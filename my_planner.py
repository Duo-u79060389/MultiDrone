import numpy as np
import random
from collections import OrderedDict
from multi_drone import MultiDrone
import time

# Set random seeds for reproducibility during demo
np.random.seed(0)
random.seed(0)


class RRTNode:
    """Node for RRT tree structure"""

    def __init__(self, configuration):
        self.configuration = np.array(configuration, dtype=np.float32)
        self.parent = None
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        child.parent = self


class RRTTree:
    """RRT Tree implementation with incremental flat cache for efficient nearest neighbor"""

    def __init__(self):
        self.nodes = []
        self._flat = None  # Incremental [n, 3K] flattened matrix cache

    def add_node(self, configuration):
        """Add a new node to the tree with deduplication"""
        # Check for near-duplicate nodes
        for n in self.nodes:
            if np.allclose(n.configuration, configuration, atol=1e-5):
                return n

        node = RRTNode(configuration)
        self.nodes.append(node)

        # Incrementally maintain flattened cache
        flat = configuration.flatten()[None, :]
        if self._flat is None:
            self._flat = flat
        else:
            self._flat = np.vstack((self._flat, flat))

        return node

    def add_edge(self, parent_config, child_config):
        """Add an edge between two configurations, returns success status"""
        parent_node = self.find_node(parent_config)
        child_node = self.find_node(child_config)
        if parent_node and child_node:
            parent_node.add_child(child_node)
            return True
        return False

    def find_node(self, configuration):
        """Find node with matching configuration using relaxed tolerance"""
        for node in self.nodes:
            if np.allclose(node.configuration, configuration, atol=1e-5):
                return node
        return None


def nearest_config(tree: RRTTree, q):
    """Fast nearest neighbor using cached flat matrix - O(N) batch distance"""
    if not tree.nodes:
        return None

    d = tree._flat - q.flatten()[None, :]
    idx = int(np.argmin(np.einsum('ij,ij->i', d, d)))
    return tree.nodes[idx].configuration


def k_nearest_configs(tree: RRTTree, q, k=5):
    """Get k nearest neighbors efficiently using argpartition - O(N) amortized"""
    if not tree.nodes:
        return []

    d = tree._flat - q.flatten()[None, :]
    distances = np.einsum('ij,ij->i', d, d)
    k = min(k, len(tree.nodes))
    idxs = np.argpartition(distances, k - 1)[:k]
    idxs = idxs[np.argsort(distances[idxs])]
    return [tree.nodes[int(i)].configuration for i in idxs]


def _get_bounds(sim):
    """Safely get environment bounds without relying on private attributes"""
    b = getattr(sim, "bounds", None)
    if b is None:
        b = getattr(sim, "_bounds", None)
    if b is None:
        lo = np.minimum(sim.initial_configuration.min(0), sim.goal_positions.min(0)) - 1.0
        hi = np.maximum(sim.initial_configuration.max(0), sim.goal_positions.max(0)) + 1.0
        b = np.stack([lo, hi], axis=1)  # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
    return b.astype(np.float32)


def _safe_clamp_to_bounds(configuration, bounds):
    """Safely clamp configuration to bounds"""
    clamped = configuration.copy()
    for i in range(clamped.shape[0]):
        for j in range(3):
            clamped[i, j] = np.clip(clamped[i, j], bounds[j, 0], bounds[j, 1])
    return clamped


class ExtendResult:
    """Result object for tree extension operations"""

    def __init__(self, success=False, final_node=None, reached_target=False, added_edges=False):
        self.success = success
        self.final_node = final_node
        self.reached_target = reached_target
        self.added_edges = added_edges


class ConnectResult:
    """Result object for tree connection operations"""

    def __init__(self, connected=False, connection_points=None):
        self.connected = connected
        self.connection_points = connection_points


class _LRU:
    """Tiny LRU for motion/config validity caching to cut hot-path calls"""
    def __init__(self, cap=20000):
        self.cap = cap
        self.d = OrderedDict()

    def get(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return self.d[k]
        return None

    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)


def _hash_cfg(cfg, prec=3):
    # Quantize to reduce cache misses from float jitter
    q = np.round(cfg.astype(np.float32), prec)
    return q.tobytes()


def _hash_segment(a, b, prec=3):
    # Order-insensitive hash for undirected segment
    ha = _hash_cfg(a, prec)
    hb = _hash_cfg(b, prec)
    return (ha, hb) if ha <= hb else (hb, ha)


class CentralizedMultiDroneBiRRT:
    """Centralized BiRRT planner for multiple drones with performance optimizations"""

    def __init__(self, max_iterations=5000, step_size=2.0, connect_threshold=3.0, safety_radius=0.6):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.connect_threshold = connect_threshold
        self.safety_radius = safety_radius

        # Adaptive sampling weights: [uniform, obstacle-based, bridge-test, goal-bias]
        self.sampling_weights = np.array([1.0, 1.0, 1.0, 1.0])

        # Adaptive step size parameters
        self.original_step_size = step_size
        self.extension_failures = 0
        self.extension_successes = 0

        # Random seed unlock flag
        self.seed_unlocked = False

        # Caches
        self._motion_cache = _LRU(cap=30000)
        self._config_cache = _LRU(cap=30000)

    # ----------------------------- Cached wrappers -----------------------------
    def _is_valid_cached(self, sim, q):
        k = _hash_cfg(q)
        v = self._config_cache.get(k)
        if v is None:
            v = bool(sim.is_valid(q))
            self._config_cache.put(k, v)
        return v

    def _motion_valid_cached(self, sim, a, b):
        k = _hash_segment(a, b)
        v = self._motion_cache.get(k)
        if v is None:
            v = bool(sim.motion_valid(a, b))
            self._motion_cache.put(k, v)
        return v

    # ----------------------------- Main planning -----------------------------
    def plan(self, sim):
        """Main planning function with timeout protection and fallback"""
        start_time = time.time()
        self._last_K = sim.N  # 记录当前无人机数量（你之前提到要存）

        # Initialize trees
        T_start = RRTTree()
        T_goal = RRTTree()

        q_start = sim.initial_configuration
        q_goal = sim.goal_positions

        # Early direct connect (fast path)
        if self._motion_valid_cached(sim, q_start, q_goal):
            return [q_start.copy(), q_goal.copy()]

        # Verify start and goal are valid
        if not self._is_valid_cached(sim, q_start):
            print("ERROR: Invalid start configuration!")
            return None

        if not self._is_valid_cached(sim, q_goal):
            print("ERROR: Invalid goal configuration!")
            return None

        T_start.add_node(q_start)
        T_goal.add_node(q_goal)

        swap_flag = False

        for i in range(self.max_iterations):
            elapsed = time.time() - start_time

            # Hard timeout protection (110 seconds = 1min 50sec, leaving 10sec buffer)
            if elapsed > 110:
                print("Timeout reached - attempting fallback solution")
                return self._timeout_fallback(sim, T_start, T_goal, q_goal)

            # Unlock random seed after 90 seconds for escape chance
            if 90 < elapsed < 91 and not self.seed_unlocked:
                np.random.seed(None)
                random.seed()
                self.seed_unlocked = True
                print("Random seed unlocked for escape attempt")

            # Adaptive step size adjustment
            if i > 0 and i % 120 == 0:
                self._adapt_step_size()

            # Dynamic connect threshold based on failure rate
            self._update_connect_threshold()

            # Select active and passive trees
            if swap_flag:
                active_tree = T_goal
                passive_tree = T_start
            else:
                active_tree = T_start
                passive_tree = T_goal
            swap_flag = not swap_flag

            # Periodically try direct connect to goal from a good node in active tree
            if i % 200 == 0 and active_tree.nodes:
                best = min(active_tree.nodes, key=lambda n: self._weighted_distance(n.configuration, q_goal))
                if self._motion_valid_cached(sim, best.configuration, q_goal):
                    active_tree.add_node(q_goal)
                    active_tree.add_edge(best.configuration, q_goal)
                    return self._shortcut_smooth_path(sim, self._trace_path_to_root(active_tree, q_goal))

            # Adaptive sampling
            q_rand, strategy_idx = self._sample_adaptive(sim, q_goal)

            # Extend active tree
            q_nearest_active = nearest_config(active_tree, q_rand)
            if q_nearest_active is None:
                continue

            extend_result = self._extend_tree_connect(sim, active_tree, q_nearest_active, q_rand)

            if extend_result.success:
                q_new = extend_result.final_node
                self.extension_successes += 1

                # Attempt connection to passive tree - try k nearest neighbors (adaptive k)
                k_nb = min(12, max(5, len(passive_tree.nodes) // 50 + 5))
                candidates = k_nearest_configs(passive_tree, q_new, k=k_nb)
                connected = False
                for q_nearest_passive in candidates:
                    connect_result = self._connect_trees_rrt_connect(sim, q_new, q_nearest_passive, passive_tree)
                    if connect_result.connected:
                        # Success! Greedy join & smooth
                        final_result = self._greedy_connect(sim, T_start, T_goal, connect_result.connection_points)
                        if final_result:
                            self.sampling_weights[strategy_idx] += 1.0
                            self.sampling_weights *= 0.999  # Weight decay
                            return final_result
                        connected = True
                        break

                if not connected and extend_result.added_edges:
                    self.sampling_weights[strategy_idx] += 0.1
                    self.sampling_weights *= 0.999
            else:
                self.extension_failures += 1

        # Final fallback if iterations exhausted
        return self._timeout_fallback(sim, T_start, T_goal, q_goal)

    # ----------------------------- Helpers -----------------------------
    def _update_connect_threshold(self):
        """Dynamic connect threshold based on failure rate"""
        base_ct = 2.0 * self.step_size
        total = self.extension_failures + self.extension_successes + 1
        fail_ratio = self.extension_failures / total
        self.connect_threshold = base_ct * (1.0 if fail_ratio < 0.6 else 1.6)

    def _timeout_fallback(self, sim, T_start, T_goal, q_goal):
        """Return best partial path instead of None on timeout"""
        try:
            if not T_start.nodes:
                return None

            best_start = min(T_start.nodes, key=lambda n: self._weighted_distance(n.configuration, q_goal))

            # Try direct connection to goal
            if self._motion_valid_cached(sim, best_start.configuration, q_goal):
                T_start.add_node(q_goal)
                T_start.add_edge(best_start.configuration, q_goal)
                partial_path = self._trace_path_to_root(T_start, q_goal)
                if len(partial_path) >= 2:
                    return [np.asarray(q, np.float32) for q in partial_path]

            # Return path to best node found (only if it makes significant progress)
            partial_path = self._trace_path_to_root(T_start, best_start.configuration)
            if len(partial_path) >= 2:
                distance_to_goal = self._weighted_distance(best_start.configuration, q_goal)
                start_to_goal = self._weighted_distance(sim.initial_configuration, q_goal)
                if distance_to_goal < 0.8 * start_to_goal:
                    print(f"Returning partial path with {len(partial_path)} waypoints")
                    return [np.asarray(q, np.float32) for q in partial_path]
        except Exception as e:
            print(f"Fallback failed: {e}")

        return None

    def _adapt_step_size(self):
        """Adaptive step size based on recent success rate"""
        total = self.extension_successes + self.extension_failures
        if total > 0:
            success_rate = self.extension_successes / total
            if success_rate < 0.3:
                self.step_size = max(0.85 * self.step_size, 0.5 * self.original_step_size)
            elif success_rate > 0.7:
                self.step_size = min(1.12 * self.step_size, 2.0 * self.original_step_size)

    def _greedy_connect(self, sim, T_start, T_goal, connection_points):
        """Greedy RRT-Connect style pushing from both connection points"""
        try:
            path = self._construct_bidirectional_path(T_start, T_goal, connection_points)
            path = self._endpoint_straighten(sim, path)
            smoothed_path = self._shortcut_smooth_path(sim, path)

            # Ensure exact goal if close and straight reachable
            if len(smoothed_path) > 0 and not sim.is_goal(smoothed_path[-1]):
                goal_positions = sim.goal_positions
                if self._weighted_distance(smoothed_path[-1], goal_positions) < self.connect_threshold:
                    if self._motion_valid_cached(sim, smoothed_path[-1], goal_positions):
                        smoothed_path.append(goal_positions.copy())

            return [np.asarray(q, np.float32) for q in smoothed_path]
        except Exception as e:
            print(f"Greedy connect failed: {e}")
            return None

    def _endpoint_straighten(self, sim, path):
        """Greedy straightening from both endpoints"""
        if len(path) <= 2:
            return path
        straightened = [path[0]]
        i = 0
        while i < len(path) - 1:
            furthest = i + 1
            for j in range(len(path) - 1, i, -1):
                if self._motion_valid_cached(sim, path[i], path[j]):
                    furthest = j
                    break
            straightened.append(path[furthest])
            i = furthest
        return straightened

    # ----------------------------- Sampling -----------------------------
    def _sample_adaptive(self, sim, q_goal):
        """Adaptive sampling with multiple strategies including direct goal sampling"""
        # 8% chance to directly sample goal
        if np.random.rand() < 0.08:
            return q_goal.copy(), 3  # Treat as goal-bias strategy

        # Normalize weights to probabilities
        probabilities = self.sampling_weights / np.sum(self.sampling_weights)
        strategy_idx = np.random.choice(4, p=probabilities)

        if strategy_idx == 0:  # Uniform sampling
            return self._sample_uniform_configuration(sim), strategy_idx
        elif strategy_idx == 1:  # Obstacle-based sampling
            return self._sample_near_obstacles(sim), strategy_idx
        elif strategy_idx == 2:  # Bridge-test sampling
            return self._sample_bridge_test(sim), strategy_idx
        else:  # Goal-biased sampling
            return self._sample_goal_bias(sim, q_goal), strategy_idx

    def _sample_uniform_configuration(self, sim):
        bounds = _get_bounds(sim)
        num_drones = sim.N
        config = np.zeros((num_drones, 3), dtype=np.float32)
        for i in range(num_drones):
            for j in range(3):
                config[i, j] = np.random.uniform(bounds[j, 0], bounds[j, 1])
        return config

    def _sample_near_obstacles(self, sim):
        bounds = _get_bounds(sim)
        for _ in range(40):
            q = self._sample_uniform_configuration(sim)
            if not self._is_valid_cached(sim, q):
                for _ in range(16):
                    noise = np.random.normal(0, 0.5 * self.step_size, q.shape)
                    cand = _safe_clamp_to_bounds(q + noise, bounds)
                    if self._is_valid_cached(sim, cand):
                        return cand
        return self._sample_uniform_configuration(sim)

    def _sample_bridge_test(self, sim):
        bounds = _get_bounds(sim)
        for attempt in range(6):
            q1 = self._sample_uniform_configuration(sim)
            tries = 0
            while self._is_valid_cached(sim, q1) and tries < 25:
                q1 = self._sample_uniform_configuration(sim)
                tries += 1
            if tries >= 25:
                return self._sample_uniform_configuration(sim)

            direction = np.random.normal(0, 1, q1.shape)
            direction = direction / (np.linalg.norm(direction.flatten()) + 1e-9)
            bridge_distance = 2.2 * self.step_size
            q2 = _safe_clamp_to_bounds(q1 + bridge_distance * direction, bounds)

            if self._is_valid_cached(sim, q2):
                q_bridge = (q1 + q2) / 2
                if self._is_valid_cached(sim, q_bridge):
                    return q_bridge
                # refine
                for _ in range(3):
                    q_mid1 = (q1 + q_bridge) / 2
                    if self._is_valid_cached(sim, q_mid1):
                        return q_mid1
                    q_mid2 = (q_bridge + q2) / 2
                    if self._is_valid_cached(sim, q_mid2):
                        return q_mid2
        return self._sample_uniform_configuration(sim)

    def _sample_goal_bias(self, sim, q_goal):
        noise = np.random.normal(0, self.step_size / 3, q_goal.shape)
        q_sample = q_goal + noise
        bounds = _get_bounds(sim)
        return _safe_clamp_to_bounds(q_sample, bounds)

    # ----------------------------- Steering / Extend -----------------------------
    def _extend_tree_connect(self, sim, tree, q_near, q_target):
        """RRT-Connect style tree extension with safe bounds clamping"""
        q_current = q_near.copy()
        added_edges = False

        while True:
            q_next = self._steer_weighted_smart(sim, q_current, q_target)

            if not self._motion_valid_cached(sim, q_current, q_next):
                # cannot move further from current along target
                if added_edges:
                    return ExtendResult(success=True, final_node=q_current,
                                        reached_target=False, added_edges=True)
                else:
                    return ExtendResult(success=False, final_node=None,
                                        reached_target=False, added_edges=False)

            tree.add_node(q_next)
            tree.add_edge(q_current, q_next)
            added_edges = True

            if self._weighted_distance(q_next, q_target) <= self.step_size:
                return ExtendResult(success=True, final_node=q_next,
                                    reached_target=True, added_edges=True)

            q_current = q_next

    def _steer_weighted_smart(self, sim, q_current, q_target):
        """Hybrid steering:
           1) Try full-fleet straight step (fast),
           2) If blocked, try grouped move (top-m farthest drones),
           3) Fallback to single-drone move.
        """
        bounds = _get_bounds(sim)
        direction = q_target - q_current
        dist = np.linalg.norm(direction)

        # 1) Full-fleet attempt
        if dist <= self.step_size:
            q_full = _safe_clamp_to_bounds(q_target.copy(), bounds)
        else:
            q_full = _safe_clamp_to_bounds(q_current + (self.step_size / (dist + 1e-9)) * direction, bounds)

        if self._motion_valid_cached(sim, q_current, q_full):
            return q_full

        # 2) Grouped move: move top-m farthest drones together (m= min(3, N))
        diffs = np.linalg.norm(q_target - q_current, axis=1)
        idxs = np.argsort(-diffs)  # descending
        m = int(min(3, len(diffs)))
        q_group = q_current.copy()
        for k in range(m):
            i = int(idxs[k])
            v = q_target[i] - q_current[i]
            di = np.linalg.norm(v)
            if di > 1e-9:
                step = v if di <= self.step_size else (self.step_size * v / di)
                q_group[i] = q_current[i] + step
        q_group = _safe_clamp_to_bounds(q_group, bounds)
        if self._motion_valid_cached(sim, q_current, q_group):
            return q_group

        # 3) Single-drone fallback: move the farthest one
        i = int(idxs[0])
        q_next = q_current.copy()
        v = q_target[i] - q_current[i]
        di = np.linalg.norm(v)
        if di <= self.step_size:
            q_next[i] = q_target[i]
        else:
            q_next[i] = q_current[i] + (self.step_size * v / (di + 1e-9))
        return _safe_clamp_to_bounds(q_next, bounds)

    def _weighted_distance(self, config1, config2):
        return np.linalg.norm((config1 - config2).flatten())

    # ----------------------------- Tree connection -----------------------------
    def _connect_trees_rrt_connect(self, sim, q_new_active, q_nearest_passive, passive_tree):
        """RRT-Connect: from passive's nearest toward q_new_active, keep extending."""
        q_curr = q_nearest_passive.copy()
        reached = False
        last = q_curr
        while True:
            q_next = self._steer_weighted_smart(sim, q_curr, q_new_active)
            if not self._motion_valid_cached(sim, q_curr, q_next):
                break
            passive_tree.add_node(q_next)
            passive_tree.add_edge(q_curr, q_next)
            last = q_next
            if self._weighted_distance(q_next, q_new_active) <= self.step_size:
                reached = True
                break
            q_curr = q_next

        if reached:
            return ConnectResult(True, (q_new_active, last))
        else:
            # Still try a direct small hop if very close
            if self._weighted_distance(last, q_new_active) <= self.connect_threshold and \
               self._motion_valid_cached(sim, last, q_new_active):
                passive_tree.add_node(q_new_active)
                passive_tree.add_edge(last, q_new_active)
                return ConnectResult(True, (q_new_active, q_new_active))
            return ConnectResult(False, None)

    # ----------------------------- Path utilities -----------------------------
    def _construct_bidirectional_path(self, T_start, T_goal, connection_points):
        q_connect_start, q_connect_goal = connection_points
        path_start = self._trace_path_to_root(T_start, q_connect_start)
        path_goal = self._trace_path_to_root(T_goal, q_connect_goal)
        path_goal.reverse()
        # Avoid duplicate in the middle
        if len(path_goal) > 0 and len(path_start) > 0 and np.allclose(path_start[-1], path_goal[0], atol=1e-5):
            full_path = path_start + path_goal[1:]
        else:
            full_path = path_start + path_goal
        return full_path

    def _trace_path_to_root(self, tree, config):
        path = []
        current_node = tree.find_node(config)
        while current_node is not None:
            path.append(current_node.configuration.copy())
            current_node = current_node.parent
        path.reverse()
        return path

    def _shortcut_smooth_path(self, sim, path):
        if len(path) <= 2:
            return path
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            shortcut = False
            while j > i + 1:
                if self._motion_valid_cached(sim, path[i], path[j]):
                    smoothed.append(path[j])
                    i = j
                    shortcut = True
                    break
                j -= 1
            if not shortcut:
                smoothed.append(path[i + 1])
                i += 1

        # Tiny noise smoothing (keep endpoints)
        if len(smoothed) > 2:
            bounds = _get_bounds(sim)
            for _ in range(2):
                newp = [smoothed[0]]
                ok = True
                for k in range(1, len(smoothed) - 1):
                    w = smoothed[k]
                    noise = np.random.normal(0, 0.1 * self.step_size, w.shape)
                    cand = _safe_clamp_to_bounds(w + noise, bounds)
                    if self._is_valid_cached(sim, cand):
                        newp.append(cand)
                    else:
                        newp.append(w)
                newp.append(smoothed[-1])
                # validate
                for k in range(len(newp) - 1):
                    if not self._motion_valid_cached(sim, newp[k], newp[k + 1]):
                        ok = False
                        break
                if ok:
                    smoothed = newp
                    break
        return smoothed


def my_planner(sim):
    """Main planner function to be called by the evaluation system"""
    num_drones = sim.N

    # 参数随规模轻量缩放（1–12 架）
    if num_drones == 1:
        max_iter = 2000
        step_size = 3.0
        safety_radius = 0.6
        connect_threshold = 2.0 * 3.0
    elif num_drones <= 3:
        max_iter = 4500
        step_size = 2.6
        safety_radius = 0.7
        connect_threshold = 2.0 * 2.6
    elif num_drones <= 6:
        max_iter = 7000
        step_size = 2.0
        safety_radius = 0.8
        connect_threshold = 2.2 * 2.0
    elif num_drones <= 9:
        max_iter = 9500
        step_size = 1.8
        safety_radius = 0.9
        connect_threshold = 2.4 * 1.8
    else:  # 10-12 drones
        max_iter = 13000
        step_size = 1.6
        safety_radius = 1.0
        connect_threshold = 2.6 * 1.6

    planner = CentralizedMultiDroneBiRRT(
        max_iterations=max_iter,
        step_size=step_size,
        connect_threshold=connect_threshold,
        safety_radius=safety_radius
    )

    start_time = time.time()
    solution_path = planner.plan(sim)
    end_time = time.time()

    print(f"Planning time: {end_time - start_time:.2f} seconds")
    print(f"Drones: {num_drones}, Max iterations: {max_iter}")

    if solution_path is not None:
        print(f"Found solution with {len(solution_path)} waypoints")
        if _verify_solution_path(sim, solution_path):
            print("Solution path verified as valid")
            # 如需可视化且 sim 支持：
            try:
                if hasattr(sim, "visualize_paths"):
                    sim.visualize_paths(solution_path)
            except Exception as e:
                print(f"Visualization skipped: {e}")
            return solution_path
        else:
            print("WARNING: Solution path validation failed!")
            return None
    else:
        print("No solution found")
        return None


def _verify_solution_path(sim, path):
    """Verify that the solution path is valid"""
    if not path or len(path) < 2:
        return False

    if not np.allclose(path[0], sim.initial_configuration, atol=1e-3):
        print(f"Start mismatch: {path[0]} vs {sim.initial_configuration}")
        return False

    if not sim.is_goal(path[-1]):
        print(f"Goal not reached: {path[-1]} vs {sim.goal_positions}")
        return False

    for i, config in enumerate(path):
        if not sim.is_valid(config):
            print(f"Invalid configuration at step {i}")
            return False
        if i > 0:
            if not sim.motion_valid(path[i - 1], config):
                print(f"Invalid motion from step {i - 1} to {i}")
                return False
    return True


# ----------------------------- Optional helpers for quick local tests -----------------------------
def create_test_environment(num_drones, bounds=None, obstacles_count=2):
    """Create a test environment configuration for given number of drones"""
    import yaml

    if bounds is None:
        bounds = {"x": [0, 50], "y": [0, 50], "z": [0, 50]}

    initial_config = []
    spacing = 2.0
    for i in range(num_drones):
        x = bounds["x"][0] + 1 + (i % 5) * spacing
        y = bounds["y"][0] + 1 + (i // 5) * spacing
        z = bounds["z"][0] + 1
        initial_config.append([x, y, z])

    goals = []
    goal_area_x_start = bounds["x"][1] - 10
    goal_area_y_start = bounds["y"][1] - 10
    for i in range(num_drones):
        x = goal_area_x_start + (i % 5) * spacing
        y = goal_area_y_start + (i // 5) * spacing
        z = bounds["z"][0] + 2
        goals.append({"position": [x, y, z], "radius": 1.0})

    obstacles = []
    if obstacles_count > 0:
        obstacle_configs = [
            {"type": "box", "position": [10, 10, 1], "size": [4, 4, 2], "rotation": [0, 0, 0], "color": "red"},
            {"type": "sphere", "position": [25, 15, 2], "radius": 2.0, "color": "red"},
            {"type": "box", "position": [15, 25, 1], "size": [3, 3, 2], "rotation": [0, 0, 0], "color": "red"},
            {"type": "cylinder", "endpoints": [[30, 10, 0], [30, 10, 10]], "radius": 1.5, "rotation": [0, 0, 0], "color": "red"},
            {"type": "box", "position": [35, 35, 1], "size": [2, 2, 2], "rotation": [0, 0, 0], "color": "red"},
            {"type": "sphere", "position": [20, 35, 3], "radius": 1.5, "color": "red"},
            {"type": "box", "position": [40, 20, 1], "size": [3, 4, 2], "rotation": [0, 0, 0], "color": "red"},
            {"type": "sphere", "position": [12, 30, 2], "radius": 1.8, "color": "red"},
        ]
        obstacles = obstacle_configs[:min(obstacles_count, len(obstacle_configs))]

    env_config = {
        "bounds": bounds,
        "initial_configuration": initial_config,
        "obstacles": obstacles,
        "goals": goals
    }

    temp_env_file = f"temp_env_{num_drones}drones.yaml"
    with open(temp_env_file, 'w') as f:
        yaml.dump(env_config, f, default_flow_style=False)

    return temp_env_file


if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZED MULTI-DRONE BiRRT PLANNER TEST")
    print("=" * 60)

    # 简单自测（如需）
    # sim = MultiDrone(num_drones=1, environment_file=create_test_environment(1, obstacles_count=0))

    # ws00_free.yaml（基线：1 架无人机、无遮挡）能跑
    # sim = MultiDrone(num_drones=1, environment_file="q4_testing_cases/ws00_free.yaml")

    # ws01_multi_no_obst.yaml（6 架、无遮挡，仅考察多机耦合）能跑
    # sim = MultiDrone(num_drones=6, environment_file="q4_testing_cases/ws01_multi_no_obst.yaml")

    # ws02_sparse_spheres.yaml（6 架、稀疏球形障碍）能跑
    # sim = MultiDrone(num_drones=6, environment_file="q4_testing_cases/ws02_sparse_spheres.yaml")

    # ws03_corridor_narrow.yaml（6 架、单一窄通道——经典“狭窄通道”）跑不动！
    # sim = MultiDrone(num_drones=6, environment_file="q4_testing_cases/ws03_corridor_narrow.yaml")

    # ws04_moderate_clutter.yaml（8 架、中等密度随机球障）
    # sim = MultiDrone(num_drones=8, environment_file="q4_testing_cases/ws04_moderate_clutter.yaml")

    # ws05_maze_boxes.yaml（10 架、箱体墙形成迷宫/走廊网络）Invalid start 跑不动！
    # sim = MultiDrone(num_drones=10, environment_file="q4_testing_cases/ws05_maze_boxes.yaml")

    # ws06_bridge_test.yaml（2 架、“bridge test” 狭缝连通）跑不了
    sim = MultiDrone(num_drones=2, environment_file="q4_testing_cases/ws06_bridge_test.yaml")

    # ws07_3d_stacks.yaml（12 架、3D 立体货架/圆柱桩，强调垂直机动）不能跑
    # sim = MultiDrone(num_drones=12, environment_file="q4_testing_cases/ws07_3d_stacks.yaml")
    # WARNING: Solution path validation failed! 不存在这样的路径莫

    my_planner(sim)

