import fcl
import yaml
import numpy as np
from vedo import Plotter, Line, Box, Sphere, Cylinder, color_map
from scipy.spatial.transform import Rotation as R


def euler_deg_to_matrix(euler_deg):
    """Convert Euler angles [roll, pitch, yaw] in degrees to a 3x3 rotation matrix."""
    if not euler_deg:
        return np.eye(3)
    r = R.from_euler('xyz', euler_deg, degrees=True)
    return r.as_matrix()


def apply_rotation(obj, euler_deg, point=(0, 0, 0)):
    """Apply Euler rotation [roll, pitch, yaw] in degrees to the given vedo object."""
    if euler_deg:
        roll, pitch, yaw = euler_deg
        obj.rotate(roll, axis=(1, 0, 0), point=point)
        obj.rotate(pitch, axis=(0, 1, 0), point=point)
        obj.rotate(yaw, axis=(0, 0, 1), point=point)


def load_obstacles_from_yaml(yaml_path, num_drones=1):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    initial_configuration = config.get("initial_configuration", None)
    assert initial_configuration is not None, "No initial_configuration provided"
    assert len(
        initial_configuration) == num_drones, "The number of points in the initial configuration must match the number of drones"
    for position in initial_configuration:
        if len(position) != 3:
            assert False, "Malformed positions in initial_configuration"

    initial_configuration = np.array(initial_configuration, dtype=np.float32)

    # Validate and parse bounds
    bounds_config = config.get("bounds", None)
    assert bounds_config is not None, "Missing required 'bounds' field in environment YAML"

    try:
        x_bounds = bounds_config["x"]
        y_bounds = bounds_config["y"]
        z_bounds = bounds_config["z"]
        assert (
                isinstance(x_bounds, list) and len(x_bounds) == 2 and
                isinstance(y_bounds, list) and len(y_bounds) == 2 and
                isinstance(z_bounds, list) and len(z_bounds) == 2
        ), "Each bound (x, y, z) must be a list of two values"

        bounds = np.array([x_bounds, y_bounds, z_bounds], dtype=np.float32)  # Shape (3, 2)
    except (KeyError, AssertionError, ValueError) as e:
        raise AssertionError(f"Invalid 'bounds' specification in YAML: {e}")

    obstacles = []
    fcl_objects = []
    obstacles_yaml = config.get("obstacles", [])
    if obstacles_yaml is not None:
        for obj in config.get("obstacles", []):
            color = obj.get("color", "gray")
            rot_euler = obj.get("rotation", None)

            if obj["type"] == "box":
                pos = np.array(obj["position"], dtype=np.float32)
                size = np.array(obj["size"], dtype=np.float32)
                box_viz = Box(pos=pos.tolist(), length=size[0], width=size[1], height=size[2]).c(color)
                apply_rotation(box_viz, rot_euler, point=pos.tolist())
                obstacles.append(box_viz)

                fcl_geom = fcl.Box(*size)
                rot_matrix = euler_deg_to_matrix(rot_euler)
                tf = fcl.Transform(rot_matrix, pos.tolist())
                fcl_objects.append(fcl.CollisionObject(fcl_geom, tf))

            elif obj["type"] == "sphere":
                pos = np.array(obj["position"], dtype=np.float32)
                radius = float(obj["radius"])
                sphere_viz = Sphere(pos=pos.tolist(), r=radius).c(color)
                apply_rotation(sphere_viz, rot_euler, point=pos.tolist())
                obstacles.append(sphere_viz)

                rot_matrix = euler_deg_to_matrix(rot_euler)
                fcl_geom = fcl.Sphere(radius)
                tf = fcl.Transform(rot_matrix, pos.tolist())
                fcl_objects.append(fcl.CollisionObject(fcl_geom, tf))

            elif obj["type"] == "cylinder":
                p1 = np.array(obj["endpoints"][0], dtype=np.float32)
                p2 = np.array(obj["endpoints"][1], dtype=np.float32)
                radius = float(obj["radius"])
                center = (p1 + p2) / 2
                vec = p2 - p1
                height = np.linalg.norm(vec)

                cylinder_viz = Cylinder(pos=[p1.tolist(), p2.tolist()], r=radius).c(color)
                apply_rotation(cylinder_viz, rot_euler, point=center.tolist())
                obstacles.append(cylinder_viz)

                fcl_geom = fcl.Cylinder(radius, height)
                rot_matrix = euler_deg_to_matrix(rot_euler)
                tf = fcl.Transform(rot_matrix, center.tolist())
                fcl_objects.append(fcl.CollisionObject(fcl_geom, tf))
            else:
                print(f"Unknown obstacle type: {obj['type']}")

    manager = fcl.DynamicAABBTreeCollisionManager()
    manager.registerObjects(fcl_objects)
    manager.setup()
    return initial_configuration, obstacles, manager, bounds


def load_goal_areas_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    goal_viz = []
    goal_positions = []
    goal_radii = []

    for entry in config.get("goals", []):
        pos = np.array(entry["position"], dtype=np.float32)
        radius = float(entry["radius"])
        color = entry.get("color", "yellow")

        sphere = Sphere(pos=pos.tolist(), r=radius).c(color).alpha(0.3)
        goal_viz.append(sphere)
        goal_positions.append(pos)
        goal_radii.append(radius)

    return goal_viz, np.array(goal_positions), np.array(goal_radii)


class MultiDrone:
    def __init__(self, num_drones, environment_file="obstacles.yaml"):
        """
        Initialize the multi-drone simulator.

        Args:
            num_drones (int): Number of drones to simulate.
            dt (float): Simulation time step (in seconds). Only used if dynamics are stepped.
        """
        self.N = num_drones
        self._drone_radius = 0.3  # Sphere radius used for collision checking

        # Placeholder for drone positions
        self.configuration = np.zeros((self.N, 3), dtype=np.float32)

        # Trajectories (used for visualization only)
        self.trajectories = [[] for _ in range(self.N)]

        # Collision geometry per drone
        self._fcl_objects = [fcl.CollisionObject(fcl.Sphere(self._drone_radius)) for _ in range(self.N)]

        # Load environment from YAML
        self.configuration, self._obstacles_viz, self._obstacles_collision, self._bounds = load_obstacles_from_yaml(
            environment_file, num_drones=self.N)
        self._initial_configuration = self.configuration.copy()

        # Load goal areas from YAML
        self._goal_viz, self._goal_positions, self._goal_radii = load_goal_areas_from_yaml(environment_file)
        assert self._goal_positions.shape[
                   0] == num_drones, "You must specify the sample number of goal as there are drones"

        self.reset(self.configuration)

    @property
    def initial_configuration(self):
        return self._initial_configuration

    @property
    def goal_positions(self):
        return self._goal_positions

    def reset(self, configuration=None):
        """
        Reset all drones to the specified configuration.

        Args:
            configuration (np.ndarray of shape (N, 3), optional): Initial configuration of all drones.
                If None, all drones are reset to the origin [0, 0, 0].
        """
        if configuration is not None:
            configuration = np.asarray(configuration, dtype=np.float32)
            assert configuration.shape == (self.N, 3), f"Expected shape ({self.N}, 3)"
            self.configuration = configuration
        else:
            self.configuration = np.zeros((self.N, 3), dtype=np.float32)

        for i in range(self.N):
            self._fcl_objects[i].setTransform(fcl.Transform(np.eye(3), self.configuration[i].tolist()))
            self.trajectories[i] = [self.configuration[i].copy()]

        self._init_plot()

    def set_configuration(self, configuration):
        """
        Set the internal configuration of all drones to the given configuration.

        Args:
            configuration (np.ndarray): Array of shape (N, 3), one position per drone.
        """
        assert configuration.shape == (self.N, 3), f"Expected shape ({self.N}, 3)"
        self.configuration = configuration.copy()

        for i in range(self.N):
            self._fcl_objects[i].setTransform(fcl.Transform(np.eye(3), configuration[i].tolist()))

    def is_valid(self, configuration):
        """
        Check whether the given configuration is valid:
        - All drones are within bounds
        - No drone collides with an obstacle
        - No drone collides with another drone

        Args:
            configuration (np.ndarray): Array of shape (N, 3), one position per drone.

        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        assert configuration.shape == (self.N, 3), f"Expected shape of configuration: ({self.N}, 3)"

        # Check bounds
        lower = self._bounds[:, 0]  # shape (3,)
        upper = self._bounds[:, 1]  # shape (3,)
        if not np.all((configuration >= lower) & (configuration <= upper)):
            return False

        # Check drone-environment collisions
        req = fcl.CollisionRequest(num_max_contacts=1, enable_contact=False)
        for i in range(self.N):
            self._fcl_objects[i].setTransform(fcl.Transform(np.eye(3), configuration[i].tolist()))
            rdata = fcl.CollisionData(request=req)
            self._obstacles_collision.collide(self._fcl_objects[i], rdata, fcl.defaultCollisionCallback)
            if rdata.result.is_collision:
                return False

        # Check drone-drone collisions
        diffs = configuration[:, np.newaxis, :] - configuration[np.newaxis, :, :]  # (N, N, 3)
        dists = np.linalg.norm(diffs, axis=-1)  # (N, N)
        mask = np.triu(np.ones((self.N, self.N), dtype=bool), k=1)  # upper triangle, no diag
        if np.any(dists[mask] < 2 * self._drone_radius):
            return False

        return True

    def motion_valid(self, configuration_0, configuration_1):
        """
        Check whether the given straight-line motion between configuration_0 and configuration_1 is valid:
        - All drones remain within bounds during the motion
        - No drone collides with an obstacle durin the motion
        - No drone collides with another drone during the motion

        Args:
            configuration_0 (np.ndarray): Start positions of all drones, shape (N, 3).
            configuration_1 (np.ndarray): End positions of all drones, shape (N, 3).

        Returns:
            bool: True if the motion is valid. False otherwise.
        """
        assert configuration_0.shape == (self.N, 3), f"Expected shape of configuration_0: ({self.N}, 3)"
        assert configuration_1.shape == (self.N, 3), f"Expected shape of configuration_1: ({self.N}, 3)"

        max_dist = np.linalg.norm(configuration_1 - configuration_0, axis=1).max()
        if max_dist < 1e-6:
            return self.is_valid(configuration_0)

        step_size = self._drone_radius * 0.5
        num_steps = int(np.ceil(max_dist / step_size))

        for i in range(num_steps + 1):
            alpha = i / num_steps
            interp = (1 - alpha) * configuration_0 + alpha * configuration_1
            if not self.is_valid(interp):
                return False
        return True

    def is_goal(self, configuration: np.ndarray) -> bool:
        """
        Check whether all drones are inside their respective goal areas for the given configuration

        Args:
            configuration (np.ndarray): Array of shape (N, 3) containing drone configuration.

        Returns:
            bool: True if all drones are inside their corresponding goal spheres.
        """
        assert hasattr(self, "_goal_positions") and hasattr(self, "_goal_radii"), \
            "Goal positions and radii must be set before calling is_goal()."
        assert configuration.shape == (self.N, 3), f"Expected configuration shape ({self.N}, 3)"

        distances = np.linalg.norm(configuration - self._goal_positions, axis=1)
        return np.all(distances <= self._goal_radii)

    def _init_plot(self):
        self._plotter = Plotter(interactive=False)
        self._drone_visuals = []

        for i in range(self.N):
            body = Sphere(r=0.1).c("cyan")
            arm1 = Cylinder(r=0.03, height=1.0).c("black")
            arm2 = Cylinder(r=0.03, height=1.0).c("black")
            traj = Line(np.array(self.trajectories[i])).lw(2).c("blue")
            self._drone_visuals.append((body, arm1, arm2, traj))

        visuals_flat = []
        for i in range(self.N):
            visuals_flat.extend(self._drone_visuals[i])
        visuals_flat.extend(self._obstacles_viz)
        visuals_flat.extend(self._goal_viz)

        self._plotter.show(
            *visuals_flat,
            axes=dict(
                xrange=(0, 50),
                yrange=(0, 50),
                zrange=(0, 50),
                xygrid=True,
                yzgrid=True,
                zxgrid=True,
            ),
            viewup='z',
            interactive=False,
            mode=8,
        )

    def _update_plot(self):
        arm_len = 0.5
        for i in range(self.N):
            pos = self.configuration[i]
            body, arm1, arm2, traj = self._drone_visuals[i]

            # Update arms
            arm1_p1 = pos + np.array([-arm_len, 0, 0])
            arm1_p2 = pos + np.array([arm_len, 0, 0])
            arm2_p1 = pos + np.array([0, -arm_len, 0])
            arm2_p2 = pos + np.array([0, arm_len, 0])

            self._plotter.remove(arm1)
            self._plotter.remove(arm2)
            self._plotter.remove(traj)

            new_arm1 = Cylinder(pos=[arm1_p1, arm1_p2], r=0.03).c('black')
            new_arm2 = Cylinder(pos=[arm2_p1, arm2_p2], r=0.03).c('black')
            new_traj = Line(np.array(self.trajectories[i])).lw(2).c(traj.color())

            self._plotter.add(new_arm1)
            self._plotter.add(new_arm2)
            self._plotter.add(new_traj)

            body.pos(pos)
            self._drone_visuals[i] = (body, new_arm1, new_arm2, new_traj)

        self._plotter.reset_camera()
        self._plotter.render()

    def visualize_paths(self, path: list[np.ndarray]):
        """
        Update each drone's trajectory line using a path through configuration space.

        Each trajectory is color-coded uniquely.

        Args:
            path (list[np.ndarray]): A list of (N, 3) configurations from start to goal.
        """
        assert len(path) >= 2, "Path must contain at least 2 configurations."
        assert all(p.shape == (self.N, 3) for p in path), "Each config must be of shape (N, 3)"

        for i in range(self.N):
            trajectory_i = np.array([config[i] for config in path])
            self.trajectories[i] = trajectory_i.tolist()

            # Remove previous trajectory line
            _, arm1, arm2, old_traj = self._drone_visuals[i]
            self._plotter.remove(old_traj)

            # Assign unique color based on drone index
            color = color_map(i, name='jet', vmin=0, vmax=self.N - 1)
            new_traj = Line(trajectory_i).lw(2).c(color)

            # Update visuals
            self._plotter.add(new_traj)
            self._drone_visuals[i] = (self._drone_visuals[i][0], arm1, arm2, new_traj)

        # Move drones to final state
        self.set_configuration(path[-1])

        # Redraw scene
        self._update_plot()
        self._plotter.interactive()