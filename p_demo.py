import numpy as np
from multi_drone import MultiDrone

sim = MultiDrone(num_drones=2, environment_file="environment.yaml")

initial_configuration = sim.initial_configuration
goal_positions = sim.goal_positions

print("Initial configuration:\n", initial_configuration, "\n")
print("Goal positions: \n", goal_positions, "\n")

configuration = np.array([
    [5.0, 4.5, 3.0],
    [3.5, 10.0, 8.0]
], dtype=np.float32)

collides = sim.is_valid(configuration)
print("Collides:\n", collides,"\n")

start = np.array([
    [5.0, 4.5, 3.0],
    [3.5, 10.0, 8.0],
], dtype=np.float32)

end = np.array([
    [10.0, 20.0, 3.0],
    [3.5, 20.0, 15.0]
], dtype=np.float32)
motion_collides = sim.motion_valid(start, end)
print("Motion collides:\n", motion_collides,"\n")

configuration = np.array([
    [5.0, 4.5, 3.0],
    [3.5, 10.0, 8.0]
], dtype=np.float32)

goal_reached = sim.is_goal(configuration)
print("Goal reached:\n", goal_reached,"\n")

paths = [
    np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
    np.array([[20, 30, 2], [2, 1, 1]], dtype=np.float32),
    np.array([[20, 100, 2], [3, 2, 2]], dtype=np.float32)
]

sim.visualize_paths(paths)