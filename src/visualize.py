import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from Environment import DroneEnv3D  # Import the environment

def visualize_environment(env):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot boundaries
    
    ax.set_xlim(0, env.space_size)
    ax.set_ylim(0, env.space_size)
    ax.set_zlim(0, env.space_size)
    
    # Plot drone position
    ax.scatter(*env.drone_pos, color='red', s=100, label="Drone")
    
    # Plot target position
    ax.scatter(*env.target, color='green', s=150, label="Target")
    
    # Plot obstacles
    for obs in env.obstacles:
        draw_cylinder(ax, *obs)
    
    # Plot drone path
    path = np.array(env.path)
    if len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color='blue', linestyle='-', marker='o', markersize=3, label="Path")
    ax.view_init(elev=90)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Drone Environment Visualization")
    ax.legend()
    plt.show()
    
def draw_cylinder(ax, x, y, r, h, num_slices=50):
    """Draws a 3D cylinder obstacle."""
    theta = np.linspace(0, 2 * np.pi, num_slices)
    z = np.linspace(0, h, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    X = r * np.cos(theta_grid) + x
    Y = r * np.sin(theta_grid) + y
    Z = z_grid
    ax.plot_surface(X, Y, Z, color='blue', alpha=0.5)

if __name__ == "__main__":
    env = DroneEnv3D()  # Create the environment
    visualize_environment(env)
