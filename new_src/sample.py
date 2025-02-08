import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DroneEnv3D(gym.Env):
    def __init__(self):
        super(DroneEnv3D, self).__init__()

        # Define the 3D grid size (10x10x10)
        self.grid_size = 11
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(3,), dtype=np.float32)

        # 6 actions: Forward, Backward, Left, Right, Up, Down
        self.action_space = spaces.Discrete(6)  

        # Define start, target, and obstacle positions
        self.start = np.array([1, 1, 1])
        self.target = np.array([10, 10, 10])
        self.obstacles = [
            np.array([6, 4, 2]), np.array([7, 7, 3]), 
            np.array([2, 3, 5]), np.array([3, 6, 6])
        ]

        # Initialize drone position
        self.drone_pos = self.start.copy()

    def reset(self):
        # Reset drone position
        self.drone_pos = self.start.copy()
        return self.drone_pos

    def step(self, action):
        # Move the drone based on the action
        if action == 0:  # Forward (Y+)
            self.drone_pos[1] += 1
        elif action == 1:  # Backward (Y-)
            self.drone_pos[1] -= 1
        elif action == 2:  # Left (X-)
            self.drone_pos[0] -= 1
        elif action == 3:  # Right (X+)
            self.drone_pos[0] += 1
        elif action == 4:  # Up (Z+)
            self.drone_pos[2] += 1
        elif action == 5:  # Down (Z-)
            self.drone_pos[2] -= 1

        # Clip position within grid boundaries
        self.drone_pos = np.clip(self.drone_pos, 0, self.grid_size - 1)

        # Calculate reward based on distance to target
        distance_to_target = np.linalg.norm(self.drone_pos - self.target)
        reward = -distance_to_target  # Closer to target = higher reward

        # Collision check
        done = False
        for obstacle in self.obstacles:
            if np.array_equal(self.drone_pos, obstacle):
                reward = -200  # Large penalty for collision
                done = True
                break

        # Check if the drone reaches the target
        if np.array_equal(self.drone_pos, self.target):
            reward = 150  # Large reward for reaching the goal
            done = True

        return self.drone_pos, reward, done, {}

    def render(self):
        # 3D Visualization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_zlim(0, self.grid_size)

        # Plot the drone position
        ax.scatter(*self.drone_pos, color='red', label='Drone')

        # Plot the target
        ax.scatter(*self.target, color='green', marker='X', s=100, label='Target')

        # Plot obstacles
        for obstacle in self.obstacles:
            ax.scatter(*obstacle, color='black', marker='s', s=80, label='Obstacle')

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('3D Drone Environment')

        plt.legend()
        plt.show()

# Example usage
env = DroneEnv3D()
env.reset()
env.render()
