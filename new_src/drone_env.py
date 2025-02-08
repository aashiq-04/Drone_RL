import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        
        # Define the grid size (10x10)
        self.grid_size = 11
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        
        # Define the start, target, and obstacle positions
        self.start = np.array([1, 1])
        self.target = np.array([10, 10])
        self.obstacles = [np.array([6, 4]), np.array([7, 7]), np.array([2, 3]),np.array([3, 6])]
        
        # Initialize the drone's position
        self.drone_pos = self.start.copy()
        
    def reset(self):
        # Reset the drone to the start position
        self.drone_pos = self.start.copy()
        return self.drone_pos
    
    def step(self, action):
        # Move the drone based on the action
        if action == 0:  # Up
            self.drone_pos[1] += 1
        elif action == 1:  # Down
            self.drone_pos[1] -= 1
        elif action == 2:  # Left
            self.drone_pos[0] -= 1
        elif action == 3:  # Right
            self.drone_pos[0] += 1
        
        # Clip the drone's position to stay within the grid
        self.drone_pos = np.clip(self.drone_pos, 0, self.grid_size - 1)
        
        # Calculate the reward
        distance_to_target = np.linalg.norm(self.drone_pos - self.target)
        reward = -distance_to_target  # Reward is negative distance to target
        
        # Check if the drone collides with an obstacle
        done = False
        for obstacle in self.obstacles:
            if np.array_equal(self.drone_pos, obstacle):
                reward = -200  # Large penalty for collision
                done = True
                break
        
        # Check if the drone reaches the target
        if np.array_equal(self.drone_pos, self.target):
            reward = 150  # Large reward for reaching the target
            done = True
        
        return self.drone_pos, reward, done, {}
    
    def render(self):
        # Visualize the environment
        plt.clf()
        plt.xlim(0, self.grid_size)
        plt.ylim(0, self.grid_size)
        
        # Plot the drone
        plt.plot(self.drone_pos[0], self.drone_pos[1], 'ro', label='Drone')
        
        # Plot the target
        plt.plot(self.target[0], self.target[1], 'gx', label='Target')
        
        # Plot the obstacles
        for obstacle in self.obstacles:
            plt.plot(obstacle[0], obstacle[1], 'ks', label='Obstacle')
        
        plt.legend()
        plt.pause(0.1)