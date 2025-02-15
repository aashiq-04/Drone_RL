import numpy as np
import gym
from gym import spaces
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
class DroneEnv3D(gym.Env):
    def __init__(self):
        super(DroneEnv3D, self).__init__()

        # Define 3D space
        self.space_size = 10
        self.target = np.array([9, 9, 9], dtype=np.float64)  # Target position

        # Continuous action space (x, y, z velocity changes)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space (position + velocity + lidar + imu)
        self.observation_space = spaces.Box(low=0, high=self.space_size, shape=(12,), dtype=np.float32)

        # Wind effect
        self.wind_strength = 0.1  

        # Define obstacles (random poles)
        self.num_obstacles = 5
        self.obstacles = []  # Initialize empty, generate on reset

        self.reset()

    def generate_obstacle(self):
        """Generate random obstacle as (x, y, radius, height)"""
        x = random.uniform(2, 8)
        y = random.uniform(2, 8)
        radius = 0.5  # Fixed radius
        height = random.uniform(5, 9)  # Tall poles
        return (x, y, radius, height)

    def reset(self):
        self.drone_pos = np.array([1, 1, 1], dtype=np.float64)  # Start position
        self.drone_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # Reset velocity
        self.path = [self.drone_pos.copy()]  # Initialize path history

        # Randomize obstacles on reset
        self.obstacles = [self.generate_obstacle() for _ in range(self.num_obstacles)]
        
        return self.get_observation()

    def step(self, action):
        action = np.clip(action, -1, 1)  # Keep actions within valid range

        # Update velocity and position
        self.drone_vel = 0.9 * self.drone_vel + 0.1 * action  # Smooth velocity update
        self.drone_pos += self.drone_vel  # Move the drone
        
        # Add wind effect
        wind = np.random.uniform(-self.wind_strength, self.wind_strength, 3)
        self.drone_pos += wind
        
        # Keep within boundaries
        self.drone_pos = np.clip(self.drone_pos, 0, self.space_size)
        
        # Sensor Data
        lidar_readings = self.get_lidar_readings()
        imu_readings = self.get_imu_readings()

        # Avoid Obstacles
        safe_action = action.copy()
    
        if lidar_readings[2] < 1.0:  # Front obstacle
            safe_action[1] = -0.5  # Move backward

        if lidar_readings[0] < 1.0:  # Right obstacle
            safe_action[0] = -0.5  # Move left

        if lidar_readings[1] < 1.0:  # Left obstacle
            safe_action[0] = 0.5  # Move right
            
        self.drone_vel = 0.9 * self.drone_vel + 0.1 * safe_action
        self.drone_pos += self.drone_vel
        
        #keeping drone withing boundaries
        self.drone_pos = np.clip(self.drone_pos, 0, self.space_size)
        
        # Collision detection (simplified for cylinder obstacles)
        collision = False
        for obs in self.obstacles:
            ox, oy, r, h = obs
            dist = np.linalg.norm(self.drone_pos[:2] - np.array([ox, oy]))
            if dist < r and self.drone_pos[2] < h:
                collision = True
                break

        # Reward function
        distance = np.linalg.norm(self.drone_pos - self.target)
        reward = -distance  # Closer = better

        if collision:
            reward -= 100  # Penalize collision
            self.drone_pos = np.array([1, 1, 1], dtype=np.float64)  # Reset position after crash

        done = distance < 0.5
        if done:
            reward += 200  # Reward reaching goal
            self.reset()

        self.path.append(self.drone_pos.copy())  # Track path
        return self.get_observation(), reward, done, {}

    def get_observation(self):
        """Combine position, velocity, and sensor data into a single observation."""
        lidar_readings = self.get_lidar_readings()
        imu_readings = self.get_imu_readings()
        return np.concatenate([self.drone_pos, self.drone_vel, lidar_readings, imu_readings])

    def get_lidar_readings(self):
        """Simulate LIDAR readings (distances in six directions: left, right, front, back, up, down)."""
        readings = np.zeros(6)
        directions = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]),
            np.array([0, -1, 0]), np.array([0, 0, 1]), np.array([0, 0, -1])
        ]
        
        for i, direction in enumerate(directions):
            for distance in np.linspace(0, self.space_size, 100):
                point = self.drone_pos + direction * distance
                if np.any(point < 0) or np.any(point > self.space_size):
                    readings[i] = distance
                    break
                for obs in self.obstacles:
                    ox, oy, r, h = obs
                    if np.linalg.norm(point[:2] - np.array([ox, oy])) < r and point[2] < h:
                        readings[i] = distance
                        break
        
        return readings

    def get_imu_readings(self):
        """Simulate IMU readings (orientation and angular velocity)."""
        # Here we use dummy data. Replace with actual calculations if available.
        orientation = np.array([0.0, 0.0, 0.0])  # Placeholder for orientation data
        angular_velocity = np.array([0.0, 0.0, 0.0])  # Placeholder for angular velocity data
        return np.concatenate([orientation, angular_velocity])

    def render(self):
        """Visualize the environment in 3D with path."""
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot boundaries
        ax.set_xlim(0, self.space_size)
        ax.set_ylim(0, self.space_size)
        ax.set_zlim(0, self.space_size)

        # Plot drone
        ax.scatter(*self.drone_pos, color='red', s=100, label="Drone")

        # Plot target
        ax.scatter(*self.target, color='green', s=150, label="Target")

        # Plot obstacles as cylinders
        for obs in self.obstacles:
            self.draw_cylinder(ax, *obs)

        # Plot path (all points where drone has been)
        path = np.array(self.path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color='blue', linestyle='-', marker='o', markersize=3, label="Path")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Drone Navigation with SAC - Random Obstacles")
        ax.legend()
        
        plt.pause(0.01)
        plt.show()

    def draw_cylinder(self, ax, x, y, r, h, num_slices=50):
        """Draws a 3D cylinder obstacle."""
        theta = np.linspace(0, 2 * np.pi, num_slices)
        z = np.linspace(0, h, 2)
        theta_grid, z_grid = np.meshgrid(theta, z)
        
        X = r * np.cos(theta_grid) + x
        Y = r * np.sin(theta_grid) + y
        Z = z_grid
        ax.plot_surface(X, Y, Z, color='blue', alpha=0.5)
