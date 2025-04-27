import numpy as np
import gymnasium as gym
import time 
from gym import spaces
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from CONFIG import OBSTACLES,MAP_SIZE,MAX_ACTION


class DroneEnv3D(gym.Env):
    def __init__(self, reward_mode="default"):  # 👈 ADD reward_mode
        super(DroneEnv3D, self).__init__()
        self.reward_mode = reward_mode  # 👈 store it
        # (keep your old code here)
        self.reset()

    def reset(self):
        # (your old reset code here)
        return self.get_observation()

    def calculate_distance(self, x1, y1, z1):
        return math.sqrt((self.target[0] - x1) ** 2 + (self.target[1] - y1) ** 2 + (self.target[2] - z1) ** 2)

    def step(self, action):
        """Move the drone using SAC actions while avoiding obstacles dynamically."""
        action = np.array(action).squeeze()
        noise = np.random.normal(0, 0.1, size=action.shape)
        action = np.clip(action + noise, -MAX_ACTION, MAX_ACTION)

        reward = 0.0
        self.calculate_distance_from_obstacle()

        delta_pos = np.array(action)
        next_position = self.drone_pos + delta_pos
        next_position = np.clip(next_position, 0, self.space_size - 1)

        if not self.is_valid_position(next_position):
            reward -= 20

        self.drone_pos = next_position
        self.visited_pos.add(tuple(self.drone_pos))
        
        collision = self.check_collision()
        if collision:
            self.collision_count += 1

        distance_total = np.linalg.norm(self.drone_pos - self.target)
        previous_distance = np.linalg.norm(self.path[-1] - self.target) if len(self.path) > 1 else distance_total

        ### 🛠 NEW: Reward modes (ablation study here)
        if self.reward_mode == "default":
            # 🔵 Your current reward structure
            reward += 10 * (previous_distance - distance_total)
            if collision:
                reward -= 75
            reward -= 0.05 * distance_total
            reward -= 0.1  # step penalty
            if abs(self.dmin) > 0.5:
                reward += 5
            else:
                reward -= (0.4 - self.dmin) * 10

        elif self.reward_mode == "distance_only":
            # 🟠 Reward ONLY for getting closer to goal
            reward += 10 * (previous_distance - distance_total)

        elif self.reward_mode == "collision_penalty_only":
            # 🟢 Penalize ONLY on collision
            if collision:
                reward -= 100
            else:
                reward += 1  # small bonus for surviving

        elif self.reward_mode == "time_penalty":
            # 🔴 Penalize each step, encourage faster goal reaching
            reward -= 1

        elif self.reward_mode == "no_collision_penalty":
            # 🟣 Ignore collision penalties
            reward += 10 * (previous_distance - distance_total)
            reward -= 0.1  # time penalty

        else:
            # fallback if invalid mode
            reward -= 1

        self.D = self.calculate_distance(self.drone_pos[0], self.drone_pos[1], self.drone_pos[2])
        done = self.D < 2
        if done:
            elapsed_time = time.time() - self.start_time
            reward += 150  # reward for success
            print(f"🎯 Goal reached in {np.round(elapsed_time)} seconds!")
            final_obs = self.get_observation()
            self.reset()
            return final_obs, reward, done, {}

        self.path.append(self.drone_pos.copy())
        return self.get_observation(), reward, done, {}
    
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
    
    def get_observation(self):
        """Return position, velocity, lidar, and imu readings."""
        lidar_readings = self.get_lidar_readings()
        relative_pos = self.target - self.drone_pos
        
        velocity_magnitude = np.linalg.norm(self.drone_vel) 
        # imu_readings = self.get_imu_readings()
        # return np.concatenate([self.drone_pos, self.drone_vel,lidar_readings])
        return np.concatenate([relative_pos, self.drone_vel, lidar_readings])

    def render(self):
        """Visualize the environment in 3D with the path."""
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlim(0, self.space_size)
        ax.set_ylim(0, self.space_size)
        ax.set_zlim(0, self.space_size)

        ax.scatter(*self.drone_pos, color='red', s=100, label="Drone")
        ax.scatter(*self.target, color='green', s=150, label="Target")

        for obs in self.obstacles:
            self.draw_cylinder(ax, *obs)

        path = np.array(self.path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color='blue', marker='o', markersize=3, label="Path")

        ax.view_init(elev=75)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Drone Navigation with A*")
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
        ax.plot_surface(X, Y, Z, color='blue', alpha=0.5,edgecolor='k')


