import numpy as np
import gymnasium as gym
import time 
from gymnasium import spaces
import random

import math
import matplotlib.pyplot as plt

from CONFIG import OBSTACLES,MAP_SIZE,MAX_ACTION

class DroneEnv3D(gym.Env):
    def __init__(self):
        super(DroneEnv3D, self).__init__()
        self.space_size = MAP_SIZE  
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=self.space_size, shape=(12,), dtype=np.float32)
        self.wind_strength = 0.1  
        self.visited_pos = set()
        self.num_obstacles = OBSTACLES
        self.obstacles = []  
        self.grid_size = 1  
        self.D =0.0
        self.ds = [0.0] * 9
        self.dmin = min(self.ds)
        self.global_dm = 0.0
        self.min_distances_at_each_step = []
        self.collision_count = 0
        self.no_progress_counter = 0
        self.distance_history = []
        self.distance_window_size = 10  # Number of steps to consider for progress
        self.min_progress_threshold = 0.05  # Minimum required progress in distance
  
        
        self.reset()
    def generate_obstacle(self):
        max_attempts = 150  # Prevent infinite loops
        min_clearance = 0.7 # Minimum required distance between obstacles
        for _ in range(max_attempts):
            x = random.uniform(3, 8)
            y = random.uniform(3, 8)
            radius = 0.35 
            height = random.uniform(5, 7)

            # Ensure new obstacle doesn't overlap with existing ones
            collision = any(
                np.linalg.norm(np.array([x, y]) - np.array([ox, oy])) < (radius + r + min_clearance)
                for ox, oy, r, h in self.obstacles
            )
            if not collision:
                return (x, y, radius, height)
        return None
    
    def is_valid_position(self, pos):
        """Check if the position is valid (not inside obstacles)"""
        clearance = 0.4
        x, y, z = pos
        for ox, oy, r, h in self.obstacles:
            if np.linalg.norm(np.array([x, y]) - np.array([ox, oy])) < (r+ clearance) :
                if 0 <= z <= (h+clearance):
                    return False
        return True

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        print("Resetting environment...")  # Debugging

        self.obstacles = []
        self.drone_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.no_progress_counter = 0
        self.distance_history = []
        self.collision_count = 0
        # self.drone_pos
        self.drone_pos = np.array([
                random.uniform(0, 2),
                random.uniform(0, 2),
                random.uniform(0, 2)
            ], dtype=np.float64)
        # z = self.drone_pos[2]
        
        self.path = [self.drone_pos.copy()]  # Path histo
        attempts = 0
        while len(self.obstacles) < self.num_obstacles and attempts < 50:
            obs = self.generate_obstacle()
            if obs:
                self.obstacles.append(obs)
            attempts += 1

        print(f"Generated {len(self.obstacles)} obstacles")  # Debugging

        attempts = 0
        while attempts < 50:
            self.drone_pos = np.array([random.uniform(0, 2), random.uniform(0, 2), random.uniform(0, 2)], dtype=np.float64)
            if self.is_valid_position(self.drone_pos):
                self.start_time = time.time()
                break
            attempts += 1
        print(f"Drone position: {self.drone_pos}")  # Debugging

        attempts = 0
        while attempts < 50:
            self.target = np.array([random.uniform(7,9), random.uniform(8, 10), random.uniform(0,5)], dtype=np.float64)
            if self.is_valid_position(self.target) and np.linalg.norm(self.drone_pos - self.target) > 4:
                break
            attempts += 1
        print(f"Target position: {self.target}")  # Debugging
        self.D = self.calculate_distance(self.drone_pos[0],self.drone_pos[1],self.drone_pos[2])
        print(f"Distance to target = {self.D}")
        return self.get_observation().astype(np.float32),{}
    
    def calculate_distance(self, x1, y1, z1):
        return math.sqrt((self.target[0] - x1) ** 2 + (self.target[1] - y1) ** 2 + (self.target[2] - z1) ** 2)
    
    # def calculate_distance_from_obstacle(self):
    #     obstacle_distance = []
    #     for obstacle in self.obstacles:
    #         x, y, z, radius = obstacle
    #         d = math.sqrt((self.drone_pos[0]-x)**2 + 
    #                       (self.drone_pos[1]-y)**2 +
    #                       (self.drone_pos[2]-z)**2) - radius
    #         obstacle_distance.append(d)
    #     obstacle_distance.sort()
    #     for i in range(min(9,len(obstacle_distance))):
    #         self.ds[i] = obstacle_distance[i]
    #     self.dmin = min(self.ds)
    #     self.global_dm = min(self.global_dm, self.dmin)
    #     self.min_distances_at_each_step.append(self.dmin)
    def calculate_distance_from_obstacle(self):
        obstacle_distance = []
        for obstacle in self.obstacles:
            ox, oy, r, h = obstacle
            d = math.sqrt((self.drone_pos[0] - ox) ** 2 + 
                          (self.drone_pos[1] - oy) ** 2 +
                          (self.drone_pos[2] - h/2) ** 2) - r
            obstacle_distance.append(d)
        
        if obstacle_distance:  # Only proceed if list is not empty
            obstacle_distance.sort()
            for i in range(min(9, len(obstacle_distance))):
                self.ds[i] = obstacle_distance[i]
            self.dmin = min(self.ds)
            self.global_dm = min(self.global_dm, self.dmin)
            self.min_distances_at_each_step.append(self.dmin)
        else:
            self.dmin = float('inf')  # If no obstacles

    def check_collision(self,safety_distance=0.4):
        # print(abs(self.dmin)
        return abs(self.dmin)<=safety_distance
    def exploration_efficiency(self):
        total_cells = self.space_size**3
        return len(self.visited_pos)/total_cells
    def check_progress(self):
        """Check if drone is making progress toward target"""
        if len(self.distance_history) >= self.distance_window_size:
            # Calculate how much progress has been made over the window
            start_distance = self.distance_history[-self.distance_window_size]
            current_distance = self.distance_history[-1]
            progress = start_distance - current_distance
            
            # If minimal progress or moving away from target
            if progress < self.min_progress_threshold:
                self.no_progress_counter += 1
                return False
            else:
                self.no_progress_counter = max(0, self.no_progress_counter - 1)  # Reduce counter if making progress
                return True
        return True  # Not enough history to determine yet
    def step(self, action):
        """Move the drone using SAC actions while avoiding obstacles dynamically."""
        action = np.array(action).squeeze()
        # noise = np.random.normal(0, 0.1, size=action.shape)  # Adjust 0.2 as needed
        action = np.clip(action , -MAX_ACTION, MAX_ACTION)  # Ensure valid action range
        self.previous_D = self.D
        reward = 0.0
        
        self.calculate_distance_from_obstacle()       
        
        
        # **Use the SAC model's action to update the drone's position**
        delta_pos = np.array(action)  # SAC outputs small movement changes (dx, dy, dz)
        self.drone_pos = np.array(self.drone_pos)
        next_position = self.drone_pos + delta_pos  # Apply SAC action
        next_position = np.clip(next_position, 0, self.space_size - 1)
        if not self.check_progress():
            # Increase penalty the longer it makes no progress
            progress_penalty = min(2 * self.no_progress_counter, 30)  # Cap at -30
            reward -= progress_penalty
            

        if np.any(next_position < 0) or np.any(next_position >= self.space_size):
            reward -= 10

        if self.is_valid_position(next_position) == False:
            reward -= 20  # Penalize invalid moves (collision risk)
        elif self.dmin<0.5:
            reward -=(0.5-self.dmin)*10
        else:
            reward +=0.5
            
        lidar_readings = self.get_lidar_readings()
        obstacle_detected = any(distance < 0.4 for distance in lidar_readings)  # Threshold of 1 unit

        if obstacle_detected and self.is_valid_position(self.drone_pos):
            reward -= 50  # Penalize only when moving near obstacles

        # **Check if the new position is valid (not colliding with obstacles)**
                
        self.drone_pos = next_position 
        self.visited_pos.add(tuple(self.drone_pos)) 
            
        if abs(self.dmin)< 0.4:
            reward-=(0.4-self.dmin) * 10
        else:
            reward+=10
        # **Collision Detection**
        collision = self.check_collision()
        if collision:
            self.collision_count+=1
            print(f"💥 Collision ! ")
            reward -= 100  # Heavy penalty for collisions
        # **Reward Function**
        distance_xy = np.linalg.norm(self.drone_pos[:2] - self.target[:2])  # XY plane distance
        distance_z = abs(self.drone_pos[2] - self.target[2])  # Z-axis distance
        distance_total = np.linalg.norm(self.drone_pos - self.target)
        previous_distance = np.linalg.norm(self.path[-1] - self.target) if len(self.path) > 1 else distance_total
        
        if distance_xy < 2 and  distance_z > 2:
            reward -= 10
            # self.drone_pos[2]=-0.05
            if self.drone_pos[2] > self.target[2]:
                reward-=5
        if distance_total < 3:
            reward += (3 - distance_total) * 100  # stronger incentive
        else:
            reward += 10 * (previous_distance - distance_total)
        if self.D < 3:
            reward += (3 - self.D) * 20 
        if abs(self.drone_pos[2] - self.target[2]) > 3:
            reward -= 5
        # reward -= 0.05 * distance_total
        reward -=0.1 
        # Small penalty for longer paths
        # print(f"📍 Drone Position: {self.drone_pos}, Target: {self.target}, Distance: {distance_total},Reward = {reward}")
        
        self.D = self.calculate_distance(self.drone_pos[0],self.drone_pos[1],self.drone_pos[2])
        done = self.D < 2.5
        truncated = False
        if done:
            elapsed_time = time.time() - self.start_time
            reward += 300  # Large reward for reaching the goal
            print(f"🎯 Goal reached in {np.round(elapsed_time)} seconds! Resetting environment.")
            final_obs = self.get_observation()
            print(f"Total collision {self.collision_count}")
            self.reset()
            return final_obs.astype(np.float32), reward, done, truncated,{}
        # **Track Path**
        self.path.append(self.drone_pos.copy())

        return self.get_observation().astype(np.float32), reward, self.D <= 2, truncated,{}
    
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
                else:
                    continue
                break
        readings = readings/self.space_size
        return readings
    
    # def get_observation(self):
    #     """Return position, velocity, lidar, and imu readings."""
    #     lidar_readings = self.get_lidar_readings()
    #     relative_pos = self.target - self.drone_pos
    #     return np.concatenate([relative_pos, self.drone_vel, lidar_readings])
    
    def get_observation(self):
        """Return normalized observation vector."""
        lidar_readings = self.get_lidar_readings()
        # Normalize relative position
        relative_pos = (self.target - self.drone_pos) / self.space_size
        # Normalize velocity
        normalized_vel = self.drone_vel / MAX_ACTION
        
        return np.concatenate([relative_pos, normalized_vel, lidar_readings])
    
    
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
        ax.set_title("Drone Navigation")
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


