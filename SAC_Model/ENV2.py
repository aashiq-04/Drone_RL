import numpy as np
import gymnasium as gym
import time 
from gym import spaces
import random
import heapq
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
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
        self.D = 0.0
        self.ds = [0.0] * 9
        self.dmin = float('inf')
        self.global_dm = float('inf')
        self.min_distances_at_each_step = []
        self.collision_count = 0
        self.safety_margin = 0.3  # Safety margin around obstacles
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
        x, y, z = pos
        for ox, oy, r, h in self.obstacles:
            # Calculate horizontal distance to obstacle center
            horizontal_dist = np.linalg.norm(np.array([x, y]) - np.array([ox, oy]))
            
            # Check if inside or too close to obstacle
            if horizontal_dist < (r + self.safety_margin) and 0 <= z <= (h + self.safety_margin):
                return False
        return True

    def calculate_obstacle_distance(self, pos):
        """Calculate minimum distance from a position to any obstacle"""
        x, y, z = pos
        min_dist = float('inf')
        
        for ox, oy, r, h in self.obstacles:
            # Calculate horizontal distance
            horizontal_dist = np.linalg.norm(np.array([x, y]) - np.array([ox, oy])) - r
            
            # Check if we're within the height of the obstacle
            if 0 <= z <= h:
                min_dist = min(min_dist, horizontal_dist)
            else:
                # If we're above the obstacle, consider vertical distance too
                vertical_dist = min(abs(z), abs(z - h))
                dist = math.sqrt(horizontal_dist**2 + vertical_dist**2) if horizontal_dist < 0 else horizontal_dist
                min_dist = min(min_dist, dist)
                
        return min_dist
    
    def find_safe_direction(self, current_pos, action, step_size=0.1):
        """Find a safe direction similar to the requested action"""
        # Original direction
        original_direction = action / (np.linalg.norm(action) + 1e-10)
        
        # Try the original direction first
        new_pos = current_pos + action
        if self.is_valid_position(new_pos):
            return action
            
        # If original direction isn't safe, try to find a safe direction
        # Generate alternative directions by rotating original direction
        best_direction = None
        best_score = -float('inf')
        
        # Try different angles in the XY plane
        for angle in np.linspace(0, 2*np.pi, 16):
            # Rotate in XY plane
            rot_direction = np.array([
                original_direction[0] * np.cos(angle) - original_direction[1] * np.sin(angle),
                original_direction[0] * np.sin(angle) + original_direction[1] * np.cos(angle),
                original_direction[2]
            ])
            
            # Try different elevation angles
            for elev in np.linspace(-np.pi/4, np.pi/4, 5):
                elev_direction = np.array([
                    rot_direction[0] * np.cos(elev),
                    rot_direction[1] * np.cos(elev),
                    rot_direction[2] + np.sin(elev)
                ])
                
                # Normalize
                elev_direction = elev_direction / (np.linalg.norm(elev_direction) + 1e-10)
                
                # Scale to similar magnitude as original action
                mag = np.linalg.norm(action)
                test_action = elev_direction * mag * step_size
                
                # Check if this direction is safe
                test_pos = current_pos + test_action
                if self.is_valid_position(test_pos):
                    # Calculate how similar this is to original direction
                    similarity = np.dot(original_direction, elev_direction)
                    
                    # Distance to target in this direction
                    target_alignment = -np.linalg.norm(test_pos - self.target)
                    
                    # Score this direction (prefer similar to original and heading toward target)
                    score = similarity * 0.4 + target_alignment * 0.6
                    
                    if score > best_score:
                        best_score = score
                        best_direction = test_action
        
        # If we found a safe direction, return it
        if best_direction is not None:
            return best_direction
            
        # If all else fails, return a very small action in the safest direction
        if np.linalg.norm(action) > 0:
            safest_direction = (self.target - current_pos)
            safest_direction = safest_direction / (np.linalg.norm(safest_direction) + 1e-10)
            return safest_direction * 0.05
        
        return np.zeros_like(action)

    def reset(self):
        print("Resetting environment...")  # Debugging

        self.obstacles = []
        self.drone_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.collision_count = 0
        self.drone_pos = np.array([
                random.uniform(0, 2),
                random.uniform(0, 2),
                random.uniform(0, 2)
            ], dtype=np.float64)
        
        self.path = [self.drone_pos.copy()]  # Path history
        attempts = 0
        while len(self.obstacles) < self.num_obstacles and attempts < 50:
            obs = self.generate_obstacle()
            if obs:
                self.obstacles.append(obs)
            attempts += 1

        print(f"Generated {len(self.obstacles)} obstacles")  # Debugging

        attempts = 0
        while attempts < 50:
            self.drone_pos = np.array([random.uniform(0, 4), random.uniform(0, 2), random.uniform(0, 4)], dtype=np.float64)
            if self.is_valid_position(self.drone_pos):
                self.start_time = time.time()
                break
            attempts += 1
        print(f"Drone position: {self.drone_pos}")  # Debugging

        attempts = 0
        while attempts < 50:
            self.target = np.array([random.uniform(7,9), random.uniform(8, 10), random.uniform(8,9)], dtype=np.float64)
            if self.is_valid_position(self.target) and np.linalg.norm(self.drone_pos - self.target) > 4:
                break
            attempts += 1
        print(f"Target position: {self.target}")  # Debugging
        self.D = self.calculate_distance(self.drone_pos[0],self.drone_pos[1],self.drone_pos[2])
        print(f"Distance to target = {self.D}")
        
        # Calculate initial obstacle distances
        self.calculate_distance_from_obstacle()
        return self.get_observation()
    
    def calculate_distance(self, x1, y1, z1):
        return math.sqrt((self.target[0] - x1) ** 2 + (self.target[1] - y1) ** 2 + (self.target[2] - z1) ** 2)
    
    def calculate_distance_from_obstacle(self):
        """Calculate distances to nearest obstacles"""
        obstacle_distances = []
        for obstacle in self.obstacles:
            ox, oy, r, h = obstacle
            horizontal_dist = np.linalg.norm(np.array([self.drone_pos[0], self.drone_pos[1]]) - np.array([ox, oy])) - r
            
            # If we're within height range of obstacle
            if 0 <= self.drone_pos[2] <= h:
                obstacle_distances.append(horizontal_dist)
            else:
                # If we're above or below, consider vertical distance too
                vertical_dist = min(abs(self.drone_pos[2]), abs(self.drone_pos[2] - h))
                dist = math.sqrt(horizontal_dist**2 + vertical_dist**2) if horizontal_dist < 0 else horizontal_dist
                obstacle_distances.append(dist)
        
        obstacle_distances.sort()
        self.ds = obstacle_distances[:9] + [float('inf')] * (9 - len(obstacle_distances[:9]))
        self.dmin = min(self.ds) if self.ds else float('inf')
        self.global_dm = min(self.global_dm, self.dmin)
        self.min_distances_at_each_step.append(self.dmin)

    def check_collision(self, safety_distance=0.2):
        return self.dmin <= safety_distance
    
    def exploration_efficiency(self):
        total_cells = self.space_size**3
        return len(self.visited_pos)/total_cells
        
    def step(self, action):
        """Move the drone using SAC actions while avoiding obstacles dynamically."""
        action = np.array(action).squeeze()
        noise = np.random.normal(0, 0.1, size=action.shape)  # Adjust 0.1 as needed
        action = np.clip(action + noise, -MAX_ACTION, MAX_ACTION)  # Ensure valid action range

        reward = 0.0
        
        # Calculate current distance from obstacles
        self.calculate_distance_from_obstacle()
        
        # Store current position before moving
        prev_pos = self.drone_pos.copy()
        prev_distance_to_target = self.D
        
        # Apply hard-coded obstacle avoidance
        # If we're too close to an obstacle, find a safe direction
        if self.dmin < self.safety_margin * 2:
            # The closer we are to an obstacle, the more we prioritize avoidance
            avoidance_weight = max(0, 1 - (self.dmin / (self.safety_margin * 2)))
            
            # Find safe direction (modified action)
            safe_action = self.find_safe_direction(self.drone_pos, action)
            
            # Blend original action with safe action based on how close we are to obstacles
            action = (1 - avoidance_weight) * action + avoidance_weight * safe_action
        
        # Calculate next position using the (potentially modified) action
        next_position = self.drone_pos + action
        
        # Keep within bounds
        next_position = np.clip(next_position, 0, self.space_size - 1)
        
        # Check if next position is valid
        if not self.is_valid_position(next_position):
            # If invalid, don't move there - find a safe alternative
            alternative_action = self.find_safe_direction(self.drone_pos, action * 0.5)
            next_position = self.drone_pos + alternative_action
            next_position = np.clip(next_position, 0, self.space_size - 1)
            
            # If still invalid, don't move
            if not self.is_valid_position(next_position):
                next_position = self.drone_pos
                reward -= 10  # Penalty for being stuck
        
        # Update drone position
        self.drone_pos = next_position
        self.visited_pos.add(tuple(self.drone_pos))
        
        # Recalculate distances after moving
        self.calculate_distance_from_obstacle()
        
        # Reward based on obstacle distance
        if self.dmin < self.safety_margin:
            reward -= (self.safety_margin - self.dmin) * 20
        else:
            reward += min(5, self.dmin)  # Reward for staying away from obstacles
            
        # Collision detection (should rarely happen with our avoidance logic)
        collision = self.check_collision()
        if collision:
            self.collision_count += 1
            print(f"💥 Collision detected! Position: {self.drone_pos}")
            reward -= 100  # Heavy penalty for collisions
            
            # Move slightly away from the obstacle
            to_target = self.target - self.drone_pos
            to_target = to_target / (np.linalg.norm(to_target) + 1e-10)
            self.drone_pos = self.drone_pos + to_target * 0.3
            
        # Calculate distance to target
        self.D = self.calculate_distance(self.drone_pos[0], self.drone_pos[1], self.drone_pos[2])
        
        # Reward for moving closer to target
        reward += 30 * (prev_distance_to_target - self.D)
        
        # Penalize zigzagging
        if len(self.path) > 2:
            direction_changes = np.linalg.norm(
                (self.drone_pos - self.path[-1]) - (self.path[-1] - self.path[-2])
            )
            reward -= direction_changes * 2
            
        # Z-axis alignment penalty
        distance_z = abs(self.drone_pos[2] - self.target[2])
        if distance_z > 3:
            reward -= 5
            
        # Small penalty for each step
        reward -= 0.1 
        
        # Check if we've reached the target
        done = self.D < 1.5
        if done:
            elapsed_time = time.time() - self.start_time
            reward += 500  # Large reward for reaching the goal
            print(f"🎯 Goal reached in {np.round(elapsed_time)} seconds! Collisions: {self.collision_count}")
            final_obs = self.get_observation()
            self.reset()
            return final_obs, reward, done, {}
            
        # Track path
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
            min_distance = self.space_size  # Start with maximum possible distance
            
            for distance in np.linspace(0, self.space_size, 100):
                point = self.drone_pos + direction * distance
                
                # Check boundaries
                if np.any(point < 0) or np.any(point >= self.space_size):
                    min_distance = distance
                    break
                    
                # Check obstacles
                for ox, oy, r, h in self.obstacles:
                    horizontal_dist = np.linalg.norm(point[:2] - np.array([ox, oy]))
                    if horizontal_dist < r and 0 <= point[2] <= h:
                        min_distance = distance
                        break
                        
                # If we found an intersection, break out of the distance loop
                if min_distance <= distance:
                    break
                    
            readings[i] = min_distance
        
        return readings
    
    def get_observation(self):
        """Return position, velocity, lidar, and imu readings."""
        lidar_readings = self.get_lidar_readings()
        relative_pos = self.target - self.drone_pos
        
        velocity_magnitude = np.linalg.norm(self.drone_vel) 
        return np.concatenate([relative_pos, self.drone_vel, lidar_readings])

    def render(self):
        """Visualize the environment in 3D with the path."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlim(0, self.space_size)
        ax.set_ylim(0, self.space_size)
        ax.set_zlim(0, self.space_size)

        # Plot obstacles
        for obs in self.obstacles:
            self.draw_cylinder(ax, *obs)
        
        # Plot path
        path = np.array(self.path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color='blue', marker='.', markersize=2, label="Path")
        
        # Plot drone and target
        ax.scatter(*self.drone_pos, color='red', s=100, label="Drone")
        ax.scatter(*self.target, color='green', s=150, label="Target")
        
        # Draw safety sphere around drone
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = self.safety_margin * np.outer(np.cos(u), np.sin(v)) + self.drone_pos[0]
        y = self.safety_margin * np.outer(np.sin(u), np.sin(v)) + self.drone_pos[1]
        z = self.safety_margin * np.outer(np.ones(np.size(u)), np.cos(v)) + self.drone_pos[2]
        ax.plot_surface(x, y, z, color='red', alpha=0.1)

        ax.view_init(elev=35, azim=45)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Drone Navigation with Obstacle Avoidance")
        ax.legend()
        
        plt.pause(0.01)
        plt.show()

    def draw_cylinder(self, ax, x, y, r, h, num_slices=30):
        """Draws a 3D cylinder obstacle."""
        theta = np.linspace(0, 2 * np.pi, num_slices)
        z = np.linspace(0, h, 2)
        theta_grid, z_grid = np.meshgrid(theta, z)
        
        X = r * np.cos(theta_grid) + x
        Y = r * np.sin(theta_grid) + y
        Z = z_grid
        ax.plot_surface(X, Y, Z, color='blue', alpha=0.5, edgecolor='k')