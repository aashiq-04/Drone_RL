import numpy as np
import gymnasium as gym
import time 
from gym import spaces
import random
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

class DroneEnv3D(gym.Env):
    def __init__(self):
        super(DroneEnv3D, self).__init__()

        self.space_size = 10  # Define 3D space size

        # Action space (movement in x, y, z)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space (position + velocity + lidar + imu)
        self.observation_space = spaces.Box(low=0, high=self.space_size, shape=(12,), dtype=np.float32)

        self.wind_strength = 0.1  # Wind effect

        self.num_obstacles = 0
        self.obstacles = []  # Obstacles will be generated on reset
        self.grid_size = 1  # Grid resolution for A*
        self.reset()
        self.start_dist  

    
    def generate_obstacle(self):
        max_attempts = 150  # Prevent infinite loops
        min_clearance = 1 # Minimum required distance between obstacles

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


    def reset(self):
        print("Resetting environment...")  # Debugging

        self.obstacles = []
        self.drone_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
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
            self.target = np.array([random.uniform(8, 10), random.uniform(8, 10), random.uniform(4,7)], dtype=np.float64)
            if self.is_valid_position(self.target) and np.linalg.norm(self.drone_pos - self.target) > 4:
                break
            attempts += 1
        print(f"Target position: {self.target}")  # Debugging

        self.a_star_path = self.a_star_search(self.drone_pos, self.target)
        self.start_dist = np.linalg.norm(self.drone_pos-self.target)
        print(f"Distance to target = {self.start_dist}")
        
        if not self.a_star_path:
            print("A* search failed! Resetting...")
            return self.reset()

        print(f"Found A* path with {len(self.a_star_path)} steps")  # Debugging
        return self.get_observation()
    # def reset(self):
    #     # Initialize drone position away from obstacles
    #     print("resetting environment . . .")
    #     valid_pos = False
    #     while not valid_pos:
    #         self.drone_pos = np.array([
    #             random.uniform(0, 3),
    #             random.uniform(0, 3),
    #             random.uniform(0, 3)
    #         ], dtype=np.float64)
    #         valid_pos = True
    #         for obs in self.obstacles:
    #             ox, oy, r, h = obs
    #             dist = np.linalg.norm(self.drone_pos[:2] - np.array([ox, oy]))
    #             if dist < r and self.drone_pos[2] < h:
    #                 valid_pos = False
    #                 break

    #     # Set random target position
    #     self.target = np.array([
    #         random.uniform(8, 10),
    #         random.uniform(8, 10),
    #         random.uniform(8, 10)
    #     ], dtype=np.float64)

    #     self.drone_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # Reset velocity
    #     self.path = [self.drone_pos.copy()]  # Path history

    #     # Generate obstacles
    #     self.obstacles = [self.generate_obstacle() for _ in range(self.num_obstacles)]

    #     # Find path using A*
    #     self.a_star_path = self.a_star_search(self.drone_pos, self.target)

    #     return self.get_observation()

    def a_star_search(self, start, goal):
        """A* algorithm to find the shortest path in 3D space avoiding obstacles."""
        start, goal = tuple(np.round(start,1).astype(int)), tuple(np.round(goal,1).astype(int))
        directions = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if not (dx == dy == dz == 0)]
        
        
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: np.linalg.norm(np.array(start) - np.array(goal))}
        
        MAX = 7000
        for _ in range(MAX):
            if not open_set:
                return [] # no path found
            _,current = heapq.heappop(open_set)
            if current==goal:
                path=[]
                while current in came_from:
                    path.append(np.array(current))
                    current=came_from[current]
                return path[::-1]
        

            for direction in directions:
                neighbor = tuple(np.array(current) + np.array(direction))
                if not (0 <= neighbor[0] < self.space_size and
                        0 <= neighbor[1] < self.space_size and 
                        0 <= neighbor[2] < self.space_size):
                    continue

                if not self.is_valid_position(neighbor):
                    continue
                
                
                
                tentative_g_score = g_score[current] + np.linalg.norm(np.array(direction))
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(goal))
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found
        # x_min,x_max = 0,10
        # y_min,y_max = 0,10
        # z_min,z_max = 0,5
        # next_position[0] = np.clip(next_position[0],x_min,x_max)
        # next_position[1] = np.clip(next_position[1],y_min,y_max)
        # next_position[2] = np.clip(next_position[2],z_min,z_max)
    def step(self, action):
        """Move the drone using SAC actions while avoiding obstacles dynamically."""
        action = np.array(action).squeeze()
        reward = 0  # Ensure reward is always initialized
        
        
        
        
        # **Use the SAC model's action to update the drone's position**
        delta_pos = np.array(action)  # SAC outputs small movement changes (dx, dy, dz)
        next_position = self.drone_pos + delta_pos  # Apply SAC action
        
        for i in range(3):
            if next_position[i]<0:
                next_position[i]=0
                reward -=30
            elif next_position[i]>self.space_size:
                next_position[i] = self.space_size-1
                reward-=30 

        # **Check if the new position is valid (not colliding with obstacles)**
        if self.is_valid_position(next_position):
            self.drone_pos = next_position  # Move if valid
        else:
            reward = -20  # Penalize invalid moves (collision risk)
            alternate = False
            for _ in range(10):
                random_adjust = np.random.uniform(-0.5,0.5,size=3)
                adjusted_pos = self.drone_pos + random_adjust
                if self.is_valid_position(adjusted_pos):
                    self.drone_pos = adjusted_pos
                    reward-=5
                    alternate = True
                    break
            if not alternate:
                print("Drone Stuck!! Applying larger movement.")
                random_adjust = np.random.uniform(-0.75, 0.75, size=3)
                adjusted_pos = self.drone_pos + random_adjust
                if self.is_valid_position(adjusted_pos):
                    self.drone_pos = adjusted_pos
                    reward -= 10
                else:
                    print("❌ Still Stuck. Resetting Drone.")
                    reward -= 50
                    return self.get_observation(), reward, False, {}
                
            

        # **Obstacle Avoidance (Using LIDAR)**
        lidar_readings = self.get_lidar_readings()
        obstacle_detected = any(distance < 0.4 for distance in lidar_readings)  # Threshold of 1 unit

        if obstacle_detected and self.is_valid_position(self.drone_pos):
            reward -= 20  # Penalize only when moving near obstacles

        # **Collision Detection**
        collision = any(
            np.linalg.norm(self.drone_pos[:2] - np.array([ox, oy])) < r
            and 0 <= self.drone_pos[2] < h  # Ensure it's within height limits
            for ox, oy, r, h in self.obstacles
        )

        if collision:
            print("💥 Collision detected! Resetting drone.")
            reward -= 100  # Heavy penalty for collisions
            self.reset()
            return self.get_observation(), reward, True, {}  # End episode

        # **Reward Function**
        distance_xy = np.linalg.norm(self.drone_pos[:2] - self.target[:2])  # XY plane distance
        distance_z = abs(self.drone_pos[2] - self.target[2])  # Z-axis distance
        distance_total = np.linalg.norm(self.drone_pos - self.target)
        previous_distance = np.linalg.norm(self.path[-1] - self.target) if len(self.path) > 1 else distance_total

        if distance_xy < 2 and  distance_z > 1.5:
            reward += 5
            self.drone_pos[2]-=0.2 # Reward for getting closer to the goal
        elif distance_total < previous_distance:
            reward += 10
        else:
            reward-=5    # Penalize moving away
    
            
        reward -= 0.1 * distance_total  # Small penalty for longer paths
        print(f"📍 Drone Position: {self.drone_pos}, Target: {self.target}, Distance: {distance_total},Reward = {reward}")
    
        # **Check if Goal is Reached**
        done = distance_total < 2.5
        if done:
            elapsed_time = time.time() - self.start_time
            reward += 300  # Large reward for reaching the goal
            print(f"🎯 Goal reached in {elapsed_time} seconds! Resetting environment.")
            final_obs = self.get_observation()
            self.reset()
            return final_obs, reward, done, {}

        # **Track Path**
        self.path.append(self.drone_pos.copy())

        return self.get_observation(), reward, done, {}


    # def step(self, action):
    #     """Move the drone along the A* planned path while avoiding obstacles dynamically."""
    #     # Get LIDAR readings
    #     lidar_readings = self.get_lidar_readings()

    #     # Check for obstacles nearby using LIDAR
    #     obstacle_detected = any(distance < 1.0 for distance in lidar_readings)  # Threshold of 1 unit

    #     # If an obstacle is detected or path is empty, recalculate A* path
    #     if obstacle_detected or not self.a_star_path:
    #         self.a_star_path = self.a_star_search(self.drone_pos, self.target)

    #     # Move along the path if available
    #     if self.a_star_path:
    #         next_position = self.a_star_path.pop(0)
    #         self.drone_pos = np.array(next_position, dtype=np.float64)

    #     # Check if the drone collides with an obstacle
    #     collision = any(
    #         np.linalg.norm(self.drone_pos[:2] - np.array([ox, oy])) < r and self.drone_pos[2] < h
    #         for ox, oy, r, h in self.obstacles
    #     )

    #     # Calculate distance to target
    #     distance = np.linalg.norm(self.drone_pos - self.target)

    #     # **Reward function**
    #     reward = 0
    #     previous_distance = np.linalg.norm(self.path[-1] - self.target) if len(self.path) > 1 else distance
        
    #     if distance < previous_distance:
    #         reward += 10  # Reward for moving closer to the goal
    #     else:
    #         reward -= 5  # Penalize moving away

    #     reward -= 0.1 * distance  # Small penalty for longer paths

    #     if collision:
    #         reward -= 100  # Heavy penalty for collision
    #         self.reset()

    #     # If goal is reached, give a high reward
    #     done = distance < 0.5
    #     if done:
    #         reward += 300  # Large reward for reaching the goal
    #         self.reset()

    #     # Track path
    #     self.path.append(self.drone_pos.copy())

    #     return self.get_observation(), reward, done, {}



    def get_observation(self):
        """Return position, velocity, lidar, and imu readings."""
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
        orientation = np.random.uniform(-0.1, 0.1, size=3)  # Simulated roll, pitch, yaw
        angular_velocity = self.drone_vel * 0.1  # Assume angular velocity depends on movement
        acceleration = np.gradient(self.drone_vel)
        return np.concatenate([orientation, angular_velocity,acceleration])

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

        ax.view_init(elev=90)
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


