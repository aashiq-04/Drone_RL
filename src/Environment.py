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
        # self.target = np.array([9, 9, 9], dtype=np.float64)  # Target position

        # Continuous action space (x, y, z velocity changes)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space (position + velocity + lidar + imu)
        self.observation_space = spaces.Box(low=0, high=self.space_size, shape=(12,), dtype=np.float32)

        # Wind effect
        self.wind_strength = 0.1  

        # Define obstacles (random poles)
        self.num_obstacles = 10
        self.obstacles = []  # Initialize empty, generate on reset

        self.reset()


    def is_valid_position(self, pos):
        """Check if the position is valid (not inside obstacles)"""
        clearance = 0.4
        x, y, z = pos
        for ox, oy, r, h in self.obstacles:
            if np.linalg.norm(np.array([x, y]) - np.array([ox, oy])) < (r+ clearance) :
                if 0 <= z <= (h+clearance):
                    return False
        return True
    def generate_obstacle(self):
        """Generate a valid obstacle that does not overlap with others"""
        max_attempts = 150
        min_clearance = 0.7 # Minimum spacing

        for _ in range(max_attempts):
            x = random.uniform(2, 8)
            y = random.uniform(2, 8)
            radius = 0.4
            height = random.uniform(5,7)

            collision = any(
                np.linalg.norm(np.array([x, y]) - np.array([ox, oy])) < (radius + r + min_clearance)
                for ox, oy, r, h in self.obstacles
            )

            if not collision:
                return (x, y, radius, height)

        return None  # Should not happen under normal conditions

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
                break
            attempts += 1
        attempts = 0
        while attempts < 50:
            self.target = np.array([random.uniform(8, 10), random.uniform(8, 10), random.uniform(8, 10)], dtype=np.float64)
            if self.is_valid_position(self.target) and np.linalg.norm(self.drone_pos - self.target) > 4:
                break
            attempts += 1 # Debugging
        return self.get_observation()

    # def step(self, action):
        
    #     action = np.clip(action, -1, 1)  # Keep actions within valid range

    #     # Update velocity and position
    #     self.drone_vel = 0.9 * self.drone_vel + 0.1 * action  # Smooth velocity update
    #     self.drone_pos += self.drone_vel  # Move the drone
        
    #     # Add wind effect
    #     wind = np.random.uniform(-self.wind_strength, self.wind_strength, 3)
    #     self.drone_pos += wind * 0.3
        
    #     # Keep within boundaries
    #     self.drone_pos = np.clip(self.drone_pos, 0, self.space_size)
        
    #     # Sensor Data
    #     lidar_readings = self.get_lidar_readings()
    #     imu_readings = self.get_imu_readings()
    #     angluar_corrections = -0.1 * imu_readings[3:]
    #     self.drone_vel += angluar_corrections

    #     # Avoid Obstacles
    #     safe_action = action.copy()
    
    #     if lidar_readings[2] < 1.0:  # Front obstacle
    #         safe_action[1] -=0.1  # Move backward

    #     if lidar_readings[0] < 1.0:  # Right obstacle
    #         safe_action[0] -=0.1  # Move left

    #     if lidar_readings[1] < 1.0:  # Left obstacle
    #         safe_action[0] += 0.1  # Move right
            
    #     self.drone_vel = 0.9 * self.drone_vel + 0.1 * safe_action
    #     self.drone_pos += self.drone_vel
        
    #     #keeping drone withing boundaries
    #     self.drone_pos = np.clip(self.drone_pos, 0, self.space_size)
        
    #     # Collision detection (simplified for cylinder obstacles)
    #     collision = not self.is_valid_position(self.drone_pos)
        

    #     # Reward function
    #     distance = np.linalg.norm(self.drone_pos - self.target)
    #     reward = -distance  # Closer = better
    #     reward -= 0.1 * np.linalg.norm(self.drone_vel)
    #     #boundary penalty 
    #     boundary_penalty = np.sum(np.clip(self.drone_pos - [0,0,0],0,2))+ np.sum(np.clip([10,10,10]-self.drone_pos,0,2))
    #     reward -=boundary_penalty

    #     if collision:
    #         reward -= 100  # Penalize collision
    #         self.drone_pos = np.array([1, 1, 1], dtype=np.float64)  # Reset position after crash

    #     done = distance < 1
    #     if done:
    #         reward += 200  # Reward reaching goal
    #         self.reset()

    #     self.path.append(self.drone_pos.copy())  # Track path
    #     return self.get_observation(), reward, done, {}
    
    def step(self, action):
        # Ensure action values are within valid range
        action = np.clip(action, -1, 1)
        reward =0
        # Smooth velocity update
        self.drone_vel = 0.9 * self.drone_vel + 0.1 * action  

        # Apply movement
        self.drone_pos += self.drone_vel  

        # Add wind effect
        wind = np.random.uniform(-self.wind_strength, self.wind_strength, 3)
        self.drone_pos += wind * 0.3  

        # Keep drone within boundaries
        self.drone_pos = np.clip(self.drone_pos, 0, self.space_size)

        # Sensor Readings
        lidar_readings = self.get_lidar_readings()
        imu_readings = self.get_imu_readings()

        # Adjust velocity based on angular drift from IMU
        angular_corrections = -0.1 * imu_readings[3:]  # Assuming last 3 IMU readings are angular
        self.drone_vel += angular_corrections[:3]
        safe_action = np.array(action,dtype=np.float64)
        # 🛑 Obstacle Avoidance (Updated with Priority System)
        if any(lidar_readings < 0.7):
    
            avoidance_vector = np.zeros(3)
            for i,distance in enumerate(lidar_readings):
                if distance < 1.0:
                    weight = (1.5 -distance)/1.5
                    if i == 0: avoidance_vector[0]-=weight
                    # elif i == 1: avoidance_vector[0] +=weight
                    elif i == 2: avoidance_vector[1] -=weight
                    elif i == 3: avoidance_vector[1] +=weight
                    elif i == 4: avoidance_vector[2] -=weight
                    elif i == 5: avoidance_vector[2] +=weight
            
            avoidance_vector = avoidance_vector/(np.linalg.norm(avoidance_vector)+1e-8)
            safe_action += 0.2 * avoidance_vector

        # Update velocity with safe action
        self.drone_vel = 0.9 * self.drone_vel + 0.1 * safe_action
        gravity = np.array([0, 0, -0.05])  # Small downward force
        self.drone_vel += gravity
        self.drone_pos += self.drone_vel  

        # Keep within boundaries again after movement
        self.drone_pos = np.clip(self.drone_pos, 0, self.space_size)

        # 🏗️ Collision Detection (More Flexible Check)
        collision = not self.is_valid_position(self.drone_pos)
        distance = np.linalg.norm(self.drone_pos - self.target)
        done = distance < 2  
        if done:
            self.drone_vel *=0.5
            reward += 500  
            print("🎯 Target Reached! Resetting Environment.")
            self.reset()
        
        reward = 0
        prev_distance = np.linalg.norm(self.path[-1] - self.target) if len(self.path) > 1 else distance
        reward += 30*(prev_distance-distance)

        # 📈 Reward Function (Improved)
        
        reward -= 0.1 * distance  
        if np.linalg.norm(self.drone_vel) < 0.3:  # Not moving much
            reward -= 20 

        # Large penalty for collision
        if collision:
            self.collision_count += 1
            print(f"💥 Collision! Total Collisions: {self.collision_count}")
            reward -= 100
            self.drone_pos -= self.drone_vel * 3

        

        # # 🎯 Goal Reward
        # if done:
        #     reward += 200  # Reward for reaching the target
        #     self.reset()  # Reset after reaching the goal

        # 📝 Track drone path for visualization/debugging
        self.path.append(self.drone_pos.copy())

        return self.get_observation(), reward, done, {}

    
    
    
    
    def get_observation(self):
        """Combine position, velocity, and sensor data into a single observation."""
        lidar_readings = self.get_lidar_readings()
        imu_readings = self.get_imu_readings()
        return np.concatenate([self.drone_pos, self.drone_vel, lidar_readings, imu_readings])

    def get_lidar_readings(self):
        """Simulate LIDAR readings (distances in six directions: left, right, front, back, up, down)."""
        readings = np.full(6, self.space_size)  # Initialize readings with max range
        directions = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),  # Right, Left
            np.array([0, 1, 0]), np.array([0, -1, 0]),  # Front, Back
            np.array([0, 0, 1]), np.array([0, 0, -1])   # Up, Down
        ]
        
        for i, direction in enumerate(directions):
            for distance in np.linspace(0, self.space_size, 100):
                point = self.drone_pos + direction * distance
                
                # Check if the point is out of bounds
                if np.any(point < 0) or np.any(point > self.space_size):
                    readings[i] = distance
                    break  # Stop checking further
                
                # Check collision with obstacles
                for obs in self.obstacles:
                    ox, oy, r, h = obs  # Obstacle position (ox, oy), radius (r), and height (h)
                    
                    # 2D collision check (X-Y plane) & height constraint
                    if np.linalg.norm(point[:2] - np.array([ox, oy])) < r and 0 <= point[2] < h:
                        readings[i] = distance
                        break  # Stop checking further

                else:
                    continue  # Continue the outer loop if no obstacle was detected
                break  # Break outer loop if an obstacle was found

        return readings


    def get_imu_readings(self, dt=0.1):  
        """Simulate IMU readings (orientation, angular velocity, and acceleration)."""
    
        # Simulated small variations in roll, pitch, and yaw
        orientation = np.random.uniform(-0.1, 0.1, size=3)  
        
        # Estimate angular velocity using finite differences (if history is available)
        if hasattr(self, "prev_orientation"):
            angular_velocity = (orientation - self.prev_orientation) / dt
        else:
            angular_velocity = np.zeros(3)  # No previous data, assume no rotation
        
        self.prev_orientation = orientation  # Store for next time step
        
        # Estimate acceleration
        if hasattr(self, "prev_velocity"):
            acceleration = (self.drone_vel - self.prev_velocity) / dt
        else:
            acceleration = np.zeros(3)  # No previous velocity, assume zero acceleration
        
        self.prev_velocity = self.drone_vel  # Store for next time step
        
        return np.concatenate([orientation, angular_velocity, acceleration])


    def render(self):
        """Render the environment dynamically without slowing down."""
        if not hasattr(self, "fig"):
            # Create figure only once
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.clear()  # Clear the previous frame

        # Set limits
        self.ax.set_xlim([0, self.space_size])
        self.ax.set_ylim([0, self.space_size])
        self.ax.set_zlim([0, self.space_size])

        # Plot obstacles
        for obs in self.obstacles:
            ox, oy, oz, r = obs
            u, v = np.mgrid[0 : 2 * np.pi : 10j, 0 : np.pi : 5j]
            x = ox + r * np.cos(u) * np.sin(v)
            y = oy + r * np.sin(u) * np.sin(v)
            z = oz + r * np.cos(v)
            self.ax.plot_wireframe(x, y, z, color="r", alpha=0.5)

        # Plot drone path
        path = np.array(self.path)
        if len(path) > 1:
            self.ax.plot(path[:, 0], path[:, 1], path[:, 2], "b-", label="Path")

        # Plot start, target, and drone position
        self.ax.scatter(*self.start, color="green", s=100, label="Start")
        self.ax.scatter(*self.target, color="red", s=100, label="Target")
        self.ax.scatter(*self.drone_pos, color="blue", s=100, label="Drone")

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Drone Navigation")

        plt.legend()
        plt.pause(0.01)  # Efficient real-time rendering
    def draw_cylinder(self, ax, x, y, r, h, num_slices=50):
        """Draws a 3D cylinder obstacle."""
        theta = np.linspace(0, 2 * np.pi, num_slices)
        z = np.linspace(0, h, 2)
        theta_grid, z_grid = np.meshgrid(theta, z)
        
        X = r * np.cos(theta_grid) + x
        Y = r * np.sin(theta_grid) + y
        Z = z_grid
        ax.plot_surface(X, Y, Z, color='blue', alpha=0.5,edgecolor='k')