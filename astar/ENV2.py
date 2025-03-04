import numpy as np
import gymnasium as gym
from gym import spaces
import random
import heapq
import matplotlib.pyplot as plt

class RRTNode:
    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent
class DroneEnv3D(gym.Env):
    def __init__(self):
        super(DroneEnv3D, self).__init__()

        self.space_size = 10  # Define 3D space size
        self.num_obstacles = 2
        self.obstacles = []
        self.grid_size = 1
        self.collision_count = 0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=self.space_size, shape=(12,), dtype=np.float32)
        self.drone_vel = 0.1
        self.recal_count =0
        self.reset()

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

    def is_valid_position(self, pos):
        """Check if the position is valid (not inside obstacles)"""
        clearance = 0.3
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
        print(f"Drone position: {self.drone_pos}")  # Debugging

        attempts = 0
        while attempts < 50:
            self.target = np.array([random.uniform(8, 10), random.uniform(8, 10), random.uniform(8, 10)], dtype=np.float64)
            if self.is_valid_position(self.target) and np.linalg.norm(self.drone_pos - self.target) > 4:
                break
            attempts += 1
        print(f"Target position: {self.target}")  # Debugging

        self.a_star_path = self.a_star_search(self.drone_pos, self.target)
        
        if not self.a_star_path:
            print("A* search failed! Resetting...")
            return self.reset()

        print(f"Found A* path with {len(self.a_star_path)} steps")  # Debugging
        return self.get_observation()

    

    def rrt_search(self, start, goal, max_iter=5000, step_size=1.0):
        """ RRT algorithm to find a feasible path in 3D avoiding obstacles """
        start_node = RRTNode(start)
        goal_node = RRTNode(goal)
        
        tree = [start_node]
        
        for _ in range(max_iter):
            # Sample a random point in space (sometimes sample the goal)
            if random.random() < 0.1:  # 10% chance to bias towards goal
                sample_point = goal
            else:
                sample_point = np.random.randint(0, self.space_size, 3)

            # Find nearest node in tree
            nearest_node = min(tree, key=lambda node: np.linalg.norm(node.position - sample_point))
            
            # Move step_size towards the sampled point
            direction = sample_point - nearest_node.position
            direction = direction / np.linalg.norm(direction)  # Normalize
            new_position = nearest_node.position + step_size * direction
            new_position = np.round(new_position).astype(int)

            # Check if the new position is valid
            if not self.is_valid_position(tuple(new_position)):
                continue
            
            # Add new node to tree
            new_node = RRTNode(new_position, parent=nearest_node)
            tree.append(new_node)

            # If goal is reached
            if np.linalg.norm(new_position - goal) < step_size:
                path = []
                while new_node:
                    path.append(new_node.position)
                    new_node = new_node.parent
                return path[::-1]  # Return reversed path

        return []  # No path found

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


    def step(self, action):
        
        """Move the drone using SAC for local movement and A* for global guidance."""
        
        # If A* path is empty or needs recomputing, find a new path
        if not self.a_star_path or not self.is_valid_position(self.a_star_path[0]) or np.linalg.norm(self.target - self.a_star_path[-1]) < 1:
            self.a_star_path = self.a_star_search(self.drone_pos, self.target)

        # Get LIDAR readings for obstacle detection
        lidar_readings = self.get_lidar_readings()  
        imu = self.get_imu_readings()
        # Define a baseline movement from SAC action
        sac_action = np.array(action, dtype=np.float64) * 1.0  # Scale SAC action
        
        # if any(lidar_readings < 0.7):

        #     avoidance_vector = np.zeros(3)
        #     for i,distance in enumerate(lidar_readings):
        #         if distance < 1.0:
        #             weight = (1.5 -distance)/1.5
        #             if i == 0: avoidance_vector[0]-=weight
        #             # elif i == 1: avoidance_vector[0] +=weight
        #             elif i == 2: avoidance_vector[1] -=weight
        #             elif i == 3: avoidance_vector[1] +=weight
        #             elif i == 4: avoidance_vector[2] -=weight
        #             elif i == 5: avoidance_vector[2] +=weight
            
        #     avoidance_vector = avoidance_vector/(np.linalg.norm(avoidance_vector)+1e-8)
        #     sac_action += 0.2 * avoidance_vector
        

        # If A* path exists, use the next point for direction
        imu_correction = imu[:3] * -0.05
        if self.a_star_path:
            a_star_target = np.array(self.a_star_path[0], dtype=np.float64)  
            a_star_direction = a_star_target - self.drone_pos  # Direction from A*
            if np.linalg.norm(a_star_direction) < 0.2:
                a_star_direction = np.array([np.random.uniform(-1, 1) for _ in range(3)])
            a_star_direction /= (np.linalg.norm(a_star_direction) + 1e-8)  # Normalize
            # Weighted movement: 70% SAC, 30% A* guidance
            movement = 0.8 * sac_action + 0.2 * a_star_direction  + 0.1* imu_correction[:3]
            
        else:
            # No A* path, SAC takes full control
            movement = sac_action  
            
        reward = 0
        distance = np.linalg.norm(self.drone_pos - self.target)
        done = distance < 2  
        if done:
            self.drone_vel = np.zeros(3)
            reward += 500  
            print("🎯 Target Reached! Resetting Environment.")
            self.reset()
            return self.get_observation(),reward,done,{}

        # Update drone position
        self.drone_vel = np.clip(movement,-0.5,1.5)
        self.drone_vel = np.reshape(self.drone_vel,(3))
        self.drone_pos += self.drone_vel  

        # Ensure drone position is correctly shaped (avoid broadcasting error)
        # self.drone_pos = np.reshape(self.drone_pos, (1,3))  

        # Collision Check
        collision = not self.is_valid_position(self.drone_pos)
        # Reward Calculation

        prev_distance = np.linalg.norm(self.path[-1] - self.target) if len(self.path) > 1 else distance

        # Reward for moving closer
        if distance < prev_distance:
            reward += 20
        else:
            reward -= 10  

        # Penalty if an obstacle is nearby
        if any(lidar_readings < 1):
            reward -= 10 

        # Penalty for longer paths
        reward -= 0.1 * distance  
        if np.linalg.norm(self.drone_vel) < 0.3:  # Not moving much
            reward -= 20 

        # Large penalty for collision
        if collision:
            self.collision_count += 1
            print(f"💥 Collision! Total Collisions: {self.collision_count}")
            reward -= 150
            # self.reset()

        # Ensure done only happens when the drone reaches the target

        # Store path history
        self.path.append(self.drone_pos.copy())
        
        

        # Pass LIDAR readings to SAC as part of the state
        state = np.concatenate([self.drone_pos, self.drone_vel,lidar_readings,self.get_imu_readings()])  

        return state, reward, done, {}

    
    # def step(self, action):
    #     """Move the drone along the A* planned path while avoiding obstacles dynamically using LIDAR."""

    #     if not self.a_star_path:
    #         self.a_star_path = self.a_star_search(self.drone_pos, self.target)

    #     lidar_readings = self.get_lidar_readings()  # Get LIDAR data

    #     # Check for close obstacles using LIDAR (Threshold distance = 2 units)
    #     close_obstacle = any(lidar_readings < 2)  

    #     # If an obstacle is too close, adjust action to avoid it
    #     if close_obstacle:
    #         # Move away from the closest obstacle (simple avoidance logic)
    #         safe_direction = np.argmax(lidar_readings)  # Pick direction with max distance
    #         adjustment = np.array([
    #             1 if safe_direction == 0 else -1 if safe_direction == 1 else 0,
    #             1 if safe_direction == 2 else -1 if safe_direction == 3 else 0,
    #             1 if safe_direction == 4 else -1 if safe_direction == 5 else 0,
    #         ])
    #         self.drone_pos += adjustment  # Move slightly away

    #     # Continue following A* path
    #     if self.a_star_path:
    #         next_position = np.array(self.a_star_path.pop(0), dtype=np.float64)
    #         self.drone_vel = (next_position - self.drone_pos) * 0.5
    #         self.drone_pos += self.drone_vel

    #     collision = not self.is_valid_position(self.drone_pos)
    #     distance = np.linalg.norm(self.drone_pos - self.target)

    #     reward = 0
    #     prev_distance = np.linalg.norm(self.path[-1] - self.target) if len(self.path) > 1 else distance

    #     # Reward/Penalty for getting closer or moving away
    #     if distance < prev_distance:
    #         reward += 10  # Reward for moving closer
    #     else:
    #         reward -= 2  # Small penalty for moving away

    #     # Extra penalty if LIDAR detects an obstacle nearby
    #     if close_obstacle:
    #         reward -= 20  # Encourage avoiding obstacles

    #     # Additional penalty for longer paths
    #     reward -= 0.1 * distance  

    #     # Large penalty for collision
    #     if collision:
    #         self.collision_count += 1
    #         print(f"💥 Collision! Total Collisions: {self.collision_count}")
    #         reward -= 100
    #         # self.reset()

    #     # Reward for reaching target
    #     done = distance < 1
    #     if done:
    #         reward += 300
    #         self.reset()

    #     self.path.append(self.drone_pos.copy())

    #     return self.get_observation(), reward, done, {}

    

    def get_observation(self):
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

