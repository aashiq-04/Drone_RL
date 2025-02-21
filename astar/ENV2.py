import numpy as np
import gymnasium as gym
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
        self.num_obstacles = 25
        self.obstacles = []
        self.grid_size = 1
        self.collision_count = 0
        self.a_star_cache = {}
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=self.space_size, shape=(12,), dtype=np.float32)
        self.drone_vel = 0.1
        self.recal_count =0
        self.reset()

    def generate_obstacle(self):
        """Generate a valid obstacle that does not overlap with others"""
        max_attempts = 100
        min_clearance = 0.7 # Minimum spacing

        for _ in range(max_attempts):
            x = random.uniform(2, 8)
            y = random.uniform(2, 8)
            radius = 0.4
            height = 7

            collision = any(
                np.linalg.norm(np.array([x, y]) - np.array([ox, oy])) < (radius + r + min_clearance)
                for ox, oy, r, h in self.obstacles
            )

            if not collision:
                return (x, y, radius, height)

        return None  # Should not happen under normal conditions

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


    def a_star_search(self, start, goal):
        """A* algorithm to find the shortest path in 3D space avoiding obstacles."""
        start, goal = tuple(np.round(start).astype(int)), tuple(np.round(goal).astype(int))
        directions = [(dx, dy, dz) for dx in (-2, -1, 0, 1, 2) for dy in (-2,-1, 0, 1,2) for dz in (-2,-1, 0, 1,2) if not (dx == dy == dz == 0)]
        
        
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: np.linalg.norm(np.array(start) - np.array(goal))}
        
        MAX = 1000
        search_steps=0
        while open_set and search_steps<MAX:
            search_steps+=1
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(np.array(current))
                    current = came_from[current]
                return path[::-1]

            for direction in directions:
                neighbor = tuple(np.array(current) + np.array(direction))
                if not (0 <= neighbor[0] < self.space_size and 0 <= neighbor[1] < self.space_size and 0 <= neighbor[2] < self.space_size):
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
        """Move the drone along the A* planned path while avoiding obstacles dynamically."""
        if not self.a_star_path:
            self.a_star_path = self.a_star_search(self.drone_pos, self.target)

        if self.a_star_path:
            next_position = np.array(self.a_star_path.pop(0),dtype=np.float64)
            self.drone_vel = (next_position - self.drone_pos) * 0.5
            self.drone_pos +=self.drone_vel

        collision = not self.is_valid_position(self.drone_pos)
        distance = np.linalg.norm(self.drone_pos - self.target)

        reward = 0
        prev_distance = np.linalg.norm(self.path[-1] - self.target) if len(self.path) > 1 else distance

        if distance < prev_distance:
            reward += 10  # Reward for moving closer
        else:
            reward -= 2  # Small penalty for moving away

        reward -= 0.1 * distance  # Small penalty for longer paths

        if collision:
            self.collision_count +=1
            print(f" Collision!! Total Collision {self.collision_count}")
            reward -= 100
            self.reset()
        done = distance < 1
        if done:
            reward += 300
            self.reset()

        self.path.append(self.drone_pos.copy())
        return self.get_observation(), reward, done, {}
    
    # def step(self, action):
        
    #     """Move the drone while dynamically updating the A* path."""
    #     if not self.a_star_path or np.linalg.norm(self.drone_pos - self.a_star_path[0]) >3:
    #         self.recal_count+=1
    #         print(f"Recalculating A* path...(Total Recalculations :{self.recal_count})")
    #         self.a_star_path = self.a_star_search(self.drone_pos, self.target)

    #     if self.a_star_path:
    #         next_position = np.array(self.a_star_path.pop(0), dtype=np.float64)
    #         self.drone_vel = (next_position - self.drone_pos) * 0.5  
    #         self.drone_pos += self.drone_vel  

    #     collision = not self.is_valid_position(self.drone_pos)
    #     distance = np.linalg.norm(self.drone_pos - self.target)

    #     reward = 0
    #     prev_distance = np.linalg.norm(self.path[-1] - self.target) if len(self.path) > 1 else distance

    #     if distance < prev_distance:
    #         reward += 10  
    #     else:
    #         reward -= 5  

    #     reward -= 0.1 * distance  

    #     if collision:
    #         self.collision_count += 1
    #         print(f"Collision!! Total Collisions: {self.collision_count}")
    #         reward -= 100
    #         self.reset()

    #     done = distance < 0.5
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
        return np.zeros(6)

    # def render(self):
    #     """Render the environment."""
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.set_xlim(0, self.space_size)
    #     ax.set_ylim(0, self.space_size)
    #     ax.set_zlim(0, self.space_size)
    #     ax.scatter(*self.drone_pos, color='red', s=100)
    #     ax.scatter(*self.target, color='green', s=150)

    #     for obs in self.obstacles:
    #         self.draw_cylinder(ax, *obs)

    #     plt.show()
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

