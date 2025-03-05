

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from SAC import SACAgent
from ENV3 import DroneEnv3D
from CONFIG import STATE_DIM, ACTION_DIM, MAX_ACTION, NUM_EPISODES, MAX_STEPS

# Load trained model paths
ACTOR_PATH = "checkpoints/sac_actor_500.pth"
CRITIC_PATH = "checkpoints/sac_critic_500.pth"

# Initialize environment and agent
env = DroneEnv3D()
agent = SACAgent(STATE_DIM, ACTION_DIM, MAX_ACTION)

# Device configuration (use GPU if available)
device =  "cpu"

# Safe model loading with error handling
try:
    agent.actor.load_state_dict(torch.load(ACTOR_PATH, map_location=device))
    agent.critic.load_state_dict(torch.load(CRITIC_PATH, map_location=device))
    agent.actor.to(device)
    agent.critic.to(device)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Model files not found. Please check paths.")
    

# Matplotlib setup for 3D visualization
plt.ion()
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

MAX_EXPLORATION = 1.2  # Example max exploration efficiency value
MAX_TIME = MAX_STEPS   # Maximum possible steps in an episode
MAX_COLLISIONS = NUM_EPISODES*10   # Define an upper limit for worst case collisions

# Tracking values
test_rewards = []
test_collisions = []
test_steps = []
test_distances = []

total_exploration_efficiency = []
total_time_efficiency = []
total_collision_rate = []
# Start episodes
for episode in range(NUM_EPISODES):
    state = env.reset()  
    total_reward = 0
    visited_positions = set()  # Track unique positions
    collision_count = 0
    env.path = [env.drone_pos.copy()]  # Store drone path
    path_array = np.array(env.path)  # Precompute path array

    for step in range(MAX_STEPS):
        with torch.no_grad():
            # Convert state to tensor
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action = agent.select_action(state_tensor).tolist()

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        # Track visited positions
        visited_positions.add(tuple(env.drone_pos))  

        # Count collisions
        if env.check_collision():
            collision_count += 1
        # Update path efficiently
        env.path.append(env.drone_pos.copy())
        path_array = np.array(env.path)  # Update path once

        # Efficient rendering (update only necessary elements)
        ax.cla()  # Clear only axes, not the entire figure
        ax.set_xlim(0, env.space_size)
        ax.set_ylim(0, env.space_size)
        ax.set_zlim(0, env.space_size)

        # Plot drone and target
        ax.scatter(*env.drone_pos, color='red', s=100, label="Drone")
        ax.scatter(*env.target, color='green', s=150, label="Target")

        # Plot obstacles
        for obs in env.obstacles:
            env.draw_cylinder(ax, *obs)

        # Plot path efficiently
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                color='blue', linestyle='-', marker='o', markersize=3, label="Path")

        # ax.view_init(elev=90)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Drone Navigation with SAC - Episode {episode + 1}")
        ax.legend()

        # Flush events instead of using plt.pause (prevents freezing)
        fig.canvas.flush_events()
        time.sleep(0.05)  # Delay for smooth visualization

        if done:
            # if total_reward>0:
                # save_path = f"saved_plots/episode_{episode+1}.png"
                # plt.savefig(save_path)
                # print("Plot Saved")
            break
     # Compute percentages
        test_rewards.append(total_reward)
        test_collisions.append(collision_count)  # Get collision count from env
        test_steps.append(step)
    exploration_percent = max(0, (1 - (env.exploration_efficiency() / MAX_EXPLORATION)) * 100)
    time_percent = max(0, (1 - (step / MAX_TIME)) * 100)
    collision_percent = max(0, (1 - (collision_count / MAX_COLLISIONS)) * 100)

    # Store for tracking
    total_exploration_efficiency.append(exploration_percent)
    total_time_efficiency.append(time_percent)
    total_collision_rate.append(collision_percent)

    
    # print(f"🎯 Episode {episode + 1}: Total Reward = {total_reward}")
    print(f"🎯 Episode {episode + 1}: Total Reward = {total_reward}, "
          f"Exploration Eff: {total_exploration_efficiency}, Time Eff: {total_time_efficiency}, "
          f"Collision Rate: {total_collision_rate}")
    
# **Print Final Metrics**
np.save("test_rewards.npy", test_rewards)
np.save("test_collisions.npy", test_collisions)
np.save("test_steps.npy", test_steps)

print("\n📊 **Overall Performance** 📊")
print(f"✅ Average Exploration Efficiency: {np.mean(total_exploration_efficiency)}%")
print(f"✅ Average Time Efficiency: {np.mean(total_time_efficiency)}%")
print(f"✅ Average Collision Rate: {np.mean(total_collision_rate)}%")

plt.ioff()  # Turn off interactive mode
plt.show()  # Keep final plot open

# Close environment after testing
env.close()
