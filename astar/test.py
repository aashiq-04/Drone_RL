# import torch
# import numpy as np
# import time  # For visualization delay
# from SAC import SACAgent
# from Env import DroneEnv3D
# import matplotlib.pyplot as plt
# from CONFIG import STATE_DIM,ACTION_DIM,MAX_ACTION,NUM_EPISODES,MAX_STEPS
# # Load trained model
# ACTOR_PATH = "checkpoints1/sac_actor_final.pth"
# CRITIC_PATH = "checkpoints1/sac_critic_final.pth"

# env = DroneEnv3D()
# agent = SACAgent(STATE_DIM, ACTION_DIM, MAX_ACTION)

# # Load model weights safely
# agent.actor.load_state_dict(torch.load(ACTOR_PATH))
# agent.critic.load_state_dict(torch.load(CRITIC_PATH))
# # agent.actor.eval()  # Set to evaluation mode
# # agent.critic.eval()

# print("✅ Model loaded successfully!")

# # Interactive mode for Matplotlib
# plt.ion()
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')


# for episode in range(NUM_EPISODES):
#     state = env.reset()  # Ensure this returns a valid state
#     total_reward = 0
#     env.path = [env.drone_pos.copy()]  # Reset path for each episode

#     for step in range(MAX_STEPS):
#         with torch.no_grad():
#             action = agent.select_action(state).tolist()  # Convert NumPy array to list

#         next_state, reward, done, _ = env.step(action)
#         total_reward += reward
#         state = next_state

#         # Render environment for visualization
#         ax.clear()
#         ax.set_xlim(0, env.space_size)
#         ax.set_ylim(0, env.space_size)
#         ax.set_zlim(0, env.space_size)

#         # Plot drone
#         ax.scatter(*env.drone_pos, color='red', s=100, label="Drone")

#         # Plot target
#         ax.scatter(*env.target, color='green', s=150, label="Target")

#         # Plot obstacles as cylinders
#         for obs in env.obstacles:
#             env.draw_cylinder(ax, *obs)

#         # Plot path (all points where drone has been)
#         path = np.array(env.path)
#         ax.plot(path[:, 0], path[:, 1], path[:, 2], color='blue', linestyle='-', marker='o', markersize=3, label="Path")
        
#         ax.view_init(elev=90)
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Z")
#         ax.set_title(f"Drone Navigation with SAC - Episode {episode + 1}")
#         ax.legend()

#         plt.draw()
#         plt.pause(0.01)  # Slow down for better visualization
#         time.sleep(0.05)  # Additional delay for better visualization

#         if done:
#             break

#     print(f"🎯 Episode {episode + 1}: Total Reward = {total_reward}")

# plt.ioff()  # Turn off interactive mode
# plt.show()  # Keep the final plot open

# # Close environment after testing
# env.close()

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from SAC import SACAgent
from Env import DroneEnv3D
from CONFIG import STATE_DIM, ACTION_DIM, MAX_ACTION, NUM_EPISODES, MAX_STEPS

# Load trained model paths
ACTOR_PATH = "checkpoints_sac/sac_actor_final.pth"
CRITIC_PATH = "checkpoints_sac/sac_critic_final.pth"

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

# Start episodes
for episode in range(NUM_EPISODES):
    state = env.reset()  
    total_reward = 0
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

        ax.view_init(elev=90)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Drone Navigation with SAC - Episode {episode + 1}")
        ax.legend()

        # Flush events instead of using plt.pause (prevents freezing)
        fig.canvas.flush_events()
        time.sleep(0.05)  # Delay for smooth visualization

        if done:
            if total_reward>0:
                save_path = f"saved_plots/episode_{episode+1}.png"
                plt.savefig(save_path)
                print("Plot Saved")
            break

    print(f"🎯 Episode {episode + 1}: Total Reward = {total_reward}")

plt.ioff()  # Turn off interactive mode
plt.show()  # Keep final plot open

# Close environment after testing
env.close()
