import torch
import numpy as np
import time  # For visualization delay
from SAC import SACAgent
from Environment import DroneEnv3D
import matplotlib.pyplot as plt

# Load trained model
ACTOR_PATH = "checkpoints1/sac_actor_final.pth"
CRITIC_PATH = "checkpoints1/sac_critic_final.pth"

# Environment and agent setup
STATE_DIM = 21  # (x, y, z, vx, vy, vz, lidar, imu)
ACTION_DIM = 3  # (vx, vy, vz)
MAX_ACTION = 1.0

env = DroneEnv3D()
agent = SACAgent(STATE_DIM, ACTION_DIM, MAX_ACTION)

# Load model weights safely
agent.actor.load_state_dict(torch.load(ACTOR_PATH,weights_only=True))
agent.critic.load_state_dict(torch.load(CRITIC_PATH,weights_only=True))
agent.actor.eval()
agent.critic.eval()
print("✅ Model loaded successfully!")
# Interactive mode for Matplotlib
plt.ion()
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Run evaluation
NUM_EPISODES = 5  # Number of test episodes
MAX_STEPS = 100  # Max steps per episode

for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    env.path = [env.drone_pos.copy()]  # Reset path for each episode

    for step in range(MAX_STEPS):
        action = agent.select_action(state)  # Use deterministic actions
        next_state, reward, done, _ = env.step(action)
        
        total_reward += reward
        state = next_state

        # Render environment for visualization
        ax.clear()
        ax.set_xlim(0, env.space_size)
        ax.set_ylim(0, env.space_size)
        ax.set_zlim(0, env.space_size)

        # Plot drone
        ax.scatter(*env.drone_pos, color='red', s=100, label="Drone")

        # Plot target
        ax.scatter(*env.target, color='green', s=150, label="Target")

        # Plot obstacles as cylinders
        for obs in env.obstacles:
            env.draw_cylinder(ax, *obs)

        # Plot path (all points where drone has been)
        path = np.array(env.path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color='blue', linestyle='-', marker='o', markersize=3, label="Path")
        # ax.view_init(elev=90)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Drone Navigation with SAC - Episode {episode + 1}")
        ax.legend()

        plt.draw()
        plt.pause(0.01)  # Slow down for better visualization
        time.sleep(0.05)  # Additional delay for better visualization

        if done:
            break

    print(f"🎯 Episode {episode + 1}: Total Reward = {total_reward}")

plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the final plot open

# Close environment after testing
env.close()
