# import numpy as np
# import matplotlib.pyplot as plt

# # Load testing metrics
# test_rewards = np.load("test_rewards.npy")
# test_collisions = np.load("test_collisions.npy")
# test_steps = np.load("test_steps.npy")
# # test_distances = np.load("test_distances.npy")

# episodes = np.arange(len(test_rewards))

# # Create a single figure with multiple subplots
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# # Plot Test Rewards
# axes[0, 0].plot(episodes, test_rewards, label="Test Reward", color="blue")
# axes[0, 0].set_xlabel("Episodes")
# axes[0, 0].set_ylabel("Reward")
# axes[0, 0].set_title("Reward Trend During Testing")
# axes[0, 0].legend()
# axes[0, 0].grid()

# # Plot Test Collisions
# axes[0, 1].plot(episodes, test_collisions, label="Test Collisions", color="red")
# axes[0, 1].set_xlabel("Episodes")
# axes[0, 1].set_ylabel("Collisions")
# axes[0, 1].set_title("Collisions Per Episode During Testing")
# axes[0, 1].legend()
# axes[0, 1].grid()

# # Plot Test Steps
# axes[1, 0].plot(episodes, test_steps, label="Steps per Episode", color="green")
# axes[1, 0].set_xlabel("Episodes")
# axes[1, 0].set_ylabel("Steps")
# axes[1, 0].set_title("Steps Per Episode During Testing")
# axes[1, 0].legend()
# axes[1, 0].grid()



# # Adjust layout
# plt.tight_layout()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # for smoothing

# Load data
test_rewards = np.load("test_rewards.npy")
test_collisions = np.load("test_collisions.npy")
test_steps = np.load("test_steps.npy")

episodes = np.arange(len(test_rewards))

# Smooth rewards for better trend visualization
smoothed_rewards = uniform_filter1d(test_rewards, size=10)

# Create reward vs episode plot (only reward trend for research paper)
plt.figure(figsize=(10, 6))
plt.plot(episodes, test_rewards, color="skyblue", label="Raw Rewards", alpha=0.4)
plt.plot(episodes, smoothed_rewards, color="blue", linewidth=2.5, label="Smoothed Rewards (Moving Avg)")

plt.title("Reward Improvement Over Episodes", fontsize=16)
plt.xlabel("Episodes", fontsize=13)
plt.ylabel("Total Reward", fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# Save the figure as high-res for paper
plt.tight_layout()
plt.savefig("reward_trend_plot.png", dpi=300)
plt.show()
