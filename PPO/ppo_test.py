import numpy as np
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from ENV import DroneEnv3D
from CONFIG import NUM_EPISODES, MAX_STEPS

# Load PPO model
MODEL_PATH = "ppo_drone_updated.zip"
model = PPO.load(MODEL_PATH)
print("✅ PPO model loaded successfully!")

# Setup environment
env = DroneEnv3D()

# Matplotlib 3D plot setup
plt.ion()
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Metrics setup
test_rewards = []
test_collisions = []
test_steps = []
total_exploration_efficiency = []
total_time_efficiency = []
total_collision_rate = []

MAX_EXPLORATION = 1.5
MAX_TIME = MAX_STEPS
MAX_COLLISIONS = NUM_EPISODES 

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()    
    total_reward = 0
    visited_positions = set()
    collision_count = 0
    env.path = [env.drone_pos.copy()]
    path_array = np.array(env.path)

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated,_ = env.step(action)
        total_reward += reward

        visited_positions.add(tuple(env.drone_pos))
        if env.check_collision():
            collision_count += 1

        env.path.append(env.drone_pos.copy())
        path_array = np.array(env.path)

        # Visualization
        ax.cla()
        ax.set_xlim(0, env.space_size)
        ax.set_ylim(0, env.space_size)
        ax.set_zlim(0, env.space_size)
        ax.scatter(*env.drone_pos, color='red', s=100, label="Drone")
        ax.scatter(*env.target, color='green', s=150, label="Target")
        for obs_ in env.obstacles:
            env.draw_cylinder(ax, *obs_)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                color='blue', linestyle='-', marker='o', markersize=3, label="Path")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Drone Navigation with PPO - Episode {episode + 1}")
        ax.legend()
        fig.canvas.flush_events()
        time.sleep(0.05)

        if done:
            break

    test_rewards.append(total_reward)
    test_collisions.append(collision_count)
    test_steps.append(step)
    
    exploration_percent = max(0, (1 - (env.exploration_efficiency() / MAX_EXPLORATION)) * 100)
    time_percent = max(0, (1 - (step / MAX_TIME)) * 100)
    collision_percent = max(0, (1 - (collision_count / MAX_COLLISIONS)) * 100)

    total_exploration_efficiency.append(exploration_percent)
    total_time_efficiency.append(time_percent)
    total_collision_rate.append(collision_percent)

    print(f"🎯 Episode {episode + 1}: Total Reward = {total_reward:.2f}, "
          f"Exploration Eff: {exploration_percent:.2f}%, "
          f"Time Eff: {time_percent:.2f}%, "
          f"Collision Rate: {collision_percent:.2f}%")

# Save metrics
np.save("ppo_test_rewards.npy", test_rewards)
np.save("ppo_test_collisions.npy", test_collisions)
np.save("ppo_test_steps.npy", test_steps)

print("\n📊 **Overall PPO Performance** 📊")
print(f"✅ Average Exploration Efficiency: {np.mean(total_exploration_efficiency):.2f}%")
print(f"✅ Average Time Efficiency: {np.mean(total_time_efficiency):.2f}%")
print(f"✅ Average Collision Rate: {np.mean(total_collision_rate):.2f}%")

plt.ioff()
plt.show()
env.close()
