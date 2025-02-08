from stable_baselines3 import PPO
from sample import DroneEnv3D

# Create the environment
env = DroneEnv3D()

# Initialize the PPO model with custom hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,  # Adjust learning rate
    gamma=0.99,            # Discount factor
    ent_coef=0.01,         # Entropy coefficient for exploration
    n_steps=2048,          # Number of steps per update
    verbose=1
)

# Training parameters
total_timesteps = 20000
obs = env.reset()
episode_reward = 0

for step in range(total_timesteps):
    action, _states = model.predict(obs)
    new_obs, reward, done, _ = env.step(action)
    
    # Update episode reward
    episode_reward += reward
    
    obs = new_obs if not done else env.reset()
    
    # Print episode reward and reset if episode is done
    if done:
        print(f"Episode finished after {step+1} timesteps with reward {episode_reward}")
        episode_reward = 0

# Train the model and save it
model.learn(total_timesteps=total_timesteps)
model.save("ppo_drone_navigation_tuned")

print("Training complete and model saved.")
