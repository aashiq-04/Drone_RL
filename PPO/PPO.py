from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from ENV import DroneEnv3D
import gymnasium as gym
import numpy as np

# Step 1: Create environment and check compatibility
env = DroneEnv3D()
check_env(env, warn=True)

# Step 2: Monitor the environment to capture episode rewards
env = Monitor(env)

# Step 3: Create PPO model
model = PPO("MlpPolicy", env, 
            verbose=1,
            learning_rate=1e-4,
            ent_coef=0.01,
            tensorboard_log="./ppo_tensorboard/")

# Step 4: Train model

model.learn(total_timesteps=100000)

# Step 5: Save model
model.save("ppo_drone_updated")

print("✅ PPO Model trained and saved!")
