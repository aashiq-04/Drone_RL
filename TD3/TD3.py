import gym
import numpy as np
import os
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from ENV import DroneEnv3D
from CONFIG import MAX_ACTION, MAX_EPISODES, MAX_STEPS

# Logging
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Environment
env = DroneEnv3D()
env = Monitor(env)



# Action noise
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# TD3 Model with action noise
model = TD3(
    "MlpPolicy",
    env,
    learning_rate=2e-4,
    buffer_size=100000,
    batch_size=128,
    verbose=1,
    gamma=0.99,
    tau=0.005,
    policy_delay=2,
    gradient_steps=1,
    train_freq=(1, "step"),
    action_noise=action_noise,  # << ADD THIS
    tensorboard_log="./td3_tensorboard/"
)

# Train the model
model.learn(total_timesteps=100000)


# Save final model
model.save(f"{CHECKPOINT_DIR}/td3_final")
print("🏁 Final model saved.")
