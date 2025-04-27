import gym
import numpy as np
import os
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from ENV import DroneEnv3D
from CONFIG import MAX_ACTION, MAX_EPISODES, MAX_STEPS

# Logging
CHECKPOINT_DIR = "checkpointDDPG"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Environment
env = DroneEnv3D()
env = Monitor(env)



# DDPG Model
model = DDPG(
    "MlpPolicy",
    env,
    learning_rate=2e-4,
    buffer_size=100000,
    batch_size=128,
    verbose=1,
    gamma=0.99,
    tau=0.005,
    tensorboard_log="./ddpg_tensorboard/"
    )

# Train the model
model.learn(total_timesteps=(MAX_EPISODES * MAX_STEPS))


# Save final model
model.save(f"{CHECKPOINT_DIR}/ddpg_final")
print("🏁 Final model saved.")
