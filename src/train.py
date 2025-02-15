import gym
import numpy as np
import torch
import os
from SAC import SACAgent
from Replay_Buffer import ReplayBuffer
from Environment import DroneEnv3D

# Hyperparameters
MAX_EPISODES = 500
MAX_STEPS = 200
BATCH_SIZE = 64
BUFFER_SIZE = 100000
STATE_DIM = 18  # (position, velocity, LIDAR, IMU)
ACTION_DIM = 3  # (vx, vy, vz)
MAX_ACTION = 1.0
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Create environment and SAC agent
env = DroneEnv3D()
agent = SACAgent(STATE_DIM, ACTION_DIM, MAX_ACTION)
replay_buffer = ReplayBuffer(BUFFER_SIZE, STATE_DIM, ACTION_DIM)

for episode in range(MAX_EPISODES):
    state = env.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.store(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if len(replay_buffer) > BATCH_SIZE:
            agent.train(replay_buffer, BATCH_SIZE)

        if done:
            break

    print(f"Episode {episode}: Reward = {episode_reward}")

    # Save the model every 50 episodes
    if episode % 50 == 0:
        actor_path = f"{CHECKPOINT_DIR}/sac_actor_{episode}.pth"
        critic_path = f"{CHECKPOINT_DIR}/sac_critic_{episode}.pth"
        torch.save(agent.actor.state_dict(), actor_path)
        torch.save(agent.critic.state_dict(), critic_path)
        print(f"✅ Model saved at Episode {episode}")

# Save the final model at the end of training
torch.save(agent.actor.state_dict(), f"{CHECKPOINT_DIR}/sac_actor_final.pth")
torch.save(agent.critic.state_dict(), f"{CHECKPOINT_DIR}/sac_critic_final.pth")
print("Final model saved!")
