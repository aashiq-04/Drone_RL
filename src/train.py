import gym
import numpy as np
import torch
import os
from SAC import SACAgent
from Replay_Buffer import ReplayBuffer
from Environment import DroneEnv3D

# Hyperparameters
MAX_EPISODES = 500
MAX_STEPS = 500
BATCH_SIZE = 64
BUFFER_SIZE = int(1e6)
STATE_DIM = 21  # (position, velocity, LIDAR, IMU)
ACTION_DIM = 3  # (vx, vy, vz)
MAX_ACTION = 1.0
CHECKPOINT_DIR = "checkpoints1"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Create environment and SAC agent
env = DroneEnv3D()
agent = SACAgent(STATE_DIM, ACTION_DIM, MAX_ACTION)
replay_buffer = ReplayBuffer(BUFFER_SIZE, STATE_DIM, ACTION_DIM)

for episode in range(MAX_EPISODES):
    state = env.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):
        # Add exploration noise in early episodes
        action = agent.select_action(state, explore=(episode < 100))  

        next_state, reward, done, _ = env.step(action)
        replay_buffer.store(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        # Train only if enough samples are collected
        if len(replay_buffer) > BATCH_SIZE:
            agent.train(replay_buffer, BATCH_SIZE)

        if done:
            break
    avg_reward = episode_reward / (step + 1)  # Avoid division by zero
    
    print(f"🎯Episode {episode}: Reward = {episode_reward:.2f}, Avg Reward: {avg_reward:.2f} ")

    # Save the model every episode (overwrite latest checkpoint)
    # latest_actor_path = f"{CHECKPOINT_DIR}/sac_actor_latest.pth"
    # latest_critic_path = f"{CHECKPOINT_DIR}/sac_critic_latest.pth"
    # torch.save(agent.actor.state_dict(), latest_actor_path)
    # torch.save(agent.critic.state_dict(), latest_critic_path)

    # Save periodic checkpoints every 50 episodes
    if episode % 50 == 0:
        actor_path = f"{CHECKPOINT_DIR}/sac_actor_{episode}.pth"
        critic_path = f"{CHECKPOINT_DIR}/sac_critic_{episode}.pth"
        torch.save(agent.actor.state_dict(), actor_path)
        torch.save(agent.critic.state_dict(), critic_path)
        print(f"✅ Model checkpoint saved at Episode {episode}")

# Save the final model at the end of training
torch.save(agent.actor.state_dict(), f"{CHECKPOINT_DIR}/sac_actor_final.pth")
torch.save(agent.critic.state_dict(), f"{CHECKPOINT_DIR}/sac_critic_final.pth")
print("✅ Final model saved!")
