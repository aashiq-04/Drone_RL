import gym
import numpy as np
import torch
import os
from SAC import SACAgent
from Replay_Buffer import ReplayBuffer
from ENV2 import DroneEnv3D
from CONFIG import MAX_ACTION,MAX_EPISODES,MAX_STEPS,BATCH_SIZE,BUFFER_SIZE,STATE_DIM,ACTION_DIM
# Hyperparameters

CHECKPOINT_DIR = "checkpoints1"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Create environment and SAC agent
env = DroneEnv3D()
agent = SACAgent(STATE_DIM, ACTION_DIM, MAX_ACTION,lr=1e-4,gamma=0.98,tau=0.002,alpha=0.01)
replay_buffer = ReplayBuffer(BUFFER_SIZE, STATE_DIM, ACTION_DIM)

for episode in range(MAX_EPISODES):
    noise_std = max(0.1,0.5-(episode/MAX_EPISODES))
    state = np.array(env.reset(), dtype=np.float32)  # Ensure correct format
    episode_reward = 0

    for step in range(MAX_STEPS):
        action = agent.select_action(state,explore=True) + np.random.normal(0,noise_std)
        scaled_action = np.clip(action, -MAX_ACTION, MAX_ACTION)  # Clip actions
        next_state, reward, done, _ = env.step(scaled_action)

        replay_buffer.store(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if len(replay_buffer) > 5 * BATCH_SIZE:  # Ensure enough experience before training
            agent.train(replay_buffer, BATCH_SIZE)

        if done:
            break

    avg_reward = episode_reward / (step + 1)  # Avoid division by zero
    print(f"🎯Episode {episode}: Reward = {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")

    # Save the model every 50 episodes
    if episode % 50 == 0:
        actor_path = f"{CHECKPOINT_DIR}/sac_actor_{episode}.pth"
        critic_path = f"{CHECKPOINT_DIR}/sac_critic_{episode}.pth"
        torch.save(agent.actor, actor_path)  # Save full model
        torch.save(agent.critic, critic_path)
        print(f"✅ Model saved at Episode {episode}")

# Save the final model at the end of training
torch.save(agent.actor.state_dict(), f"{CHECKPOINT_DIR}/sac_actor_final.pth")
torch.save(agent.critic.state_dict(), f"{CHECKPOINT_DIR}/sac_critic_final.pth")
print("🏆 Final model saved!")
