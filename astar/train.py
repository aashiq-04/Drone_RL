import gym
import numpy as np
import torch
import os
from SAC import SACAgent
from Replay_Buffer import ReplayBuffer
from Env import DroneEnv3D
from CONFIG import MAX_ACTION,MAX_EPISODES,MAX_STEPS,BATCH_SIZE,BUFFER_SIZE,STATE_DIM,ACTION_DIM
# Hyperparameters

CHECKPOINT_DIR = "New_checkpoint_sac"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Create environment and SAC agent
env = DroneEnv3D()
agent = SACAgent(STATE_DIM, ACTION_DIM, MAX_ACTION,lr=3e-4,gamma=0.99,tau=0.005,alpha=0.02)
replay_buffer = ReplayBuffer(BUFFER_SIZE, STATE_DIM, ACTION_DIM)

best_reward = -float('inf')

for episode in range(1,MAX_EPISODES+1):
    noise_std = max(0.1,0.6-(episode/(MAX_EPISODES/2)))
    state = np.array(env.reset(), dtype=np.float32)  # Ensure correct format
    episode_reward = 0
    # losses =[]

    for step in range(MAX_STEPS):
        action = agent.select_action(state,explore=True) + np.random.normal(0,noise_std)
        action = np.clip(action, -MAX_ACTION, MAX_ACTION)  # Clip actions
        next_state, reward, done, _ = env.step(action)

        replay_buffer.store(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if len(replay_buffer) > 5 * BATCH_SIZE:  # Ensure enough experience before training
            agent.train(replay_buffer, BATCH_SIZE)
            
        if done:
            break

    avg_reward = episode_reward / (step + 1)  # Avoid division by zero
    # avg_loss = np.mean(losses) if losses else 0
    print(f"🎯Episode {episode}: Reward = {episode_reward:.2f}, Avg Reward: {avg_reward:.2f} ")
    
    if episode_reward>best_reward:
        best_reward = episode_reward
        torch.save(agent.actor.state_dict(),f"{CHECKPOINT_DIR}/sac_actor_best.pth")
        torch.save(agent.critic.state_dict(),f"{CHECKPOINT_DIR}/sac_critic_best.pth")
        print(f"🏆 New Best Model Saved at Episode {episode} with Reward {episode_reward:.2f}")

    # Save the model every 50 episodes
    if episode % 100 == 0:
        actor_path = f"{CHECKPOINT_DIR}/sac_actor_{episode}.pth"
        critic_path = f"{CHECKPOINT_DIR}/sac_critic_{episode}.pth"
        torch.save(agent.actor, actor_path)  # Save full model
        torch.save(agent.critic, critic_path)
        print(f"✅ Model saved at Episode {episode}")

# Save the final model at the end of training
torch.save(agent.actor.state_dict(), f"{CHECKPOINT_DIR}/sac_actor_final.pth")
torch.save(agent.critic.state_dict(), f"{CHECKPOINT_DIR}/sac_critic_final.pth")
print("🏆 Final model saved!")
