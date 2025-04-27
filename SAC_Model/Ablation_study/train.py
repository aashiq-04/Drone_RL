import gym
import numpy as np
import torch,time
from torch.utils.tensorboard import SummaryWriter
import os
from SAC_Model.SAC import SACAgent
from SAC_Model.Replay_Buffer import ReplayBuffer
from Ablation_study.ENV_reward import DroneEnv3D
from SAC_Model.CONFIG import MAX_ACTION,MAX_EPISODES,MAX_STEPS,BATCH_SIZE,BUFFER_SIZE,STATE_DIM,ACTION_DIM, LOAD_CHECKPOINT
# Hyperparameters

CHECKPOINT_DIR = "checkpoint_ENV2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Create environment and SAC agent
env = DroneEnv3D(reward_mode="collision_penalty_only")
agent = SACAgent(STATE_DIM, ACTION_DIM, MAX_ACTION,lr=2e-4,gamma=0.99,tau=0.005,alpha=0.02)
replay_buffer = ReplayBuffer(BUFFER_SIZE, STATE_DIM, ACTION_DIM)

best_reward = -float('inf')

# TensorBoard Setup
LOG_DIR = "runs"
writer = SummaryWriter(LOG_DIR)
if LOAD_CHECKPOINT:
    agent.load_model(torch.load(f"{CHECKPOINT_DIR}/sac_actor_best.pth"), torch.load(f"{CHECKPOINT_DIR}/sac_critic_best.pth"))
MAX_EXPLORATION = 1.0  # Example max exploration efficiency value
MAX_TIME = MAX_STEPS   # Maximum possible steps in an episode
MAX_COLLISIONS = 20
total_collision_rate = []
total_exploration_efficiency = []
total_time_efficiency=[]
collision=0

for episode in range(1,MAX_EPISODES+1):
    noise_std = max(0.1,0.4-(episode/(MAX_EPISODES/2)))
    state = np.array(env.reset(), dtype=np.float32)  # Ensure correct format
    episode_reward = 0
    visited_pos = set()
    collision=0
    step=0
    # losses =[]

    for step in range(MAX_STEPS):
        action = agent.select_action(state,explore=True) 
        action = np.clip(action, -MAX_ACTION, MAX_ACTION)  # Clip actions
        next_state, reward, done, _ = env.step(action)
        
        visited_pos.add(tuple(env.drone_pos))
        if env.check_collision():
            collision+=1

        replay_buffer.store(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if len(replay_buffer) > 5 * BATCH_SIZE:  # Ensure enough experience before training
            agent.train(replay_buffer, BATCH_SIZE)
         
        
           
        if done:
            break
    exploration_percent = max(0, (1 - (env.exploration_efficiency() / MAX_EXPLORATION)) * 100)
    time_percent = max(0, (1 - (step / MAX_TIME)) * 100)
    collision_percent = max(0, (1 - (collision / MAX_COLLISIONS)) * 100)

    # Store for tracking
    total_exploration_efficiency.append(exploration_percent)
    total_time_efficiency.append(time_percent)
    total_collision_rate.append(collision_percent)

    avg_reward = episode_reward / (step + 1)  # Avoid division by zero
    # avg_loss = np.mean(losses) if losses else 0
    # print(f"🎯Episode {episode}: Reward = {episode_reward:.2f}, Avg Reward: {avg_reward:.2f} ")
    print(f"🎯 Episode {episode + 1}: Episode reward = {episode_reward},Avg Reward = {avg_reward} "
          f"Exploration Eff: {total_exploration_efficiency}, Time Eff: {total_time_efficiency}, "
          f"Collision Rate: {total_collision_rate}")
    
    if episode_reward>best_reward:
        best_reward = episode_reward
        torch.save(agent.actor.state_dict(),f"{CHECKPOINT_DIR}/sac_actor_best.pth")
        torch.save(agent.critic.state_dict(),f"{CHECKPOINT_DIR}/sac_critic_best.pth")
        print(f"🏆 New Best Model Saved at Episode {episode} with Reward {episode_reward:.2f}")

    # Save the model every 50 episodes
    if episode % 50 == 0:
        actor_path = f"{CHECKPOINT_DIR}/sac_actor_{episode}.pth"
        critic_path = f"{CHECKPOINT_DIR}/sac_critic_{episode}.pth"
        torch.save(agent.actor.state_dict(), actor_path)  # Save full model
        torch.save(agent.critic.state_dict(), critic_path)
        print(f"✅ Model saved at Episode {episode}")

# Save the final model at the end of training
torch.save(agent.actor.state_dict(), f"{CHECKPOINT_DIR}/sac_actor_final.pth")
torch.save(agent.critic.state_dict(), f"{CHECKPOINT_DIR}/sac_critic_final.pth")
print("🏆 Final model saved!")
# **Print Final Metrics**
print("\n📊 **Overall Performance** 📊")
print(f"✅ Average Exploration Efficiency: {np.mean(total_exploration_efficiency)}%")
print(f"✅ Average Time Efficiency: {np.mean(total_time_efficiency)}%")
print(f"✅ Average Collision Rate: {np.mean(total_collision_rate)}%")
