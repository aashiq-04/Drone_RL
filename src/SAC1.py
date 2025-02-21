import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from collections import deque
import random
from Environment import DroneEnv3D

# Hyperparameters
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # Entropy coefficient
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
BUFFER_SIZE = 1000000
UPDATE_INTERVAL = 50

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural networks (Q-values and policy)
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SAC:
    def __init__(self, state_dim, action_dim):
        # Q networks
        self.q1 = MLP(state_dim + action_dim, 1).to(device)
        self.q2 = MLP(state_dim + action_dim, 1).to(device)
        self.q1_target = MLP(state_dim + action_dim, 1).to(device)
        self.q2_target = MLP(state_dim + action_dim, 1).to(device)
        
        # Initialize target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Policy network (actor)
        self.policy = MLP(state_dim, action_dim).to(device)
        
        # Value network (for computing entropy)
        self.value = MLP(state_dim, 1).to(device)
        self.value_target = MLP(state_dim, 1).to(device)
        self.value_target.load_state_dict(self.value.state_dict())
        
        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=LEARNING_RATE)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=LEARNING_RATE)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=LEARNING_RATE)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        mean = self.policy(state)
        # Add noise for exploration (reparameterization trick)
        std = torch.full_like(mean, 0.1)  # Constant exploration noise
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        return action.clamp(-1.0, 1.0).cpu().detach().numpy()

    def update(self, replay_buffer):
        # Sample batch from replay buffer
        batch = replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, done = batch

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(1)

        # Compute Q-values and target value
        next_actions = self.policy(next_states)
        next_q1 = self.q1_target(torch.cat([next_states, next_actions], 1))
        next_q2 = self.q2_target(torch.cat([next_states, next_actions], 1))
        next_value = self.value_target(next_states)

        # Bellman backup
        target_q = rewards + GAMMA * (1 - done) * (torch.min(next_q1, next_q2) - ALPHA * next_value)

        # Update Q networks
        q1_loss = torch.mean((self.q1(torch.cat([states, actions], 1)) - target_q) ** 2)
        q2_loss = torch.mean((self.q2(torch.cat([states, actions], 1)) - target_q) ** 2)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update value network
        value_loss = torch.mean((self.value(states) - (torch.min(self.q1(states, actions), self.q2(states, actions)) - ALPHA * next_value)) ** 2)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update policy network
        policy_loss = torch.mean(ALPHA * torch.log(torch.tensor(0.1)) - torch.min(self.q1(states, actions), self.q2(states, actions)))

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update target networks
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)
        self.soft_update(self.value, self.value_target)

    def soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)


# Replay Buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done)

    def size(self):
        return len(self.buffer)


# Main training loop
def train():
    # Initialize environment, SAC agent, and replay buffer
    env = DroneEnv3D()  # Replace with your environment
    agent = SAC(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    replay_buffer = ReplayBuffer()

    # Training loop
    for episode in range(1000):
        state = env.reset()
        episode_reward = 0
        for t in range(1000):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            if replay_buffer.size() > BATCH_SIZE:
                agent.update(replay_buffer)

            state = next_state
            episode_reward += reward

            if done:
                break

        print(f"Episode {episode} Reward: {episode_reward}")

if __name__ == "__main__":
    train()
