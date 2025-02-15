import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    """Actor Network: Outputs continuous action based on state"""
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Outputs actions between [-1, 1]
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    """Critic Network: Estimates Q-values"""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)  # Double Q-learning

class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, lr=3e-4):
        self.actor = Actor(state_dim, action_dim, max_action).float()
        self.critic = Critic(state_dim, action_dim).float()
        self.critic_target = Critic(state_dim, action_dim).float()
        self.critic_target.load_state_dict(self.critic.state_dict())  # Initialize target network

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        """Returns action from policy (no exploration noise needed)"""
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64):
        """Samples from buffer and updates networks"""
        actual_batch_size = min(batch_size, len(replay_buffer))
        states, actions, rewards, next_states, dones = replay_buffer.sample(actual_batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Get next action from actor
        next_actions = self.actor(next_states)
        
        # Get target Q values
        q1_target, q2_target = self.critic_target(next_states, next_actions)
        q_target = torch.min(q1_target, q2_target)
        y = rewards + (1 - dones) * self.gamma * q_target.detach()

        # Compute current Q estimates
        q1, q2 = self.critic(states, actions)
        critic_loss = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update policy (Actor)
        actor_loss = -self.critic.q1(torch.cat([states, self.actor(states)], dim=-1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Target Networks (Soft Update)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
