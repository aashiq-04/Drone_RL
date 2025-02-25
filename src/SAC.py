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
        return torch.clamp(self.max_action * self.net(state), -self.max_action, self.max_action)

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
        q1_value = self.q1(sa)
        q2_value = self.q2(sa)
        return q1_value, q2_value  # Double Q-learning

class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, lr=3e-4, alpha=0.02):
        self.actor = Actor(state_dim, action_dim, max_action).float()
        self.critic = Critic(state_dim, action_dim).float()
        self.critic_target = Critic(state_dim, action_dim).float()
        self.critic_target.load_state_dict(self.critic.state_dict())  # Initialize target network

        self.max_action = max_action
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        
        # Automatic Temperature (Alpha) Adjustment
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = alpha  # Initial alpha value

        self.update_counter = 0  # Counter for delayed target updates

    def select_action(self, state, explore=True):
        """Returns action from policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).cpu().detach().numpy()[0]

        if explore:
            noise = np.random.normal(0, 0.2, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)

        return action

    def train(self, replay_buffer, batch_size=64):
        """Samples from buffer and updates networks"""
        actual_batch_size = min(batch_size, len(replay_buffer))
        states, actions, rewards, next_states, dones = replay_buffer.sample(actual_batch_size)

        # Convert to PyTorch tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).view(-1, 1)  
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).view(-1, 1)  

        # Get next action from actor
        next_actions = self.actor(next_states)

        # Get target Q values with entropy regularization
        with torch.no_grad():
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            
            # Compute log probabilities for entropy regularization
            log_probs = -((next_actions**2).sum(dim=-1, keepdim=True))
            
            y = rewards + (1 - dones) * self.gamma * (next_q - self.alpha * log_probs)

        # Compute current Q estimates
        q1, q2 = self.critic(states, actions)
        critic_loss = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update policy (Actor)
        new_actions = self.actor(states)
        q1_actor, _ = self.critic(states, new_actions)

        log_probs = -((new_actions**2).sum(dim=-1, keepdim=True))  # Corrected entropy approximation
        actor_loss = (self.alpha * log_probs - q1_actor).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Automatic Temperature Adjustment (Entropy tuning)
        alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()  # Update alpha dynamically

        # Delayed Target Network Update (Every 2 steps)
        self.update_counter += 1
        if self.update_counter % 2 == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


