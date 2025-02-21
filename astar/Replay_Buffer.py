import numpy as np
import random
from collections import deque
class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0  # Track the current size of the buffer

        # Storage
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        
        

    def __len__(self):
        return self.size  # Corrected: Return actual buffer size

    def store(self, state, action, reward, next_state, done):
        """Stores a transition into the buffer"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        


        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)  # Track the size

    def sample(self, batch_size):
        """Samples a batch of transitions randomly"""
        if self.size < batch_size:
            raise ValueError("Cannot sample more than available in the replay buffer.")
        
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.dones[idxs]
# import numpy as np
# import random
# from collections import deque

# class ReplayBuffer:
#     def __init__(self, max_size, state_dim, action_dim):
#         self.max_size = max_size
#         self.buffer = deque(maxlen=max_size)  # Dynamically growable buffer

#     def __len__(self):
#         return len(self.buffer)

#     def store(self, state, action, reward, next_state, done):
#         """Stores a transition in the buffer."""
#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         """Randomly samples a batch of transitions."""
#         if len(self.buffer) < batch_size:
#             raise ValueError("Not enough samples in the buffer to draw a batch.")

#         batch = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)

#         return (
#             np.array(states, dtype=np.float32),
#             np.array(actions, dtype=np.float32),
#             np.array(rewards, dtype=np.float32).reshape(-1, 1),  # Keep reward shape consistent
#             np.array(next_states, dtype=np.float32),
#             np.array(dones, dtype=np.bool_).reshape(-1, 1)  # Store done flags as boolean
#         )
