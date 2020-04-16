import numpy as np
import random
from collections import namedtuple, deque, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.loss = torch.nn.MSELoss()

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        self.qnetwork_target.eval()
        with torch.no_grad():
            action_values_target = self.qnetwork_target(next_states)
        target_max = torch.max(action_values_target, dim=1)[0]
        targets = rewards.squeeze() + (gamma * target_max  * (1 - dones.squeeze()))
            
        self.qnetwork_local.train()
        self.optimizer.zero_grad()
        action_values_current = self.qnetwork_local(states)[range(len(states)), actions.squeeze()]
        
        # do gradient step with optimizer
        self.loss(action_values_current, targets).backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
            
            
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_size = state_size
        self.output_size = action_size
        
        # smaller FC network for task with 1D states
        self.hidden_sizes = [300, 200, 100]
        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.input_size, self.hidden_sizes[0])),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])),
            ('relu3', nn.ReLU()),
            ('q_values', nn.Linear(self.hidden_sizes[2], self.output_size))
        ]))
        
        
        """
        # Huge network from DeepMind's DQN paper
        self.input_channels = 4
        self.filter_sizes = [8, 4, 3]
        self.conv_channels = [32, 64, 64]
        self.strides = [4, 2, 1]
        self.fc_size = 512
        self.model = nn.Sequential(OrderedDict([
                      ('conv1', nn.Conv2d(self.input_channels, self.conv_channels[0], self.filter_sizes[0], stride=self.strides[0])),
                      ('relu1', nn.ReLU()),
                      ('conv2', nn.Conv2d(self.input_channels, self.conv_channels[1], self.filter_sizes[1], stride=self.strides[1])),
                      ('relu2', nn.ReLU()),
                      ('conv3', nn.Conv2d(self.input_channels, self.conv_channels[2], self.filter_sizes[2], stride=self.strides[2])),
                      ('relu3', nn.ReLU()),
                      ('squeeze', Lambda(lambda x: x.squeeze())),
                      ('fc1', nn.Linear(hidden_sizes[0], self.fc_size)),
                      ('relu_fc', nn.ReLU()),
                      ('q_values', nn.Linear(self.fc_size, self.output_size))]))
        """

    def forward(self, state):
        return self.model(state)

            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)