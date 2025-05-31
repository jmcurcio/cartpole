import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseAgent

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(action_dim)
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config['lr'])
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.update_target()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update(self, batch, config):
        states = torch.FloatTensor([b.state for b in batch])
        actions = torch.LongTensor([b.action for b in batch]).unsqueeze(1)
        rewards = torch.FloatTensor([b.reward for b in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([b.next_state for b in batch])
        dones = torch.FloatTensor([b.done for b in batch]).unsqueeze(1)

        # Q(s, a)
        q_values = self.q_network(states).gather(1, actions)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target = rewards + (1 - dones) * self.gamma * next_q_values

        loss = torch.nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 