import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseAgent

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class REINFORCEAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(action_dim)
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config['lr'])
        self.gamma = config['gamma']
        self.episode_log_probs = []
        self.episode_rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.episode_log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.episode_rewards.append(reward)

    def update(self, *args, **kwargs):
        # Compute discounted returns
        R = 0
        returns = []
        for r in reversed(self.episode_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, R in zip(self.episode_log_probs, returns):
            loss -= log_prob * R

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear memory
        self.episode_log_probs = []
        self.episode_rewards = [] 