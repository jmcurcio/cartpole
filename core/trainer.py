import torch
import numpy as np

class Trainer:
    def __init__(self, env, agent, buffer, config):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.config = config

    def train_dqn(self, logger=None):
        num_episodes = self.config.get('num_episodes', 500)
        batch_size = self.config['batch_size']
        target_update_freq = self.config['target_update_freq']
        min_buffer_size = batch_size
        rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                done = terminated or truncated

                if len(self.buffer) >= min_buffer_size:
                    batch = self.buffer.sample(batch_size)
                    self.agent.update(batch, self.config)

            if hasattr(self.agent, 'epsilon'):
                self.agent.epsilon = max(
                    self.agent.epsilon * self.agent.epsilon_decay,
                    self.agent.epsilon_min
                )

            if hasattr(self.agent, 'update_target') and episode % target_update_freq == 0:
                self.agent.update_target()

            rewards.append(episode_reward)
            epsilon = getattr(self.agent, 'epsilon', None)
            if epsilon is not None:
                epsilon_str = f"{epsilon:.3f}"
            else:
                epsilon_str = "N/A"
            msg = f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {epsilon_str}"
            if logger:
                logger.info(msg)
            else:
                print(msg)
        return rewards

    def train_reinforce(self, logger=None):
        num_episodes = self.config.get('num_episodes', 500)
        rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.agent.store_reward(reward)
                state = next_state
                episode_reward += reward
                done = terminated or truncated

            self.agent.update()
            rewards.append(episode_reward)
            logger.info(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

        logger.info("REINFORCE training finished.")
        return rewards