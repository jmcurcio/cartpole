import torch
import numpy as np
from utils.video import wrap_env_for_video

class Trainer:
    def __init__(self, env, agent, buffer, config):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.config = config

    def train(self, logger=None, video_dir=None, video_episodes=None, agent_name="agent"):
        num_episodes = self.config.get('num_episodes', 500)
        batch_size = self.config['batch_size']
        target_update_freq = self.config['target_update_freq']
        min_buffer_size = batch_size
        rewards = []

        for episode in range(num_episodes):
            # Video logic
            env_to_use, recording = (self.env, False)
            if video_dir is not None and video_episodes is not None:
                env_to_use, recording = wrap_env_for_video(self.env, video_dir, video_episodes, episode, agent_name)
            state, _ = env_to_use.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env_to_use.step(action)
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
            msg = f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {getattr(self.agent, 'epsilon', 'N/A'):.3f}"
            if logger:
                logger.info(msg)
            else:
                print(msg)
            # Close video recorder if used
            if recording and hasattr(env_to_use, "close_video_recorder"):
                env_to_use.close_video_recorder()

        return rewards 