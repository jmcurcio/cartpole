import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
import yaml
import argparse

from agents.random import RandomAgent
from agents.dqn import DQNAgent
from agents.reinforce import REINFORCEAgent
from core.replay_buffer import ReplayBuffer
from core.trainer import Trainer
from utils.logger import get_logger
from utils.plotting import plot_rewards
from utils.video import wrap_env_for_video

logger = get_logger(__name__)

def get_agent(agent_name, env, config):
    if agent_name == "random":
        return RandomAgent(env.action_space)
    elif agent_name == "dqn":
        return DQNAgent(env.observation_space.shape[0], env.action_space.n, config)
    elif agent_name == "reinforce":
        return REINFORCEAgent(env.observation_space.shape[0], env.action_space.n, config)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, choices=["random", "dqn", "reinforce"], help="Agent to use")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment ID")
    parser.add_argument("--show_plot", action="store_true", help="Show plot after training")
    parser.add_argument("--record_videos", action="store_true", help="Record videos")
    parser.add_argument("--video_episodes", type=str, default="0,-1", help="Comma-separated list of episode indices to record (e.g., 0,249,499)")
    args = parser.parse_args()


    logger.info(f"Starting experiment with agent: {args.agent}, env: {args.env}, config: {args.config}")

    env = gym.make(args.env, render_mode="rgb_array")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    agent = get_agent(args.agent, env, config)

    # Parse video episodes
    num_episodes = config.get('num_episodes', 500)
    # Support -1 as last episode, -2 as second to last, etc.
    raw_indices = [int(x) for x in args.video_episodes.split(",") if x.strip() != ""]
    episode_indices = set([i if i >= 0 else num_episodes + i for i in raw_indices])

    if args.record_videos:
        env = wrap_env_for_video(env, args.agent, episode_indices)

    trainer = Trainer(env=env, agent=agent, buffer=ReplayBuffer(0), config=config)
    if args.agent == "dqn":
        trainer.buffer = ReplayBuffer(config.get('buffer_size', 10000))
        rewards = trainer.train_dqn(logger=logger)
    elif args.agent == "reinforce":
        rewards = trainer.train_reinforce(logger=logger)
    else:
        rewards = trainer.train_dqn(logger=logger)

    plot_rewards(rewards, show=args.show_plot)
    env.close()

if __name__ == "__main__":
    main() 