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
    parser.add_argument("--video_episodes", type=str, default="0,-1", help="Comma-separated list of episode indices to record (e.g., 0,249,499)")
    args = parser.parse_args()

    logger = get_logger(f"{args.agent}_{args.env}")
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

    def should_record(episode):
        return episode in episode_indices

    video_folder = os.path.join('videos', args.agent)
    os.makedirs(video_folder, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=should_record,
        name_prefix=f"{args.agent}"
    )

    rewards = []
    if args.agent == "reinforce":
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.store_reward(reward)
                state = next_state
                episode_reward += reward
                done = terminated or truncated
            agent.update()
            rewards.append(episode_reward)
            logger.info(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}")
        logger.info("Training finished.")
        if hasattr(env, "close"):
            env.close()
    else:
        buffer = ReplayBuffer(config.get('buffer_size', 10000))
        trainer = Trainer(env, agent, buffer, config)
        rewards = trainer.train(logger=logger, video_episodes=episode_indices, agent_name=args.agent)

    plot_rewards(rewards, show=args.show_plot)
    env.close()

if __name__ == "__main__":
    main() 