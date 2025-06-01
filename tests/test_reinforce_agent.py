import gymnasium as gym
from agents.reinforce import REINFORCEAgent
import yaml

def test_reinforce_agent_action():
    env = gym.make('CartPole-v1')
    with open('configs/reinforce_config.yaml') as f:
        config = yaml.safe_load(f)
    agent = REINFORCEAgent(env.observation_space.shape[0], env.action_space.n, config)
    state, _ = env.reset()
    action = agent.select_action(state)
    assert 0 <= action < env.action_space.n 