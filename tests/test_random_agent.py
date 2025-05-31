import gymnasium as gym
from agents.random import RandomAgent

def test_random_agent_action():
    env = gym.make('CartPole-v1')
    agent = RandomAgent(env.action_space)
    state, _ = env.reset()
    action = agent.select_action(state)
    assert env.action_space.contains(action) 