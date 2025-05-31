from .base import BaseAgent

class RandomAgent(BaseAgent):
    def select_action(self, state):
        return self.action_space.sample() 