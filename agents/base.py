from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, action_space):
        self.action_space = action_space

    @abstractmethod
    def select_action(self, state):
        pass

    def update(self, *args, **kwargs):
        pass 