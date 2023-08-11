import numpy as np
from abc import ABC, abstractmethod

class Feedback(ABC):
    def __init__(self, env, objective) -> None:
        super().__init__()
        self.objective = objective
    
    @abstractmethod
    def step_single(self, state: int, action: int):
        ...

    @abstractmethod
    def step_episode(self):
        ...

class EmptyFeedback(Feedback):
    def __init__(self, env = None, objective = None) -> None:
        super().__init__(env, objective)
        self.objective = objective
    
    def step_single(self, state:int, action: int):
        pass

    def step_episode(self):
        pass

class SimpleFeedback(Feedback):
    def __init__(self, env, objective) -> None:
        super().__init__(env, objective)
        self.state_trajectory = []
        self.action_trajectory = []
    
    def step_single(self, state: int, action: int):
        self.state_trajectory.append(state)
        self.action_trajectory.append(action)
        self.step_update()

    def step_episode(self):
        self.episode_update()
        # restart the trajectories
        self.state_trajectory = []
        self.action_trajectory = []
    
    @abstractmethod
    def step_update(self):
        ...

    @abstractmethod
    def episode_update(self):
        ...