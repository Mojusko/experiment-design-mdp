import autograd.numpy as np
from abc import ABC, abstractmethod
from typing import Callable

from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.env.linear_system import ContinuousEnv
from mdpexplore.policies.policy_base import Policy


class DiscreteSolver(ABC):
    def __init__(
        self,
        env: DiscreteEnv,
        reward: np.ndarray,
    ) -> None:

        self.env = env
        self.reward = reward

    @abstractmethod
    def solve(self) -> Policy:
        ...
    
    def initialize(self, params = None):
        pass

    def initialization_params(self):
        pass


class ContinuousSolver(ABC):
    def __init__(
        self,
        env: ContinuousEnv,
        reward: Callable,
    ) -> None:

        self.env = env
        self.reward = reward

    @abstractmethod
    def solve(self) -> Policy:
        ...
    
    def initialize(self, params = None):
        pass

    def initialization_params(self):
        pass
