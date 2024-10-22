import autograd.numpy as np
from abc import ABC, abstractmethod
from typing import Callable
import torch 
from doexpy.env.discrete_env import DiscreteEnv
from doexpy.env.linear_system import ContinuousEnv
from doexpy.policies.policy_base import Policy


class DiscreteSolver(ABC):
    def __init__(
        self,
        env: DiscreteEnv,
        reward: torch.Tensor,
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
