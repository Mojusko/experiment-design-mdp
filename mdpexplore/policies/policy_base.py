import torch 
from abc import ABC, abstractmethod

from mdpexplore.env.discrete_env import DiscreteEnv, Environment


class Policy(ABC):
    def __init__(self, env: Environment) -> None:
        self.env = env

    @abstractmethod
    def next_action(self, state):
        ...


class SummarizedPolicy(Policy, ABC):
    def __init__(self, env: DiscreteEnv) -> None:
        super().__init__(env)
