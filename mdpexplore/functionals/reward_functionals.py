import autograd.numpy as np
import autograd.numpy.linalg as la
import torch
from typing import List, Union
from abc import ABC, abstractmethod
from mdpexplore.env.discrete_env import Environment


class RewardFunctional(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def eval(self,
             emissions: np.ndarray,
             distribution: np.ndarray,
             episodes: int):
        pass

    def get_type(self):
        return self.type

