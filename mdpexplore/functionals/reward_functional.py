import autograd.numpy as np
import autograd.numpy.linalg as la
import torch
from typing import List, Union
from abc import ABC, abstractmethod
from mdpexplore.env.discrete_env import Environment, DiscreteEnv


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
    
    def build_density_from_trajectories(self,
                                        trajectories: List[np.ndarray]):

        t = len(trajectories)
        H = self.env.max_episode_length
        S = self.env.states_num
        A = self.env.actions_num

        d = np.zeros((H, S, A))

        for tau in trajectories:
            # add visitations to density
            for h in range(H):
                d[h, tau[0][h], tau[1][h]] += 1
        
        # normalize densities (each has a single action-state pair per time step)
        d = d / t

        return d