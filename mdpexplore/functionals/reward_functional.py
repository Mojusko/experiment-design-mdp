import autograd.numpy as np
import autograd.numpy.linalg as la
import torch
from typing import List, Union
from abc import ABC, abstractmethod
from mdpexplore.env.discrete_env import Environment, DiscreteEnv
from mdpexplore.densities.continous_densities import SimpleDeltaDensity, NonStationaryDeltaDensity


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
            for h in range(len(tau[1])):
                d[h, tau[0][h], tau[1][h]] += 1
        
        # normalize densities (each has a single action-state pair per time step)
        for h in range(H):
            if d[h].sum() > 0:
                d[h] = d[h] / d[h].sum()

        return d

class ContinuousRewardFunctional(ABC):

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

        densities = [SimpleDeltaDensity(self.env) for _ in range(self.env.max_episode_length)]

        for tau in trajectories:
            # add visitations to density
            for h in range(len(tau[1])):
                densities[h] = densities[h] + SimpleDeltaDensity(self.env, np.expand_dims(tau[0][h], axis=0), np.expand_dims(tau[1][h], axis=0))
        
        return NonStationaryDeltaDensity(self.env, densities)