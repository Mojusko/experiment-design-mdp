import torch
from typing import List, Union
from abc import ABC, abstractmethod
from doexpy.env.discrete_env import Environment, DiscreteEnv
from doexpy.densities.continous_densities import SimpleDeltaDensity, NonStationaryDeltaDensity


class RewardFunctional(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def eval(self,
             emissions: torch.Tensor,
             distribution: torch.Tensor,
             episodes: int):
        pass

    def get_type(self):
        return self.type
    
    def build_density_from_trajectories(self,
                                        trajectories: List[torch.Tensor]):

        t = len(trajectories)
        H = self.env.max_episode_length
        S = self.env.states_num
        A = self.env.actions_num

        d = torch.zeros(size = (H, S, A), dtype = torch.float64)

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
             emissions: torch.Tensor,
             distribution: torch.Tensor,
             episodes: int):
        pass

    def get_type(self):
        return self.type
    
    def build_density_from_trajectories(self,
                                        trajectories: List[torch.Tensor]):

        densities = [SimpleDeltaDensity(self.env) for _ in range(self.env.max_episode_length)]

        for tau in trajectories:
            # add visitations to density
            for h in range(len(tau[1])):
                densities[h] = densities[h] + SimpleDeltaDensity(self.env, tau[0][h], tau[1][h])
        
        return NonStationaryDeltaDensity(self.env, densities)