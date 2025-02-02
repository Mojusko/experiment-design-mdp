import torch.linalg as la
import torch
import cvxpy as cp 
from typing import List, Union, Callable
from abc import ABC, abstractmethod
from doexpy.env.discrete_env import Environment
from doexpy.functionals.reward_functional import RewardFunctional
from doexpy.functionals.doe_static_functionals import MultiPolicyOrigDesignD, MultiPolicyOrigDesignA

class AdaptiveDesignD(RewardFunctional):

    def __init__(self,
                 env: Environment,
                 lambd: float = 1e-3,
                 scale_reg: bool = True,
                 uniform_alpha: bool = False,
                 sigma: float = 1.0):

        super().__init__()

        self.dim = env.get_dim()
        self.scale_reg = scale_reg
        self.uniform_alpha = uniform_alpha
        self.env = env
        self.lambd = lambd * torch.eye(self.dim, dtype=torch.float64)

        if isinstance(lambd, float):
            self.Sigma = sigma * torch.ones(self.env.get_states_num())
            self.Sigma_true = self.Sigma
        else:
            self.Sigma = sigma
            self.Sigma_true = self.Sigma

        self.type = "adaptive"

    def eval_basic(self,
                   emissions: torch.Tensor,
                   distribution: torch.Tensor,
                   unrolls: List[torch.Tensor],
                   episodes: int,
                   ) -> float:
        """

        """

        if len(unrolls) > 0:
            aggregated_density = self.build_density_from_trajectories(unrolls)
        else:
            aggregated_density = torch.zeros((self.env.max_episode_length, self.env.states_num, self.env.actions_num), dtype = torch.float64)

        alpha = len(unrolls) / episodes
        
        distribution = torch.sum(torch.sum(distribution, dim = 2), dim = 0)
        aggregated_density = torch.sum(torch.sum(aggregated_density, dim = 2), dim = 0)


        new_z = torch.multiply(emissions.T, distribution / (self.Sigma ** 2)) @ emissions
        agg_z = torch.multiply(emissions.T, aggregated_density / (self.Sigma ** 2)) @ emissions

        if self.uniform_alpha:
            z = 1. / episodes * new_z + \
                alpha * agg_z
        else:
            z = (1 - alpha) * new_z + \
                alpha * agg_z
        return z

    def eval(self,
             emissions: torch.Tensor,
             distribution: torch.Tensor,
             unrolls: List[torch.Tensor],
             episodes: int,
             ) -> float:
        alpha = len(unrolls) / episodes

        z = self.eval_basic(emissions, distribution, unrolls, episodes)

        if not self.scale_reg:
            return la.slogdet(z + (1 - alpha) * self.lambd)[1]
        else:
            return la.slogdet(z + self.lambd / episodes)[1]

    def eval_basic_cvxpy(self,
                   emissions: torch.Tensor,
                   distribution: cp.Variable,
                   unrolls: List[torch.Tensor],
                   episodes: int,
                   ) -> float:
        """

        """

        if len(unrolls) > 0:
            aggregated_density = self.build_density_from_trajectories(unrolls)
        else:
            aggregated_density = torch.zeros((self.env.max_episode_length, self.env.states_num, self.env.actions_num))

        alpha = len(unrolls) / episodes
        distribution = cp.sum(distribution, dim = 1)
        aggregated_density = torch.sum(torch.sum(aggregated_density, dim = 2), dim = 0)

        new_z = emissions.T @ cp.diag(distribution / (self.Sigma ** 2)) @ emissions
        agg_z = emissions.T @ torch.diag(aggregated_density / (self.Sigma ** 2)) @ emissions

        if self.uniform_alpha:
            z = 1. / episodes * new_z + \
                alpha * agg_z
        else:
            z = (1 - alpha) * new_z + \
                alpha * agg_z
        return z

    def get_eval_cvxpy(self, 
             emissions: torch.Tensor,
             distribution: cp.Variable,
             unrolls: List[torch.Tensor],
             episodes: int,
             )->cp.Expression:

        distribution_summed = 0
        for h in range(self.env.max_episode_length):
            distribution_summed += distribution[h]
        
        alpha = len(unrolls) / episodes
        z = self.eval_basic_cvxpy(emissions, distribution_summed, unrolls, episodes)
        if not self.scale_reg:
            return cp.log_det(z + (1 - alpha) * self.lambd)
        else:
            return cp.log_det(z + self.lambd / episodes)
        
    def eval_full(self,
                  emissions: torch.Tensor,
                  distribution: torch.Tensor,
                  episodes: int,
                  ) -> float:

        distribution = torch.sum(torch.sum(distribution, dim = 2), dim = 0)

        z = emissions.T @ torch.diag(distribution / (self.Sigma_true ** 2)) @ emissions

        if not self.scale_reg:
            return la.slogdet(z + self.lambd)[1]
        else:
            return la.slogdet(z + self.lambd / episodes)[1]


class AdaptiveDesignC(AdaptiveDesignD):
    def __init__(self, env: Environment, lambd: float = 1e-3, scale_reg: bool = False, sigma: float = 1.0, C=None):
        super().__init__(env, lambd=lambd, scale_reg=scale_reg, sigma=sigma)
        self.C = C

    def eval(self,
             emissions: torch.Tensor,
             distribution: torch.Tensor,
             unrolls: List[torch.Tensor],
             episodes: int,
             ) -> float:
        z = self.eval_basic(emissions, distribution, unrolls, episodes)
        if isinstance(self.C, list):
            return torch.max(torch.stack([torch.trace(la.inv(C @ la.inv(z + (1. / episodes) * self.lambd) @ C.T)) for C in self.C]))
        else:
            return torch.trace(la.inv(self.C @ la.inv(z + (1. / episodes) * self.lambd) @ self.C.T))

    def eval_full(self,
                  emissions: torch.Tensor,
                  distribution: torch.Tensor,
                  episodes: int,
                  ) -> float:
        distribution = torch.sum(torch.sum(distribution, dim = 2), dim = 0)
        z = torch.multiply(emissions.T, distribution / (self.Sigma_true ** 2)) @ emissions
        if isinstance(self.C, list):
            return torch.max(torch.stack([torch.trace(la.inv(C @ la.inv(z + (1. / episodes) * self.lambd) @ C.T)) for C in self.C]))
        else:
            return torch.trace(la.inv(self.C @ la.inv(z + (1. / episodes) * self.lambd) @ self.C.T))

class AdaptiveDesignHeteroD(AdaptiveDesignD):
    def __init__(self,
                 env: Environment,
                 lambd: float = 1e-3,
                 scale_reg: bool = True,
                 sigma: float = 1.0,
                 sigma_fun: Callable = lambda x: 1.0):
        super().__init__(env, lambd=lambd, scale_reg=scale_reg, sigma=sigma)
        self.sigma_fun = sigma_fun
    def eval_basic(self,
                   emissions: torch.Tensor,
                   distribution: torch.Tensor,
                   unrolls: List[torch.Tensor],
                   episodes: int,
                   ) -> float:
        """

        """

        if len(unrolls) > 0:
            aggregated_density = self.build_density_from_trajectories(unrolls)
        else:
            aggregated_density = torch.zeros((self.env.max_episode_length, self.env.states_num, self.env.actions_num))

        alpha = len(unrolls) / episodes

        distribution = torch.sum(distribution, dim = 0)
        distribution = torch.sum(self.sigma_fun(distribution), dim = 1)

        aggregated_density = torch.sum(aggregated_density, dim = 0)
        aggregated_density = torch.sum(self.sigma_fun(aggregated_density), dim = 1)

        new_z = torch.multiply(emissions.T, distribution) @ emissions
        agg_z = torch.multiply(emissions.T, aggregated_density) @ emissions

        if self.uniform_alpha:
            z = 1. / episodes * new_z + \
                alpha * agg_z
        else:
            z = (1 - alpha) * new_z + \
                alpha * agg_z
        return z

class AdaptiveDesignA(RewardFunctional):
    def eval(self,
             emissions: torch.Tensor,
             distribution: torch.Tensor,
             unrolls: List[torch.Tensor],
             episodes: int,
             ) -> float:
        alpha = len(unrolls) / episodes
        distribution = torch.sum(torch.sum(distribution, dim = 2), dim = 0)
        z = self.eval_basic(emissions, distribution, unrolls, episodes)
        if not self.scale_reg:
            return -torch.trace(la.inv(z + (1 - alpha) * self.lambd))
        else:
            return -torch.trace(la.inv(z + (1. / episodes) * self.lambd))

class AdaptiveOrigDesignD(MultiPolicyOrigDesignD):
    def __init__(self, env, lambd=1e-3, dim=0, uniform_alpha=False):
        super().__init__(env, lambd, dim)
        self.type = "adaptive"
        self.uniform_alpha = uniform_alpha

    def eval(self, emissions, distributions, visitations_per_policy, episodes):
        # For training - handle adaptive weighting
        agg_densities = [
            self.build_density_from_trajectories(visitations) 
            for visitations in visitations_per_policy
        ]
        
        alpha = len(visitations_per_policy[0]) / episodes
        
        # Calculate information matrices
        new_z = super()._calculate_z(emissions, distributions, episodes)
        agg_z = super()._calculate_z(emissions, agg_densities, episodes)
        
        # Weight combination based on alpha
        z = (1.0 / episodes) * new_z + alpha * agg_z if self.uniform_alpha else (1 - alpha) * new_z + alpha * agg_z
        
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        return torch.linalg.slogdet(z + self.lambd/episodes * eye)[1]

    def eval_full(self, emissions, distributions, episodes):
        # For final evaluation - just use distributions directly
        z = super()._calculate_z(emissions, distributions, episodes)
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        return torch.linalg.slogdet(z + self.lambd/episodes * eye)[1]

class AdaptiveOrigDesignA(MultiPolicyOrigDesignA):
    def __init__(self, env, lambd=1e-3, dim=0, uniform_alpha=False):
        super().__init__(env, lambd, dim)
        self.type = "adaptive"
        self.uniform_alpha = uniform_alpha

    def eval(self, emissions, distributions, visitations_per_policy, episodes):
        # Handle adaptive weighting based on history
        agg_densities = [
            self.build_density_from_trajectories(visitations) 
            for visitations in visitations_per_policy
        ]
        
        alpha = len(visitations_per_policy[0]) / episodes
        
        # Calculate information matrices
        new_z = super()._calculate_z(emissions, distributions, episodes)
        agg_z = super()._calculate_z(emissions, agg_densities, episodes)
        
        # Weight combination based on alpha
        z = (1.0 / episodes) * new_z + alpha * agg_z if self.uniform_alpha else (1 - alpha) * new_z + alpha * agg_z
        
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        if self.V is None:
            return -torch.trace(torch.linalg.inv(z + self.lambd/episodes * eye))
        else:
            return -torch.trace(self.V @ torch.linalg.inv(z + self.lambd/episodes * eye))

    def eval_full(self, emissions, distributions, episodes):
        # For final evaluation - just use distributions directly
        z = super()._calculate_z(emissions, distributions, episodes)
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        if self.V is None:
            return -torch.trace(torch.linalg.inv(z + self.lambd/episodes * eye))
        else:
            return -torch.trace(self.V @ torch.linalg.inv(z + self.lambd/episodes * eye))
