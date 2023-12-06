import autograd.numpy as np
import autograd.numpy.linalg as la
import torch
import cvxpy as cp 
from typing import List, Union
from abc import ABC, abstractmethod
from mdpexplore.env.discrete_env import Environment
from mdpexplore.functionals.reward_functional import RewardFunctional

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
        self.lambd = lambd * np.eye(self.dim)

        if isinstance(lambd, float):
            self.Sigma = sigma * np.ones(self.env.get_states_num())
            self.Sigma_true = self.Sigma
        else:
            self.Sigma = sigma
            self.Sigma_true = self.Sigma

        self.type = "adaptive"

    def eval_basic(self,
                   emissions: np.ndarray,
                   distribution: np.ndarray,
                   unrolls: List[np.ndarray],
                   episodes: int,
                   ) -> float:
        """

        """

        if len(unrolls) > 0:
            aggregated_density = self.build_density_from_trajectories(unrolls)
        else:
            aggregated_density = np.zeros((self.env.max_episode_length, self.env.states_num, self.env.actions_num))

        alpha = len(unrolls) / episodes
        distribution = np.sum(np.sum(distribution, axis = 2), axis = 0)
        aggregated_density = np.sum(np.sum(aggregated_density, axis = 2), axis = 0)

        new_z = np.multiply(emissions.T, distribution / (self.Sigma ** 2)) @ emissions
        agg_z = np.multiply(emissions.T, aggregated_density / (self.Sigma ** 2)) @ emissions

        if self.uniform_alpha:
            z = 1. / episodes * new_z + \
                alpha * agg_z
        else:
            z = (1 - alpha) * new_z + \
                alpha * agg_z
        return z

    def eval(self,
             emissions: np.ndarray,
             distribution: np.ndarray,
             unrolls: List[np.ndarray],
             episodes: int,
             ) -> float:
        alpha = len(unrolls) / episodes

        z = self.eval_basic(emissions, distribution, unrolls, episodes)

        if not self.scale_reg:
            return la.slogdet(z + (1 - alpha) * self.lambd)[1]
        else:
            return la.slogdet(z + self.lambd / episodes)[1]

    def eval_basic_cvxpy(self,
                   emissions: np.ndarray,
                   distribution: cp.Variable,
                   unrolls: List[np.ndarray],
                   episodes: int,
                   ) -> float:
        """

        """

        if len(unrolls) > 0:
            aggregated_density = self.build_density_from_trajectories(unrolls)
        else:
            aggregated_density = np.zeros((self.env.max_episode_length, self.env.states_num, self.env.actions_num))

        alpha = len(unrolls) / episodes
        distribution = cp.sum(distribution, axis = 1)
        aggregated_density = np.sum(np.sum(aggregated_density, axis = 2), axis = 0)

        new_z = emissions.T @ cp.diag(distribution / (self.Sigma ** 2)) @ emissions
        agg_z = emissions.T @ np.diag(aggregated_density / (self.Sigma ** 2)) @ emissions

        if self.uniform_alpha:
            z = 1. / episodes * new_z + \
                alpha * agg_z
        else:
            z = (1 - alpha) * new_z + \
                alpha * agg_z
        return z

    def get_eval_cvxpy(self, 
             emissions: np.ndarray,
             distribution: cp.Variable,
             unrolls: List[np.ndarray],
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
                  emissions: np.ndarray,
                  distribution: np.ndarray,
                  episodes: int,
                  ) -> float:

        distribution = np.sum(np.sum(distribution, axis = 2), axis = 0)

        z = emissions.T @ np.diag(distribution / (self.Sigma_true ** 2)) @ emissions

        if not self.scale_reg:
            return la.slogdet(z + self.lambd)[1]
        else:
            return la.slogdet(z + self.lambd / episodes)[1]


class AdaptiveDesignC(AdaptiveDesignD):
    def __init__(self, env: Environment, lambd: float = 1e-3, scale_reg: bool = False, sigma: float = 1.0, C=None):
        super().__init__(env, lambd=lambd, scale_reg=scale_reg, sigma=sigma)
        self.C = C

    def eval(self,
             emissions: np.ndarray,
             distribution: np.ndarray,
             unrolls: List[np.ndarray],
             episodes: int,
             ) -> float:
        z = self.eval_basic(emissions, distribution, unrolls, episodes)
        if isinstance(self.C, list):
            return np.max([np.trace(la.inv(C @ la.inv(z + (1. / episodes) * self.lambd) @ C.T)) for C in self.C])
        else:
            return np.trace(la.inv(self.C @ la.inv(z + (1. / episodes) * self.lambd) @ self.C.T))

    def eval_full(self,
                  emissions: np.ndarray,
                  distribution: np.ndarray,
                  episodes: int,
                  ) -> float:
        z = np.multiply(emissions.T, distribution / (self.Sigma_true ** 2)) @ emissions
        if isinstance(self.C, list):
            return np.max([np.trace(la.inv(C @ la.inv(z + (1. / episodes) * self.lambd) @ C.T)) for C in self.C])
        else:
            return np.trace(la.inv(self.C @ la.inv(z + (1. / episodes) * self.lambd) @ self.C.T))


class AdaptiveDesignA(RewardFunctional):
    def eval(self,
             emissions: np.ndarray,
             distribution: np.ndarray,
             unrolls: List[np.ndarray],
             episodes: int,
             ) -> float:
        alpha = len(unrolls) / episodes
        z = self.eval_basic(emissions, distribution, unrolls, episodes)
        if not self.scale_reg:
            return -np.trace(la.inv(z + (1 - alpha) * self.lambd))
        else:
            return -np.trace(la.inv(z + (1. / episodes) * self.lambd))
