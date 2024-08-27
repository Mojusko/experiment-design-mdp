import autograd.numpy as np
import autograd.numpy.linalg as la
import torch
from typing import List, Union
from abc import ABC, abstractmethod
from mdpexplore.env.discrete_env import Environment
from mdpexplore.functionals.reward_functional import RewardFunctional

class ExperimentDesignFunctional(RewardFunctional):

    def __init__(self, dim = 0):
        super().__init__()
        self.dim = dim

class DesignA(ExperimentDesignFunctional):

    def __init__(self,
                 env: Environment,
                 lambd: float = 1e-3,
                 dim = 0):
        super().__init__(dim = dim)
        self.lambd = lambd
        self.type = "static"
        self.env = env

    def eval(self,
             emissions: np.ndarray,
             distribution: np.ndarray,
             episodes: int = 0
             ) -> float:
        if self.dim == 0:
            distribution = np.sum(np.sum(distribution, axis = 2), axis = 0)
        elif self.dim == 1: # actions matter 
            distribution = np.sum(np.sum(distribution, axis = 1), axis = 0)
        #z = emissions.T @ np.diag(distribution) @ emissions
        z = np.einsum('ij,j,jk->ik', emissions.T, distribution, emissions)
        return -np.trace(la.inv(z + self.lambd/episodes * np.eye(z.shape[0])))

    def eval_full(self,
                  emissions: np.ndarray,
                  distribution: np.ndarray,
                  episodes: int,
                  ) -> float:
        if self.dim == 0:
            distribution = np.sum(np.sum(distribution, axis = 2), axis = 0)
        elif self.dim == 1: # actions matter 
            distribution = np.sum(np.sum(distribution, axis = 1), axis = 0)
        #z = emissions.T @ np.diag(distribution) @ emissions
        z = np.einsum('ij,j,jk->ik', emissions.T, distribution, emissions)
        return -np.trace(la.inv(z + self.lambd/episodes * np.eye(z.shape[0])))

class DesignD(ExperimentDesignFunctional):
    def __init__(self,
                 env: Environment,
                 lambd: float = 1e-3,
                 scale_reg=True,
                 sigma: float = 1.):
        super().__init__()
        self.env = env
        self.dim = self.env.get_dim()
        self.lambd = lambd
        self.scale_reg = scale_reg
        self.type = "static"
        self.lambd = lambd * np.eye(self.dim)

        if isinstance(lambd, float):
            self.Sigma = sigma * np.ones(self.env.get_states_num())
            self.Sigma_true = self.Sigma
        else:
            self.Sigma = sigma
            self.Sigma_true = self.Sigma

    def eval(self,
             emissions: np.ndarray,
             distribution: np.ndarray,
             episodes: int
             ) -> float:
        if self.dim == 0: #states matter 
            distribution = np.sum(np.sum(distribution, axis = 2), axis = 0)
        elif self.dim == 1: # actions matter
            distribution = np.sum(np.sum(distribution, axis = 1), axis = 0)
        z = emissions.T @ np.diag(distribution / (self.Sigma ** 2)) @ emissions
        return np.linalg.slogdet(z + self.lambd / episodes)[1]

    def eval_full(self,
                  emissions: np.ndarray,
                  distribution: np.ndarray,
                  episodes: int,
                  ) -> float:
        if self.dim == 0:
            distribution = np.sum(np.sum(distribution, axis = 2), axis = 0)
        elif self.dim == 1: 
            distribution = np.sum(np.sum(distribution, axis = 1), axis = 0)
        z = emissions.T @ np.diag(distribution / (self.Sigma_true ** 2)) @ emissions
        return np.linalg.slogdet(z + self.lambd / episodes)[1]


class DesignE(DesignD):
    def eval(self,
             emissions: np.ndarray,
             distribution: np.ndarray,
             episodes: int
             ) -> float:
        if self.dim == 0: #states matter 
            distribution = np.sum(np.sum(distribution, axis = 2), axis = 0)
        elif self.dim == 1: # actions matter
            distribution = np.sum(np.sum(distribution, axis = 1), axis = 0)

        z = emissions.T @ np.diag(distribution / (self.Sigma ** 2)) @ emissions
        return np.linalg.eigvalsh(z + self.lambd / episodes)[0]

    def eval_full(self,
                  emissions: np.ndarray,
                  distribution: np.ndarray,
                  episodes: int,
                  ) -> float:
        if self.dim == 0:
            distribution = np.sum(np.sum(distribution, axis = 2), axis = 0)
        elif self.dim == 1: 
            distribution = np.sum(np.sum(distribution, axis = 1), axis = 0)
        z = emissions.T @ np.diag(distribution / (self.Sigma_true ** 2)) @ emissions
        return np.linalg.eigvalsh(z + self.lambd / episodes)[0]
    
class DesignC(ExperimentDesignFunctional):

    def __init__(self,
                 env: Environment,
                 lambd: float = 1e-3,
                 sigma: float = 1.,
                 C: Union[np.array, List, None] = None):
        super().__init__()

        self.env = env
        self.dim = self.env.get_dim()
        self.lambd = lambd * np.eye(self.dim)
        self.sigma = sigma

        self.C = C
        self.type = "static"

    def eval(self,
             emissions: np.ndarray,
             distribution: np.ndarray,
             episodes: int = 0) -> float:
        distribution = np.sum(np.sum(distribution, axis = 2), axis = 0)
        z = np.multiply(emissions.T, distribution / (self.sigma ** 2)) @ emissions
        if isinstance(self.C, list):
            return np.max([np.trace(la.inv(C @ la.inv(z + self.lambd / episodes) @ C.T)) for C in self.C])
        else:
            return np.trace(la.inv(self.C @ la.inv(z + self.lambd / episodes) @ self.C.T))

    def eval_full(self,
                  emissions: np.ndarray,
                  distribution: np.ndarray,
                  episodes: int = 0) -> float:
        return self.eval(emissions, distribution, episodes)
