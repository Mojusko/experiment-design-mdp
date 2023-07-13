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


class DesignA(RewardFunctional):

    def __init__(self,
                 env: Environment,
                 lambd: float = 1e-3):
        super().__init__()
        self.lambd = lambd
        self.type = "static"
        self.env = env

    def eval(self,
             emissions: np.ndarray,
             distribution: np.ndarray,
             episodes: int = 0
             ) -> float:
        z = emissions.T @ np.diag(distribution) @ emissions
        return -np.trace(la.inv(z + self.lambd * np.eye(z.shape[0])))


class DesignD(RewardFunctional):
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
        z = emissions.T @ np.diag(distribution / (self.Sigma ** 2)) @ emissions
        return np.linalg.slogdet(z + self.lambd / episodes)[1]

    def eval_full(self,
                  emissions: np.ndarray,
                  distribution: np.ndarray,
                  episodes: int,
                  ) -> float:
        z = emissions.T @ np.diag(distribution / (self.Sigma_true ** 2)) @ emissions
        return np.linalg.slogdet(z + self.lambd / episodes)[1]


class DesignC(RewardFunctional):

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


class DesignBayesD(RewardFunctional):

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
        alpha = len(unrolls) / episodes
        aggregated_unrolls = sum(unrolls) / len(unrolls) if len(unrolls) > 0 else np.zeros(emissions.shape[0])

        new_z = np.multiply(emissions.T, distribution / (self.Sigma ** 2)) @ emissions
        agg_z = np.multiply(emissions.T, aggregated_unrolls / (self.Sigma ** 2)) @ emissions

        # new_z = emissions.T @ np.diag(distribution/(self.Sigma**2)) @ emissions
        # agg_z = emissions.T @ np.diag(aggregated_unrolls/(self.Sigma**2)) @ emissions

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
            return np.linalg.slogdet(z + (1 - alpha) * self.lambd)[1]
        else:
            return np.linalg.slogdet(z + self.lambd / episodes)[1]

    def eval_full(self,
                  emissions: np.ndarray,
                  distribution: np.ndarray,
                  episodes: int,
                  ) -> float:

        z = emissions.T @ np.diag(distribution / (self.Sigma_true ** 2)) @ emissions

        if not self.scale_reg:
            return np.linalg.slogdet(z + self.lambd)[1]
        else:
            return np.linalg.slogdet(z + self.lambd / episodes)[1]


class DesignBayesC(DesignBayesD):
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


class DesignBayesA(RewardFunctional):
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


class DesignRewardBandit(RewardFunctional):

    def __init__(self, action_space_size):
        super().__init__()
        # This throws a warning
        self.mu = np.ones(action_space_size) * np.inf
        self.type = "adaptive"

    def eval(self, emissions, distribution, visitations, episodes):
        return distribution @ self.mu - np.max(self.mu)

    def eval_full(self,
                  emissions: np.ndarray,
                  distribution: np.ndarray,
                  episodes: int = 0) -> float:
        return distribution @ self.mu - np.max(self.mu)


class DesignBestArmLinearBandit(RewardFunctional):

    def __init__(self, action_space_size, lambd, variant: int = 0, eps=0.01):

        super().__init__()

        self.ucbs = np.ones(action_space_size) * np.inf
        self.lcbs = -1 * np.ones(action_space_size) * np.inf
        self.diff_ucbs = np.ones((action_space_size, action_space_size)) * np.inf
        self.diff_lcbs = -1 * np.ones((action_space_size, action_space_size)) * np.inf
        self.type = "adaptive"
        self.lambd = lambd
        self.variant = variant
        self.eps = eps

    def _restrict_best_action_space(self, emissions):

        best_pred = np.argmax(self.ucbs)
        best_lb = self.lcbs[best_pred]
        restriction_mask = self.ucbs >= best_lb
        return restriction_mask

    def _estimate_building_blocks(self, emissions, V_eta_inv):
        restriction_mask = self._restrict_best_action_space(emissions)
        restricted_action_space = emissions[restriction_mask]
        n, m = restricted_action_space.shape

        # Reshape the arrays to have compatible shapes for broadcasting
        # and compute differences between all possible row pairs. Choosing
        # a max over this set upperbounds max_z ||z - z^*||_V_eta_inv
        arr1_reshaped = restricted_action_space.reshape(n, 1, m)
        arr2_reshaped = restricted_action_space.reshape(1, n, m)
        diffs = arr1_reshaped - arr2_reshaped
        diffs = diffs.reshape((-1, diffs.shape[-1]))

        # Restrict diff UCBs and LCBs
        restriction_mask_2d = np.outer(restriction_mask, restriction_mask)
        restricted_diff_ucbs = self.diff_ucbs[restriction_mask_2d].reshape(-1)
        restricted_diff_lcbs = self.diff_lcbs[restriction_mask_2d].reshape(-1)

        # Compute the lower bounds on the possible denominators.
        denominators = np.ones_like(restricted_diff_ucbs) * self.eps
        denominators[restricted_diff_ucbs < 0] = restricted_diff_ucbs[restricted_diff_ucbs < 0] ** 2
        denominators[restricted_diff_lcbs > 0] = restricted_diff_lcbs[restricted_diff_lcbs > 0] ** 2

        nominators = np.apply_along_axis(lambda x: x @ V_eta_inv @ x.T, 1, diffs)

        if self.variant == 0:
            star_id = np.argmax(nominators)
            diff_star = diffs[star_id]

            d = np.min(denominators)
        elif self.variant == 1:
            star_id = np.argmax(nominators / denominators)
            diff_star = diffs[star_id]

            d = denominators[star_id]
        else:
            raise NotImplemented

        val_star = (nominators / denominators)[star_id]
        return d, diff_star, val_star

    def eval(self, emissions, distribution, visitations, episodes):
        V_eta = emissions.T @ np.diag(distribution) @ emissions
        V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(V_eta.shape[0]))

        _, _, val_star = self._estimate_building_blocks(emissions, V_eta_inv)
        return val_star

    def eval_full(self,
                  emissions: np.ndarray,
                  distribution: np.ndarray,
                  episodes: int = 0) -> float:
        V_eta = emissions.T @ np.diag(distribution) @ emissions
        V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(V_eta.shape[0]))

        _, _, val_star = self._estimate_building_blocks(emissions, V_eta_inv)
        return val_star

    def gradient(self, emissions: np.ndarray, distribution: np.array):
        """Implements a custom gradient.

        Gradient is custom implemented and uses Danskin's theorem due to the "non-smoothness" of the objective.
        """
        V_eta = emissions.T @ np.diag(distribution) @ emissions
        V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(V_eta.shape[0]))

        d, diff_star, _ = self._estimate_building_blocks(emissions, V_eta_inv)

        def partial(x):
            return - np.sum(
                (diff_star[:, None] @ diff_star[None, :]) * (V_eta_inv @ x[:, None] @ x[None, :] @ V_eta_inv))

        gradient = np.apply_along_axis(partial, 1, emissions)
        return gradient / d
