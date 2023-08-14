import autograd.numpy as np
import autograd.numpy.linalg as la
import torch
from typing import List, Union
from abc import ABC, abstractmethod
from mdpexplore.env.discrete_env import Environment
from mdpexplore.functionals.reward_functional import RewardFunctional
import cvxpy as cp 



class DesignRewardBandit(RewardFunctional):

    def __init__(self, action_space_size):
        super().__init__()
        # This throws a warning
        self.ucb = np.ones(action_space_size) * np.inf
        self.type = "adaptive"

    def eval(self, emissions, distribution, visitations, episodes):
        return distribution @ self.ucb

    def eval_full(self,
                  emissions: np.ndarray,
                  distribution: np.ndarray,
                  episodes: int = 0) -> float:
        return distribution @ self.ucb
    
    def eval_cvxpy(self, emissions, distribution, visitations, episodes):
        return distribution @ self.ucb


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
        distribution = np.sum(np.sum(distribution, axis = 2),axis = 0)
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


class DesignBestArmLinearBanditNoDenominator(RewardFunctional):

    def __init__(self, env, lambd, variant: int = 0, eps=0.01, sigma=0.01, scale_reg=True, mix_objectives=(False, 0), init_ucb = np.inf):

        super().__init__()

        self.env = env
        action_space_size = env.actions_num

        self.ucbs = np.ones(action_space_size) * init_ucb
        self.lcbs = -1 * np.ones(action_space_size) * init_ucb
        self.diff_ucbs = np.ones((action_space_size, action_space_size)) * init_ucb * 2
        self.diff_lcbs = -1 * np.ones((action_space_size, action_space_size)) * init_ucb * 2
        self.type = "adaptive"
        self.lambd = lambd
        self.variant = variant
        self.eps = eps
        self.sigma = sigma
        self.uniform_alpha = False
        self.scale_reg = scale_reg
        # objectives trade-off
        self.mix_objectives, self.mix_ratio = mix_objectives

    def _restrict_best_action_space(self, emissions):

        best_lcb = np.max(self.lcbs)
        restriction_mask = self.ucbs >= best_lcb

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

        nominators = np.apply_along_axis(lambda x: x @ V_eta_inv @ x.T, 1, diffs)
        star_id = np.argmax(nominators)
        diff_star = diffs[star_id]

        val_star = np.max(nominators)

        return None, diff_star, val_star

    def eval(self, emissions, distribution, unrolls, episodes):

        alpha = len(unrolls) / episodes
        #TODO change this to allow for different types of unrolls
        alpha = len(unrolls[0][1]) / self.env.max_episode_length

        # calculate the aggregated action state
        aggregated_unrolls = 0
        if len(unrolls) > 0:
            # for t in range(len(unrolls)):
            #     aggregated_unrolls += unrolls[t]
            # aggregated_unrolls = aggregated_unrolls / len(unrolls)
            aggregated_unrolls = self.build_density_from_trajectories(unrolls)
        else:
            aggregated_unrolls = np.zeros((self.env.max_episode_length, self.env.states_num, self.env.actions_num))
        
        distribution = np.sum(np.sum(distribution, axis = 2), axis = 0)
        aggregated_unrolls = np.sum(np.sum(aggregated_unrolls, axis = 2), axis = 0)

        new_V_eta = np.multiply(emissions.T, distribution / (self.sigma ** 2)) @ emissions
        agg_V_eta = np.multiply(emissions.T, aggregated_unrolls / (self.sigma ** 2)) @ emissions

        if self.uniform_alpha:
            V_eta = 1. / episodes * new_V_eta + \
                alpha * agg_V_eta
        else:
            V_eta = (1 - alpha) * new_V_eta + \
                alpha * agg_V_eta

        if not self.scale_reg:
            V_eta_inv = np.linalg.inv(V_eta + (1 - alpha) * self.lambd * np.identity(V_eta.shape[0]))
        else:
            V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(V_eta.shape[0]))

        _, _, val_star = self._estimate_building_blocks(emissions, V_eta_inv)

        if self.mix_objectives:
            return - (1 - self.mix_ratio) * val_star + self.mix_ratio * self.ucbs @ distribution
        else:
            return - val_star

    def get_eval_cvxpy(self, emissions, distribution, unrolls, episodes):
        alpha = len(unrolls) / episodes

        # calculate the aggregated action state
        aggregated_unrolls = 0
        if len(unrolls) > 0:
            # for t in range(len(unrolls)):
            #     aggregated_unrolls += unrolls[t]
            # aggregated_unrolls = aggregated_unrolls / len(unrolls)
            aggregated_unrolls = self.build_density_from_trajectories(unrolls)
        else:
            aggregated_unrolls = np.zeros((self.env.max_episode_length, self.env.states_num, self.env.actions_num))
        
        distribution_summed = 0
        for h in range(self.env.max_episode_length):
            distribution_summed += distribution[h]
        distribution = cp.sum(distribution_summed, axis = 1)
        
        aggregated_unrolls = np.sum(np.sum(aggregated_unrolls, axis = 2), axis = 0)

        new_V_eta = emissions.T @ cp.diag(distribution / (self.sigma ** 2)) @ emissions
        agg_V_eta = np.multiply(emissions.T, aggregated_unrolls / (self.sigma ** 2)) @ emissions

        if self.uniform_alpha:
            V_eta = 1. / episodes * new_V_eta + \
                alpha * agg_V_eta
        else:
            V_eta = (1 - alpha) * new_V_eta + \
                alpha * agg_V_eta

        if not self.scale_reg:
            V_eta = V_eta + (1 - alpha) * self.lambd * np.identity(V_eta.shape[0])
        else:
            V_eta = V_eta + self.lambd * np.identity(V_eta.shape[0])

        restriction_mask = self._restrict_best_action_space(emissions)
        restricted_action_space = emissions[restriction_mask]
        n, m = restricted_action_space.shape

        arr1_reshaped = restricted_action_space.reshape(n, 1, m)
        arr2_reshaped = restricted_action_space.reshape(1, n, m)
        diffs = arr1_reshaped - arr2_reshaped
        diffs = diffs.reshape((-1, diffs.shape[-1]))

        # Restrict diff UCBs and LCBs
        restriction_mask_2d = np.outer(restriction_mask, restriction_mask)
        restricted_diff_ucbs = self.diff_ucbs[restriction_mask_2d].reshape(-1)
        restricted_diff_lcbs = self.diff_lcbs[restriction_mask_2d].reshape(-1)

        # take the mean instead of the max
        nominators = cp.matrix_frac(diffs.T, V_eta) / diffs.shape[0]
        val_star = cp.max(nominators)

        # _, _, val_star = self._estimate_building_blocks(emissions, V_eta_inv)

        if self.mix_objectives:
            return - (1 - self.mix_ratio) * val_star + self.mix_ratio * self.ucbs @ distribution
        else:
            return - val_star

    def eval_full(self,
                  emissions: np.ndarray,
                  distribution: np.ndarray,
                  episodes: int = 0) -> float:
        
        distribution = np.sum(np.sum(distribution, axis = 2), axis = 0)

        V_eta = emissions.T @ np.diag(distribution) @ emissions
        V_eta_inv = np.linalg.inv(V_eta + self.lambd * np.identity(V_eta.shape[0]))

        _, _, val_star = self._estimate_building_blocks(emissions, V_eta_inv)

        if self.mix_objectives:
            return - (1 - self.mix_ratio) * val_star + self.mix_ratio * self.ucbs @ distribution
        else:
            return - val_star