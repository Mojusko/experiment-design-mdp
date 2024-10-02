from typing import Union
import autograd.numpy as np
import autograd.numpy.linalg as la
import torch
from typing import List, Union
from abc import ABC, abstractmethod
from mdpexplore.env.discrete_env import Environment
from mdpexplore.functionals.reward_functional import RewardFunctional
from mdpexplore.utils.embedding import EmptyEmbedding
import cvxpy as cp

# for dummy EI we need the cdf and pdf of the normal distribution
from scipy.stats import norm

class DesignRewardBandit(RewardFunctional):
    """
    Classical Bandit Reward Functional, corresponding to cumulative reward.
    """
    def __init__(self, action_space_size):
        super().__init__()
        # This throws a warning
        self.ucb = torch.ones(action_space_size) * torch.inf
        self.type = "adaptive"

    def eval(self, emissions, distribution, visitations, episodes):
        return distribution @ self.ucb

    def eval_full(self,
                  emissions: torch.Tensor,
                  distribution: torch.Tensor,
                  episodes: int = 0) -> float:
        return distribution @ self.ucb
    
    def eval_cvxpy(self, emissions, distribution, visitations, episodes):
        return distribution @ self.ucb


class DesignBestArmLinearBandit(RewardFunctional):

    def __init__(self, action_space_size, lambd, variant: int = 0, eps=0.01):

        super().__init__()

        self.ucbs = torch.ones(action_space_size) * torch.inf
        self.lcbs = -1 * torch.ones(action_space_size) * torch.inf
        self.diff_ucbs = torch.ones((action_space_size, action_space_size)) * torch.inf
        self.diff_lcbs = -1 * torch.ones((action_space_size, action_space_size)) * torch.inf
        self.type = "adaptive"
        self.lambd = lambd
        self.variant = variant
        self.eps = eps

    def _restrict_best_action_space(self, emissions):

        best_pred = torch.argmax(self.ucbs)
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
        restriction_mask_2d = torch.outer(restriction_mask, restriction_mask)
        restricted_diff_ucbs = self.diff_ucbs[restriction_mask_2d].reshape(-1)
        restricted_diff_lcbs = self.diff_lcbs[restriction_mask_2d].reshape(-1)

        # Compute the lower bounds on the possible denominators.
        denominators = torch.ones_like(restricted_diff_ucbs) * self.eps
        denominators[restricted_diff_ucbs < 0] = restricted_diff_ucbs[restricted_diff_ucbs < 0] ** 2
        denominators[restricted_diff_lcbs > 0] = restricted_diff_lcbs[restricted_diff_lcbs > 0] ** 2

#        nominators = torch.apply_along_axis(lambda x: x @ V_eta_inv @ x.T, 1, diffs)
        variance = lambda x: x @ V_eta_inv @ x.T
        results = [variance(x) for x in diffs] 
        nominators = torch.vstack(results)

        if self.variant == 0:
            star_id = torch.argmax(nominators)
            diff_star = diffs[star_id]

            d = torch.min(denominators)
        elif self.variant == 1:
            star_id = torch.argmax(nominators / denominators)
            diff_star = diffs[star_id]

            d = denominators[star_id]
        else:
            raise NotImplemented

        val_star = (nominators / denominators)[star_id]
        return d, diff_star, val_star

    def eval(self, emissions, distribution, visitations, episodes):
        #V_eta = emissions.T @ torch.diag(distribution) @ emissions
        V_eta = torch.einsum('ij,j,jk->ik', emissions.T, distribution, emissions)
        V_eta_inv = torch.linalg.inv(V_eta + self.lambd * torch.eye(V_eta.shape[0]).double())
        _, _, val_star = self._estimate_building_blocks(emissions, V_eta_inv)
        return val_star

    def eval_full(self,
                  emissions: torch.Tensor,
                  distribution: torch.Tensor,
                  episodes: int = 0) -> float:
        distribution = torch.sum(torch.sum(distribution, dim = 2),dim = 0)
        V_eta = emissions.T @ torch.diag(distribution) @ emissions
        V_eta_inv = torch.linalg.inv(V_eta + self.lambd * torch.eye(V_eta.shape[0]).double())

        _, _, val_star = self._estimate_building_blocks(emissions, V_eta_inv)
        return val_star

    def gradient(self,
                 emissions: torch.Tensor,
                 distribution: torch.Tensor):
        """Implements a custom gradient.

        Gradient is custom implemented and uses Danskin's theorem due to the "non-smoothness" of the objective.
        """
        V_eta = emissions.T @ torch.diag(distribution) @ emissions
        V_eta_inv = torch.linalg.inv(V_eta + self.lambd * torch.eye(V_eta.shape[0]))

        d, diff_star, _ = self._estimate_building_blocks(emissions, V_eta_inv)

        def partial(x):
            return - torch.sum(
                (diff_star[:, None] @ diff_star[None, :]) * (V_eta_inv @ x[:, None] @ x[None, :] @ V_eta_inv))

        results = [partial(emission) for emission in emissions]  # emissions is iterated along axis=1
        gradient = torch.stack(results)
        #gradient = torch.apply_along_axis(partial, 1, emissions)
        return gradient / d


class DesignBestArmLinearBanditNoDenominator(RewardFunctional):

    def __init__(self,
                 env,
                 lambd,
                 variant: int = 0,
                 eps=0.01,
                 sigma=0.01,
                 scale_reg=True,
                 init_ucb = torch.inf,
                 sigma_fun = None):

        super().__init__()

        self.env = env
        action_space_size = env.actions_num

        self.ucbs = torch.ones(action_space_size, dtype = torch.float64).view(-1,1) * init_ucb
        self.lcbs = -1 * torch.ones(action_space_size, dtype = torch.float64).view(-1,1) * init_ucb
        self.diff_ucbs = torch.ones((action_space_size, action_space_size), dtype = torch.float64).view(-1,1) * init_ucb * 2
        self.diff_lcbs = -1 * torch.ones((action_space_size, action_space_size), dtype = torch.float64).view(-1,1) * init_ucb * 2
        self.type = "adaptive"
        self.lambd = lambd
        self.variant = variant
        self.eps = eps
        self.sigma_fun = sigma_fun
        self.sigma = sigma
        self.uniform_alpha = False
        self.scale_reg = scale_reg

    def _restrict_best_action_space(self, emissions):

        best_lcb = torch.max(self.lcbs)
        restriction_mask = self.ucbs >= best_lcb

        return restriction_mask

    def _estimate_building_blocks(self, emissions, V_eta_inv):
        restriction_mask = self._restrict_best_action_space(emissions).view(-1)
        restricted_action_space = emissions[restriction_mask.view(-1),:]
        n, m = restricted_action_space.size()

        # Reshape the arrays to have compatible shapes for broadcasting
        # and compute differences between all possible row pairs. Choosing
        # a max over this set upperbounds max_z ||z - z^*||_V_eta_inv
        arr1_reshaped = restricted_action_space.reshape(n, 1, m)
        arr2_reshaped = restricted_action_space.reshape(1, n, m)
        diffs = arr1_reshaped - arr2_reshaped
        diffs = diffs.reshape((-1, diffs.shape[-1]))

        # Restrict diff UCBs and LCBs
        restriction_mask_2d = torch.outer(restriction_mask, restriction_mask)
        #restricted_diff_ucbs = self.diff_ucbs[restriction_mask_2d].reshape(-1)
        #restricted_diff_lcbs = self.diff_lcbs[restriction_mask_2d].reshape(-1)
        #print (restricted_diff_ucbs.size())
#       nominators = torch.apply_along_axis(lambda x: x @ V_eta_inv @ x.T, 1, diffs)

        variance = lambda x: x @ V_eta_inv @ x.T
        results = [variance(x.view(1,-1)) for x in diffs] 
        nominators = torch.vstack(results)

        star_id = torch.argmax(nominators)
        diff_star = diffs[star_id]

        val_star = torch.max(nominators)

        return None, diff_star, val_star

    def eval(self, emissions, distribution, unrolls, episodes):
        if len(unrolls) > 0:
            alpha = (len(unrolls[:-1]) * self.env.max_episode_length + len(unrolls[0][1])) / (episodes * self.env.max_episode_length)
        else:
            alpha = 0

        # calculate the aggregated action state
        aggregated_unrolls = 0
        if len(unrolls) > 0:
            # for t in range(len(unrolls)):
            #     aggregated_unrolls += unrolls[t]
            # aggregated_unrolls = aggregated_unrolls / len(unrolls)
            aggregated_unrolls = self.build_density_from_trajectories(unrolls)
        else:
            aggregated_unrolls = torch.zeros((self.env.max_episode_length, self.env.states_num, self.env.actions_num), dtype = torch.float64)


        if self.sigma_fun is not None:
            distribution = torch.sum(distribution, dim = 0)
            distribution = torch.sum(self.sigma_fun(distribution), dim = 1)

            aggregated_density = torch.sum(aggregated_unrolls, dim = 0)
            aggregated_density = torch.sum(self.sigma_fun(aggregated_density), dim = 1)

            new_V_eta = torch.multiply(emissions.T, distribution) @ emissions
            agg_V_eta = torch.multiply(emissions.T, aggregated_density) @ emissions
        else:
            distribution = torch.sum(torch.sum(distribution, dim = 2), dim = 0)
            aggregated_unrolls = torch.sum(torch.sum(aggregated_unrolls, dim = 2), dim = 0)

            new_V_eta = torch.multiply(emissions.T, distribution / (self.sigma ** 2)) @ emissions
            agg_V_eta = torch.multiply(emissions.T, aggregated_unrolls / (self.sigma ** 2)) @ emissions


        if self.uniform_alpha:
            V_eta = 1. / episodes * new_V_eta + \
                alpha * agg_V_eta
        else:
            V_eta = (1 - alpha) * new_V_eta + \
                alpha * agg_V_eta

        if not self.scale_reg:
            V_eta_inv = torch.linalg.inv(V_eta + (1 - alpha) * self.lambd * torch.eye(V_eta.shape[0]).double())
        else:
            V_eta_inv = torch.linalg.inv(V_eta + self.lambd * torch.eye(V_eta.shape[0]).double())

        _, _, val_star = self._estimate_building_blocks(emissions, V_eta_inv)

        return - val_star

    def get_eval_cvxpy(self,
                    emissions : torch.Tensor,
                    distribution: torch.Tensor,
                    unrolls: List,
                    episodes: int):
        
        alpha = len(unrolls) / episodes

        # calculate the aggregated action state
        aggregated_unrolls = 0
        if len(unrolls) > 0:
            # for t in range(len(unrolls)):
            #     aggregated_unrolls += unrolls[t]
            # aggregated_unrolls = aggregated_unrolls / len(unrolls)
            aggregated_unrolls = self.build_density_from_trajectories(unrolls)
        else:
            aggregated_unrolls = torch.zeros((self.env.max_episode_length, self.env.states_num, self.env.actions_num), dtype=torch.float64)
        
        distribution_summed = 0
        for h in range(self.env.max_episode_length):
            distribution_summed += distribution[h]
        distribution = cp.sum(distribution_summed, dim = 1)
        
        aggregated_unrolls = torch.sum(torch.sum(aggregated_unrolls, dim = 2), dim = 0)

        new_V_eta = emissions.T @ cp.diag(distribution / (self.sigma ** 2)) @ emissions
        agg_V_eta = torch.multiply(emissions.T, aggregated_unrolls / (self.sigma ** 2)) @ emissions

        if self.uniform_alpha:
            V_eta = 1. / episodes * new_V_eta + \
                alpha * agg_V_eta
        else:
            V_eta = (1 - alpha) * new_V_eta + \
                alpha * agg_V_eta

        if not self.scale_reg:
            V_eta = V_eta + (1 - alpha) * self.lambd * torch.eye(V_eta.shape[0]).double()
        else:
            V_eta = V_eta + self.lambd * torch.eye(V_eta.shape[0]).double()

        restriction_mask = self._restrict_best_action_space(emissions)
        restricted_action_space = emissions[restriction_mask]
        n, m = restricted_action_space.shape

        arr1_reshaped = restricted_action_space.reshape(n, 1, m)
        arr2_reshaped = restricted_action_space.reshape(1, n, m)
        diffs = arr1_reshaped - arr2_reshaped
        diffs = diffs.reshape((-1, diffs.shape[-1]))

        # Restrict diff UCBs and LCBs
        restriction_mask_2d = torch.outer(restriction_mask, restriction_mask)
        restricted_diff_ucbs = self.diff_ucbs[restriction_mask_2d].reshape(-1)
        restricted_diff_lcbs = self.diff_lcbs[restriction_mask_2d].reshape(-1)

        nominators = cp.matrix_frac(diffs.T, V_eta) / diffs.shape[0]
        val_star = cp.max(nominators)

        # _, _, val_star = self._estimate_building_blocks(emissions, V_eta_inv)
        return - val_star

    def eval_full(self,
                  emissions: torch.Tensor,
                  distribution: torch.Tensor,
                  episodes: int = 0) -> float:
        
        distribution = torch.sum(torch.sum(distribution, dim = 2), dim = 0)
        V_eta = torch.einsum('ij,j,jk->ik', emissions.T, distribution, emissions)
        V_eta_inv = torch.linalg.inv(V_eta + self.lambd * torch.eye(V_eta.shape[0]).double())

        _, _, val_star = self._estimate_building_blocks(emissions, V_eta_inv)

        return - val_star

    def gradient(self, emissions: torch.Tensor, distribution: torch.Tensor, unrolls, episodes):
        """Implements a custom gradient.

        Gradient is custom implemented and uses Danskin's theorem due to the "non-smoothness" of the objective.
        """

        if len(unrolls) > 0:
            alpha = (len(unrolls[:-1]) * self.env.max_episode_length + len(unrolls[0][1])) / (episodes * self.env.max_episode_length)
        else:
            alpha = 0

        # calculate the aggregated action state
        aggregated_unrolls = 0
        if len(unrolls) > 0:
            # for t in range(len(unrolls)):
            #     aggregated_unrolls += unrolls[t]
            # aggregated_unrolls = aggregated_unrolls / len(unrolls)
            aggregated_unrolls = self.build_density_from_trajectories(unrolls)
        else:
            aggregated_unrolls = torch.zeros((self.env.max_episode_length, self.env.states_num, self.env.actions_num))
        
        # obtain the distribution shape before summing
        H, S, A = distribution.shape
        
        if self.sigma_fun is not None:
            distribution = torch.sum(distribution, dim = 0)
            distribution = torch.sum(self.sigma_fun(distribution), dim = 1)

            aggregated_density = torch.sum(aggregated_unrolls, dim = 0)
            aggregated_density = torch.sum(self.sigma_fun(aggregated_density), dim = 1)

            new_V_eta = torch.multiply(emissions.T, distribution) @ emissions
            agg_V_eta = torch.multiply(emissions.T, aggregated_density) @ emissions
        else:
            distribution = torch.sum(torch.sum(distribution, dim = 2), dim = 0)
            aggregated_unrolls = torch.sum(torch.sum(aggregated_unrolls, dim = 2), dim = 0)

            new_V_eta = torch.multiply(emissions.T, distribution / (self.sigma ** 2)) @ emissions
            agg_V_eta = torch.multiply(emissions.T, aggregated_unrolls / (self.sigma ** 2)) @ emissions

        if self.uniform_alpha:
            V_eta = 1. / episodes * new_V_eta + \
                alpha * agg_V_eta
        else:
            V_eta = (1 - alpha) * new_V_eta + \
                alpha * agg_V_eta

        if not self.scale_reg:
            V_eta_inv = torch.linalg.inv(V_eta + (1 - alpha) * self.lambd * torch.eye(V_eta.shape[0]).double())
        else:
            V_eta_inv = torch.linalg.inv(V_eta + self.lambd * torch.eye(V_eta.shape[0]).double())

        d, diff_star, _ = self._estimate_building_blocks(emissions, V_eta_inv)
        diff_star = diff_star.reshape(1, -1)


        # compute the first part of the matrix multiplication
        mat_1 = V_eta_inv @ diff_star.T @ diff_star @ V_eta_inv
        # calculated batched outer product of the emissions
        mat_2 = torch.einsum('ij,ik->ijk', emissions, emissions)
        # mat_2 = torch.bmm(phi_x.unsqueeze(2), phi_x.unsqueeze(1))
        # multiply together
        mat = torch.matmul(mat_1, mat_2)
        # mat = torch.matmul(mat_1.unsqueeze(0), mat_2)
        # take the trace of each matrix to obtain the gradient
        #gradient = torch.diagonal(mat, axis1=1, axis2=2).sum(axis=1, keepdims = True)
        gradient = torch.diagonal(mat, dim1=1, dim2=2).sum(dim = 1, keepdim=True)

        # gradient = torch.diagonal(mat, dim1=1, dim2=2).sum(dim=1, keepdim = True)
    


        if self.sigma_fun is not None:
            # First, reshape the gradient to be of shape (S, 1)
            gradient = gradient.reshape(-1, 1)
            
            # Repeat the gradient to be of shape (S, A)
            gradient = gradient.repeat(1, A)  # Repeating along the second dimension (axis 1)
            
            # Now scale by the noise using self.sigma_fun
            gradient = self.sigma_fun(gradient)
            
            # Reshape the gradient to be of shape (1, S, A)
            gradient = gradient.reshape(1, S, A)
            
            # Repeat the gradient to be of shape (H, S, A)
            gradient = gradient.repeat(H, 1, 1)  # Repeating along the first dimension (axis 0)

            return gradient

        else:
            gradient = gradient.reshape(1, -1, 1)
            
            # Repeat the gradient to be of shape (H, S, A)
            gradient = gradient.repeat(H, 1, 1)  # Repeating along the first dimension (axis 0)

            gradient = gradient.repeat(1, 1, A)  # Repeating along the third dimension (axis 2)

        # if self.sigma_fun is not None:
        #     # first repeat the gradient to be of shape (S, A)
        #     gradient = gradient.reshape(-1, 1)
        #     gradient = np.repeat(gradient, A, dim = 1)
        #     # now scale by the noise
        #     gradient = self.sigma_fun(gradient)
        #     # now repeat the gradient to be of shape (H, S, A)
        #     gradient = gradient.reshape(1, S, A)

        #     # repeat the gradient to be of shape (H, S, A)
        #     gradient = np.repeat(gradient, H, dim = 0)

        #     return gradient
        
        # else:
        #     # repeat the gradient to be of shape (H, S, A)
        #     gradient = gradient.reshape(1, -1, 1)

        #     gradient = np.repeat(gradient, H, dim = 0)
        #     gradient = np.repeat(gradient, A, dim = 2)

        return gradient


