import torch.linalg as la
import torch
import cvxpy as cp 
from typing import List, Union, Callable
from abc import ABC, abstractmethod
from doexpy.env.discrete_env import Environment
from doexpy.functionals.reward_functional import RewardFunctional
from doexpy.functionals.doe_static_functionals import MultiPolicyOrigDesignD, MultiPolicyOrigDesignA, StochasticMultiPolicyRewardFunctionalMixin, compute_mask

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

class StochasticAdaptiveOrigDesignA(StochasticMultiPolicyRewardFunctionalMixin, MultiPolicyOrigDesignA):
    def __init__(self, env, lambd=1e-3, dim=0, uniform_alpha=False, V=None, batch_size=500):
        super().__init__(env, lambd, dim, batch_size=batch_size)
        self.type = "adaptive"
        self.uniform_alpha = uniform_alpha
        self.V = V

    def eval(self, emissions, distributions, visitations_per_policy, episodes, should_mask=True):
        # Compute agg_densities for each policy's visitation history.
        agg_densities = [
            self.build_density_from_trajectories(visitations)
            for visitations in visitations_per_policy
        ]

        # For Stationary distributions, convert history density to stationary (S x A) format (TODO: refactor this, only applies to LLM)
        for i in range(len(distributions)):
            if len(distributions[i].shape) < len(agg_densities[i].shape):
                agg_densities[i] = agg_densities[i].diagonal(dim1=0, dim2=1).T

        union_mask = None
        
        if should_mask:
            # Combine all visitation histories. For instance, if each agg_density is (S x A),
            # you can sum over S for each policy and then combine them.
            history_aggregated = torch.sum(torch.stack([torch.sum(agg, dim=0) for agg in agg_densities]), dim=0)
            
            # Aggregate current distribution from, say, the first policy.
            # For a 2D distribution with shape (S x A), sum over S to get a 1D vector of length A.
            current_aggregated = torch.sum(distributions[0], dim=0)
            
            # Compute the union mask.
            union_mask = combined_mask(current_aggregated, history_aggregated, self.batch_size)
            
            # Apply the union mask to all inputs.
            # Assuming emissions is defined over actions (shape: (A, d)):
            emissions = emissions[union_mask]
            
            # Mask each current distribution (for a 2D case, assume shape (S, A)):
            distributions = [d[:, union_mask] for d in distributions]
            
            # Also mask each agg_density accordingly.
            agg_densities = [agg[:, union_mask] for agg in agg_densities]
        
        # Now both current and historical inputs have the same action dimension.
        new_z = super()._calculate_z(emissions, distributions, episodes, mask=union_mask)
        agg_z = super()._calculate_z(emissions, agg_densities, episodes, mask=union_mask)
        
        alpha = len(visitations_per_policy[0]) / episodes
        z = ((1.0 / episodes) * new_z + alpha * agg_z) if self.uniform_alpha else ((1 - alpha) * new_z + alpha * agg_z)
        
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        if self.V is None:
            return torch.trace(torch.linalg.inv(z + self.lambd/(self.horizon*episodes) * eye))
        else:
            return torch.trace(self.V @ torch.linalg.inv(z + self.lambd/(self.horizon*episodes) * eye))
    
    def eval_full(self, emissions, distributions, episodes):
        # Final evaluation uses the full (unmasked) distributions.
        z = super()._calculate_z(emissions, distributions, episodes)
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        if self.V is None:
            return torch.trace(torch.linalg.inv(z + self.lambd/(self.horizon*episodes) * eye))
        else:
            return torch.trace(self.V @ torch.linalg.inv(z + self.lambd/(self.horizon*episodes) * eye))

def combined_mask(current_aggregated: torch.Tensor,
                  history_aggregated: torch.Tensor,
                  additional: int) -> torch.Tensor:
    """
    current_aggregated: 1D tensor of length A computed from the current distribution.
    history_aggregated: 1D tensor of length A computed from visitation history.
    additional: number of additional (zero in current) indices to sample.
    
    Returns a sorted tensor containing the union of:
      - All indices where history_aggregated is nonzero, and
      - Exactly `additional` indices sampled stochastically from indices where current_aggregated is zero.
    """
    # Compute the stochastic mask from the current distribution.
    current_mask = compute_mask(current_aggregated, additional)
    
    # Compute the history mask: all indices with nonzero visitation.
    history_mask = (history_aggregated != 0).nonzero(as_tuple=True)[0]
    
    # Union the two masks.
    combined = torch.cat([current_mask, history_mask])
    combined = torch.unique(combined)  # remove duplicates
    combined, _ = torch.sort(combined)
    return combined

class StochasticAdaptiveOrigDesignD(StochasticMultiPolicyRewardFunctionalMixin, MultiPolicyOrigDesignD):
    def __init__(self, env, lambd=1e-3, dim=0, uniform_alpha=False, batch_size=500):
        super().__init__(env, lambd, dim, batch_size=batch_size)
        self.type = "adaptive"
        self.uniform_alpha = uniform_alpha

    def eval(self, emissions, distributions, visitations_per_policy, episodes, should_mask=True):
        # Compute aggregated densities from each policy's visitation history.
        agg_densities = [
            self.build_density_from_trajectories(visitations)
            for visitations in visitations_per_policy
        ]
        
        # For stationary distributions: if the aggregated density has more dimensions,
        # convert it to (S x A) format (e.g., via diagonal extraction).
        for i in range(len(distributions)):
            if len(distributions[i].shape) < len(agg_densities[i].shape):
                agg_densities[i] = agg_densities[i].diagonal(dim1=0, dim2=1).T

        union_mask = None
        
        if should_mask:
            # Combine all visitation histories. For example, if each agg_density is (S x A),
            # sum over states (S) to get an action-level statistic.
            history_aggregated = torch.sum(
                torch.stack([torch.sum(agg, dim=0) for agg in agg_densities]), dim=0
            )
            
            # Aggregate current distribution from the first policy (assuming (S x A)).
            current_aggregated = torch.sum(distributions[0], dim=0)
            
            # Compute the union mask (requires a helper function `combined_mask`).
            union_mask = combined_mask(current_aggregated, history_aggregated, self.batch_size)
            
            # Apply the union mask to emissions, current distributions, and aggregated densities.
            emissions = emissions[union_mask]
            distributions = [d[:, union_mask] for d in distributions]
            agg_densities = [agg[:, union_mask] for agg in agg_densities]

        # Calculate the two information matrices:
        new_z = super()._calculate_z(emissions, distributions, episodes, mask=union_mask)
        agg_z = super()._calculate_z(emissions, agg_densities, episodes, mask=union_mask)

        # Determine the weighting factor alpha.
        alpha = len(visitations_per_policy[0]) / episodes
        if self.uniform_alpha:
            z = (1.0 / episodes) * new_z + alpha * agg_z
        else:
            z = (1 - alpha) * new_z + alpha * agg_z

        # Add ridge regularization.
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        regularized_z = z + self.lambd / (self.horizon*episodes) * eye

        # Return the log-determinant (second element of slogdet output).
        return torch.linalg.slogdet(regularized_z)[1]

    def eval_full(self, emissions, distributions, episodes):
        # For final evaluation, use the full (unmasked) current distributions.
        z = super()._calculate_z(emissions, distributions, episodes)
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        regularized_z = z + self.lambd / (self.horizon*episodes) * eye
        return torch.linalg.slogdet(regularized_z)[1]
