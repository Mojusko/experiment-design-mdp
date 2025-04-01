import torch.linalg as la
import torch
import cvxpy as cp 
from typing import List, Union, Callable
from abc import ABC, abstractmethod
from doexpy.env.discrete_env import Environment
from doexpy.functionals.reward_functional import RewardFunctional
import logging
from doexpy.functionals.doe_static_functionals import MultiPolicyOrigDesignD, MultiPolicyOrigDesignA, MultiPolicyOrigDesignC

logger = logging.getLogger(__name__)

# ============================================================================
# Base Adaptive Design Classes
# ============================================================================

class AdaptiveDesignD(RewardFunctional):
    """
    Adaptive D-optimal design for experiment design.
    """
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
    """
    Adaptive C-optimal design for experiment design.
    """
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
    """
    Adaptive D-optimal design with heteroscedastic noise.
    """
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
    """
    Adaptive A-optimal design for experiment design.
    """
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

# ============================================================================
# Adaptive Original Design Classes
# ============================================================================

class AdaptiveOrigDesignD(MultiPolicyOrigDesignD):
    """
    Adaptive D-optimal design for original design functionals.
    """
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

        # Process agg_densities: Average over time if they are HxSxA and distributions are SxA
        processed_agg_densities = []
        for i in range(len(distributions)):
            agg_density = agg_densities[i]
            # Check if candidate distribution is stationary (e.g., S, A) while agg_density is not (e.g., H, S, A)
            if distributions[i].ndim < agg_density.ndim and agg_density.ndim == 3:
                 # Average over the time dimension (H)
                 processed_agg_densities.append(torch.mean(agg_density, dim=0))
            elif distributions[i].ndim == agg_density.ndim:
                 # Dimensions match, use as is (e.g., both are S, A or both H, S, A)
                 processed_agg_densities.append(agg_density)
            else:
                 # Handle unexpected dimension mismatch
                 raise ValueError(f"Inconsistent dimensions between candidate distribution ({distributions[i].shape}) and aggregated density ({agg_density.shape})")

        alpha = len(visitations_per_policy[0]) / episodes

        # Calculate information matrices using the processed densities
        new_z = super()._calculate_z(emissions, distributions, episodes)
        # Use the processed (averaged) aggregate densities here
        agg_z = super()._calculate_z(emissions, processed_agg_densities, episodes)

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
    """
    Adaptive A-optimal design for original design functionals.
    """
    def __init__(self, env, lambd=1e-3, dim=0, uniform_alpha=False, V=None):
        super().__init__(env, lambd, dim)
        self.type = "adaptive"
        self.uniform_alpha = uniform_alpha
        self.V = V

    def eval(self, emissions, distributions, visitations_per_policy, episodes):
        # For training - handle adaptive weighting without any masking
        agg_densities = [
            self.build_density_from_trajectories(visitations)
            for visitations in visitations_per_policy
        ]

        # Process agg_densities: Average over time if they are HxSxA and distributions are SxA
        processed_agg_densities = []
        for i in range(len(distributions)):
            agg_density = agg_densities[i]
            # Check if candidate distribution is stationary (e.g., S, A) while agg_density is not (e.g., H, S, A)
            if distributions[i].ndim < agg_density.ndim and agg_density.ndim == 3:
                 # Average over the time dimension (H)
                 processed_agg_densities.append(torch.mean(agg_density, dim=0))
            elif distributions[i].ndim == agg_density.ndim:
                 # Dimensions match, use as is (e.g., both are S, A or both H, S, A)
                 processed_agg_densities.append(agg_density)
            else:
                 # Handle unexpected dimension mismatch
                 raise ValueError(f"Inconsistent dimensions between candidate distribution ({distributions[i].shape}) and aggregated density ({agg_density.shape})")

        alpha = len(visitations_per_policy[0]) / episodes

        # Calculate the information matrices using the processed densities
        new_z = super()._calculate_z(emissions, distributions, episodes)
        # Use the processed (averaged) aggregate densities here
        agg_z = super()._calculate_z(emissions, processed_agg_densities, episodes)

        # Combine the matrices based on alpha
        if self.uniform_alpha:
            z = (1.0 / episodes) * new_z + alpha * agg_z
        else:
            z = (1 - alpha) * new_z + alpha * agg_z
        
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        if self.V is None:
            return -torch.trace(torch.linalg.inv(z + self.lambd/(self.horizon * episodes) * eye))
        else:
            return -torch.trace(self.V @ torch.linalg.inv(z + self.lambd/(self.horizon * episodes) * eye))

    def eval_full(self, emissions, distributions, episodes):
        # For final evaluation - directly use the provided distributions
        z = super()._calculate_z(emissions, distributions, episodes)
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        if self.V is None:
            return -torch.trace(torch.linalg.inv(z + self.lambd/(self.horizon * episodes) * eye))
        else:
            return -torch.trace(self.V @ torch.linalg.inv(z + self.lambd/(self.horizon * episodes) * eye))

# ============================================================================
# Utility Functions
# ============================================================================

class AdaptiveOrigDesignC(MultiPolicyOrigDesignC):
    """
    Adaptive C-optimal design for original design functionals.
    Updates C based on estimator unless adaptive_estimation_frequency is 0.
    """
    def __init__(self, env, lambd=1e-3, dim=0, uniform_alpha=False, C=None, adaptive_estimation_frequency=1, **kwargs):
        """
        Initializes AdaptiveOrigDesignC.

        Args:
            env: Environment object.
            lambd: Regularization parameter.
            dim: Dimension parameter.
            uniform_alpha: Flag for alpha weighting.
            C: Initial C vector (e.g., prior or GT weights). If None, expects update_estimator.
            adaptive_estimation_frequency: Frequency of estimator updates. If 0, C will not be updated.
            **kwargs: Additional arguments for parent classes.
        """
        self.adaptive_estimation_frequency = adaptive_estimation_frequency
        # Initialize the parent MultiPolicyOrigDesignC first
        # We temporarily allow C=None here, but log a warning if estimation freq is > 0.
        # We handle the C=None case specifically for the adaptive scenario.
        try:
            super().__init__(env, lambd, dim, C=C, **kwargs)
        except ValueError as e:
            # If C is None and estimation frequency is > 0, it's an issue unless update_estimator is called first.
            if C is None and self.adaptive_estimation_frequency > 0:
                logger.warning("AdaptiveOrigDesignC initialized with C=None and adaptive_estimation_frequency > 0. "
                               "Expecting `update_estimator` to be called before evaluation.")
                # Call grandparent init to set up basic attributes
                super(MultiPolicyOrigDesignC, self).__init__(env, lambd, dim, **kwargs)
                self.C = None # Explicitly set C to None
            # If C is None and frequency is 0, this is an error because C won't be updated.
            elif C is None and self.adaptive_estimation_frequency == 0:
                 logger.error("AdaptiveOrigDesignC initialized with C=None and adaptive_estimation_frequency=0. "
                              "C must be provided if no updates are planned.")
                 raise ValueError("C cannot be None for AdaptiveOrigDesignC when adaptive_estimation_frequency is 0.")
            else:
                # If C was not None and still failed, re-raise the error
                raise e
        else:
             # If super().__init__ succeeded (meaning C was not None)
             if C is not None:
                 c_norm_l2 = torch.linalg.norm(C).item()
                 c_norm_l1 = torch.linalg.norm(C, ord=1).item()
                 logger.info(f"AdaptiveOrigDesignC initialized with C vector: L2 norm={c_norm_l2:.4f}, L1 norm={c_norm_l1:.4f}")
             # Log update behavior based on frequency
             if self.adaptive_estimation_frequency == 0:
                 logger.info("adaptive_estimation_frequency is 0. C vector will NOT be updated.")
             else:
                 logger.info(f"adaptive_estimation_frequency is {self.adaptive_estimation_frequency}. C vector WILL be updated.")

        self.type = "adaptive"
        self.uniform_alpha = uniform_alpha

    def update_estimator(self, estimator, emissions):
        """
        Update the C vector based on the estimator, unless adaptive_estimation_frequency is 0.
        """
        if self.adaptive_estimation_frequency == 0:
            logger.info("adaptive_estimation_frequency is 0. Skipping update_estimator for AdaptiveOrigDesignC.")
            return
        else:
            # Proceed with the normal update from the estimator via the parent method
            logger.info(f"adaptive_estimation_frequency > 0. Updating C based on estimator {type(estimator).__name__}.")
            super().update_estimator(estimator, emissions)

    def eval(self, emissions, distributions, visitations_per_policy, episodes):
        # Compute agg_densities for each policy's visitation history
        agg_densities = [
            self.build_density_from_trajectories(visitations)
            for visitations in visitations_per_policy
        ]

        # Process agg_densities: Average over time if they are HxSxA and distributions are SxA
        processed_agg_densities = []
        for i in range(len(distributions)):
            agg_density = agg_densities[i]
            # Check if candidate distribution is stationary (e.g., S, A) while agg_density is not (e.g., H, S, A)
            if distributions[i].ndim < agg_density.ndim and agg_density.ndim == 3:
                 # Average over the time dimension (H)
                 processed_agg_densities.append(torch.mean(agg_density, dim=0))
            elif distributions[i].ndim == agg_density.ndim:
                 # Dimensions match, use as is (e.g., both are S, A or both H, S, A)
                 processed_agg_densities.append(agg_density)
            else:
                 # Handle unexpected dimension mismatch
                 raise ValueError(f"Inconsistent dimensions between candidate distribution ({distributions[i].shape}) and aggregated density ({agg_density.shape})")

        alpha = len(visitations_per_policy[0]) / episodes

        # Calculate information matrices using the processed densities
        new_z = super()._calculate_z(emissions, distributions, episodes)
        # Use the processed (averaged) aggregate densities here
        agg_z = super()._calculate_z(emissions, processed_agg_densities, episodes)

        # Weight combination based on alpha
        if self.uniform_alpha:
            z = (1.0 / episodes) * new_z + alpha * agg_z
        else:
            z = (1 - alpha) * new_z + alpha * agg_z

        # Create identity matrix for regularization
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        z_reg = z + self.lambd/(self.horizon*episodes) * eye
        
        # Compute inverse of regularized z
        inv_z_reg = torch.linalg.inv(z_reg)
        
        # Use parent class method to compute C-optimal value
        return super()._compute_c_optimal_value(inv_z_reg)

    def eval_full(self, emissions, distributions, episodes):
        # For final evaluation - directly use the provided distributions
        z = super()._calculate_z(emissions, distributions, episodes)
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        z_reg = z + self.lambd/(self.horizon*episodes) * eye
        inv_z_reg = torch.linalg.inv(z_reg)
        
        # Use parent class method to compute C-optimal value
        return super()._compute_c_optimal_value(inv_z_reg)


