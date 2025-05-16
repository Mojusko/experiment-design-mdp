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
    def __init__(self, env, lambd=1e-3, dim=0, uniform_alpha=False, V=None, **kwargs):
        super().__init__(env, lambd, dim, **kwargs)
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
    def __init__(self, env, lambd=1e-3, dim=0, uniform_alpha=False, C=None, adaptive_estimation_frequency=1, update_C_from_estimator: bool = True, **kwargs):
        """
        Initializes AdaptiveOrigDesignC.

        Args:
            env: Environment object.
            lambd: Regularization parameter.
            dim: Dimension parameter.
            uniform_alpha: Flag for alpha weighting.
            C: Initial C vector (e.g., prior or GT weights). If None, expects update_estimator.
            adaptive_estimation_frequency: Frequency of estimator updates. If 0, C will not be updated.
            update_C_from_estimator: If True (default), C will be updated from the estimator
                                     when adaptive_estimation_frequency > 0.
            **kwargs: Additional arguments for parent classes.
        """
        self.adaptive_estimation_frequency = adaptive_estimation_frequency
        self.update_C_from_estimator = update_C_from_estimator
        # Initialize the parent MultiPolicyOrigDesignC first
        super().__init__(env, lambd, dim, C=C, **kwargs) # This will set self.C

        self.adaptive_estimation_frequency = adaptive_estimation_frequency
        self.update_C_from_estimator = update_C_from_estimator
        
        # Log behavior if C is None at initialization
        if self.C is None:
            if self.adaptive_estimation_frequency > 0 and self.update_C_from_estimator:
                logger.warning(f"{type(self).__name__} initialized with C=None. "
                               "Evaluation will use A-optimal design until `update_estimator` provides a C vector.")
            elif self.adaptive_estimation_frequency == 0 and not self.update_C_from_estimator : # C is None, will not be updated by estimator
                 logger.warning(f"{type(self).__name__} initialized with C=None, adaptive_estimation_frequency=0 and update_C_from_estimator=False. "
                               "Evaluation will permanently use A-optimal design unless C is set externally.")
            elif self.adaptive_estimation_frequency == 0 and self.update_C_from_estimator : # C is None, will not be updated by estimator as freq is 0
                 logger.warning(f"{type(self).__name__} initialized with C=None and adaptive_estimation_frequency=0. "
                               "Evaluation will permanently use A-optimal design unless C is set externally, as estimator updates are off.")
            elif not self.update_C_from_estimator: # C is None, freq > 0, but update_C_from_estimator is False
                 logger.warning(f"{type(self).__name__} initialized with C=None and update_C_from_estimator=False. "
                                "Evaluation will permanently use A-optimal design unless C is set externally, as C will not be updated from estimator.")


        self.type = "adaptive"
        self.uniform_alpha = uniform_alpha

    def update_estimator(self, estimator, emissions):
        """
        Update the C vector based on the estimator, if adaptive_estimation_frequency > 0
        and update_C_from_estimator is True.
        """
        if self.adaptive_estimation_frequency == 0:
            logger.info(f"adaptive_estimation_frequency is 0 for {type(self).__name__}. Skipping C update from estimator.")
            return

        if not self.update_C_from_estimator:
            logger.info(f"update_C_from_estimator is False for {type(self).__name__}. "
                        f"Skipping C update from estimator, even though adaptive_estimation_frequency is {self.adaptive_estimation_frequency}.")
            return
        
        # If we reach here, adaptive_estimation_frequency > 0 AND update_C_from_estimator is True
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
        
        if self.C is None:
            logger.debug(f"{type(self).__name__}.eval: C is None, using A-optimal criterion.")
            return -torch.trace(torch.linalg.inv(z_reg))

        # Compute inverse of regularized z
        inv_z_reg = torch.linalg.inv(z_reg)
        
        # Use parent class method to compute C-optimal value
        return super()._compute_c_optimal_value(inv_z_reg)

    def eval_full(self, emissions, distributions, episodes):
        # For final evaluation - directly use the provided distributions
        z = super()._calculate_z(emissions, distributions, episodes) # Calls MultiPolicyOrigDesignC._calculate_z
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        z_reg = z + self.lambd/(self.horizon*episodes) * eye

        if self.C is None:
            logger.debug(f"{type(self).__name__}.eval_full: C is None, using A-optimal criterion.")
            return -torch.trace(torch.linalg.inv(z_reg))
        
        inv_z_reg = torch.linalg.inv(z_reg)
        
        # Use parent class method to compute C-optimal value
        return super()._compute_c_optimal_value(inv_z_reg)


