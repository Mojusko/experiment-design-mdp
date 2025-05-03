import autograd.numpy as np
#import autograd.numpy.linalg as la
import torch
from typing import List, Union
from abc import ABC, abstractmethod
from doexpy.env.discrete_env import Environment
from doexpy.functionals.reward_functional import RewardFunctional
import torch.linalg as la
import logging
from stpy.regression.regularized_dictionary.regularized_multinomial_estimator import RegularizedMultinomialEstimator

logger = logging.getLogger(__name__)

class ExperimentDesignFunctional(RewardFunctional):

    def __init__(self, dim = 0):
        super().__init__()
        self.dim = dim


    def _prepare(self,
            emissions: torch.Tensor,
            distribution: torch.Tensor,
            episodes: int = 0,
            Sigma: Union[None,float] = None
        )->torch.Tensor:

        # If Sigma not specified just divide by 1. 
        if Sigma is None:
            Sigma = 1.

        # Move distribution to same device as emissions
        distribution = distribution.to(emissions.device)

        if self.dim == 0:
            distribution = torch.sum(torch.sum(distribution, dim = 2), dim = 0)
        
        elif self.dim == 1: # actions matter 
            distribution = torch.sum(torch.sum(distribution, dim = 1), dim = 0)
        z = torch.einsum('ij,j,jk->ik', emissions.T, distribution/ Sigma**2, emissions)
        return z

class DesignA(ExperimentDesignFunctional):
    def __init__(self, env: Environment, lambd: float = 1e-3, dim=0, V=None):
        super().__init__(dim=dim)
        self.lambd = lambd
        self.type = "static"
        self.env = env
        self.V = V.to(env.device) if V is not None else None

    def eval(self, emissions: torch.Tensor, distribution: torch.Tensor, episodes: int = 0) -> float:
        z = self._prepare(emissions, distribution, episodes)
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        if self.V is None:
            return -torch.trace(la.inv(z + self.lambd/episodes * eye))
        else:
            return -torch.trace(self.V @ la.inv(z + self.lambd/episodes * eye))

    def eval_full(self, emissions: torch.Tensor, distribution: torch.Tensor, episodes: int) -> float:
        z = self._prepare(emissions, distribution, episodes)
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        if self.V is None:
            return -torch.trace(la.inv(z + self.lambd/episodes * eye))
        else:
            return -torch.trace(self.V @ la.inv(z + self.lambd/episodes * eye))

class DesignD(DesignA):
    def __init__(self, env: Environment, lambd: float = 1e-3, dim=0, V=None):
        super().__init__(env=env, lambd=lambd, dim=dim, V=V)

    def eval(self,
             emissions: torch.Tensor,
             distribution: torch.Tensor,
             episodes: int) -> float:
        z = self._prepare(emissions, distribution, episodes)
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        return torch.linalg.slogdet(z + self.lambd/episodes * eye)[1]

    def eval_full(self,
             emissions: torch.Tensor,
             distribution: torch.Tensor,
             episodes: int) -> float:
        return self.eval(emissions, distribution, episodes)

class DesignE(DesignD):
    def eval(self,
             emissions: torch.Tensor,
             distribution: torch.Tensor,
             episodes: int
             ) -> float:
        z = self._prepare(emissions, distribution, Sigma = self.Sigma)
        return torch.linalg.eigvalsh(z + self.lambd / episodes)[0]

    def eval_full(self,
                  emissions: torch.Tensor,
                  distribution: torch.Tensor,
                  episodes: int,
                  ) -> float:
        z = self._prepare(emissions, distribution, Sigma = self.Sigma_true)
        return torch.linalg.eigvalsh(z + self.lambd / episodes)[0]
    
class DesignC(ExperimentDesignFunctional):

    def __init__(self,
                 env: Environment,
                 lambd: float = 1e-3,
                 Sigma: float = 1.,
                 C: Union[torch.Tensor, List, None] = None):
        super().__init__()

        self.env = env
        self.dim = self.env.get_dim()
        self.lambd = lambd * np.eye(self.dim)
        self.Sigma = Sigma

        self.C = C
        self.type = "static"

    def eval(self,
             emissions: torch.Tensor,
             distribution: torch.Tensor,
             episodes: int = 0) -> float:
        z = self._prepare(emissions, distribution, Sigma = self.Sigma)
        if isinstance(self.C, list):
            return torch.max([torch.trace(la.inv(C @ la.inv(z + self.lambd / episodes) @ C.T)) for C in self.C])
        else:
            return torch.trace(la.inv(self.C @ la.inv(z + self.lambd / episodes) @ self.C.T))

    def eval_full(self,
                  emissions: torch.Tensor,
                  distribution: torch.Tensor,
                  episodes: int = 0) -> float:
        return self.eval(emissions, distribution, episodes)

class MultiPolicyAggDesignA(ExperimentDesignFunctional):
    def __init__(self,
                 env: Environment,
                 lambd: float = 1e-3,
                 dim = 0,
                 V = None,
                 time_weigh = True):
        super().__init__(dim=dim)
        self.lambd = lambd 
        self.type = "static"
        self.env = env
        self.V = V
        self.time_weigh = time_weigh


    def _calculate_z(self,
                     emissions: torch.Tensor,
                     distributions: List[torch.Tensor],
                     episodes: int = 0,
                     Sigma: Union[None, float] = None) -> torch.Tensor:
        if Sigma is None:
            Sigma = 1.
            
        # Ensure distributions are on the same device as emissions.
        distributions = [d.to(emissions.device) for d in distributions]
        # <-- New: If distributions are 2D, add a horizon dimension (like in OrigDesign)
        if distributions[0].ndim == 2:
            distributions = [d.unsqueeze(0) for d in distributions]
    
        # Initialize z using the feature dimension from emissions.
        z = torch.zeros((emissions.shape[1], emissions.shape[1]),
                        dtype=distributions[0].dtype, device=emissions.device)
        emissions = emissions.type(distributions[0].dtype)
        H = distributions[0].shape[0]
    
        for h in range(H):
            time_weight = (H - h) / H if self.time_weigh else 1.0
            if self.dim == 0:
                # Sum over the first non-horizon dimension.
                d_h_sum = sum(torch.sum(d[h], dim=1) for d in distributions) / Sigma**2
            elif self.dim == 1:
                d_h_sum = sum(torch.sum(d[h], dim=0) for d in distributions) / Sigma**2
    
            # Compute the two terms via Einstein summation.
            z_diag = torch.einsum('ij,j,jk->ik', emissions.T, d_h_sum, emissions)
            z_outer = 0.5 * torch.einsum('ij,j,k,kl->il', emissions.T, d_h_sum, d_h_sum, emissions)
            z += time_weight * (z_diag - z_outer)
    
        return z
    def eval(self,
         emissions: torch.Tensor,
         distributions: List[torch.Tensor],
         episodes: int = 0) -> float:
        z = self._calculate_z(emissions, distributions, episodes)
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        if self.V is None:
            return -torch.trace(la.inv(z + self.lambd/episodes * eye))
        else:
            return -torch.trace(self.V @ la.inv(z + self.lambd/episodes * eye))

    def eval_full(self,
              emissions: torch.Tensor,
              distributions: List[torch.Tensor],
              episodes: int) -> float:
        return self.eval(emissions, distributions, episodes)

class MultiPolicyAggDesignD(MultiPolicyAggDesignA):
    def eval(self,
             emissions: torch.Tensor,
             distributions: List[torch.Tensor],
             episodes: int = 0) -> float:
        z = self._calculate_z(emissions, distributions, episodes)
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        return torch.linalg.slogdet(z + self.lambd/episodes * eye)[1]

class MultiPolicyOrigDesignD(RewardFunctional):
    def __init__(self, env, lambd=1e-3, dim=0, V=None):
        super().__init__()
        self.lambd = lambd
        self.type = "static"
        self.env = env
        self.V = V
        self.horizon = env.max_episode_length # Keep horizon for regularization scaling
        self.estimator = None
        self.dim = dim

    def update_estimator(self, estimator, emissions):
        """Update the estimator and set C to be the parameter vector."""
        self.estimator = estimator
        theta_fit = estimator.theta_fit  # Direct access to theta_fit
        
        # Ensure theta_fit is properly shaped for C-optimal calculations
        # For C-optimal design, we typically need a row vector (1xN)
        # If theta_fit is a column vector (Nx1), reshape it to a row vector
        if theta_fit.dim() == 2 and theta_fit.shape[1] == 1:
            self.C = theta_fit  # Keep as column vector, we'll transpose when needed
        else:
            self.C = theta_fit

    # def _compute_diagonal_terms(self, emissions, prob_matrix, d1_h, d2_h):
    #     """Compute diagonal terms of the Fisher."""
    #     d2_h_or_unif = torch.ones_like(d2_h) / d2_h.shape[0] if d2_h.sum() == 0 else d2_h
    #     d1_h_or_unif = torch.ones_like(d1_h) / d1_h.shape[0] if d1_h.sum() == 0 else d1_h
    #     p_q1 = torch.mm(prob_matrix, d2_h_or_unif.view(-1,1))
    #     p_q2 = torch.mm(prob_matrix, d1_h_or_unif.view(-1,1))
    #
    #     term1 = emissions.T @ torch.diag(p_q1.squeeze()) @ torch.diag(d1_h) @ emissions
    #     term2 = emissions.T @ torch.diag(p_q2.squeeze()) @ torch.diag(d2_h) @ emissions
    #
    #     return term1 + term2
    #
    # def _compute_cross_terms(self, emissions, prob_matrix, d1_h, d2_h):
    #     # Since prob_matrix is always 0.5, prob * (1 - prob) is always 0.25
    #     probs = 0.25
    #     d1d2 = d1_h.unsqueeze(1) @ d2_h.unsqueeze(0)  # [n_states, n_states]
    #     d2d1 = d2_h.unsqueeze(1) @ d1_h.unsqueeze(0)  # [n_states, n_states]
    #
    #     term1 = emissions.T @ (probs * d1d2) @ emissions
    #     term2 = emissions.T @ (probs * d2d1) @ emissions
    #
    #     return term1 + term2

    def _calculate_z(self, emissions, distributions, episodes):
        distributions = [d.to(emissions.device) for d in distributions]
        emissions = emissions.type(distributions[0].dtype) # Shape: (n_elements, d_features) where n_elements depends on self.dim
        feature_dim = emissions.shape[1]
        z = torch.zeros((feature_dim, feature_dim),
                        dtype=distributions[0].dtype, device=emissions.device)

        if distributions[0].ndim == 2:
            # If input is 2D (S, A), add a singleton horizon dimension -> (1, S, A)
            distributions = [dist.unsqueeze(0) for dist in distributions]

        H = distributions[0].shape[0]
        assert H == 1, f"This functional only supports stationary policies (horizon H=1). Got H={H}."
        K = len(distributions) # Number of policies

        # Since H=1, we only consider the first time step (index 0)
        h = 0
        d_h = [] # List to store marginal distributions for each policy at step h
        for q in range(K):
            if self.dim == 0: # Sum over actions -> marginal state distribution d(s)
                # distributions[q][h] has shape (S, A)
                # emissions should have shape (S, d_features) - assuming state features
                d_h_q = torch.sum(distributions[q][h], dim=1) # Shape (S,)
            elif self.dim == 1: # Sum over states -> marginal action distribution d(a)
                # distributions[q][h] has shape (S, A)
                # emissions should have shape (A, d_features) - assuming action features
                d_h_q = torch.sum(distributions[q][h], dim=0) # Shape (A,)
            else:
                raise ValueError(f"Unsupported dim value: {self.dim}. Must be 0 or 1.")
            d_h.append(d_h_q)

        # Check if emissions shape matches the marginal distribution dimension
        if emissions.shape[0] != d_h[0].shape[0]:
             raise ValueError(f"Dimension mismatch: emissions first dimension ({emissions.shape[0]}) "
                              f"does not match the marginal distribution dimension ({d_h[0].shape[0]}) "
                              f"based on self.dim={self.dim}.")

        # Calculate the approximate Fisher Information Matrix I_approx
        # I_approx = (1/K^2) * [ (K-1) * sum_q (E_q[phi phi^T]) - sum_{q!=q'} (E_q[phi])(E_{q'}[phi^T]) ]

        # Term 1: (K-1) * sum_q E_q[phi phi^T]
        # E_q[phi phi^T] = sum_s d_h_q(s) phi(s) phi(s)^T = emissions.T @ diag(d_h_q) @ emissions
        term1 = torch.zeros_like(z)
        for q in range(K):
            # Ensure d_h[q] is treated as weights for the diagonal
            term1 += emissions.T @ torch.diag(d_h[q]) @ emissions

        term1 *= K

        # Term 2: sum_{q!=q'} (E_q[phi])(E_{q'}[phi^T])
        # E_q[phi] = sum_s d_h_q(s) phi(s) = emissions.T @ d_h_q
        # E_{q'}[phi^T] = sum_{s'} d_h_{q'}(s') phi(s')^T = d_h_{q'}.T @ emissions
        term2 = torch.zeros_like(z)
        expected_phis = [] # Store E_q[phi] for each q
        for q in range(K):
            # d_h[q] has shape (N,), emissions has shape (N, d)
            # emissions.T @ d_h[q] gives shape (d,)
            expected_phis.append(emissions.T @ d_h[q]) # Shape (d,)

        for q in range(K):
            for q_prime in range(K):
                    # expected_phis[q] is (d,), expected_phis[q_prime] is (d,)
                    # We need outer product: (d,) x (d,) -> (d, d)
                    term2 += torch.outer(expected_phis[q], expected_phis[q_prime])

        # Combine terms and scale
        z = (term1 - term2) / (K**2)

        return z

    def eval(self, emissions, distributions, episodes):
        z = self._calculate_z(emissions, distributions, episodes)
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        if self.V is None:
            return torch.linalg.slogdet(z + self.lambd/(self.horizon*episodes) * eye)[1]
        else:
            return torch.linalg.slogdet((self.V @ z) + self.lambd/(self.horizon*episodes) * eye)[1]

    def eval_full(self, emissions, distributions, episodes):
        return self.eval(emissions, distributions, episodes)

class MultiPolicyOrigDesignA(MultiPolicyOrigDesignD):
    def eval(self, emissions, distributions, episodes):

        z = self._calculate_z(emissions, distributions, episodes)
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        if self.V is None:
            return -torch.trace(la.inv(z + self.lambd/(self.horizon*episodes) * eye))
        else:
            return -torch.trace(self.V @ la.inv(z + self.lambd/(self.horizon*episodes) * eye))

    def eval_full(self, emissions, distributions, episodes):
        return self.eval(emissions, distributions, episodes)

class MultiPolicyOrigDesignC(MultiPolicyOrigDesignD): # Inherits _calculate_z from D
    def __init__(self, env, lambd, dim=0, C=None, **kwargs):
        """
        Initialize the MultiPolicyOrigDesignC class using Subspace D-optimality.

        Parameters:
        - env: The environment object.
        - lambd (float): The regularization parameter lambda.
        - dim (int): Dimension for marginalization (0 for states, 1 for actions).
        - C (List[torch.Tensor]): List of C vectors (e.g., embedded keywords).
          Each tensor should be (1, d). The list will be stacked into (k, d).
        - **kwargs: Additional keyword arguments (V is ignored for this objective).
        """
        # Call the parent class's __init__ but V is not used here.
        super().__init__(env=env, lambd=lambd, dim=dim, V=None) # Pass V=None

        if C is None or not isinstance(C, list) or not C:
            raise ValueError("C must be a non-empty list of torch.Tensors for MultiPolicyOrigDesignC.")

        # Normalize each vector before stacking (recommended)
        # Move to device during normalization
        try:
            C_normalized = [(c.to(env.device) / torch.linalg.norm(c)).to(env.device) for c in C]
            self.C_stack = torch.cat(C_normalized, dim=0) # Shape (k, d)
            logger.info(f"Initialized {type(self).__name__} with {len(C)} C vectors stacked into shape {self.C_stack.shape}.")
        except Exception as e:
            logger.error(f"Error during C vector normalization/stacking: {e}")
            # Check shapes if possible
            for i, c_vec in enumerate(C):
                if not isinstance(c_vec, torch.Tensor) or c_vec.ndim != 2 or c_vec.shape[0] != 1:
                     logger.error(f"C vector at index {i} has unexpected shape/type: {type(c_vec)}, shape={c_vec.shape if isinstance(c_vec, torch.Tensor) else 'N/A'}. Expected (1, d).")
            raise ValueError("Failed to process C vectors. Check logs for details.") from e


        # Store k (number of C vectors)
        self.num_c_vectors = self.C_stack.shape[0]


    def update_estimator(self, estimator, emissions):
        """
        Update the estimator and set C based on the estimator's parameters.
        
        NOTE: This method is currently commented out. The intention is to use
        the C vector(s) provided during initialization (a priori) and not
        update them based on the fitted estimator during the experiment run.
        If adaptive C-optimality based on the estimator is desired, this
        method needs to be uncommented and potentially revised.

        Parameters:
        - estimator: A RegularizedMultinomialEstimator with theta_fit property
        - emissions: The emissions tensor
        """
        # # Log update intention
        # logger.info(f"Updating C vector in {type(self).__name__}.")
        #
        # # Call parent's update_estimator method (if applicable, though not strictly needed here as we override C logic)
        # super().update_estimator(estimator, emissions)
        #
        # # Get the fitted parameter vector
        # theta_fit = estimator.theta_fit
        #
        # # Normalize the parameter vector before using it as C
        # norm = torch.linalg.norm(theta_fit)
        # if norm > 1e-9: # Avoid division by zero or near-zero
        #     # Assume theta_fit is a 1D vector or (d, 1) or (1, d). Normalize and reshape to (1, d).
        #     new_C = (theta_fit / norm).view(1, -1)
        #     new_c_norm_l2 = torch.linalg.norm(new_C).item() # Should be ~1.0
        #     new_c_norm_l1 = torch.linalg.norm(new_C, ord=1).item()
        #     logger.info(f"New C vector set from estimator {type(estimator).__name__} (reshaped to {new_C.shape}): L2 norm={new_c_norm_l2:.4f}, L1 norm={new_c_norm_l1:.4f}")
        #     self.C = new_C
        # else:
        #     # Handle zero vector case - raise error as C cannot be None or zero for C-optimality trace calculation
        #     logger.error("Estimator theta_fit has near-zero norm. Cannot compute C-optimal design. Raising ValueError.")
        #     raise ValueError("Estimator theta_fit has near-zero norm, cannot set C for C-optimal design.")
        pass # Method is disabled - C vectors are fixed

    # Inherits _calculate_z from MultiPolicyOrigDesignD

    def eval(self, emissions, distributions, episodes):
        """
        Evaluate the design using the emissions, distributions, and number of episodes.

        Parameters:
        - emissions (torch.Tensor): The emissions tensor.
        - distributions (torch.Tensor): The distributions tensor.
        - episodes (int): The number of episodes.

        Evaluate the design using Subspace D-optimality: logdet(C @ Z_reg @ C.T).

        Parameters:
        - emissions (torch.Tensor): The emissions tensor.
        - distributions (List[torch.Tensor]): List of distribution tensors for each policy.
        - episodes (int): The number of episodes.

        Returns:
        - float: The log-determinant objective value.
        """
        # Compute the approximate Fisher Information Matrix Z
        # Z shape: (d, d) where d is feature_dim
        z = self._calculate_z(emissions, distributions, episodes)
        feature_dim = z.shape[0]

        # Ensure C_stack's dimension matches feature dimension
        if self.C_stack.shape[1] != feature_dim:
            raise ValueError(f"Dimension mismatch: C_stack second dimension ({self.C_stack.shape[1]}) "
                             f"does not match feature dimension ({feature_dim}).")

        # Create identity matrix for regularization
        eye_d = torch.eye(feature_dim, device=z.device, dtype=z.dtype)

        # Apply regularization to Z: Z_reg = Z + (lambda / (horizon * episodes)) * I_d
        # Use self.horizon inherited from MultiPolicyOrigDesignD
        # Ensure episodes is not zero to avoid division by zero
        if episodes <= 0:
             logger.warning("Episodes <= 0 in eval. Setting regularization to a large value.")
             regularization = self.lambd * 1e10 # Effectively infinite regularization
        else:
            regularization = self.lambd / (self.horizon * episodes)
        z_reg = z + regularization * eye_d

        # Project Z_reg onto the subspace defined by C_stack
        # C_stack shape: (k, d)
        # Z_reg shape: (d, d)
        # C_stack.T shape: (d, k)
        # projected_fisher shape: (k, k)
        projected_fisher = self.C_stack @ z_reg @ self.C_stack.T

        # Optional: Add a small regularization to the projected matrix itself
        # This helps if C_stack @ C_stack.T is ill-conditioned (e.g., keywords are very similar)
        # eye_k = torch.eye(self.num_c_vectors, device=projected_fisher.device, dtype=projected_fisher.dtype)
        # projected_fisher_reg = projected_fisher + 1e-6 * eye_k # Add small diagonal jitter

        # Compute the objective: log determinant of the projected Fisher matrix
        # Use slogdet for numerical stability: returns (sign, logabsdet)
        sign, logabsdet = torch.linalg.slogdet(projected_fisher) # Use projected_fisher_reg if adding jitter

        # We expect the matrix to be positive semi-definite, so sign should be +1
        # If sign is not +1 or logabsdet is -inf, it might indicate numerical issues or
        # insufficient exploration/regularization.
        if sign <= 0 or torch.isinf(logabsdet):
             logger.warning(f"slogdet returned sign={sign} or logabsdet={logabsdet}. "
                            f"Matrix might be ill-conditioned. Regularization: {regularization:.2e}. "
                            f"Returning large negative value.")
             # Return a large negative value to avoid selecting this design
             # Ensure it's a tensor on the correct device
             return torch.tensor(-1e20, device=projected_fisher.device, dtype=projected_fisher.dtype)

        # Return the log-determinant tensor directly for autograd
        return logabsdet


    def eval_full(self, emissions, distributions, episodes):
        """
        Full evaluation method, which delegates to eval.

        Parameters:
        - emissions (torch.Tensor): The emissions tensor.
        - distributions (torch.Tensor): The distributions tensor.
        - episodes (int): The number of episodes.

        Returns:
        - float: The evaluation result.
        """
        return self.eval(emissions, distributions, episodes)


