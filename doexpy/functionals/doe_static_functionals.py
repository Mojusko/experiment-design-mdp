import autograd.numpy as np
#import autograd.numpy.linalg as la
import torch
from typing import List, Union
from abc import ABC, abstractmethod
from doexpy.env.discrete_env import Environment
from doexpy.functionals.reward_functional import RewardFunctional
import torch.linalg as la
import logging
import torch.nn.functional as F # Added for cosine_similarity
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

        # If distribution is 2D (e.g., S x A), add a singleton horizon dimension
        # to make it compatible with the 3D summation logic (H x S x A).
        if distribution.ndim == 2:
            distribution = distribution.unsqueeze(0)

        if self.dim == 0: # Sum over actions and horizon to get marginal state distribution d(s)
            # Input distribution is (H,S,A). Sum over A (dim 2), then H (dim 0) -> (S,)
            distribution = torch.sum(torch.sum(distribution, dim = 2), dim = 0)
        
        elif self.dim == 1: # Sum over states and horizon to get marginal action distribution d(a)
            # Input distribution is (H,S,A). Sum over S (dim 1), then H (dim 0) -> (A,)
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
        """
        Update the C vector based on the estimator's theta_fit.
        """
        if not hasattr(estimator, 'theta_fit') or estimator.theta_fit is None:
            logger.warning(f"Estimator {type(estimator).__name__} has no 'theta_fit' or it's None. Cannot update C in {type(self).__name__}.")
            return

        new_C_candidate = estimator.theta_fit.detach().clone()

        # Normalize the new C candidate
        norm = torch.linalg.norm(new_C_candidate)
        if norm > 1e-9:  # Avoid division by zero
            normalized_new_C = (new_C_candidate / norm)
        else:
            logger.error(f"New C candidate (estimator.theta_fit) for {type(self).__name__} has near-zero norm. C will not be updated.")
            return

        # Ensure normalized_new_C is a column vector (d, 1)
        if normalized_new_C.dim() == 1:
            normalized_new_C = normalized_new_C.unsqueeze(1) # Make it (d,1)
        elif normalized_new_C.dim() == 2 and normalized_new_C.shape[0] == 1: # if (1,d)
            normalized_new_C = normalized_new_C.T # Make it (d,1)
        elif not (normalized_new_C.dim() == 2 and normalized_new_C.shape[1] == 1): # if not (d,1)
            logger.warning(f"New C for {type(self).__name__} after normalization is not a column vector (d,1), shape is {normalized_new_C.shape}. Using as is.")
            
        self.C = normalized_new_C  # Update self.C
        logger.debug(f"C vector in {type(self).__name__} updated from estimator. New C shape: {self.C.shape}")

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

class MultiPolicyOrigDesignC(MultiPolicyOrigDesignD):
    def __init__(self, env, lambd, dim=0, C=None, **kwargs):
        """
        Initialize the MultiPolicyOrigDesignC class.

        Parameters:
        - env: The environment object.
        - lambd (float): The regularization parameter lambda.
        - Sigma (float): The Sigma parameter.
        - C (torch.Tensor, list, or None): The C parameter, which can be a tensor, list of tensors, or None.
        - **kwargs: Additional keyword arguments passed to the parent class (time_weigh is ignored).
        """
        # Call the parent class's __init__ to set up common attributes
        # Note: time_weigh is no longer accepted by the parent __init__
        super().__init__(env, lambd, dim, **kwargs) 
        # Set the C attribute specific to this class
        self.C = C

    def _compute_c_optimal_value(self, inv_z_reg):
        """
        Compute C-optimal design value using the inverse regularized z matrix.
        If self.C is None, computes A-optimal design value instead.
        
        Parameters:
        - inv_z_reg (torch.Tensor): The inverse of the regularized z matrix.
        
        Returns:
        - float: The C-optimal value (trace or max trace), or A-optimal value if C is None.
        """
        target_device = inv_z_reg.device  # Get the device of inv_z_reg

        if self.C is None:
            logger.debug(f"{type(self).__name__}._compute_c_optimal_value: C is None, using A-optimal criterion.")
            return -torch.trace(inv_z_reg)

        # Handle C being a single tensor by converting it to a list containing its transpose,
        # or process C if it's already a list.
        if not isinstance(self.C, list):
            # self.C is a single tensor. update_estimator makes it (d,1).
            # Transpose to (1,d) to match the list item expectation for c M_inv c.T formula.
            if not (self.C.dim() == 2 and self.C.shape[1] == 1):
                # This case should ideally not happen if C is set by update_estimator
                # or initialized as a proper column vector.
                logger.warning(f"Single self.C is not a column vector (d,1), shape is {self.C.shape}. Attempting transpose anyway.")
            C_list_to_process = [self.C.T]
            was_single_c = True
        else:
            # self.C is already a list of tensors.
            C_list_to_process = self.C
            was_single_c = False

        processed_traces = []
        for C_item_from_list in C_list_to_process:
            # Each item in C_list_to_process should be a row vector (1,d)
            # for the formula: trace(inv(c @ M_inv @ c.T))
            if not (C_item_from_list.dim() == 2 and C_item_from_list.shape[0] == 1 and C_item_from_list.shape[1] == inv_z_reg.shape[0]):
                error_msg = (f"Item in C list (or transformed single C) is expected to be a row vector (1,d) "
                             f"matching inv_z_reg dim {inv_z_reg.shape[0]}, but got shape {C_item_from_list.shape}.")
                if was_single_c: # Add info about original self.C if it was a single tensor
                    error_msg += f" Original self.C shape was {self.C.shape}."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            C_item_dev = C_item_from_list.to(target_device)
            
            # scalar_variance_term = C_item_dev @ inv_z_reg @ C_item_dev.T
            # This term is (1,d) @ (d,d) @ (d,1) -> (1,1)
            # It represents the variance in the direction of C_item_dev.
            # We want to maximize 1/variance (precision).
            # torch.linalg.inv of a (1,1) tensor is its reciprocal.
            precision_val = torch.linalg.inv(C_item_dev @ inv_z_reg @ C_item_dev.T)
            processed_traces.append(torch.trace(precision_val)) # trace of (1,1) is the element itself

        if was_single_c:
            # If originally a single C, return its computed precision value directly.
            return processed_traces[0]
        else:
            # If originally a list of C vectors, return the sum of log-precisions.
            # This matches the previous behavior for a list of C.
            return torch.log(torch.stack(processed_traces)).sum()

    def eval(self, emissions, distributions, episodes):
        """
        Evaluate the design using the emissions, distributions, and number of episodes.

        Parameters:
        - emissions (torch.Tensor): The emissions tensor.
        - distributions (torch.Tensor): The distributions tensor.
        - episodes (int): The number of episodes.

        Returns:
        - float: The evaluation result (trace or max trace).
        """
        # Compute z using the inherited _calculate_z method
        z = self._calculate_z(emissions, distributions, episodes)
        
        # Create an identity matrix matching z's shape and device
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        
        # Apply horizon*T regularization: z + (lambda / horizon * episodes)
        z_reg = z + (self.lambd / (self.horizon * episodes)) * eye
        
        # Compute the inverse of the regularized z
        inv_z_reg = torch.linalg.inv(z_reg)

        # Calculate the original C-optimal value (log-product of traces)
        c_optimal_design = self._compute_c_optimal_value(inv_z_reg)

        # Return the weighted sum
        return c_optimal_design

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


