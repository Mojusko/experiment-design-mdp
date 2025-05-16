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

    def update_estimator(self, estimator, emissions, scorer_model_gt_weight=None):
        """
        Update the C vector based on the estimator's theta_fit.
        Logs cosine similarity of old and new C against scorer_model_gt_weight if provided.
        """
        logger.info(f"Attempting to update C vector in {type(self).__name__} based on estimator {type(estimator).__name__}.")

        if not hasattr(estimator, 'theta_fit') or estimator.theta_fit is None:
            logger.warning("Estimator has no 'theta_fit' or it's None. Cannot update C.")
            return

        new_C_candidate = estimator.theta_fit.detach().clone()

        # Normalize the new C candidate
        norm = torch.linalg.norm(new_C_candidate)
        if norm > 1e-9:  # Avoid division by zero
            normalized_new_C = (new_C_candidate / norm)
        else:
            logger.error("New C candidate (estimator.theta_fit) has near-zero norm. Cannot use for C-optimal design. C will not be updated.")
            return

        # Ensure normalized_new_C is a column vector (d, 1)
        # This assumes C is a single vector. If C can be a list, this part needs adjustment.
        if normalized_new_C.dim() == 1:
            normalized_new_C = normalized_new_C.unsqueeze(1) # Make it (d,1)
        elif normalized_new_C.dim() == 2 and normalized_new_C.shape[0] == 1: # if (1,d)
            normalized_new_C = normalized_new_C.T # Make it (d,1)
        # If it's already (d,1), it's fine. If (d, k) where k!=1, it might be an issue for single C.

        if scorer_model_gt_weight is not None:
            gt_weight_flat = scorer_model_gt_weight.detach().clone().to(normalized_new_C.device).flatten()

            if hasattr(self, 'C') and self.C is not None:
                # Assuming self.C is a tensor that can be flattened for comparison
                old_C_flat = self.C.detach().clone().to(gt_weight_flat.device).flatten()
                if old_C_flat.shape == gt_weight_flat.shape and old_C_flat.numel() > 0 :
                    cos_sim_old = F.cosine_similarity(old_C_flat, gt_weight_flat, dim=0)
                    logger.info(f"  Old C vs GT weight: Cosine Similarity = {cos_sim_old.item():.4f}")
                else:
                    logger.warning(f"  Could not compare Old C (shape {old_C_flat.shape}, numel {old_C_flat.numel()}) with GT weight (shape {gt_weight_flat.shape}, numel {gt_weight_flat.numel()}). Old C might be a list or incompatible.")

            new_C_flat = normalized_new_C.detach().clone().to(gt_weight_flat.device).flatten()
            if new_C_flat.shape == gt_weight_flat.shape and new_C_flat.numel() > 0:
                cos_sim_new = F.cosine_similarity(new_C_flat, gt_weight_flat, dim=0)
                logger.info(f"  New C vs GT weight: Cosine Similarity = {cos_sim_new.item():.4f}")
            else:
                 logger.warning(f"  Could not compare New C (shape {new_C_flat.shape}) with GT weight (shape {gt_weight_flat.shape}).")

        self.C = normalized_new_C  # Update self.C
        logger.info(f"C vector in {type(self).__name__} updated. New C shape: {self.C.shape}, L2 norm: {torch.linalg.norm(self.C).item():.4f}")

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
        if C is None:
            raise ValueError("C cannot be None for MultiPolicyOrigDesignC. It must be provided or set via update_estimator.")
        self.C = C

    def _compute_c_optimal_value(self, inv_z_reg):
        """
        Compute C-optimal design value using the inverse regularized z matrix.
        
        Parameters:
        - inv_z_reg (torch.Tensor): The inverse of the regularized z matrix.
        
        Returns:
        - float: The C-optimal value (trace or max trace).
        """
        target_device = inv_z_reg.device  # Get the device of inv_z_reg

        if self.C is None:
             # This case should ideally not be reached due to checks in __init__ and update_estimator
             raise ValueError("C is None during C-optimal value computation. This should not happen.")

        # Handle C being a list of vectors
        if isinstance(self.C, list):
            # Assume each C_item is a (1, d) tensor
            # Move each C_item to the target device before computation
            traces = []
            for i, C_item in enumerate(self.C):
                C_item_dev = C_item.to(target_device)
                # Calculate precision in direction C_item
                traces.append(torch.trace(torch.linalg.inv(C_item_dev @ inv_z_reg @ C_item_dev.T)))
            # Return the mean of the precisions instead of the max
            #return torch.mean(torch.stack(traces)) # Original: Mean
            return torch.log(torch.stack(traces)).sum() # New: Log-Product (Sum of Logs)

        # Handle C being a single vector
        # Assume self.C is a (1, d) tensor
        C_dev = self.C.to(target_device) # Move self.C to the target device
        return torch.trace(torch.linalg.inv(C_dev @ inv_z_reg @ C_dev.T))

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


