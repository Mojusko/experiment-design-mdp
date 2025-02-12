import autograd.numpy as np
#import autograd.numpy.linalg as la
import torch
from typing import List, Union
from abc import ABC, abstractmethod
from doexpy.env.discrete_env import Environment
from doexpy.functionals.reward_functional import RewardFunctional
import torch.linalg as la 

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
            
        distributions = [d.to(emissions.device) for d in distributions]
        z = torch.zeros((emissions.shape[1], emissions.shape[1]), dtype=distributions[0].dtype, device=emissions.device)
        emissions = emissions.type(distributions[0].dtype)
        H = distributions[0].shape[0]

        for h in range(H):
            time_weight = (H - h)/H if self.time_weigh else 1.0
            if self.dim == 0:
                d_h_sum = sum(torch.sum(d[h], dim=1) for d in distributions)/Sigma**2
            elif self.dim == 1:
                d_h_sum = sum(torch.sum(d[h], dim=0) for d in distributions)/Sigma**2
            
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
    def __init__(self, env, lambd=1e-3, dim=0, V=None, time_weigh=True):
        super().__init__()
        self.lambd = lambd
        self.type = "static"
        self.env = env
        self.V = V
        self.time_weigh = time_weigh
        self.estimator = None
        self.prob_matrix = None
        self.dim = dim

    def update_estimator(self, estimator, emissions):
        """Update the estimator and recompute probability matrix."""
        self.estimator = estimator
        self._update_probability_matrix(emissions)

    def _update_probability_matrix(self, emissions):
        """Update pairwise probability matrix based on emissions."""
        logits = self.estimator.mean(emissions)
        logits = logits.to(emissions.device)
        exp_logits = torch.exp(logits)
        exp_logits_i = exp_logits.view(-1, 1)
        exp_logits_j = exp_logits.view(1, -1)
        denominators = exp_logits_i + exp_logits_j
        
        self.prob_matrix = exp_logits_i / denominators

    def _get_prob_matrix(self, emissions):
        """Return pairwise probability matrix or default to 0.5 on the specified device and dtype."""
        n = emissions.shape[0]
        if self.prob_matrix is None:
            return 0.5 * torch.ones((n, n), device=emissions.device, dtype=emissions.dtype)
        return self.prob_matrix

    def _compute_diagonal_terms(self, emissions, prob_matrix, d1_h, d2_h):
        """Compute diagonal terms of the Fisher."""
        d2_h_or_unif = torch.ones_like(d2_h) / d2_h.shape[0] if d2_h.sum() == 0 else d2_h
        d1_h_or_unif = torch.ones_like(d1_h) / d1_h.shape[0] if d1_h.sum() == 0 else d1_h
        p_q1 = torch.mm(prob_matrix, d2_h_or_unif.view(-1,1))
        p_q2 = torch.mm(prob_matrix, d1_h_or_unif.view(-1,1))

        term1 = emissions.T @ torch.diag(p_q1.squeeze()) @ torch.diag(d1_h) @ emissions       
        term2 = emissions.T @ torch.diag(p_q2.squeeze()) @ torch.diag(d2_h) @ emissions       
        
        return term1 + term2

    def _compute_cross_terms(self, emissions, prob_matrix, d1_h, d2_h):
        probs = prob_matrix * (1 - prob_matrix)
        d1d2 = d1_h.unsqueeze(1) @ d2_h.unsqueeze(0)  # [n_states, n_states]
        d2d1 = d2_h.unsqueeze(1) @ d1_h.unsqueeze(0)  # [n_states, n_states]
        
        term1 = emissions.T @ (probs * d1d2) @ emissions
        term2 = emissions.T @ (probs * d2d1) @ emissions

        return term1 + term2

    def _calculate_z(self, emissions, distributions, episodes):
        distributions = [d.to(emissions.device) for d in distributions]
        emissions = emissions.type(distributions[0].dtype)
        z = torch.zeros((emissions.shape[1], emissions.shape[1]), 
                       dtype=distributions[0].dtype, device=emissions.device)
        if len(distributions[0].shape) == 2:
            distributions = [dist[None,:] for dist in distributions]
        H = distributions[0].shape[0]
        
        for h in range(H):
            time_weight = (H - h)/H if self.time_weigh else 1.0
            if self.dim == 0:
                d1_h = torch.sum(distributions[0][h], dim=1)
                d2_h = torch.sum(distributions[1][h], dim=1)
            elif self.dim == 1:
                d1_h = torch.sum(distributions[0][h], dim=0)  
                d2_h = torch.sum(distributions[1][h], dim=0)
                
            prob_matrix = self._get_prob_matrix(emissions)
            prob_matrix = prob_matrix.type(d1_h.dtype)
            
            diag_terms = self._compute_diagonal_terms(emissions, prob_matrix, d1_h, d2_h)
            cross_terms = self._compute_cross_terms(emissions, prob_matrix, d1_h, d2_h)
            z += time_weight * (diag_terms - cross_terms)
            
        return z

    def eval(self, emissions, distributions, episodes):
        z = self._calculate_z(emissions, distributions, episodes)
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        if self.V is None:
            return torch.linalg.slogdet(z + self.lambd/episodes * eye)[1]
        else:
            return torch.linalg.slogdet((self.V @ z) + self.lambd/episodes * eye)[1]

    def eval_full(self, emissions, distributions, episodes):
        return self.eval(emissions, distributions, episodes)


class MultiPolicyOrigDesignA(MultiPolicyOrigDesignD):
    def eval(self, emissions, distributions, episodes):
        z = self._calculate_z(emissions, distributions, episodes)
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        if self.V is None:
            return -torch.trace(la.inv(z + self.lambd/episodes * eye))
        else:
            return -torch.trace(self.V @ la.inv(z + self.lambd/episodes * eye))

    def eval_full(self, emissions, distributions, episodes):
        return self.eval(emissions, distributions, episodes)


class MultiPolicyOrigDesignE(MultiPolicyOrigDesignD):
    def eval(self, emissions, distributions, episodes):
        z = self._calculate_z(emissions, distributions, episodes)
        eye = torch.eye(z.shape[0], device=z.device, dtype=z.dtype)
        return torch.linalg.eigvalsh(z + self.lambd/episodes * eye)[0]

    def eval_full(self, emissions, distributions, episodes):
        return self.eval(emissions, distributions, episodes)


def compute_mask(aggregated: torch.Tensor, additional: int) -> torch.Tensor:
    """
    Given a 1D tensor `aggregated` (e.g. aggregated action weights),
    returns a sorted tensor of indices that includes:
      - all indices where the value is nonzero, and
      - exactly `additional` indices randomly sampled among the zero entries (if available).
      
    If there are fewer than `additional` zero indices, all of them are included.
    """
    nonzero_idx = (aggregated != 0).nonzero(as_tuple=True)[0]
    zero_idx = (aggregated == 0).nonzero(as_tuple=True)[0]
    
    if additional > 0 and len(zero_idx) > 0:
        if additional >= len(zero_idx):
            sampled_zero_idx = zero_idx
        else:
            perm = torch.randperm(len(zero_idx))
            sampled_zero_idx = zero_idx[perm[:additional]]
        mask = torch.cat([nonzero_idx, sampled_zero_idx])
    else:
        mask = nonzero_idx

    mask, _ = torch.sort(mask)
    return mask


class StochasticMultiPolicyRewardFunctionalMixin:
    def __init__(self, *args, batch_size: int = None, **kwargs):
        """
        batch_size: Number of additional (zero) action indices to include alongside all nonzero indices.
                    If None, no masking is performed.
        """
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size



