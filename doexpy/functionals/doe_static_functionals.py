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

        if self.dim == 0:
            distribution = torch.sum(torch.sum(distribution, dim = 2), dim = 0)
        
        elif self.dim == 1: # actions matter 
            distribution = torch.sum(torch.sum(distribution, dim = 1), dim = 0)
        z = torch.einsum('ij,j,jk->ik', emissions.T, distribution/ Sigma**2, emissions)
        return z

class DesignA(ExperimentDesignFunctional):

    def __init__(self,
                 env: Environment,
                 lambd: float = 1e-3,
                 dim = 0,
                 V = None):
        super().__init__(dim = dim)
        self.lambd = lambd
        self.type = "static"
        self.env = env
        self.V = V

    def eval(self,
             emissions: torch.Tensor,
             distribution: torch.Tensor,
             episodes: int = 0
             ) -> float:
        z = self._prepare(emissions, distribution, episodes)
        if self.V is None:
            return -torch.trace(la.inv(z + self.lambd/episodes * torch.eye(z.shape[0])))
        else:
            return -torch.trace(self.V@la.inv(z + self.lambd/episodes * torch.eye(z.shape[0])))

    def eval_full(self,
                  emissions: torch.Tensor,
                  distribution: torch.Tensor,
                  episodes: int,
                  ) -> float:
        z = self._prepare(emissions, distribution, episodes)
        if self.V is None:
            return -torch.trace(la.inv(z + self.lambd/episodes * torch.eye(z.shape[0])))
        else:
            return -torch.trace(self.V@la.inv(z + self.lambd/episodes * torch.eye(z.shape[0])))

class DesignD(ExperimentDesignFunctional):
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
        self.lambd = lambd * torch.eye(self.dim)

        if isinstance(lambd, float):
            self.Sigma = sigma * torch.ones(self.env.get_states_num(), dtype=torch.float64)
            self.Sigma_true = self.Sigma
        else:
            self.Sigma = sigma
            self.Sigma_true = self.Sigma

    def eval(self,
             emissions: torch.Tensor,
             distribution: torch.Tensor,
             episodes: int
             ) -> float:
        z = self._prepare(emissions, distribution, Sigma = self.Sigma)
        return torch.linalg.slogdet(z + self.lambd / episodes)[1]

    def eval_full(self,
                  emissions: torch.Tensor,
                  distribution: torch.Tensor,
                  episodes: int,
                  ) -> float:
        z = self._prepare(emissions, distribution, Sigma = self.Sigma_true)
        return torch.linalg.slogdet(z + self.lambd / episodes)[1]


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
                 V = None):
        super().__init__(dim=dim)
        self.lambd = lambd 
        self.type = "static"
        self.env = env
        self.V = V

    def _calculate_z(self,
                    emissions: torch.Tensor,
                    distributions: List[torch.Tensor],
                    episodes: int = 0,
                    Sigma: Union[None, float] = None) -> torch.Tensor:
        if Sigma is None:
            Sigma = 1.
    
        z = torch.zeros((emissions.shape[1], emissions.shape[1]))
        emissions = emissions.type(distributions[0].dtype)
        
        # For each horizon step
        for h in range(distributions[0].shape[0]):
            if self.dim == 0:
                d_h_sum = sum(torch.sum(d[h], dim=1) for d in distributions)/Sigma**2  # Sum over states
            elif self.dim == 1:
                d_h_sum = sum(torch.sum(d[h], dim=0) for d in distributions)/Sigma**2  # Sum over actions
            
            z_diag = torch.einsum('ij,j,jk->ik', emissions.T, d_h_sum, emissions)
            z_outer = 0.5 * torch.einsum('ij,j,k,kl->il', emissions.T, d_h_sum, d_h_sum, emissions)
            z += z_diag - z_outer
    
        return z

    def eval(self,
         emissions: torch.Tensor,
         distributions: List[torch.Tensor],
         episodes: int = 0) -> float:
        z = self._calculate_z(emissions, distributions, episodes)
        if self.V is None:
            return -torch.trace(la.inv(z + self.lambd/episodes * torch.eye(z.shape[0])))
        else:
            return -torch.trace(self.V@la.inv(z + self.lambd/episodes * torch.eye(z.shape[0])))

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
        return torch.linalg.slogdet(z + self.lambd/episodes * torch.eye(z.shape[0]))[1]

class MultiPolicyOrigDesignD(ExperimentDesignFunctional):
    def __init__(self,
                 env: Environment,
                 lambd: float = 1e-3,
                 dim = 0,
                 V = None):
        super().__init__(dim=dim)
        self.lambd = lambd 
        self.type = "static"
        self.env = env
        self.V = V

    def _calculate_z(self,
                    emissions: torch.Tensor,
                    distributions: List[torch.Tensor],
                    episodes: int = 0,
                    Sigma: Union[None, float] = None) -> torch.Tensor:
        if Sigma is None:
            Sigma = 1.
            
        z = torch.zeros((emissions.shape[1], emissions.shape[1]), dtype=distributions[0].dtype)
        emissions = emissions.type(distributions[0].dtype)
        
        for h in range(distributions[0].shape[0]):
            if self.dim == 0:
                d_h_sum = sum(torch.sum(d[h], dim=1) for d in distributions)/Sigma**2
            elif self.dim == 1:
                d_h_sum = sum(torch.sum(d[h], dim=0) for d in distributions)/Sigma**2
            
            z_diag = torch.einsum('ij,j,jk->ik', emissions.T, d_h_sum, emissions)
            z += z_diag
            
            for d1 in distributions:
                for d2 in distributions:
                    if self.dim == 0:
                        d1_h = torch.sum(d1[h], dim=1)/Sigma**2
                        d2_h = torch.sum(d2[h], dim=1)/Sigma**2
                    elif self.dim == 1:
                        d1_h = torch.sum(d1[h], dim=0)/Sigma**2
                        d2_h = torch.sum(d2[h], dim=0)/Sigma**2
                    
                    z -= torch.einsum('ij,j,k,kl->il', emissions.T, d1_h, d2_h, emissions)
        
        return z

    def eval(self,
             emissions: torch.Tensor,
             distributions: List[torch.Tensor],
             episodes: int = 0) -> float:
        z = self._calculate_z(emissions, distributions, episodes)
        return torch.linalg.slogdet(z + self.lambd/episodes * torch.eye(z.shape[0]))[1]

    def eval_full(self,
                  emissions: torch.Tensor,
                  distributions: List[torch.Tensor],
                  episodes: int) -> float:
        return self.eval(emissions, distributions, episodes)
