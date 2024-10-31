
import numpy as np 
import torch 
from scipy.stats import norm
from doexpy.functionals.reward_functional import RewardFunctional


class DesignBestArmLinearBanditEIDummy(RewardFunctional):

    def __init__(self, env, init_ucb = torch.inf, prior_mean = None):

        super().__init__()

        self.env = env
        action_space_size = env.actions_num

        self.ucbs = torch.ones(action_space_size) * init_ucb
        self.lcbs = -1 * torch.ones(action_space_size) * init_ucb
        self.stds = torch.ones(action_space_size)
        self.best_obs = -1 * init_ucb

        if prior_mean is None:
            self.means = torch.zeros(action_space_size)
        else:
            self.means = prior_mean

        self.type = "adaptive"

    def eval(self, emissions, distribution, unrolls, episodes):
        return 0

    def eval_full(self,
                  emissions: torch.Tensor,
                  distribution: torch.Tensor,
                  episodes: int = 0) -> float:
        
        return 0

    def gradient(self, emissions: torch.Tensor, distribution: torch.Tensor, unrolls, episodes):
        """
        Calculate the Expected Improvement at each emission point.

        The formula is given by:

        EI(x) = (mu(x) - f(x)) * Phi(mu(x) - f(x)) + sigma(x) * phi(mu(x) - f(x))

        """
        # obtain the distribution shape before summing
        H, S, A = distribution.shape
        
        EI = (self.means - self.best_obs) * norm.cdf((self.means - self.best_obs) / self.stds) + self.stds * norm.pdf((self.means - self.best_obs) / self.stds)

        EI = EI.reshape(1, S, 1)

        EI = np.repeat(EI, H, axis = 0)
        EI = np.repeat(EI, A, axis = 2)

        return EI

class GreedyEIDummy(RewardFunctional):

    def __init__(self, env, init_ucb = torch.inf, prior_mean = None):

        super().__init__()

        self.env = env
        action_space_size = env.actions_num

        self.ucbs = torch.ones(action_space_size) * init_ucb
        self.lcbs = -1 * torch.ones(action_space_size) * init_ucb
        self.stds = torch.ones(action_space_size)
        self.best_obs = -1 * init_ucb

        if prior_mean is None:
            self.means = torch.zeros(action_space_size)
        else:
            self.means = prior_mean

        self.type = "adaptive"

    def eval(self, emissions, distribution, unrolls, episodes):
        return 0

    def eval_full(self,
                  emissions: torch.Tensor,
                  distribution: torch.Tensor,
                  episodes: int = 0) -> float:
        
        return 0

    def gradient(self, emissions: torch.Tensor, distribution: torch.Tensor, unrolls, episodes):
        """
        Calculate the Expected Improvement at each emission point.

        The formula is given by:

        EI(x) = (mu(x) - f(x)) * Phi(mu(x) - f(x)) + sigma(x) * phi(mu(x) - f(x))

        """
        # obtain the distribution shape before summing
        H, S, A = distribution.shape
        
        # EI = (self.means - self.best_obs) * norm.cdf((self.means - self.best_obs) / self.stds) + self.stds * norm.pdf((self.means - self.best_obs) / self.stds)

        # create a copy of ucbs
        EI = self.ucbs.clone()

        EI = EI.reshape(1, S, 1)

        # repeat the EI to match the distribution shape using torch functionality
        EI = EI.repeat(H, 1, A)

        # for every future step make the reward zero
        EI[2:, :, :] = 0

        return EI
