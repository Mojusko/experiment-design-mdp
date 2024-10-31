from doexpy.feedback.bandit_feedback import BanditFeedback
from doexpy.env.discrete_env import Environment
from typing import Union, Callable

from doexpy.solvers.lp import LP
from doexpy.solvers.dp import DP
from doexpy.convex_solvers.frank_wolfe import FrankWolfe
from doexpy.mdpexplore import MdpExplore
from doexpy.env.bandits import Bandits, Bandits_Left_Right, ConstrainedMaxMovement, MovementConstrainedBayesianOptimization
from doexpy.functionals.bandit_functionals import DesignRewardBandit, DesignBestArmLinearBandit, DesignBestArmLinearBanditNoDenominator
from doexpy.functionals.doe_adaptive_functionals import AdaptiveDesignD
from doexpy.functionals.reward_functional import RewardFunctional
from doexpy.policies.summary_policies.density_policy import DensityPolicy, MarginalDensityPolicy
from doexpy.policies.summary_policies.mixture_policy import MixturePolicy

from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.continuous_processes.kernelized_features import KernelizedFeatures

import numpy as np 
import torch 

class BanditFeedbackState(BanditFeedback):
    """
    Bandit feedback class for state-based feedback. Different to classical feedback it gives feedback given 
    the state of the environment; not its emissions (embeddings).
    """
    def __init__(self,
                env:Environment, 
                objective:RewardFunctional, 
                estimator:Union[GaussianProcess, KernelizedFeatures],
                theta_star:Union[torch.Tensor, Callable], 
                sigma:float,
                sigma_fn: Union[Callable,Callable] = None,
                video:bool = False,
                wort_case:bool = True,
                markovian:bool = False,    
                update_mean: bool = False,
                prior_mean: Union[torch.Tensor,None] = None,
                aggregation: bool = False) -> None:      # sums all the past feedbacks
        
        super().__init__(env, objective, estimator, theta_star, sigma, sigma_fn, video, wort_case, markovian, update_mean, prior_mean)
        self.aggregation = aggregation

    def step_update(self):
        if self.markovian:
            action_list = self.action_trajectory
            if self.aggregation:
                # TODO: needs to be finished and made compatible with the rest of the code
                # sum all the past feedbacks
                actions = action_list
            else:
                # only the last feedback
                actions = [action_list[-1]]
            
            print ("Giving feedback on", actions)

            eps = torch.randn(size = (len(actions),1))*self.sigma

            if callable(self.theta_star):
                     fun_value = self.theta_star(torch.Tensor(actions)).int().view(-1,1) + eps
            else:
                raise ValueError("Non-callable theta_star needs to be callable")
            print ("Feedback given:", fun_value)

            for i,a in enumerate(actions):
                self.estimator.add_data_point(torch.Tensor([a]).int().view(-1,1), fun_value[i].view(-1,1))
            
            
            #print('actions taken: ', action_list)


    def episode_update(self):
        if self.markovian:
            self.estimator.fit()
            print ("Updating the model.")
        else:
            raise NotImplementedError("Non-Markovian feedback not implemented yet for this feedback model")