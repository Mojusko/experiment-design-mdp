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
                prior_mean: Union[torch.Tensor,None] = None) -> None:     
        
        super().__init__(env, objective, estimator, theta_star, sigma, sigma_fn, video, wort_case, markovian, update_mean, prior_mean)

    def step_episode(self):
        pass

    def step_update(self):
        pass 

    def episode_update(self):
         if self.markovian:
            action_list = self.action_trajectory
            print('actions taken: ', action_list)

            for action in action_list:
                # obtain the value of the action
                state = self.env.action_space_pre_embedding[action].reshape(1, -1)
                
                # obtain the noise
                eps = np.random.normal(0, self.sigma)

                if callable(self.theta_star):
                    fun_value = self.theta_star(state) + eps - self.prior_mean[action]
                else:
                    z = self.estimator.embed(state)
                    fun_value = z @ self.theta_star + eps - self.prior_mean[action]
                
                fun_value = fun_value.reshape(-1, 1) # .reshape(-1) #.item()

                self.estimator.add_data_point(state, fun_value)
                
                if self.update_mean:
                    self.objective.best_obs = max(self.objective.best_obs, (fun_value + self.prior_mean[action]).item())
            
            self.estimator.fit()
            self.objective.ucbs = self.estimator.ucb(self.action_space).reshape(-1) + self.prior_mean.reshape(-1)
            self.objective.lcbs = self.estimator.lcb(self.action_space).reshape(-1) + self.prior_mean.reshape(-1)
            # update the mean if required
            if self.update_mean:
                means, stds = self.estimator.mean_std(self.action_space)
                self.objective.means = means.reshape(-1) + self.prior_mean.reshape(-1)
                self.objective.stds = stds.reshape(-1)

            # calculate the best arm guess
            mean_estimates = self.estimator.mean(self.action_space).reshape(-1) + self.prior_mean.reshape(-1)
            self.best_arm.append(torch.argmax(mean_estimates).item())       
        else:
            pass