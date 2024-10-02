from typing import Union, Callable
from enum import Enum
import random
import numpy as np
import torch
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.continuous_processes.kernelized_features import KernelizedFeatures
from stpy.kernels import KernelFunction
from stpy.embeddings.embedding import HermiteEmbedding
from stpy.continuous_processes.nystrom_fea import NystromFeatures
import argparse
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp

from mdpexplore.solvers.lp import LP
from mdpexplore.solvers.dp import DP
from mdpexplore.convex_solvers.frank_wolfe import FrankWolfe
from mdpexplore.mdpexplore import MdpExplore
from mdpexplore.env.bandits import Bandits, Bandits_Left_Right, ConstrainedMaxMovement, MovementConstrainedBayesianOptimization
from mdpexplore.functionals.bandit_functionals import DesignRewardBandit, DesignBestArmLinearBandit, DesignBestArmLinearBanditNoDenominator
from mdpexplore.functionals.doe_adaptive_functionals import AdaptiveDesignD
from mdpexplore.functionals.reward_functional import RewardFunctional
from mdpexplore.policies.summary_policies.density_policy import DensityPolicy, MarginalDensityPolicy
from mdpexplore.policies.summary_policies.mixture_policy import MixturePolicy

from scipy.optimize import minimize
from scipy.stats import norm

from mdpexplore.feedback.feedback_base import SimpleFeedback
import torch

class BanditFeedback(SimpleFeedback):
    def __init__(self, 
                env:MovementConstrainedBayesianOptimization, 
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
        super().__init__(env, objective)
        self.estimator = estimator
        self.theta_star = theta_star
        self.sigma = sigma
        self.sigma_fn = sigma_fn
        self.embedded_action_space = env.action_space
        self.action_space = env.action_space_pre_embedding
        self.video = video
        self.worst_case = wort_case
        self.markovian = markovian
        self.update_mean = update_mean
        # keep track of best arm guess
        self.best_arm = []
        # define the prior mean

        if prior_mean is None:
            self.prior_mean = torch.zeros(size = (self.action_space.size()[0], 1), dtype = torch.float64)
        else:
            self.prior_mean = prior_mean
    
    def step_update(self):
        if self.markovian:
            action = self.action_trajectory[-1]
            print('actions taken: ', action)

            if self.video:
                # save the relevant stuff for plotting
                if self.estimator.fitted:
                    lcb = self.estimator.lcb(self.action_space).squeeze()
                    ucb = self.estimator.ucb(self.action_space).squeeze()
                    mean = self.estimator.mean(self.action_space).squeeze()
                else:
                    lcb = self.objective.lcbs
                    ucb = self.objective.ucbs
                    mean = np.zeros_like(lcb)

                # load the arrays from memory
                actions = np.load('actions.npy')
                lcbs = np.load('lcb.npy')
                means = np.load('mean.npy')
                ucbs = np.load('ucb.npy')

                # append the new values
                actions = np.vstack((actions, np.array(action)))
                lcbs = np.vstack((lcbs, lcb))
                means = np.vstack((means, mean))
                ucbs = np.vstack((ucbs, ucb))

                # save them in memory
                np.save('actions.npy', actions)
                np.save('lcb.npy', lcbs)
                np.save('mean.npy', means)
                np.save('ucb.npy', ucbs)
        else:
            action = self.action_trajectory[-1]
            state = self.state_trajectory[-1]
            print('actions taken: ', action, 'in state:', state)

            # state_coord = self.env.convert_to_grid(state)
            # action_coord = self.env.convert_to_grid(action)

            # obtain the value of the action
            state_x = self.env.action_space_pre_embedding[action].reshape(1, -1)
            print ('corresponding x:', state_x)
            # obtain the noise
            if self.sigma_fn is None:
                eps = np.random.normal(0, self.sigma)
                Sigma = self.sigma
            else:
                state_coord = self.env.convert_to_grid(state)
                action_coord = self.env.convert_to_grid(action)
                Sigma = self.sigma_fn(state_coord,action_coord)
                eps = np.random.normal(0, Sigma)



            if callable(self.theta_star):
                fun_value = self.theta_star(state_x) + eps - self.prior_mean[action]
            else:
                z = self.estimator.embed(state_x)
                fun_value = z @ self.theta_star + eps - self.prior_mean[action]

            fun_value = fun_value#.item()
            

            print ('y:', fun_value)
            if not self.worst_case:
                if self.sigma_fn is not None:
                    print ("constrained sigma:", Sigma)
                    self.estimator.add_data_point(state_x, fun_value, Sigma = Sigma.view(1,1))
                else:
                    self.estimator.add_data_point(state_x, fun_value)
            else:
                if self.sigma_fn is not None:
                    print ("worst-case:", self.sigma)
                    self.estimator.add_data_point(state_x, fun_value, Sigma = torch.Tensor([self.sigma]).double().view(1,1))
                else:
                    self.estimator.add_data_point(state_x, fun_value)

            self.estimator.fit()
            self.objective.ucbs = self.estimator.ucb(self.action_space) + self.prior_mean
            self.objective.lcbs = self.estimator.lcb(self.action_space) + self.prior_mean
            # update the mean if required
            if self.update_mean:
                means, stds = self.estimator.mean_std(self.action_space).reshape(-1)
                self.objective.means = means.reshape(-1) + self.prior_mean.reshape(-1)
                self.objective.stds = stds.reshape(-1)
                self.objective.best_obs = torch.maximum(self.objective.best_obs, fun_value + self.prior_mean[action])

            # Compute the differences to get the UCBs and LCBs for the objective denominator, no longer used
            # calculate the best arm guess
            mean_estimates = self.estimator.mean(self.action_space).reshape(-1) + self.prior_mean.reshape(-1)
            self.best_arm.append(torch.argmax(mean_estimates))

            if self.video:
                # save the relevant stuff for plotting
                lcb = self.estimator.lcb(self.action_space).reshape(-1)
                ucb = self.estimator.ucb(self.action_space).reshape(-1)
                mean = self.estimator.mean(self.action_space).reshape(-1)

                # load the arrays from memory
                actions = np.load('actions.npy')
                lcbs = np.load('lcb.npy')
                means = np.load('mean.npy')
                ucbs = np.load('ucb.npy')

                # append the new values
                actions = np.vstack((actions, np.array(action)))
                lcbs = np.vstack((lcbs, lcb))
                means = np.vstack((means, mean))
                ucbs = np.vstack((ucbs, ucb))

                # save them in memory
                np.save('actions.npy', actions)
                np.save('lcb.npy', lcbs)
                np.save('mean.npy', means)
                np.save('ucb.npy', ucbs)

        

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
                
                fun_value = fun_value.item()

                self.estimator.add_data_point(state, fun_value)
                
                if self.update_mean:
                    self.objective.best_obs = torch.maximum(self.objective.best_obs, fun_value + self.prior_mean[action])
            
            self.estimator.fit()
            self.objective.ucbs = self.estimator.ucb(self.action_space).reshape(-1) + self.prior_mean.reshape(-1)
            self.objective.lcbs = self.estimator.lcb(self.action_space).reshape(-1) + self.prior_mean.reshape(-1)
            # update the mean if required
            if self.update_mean:
                means, stds = self.estimator.mean_std(self.action_space).reshape(-1)
                self.objective.means = means.reshape(-1) + self.prior_mean.reshape(-1)
                self.objective.stds = stds.reshape(-1)

            # calculate the best arm guess
            mean_estimates = self.estimator.mean(self.action_space).reshape(-1) + self.prior_mean.reshape(-1)
            self.best_arm.append(torch.argmax(mean_estimates))       
        else:
            pass
