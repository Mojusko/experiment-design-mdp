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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from mdpexplore.feedback.feedback_base import SimpleFeedback

class BanditFeedback(SimpleFeedback):
    def __init__(self, 
                env:MovementConstrainedBayesianOptimization, 
                objective:RewardFunctional, 
                estimator:Union[GaussianProcess, KernelizedFeatures], 
                theta_star:Union[np.array, Callable], 
                sigma:float, video:bool = False, 
                markovian:bool = False,
                update_mean: bool = False,
                prior_mean: np.array = None) -> None:
        
        super().__init__(env, objective)
        self.estimator = estimator
        self.theta_star = theta_star
        self.sigma = sigma
        self.embedded_action_space = env.action_space
        self.action_space = env.action_space_pre_embedding
        self.video = video
        self.markovian = markovian
        self.update_mean = update_mean
        # keep track of best arm guess
        self.best_arm = []
        # define the prior mean
        if prior_mean is None:
            self.prior_mean = np.zeros((self.action_space.shape[0], 1))
        else:
            self.prior_mean = prior_mean
    
    def step_update(self):
        if self.markovian:
            action = self.action_trajectory[-1]
            print('actions taken: ', action)

            if self.video:
                # save the relevant stuff for plotting
                if self.estimator.fitted:
                    lcb = self.estimator.lcb(torch.tensor(self.action_space)).numpy().squeeze()
                    ucb = self.estimator.ucb(torch.tensor(self.action_space)).numpy().squeeze()
                    mean = self.estimator.mean(torch.tensor(self.action_space)).numpy().squeeze()
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
            print('actions taken: ', action)

            # obtain the value of the action
            state = self.env.action_space_pre_embedding[action].reshape(1, -1)
            # obtain the noise
            eps = np.random.normal(0, self.sigma)

            if callable(self.theta_star):
                fun_value = self.theta_star(state) + eps - self.prior_mean[action]
            else:
                z = self.estimator.embed(torch.tensor(state)).numpy()
                fun_value = z @ self.theta_star + eps - self.prior_mean[action]
            
            self.estimator.add_data_point(torch.tensor(state),
                                        torch.tensor([[fun_value]]))
            self.estimator.fit()
            self.objective.ucbs = self.estimator.ucb(torch.tensor(self.action_space)).numpy().squeeze() + self.prior_mean
            self.objective.lcbs = self.estimator.lcb(torch.tensor(self.action_space)).numpy().squeeze() + self.prior_mean
            # update the mean if required
            if self.update_mean:
                means, stds = self.estimator.mean_std(torch.tensor(self.action_space))
                self.objective.means = means.numpy().squeeze() + self.prior_mean
                self.objective.stds = stds.numpy().squeeze()
                self.objective.best_obs = np.maximum(self.objective.best_obs, fun_value + self.prior_mean[action])

            # calculate the best arm guess
            mean_estimates = self.estimator.mean(torch.tensor(self.action_space)).numpy().squeeze() + self.prior_mean
            self.best_arm.append(np.argmax(mean_estimates))

            if self.video:
                # save the relevant stuff for plotting
                lcb = self.estimator.lcb(torch.tensor(self.action_space)).numpy().squeeze()
                ucb = self.estimator.ucb(torch.tensor(self.action_space)).numpy().squeeze()
                mean = self.estimator.mean(torch.tensor(self.action_space)).numpy().squeeze()

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
                    z = self.estimator.embed(torch.tensor(state)).numpy()
                    fun_value = z @ self.theta_star + eps - self.prior_mean[action]
                
                self.estimator.add_data_point(torch.tensor(state),
                                            torch.tensor([[fun_value]]))
                
                if self.update_mean:
                    self.objective.best_obs = np.maximum(self.objective.best_obs, fun_value + self.prior_mean[action])
            
            self.estimator.fit()
            self.objective.ucbs = self.estimator.ucb(torch.tensor(self.embedded_action_space)).numpy().squeeze() + self.prior_mean
            self.objective.lcbs = self.estimator.lcb(torch.tensor(self.embedded_action_space)).numpy().squeeze() + self.prior_mean
            # update the mean if required
            if self.update_mean:
                means, stds = self.estimator.mean_std(torch.tensor(self.action_space))
                self.objective.means = means.numpy().squeeze() + self.prior_mean
                self.objective.stds = stds.numpy().squeeze()

            # calculate the best arm guess
            mean_estimates = self.estimator.mean(torch.tensor(self.action_space)).numpy().squeeze() + self.prior_mean
            self.best_arm.append(np.argmax(mean_estimates))

            # if self.video:
            #     # save the relevant stuff for plotting
            #     lcb = self.estimator.lcb(torch.tensor(self.embedded_action_space)).numpy().squeeze()
            #     ucb = self.estimator.ucb(torch.tensor(self.embedded_action_space)).numpy().squeeze()
            #     mean = self.estimator.mean(torch.tensor(self.embedded_action_space)).numpy().squeeze()

            #     # load the arrays from memory
            #     actions = np.load('actions.npy')
            #     lcbs = np.load('lcb.npy')
            #     means = np.load('mean.npy')
            #     ucbs = np.load('ucb.npy')

            #     # append the new values
            #     actions = np.vstack((actions, np.array(action_list)))
            #     lcbs = np.vstack((lcbs, lcb))
            #     means = np.vstack((means, mean))
            #     ucbs = np.vstack((ucbs, ucb))

            #     # save them in memory
            #     np.save('actions.npy', actions)
            #     np.save('lcb.npy', lcbs)
            #     np.save('mean.npy', means)
            #     np.save('ucb.npy', ucbs)
        
        else:
            pass

class ContinuousBanditFeedback(SimpleFeedback):
    def __init__(self, 
                env:MovementConstrainedBayesianOptimization, 
                objective:RewardFunctional, 
                estimator:Union[GaussianProcess, KernelizedFeatures], 
                theta_star:Union[np.array, Callable], 
                sigma:float, 
                video:bool = False, 
                markovian:bool = False,
                maximization_set_method:str = 'thompson_sampling') -> None:
        
        super().__init__(env, objective)
        self.estimator = estimator
        self.theta_star = theta_star
        self.sigma = sigma
        self.embedding = self.objective.embedding
        self.video = video
        self.markovian = markovian
        self.maximization_set_method = maximization_set_method
        if self.maximization_set_method == 'ucb':
            self.create_maximizers_grid()
    
    def thompson_sample_potential_maximizers(self):
        if type(self.estimator) == KernelizedFeatures:

            # sample thetas from the posterior and maximize
            maximizers, _ = self.estimator.sample_and_optimize(size = self.objective.num_of_maximizers)
            
            return maximizers.numpy()

        else:
            raise NotImplementedError('Thompson sampling for GPs not implemented yet')
    
    def create_maximizers_grid(self):
        if self.env.states_dim == 1:
            self.maximizers_grid = np.linspace(-0.5, 0.5, 101).reshape(-1, 1)
        elif self.env.states_dim == 2:
            self.maximizers_grid = np.array(np.meshgrid(np.linspace(-0.5, 0.5, 21), np.linspace(-0.5, 0.5, 21))).T.reshape(-1, 2)
        else:
            raise NotImplementedError('Maximizers grid for higher dimensions not implemented yet')

    def ucb_potential_maximizers(self):
        
        lcbs = self.estimator.lcb(torch.tensor(self.maximizers_grid)).numpy().squeeze()
        highest_lcb = np.max(lcbs)

        ucbs = self.estimator.ucb(torch.tensor(self.maximizers_grid)).numpy().squeeze()

        return self.maximizers_grid[ucbs >= highest_lcb]
    
    def step_update(self):
        if self.markovian:
            pass
        else:
            action = self.action_trajectory[-1]
            # see the next state
            next_state = self.env.next(self.env.state, action)
            print('next design: ', next_state)
            # obtain the corresponding observation
            eps = np.random.normal(0, self.sigma**2)
            if callable(self.theta_star):
                fun_value = self.theta_star(next_state)
            else:
                z = self.embedding.embed(torch.tensor(next_state)).numpy()
                eps = np.random.normal(0, self.sigma)
                fun_value = z @ self.theta_star + eps

            self.estimator.add_data_point(torch.tensor(next_state),
                                        torch.tensor(fun_value).reshape((-1, 1)))
            self.estimator.fit()

            if self.video:
                # save the relevant stuff for plotting
                memory_states = np.load('states.npy')
                memory_maximizers = np.load('maximizers.npy')
                memory_mean = np.load('mean.npy')
                memory_ucb = np.load('ucb.npy')
                memory_lcb = np.load('lcb.npy')

                # need to save the state
                np.save('states.npy', np.vstack((memory_states, np.array(next_state))))

                # need to save the set of maximizers
                # create copy of the set of maximizers
                maximizers = self.objective.set_of_maximizers.copy()

                # extend with empty arrays to match the number of maximizers
                # maximizers = np.vstack((maximizers, np.zeros((self.objective.num_of_maximizers - maximizers.shape[0], maximizers.shape[1]))))

                np.save('maximizers.npy', np.vstack((memory_maximizers, np.expand_dims(maximizers, 0))))

                if self.env.states_dim == 1:
                    # need to save the state of the GP
                    discrete_state_space = np.linspace(-0.5, 0.5, 101).reshape(-1, 1)

                    lcb = self.estimator.lcb(torch.tensor(discrete_state_space)).numpy().squeeze()
                    ucb = self.estimator.ucb(torch.tensor(discrete_state_space)).numpy().squeeze()
                    mean = self.estimator.mean(torch.tensor(discrete_state_space)).numpy().squeeze()

                    np.save('mean.npy', np.vstack((memory_mean, mean)))
                    np.save('ucb.npy', np.vstack((memory_ucb, ucb)))
                    np.save('lcb.npy', np.vstack((memory_lcb, lcb)))

            if self.maximization_set_method == 'ucb':
                self.objective.set_of_maximizers = self.ucb_potential_maximizers()
            elif self.maximization_set_method == 'thompson_sampling':
                self.objective.set_of_maximizers = self.thompson_sample_potential_maximizers()
            else:
                raise NotImplementedError('Maximization set method not implemented yet')

    def episode_update(self):
        if self.markovian:
            final_state = self.env.next(self.env.state, self.action_trajectory[-1])
            state_list = self.state_trajectory[1:] + [final_state]
            print('actions taken: ', state_list)

            for state in state_list:
                eps = np.random.normal(0, self.sigma**2)
                if callable(self.theta_star):
                    fun_value = self.theta_star(state) + eps
                else:
                    z = self.embedding.embed(torch.tensor(state)).numpy()
                    fun_value = z @ self.theta_star + eps
                
                self.estimator.add_data_point(torch.tensor(np.expand_dims(z, 0)),
                                        torch.tensor([[fun_value]]))
            
            self.estimator.fit()
            self.objective.set_of_maximizers = self.thompson_sample_potential_maximizers()


            if self.video:
                # save the relevant stuff for plotting

                # need to save the state

                # need to save the set of maximizers

                # need to save the state of the GP

                pass
        
        else:
            pass