from typing import Union
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
    def __init__(self, env:MovementConstrainedBayesianOptimization, objective:RewardFunctional, estimator:Union[GaussianProcess, KernelizedFeatures], theta_star:np.array, sigma:float, video:bool = False, markovian:bool = False) -> None:
        super().__init__(env, objective)
        self.estimator = estimator
        self.theta_star = theta_star
        self.sigma = sigma
        self.embedded_action_space = env.action_space
        self.video = video
        self.markovian = markovian
    
    def step_update(self):
        if self.markovian:
            pass
        else:
            action = self.action_trajectory[-1]
            print('actions taken: ', action)
            
            z = self.embedded_action_space[action]
            eps = np.random.normal(0, self.sigma)
            fun_value = z @ self.theta_star + eps
            self.estimator.add_data_point(torch.tensor(np.expand_dims(z, 0)),
                                        torch.tensor([[fun_value]]))
            self.estimator.fit()
            self.objective.ucbs = self.estimator.ucb(torch.tensor(self.embedded_action_space)).numpy().squeeze()
            self.objective.lcbs = self.estimator.lcb(torch.tensor(self.embedded_action_space)).numpy().squeeze()

            # Compute the differences to get the UCBs and LCBs for the objective denominator
            n, m = self.embedded_action_space.shape
            arr1_reshaped = self.embedded_action_space.reshape(n, 1, m)
            arr2_reshaped = self.embedded_action_space.reshape(1, n, m)
            diffs = arr1_reshaped - arr2_reshaped
            self.objective.diff_ucbs = self.estimator.ucb(torch.tensor(diffs.reshape((-1, m)))).numpy().reshape((n, n))
            self.objective.diff_lcbs = self.estimator.lcb(torch.tensor(diffs.reshape((-1, m)))).numpy().reshape((n, n))

            if self.video:
                # save the relevant stuff for plotting
                lcb = self.estimator.lcb(torch.tensor(self.embedded_action_space)).numpy().squeeze()
                ucb = self.estimator.ucb(torch.tensor(self.embedded_action_space)).numpy().squeeze()
                mean = self.estimator.mean(torch.tensor(self.embedded_action_space)).numpy().squeeze()

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
                z = self.embedded_action_space[action]
                eps = np.random.normal(0, self.sigma)
                fun_value = z @ self.theta_star + eps
                self.estimator.add_data_point(torch.tensor(np.expand_dims(z, 0)),
                                        torch.tensor([[fun_value]]))
            
            self.estimator.fit()
            self.objective.ucbs = self.estimator.ucb(torch.tensor(self.embedded_action_space)).numpy().squeeze()
            self.objective.lcbs = self.estimator.lcb(torch.tensor(self.embedded_action_space)).numpy().squeeze()

            # Compute the differences to get the UCBs and LCBs for the objective denominator
            n, m = self.embedded_action_space.shape
            arr1_reshaped = self.embedded_action_space.reshape(n, 1, m)
            arr2_reshaped = self.embedded_action_space.reshape(1, n, m)
            diffs = arr1_reshaped - arr2_reshaped
            self.objective.diff_ucbs = self.estimator.ucb(torch.tensor(diffs.reshape((-1, m)))).numpy().reshape((n, n))
            self.objective.diff_lcbs = self.estimator.lcb(torch.tensor(diffs.reshape((-1, m)))).numpy().reshape((n, n))

            if self.video:
                # save the relevant stuff for plotting
                lcb = self.estimator.lcb(torch.tensor(self.embedded_action_space)).numpy().squeeze()
                ucb = self.estimator.ucb(torch.tensor(self.embedded_action_space)).numpy().squeeze()
                mean = self.estimator.mean(torch.tensor(self.embedded_action_space)).numpy().squeeze()

                # load the arrays from memory
                actions = np.load('actions.npy')
                lcbs = np.load('lcb.npy')
                means = np.load('mean.npy')
                ucbs = np.load('ucb.npy')

                # append the new values
                actions = np.vstack((actions, np.array(action_list)))
                lcbs = np.vstack((lcbs, lcb))
                means = np.vstack((means, mean))
                ucbs = np.vstack((ucbs, ucb))

                # save them in memory
                np.save('actions.npy', actions)
                np.save('lcb.npy', lcbs)
                np.save('mean.npy', means)
                np.save('ucb.npy', ucbs)
        
        else:
            pass

class ContinuousBanditFeedback(SimpleFeedback):
    def __init__(self, env:MovementConstrainedBayesianOptimization, objective:RewardFunctional, estimator:Union[GaussianProcess, KernelizedFeatures], theta_star:np.array, sigma:float, video:bool = False, markovian:bool = False) -> None:
        super().__init__(env, objective)
        self.estimator = estimator
        self.theta_star = theta_star
        self.sigma = sigma
        self.embedding = self.objective.embedding
        self.video = video
        self.markovian = markovian
    
    def thompson_sample_potential_maximizers(self):
        if type(self.estimator) == KernelizedFeatures:

            # sample thetas from the posterior and maximize
            maximizers = np.zeros((self.objective.num_of_maximizers, self.env.states_dim))
            for i in range(self.objective.num_of_maximizers):
                max_x, max_f = self.estimator.sample_and_optimize()
                maximizers[i] = max_x
            
            return maximizers

        else:
            raise NotImplementedError('Thompson sampling for GPs not implemented yet')
    
    def step_update(self):
        if self.markovian:
            pass
        else:
            action = self.action_trajectory[-1]
            # see the next state
            next_state = self.env.next(self.env.state, action)
            print('next design: ', next_state)
            # obtain the corresponding observation
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
                np.save('maximizers.npy', np.vstack((memory_maximizers, self.objective.set_of_maximizers[None, ...])))

                if self.env.states_dim == 1:
                    # need to save the state of the GP
                    discrete_state_space = np.linspace(-0.5, 0.5, 101).reshape(-1, 1)

                    lcb = self.estimator.lcb(torch.tensor(discrete_state_space)).numpy().squeeze()
                    ucb = self.estimator.ucb(torch.tensor(discrete_state_space)).numpy().squeeze()
                    mean = self.estimator.mean(torch.tensor(discrete_state_space)).numpy().squeeze()

                    np.save('mean.npy', np.vstack((memory_mean, mean)))
                    np.save('ucb.npy', np.vstack((memory_ucb, ucb)))
                    np.save('lcb.npy', np.vstack((memory_lcb, lcb)))

            self.objective.set_of_maximizers = self.thompson_sample_potential_maximizers()

    def episode_update(self):
        if self.markovian:
            final_state = self.env.next(self.env.state, self.action_trajectory[-1])
            state_list = self.state_trajectory[1:] + [final_state]
            print('actions taken: ', state_list)

            for state in state_list:
                z = self.embedding.embed(torch.tensor(state)).numpy()
                eps = np.random.normal(0, self.sigma)
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