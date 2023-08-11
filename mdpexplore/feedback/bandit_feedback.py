from enum import Enum
import random

import numpy as np
import torch
from stpy.continuous_processes.gauss_procc import GaussianProcess
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
from mdpexplore.policies.density_policy import DensityPolicy, MarginalDensityPolicy
from mdpexplore.policies.mixture_policy import MixturePolicy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from mdpexplore.feedback.feedback_base import SimpleFeedback

class BanditFeedback(SimpleFeedback):
    def __init__(self, env:MovementConstrainedBayesianOptimization, objective:RewardFunctional, estimator:GaussianProcess, theta_star:np.array, sigma:float, video:bool = False) -> None:
        super().__init__(env, objective)
        self.estimator = estimator
        self.theta_star = theta_star
        self.sigma = sigma
        self.embedded_action_space = env.action_space
        self.video = video
    
    def step_update(self):
        pass

    def episode_update(self):
        action_list = self.action_trajectory
        print('actions taken: ', action_list)

        for action in action_list:
            z = self.embedded_action_space[action]
            eps = np.random.normal(0, self.sigma)
            fun_value = z @ self.theta_star + eps
            self.estimator.add_data_point(torch.tensor(np.expand_dims(z, 0)),
                                    torch.tensor([[fun_value]]))
        
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