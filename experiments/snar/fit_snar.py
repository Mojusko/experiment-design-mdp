from summit.benchmarks import SnarBenchmark
from summit.utils.dataset import DataSet

import numpy as np
from doexpy.densities.density_estimators import DeltaDensityEstimator
from doexpy.env.continuous_env import ContinuousEnv
from torch.nn import ModuleList
from doexpy.utils.nn import DeterministicModule
from doexpy.policies.summary_policies.mixture_policy import MixturePolicy
from doexpy.convex_solvers.convex_solvers_base import ConvexSolverBase
from doexpy.policies.base_policies.non_stationary_policy import NonStationaryPolicyContinuous
from scipy.optimize import minimize
from scipy.stats import norm
from stpy.continuous_processes.kernelized_features import KernelizedFeatures

import torch


class SnAr():
    def __init__(self):
        self.dim = 4

        self.gamma = 0.2
        self.kappa = 0.8

        self.optimum = 1.42998

        self.name = 'SnarBenchmark'

        self.snar_bench = SnarBenchmark()

    def query_function(self, x):
        x = x.reshape(1, -1) + 0.5
        temp = x[:, 0] * 80 + 40
        conc_dfnb = x[:, 1] * 0.4 + 0.1
        residence_time = x[:, 2] * 1.5 + 0.5
        equiv_pldn = x[:, 3] * 4 + 1

        values = {
            ("tau", "DATA"): [residence_time],
            ("equiv_pldn", "DATA"): [equiv_pldn],
            ("conc_dfnb", "DATA"): [conc_dfnb],
            ("temperature", "DATA"): [temp],
        }

        conditions = DataSet(values)
        experiments = self.snar_bench.run_experiments(conditions, computation_time = False)
        return (np.maximum(experiments['sty'][0] / 10000 - experiments['e_factor'][0] / 10, -5) + 3) / 2
    
class LSR(ConvexSolverBase):
    def __init__(self, env : ContinuousEnv, objective, estimator : KernelizedFeatures, verbosity = 0, accuracy = 1e-4, gamma = 0.01) -> None:
        super().__init__(env, objective, verbosity = verbosity, accuracy = accuracy)
        self.type = 'LSR'
        self.stationary = False
        self.SummarizedPolicyType = MixturePolicy
        self.density_estimator = DeltaDensityEstimator(self.env, self.objective)
        self.gamma = gamma
        self.estimator = estimator

        # start an array of visited states
        self.visited_states = np.zeros((0, self.env.states_dim))
    
    def optimize(self, emissions, visitations, episodes) -> None:
        '''
        Builds and optimization path using the SnAKe algorithm.
        '''
        # pre-set variables for emmissions, visitations and episodes
        self.emissions = emissions
        self.visitations = visitations
        self.episodes = episodes
        self.dim = self.env.states_dim

        # first optimize EI locally
        current_state = self.env.state
        # create the local bounds
        bounds = []
        for d in range(self.env.states_dim):
            lower_bound = self.env.min_action + current_state[0, d]
            lower_bound = np.clip(lower_bound, -0.5, 0.5)
            upper_bound = self.env.max_action + current_state[0, d]
            upper_bound = np.clip(upper_bound, -0.5, 0.5)
            bounds.append((lower_bound, upper_bound))
        
        # now optimize
        res = minimize(self.EI, current_state.reshape(-1), bounds = bounds, tol = 1e-4)
        # get the optimal state
        optimal_state = torch.from_numpy(res.x)
        # get optimal value
        optimal_value = res.fun * -1
        # check if optimal values is larger than gamma
        if optimal_value > self.gamma:
            # choose this action to return
            action = optimal_state - current_state
        # otherwise optimize EI globally
        else:
            # create the global bounds
            bounds = [(-0.5, 0.5) for _ in range(self.env.states_dim)]
            
            best_EI = 1e-10
            best_state = None

            # run with 20 random initializations
            for _ in range(20):
                x0 = np.random.uniform(-0.5, 0.5, self.env.states_dim)
                # now optimize
                res = minimize(self.EI, x0, bounds = bounds, tol = 1e-4)
                # get the optimal state
                optimal_state = torch.from_numpy(res.x)
                # get optimal value
                optimal_value = res.fun
                # check if optimal values is larger than gamma
                if optimal_value < best_EI:
                    # choose this action to return
                    best_EI = optimal_value
                    best_state = optimal_state
            
            action = best_state - current_state
        
        actions = np.zeros((self.env.max_episode_length - self.env.h, self.env.actions_dim))
        actions[0, :] = action

        # now create the policy
        policy = ModuleList()
        # define actions as a tensor
        actions = torch.tensor(actions, requires_grad = False)
        for h in range(self.env.max_episode_length - self.env.h):
            policy.append(DeterministicModule(actions[h, :]))

        output_policy = NonStationaryPolicyContinuous(self.env, policy)
        # create summarized policy
        self.policies.append(output_policy)
        self.weights = [1.0]
        self.summarize()

        return self.summarized_policy, self.policies, self.weights, self.densities

    def EI(self, x):
        '''
        Calculates the expected improvement of a point x.
        '''
        # first calculate the mean and variance of the current state
        mean, std = self.estimator.mean_std(torch.tensor(x).reshape(1, -1))
        # now calculate the EI
        best_obs = np.max(self.estimator.y.numpy())

        # calculate the EI
        EI = (mean - best_obs) * norm.cdf((mean - best_obs) / std) + std * norm.pdf((mean - best_obs) / std)

        return -EI