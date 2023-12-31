import numpy as np
from mdpexplore.policies.summary_policies.density_policy import DensityPolicy
from mdpexplore.policies.summary_policies.mixture_policy import MixturePolicy
from mdpexplore.densities.continous_densities import NonStationaryDeltaDensity
from mdpexplore.densities.density_estimators import DeltaDensityEstimator
from mdpexplore.env.continuous_env import ContinuousEnv
from torch.nn import ModuleList
from mdpexplore.utils.nn import DeterministicModule
from typing import Callable, Type, Union, Tuple
from mdpexplore.convex_solvers.convex_solvers_base import ConvexSolverBase
from mdpexplore.policies.base_policies.non_stationary_policy import NonStationaryPolicyContinuous
from mdpexplore.densities.continous_densities import ContinuousDensity
from scipy.optimize import minimize

import torch

import time


class ContinuousGreedyApproximation(ConvexSolverBase):
    '''
    Greedy approximation algorithm for continuous environments. We find the maximum of $\nalba$ U and find the
    shortest path to it. This means using a single component of the Frank-Wolfe algorithm.
    '''
    def __init__(self, env : ContinuousEnv, objective, verbosity = 0, accuracy = 1e-4) -> None:
        super().__init__(env, objective, verbosity = verbosity, accuracy = accuracy)
        self.type = 'greedy_approximation'
        self.stationary = False
        self.SummarizedPolicyType = MixturePolicy
        self.density_estimator = DeltaDensityEstimator(self.env, self.objective)

    
    def optimize(self, emissions, visitations, episodes) -> None:
        '''
        Returns the greedy policy for the current state distribution.
        '''
        # pre-set variables for emmissions, visitations and episodes
        self.emissions = emissions
        self.visitations = visitations
        self.episodes = episodes
        self.dim = self.env.states_dim

        # first look at the planning horizon
        H_plan = self.env.max_episode_length - self.env.h

        # calculate the current density function
        density = self.density_estimator.density_oracle(self.policies, self.weights, self.densities, self.stationary)
        # pre-compute parts of the objective function
        self.objective.pre_compute(self.emissions, density, self.visitations, self.episodes)

        # to initialize the optimization problem, solve the greedy problem
        greedy_state, greedy_action = self.calculate_optimal_greedy_state_action(density, emissions, visitations, episodes, H_plan)

        # create a path from the current state to the greedy state
        path_state = np.array(self.env.state)
        states = np.zeros((H_plan, self.dim))
        actions = np.zeros((H_plan, self.dim))
        init_h = 0

        while init_h < H_plan - 1:
            states[init_h, :] = path_state

            # take a greedy action
            direction = greedy_state - path_state

            # check if step is achievable in a single step
            if np.all(direction <= self.env.max_action) and np.all(direction >= self.env.min_action):
                actions[init_h, :] = direction
                # set all remaining states to the greedy state
                states[init_h + 1:, :] = greedy_state
                init_h += 1
                break

            else:
                # clip wrt to min and max action
                direction = np.clip(direction, self.env.min_action, self.env.max_action)
                actions[init_h, :] = direction
                init_h += 1
                path_state = path_state + direction


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
    
    def _reward_fn_gradient(self, distribution: Union[np.ndarray, ContinuousDensity], emissions, visitations, episodes) -> Union[np.ndarray, Callable]:
        """Computes the reward functional differentiated wrt to the state distribution

        Args:
            distribution (np.ndarray): state distribution to compute the reward function

        Returns:
            np.ndarray: gradient of the functional wrt to the state distribution - i.e. the reward function
        """

        if self.objective.get_type() == "adaptive":
            if self.stationary:
                grad_fn = lambda h, s, a: self.objective.get_gradient_density(emissions, distribution, visitations, episodes, s, a)
            else:
                grad_fn = lambda h, s, a: self.objective.get_gradient_density(emissions, distribution, visitations, episodes, s, a)
        else:
            if self.stationary:
                grad_fn = lambda h, s, a: self.objective.get_gradient_density(emissions, distribution, visitations, episodes, s, a)
            else:
                grad_fn = lambda h, s, a: self.objective.get_gradient_density(emissions, distribution, visitations, episodes, s, a)
        
        return grad_fn
    
    def calculate_optimal_greedy_state_action(self, distribution, emissions, visitations, episodes, H_plan):
        grad_reward_fn = self._reward_fn_gradient(distribution, emissions, visitations, episodes)

        def greedy_objective(x):
            return - grad_reward_fn(0, x[:self.dim].reshape(1, -1), x[self.dim:].reshape(1, -1))
        
        # define the bounds
        bounds_states = [(-0.5, 0.5) for _ in range(self.env.states_dim)]
        bounds_actions = [(self.env.min_action, self.env.max_action) for _ in range(self.env.actions_dim)]

        # now solve the optimization problem, doing a few random restarts
        best_obj = 1e10

        optimal_state_actions = []
        optimal_objs = []

        for _ in range(100):
            x_init = np.random.uniform(-0.5, 0.5, size = (self.env.states_dim))
            a_init = np.zeros((self.env.actions_dim))

            x_init = np.concatenate((x_init, a_init))

            res = minimize(greedy_objective, x_init, bounds = bounds_states + bounds_actions, options = {'disp' : False}, tol = 1e-20)

            if res.fun < best_obj:
                best_obj = res.fun
                best_state_action = res.x
            
            if res.success:
                optimal_state_actions.append(res.x)
                optimal_objs.append(res.fun)
        
        # if there were no successful runs, just return the best one and print a warning
        if len(optimal_state_actions) == 0:
            print('WARNING: No successful runs for greedy optimization')
            state = best_state_action[:self.env.states_dim]
            action = best_state_action[self.env.states_dim:]

            return state, action

        # check each of the optimal state-actions paths
        best_rewards = 1e10
        best_idx = 0
        for curr_idx, op_state_action in enumerate(optimal_state_actions):

            states = np.zeros((H_plan, self.dim))
            actions = np.zeros((H_plan, self.dim))
            sum_rewards = 0.0

            path_state = np.array(self.env.state)
            greedy_state = op_state_action[:self.env.states_dim]
            greedy_action = op_state_action[self.env.states_dim:]

            init_h = 0
            while init_h < H_plan - 1:
                states[init_h, :] = path_state

                # take a greedy action
                direction = greedy_state - path_state

                # check if step is achievable in a single step
                if np.all(direction <= self.env.max_action) and np.all(direction >= self.env.min_action):
                    actions[init_h, :] = direction
                    # set all remaining states to the greedy state
                    states[init_h + 1:, :] = greedy_state
                    init_h += 1
                    break

                else:
                    # clip wrt to min and max action
                    direction = np.clip(direction, self.env.min_action, self.env.max_action)
                    actions[init_h, :] = direction
                    init_h += 1
                    path_state = path_state + direction
            
            # now calculate the reward
            for h in range(H_plan):
                state_action = np.concatenate((states[h, :], actions[h, :]))
                sum_rewards += greedy_objective(state_action)
            
            if sum_rewards < best_rewards:
                best_rewards = sum_rewards
                best_idx = curr_idx

        # extract the optimal state-action
        optimal_state_action = optimal_state_actions[best_idx]

        state = optimal_state_action[:self.env.states_dim]
        action = optimal_state_action[self.env.states_dim:]

        print('Greedy state: ', state)

        return state, action