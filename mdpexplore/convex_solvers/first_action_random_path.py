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

class RandomPaths(ConvexSolverBase):
    def __init__(self, env : ContinuousEnv, objective, verbosity = 0, accuracy = 1e-4, num_paths = 100) -> None:
        super().__init__(env, objective, verbosity = verbosity, accuracy = accuracy)
        self.type = 'adaptive'
        self.stationary = False
        self.SummarizedPolicyType = MixturePolicy
        self.density_estimator = DeltaDensityEstimator(self.env, self.objective)
        self.num_paths = num_paths
    
    def optimize(self, emissions, visitations, episodes) -> None:
        '''
        Builds an optimization problem and solves it using cyipopt. The problem has to be of the form:

        min U(sum_h delta_h(x, a))

        s.t.
        x_h = x_{h-1} + a_{h-1}
        and 
        a_min <= a_h <= a_max
        and
        x_h in X

        We optimize over the first action only, and the rest are drawn randomly. We then check the best solution and
        use it as the policy.

        '''
        # pre-set variables for emmissions, visitations and episodes
        self.emissions = emissions
        self.visitations = visitations
        self.episodes = episodes
        self.dim = self.env.states_dim
        plot = False

        # first look at the planning horizon
        H_plan = self.env.max_episode_length - self.env.h

        # calculate the current density function
        density = self.density_estimator.density_oracle(self.policies, self.weights, self.densities, self.stationary)
        # pre-compute parts of the objective function
        self.objective.pre_compute(self.emissions, density, self.visitations, self.episodes)

        # now build the objective function
        def objective(X):
            # X has shape (H_plan * 2 * x_dim)
            # first split X into states and actions
            states = X[:H_plan * self.dim].reshape((H_plan, self.dim))
            actions = X[H_plan * self.dim:].reshape((H_plan, self.dim))

            # now define a density function
            # density = NonStationaryDeltaDensity(self.env, initial_states = states, initial_actions = actions)
            # now compute the objective
            return - self.objective.eval_quick(self.emissions, states, self.visitations, self.episodes)

        # sample the initial paths
        paths = self.sample_paths(self.num_paths, H_plan)
        
        # initialize the best objective and the best path
        best_obj = 1e10
        best_path = None

        # now optimize the first state for each path (i.e. the first two actions)
        for path in paths:
            # define the bounds
            bounds = []
            states = path[:H_plan * self.dim].reshape((H_plan, self.dim))
            actions = path[H_plan * self.dim:].reshape((H_plan, self.dim))

            x0 = states[0]
            x2 = states[2]

            x_min = np.maximum(-0.5, x0 + self.env.min_action, x2 - self.env.max_action)
            x_max = np.minimum(0.5, x0 + self.env.max_action, x2 - self.env.min_action)

            for d in range(self.env.states_dim):
                bounds.append((x_min[d], x_max[d]))
            
            # now optimize
            def objective_first_action(x):
                states_obj = np.concatenate((states[0, :].reshape(1, -1), x.reshape(1, -1), states[2:, :]), axis = 0)
                act0 = states_obj[1, :] - states_obj[0, :]
                act1 = states_obj[2, :] - states_obj[1, :]
                actions_obj = np.concatenate((act0.reshape(1, -1), act1.reshape(1, -1), actions[2:, :]), axis = 0)
                # now evaluate the objective
                obj = objective(np.concatenate((states_obj.reshape(1, -1), actions_obj.reshape(1, -1)), axis = 1).reshape(-1))
                return obj

            # get the gradient of the objective
            def gradient(X):
                # set the initial state to be differentiable

                states_for_grad = states.copy()
                states_for_grad[1, :] = np.zeros((self.dim))
                states_for_grad = torch.tensor(states_for_grad, requires_grad = False)
                state_1 = torch.tensor(X, requires_grad = True)
                states_for_grad[1, :] += state_1
                states_for_grad.retain_grad()

                # compute the objective
                obj = self.objective.eval_quick(self.emissions, states_for_grad, self.visitations, self.episodes)

                # compute the gradient
                obj.backward()

                # extract the gradient
                state1_grad = state_1.grad.detach().numpy()

                return - state1_grad

            x1_init = states[1, :]
            res = minimize(objective_first_action, x1_init, bounds = bounds, tol = 1e-4, jac = gradient)

            # get the best objective
            if res.fun < best_obj:
                best_obj = res.fun
                # build the best path
                best_states = np.concatenate((states[0, :].reshape(1, -1), res.x.reshape(1, -1), states[2:, :]), axis = 0)
                a0 = best_states[1, :] - best_states[0, :]
                a1 = best_states[2, :] - best_states[1, :]
                best_actions = np.concatenate((a0.reshape(1, -1), a1.reshape(1, -1), actions[2:, :]), axis = 0)
                best_path = np.concatenate((best_states.reshape(1, -1), best_actions.reshape(1, -1)), axis = 1).reshape(-1)
                
        # extract the best path
        states = best_states
        actions = best_actions

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
    
    def sample_paths(self, num_paths : int, H_plan : int) -> np.ndarray:
        '''
        Samples paths from the environment
        '''
        # initialize the output
        paths = np.zeros((num_paths, H_plan * 2 * self.env.states_dim))
        # get the current state
        init_state = self.env.state
        # use for loop
        for i in range(num_paths):
            current_state = init_state
            acts = np.zeros((H_plan, self.env.actions_dim))
            states = np.zeros((H_plan, self.env.states_dim))
            states[0, :] = init_state

            for h in range(H_plan - 1):
                # get bounds for current actions
                min_action = np.maximum(self.env.min_action, -0.5 - current_state)
                max_action = np.minimum(self.env.max_action, 0.5 - current_state)
                # sample the action
                act = np.random.uniform(min_action, max_action, size = (1, self.env.actions_dim))
                # update current state
                current_state = current_state + act
                # save the action and state
                acts[h, :] = act
                states[h + 1, :] = current_state
            
            # now concatenate the states and actions
            paths[i, :] = np.concatenate((states.reshape((1, H_plan * self.env.states_dim)), acts.reshape((1, H_plan * self.env.actions_dim))), axis = 1)

        return paths