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

# cvxpy imports
from cyipopt import minimize_ipopt

import matplotlib.pyplot as plt


class InteriorPoint(ConvexSolverBase):
    def __init__(self, env : ContinuousEnv, objective, verbosity = 0, accuracy = 1e-4) -> None:
        super().__init__(env, objective, verbosity = verbosity, accuracy = accuracy)
        self.type = 'interior-point'
        self.stationary = False
        self.SummarizedPolicyType = MixturePolicy
        self.density_estimator = DeltaDensityEstimator(self.env, self.objective)

        # initialize a plot
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlim(-0.5, 0.5)
        self.ax.set_ylim(-0.5, 0.5)

        # start an array of visited states
        self.visited_states = np.zeros((0, self.env.states_dim))
    
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

        # now build the objective function
        def objective(X):
            # X has shape (H_plan * 2, x_dim)
            # first split X into states and actions
            states = X[:H_plan * self.dim].reshape((H_plan, self.dim))
            actions = X[H_plan * self.dim:].reshape((H_plan, self.dim))

            # now define a density function
            # density = NonStationaryDeltaDensity(self.env, initial_states = states, initial_actions = actions)
            # now compute the objective
            return - self.objective.eval_quick(self.emissions, states, self.visitations, self.episodes)

        def gradient(X):
            # calculate the gradient of the objective function using pytorch backward

            # X has shape (H_plan * 2, x_dim)
            # first split X into states and actions
            states = X[:H_plan * self.dim].reshape((H_plan, self.dim))
            actions = X[H_plan * self.dim:].reshape((H_plan, self.dim))

            # initialize the gradient
            grad = np.zeros((H_plan * 2, self.dim))

            # set states to be differentiable
            states = states.copy()
            states = torch.tensor(states, requires_grad = True)
            states.retain_grad()

            # compute the objective
            obj = self.objective.eval_quick(self.emissions, states, self.visitations, self.episodes)

            # compute the gradient
            obj.backward()

            # extract the gradient
            state_grads = states.grad.detach().numpy()
            # set the first gradient to zero due to the initial constraint
            state_grads[0, :] = 0

            grad[:H_plan, :] = state_grads

            # as a_{h-1} = x_h - x_{h-1} we can compute the gradient of the actions
            grad[H_plan:2*H_plan - 1, :] = grad[1:H_plan, :] - grad[:H_plan - 1, :]

            return - grad.flatten()
            
        
        # now build the constraints
        def path_constraint(X):
            # enforces that x_h = x_{h-1} + a_{h-1}
            # X has shape (H_plan * 2, x_dim)

            # first split X into states and actions
            states = X[:H_plan * self.dim].reshape((H_plan, self.dim))
            actions = X[H_plan * self.dim:].reshape((H_plan, self.dim))
            # now compute the constraint
            constraint = states[1:, :] - states[:-1, :] - actions[:-1, :]

            return constraint.flatten()

        def initial_constraint(X):
            # enforces that x_0 = x_init
            # X has shape (H_plan * 2, x_dim)
            x_0 = X[:self.dim]
            x_init = self.env.state

            return (x_0 - x_init).flatten()

        def final_action_constraint(X):
            # set the final action to zero without loss of generality
            # X has shape (H_plan * 2, x_dim)
            final_action = X[-self.dim:]

            return final_action.flatten()
        
        constr = [{'type' : 'eq', 'fun' : path_constraint}, {'type' : 'eq', 'fun' : initial_constraint}, {'type' : 'eq', 'fun' : final_action_constraint}]
        
        # the remaining constraints are bounds
        bounds_states = [(-0.5, 0.5) for _ in range(H_plan * self.env.states_dim)]
        bounds_actions = [(self.env.min_action, self.env.max_action) for _ in range(H_plan * self.env.actions_dim)]
        bounds = bounds_states + bounds_actions

        # to initialize the optimization problem, solve the greedy problem
        greedy_state, greedy_action = self.calculate_optimal_greedy_state_action(density, emissions, visitations, episodes)

        # create a path from the current state to the greedy state
        path_state = np.array(self.env.state)
        init_states = np.zeros((H_plan, self.dim))
        init_actions = np.zeros((H_plan, self.dim))
        init_h = 0

        while init_h < H_plan - 1:
            init_states[init_h, :] = path_state

            # take a greedy action
            direction = greedy_state - path_state

            # check if step is achievable in a single step
            if np.all(direction <= self.env.max_action) and np.all(direction >= self.env.min_action):
                init_actions[init_h, :] = direction
                # set all remaining states to the greedy state
                init_states[init_h + 1:, :] = greedy_state
                init_h += 1
                break

            else:
                # clip wrt to min and max action
                direction = np.clip(direction, self.env.min_action, self.env.max_action)
                init_actions[init_h, :] = direction
                init_h += 1
                path_state = path_state + direction
        
        # set the greedy path
        greedy_path = np.zeros((H_plan * 2 * self.dim))
        greedy_path[:H_plan * self.dim] = init_states.flatten()
        greedy_path[H_plan * self.dim:] = init_actions.flatten()

        # sample a ton of initial paths and pick the best one
        num_initial_paths = 1
        # set all initial paths to the greedy path
        paths = greedy_path.reshape((1, -1)).repeat(num_initial_paths, axis = 0)

        for h in range(init_h, H_plan):
            # fill the remaining path with random actions
            acts = np.random.uniform(self.env.min_action, self.env.max_action, size = (num_initial_paths, self.env.actions_dim))
            # clamp the actions by making sure that the next state is in [-0.5, 0.5]
            acts = np.clip(acts, -0.5 - path_state, 0.5 - path_state)
            # save the paths
            paths[:, h * self.dim : (h+1) * self.dim] = path_state
            paths[:, H_plan * self.dim + h * self.dim : H_plan * self.dim + (h + 1) * self.dim] = acts

            # update the path state
            path_state  = path_state + acts
        
        # now evaluate the objective function for each path
        best_idx = 0
        best_obj = 1e10

        import time
        t1 = time.time()

        for i in range(num_initial_paths):
            obj = objective(paths[i, :])
            if obj < best_obj:
                best_obj = obj
                best_idx = i

        t2 = time.time()
        # print('Average time per function call: ', (t2 - t1) / num_initial_paths)
        # print('Best initial objective: ', best_obj)

        # select the best path
        best_init_path = paths[best_idx, :]

        # now solve the optimization problem
        # t1 = time.time()
        # res = minimize_ipopt(objective, best_init_path, bounds = bounds, constraints = constr, jac = gradient, options = {'disp' : 5, 'max_iter' : 50})
        # t2 = time.time()
        # print('Time to solve: ', t2 - t1)
        # print('Objective: ', res.fun)
        # now extract the optimal policy
        # states = res.x[:H_plan * self.dim].reshape((H_plan, self.dim))
        # actions = res.x[H_plan * self.dim:].reshape((H_plan, self.dim))
        # print('Check against objective', objective(res.x))
        # see what happens when we play the greedy + random policy
        states = best_init_path[:H_plan * self.dim].reshape((H_plan, self.dim))
        actions = best_init_path[H_plan * self.dim:].reshape((H_plan, self.dim))

        self.visited_states = np.concatenate((self.visited_states, states[0, :].reshape(1, -1)), axis = 0)

        # create whole path using visited states and future ones
        future_path = np.concatenate((self.visited_states, states[1:, :]), axis = 0).reshape(1, 30, 2)
        # save for plotting
        future_path_old = np.load('future_states.npy')
        future_path = np.concatenate((future_path_old, future_path), axis = 0)
        np.save('future_states.npy', future_path)

        # also save the greedy reward contour for visualization
        grad_reward_fn = self._reward_fn_gradient(density, emissions, visitations, episodes)
        x_grid = np.linspace(-0.5, 0.5, 101)
        y_grid = np.linspace(-0.5, 0.5, 101)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = grad_reward_fn(0, np.array([X[i, j], Y[i, j]]).reshape(1, -1), np.array([0, 0]).reshape(1, -1))

        reward_contours = np.load('reward_contours.npy')
        reward_contours = np.concatenate((reward_contours, Z.reshape(1, 101, 101)), axis = 0)
        np.save('reward_contours.npy', reward_contours)

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
    
    def calculate_optimal_greedy_state_action(self, distribution, emissions, visitations, episodes):
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

        # remove from best objective if best is not < 90% of the best objective
        [optimal_state_actions, optimal_objs] = zip(*[(x, y) for x, y in zip(optimal_state_actions, optimal_objs) if y < 0.9 * best_obj])

        # extract the optimal state-action
        dist_to_current_state = [np.linalg.norm(x[:self.env.states_dim] - self.env.state) for x in optimal_state_actions]
        idx = np.argmin(dist_to_current_state)
        optimal_state_action = optimal_state_actions[idx]

        state = optimal_state_action[:self.env.states_dim]
        action = optimal_state_action[self.env.states_dim:]

        print('Greedy state: ', state)

        return state, action