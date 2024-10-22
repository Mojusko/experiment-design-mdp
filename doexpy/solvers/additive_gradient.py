from typing import Callable
import autograd.numpy as np
from doexpy.env.linear_system import ContinuousEnv

from doexpy.solvers.solver_base import ContinuousSolver
from doexpy.policies.policy_base import Policy
from doexpy.policies.base_policies.non_stationary_policy import NonStationaryPolicyContinuous

import torch
import torch.nn as nn
from torch.nn import ModuleList
from doexpy.utils.nn import DeterministicModule

from torchmin import minimize_constr

class AdditiveGradient(ContinuousSolver):
    def __init__(self, env: ContinuousEnv,
                reward: Callable, 
                gradient_steps = 256,
                num_multistarts = 256,
                verbosity = 0) -> None:
        
        super().__init__(env, reward)
        self.gradient_steps = gradient_steps
        self.verbosity = verbosity
        self.num_multistarts = num_multistarts
    
    def solve(self) -> Policy:
        '''
        Solves the MDP and returns the optimal policy.

        It does so by solving the following optimization problem:
        
            max sum_h reward(h, s, a)

        where s_h = s_{h-1} + a_{h-1}
              s_0 = initial_state

        we optimize across the actions a_0, ..., a_{H-1}.
        '''

        # save the reward contour for visualization
        x_grid = np.linspace(-0.5, 0.5, 101)
        y_grid = np.linspace(-0.5, 0.5, 101)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.reward(0, np.array([X[i, j], Y[i, j]]).reshape(1, -1), np.array([0, 0]).reshape(1, -1))

        reward_contours = np.load('reward_contours.npy')
        reward_contours = np.concatenate((reward_contours, Z.reshape(1, 101, 101)), axis = 0)
        np.save('reward_contours.npy', reward_contours)

        actions_dim = self.env.actions_dim

        # initialize the optimization variables
        actions = torch.zeros((self.num_multistarts, self.env.max_episode_length - self.env.h, actions_dim))
        # initialize the actions randomly
        actions = actions + torch.randn((self.num_multistarts, self.env.max_episode_length - self.env.h, actions_dim))
        # clamp the actions
        state = torch.tensor(self.env.state)
        state = state.expand(self.num_multistarts, -1)
        for h in range(self.env.max_episode_length - self.env.h):
            a_min = torch.maximum(-0.5 - state, torch.tensor(self.env.min_action))
            a_max = torch.minimum(0.5 - state, torch.tensor(self.env.max_action))
            actions[:, h, :] = torch.clamp(actions[:, h, :], a_min, a_max)
        
        # initialize the actions to the required gradient
        actions.requires_grad = True
        optimizer = torch.optim.Adam([actions], lr=1e-3)

        for j in range(self.gradient_steps):
            # start the optimization step
            loss = 0
            optimizer.zero_grad()
            # initialize the state
            state = torch.tensor(self.env.state)
            # expand the state to the batch size
            state = state.expand(self.num_multistarts, -1)

            for h in range(self.env.max_episode_length - self.env.h):
                loss = loss - self.reward(h, state, actions[:, h, :])
                state = state + actions[:, h, :]
            
            # consider the terminal reward
            loss = loss - self.reward(self.env.max_episode_length - self.env.h, state, torch.zeros((self.num_multistarts, actions_dim)))

            loss = loss.sum()
            # compute the gradients and take a step
            loss.backward()
            optimizer.step()

            # clamp the actions
            state = torch.tensor(self.env.state)
            state = state.expand(self.num_multistarts, -1)
            for h in range(self.env.max_episode_length - self.env.h):
                a_min = torch.maximum(-0.5 - state, torch.tensor(self.env.min_action))
                a_max = torch.minimum(0.5 - state, torch.tensor(self.env.max_action))
                actions[:, h, :].clamp(a_min, a_max)
            
            if (self.verbosity > 0) & (j % 10 == 0):
                print(loss.item())
        
        # find the best action from the multistarts
        with torch.no_grad():

            loss = 0

            state = torch.tensor(self.env.state)
            state = state.expand(self.num_multistarts, -1)

            for h in range(self.env.max_episode_length - self.env.h):
                state = state + actions[:, h, :]
                loss = loss - self.reward(h, state, actions[:, h, :])
        
        best_action = actions[torch.argmin(loss), :, :]

        policy = ModuleList()
        for h in range(self.env.max_episode_length - self.env.h):
            policy.append(DeterministicModule(best_action[h, :]))
        
        return NonStationaryPolicyContinuous(self.env, policy)


class AdditiveGradientTorchMinimize(ContinuousSolver):
    def __init__(self, env: ContinuousEnv,
                reward: Callable, 
                gradient_steps = 256,
                num_multistarts = 256,
                verbosity = 0) -> None:
        
        super().__init__(env, reward)
        self.gradient_steps = gradient_steps
        self.verbosity = verbosity
        self.num_multistarts = num_multistarts
    
    def solve(self) -> Policy:
        '''
        Solves the MDP and returns the optimal policy.

        It does so by solving the following optimization problem:
        
            max sum_h reward(h, s, a)

        where s_h = s_{h-1} + a_{h-1}
              s_0 = initial_state

        we optimize across the actions a_0, ..., a_{H-1}.
        '''
        actions_dim = self.env.actions_dim

        # initialize the optimization variables
        actions = torch.zeros((self.num_multistarts, self.env.max_episode_length - self.env.h, actions_dim))
        # initialize the actions randomly
        actions = actions + torch.randn((self.num_multistarts, self.env.max_episode_length - self.env.h, actions_dim))
        # clamp the actions
        state = torch.tensor(self.env.state)
        state = state.expand(self.num_multistarts, -1)
        for h in range(self.env.max_episode_length - self.env.h):
            a_min = torch.maximum(-0.5 - state, torch.tensor(self.env.min_action))
            a_max = torch.minimum(0.5 - state, torch.tensor(self.env.max_action))
            actions[:, h, :] = torch.clamp(actions[:, h, :], a_min, a_max)
        
        # initialize the actions to the required gradient
        actions.requires_grad = True

        def fn(x):
            loss = 0
            state = torch.tensor(self.env.state)
            state = state.expand(self.num_multistarts, -1)

            for h in range(self.env.max_episode_length - self.env.h):
                loss = loss - self.reward(h, state, x[:, h, :])
                state = state + x[:, h, :]
            
            # consider the terminal reward
            loss = loss - self.reward(self.env.max_episode_length - self.env.h, state, torch.zeros((self.num_multistarts, actions_dim)))

            loss = loss.sum()
            return loss


        # create constraints

        constraints = []

        def fn_constraint_movement(x):
            return x
        
        constraints.append(dict(fun = fn_constraint_movement, lb = self.env.min_action, ub = self.env.max_action))

        # box constraints
        init_state = torch.tensor(self.env.state)
        init_state = init_state.expand(self.num_multistarts, -1)
        def make_fn_constraint_box(h):
            def fn_constraint_box(x):
                state = init_state.clone()
                state += torch.sum(x[:, :h+1, :], dim = 1)
                return state
            return fn_constraint_box
        
        for h in range(self.env.max_episode_length - self.env.h):
            constraints.append(dict(fun = make_fn_constraint_box(h), lb = -0.5, ub = 0.5))

        # solve the optimization problem
        result = minimize_constr(fn, actions, constraints, disp=2)
        
        # find the best action from the multistarts
        with torch.no_grad():

            loss = 0

            state = torch.tensor(self.env.state)
            state = state.expand(self.num_multistarts, -1)

            for h in range(self.env.max_episode_length - self.env.h):
                state = state + actions[:, h, :]
                loss = loss - self.reward(h, state, actions[:, h, :])
        
        best_action = actions[torch.argmin(loss), :, :]

        policy = ModuleList()
        for h in range(self.env.max_episode_length - self.env.h):
            policy.append(DeterministicModule(best_action[h, :]))
        
        return NonStationaryPolicyContinuous(self.env, policy)