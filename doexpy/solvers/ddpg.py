from typing import Callable
import autograd.numpy as np
from doexpy.env.linear_system import ContinuousEnv

from doexpy.solvers.solver_base import ContinuousSolver
from doexpy.policies.policy_base import Policy
from doexpy.policies.base_policies.non_stationary_policy import NonStationaryPolicyContinuous

import torch
import torch.nn as nn
from torch.nn import ModuleList
from doexpy.utils.nn import MLP

import matplotlib.pyplot as plt

class DDPG(ContinuousSolver):
    def __init__(self, env: ContinuousEnv,
                reward: Callable, 
                hidden_dim = 64, 
                buffer_size = 256, 
                update_iterations = 32, 
                gradient_steps = 32, 
                exploration_noise = 0.1, 
                polyak = 0.995, 
                discount_factor = 0.99,
                target_smoothing_clip = 0.3,
                verbosity = 0) -> None:
        
        super().__init__(env, reward)
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.update_iterations = update_iterations
        self.gradient_steps = gradient_steps
        self.exploration_noise = exploration_noise
        self.polyak = polyak
        self.discount_factor = discount_factor
        self.target_policy_smoothing_c = target_smoothing_clip
        self.verbosity = verbosity

        # we define a Q-function for each time-step
        self.q_functions = ModuleList([])
        self.target_q_functions = ModuleList([])
        for h in range(self.env.max_episode_length - self.env.h):
            self.q_functions.append(MLP(self.env.states_dim + self.env.actions_dim, 1, hidden_dim = self.hidden_dim))
            self.target_q_functions.append(MLP(self.env.states_dim + self.env.actions_dim, 1, hidden_dim = self.hidden_dim))
            self.target_q_functions[h].load_state_dict(self.q_functions[h].state_dict())
        
        # we define a policy for each time-step
        self.policies = ModuleList([])
        self.target_policies = ModuleList([])
        for h in range(self.env.max_episode_length - self.env.h):
            self.policies.append(MLP(self.env.states_dim, self.env.actions_dim, hidden_dim = self.hidden_dim))
            self.target_policies.append(MLP(self.env.states_dim, self.env.actions_dim, hidden_dim = self.hidden_dim))
            self.target_policies[h].load_state_dict(self.policies[h].state_dict())

        self.buffer = []

    def solve(self) -> Policy:
        '''
        Solves the MDP and returns the optimal policy.
        '''

        # define loss
        mse_loss = nn.MSELoss()
        
        for i in range(self.update_iterations):
            self.fill_buffer(i)
            for h, buffer_h in reversed(list(enumerate(self.buffer))):
                states, actions, reward, next_states = buffer_h

                # define optimizer
                optimizer_q_function = torch.optim.Adam(self.q_functions[h].parameters(), lr = 1e-3)
                optimizer_policy = torch.optim.Adam(self.policies[h].parameters(), lr = 1e-3)

                states = torch.tensor(states)
                actions = torch.tensor(actions)

                for j in range(self.gradient_steps):

                    self.compute_targets(reward, next_states, h)

                    loss_q_function = mse_loss(self.q_functions[h](torch.cat((states, actions), dim = -1)), self.targets)

                    optimizer_q_function.zero_grad()
                    loss_q_function.backward()
                    optimizer_q_function.step()

                    # update target networks
                    for target_param, param in zip(self.target_q_functions[h].parameters(), self.q_functions[h].parameters()):
                        target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)

                    # update policy once every two steps
                    if j % 2 == 1:
                        policy_acts = self.policies[h](states)
                        loss_policy = -self.q_functions[h](torch.cat((states, policy_acts), dim = -1)).mean()
                        # add penalty for actions outside of the action space
                        loss_policy += torch.mean(100 * torch.max(policy_acts - self.env.max_action, torch.zeros_like(policy_acts))**2)
                        loss_policy += torch.mean(100 * torch.max(self.env.min_action - policy_acts, torch.zeros_like(policy_acts))**2)

                        optimizer_policy.zero_grad()
                        loss_policy.backward()
                        optimizer_policy.step()
                    
                        # update target networks
                        for target_param, param in zip(self.target_policies[h].parameters(), self.policies[h].parameters()):
                            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)
                    
                    if (self.verbosity > 3) & (j % 20 == 1):
                        print(f'Gradient step {j+1}/{self.gradient_steps} completed.')
                        print(f'Loss Q-Function with h = {h}: {loss_q_function.item()}')
                        if j % 2 == 1:
                            print(f'Loss Policy with h = {h}: {loss_policy.item()}')
            
            self.buffer = []

            if self.verbosity > 2:
                print(f'Iteration {i+1}/{self.update_iterations} completed.')
                print(f'Loss Q-Function: {loss_q_function.item()}')
                print(f'Loss Policy: {loss_policy.item()}')

        return NonStationaryPolicyContinuous(self.env, self.policies)

    def fill_buffer(self, i:int):
        '''
        Runs the policy in the environment to fill the buffer.
        '''
        # on the first iteration, initialize the buffer randomly to obtain a diverse set of states
        if i == 0:
            states = torch.rand((self.buffer_size, self.env.states_dim)) - 0.5
        
        # on the following iterations, initialize the buffer with the initial state to focus on the region of interest
        else:
            initial_state = self.env.state
            states = initial_state
            # repeat state for buffer size
            # states = np.repeat(states, self.buffer_size, axis = 0)
            # do the same but with torch
            states = states.repeat(self.buffer_size, 1)

        # states = np.random.uniform(-0.5, 0.5, size = (self.buffer_size, self.env.states_dim))

        for h in range(self.env.max_episode_length - self.env.h):

            actions = self.policies[h](torch.tensor(states)).detach()
            noise = torch.randn_like(actions) * self.exploration_noise
            actions += noise
            actions = torch.clip(actions, self.env.min_action, self.env.max_action)

            reward = self.reward(h, states, actions).reshape(self.buffer_size, 1)

            next_states = self.env.next(states, actions)

            self.buffer.append((states, actions, reward, next_states))

            states = next_states

    def compute_targets(self, reward, next_states, h):
        '''
        Compute the targets y(r, s', d) = r + gamma * (1 - d) * Q(s', pi(s'))
        '''
        next_states = torch.tensor(next_states)
        reward = torch.tensor(reward)

        if h == self.env.max_episode_length - self.env.h - 1:
            target = reward
        
        else:
            action = self.target_policies[h](next_states)
            noise = torch.randn_like(action) * self.exploration_noise
            noise = torch.clip(noise, - self.target_policy_smoothing_c, self.target_policy_smoothing_c)
            action += noise
            action = torch.clip(action, self.env.min_action, self.env.max_action)

            target = reward + self.discount_factor * self.target_q_functions[h+1](torch.cat((next_states, action), dim = -1))
        
        self.targets = target