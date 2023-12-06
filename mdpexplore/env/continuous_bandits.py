from abc import ABC
import autograd.numpy as np
import numpy as np
from mdpexplore.env.continuous_env import ContinuousEnv
import torch
from scipy.linalg import norm

class ContinuousMovementConstrainedBayesianOptimization(ContinuousEnv, ABC):
    def __init__(self, states_dim: int, 
                actions_dim: int, 
                theta_star: np.array, 
                sigma: float, 
                discount_factor: float = 0.99, 
                max_episode_length:int = 10, 
                min_action:float = -0.1, 
                max_action:float = 0.1,
                init_state:np.array = None) -> None:
        
        if init_state is None:
            init_state = np.ones(states_dim).reshape(1, -1) * -0.5

        super().__init__(init_state=init_state)

        self.states_dim = states_dim
        self.actions_dim = actions_dim
        self.theta_star = theta_star
        self.sigma = sigma
        # Emissions are features
        self.emissions = None
        self.max_episode_length = max_episode_length
        self.terminal_state = None
        # set the discount factor
        self.discount_factor = discount_factor
        # whether the environment is constrained or not
        self.constrained = False
        self.stationary = False
        # set environment's constraints
        self.min_action = min_action
        self.max_action = max_action

    def next(self, state, action):
        # check if state is numpy array or pytorch tensor
        if isinstance(state, np.ndarray):
            next_state = np.clip(state + action, -0.5, 0.5)
        elif isinstance(state, torch.Tensor):
            next_state = torch.clip(state + action, 0.5, 0.5)
        
        return next_state

    def step(self, action: int):
        self.state = self.next(self.state, action)
        self.h += 1
        return action

    def convert(self, state):
        pass

    def is_valid_action(self, action, state) -> bool:
        if np.any(action >= self.max_action):
            return False
        
        elif np.any(action <= self.min_action):
            return False
        
        elif np.any(action + state >= 0.5):
            return False

        elif np.any(action + state <= -0.5):
            return False
        
        else:
            return True
    
    def get_transition_matrix(self):
        pass

    def reset(self) -> None:
        self.state = self.init_state
        self.h = 0
    
    def get_dim(self):
        return self.states_dim