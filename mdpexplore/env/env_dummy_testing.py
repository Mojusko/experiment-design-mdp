import numpy as np
import torch
from mdpexplore.solvers.dp import DP
from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.env.continuous_env import ContinuousEnv
from scipy.linalg import norm

class DummyTestEnv(DiscreteEnv):
    def __init__(self, states_num = 5, actions_num = 5, max_episode_length = 10, constrained = False):
        super().__init__(init_state=0)
        self.states_num = states_num
        self.actions_num = actions_num
        self.max_episode_length = max_episode_length
        self.terminal_state = None
        self.constrained = constrained
        self.emissions = np.arange(self.actions_num)

    def available_actions(self, state):
        return [a for a in range(self.actions_num) if self.is_valid_action(a, state)]

    def next(self, state, action):
        return action
    
    def convert(self, state):
        pass

    def step(self, action: int):
        self.visitations[action] += 1
        return action

    def get_transition_matrix(self) -> np.ndarray:
        tm = np.zeros((self.states_num, self.actions_num, self.states_num))
        for s in range(self.states_num):
            for a in range(self.states_num):
                if self.is_valid_action(a, s):
                    tm[s, a, a] = 1.
        return tm

    def is_valid_action(self, action, state) -> bool:
        if state == 0:
            return action in [0, 1]
            
        if state == 1:
            return action in [0, 1, 2]

        
        if state == 2:
            return action in [1, 2, 3]
        
        if state == 3:
            return action in [2, 3, 4]
        
        if state == 4:
            return action in [3, 4]

    def reset(self) -> None:
        self.state = self.init_state
    
    def get_dim(self):
        return 1
    
    def get_states_num(self):
        return self.states_num

class DummyTestEnvContinuous(ContinuousEnv):
    def __init__(self, max_episode_length = 3, constrained = False, min_action = -0.2, max_action = 0.2, init_state = np.array([-0.5]).reshape(1, 1)):
        super().__init__(init_state)
        self.states_dim = 1
        self.actions_dim = 1
        self.max_episode_length = max_episode_length
        self.terminal_state = None
        self.constrained = constrained

        # add constraints on action
        self.min_action = min_action
        self.max_action = max_action

    def next(self, state, action):
        # check if state is numpy array or pytorch tensor
        if isinstance(state, np.ndarray):
            next_state = np.clip(state + action, -0.5, 0.5)
        elif isinstance(state, torch.Tensor):
            next_state = torch.clip(state + action, 0.5, 0.5)
        
        return next_state
    
    def convert(self, state):
        pass

    def step(self, action: int):
        self.visitations[action] += 1
        return action

    def get_transition_matrix(self) -> np.ndarray:
        tm = np.zeros((self.states_num, self.actions_num, self.states_num))
        for s in range(self.states_num):
            for a in range(self.states_num):
                if self.is_valid_action(a, s):
                    tm[s, a, a] = 1.
        return tm

    def is_valid_action(self, action, state) -> bool:
        if norm(action) >= 0.2:
            return False
        
        elif action + state >= 0.5:
            return False

        elif action + state <= -0.5:
            return False
        
        else:
            return True

    def reset(self) -> None:
        self.state = self.init_state
    
    def get_dim(self):
        return self.states_dim