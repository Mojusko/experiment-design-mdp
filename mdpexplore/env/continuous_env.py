from abc import ABC, abstractmethod
from mdpexplore.env.discrete_env import Environment
import torch 

class ContinuousEnv(Environment):
    def __init__(self, init_state) -> None:
        super().__init__()
        self.type = 'continuous'
        self.init_state = init_state
        self.state = init_state
        self.states_dim = init_state.size()[1]
        self.actions_dim = None
        self.min_action = None
        self.max_action = None
        self.visitations = None
        self.max_episode_length = None
        self.h = 0

    @abstractmethod
    def next(self, state, action):
        '''
        Returns the state reached from given state and action
        '''
        ...

    @abstractmethod
    def step(self, action):
        '''
        Takes the given action, updates current state and returns the emission
        '''
        ...

    @abstractmethod
    def convert(self, state):
        '''
        Takes the given action, updates current state and returns the emission
        '''
        ...


    @abstractmethod
    def get_transition_matrix(self) -> torch.Tensor:
        '''
        Returns the transition matrix P(s'|s,a)
        '''
        ...

    @abstractmethod
    def is_valid_action(self, action, state) -> bool:
        ...

    @abstractmethod
    def reset(self) -> None:
        self.state = self.init_state
        self.h = 0
