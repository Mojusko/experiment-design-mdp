import autograd.numpy as np
from abc import ABC, abstractmethod

class Environment(ABC):

    def __init__(self) -> None:
        pass

class DiscreteEnv(Environment):
    def __init__(self, init_state) -> None:
        super().__init__()
        self.type = 'discrete'
        self.init_state = init_state
        self.state = init_state
        self.states_num = None
        self.actions_num = None
        self.visitations = None
        self.max_episode_length = None
    
    @abstractmethod
    def available_actions(self, state):
        '''
        Returns available actions at current state
        '''
        ...

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
    def get_transition_matrix(self) -> np.ndarray:
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
