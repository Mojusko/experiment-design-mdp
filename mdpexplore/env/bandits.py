import numpy as np
from abc import ABC, abstractmethod

from mdpexplore.env.discrete_env import DiscreteEnv


class Bandits(DiscreteEnv, ABC):

    def __init__(self, action_space: np.array, theta_star: np.array, sigma: float) -> None:
        super().__init__(init_state=0)
        self.action_space = action_space
        self.theta_star = theta_star
        self.sigma = sigma
        # TODO: For backward compatibility we treat actions as states for now,
        #       it would be better to make the code support space actions everywhere
        self.states_num = action_space.shape[0]
        self.actions_num = action_space.shape[0]
        # Emissions are features
        self.emissions = self.action_space
        self.max_episode_length = 1
        self.terminal_state = None
        self.visitations = np.zeros(self.states_num)

    def available_actions(self, state):
        return list(range(self.actions_num))

    def next(self, state, action):
        return action

    def step(self, action: int):
        self.visitations[action] += 1
        return action

    def convert(self, state):
        pass

    def get_transition_matrix(self) -> np.ndarray:
        return np.array([1.])

    def is_valid_action(self, action, state) -> bool:
        return action < self.action_num

    def reset(self) -> None:
        self.state = self.init_state

class Bandits_Left_Right(DiscreteEnv, ABC):

    def __init__(self, action_space: np.array, action_space_pre_embedding: np.array, theta_star: np.array, sigma: float, discount_factor: float = 0.99) -> None:
        super().__init__(init_state=0)
        self.action_space = action_space
        self.action_space_pre_embedding = action_space_pre_embedding
        self.theta_star = theta_star
        self.sigma = sigma
        # TODO: For backward compatibility we treat actions as states for now,
        #       it would be better to make the code support space actions everywhere
        self.states_num = action_space.shape[0]
        self.actions_num = action_space.shape[0]
        # Emissions are features
        self.emissions = self.action_space
        self.max_episode_length = 5
        self.terminal_state = None
        self.visitations = np.zeros(self.states_num)
        # initialize the transition matrix
        self.transition_matrix = None
        # set the discount factor
        self.discount_factor = discount_factor
        # whether the environment is constrained or not
        self.constrained = False
        self.stationary = False

    def available_actions(self, state):
        return [action for action, _ in enumerate(self.action_space) if self.is_valid_action(action, state)]

    def next(self, state, action):
        return action

    def step(self, action: int):
        self.state = self.next(self.state, action)
        self.visitations[action] += 1
        return action

    def convert(self, state):
        pass

    def get_transition_matrix(self) -> np.ndarray:
        # transition matrix get initialized in the first call to this function
        if self.transition_matrix is not None:
            return self.transition_matrix
        
        P = np.zeros((self.states_num, self.actions_num, self.states_num))

        for s in range(self.states_num):
            for a in range(self.actions_num):
                if self.is_valid_action(a, s):
                    s_next = self.next(s, a)
                    P[s, a, s_next] = 1.0
        
        self.transition_matrix = P
        return P

    def is_valid_action(self, action, state) -> bool:
        current_x = self.action_space_pre_embedding[state]
        action_x = self.action_space_pre_embedding[action]
        # we alternate between sampling in <= 0 and >= 0
        if current_x <= 0:
            return action_x > 0
        else:
            return action_x <= 0

    def reset(self) -> None:
        self.state = self.init_state