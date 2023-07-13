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
        self.visitations = np.ones(self.states_num)
        self.visitations[self.init_state] += 1

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
