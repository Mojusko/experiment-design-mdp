import torch 
from abc import ABC, abstractmethod

from doexpy.env.discrete_env import DiscreteEnv


class Bandits(DiscreteEnv, ABC):

    def __init__(self, action_space:  torch.Tensor) -> None:
        super().__init__(init_state=0)

        self.action_space = action_space

        self.states_num = 1
        self.actions_num = action_space.size()[0]
        
        # Emissions are features
        self.emissions = self.action_space
        self.max_episode_length = 1
        self.emiss_num = self.actions_num
        self.terminal_state = None
        self.transition_matrix = None
        self.h = 0 
        self.visitations = torch.zeros((self.states_num, self.actions_num), dtype = torch.float64)
        self.constrained = False

    def available_actions(self, state):
        return list(range(self.actions_num))

    def convert(self, state):
        return super().convert(state)

    def next(self, state, action):
        return state

    def step(self, action: int):
        self.state = self.next(self.state, action)
        self.visitations[self.state,action] += 1
        self.h += 1
        return action

    def get_transition_matrix(self) -> torch.Tensor:
        if self.transition_matrix is not None:
            return self.transition_matrix      
        P = torch.ones((1, self.actions_num, 1), dtype = torch.float64)
        self.transition_matrix = P
        return P
    
    def is_valid_action(self, action, state) -> bool:
        return action < self.action_num

    def reset(self) -> None:
        self.state = self.init_state

class MovementConstrainedBayesianOptimization(DiscreteEnv, ABC):
    def __init__(self, action_space: torch.Tensor, action_space_pre_embedding: torch.Tensor, theta_star: torch.Tensor, sigma: float, discount_factor: float = 0.99, max_episode_length:int = 10, init_state:int = 0) -> None:
        super().__init__(init_state=init_state)
        self.action_space = action_space
        self.action_space_pre_embedding = action_space_pre_embedding
        self.theta_star = theta_star
        self.sigma = sigma
        self.states_num = action_space.size()[0]
        self.actions_num = action_space.size()[0]
        # Emissions are features
        self.emissions = self.action_space
        self.max_episode_length = max_episode_length
        self.terminal_state = None
        self.visitations =torch.zeros(self.states_num, dtype = torch.float64)
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
        self.h += 1
        return action

    def convert(self, state):
        pass

    def get_transition_matrix(self) -> torch.Tensor:
        # transition matrix get initialized in the first call to this function
        if self.transition_matrix is not None:
            return self.transition_matrix
        
        P = torch.zeros((self.states_num, self.actions_num, self.states_num), dtype=torch.float64)

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
        self.h = 0
    
    def get_dim(self):
        return self.action_space.size()[1]
    
class Bandits_Left_Right(MovementConstrainedBayesianOptimization, ABC):
    def __init__(self, action_space: torch.Tensor, action_space_pre_embedding: torch.Tensor, theta_star: torch.Tensor, sigma: float, discount_factor: float = 0.99) -> None:
        super().__init__(action_space, action_space_pre_embedding, theta_star, sigma, discount_factor)
    
    def is_valid_action(self, action, state) -> bool:
        current_x = self.action_space_pre_embedding[state]
        action_x = self.action_space_pre_embedding[action]
        # we alternate between sampling in <= 0 and >= 0
        if current_x <= 0:
            return action_x > 0
        else:
            return action_x <= 0

class ConstrainedMaxMovement(MovementConstrainedBayesianOptimization, ABC):
    def __init__(self, action_space: torch.Tensor, action_space_pre_embedding: torch.Tensor, theta_star: torch.Tensor, sigma: float, discount_factor: float = 0.99, max_episode_length:int = 10, delta: float = 0.1, init_state: int = 0) -> None:
        super().__init__(action_space, action_space_pre_embedding, theta_star, sigma, discount_factor, max_episode_length, init_state)
        self.delta = delta
        # self.constrained = True
        # self.terminal_state = self.states_num - 1

    def is_valid_action(self, action, state) -> bool:
        current_x = self.action_space_pre_embedding[state]
        action_x = self.action_space_pre_embedding[action]
        # calculate the distance between the current state and the action
        distance = torch.linalg.norm(current_x - action_x)
        return distance <= self.delta