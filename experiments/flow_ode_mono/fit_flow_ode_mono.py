import numpy as np
from scipy.integrate import odeint
import torch
from doexpy.env.bandits import MovementConstrainedBayesianOptimization

# define the environment class
class ODEMonoEnv(MovementConstrainedBayesianOptimization):
    def __init__(self,
                 action_space: np.array, 
                 action_space_pre_embedding: np.array, 
                 theta_star: np.array, sigma: float, 
                 discount_factor: float = 0.99, 
                 max_episode_length: int = 10,
                 delta: float = 0.1,
                 init_state: int = 0,
                 terminal_state: int = None) -> None:
        
        super().__init__(action_space, action_space_pre_embedding, theta_star, sigma, discount_factor, max_episode_length, init_state)
        self.delta = delta
        if terminal_state is None:
            self.constrained = False
        else:
            self.constrained = True
            self.terminal_state = terminal_state
    
    def is_valid_action(self, action, state) -> bool:
        current_state = self.action_space_pre_embedding[state]
        next_state = self.action_space_pre_embedding[action]

        # check that the current distance is less than delta_mov in each dimension
        if np.abs(current_state[0] - next_state[0]) > self.delta:
            return False
        if np.abs(current_state[1] - next_state[1]) > self.delta:
            return False
        
        # finally check that the residence time is increasing monotically
        if current_state[0] > next_state[0]:
            return False

        return True