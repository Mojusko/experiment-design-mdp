from mdpexplore.env.discrete_env import DiscreteEnv
from typing import List
import numpy as np
import copy 

class StochasticGridParticles(DiscreteEnv):
    def __init__(
            self,
            no_particles: int = 2,
            init_state: int = 0,
            prob: float = 0.2,
            max_episode_length: int = 10,
            width: int = 100,
            seed=None
    ):
        self.prob = prob
        self.actions = {0: 1, 1: -1, 2: 0, 3: None} # left, right, stay
        self.t = 0
        self.no_particles = no_particles 
        self.width = 100
        self.max_episode_length = max_episode_length
        self.init_state = init_state


    def next(self, state: np.array, action: int) -> int:
        act = self.actions[action]
        s = state
        next_s = copy.deepcopy(state)

        for i in range(self.no_particles):
            if act is not None:
                coin_toss = np.random.uniform()
                if coin_toss > self.prob:
                    
                    next_s[i] = max(0,min(s[i] + act - 1,self.width))
                else:
                    next_s[i] = max(0,min(s[i] + 1 + act, self.width))
            else:
                pass 
        return next_s

    def is_valid_action(self, action: int, state: np.array, h: int) -> bool:
        
        if h % 2 == 0:
            if action is not None:
                return True
            else:
                return False
        else:
            if action is not None:
                return False
            else:
                return True
            
    def reset(self) -> None:
        """Resets the environment to its initial state
        """        
        self.visitations = np.zeros(self.states_num)
        self.visitations[self.init_state] = 1
        super().reset()


    def step(self, action: int) -> Tuple[Any, Union[float, Any]]:
        """Takes the given action, updates current state and returns the emission.

        Args:
            action (int): ID of action to be taken

        Returns:
            float: transformed noisy emission
        """        
        self.state = self.next(self.state, action)
        self.visitations[self.state] += 1
        return self.state