from mdpexplore.env.grid_world_base import DeterministicGridWorldBase
import numpy as np 
from abc import ABC, abstractmethod
import torch 

class StochasticGridWorldBase(DeterministicGridWorldBase, ABC):

	def __init__(self, init_state: int = 0,
				 width: int = 7,
				 height: int = 6,
				 max_episode_length: int = 50,
				 discount_factor: float = 0.99, max_sectors_num: int = 10,
				 seed: int = None,
				 teleport: int = None,
				 constrained: bool = False,
				 terminal_state: int = None) -> None:
		super().__init__(init_state, width, height, max_episode_length, discount_factor, max_sectors_num, seed, teleport, constrained, terminal_state)
	
		
	def get_transition_matrix(self) -> torch.Tensor:
		if self.transition_matrix is not None:
			return self.transition_matrix
		
		P = torch.zeros(size= (self.states_num, self.actions_num, self.states_num), dtype = torch.float64)
		
		for s in range(self.states_num):

			for a in range(self.actions_num):
				if self.is_valid_action(a, s):
					probs = self.p_next(s, a)
					for s_state in probs.keys():
						P[s, a, s_state] = probs[s_state]

		self.transition_matrix = P
		return P