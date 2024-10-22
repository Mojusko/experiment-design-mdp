from typing import List
import numpy as np 
from doexpy.env.grid_world_base import DeterministicGridWorldBase
from doexpy.env.stochastic_grid_world_base import StochasticGridWorldBase

class ErgodicGridWorld(DeterministicGridWorldBase):
	def __init__(
		self,
		init_state: int = 0,
		width: int = 7,
		height: int = 6,
		max_episode_length: int = 50,
		discount_factor: float = 0.99,
		max_sectors_num: int = 10,
		seed: int = None,
		teleport: int = None,
		constrained: bool = False,
		terminal_state: int = None
	) -> None:

		self.actions = {0:(0,0), 1:(1,0), 2:(0,1), 3:(-1,0), 4:(0,-1)}
		super().__init__(
			init_state=init_state,
			width=width,
			height=height,
			max_episode_length=max_episode_length,
			discount_factor=discount_factor,
			max_sectors_num=max_sectors_num,
			seed=seed,
			teleport=teleport,
			constrained=constrained,
			terminal_state=terminal_state
		)
		if not constrained:
			self.terminal_state = None

class NonErgodicGridWorld(DeterministicGridWorldBase):
	def __init__(
		self,
		init_state: int = 0,
		width: int = 7,
		height: int = 6,
		max_episode_length: int = 50,
		discount_factor: float = 0.99,
		max_sectors_num: int = 10,
		seed: int = None,
		teleport: int = None,
		constrained: bool = False,
		terminal_state: int = None
	) -> None:

		self.actions = {0:(0,0), 1:(1,0), 2:(0,1)}
		super().__init__(
			init_state=init_state,
			width=width,
			height=height,
			max_episode_length=max_episode_length,
			discount_factor=discount_factor,
			max_sectors_num=max_sectors_num,
			seed=seed,
			teleport=teleport,
			constrained=constrained,
			terminal_state=terminal_state
		)

		self.max_episode_length = min(self.max_episode_length, width + height - 2)


class DummyGridWorld(NonErgodicGridWorld):
	def __init__(
		self,
		max_episode_length: int = 20,
		teleport: int = None,
		constrained: bool = False,
		terminal_state: int = None,
	):
		super().__init__(
			init_state=0,
			width=7,
			height=6,
			max_episode_length=max_episode_length,
			discount_factor=0.99,
			max_sectors_num=6,
			seed=None,
			teleport=teleport,
			constrained=constrained,
			terminal_state=terminal_state
		)

	def _generate_sector_ids(self, sectors_num: int) -> List[List[int]]:
		return [
				[0, 1, 2, 7, 8, 9, 32, 33, 34, 39, 40, 41],
				[21, 22, 23, 28, 29, 30, 35, 36, 37],
				[3, 10, 14, 15, 16],
				[17, 24, 31, 38],
				[18, 19, 20, 25, 26, 27],
				[4, 5, 6, 11, 12, 13],
			]


class TunnelGridWorld(NonErgodicGridWorld):
	def __init__(
		self,
		width: int = 10,
		max_episode_length: int = 20,
	):
		super().__init__(
			init_state=0,
			width=width,
			height=3,
			max_episode_length=max_episode_length,
			discount_factor=0.99,
			max_sectors_num=width+1,
			seed=None,
			teleport=None,
			constrained=None,
			terminal_state=None
		)

	def _generate_sector_ids(self, sectors_num: int) -> List[List[int]]:
		tunnels = sectors_num - 1
		ret = [list(range(0, tunnels)) + list(range(2*tunnels, 3*tunnels))]
		for i in range (tunnels, 2*tunnels):
			ret.append([i])
		return ret

	def is_valid_action(self, action: int, state: int) -> bool:
		s = self.convert_to_grid(state)
		if s[1] == 1:
			return self.actions[action] == (0, 1)
		return super().is_valid_action(action, state)


class StochasticDummyGridWorld(StochasticGridWorldBase):
	def __init__(
		self,
		max_episode_length: int = 20,
		teleport: int = None,
		constrained: bool = False,
		terminal_state: int = None,
		width  : int  = 6,
		height : int = 7
	):
		
		self.actions = {0:'V', 1:'H', 2:'S'}
		super().__init__(
			init_state=0,
			width=width,
			height=height,
			max_episode_length=max_episode_length,
			discount_factor=0.99,
			max_sectors_num=5,
			seed=None,
			teleport=teleport,
			constrained=constrained,
			terminal_state=terminal_state
		)
		# action to coresponding to vertical movement, horizontal movement and


	def _generate_sector_ids(self, sectors_num: int) -> List[List[int]]:
		return [
				[0, 1, 2, 7, 8, 9, 32, 33, 34, 39, 40, 41],
				[4, 5, 6, 11, 12, 13, 21, 22, 23, 28, 29, 30, 35, 36, 37],
				[3, 10, 14, 15, 16],
				[17, 24, 31, 38],
				[18, 19, 20, 25, 26, 27]
			]


	def next(self, state: int, action: int) -> int:
		act = self.actions[action]
		s = self.convert_to_grid(state)
		
		if act == 'V':
			a0 = 0
			a1 = 1 if np.random.randn() > 0 else -1 
		elif act == 'H':
			a0 = 1 if np.random.randn() > 0 else -1 
			a1 = 0
		elif act == 'S':
			a1 = 0 
			a0 = 0 
		else:
			a1 = 0 
			a0 = 0 
		return self.convert_from_grid(
			(  max(0,min(s[0] + a0,self.width-1)), max(0,min(s[1] + a1,self.height-1))   )
		)
	

	def p_next(self, state:int, action:int)->dict:
		act = self.actions[action]
		s = self.convert_to_grid(state)
		
		if act == 'H':
			if s[0] == self.width - 1:
				new_state = self.convert_from_grid((s[0] -1, s[1]))
				probs={new_state:1.}
			elif s[0] == 0:
				new_state = self.convert_from_grid((s[0] +1, s[1]))
				probs={new_state:1.}
			else:
				new_state1 = self.convert_from_grid((s[0] -1, s[1]))
				new_state2 = self.convert_from_grid((s[0] +1, s[1]))
				probs={new_state1:0.5, new_state2:0.5}
		
		elif act == 'V':
			if s[1] == self.height - 1:
				new_state = self.convert_from_grid((s[0], s[1]-1))
				probs={new_state:1.}
			elif s[1] == 0:
				new_state = self.convert_from_grid((s[0], s[1]+1))
				probs={new_state:1.}

			else:
				new_state1 = self.convert_from_grid((s[0] , s[1]-1))
				new_state2 = self.convert_from_grid((s[0] , s[1]+1))
				probs={new_state1:0.5, new_state2:0.5}
		elif act == 'S':
			probs={state:1.}

		return probs 

	def is_valid_action(self, action: int, state: int) -> bool:
		if action not in self.actions:
			return False
		return True 
