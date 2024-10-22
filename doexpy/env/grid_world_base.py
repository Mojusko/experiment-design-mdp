import torch 
from typing import List, Tuple, Union, Any
from numpy.random import default_rng
from abc import ABC, abstractmethod

from doexpy.env.discrete_env import DiscreteEnv

class DeterministicGridWorldBase(DiscreteEnv, ABC):

    @abstractmethod
    def __init__(
        self,
        init_state: int = 0,
        width: int = 7,
        height: int = 6,
        max_episode_length: int = 50,
        discount_factor: float = 0.99,
        max_sectors_num: int = 10,
        seed: int = None,
        sigma: float = 1.0,
        teleport: int = None,
        constrained: bool = False,
        terminal_state: int = None,
        ) -> None:
        """GridWorld class, representing a width*height environment.

        Args:
            init_state (int, optional): initial state. Defaults to 0.
            width (int, optional): width of the environment. Defaults to 7.
            height (int, optional): height of the environment. Defaults to 6.
            max_episode_length (int, optional): maximum episode length. Defaults to 50.
            discount_factor (float, optional): discount factor on the reward in the environment. Defaults to 0.99.
            max_sectors_num (int, optional): maximum number of randomly generated sectors. Defaults to 10.
            seed (int, optional): random seed. Defaults to None.
            sigma (float, optional): standard deviation of noise oracle.
            teleport (int, optional): if not None, the chosen state is a teleport back to the initial state, bypassing non-ergodicity otherwise. Defaults to None.
            constrained (bool, optional): if True, the environment is constrained (the agent must get to the terminal state at the end of the episode). Defaults to False.
            terminal_state (int, optional): if not None, sets the state in which the agent must find itself at the end of the episode. Defaults to None.
        """

        super().__init__(init_state)
        self.rng = default_rng(seed)
        self.sigma = sigma
        self.width = width
        self.height = height
        self.states_num = self.width * self.height
        
        if terminal_state is None:
            self.terminal_state = self.states_num - 1
        else:
            self.terminal_state = terminal_state
        
        self.actions_num = len(self.actions)

        self.max_episode_length = max_episode_length
        self.discount_factor = discount_factor

        self.max_sectors_num = max_sectors_num
        self.emissions = {}

        self._generate_emissions()

        # tracking to plot later
        self.visitations = torch.zeros(self.states_num, dtype = torch.float64)
        self.visitations[self.init_state] = 1

        # to be initialized when get_transition_matrix is first called
        self.transition_matrix = None

        self.teleport = teleport

        self.constrained = constrained

    def get_dim(self):
        return self.dim

    def get_emiss_dim(self):
        return self.emiss_num

    def get_states_num(self):
        return self.states_num

    def _generate_emissions(self):
        vectors = torch.eye(self.max_sectors_num).double()
        sector_ids = self._generate_sector_ids(self.max_sectors_num)

        for i, ids in enumerate(sector_ids):
            for idx in ids:
                self.emissions[idx] = vectors[i]
        
        self.dim = self.max_sectors_num
        self.theta = self.rng.random(self.max_sectors_num)
        self.emiss_num = len(self.emissions.values())

    def _generate_sector_ids(self, sectors_num: int) -> List[List[int]]:
        """Generates a list of sector ID allocations

        Args:
            sectors_num (int): number of sectors to generate

        Returns:
            List[List[int]]: list of sector ID allocations, i.e. 0th entry contains state IDs included in sector 0.
        """        
        sector_ids = []
        for _ in range(sectors_num):
            sector_ids.append([])

        states = set(range(self.states_num))

        sector = None

        for s in range(self.states_num):
            if s in states:
                if sector is None:
                    sector = 0
                else:
                    choices = list(range(sectors_num)); choices.remove(sector)
                    sector = self.rng.choice(choices)
                grid_coordinates = self.convert_to_grid(s)
                
                sector_width = self.rng.integers(2, 4)
                sector_height = self.rng.integers(2, 4)

                for i in range(sector_width):
                    for j in range(sector_height):
                        candidate = self.convert_from_grid((grid_coordinates[0] + i, grid_coordinates[1] + j))
                        if candidate in states:
                            sector_ids[sector].append(candidate)
                            states.remove(candidate)

        return sector_ids

    def convert(self, state:int):
        return self.convert_to_grid(state)


    def convert_to_grid(self, index: int) -> Tuple[int, int]:
        """Converts a state ID to grid coordinates

        Args:
            index (int): state ID

        Returns:
            Tuple[int, int]: grid coordinates of the state
        """        
        return (index % self.width, index // self.width)

    def convert_from_grid(self, coordinates: Tuple[int, int]) -> int:
        """Converts grid coordinates to a state ID

        Args:
            coordinates (Tuple[int, int]): grid coordinates

        Returns:
            int: state ID of the given grid coordinates
        """        
        return coordinates[0] + coordinates[1]*self.width

    def available_actions(self, state: int) -> List[int]:
        """Returns available actions at the given state

        Args:
            state (int): state ID

        Returns:
            List[int]: list of available actions
        """        
        return [action for action in self.actions if self.is_valid_action(action, state)]

    def next(self, state: int, action: int) -> int:
        """Returns the next state after taking the given action from the given state

        Args:
            state (int): current state ID
            action (int): action ID

        Returns:
            int: state ID after taking the given action from the given state
        """        
        if state == self.teleport:
            return self.init_state

        act = self.actions[action]
        s = self.convert_to_grid(state)

        return self.convert_from_grid((s[0] + act[0], s[1] + act[1]))

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

    def get_transition_matrix(self) -> torch.Tensor:
        """Returns the transition matrix P(s'|s,a)

        Returns:
            np.ndarray: transition matrix
        """        
        if self.transition_matrix is not None:
            return self.transition_matrix
        
        P = torch.zeros((self.states_num, self.actions_num, self.states_num), dtype = torch.float64)
        for s in range(self.states_num):
            
            if s == self.teleport:
                for a in range(self.actions_num):
                    if self.is_valid_action(a, s):
                        P[s, a, self.init_state] = 1.0

            else:
                for a in range(self.actions_num):
                    if self.is_valid_action(a, s):
                        s_next = self.next(s, a)
                        P[s, a, s_next] = 1.0
        
        self.transition_matrix = P
        return P

    def is_valid_action(self, action: int, state: int) -> bool:
        """Checks if the given action is valid in the given state

        Args:
            action (int): action ID
            state (int): state ID

        Returns:
            bool: True if the given action is valid in the given state
        """        
        if action not in self.actions:
            return False
        act = self.actions[action]
        if act == (0, 0):
            return state == self.terminal_state
        state = self.convert_to_grid(state)
        return (
            state[0] + act[0] >= 0 and
            state[0] + act[0] < self.width and
            state[1] + act[1] >= 0 and
            state[1] + act[1] < self.height
        )

    def reset(self) -> None:
        """Resets the environment to its initial state
        """        
        self.visitations = torch.zeros(self.states_num, dtype=torch.float64)
        self.visitations[self.init_state] = 1
        super().reset()
