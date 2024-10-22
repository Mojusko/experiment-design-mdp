from typing import List, Tuple, Union, Any, Callable
import autograd.numpy as np
from numpy.random import default_rng
from abc import ABC, abstractmethod
import torch
from doexpy.env.discrete_env import DiscreteEnv


class TimeChain(DiscreteEnv, ABC):

    def __init__(
            self,
            embed: Callable,
            time_period: int = 100,
            max_events: int = 5,
            min_event_distance: int = 2,
            max_episode_length: int = 50,
            discount_factor: float = 0.99,
            seed: int = None,
            dt: float = 0.1,
            sigma: float = 1.0,
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
        """
        super().__init__(0)

        self.rng = default_rng(seed)
        self.sigma = sigma
        self.dt = dt
        self.embed = embed
        self.time_period = time_period
        self.max_events = max_events
        self.min_event_distance = min_event_distance
        self.states_num = self.time_period *  self.max_events * (self.min_event_distance+1)
        self.emiss_num = self.states_num

        self.terminal_state = self.convert_from_grid((511,max_events,min_event_distance))
        self.init_state = self.convert_from_grid((0,0,min_event_distance))


        self.actions = [0,1]
        self.actions_num = len(self.actions)

        self.max_episode_length = max_episode_length
        self.discount_factor = discount_factor
        self.emissions = {}

        self._generate_emissions()

        # tracking to plot later
        self.visitations = torch.zeros(size = (1, self.states_num, 1), dtype = torch.float64)
        self.visitations[0, self.init_state,0] = 1

        # to be initialized when get_transition_matrix is first called
        self.transition_matrix = None
        self.constrained = False
        
    def get_dim(self):
        return self.dim

    def get_states_num(self):
        return self.states_num

    def _generate_emissions(self):

        for i in range(self.states_num):
            time, count ,distance = self.convert_to_grid(i)
            if distance == 0:
                t = torch.Tensor([self.dt*time]).view(1,1).double()
                self.emissions[i] = self.embed(t).view(-1)
            else:
                self.emissions[i] = self.embed(t).view(-1)*0
        self.dim = self.emissions[0].size()[0]

    def convert(self, state: int):
        return self.convert_to_grid(state)

    def convert_to_grid(self, index: int) -> Tuple[int, int, int]:
        """Converts a state ID to grid coordinates

        Args:
            index (int): state ID
        Returns:
            Tuple[int, int, int]: grid coordinates of the state
        """
        event_distance = index // (self.time_period * self.max_events)
        events = (index % (self.time_period * self.max_events))//self.time_period
        time = (index % (self.time_period * self.max_events)) % self.time_period
        return time, events, event_distance

    def convert_from_grid(self, coordinates: Tuple[int, int, int]) -> int:
        """Converts grid coordinates to a state ID

        Args:
            coordinates (Tuple[int, int]): grid coordinates

        Returns:
            int: state ID of the given grid coordinates
        """
        return coordinates[0] + coordinates[1] * self.time_period + coordinates[2] * self.time_period * self.max_events

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
        act = self.actions[action]
        time, count, distance = self.convert_to_grid(state)
        newtime = min(time + 1, self.time_period-1)
        count = count + act
        if act == 1:
            distance = 0
        else:
            distance = min(distance + (newtime-time),self.min_event_distance)
        return self.convert_from_grid((newtime, count, distance))

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

        P = torch.zeros(size = (self.states_num, self.actions_num, self.states_num), dtype = torch.float64)
        for s in range(self.states_num):
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
        time, count, distance = self.convert_to_grid(state)
        if act == 1:
            if count < self.max_events-1 and distance >= self.min_event_distance:
                return True
            else:
                return False
        else:
            return True

    def reset(self) -> None:
        """Resets the environment to its initial state
        """
        self.visitations = torch.zeros(self.states_num, dtype=torch.float64)
        self.visitations[self.init_state] = 1
        super().reset()
