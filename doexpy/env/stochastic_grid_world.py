from doexpy.env.stochastic_grid_world_base import StochasticGridWorldBase
from typing import List
import numpy as np


class StochasticGridWorld(StochasticGridWorldBase):
    def __init__(
            self,
            init_state: int = 0,
            prob: float = 0.1,
            max_episode_length: int = 10,
            terminal_state: int = None,
            discount_factor=0.99,
            width: int = 10,
            height: int = 10,
            seed=None,
            max_sectors_num: int = 5
    ):
        self.prob = prob

        self.actions = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, -1)}
        super().__init__(
            init_state=init_state,
            width=width,
            height=height,
            max_episode_length=max_episode_length,
            discount_factor=discount_factor,
            max_sectors_num=max_sectors_num,
            seed=seed,
            terminal_state=terminal_state
        )

    def next(self, state: int, action: int) -> int:
        act = self.actions[action]
        s = self.convert_to_grid(state)

        coin_toss = np.random.uniform()
        if coin_toss > self.prob:
            return self.convert_from_grid((s[0] + act[0], s[1] + act[1]))
        else:
            valid_actions = []
            for a in self.available_actions(state):
                if a != action:
                    valid_actions.append(a)

            if len(valid_actions) == 0:
                return self.convert_from_grid((s[0] + act[0], s[1] + act[1]))

            else:
                next_a = np.random.choice(valid_actions, size=1)
                act = self.actions[next_a[0]]
                return self.convert_from_grid((s[0] + act[0], s[1] + act[1]))

    def p_next(self, state: int, action: int) -> dict:
        act = self.actions[action]
        s = self.convert_to_grid(state)

        new_state = (s[0] + act[0], s[1] + act[1])
        if len(self.available_actions(state)) == 1 and action == self.available_actions(state)[0]:
            probs = {self.convert_from_grid(new_state): 1.0}
        else:
            probs = {self.convert_from_grid(new_state): 1 - self.prob}
            number_of_alternative_actions = len(self.available_actions(state)) - 1
            for a in self.available_actions(state):
                if a != action:
                    new_act = self.actions[a]
                    new_state = (s[0] + new_act[0], s[1] + new_act[1])
                    probs[self.convert_from_grid(new_state)] = self.prob / number_of_alternative_actions
        return probs

    def is_valid_action(self, action: int, state: int) -> bool:
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


class StochasticDummyGridWorld(StochasticGridWorld):
    def __init__(
            self,
            prob: float = 0.1,
            max_episode_length: int = 20,
            terminal_state: int = None,
    ):
        super().__init__(
            prob = prob,
            init_state=0,
            width=7,
            height=6,
            max_episode_length=max_episode_length,
            discount_factor=0.99,
            max_sectors_num=5,
            seed=None,
            terminal_state=terminal_state
        )

    def _generate_sector_ids(self, sectors_num: int) -> List[List[int]]:
        return [
            [0, 1, 2, 7, 8, 9, 32, 33, 34, 39, 40, 41],
            [4, 5, 6, 11, 12, 13, 21, 22, 23, 28, 29, 30, 35, 36, 37],
            [3, 10, 14, 15, 16],
            [17, 24, 31, 38],
            [18, 19, 20, 25, 26, 27]
        ]
