from mdpexplore.env.grid_worlds import DeterministicGridWorldBase
import numpy as np
from stpy.helpers.helper import interval
class GlobalTransGridWorld(DeterministicGridWorldBase):

    def __init__(self, size, max_episode_length):
        no_states = size*size

        self.actions = dict([(j,j) for j in range(no_states)])
        self.m = 100
        self.dim = self.m

        super().__init__(
            init_state=0,
            width=size,
            height=size,
            max_episode_length=max_episode_length,
            discount_factor=0.99,
            max_sectors_num=10,
            seed=None,
            teleport=None,
            constrained=False,
            terminal_state=None
        )

        self.action_space_pre_embedding = interval(size, 2, L_infinity_ball= 0.5)
        self.action_space = self._generate_emissions()

    def _generate_emissions(self):
        vectors = np.eye(self.m)
        for i in range(self.states_num):
            self.emissions[i] = vectors[i % self.m]
        self.theta = self.rng.random(self.max_sectors_num)
        return self.emissions
    def next(self, state: int, action: int) -> int:
        act = self.actions[action]
        return act

    def p_next(self, state: int, action: int) -> dict:
        act = self.actions[action]
        probs = {state: act}
        return probs

    def is_valid_action(self, action: int, state: int) -> bool:
        if action not in self.actions:
            return False
        return True
