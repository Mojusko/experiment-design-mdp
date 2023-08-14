import autograd.numpy as np

from mdpexplore.policies.policy_base import Policy
from mdpexplore.env.discrete_env import DiscreteEnv


class NonStationaryPolicy(Policy):
    def __init__(self, env: DiscreteEnv, ps: np.ndarray) -> None:
        self.ps = ps
        self.time = 0
        super().__init__(env)

    def next_action(self, state: int):
        state_policy = self.ps[self.time, state]
        actions = self.env.available_actions(state)
        reduced_state_policy = state_policy[actions]/np.sum(state_policy[actions])
        self.time += 1
        if self.time == self.env.max_episode_length:
            self._reset()
        return self.rng.choice(actions, p=reduced_state_policy)

    def _reset(self):
        self.time = 0