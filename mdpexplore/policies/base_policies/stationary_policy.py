import autograd.numpy as np

from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.policies.policy_base import Policy


class StationaryPolicy(Policy):
    def __init__(self, env: DiscreteEnv, p: np.ndarray) -> None:
        self.p = p
        super().__init__(env)

    def next_action(self, state: int):
        state_policy = self.p[state]
        actions = self.env.available_actions(state)
        reduced_state_policy = state_policy[actions]/np.sum(state_policy[actions])
        return self.rng.choice(actions, p=reduced_state_policy)