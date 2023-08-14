from typing import List

from mdpexplore.policies.policy_base import Policy, SummarizedPolicy
from mdpexplore.env.discrete_env import DiscreteEnv


class MixturePolicy(SummarizedPolicy):
    def __init__(self, env: DiscreteEnv, ps: List[Policy], weights: List[float]) -> None:
        super().__init__(env)
        self.ps = ps
        self.p_weights = weights
        self.policy_picked = self.rng.choice(self.ps, p=self.p_weights)

    def next_action(self, state):
        return self.policy_picked.next_action(state)