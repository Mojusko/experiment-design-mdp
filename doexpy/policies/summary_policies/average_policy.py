import autograd.numpy as np
from typing import List

from doexpy.env.discrete_env import DiscreteEnv
from doexpy.policies.policy_base import SummarizedPolicy
from doexpy.policies.base_policies.stationary_policy import StationaryPolicy


class AveragePolicy(SummarizedPolicy):
    def __init__(self, env: DiscreteEnv, ps: List[StationaryPolicy], weights: List[float]) -> None:
        super().__init__(env)
        self.ps = ps
        self.p_weights = weights

        p_avg = np.zeros((env.states_num, env.actions_num))
        for policy in self.ps:
            p_avg += policy.p
        p_avg /= len(self.ps)
        self.average_policy = StationaryPolicy(env, p_avg)

    def next_action(self, state):
        return self.average_policy.next_action(state)