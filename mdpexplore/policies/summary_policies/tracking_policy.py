from typing import List
import numpy as np
from mdpexplore.policies.policy_base import Policy, SummarizedPolicy
from mdpexplore.env.discrete_env import DiscreteEnv


class TrackingPolicy(SummarizedPolicy):
    def __init__(self, env: DiscreteEnv, ps: List[Policy], weights: List[float], empirical_counts: np.array ) -> None:
        super().__init__(env)
        self.ps = ps
        self.p_weights = weights

        # normalize
        weights = np.array(weights)
        if np.sum(empirical_counts) > 1e-5:
            empirical_prob = empirical_counts/np.sum(empirical_counts)
        else:
            empirical_prob = empirical_counts

        self.policy_id = np.argmax(weights - empirical_prob)
        #self.policy_id = self.rng.choice(np.arange(0,len(self.ps),1), p = self.p_weights)
        self.policy_picked = self.ps[self.policy_id]

    def get_picked_policy_id(self):
        return self.policy_id

    def next_action(self, state):
        return self.policy_picked.next_action(state)