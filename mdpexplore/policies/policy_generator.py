import autograd.numpy as np

from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.env.linear_system import ContinuousEnv

from mdpexplore.policies.base_policies.stationary_policy import StationaryPolicy
from mdpexplore.policies.base_policies.linear_policy import LinearPolicy

class PolicyGenerator():
    # TODO: add a constrained policy?
    def __init__(self, env: DiscreteEnv) -> None:
        self.env = env

    def uniform_policy(self):
        '''
        Returns a uniform policy within the environment
        '''
        p = np.ones((self.env.states_num, self.env.actions_num))
        for s in range(self.env.states_num):
            for a in range(self.env.actions_num):
                if not self.env.is_valid_action(a, s):
                    p[s, a] = 0
        p /= np.sum(p, axis=1, keepdims=True)

        return StationaryPolicy(self.env, p)

class ContinuousPolicyGenerator():

    def __init__(self, env: ContinuousEnv) -> None:
        self.env = env

    def uniform_policy(self):
        '''
        Returns a uniform policy within the environment
        '''
        d = self.env.state_dim
        K = np.random.randn(d,d)
        return LinearPolicy(self.env, K)
