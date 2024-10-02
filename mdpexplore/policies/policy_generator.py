import torch 

from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.env.linear_system import ContinuousEnv

from mdpexplore.policies.base_policies.stationary_policy import StationaryPolicy
from mdpexplore.policies.base_policies.non_stationary_policy import NonStationaryPolicy
from mdpexplore.policies.base_policies.linear_policy import LinearPolicy

class PolicyGenerator():
    # TODO: add a constrained policy?
    def __init__(self, env: DiscreteEnv) -> None:
        self.env = env

    def uniform_policy(self, stationary = False):
        '''
        Returns a uniform policy within the environment
        '''
        if stationary:
            p = torch.ones(size = (self.env.states_num, self.env.actions_num), dtype = torch.float64)
            for s in range(self.env.states_num):
                for a in range(self.env.actions_num):
                    if not self.env.is_valid_action(a, s):
                        p_h[s, a] = 0

            return StationaryPolicy(self.env, p)
        
        else:
            p = torch.ones((self.env.max_episode_length, self.env.states_num, self.env.actions_num),dtype=torch.float64)
            for h in range(self.env.max_episode_length):

                p_h = torch.ones((self.env.states_num, self.env.actions_num),dtype=torch.float64)
                for s in range(self.env.states_num):
                    for a in range(self.env.actions_num):
                        if not self.env.is_valid_action(a, s):
                            p_h[s, a] = 0

                p_h /= torch.sum(p_h, dim=1, keepdims=True)
                p[h] = p_h

            return NonStationaryPolicy(self.env, p)

class ContinuousPolicyGenerator():

    def __init__(self, env: ContinuousEnv) -> None:
        self.env = env

    def uniform_policy(self):
        '''
        Returns a uniform policy within the environment
        '''
        d = self.env.state_dim
        K = torch.randn(d,d, dtype=torch.float64)
        return LinearPolicy(self.env, K)
