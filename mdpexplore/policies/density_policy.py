import autograd.numpy as np

from mdpexplore.policies.policy_base import SummarizedPolicy
from mdpexplore.policies.simple_policy import SimplePolicy
from mdpexplore.policies.non_stationary_policy import NonStationaryPolicy
from mdpexplore.env.discrete_env import DiscreteEnv


class DensityPolicy(SummarizedPolicy):
    def __init__(self, env: DiscreteEnv, density: np.ndarray, density_sa: np.ndarray) -> None:
        super().__init__(env)
        self.density = density
        self.density_sa = density_sa

        # check density shape to determine if policy is stationary or not: density_sa has shape (S,A) or (H, S, A)
        if len(self.density_sa.shape) == 2:
            policy = np.zeros(shape = self.density_sa.shape)
            temp = np.tile(self.density.reshape(-1,1), (1,self.density_sa.shape[1]))
            mask = temp > 0
            policy[mask] = self.density_sa[mask] / temp[mask]
            self.policy = SimplePolicy(env, policy)
        # if non-stationary reshapings are different and we return a non-stationary policy
        elif len(self.density_sa.shape) == 3:
            policy = np.zeros(shape = self.density_sa.shape)
            temp = np.tile(np.expand_dims(self.density, -1), (1, 1, self.density_sa.shape[1]))
            mask = temp > 0
            policy[mask] = self.density_sa[mask] / temp[mask]
            self.policy = NonStationaryPolicy(env, policy)

    def next_action(self, state):
        return self.policy.next_action(state)

class ActionDensityPolicy(SummarizedPolicy):
    def __init__(self, env: DiscreteEnv, density: np.ndarray, density_sa: np.ndarray) -> None:
        super().__init__(env)
        self.density = density
        self.density_sa = density_sa

        # check density shape to determine if policy is stationary or not: density_sa has shape (S,A) or (H, S, A)
        if len(self.density_sa.shape) == 2:
            policy = np.zeros(shape = self.density_sa.shape)
            for state in range(self.density.shape[0]):
                mask = self.env.available_actions(state)
                denom = np.sum(self.density[mask])
                if denom > 0:
                    policy[state, mask] = self.density[mask] / np.sum(self.density[mask])
        
            self.policy = SimplePolicy(env, policy)

        # if non-stationary reshapings are different and we return a non-stationary policy
        elif len(self.density_sa.shape) == 3:
            policy = np.zeros(shape = self.density_sa.shape)
            for h in range(self.density_sa.shape[0] - 1):
                for state in range(self.density_sa.shape[1]):
                    # mask over available actions
                    mask = self.env.available_actions(state)
                    denom = np.sum(self.density[h + 1, mask])
                    if denom > 0:
                        policy[h, state, mask] = self.density[h + 1, mask] / denom
            
            # for final timestep, we set policy according to state-action density
            temp = np.tile(np.expand_dims(self.density[-1], -1), (1, 1, self.density_sa[-1].shape[1]))
            mask = (temp > 0)[0]
            policy[-1, mask] = self.density_sa[-1, mask] / temp[0][mask]
            
            self.policy = NonStationaryPolicy(env, policy)

    def next_action(self, state):
        return self.policy.next_action(state)