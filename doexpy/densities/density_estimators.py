from abc import ABC, abstractmethod
import numpy as np
import torch 

from doexpy.policies.policy_base import Policy
from doexpy.policies.base_policies.non_stationary_policy import NonStationaryPolicy, NonStationaryPolicyContinuous
from doexpy.policies.base_policies.stationary_policy import StationaryPolicy, StationaryPolicyContinuous

from doexpy.env.discrete_env import DiscreteEnv
from doexpy.functionals.reward_functional import RewardFunctional
from doexpy.env.continuous_env import ContinuousEnv

from doexpy.densities.continous_densities import SimpleDeltaDensity, NonStationaryDeltaDensity

class DensityEstimator(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def density_oracle_single(self, policy: Policy) -> np.ndarray:
        ...
    
    @abstractmethod
    def density_oracle(self, policy: Policy) -> np.ndarray:
        ...

class TabularDensity(DensityEstimator):
    def __init__(self, env:DiscreteEnv, objective:RewardFunctional) -> None:
        super().__init__()
        self.env = env
        self.objective = objective
    
    def density_oracle_single(self, policy: Policy) -> np.ndarray:
            """Computes state distribution induced by the given policy over a specified horizon

            Args:
                policy (Policy): inducing policy

            Returns:
                np.ndarray: S x A (stationary) or H x S x A (non-stationary) array with density for each state
            """

            if type(policy) is StationaryPolicy:

                v0 = torch.zeros(size = (self.env.states_num, self.env.actions_num), dtype = torch.float64)
                # initialize with the initial state and corresponding actions
                for act in self.env.available_actions(self.env.init_state):
                    v0[self.env.init_state, act] = policy.p[self.env.init_state, act]

                v = torch.Tensor(v0, dtype = torch.float64)
                temp = torch.Tensor(v0).sum(dim = -1)
                
                p_pi = (self.env.get_transition_matrix() * torch.unsqueeze(policy.ps[i], dim=2)).sum(dim = 1)

                assert (torch.allclose(p_pi.sum(dim=1), 1, rtol=1e-05, atol=1e-05))

                for _ in range(self.env.max_episode_length):
                    temp = p_pi.T @ temp
                    expanded_temp = torch.unsqueeze(temp, dim=-1)
                    v += expanded_temp * policy.p
                
                v = v / v.sum()

            elif type(policy) is NonStationaryPolicy:

                v0 = torch.zeros(size = (self.env.max_episode_length - self.env.h, self.env.states_num, self.env.actions_num), dtype = torch.float64)
                # initialize with the initial state and corresponding actions
                for act in self.env.available_actions(self.env.state):
                    v0[0, self.env.state, act] = policy.ps[0, self.env.state, act]

                v = torch.Tensor(v0)
                # get marginal state distribution for initial temp
                temp = torch.Tensor(v0)[0].sum(dim = -1)
                
                for i in range(self.env.max_episode_length - self.env.h - 1):
                    p_pi = (self.env.get_transition_matrix() * torch.unsqueeze(policy.ps[i], dim=2)).sum(dim = 1)
                    # assert (np.allclose(p_pi.sum(axis=1), 1, rtol=1e-05, atol=1e-05))
                    temp = p_pi.T @ temp
                    expanded_temp = torch.unsqueeze(temp, dim=-1)
                    v[i + 1] += expanded_temp * policy.ps[i + 1]
            return v

    def density_oracle(self, policies, weights, densities, stationary = False) -> torch.Tensor:
        """Computes the combined state (or state-action) distribution induced by the saved policies

        Args:
            actions (bool, optional): if True, computes state-action distribution instead of state distribution. Defaults to False.

        Returns:
            np.ndarray: 1-D array with density for each state

        Raises:
            TypeError: if the saved policies are non-stationary
        """

        # if the first policy is non-stationary, we define a total density with time index
        if stationary:
            total_density = torch.zeros(size = (self.env.states_num, self.env.actions_num))
        else:
            total_density = torch.zeros(size = (self.env.max_episode_length - self.env.h, self.env.states_num, self.env.actions_num))

        # TODO: this can be sped up 
        for i, policy in enumerate(policies):
            if i >= len(densities):
                d = self.density_oracle_single(policy)
                densities.append(d)
            total_density += weights[i] * densities[i]
        return total_density
    
class DeltaDensityEstimator(DensityEstimator):
    def __init__(self, env:ContinuousEnv, objective:RewardFunctional) -> None:
        super().__init__()
        self.env = env
        self.objective = objective
    
    def density_oracle_single(self, policy: Policy):
        # run the policy and accumulate the states and actions
        state = self.env.state

        if type(policy) is StationaryPolicyContinuous:

            density = SimpleDeltaDensity(self.env)

            for _ in range(self.env.max_episode_length - self.env.h):
                action = policy.next_action(state)
                density = density + SimpleDeltaDensity(self.env, state, action)
                state = self.env.next(state, action)

        elif type(policy) is NonStationaryPolicyContinuous:

            density = NonStationaryDeltaDensity(self.env)
            densities = []
            policy._reset()

            # record the current time at the policy
            policy_time = policy.time
            for _ in range(self.env.max_episode_length - self.env.h):
                action = policy.next_action(state)
                densities.append(SimpleDeltaDensity(self.env, state, action))
                state = self.env.next(state, action)
            # reset the policy to the original time
            policy.time = policy_time
            
            density = density + NonStationaryDeltaDensity(self.env, densities)
            
        else:
            raise TypeError('invalid policy type')
        
        return density

    def density_oracle(self, policies, weights, densities, stationary = False):
        
        if stationary:
            total_density = SimpleDeltaDensity(self.env)
        else:
            total_density = NonStationaryDeltaDensity(self.env)
        
        # aggregate the densities depending on the weights
        for i, policy in enumerate(policies):
            if i >= len(densities):
                d = self.density_oracle_single(policy)
                densities.append(d)
            total_density += weights[i] * densities[i]
        return total_density