from doexpy.policies.policy_base import Policy
from doexpy.env.discrete_env import Environment, DiscreteEnv
from doexpy.env.continuous_env import ContinuousEnv
import torch
import torch.nn as nn


class NonStationaryPolicy(Policy):
    def __init__(self, env: DiscreteEnv, ps: torch.Tensor) -> None:
        self.ps = ps
        self.time = 0
        super().__init__(env)

    def next_action(self, state: int):
        state_policy = self.ps[self.time, state]
        actions = self.env.available_actions(state)
        reduced_state_policy = state_policy[actions]/torch.sum(state_policy[actions])
        self.time += 1
        if self.time == self.env.max_episode_length:
            self._reset()      
        return actions[torch.multinomial(reduced_state_policy, 1)]
       

    def _reset(self):
        self.time = 0

class NonStationaryPolicyContinuous(Policy):
    def __init__(self, env: ContinuousEnv, ps: nn.Module) -> None:
        super().__init__(env)
        self.ps = ps
        self.time = 0
    
    def next_action(self, state: torch.Tensor):

        with torch.no_grad():
            state = state
            action = torch.clip(self.ps[self.time](state), self.env.min_action, self.env.max_action)
    
        self.time += 1
        if self.time == self.env.max_episode_length:
            self._reset()
        
        return action

    def _reset(self):
        self.time = 0