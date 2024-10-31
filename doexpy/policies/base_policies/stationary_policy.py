import autograd.numpy as np

from doexpy.env.discrete_env import DiscreteEnv
from doexpy.env.continuous_env import ContinuousEnv
from doexpy.policies.policy_base import Policy

import torch


class StationaryPolicy(Policy):
    def __init__(self, env: DiscreteEnv, p: np.ndarray) -> None:
        self.p = p
        super().__init__(env)

    def next_action(self, state: int):
        state_policy = self.p[state]
        actions = self.env.available_actions(state)
        reduced_state_policy = state_policy[actions]/np.sum(state_policy[actions])
        return actions[torch.multinomial(reduced_state_policy, 1)]

class StationaryPolicyContinuous(Policy):
    def __init__(self, env: ContinuousEnv, p: torch.nn.Module) -> None:
        super().__init__(env)
        self.p = p
    
    def next_action(self, state: torch.Tensor):
        with torch.no_grad():
            action = torch.clip(self.p(state), self.env.min_action, self.env.max_action)
        return action