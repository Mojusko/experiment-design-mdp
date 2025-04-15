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
        # Ensure we are working with a tensor slice and use torch.sum
        state_policy_tensor = torch.as_tensor(state_policy[actions], dtype=torch.float64) # Match self.p dtype
        sum_state_policy = torch.sum(state_policy_tensor)

        # Avoid division by zero or near-zero
        if sum_state_policy > 1e-9: # Use a small threshold
            reduced_state_policy = state_policy_tensor / sum_state_policy
        else:
            # If the sum is near zero, it means the policy assigns no probability mass
            # to any of the currently available actions. This indicates a potential
            # issue with the policy definition or an environment/policy mismatch.
            raise ValueError(
                f"Policy assigns zero or near-zero probability ({sum_state_policy.item():.2e}) "
                f"to all available actions {actions} in state {state}. "
                f"Check policy definition or environment constraints."
            )

        # Sample from the normalized probabilities over available actions
        # Ensure reduced_state_policy is 1D for multinomial
        action_index = torch.multinomial(reduced_state_policy.view(-1), 1).item()
        return actions[action_index]

class StationaryPolicyContinuous(Policy):
    def __init__(self, env: ContinuousEnv, p: torch.nn.Module) -> None:
        super().__init__(env)
        self.p = p
    
    def next_action(self, state: torch.Tensor):
        with torch.no_grad():
            action = torch.clip(self.p(state), self.env.min_action, self.env.max_action)
        return action
