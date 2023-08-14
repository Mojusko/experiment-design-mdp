import autograd.numpy as np

from mdpexplore.policies.policy_base import Policy
from mdpexplore.env.discrete_env import DiscreteEnv
from convex_solvers.convex_solvers_base import ConvexSolverBase


class NonMarkovianPolicy(Policy):
    def __init__(self, env: DiscreteEnv, convex_solver: ConvexSolverBase) -> None:
        self.time = 0
        self.convex_solver = convex_solver
        super().__init__(env)
    
    def optimize(self):
        self.summarized_policy, self.policies, self.weights, self.densities = self.convex_solver.optimize()

    def next_action(self, state: int):
        self.optimize()
        action = self.summarized_policy.next_action(state)
        self.time += 1
        if self.time == self.env.max_episode_length:
            self._reset()
        return action

    def _reset(self):
        self.time = 0