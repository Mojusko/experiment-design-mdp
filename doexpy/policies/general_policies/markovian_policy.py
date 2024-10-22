import autograd.numpy as np

from doexpy.policies.policy_base import Policy
from doexpy.env.discrete_env import DiscreteEnv
from doexpy.convex_solvers.convex_solvers_base import ConvexSolverBase


class MarkovianPolicy(Policy):
    def __init__(self, env: DiscreteEnv, convex_solver: ConvexSolverBase) -> None:
        self.time = 0
        self.summarized_policy = None
        self.convex_solver = convex_solver
        super().__init__(env)
    
    def optimize(self, emissions, visitations, episodes):
        self.summarized_policy, self.policies, self.weights, self.densities = self.convex_solver.optimize(emissions, visitations, episodes)

    def next_action(self, state: int, emissions, visitations, episodes, keep = False):
        if self.time == 0 and keep == False:
            self.optimize(emissions, visitations, episodes)

        action = self.summarized_policy.next_action(state)
        self.time += 1
        if self.time == self.env.max_episode_length:
            self._reset()
        return action

    def return_density(self):
        if self.summarized_policy is None:
            raise AssertionError("Trying to return density before being optimizer")
        else:
            return self.summarized_policy.density_sa

    def _reset(self):
        self.time = 0