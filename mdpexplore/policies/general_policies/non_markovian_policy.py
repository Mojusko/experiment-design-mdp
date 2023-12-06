import autograd.numpy as np

from mdpexplore.policies.policy_base import Policy
from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.convex_solvers.convex_solvers_base import ConvexSolverBase


class NonMarkovianPolicy(Policy):
    def __init__(self, env: DiscreteEnv, convex_solver: ConvexSolverBase) -> None:
        self.time = 0
        self.state_trajectory = []
        self.action_trajectory = []
        self.convex_solver = convex_solver
        super().__init__(env)
    
    def optimize(self, emissions, visitations, episodes):
        self.summarized_policy, self.policies, self.weights, self.densities = self.convex_solver.optimize(emissions, visitations, episodes)

    def next_action(self, state: int, emissions, visitations, episodes):

        self.state_trajectory.append(state)

        self.convex_solver.reset()
        visitations_extended = visitations + [(self.state_trajectory, self.action_trajectory)]
        self.optimize(emissions, visitations_extended, episodes)

        action = self.summarized_policy.next_action(state)
        # print('action plan', [np.argmax(self.summarized_policy.policy.ps[i].sum(axis = 0)) for i in range(len(self.summarized_policy.policy.ps))])
        self.action_trajectory.append(action)

        self.time += 1
        if self.time == self.env.max_episode_length:
            self._reset()
        return action

    def _reset(self):
        self.time = 0
        self.convex_solver.reset()
        self.state_trajectory = []
        self.action_trajectory = []