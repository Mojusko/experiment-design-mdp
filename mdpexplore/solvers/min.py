import autograd.numpy as np

from mdpexplore.solvers.solver_base import DiscreteSolver
from mdpexplore.policies.policy_base import Policy
from mdpexplore.policies.simple_policy import SimplePolicy


class MinSolver(DiscreteSolver):
    '''
    Dummy solver for bandit problem.
    '''

    def solve(self) -> Policy:
        action_id = np.random.choice(np.flatnonzero(self.reward == self.reward.min()))

        p = np.zeros((self.env.states_num, self.env.actions_num))
        # TODO: For backward compatibility we have more states than 1, this should be changed later.
        for s in range(self.env.states_num):
            p[s, action_id] = 1.

        return SimplePolicy(self.env, p)
