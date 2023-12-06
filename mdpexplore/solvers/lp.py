import cvxpy as cp
import autograd.numpy as np

from mdpexplore.solvers.solver_base import DiscreteSolver
from mdpexplore.policies.policy_base import Policy
from mdpexplore.policies.base_policies.stationary_policy import StationaryPolicy
# import mosek


class LP(DiscreteSolver):
    def solve(self) -> Policy:
        '''
        Solves the MDP and returns the value function.
        '''
        transition_matrix = self.env.get_transition_matrix()

        v = cp.Variable(self.env.states_num)  # number of states
        objective = cp.Minimize(cp.sum(v))

        constraints = []
        for s in range(self.env.states_num):
            if self.env.terminal_state == s:
                constraints.append(v[s] >= self.reward[s])
                break
        # for a in range(self.env.actions_num):
        # 	if self.env.is_valid_action(a, s):
        # 		constraints.append(v[s] >= self.reward[s] + self.env.discount_factor * transition_matrix[s, a] @ v)
        for a in range(self.env.actions_num):
            constraints += [v >= self.reward + self.env.discount_factor * transition_matrix[:, a] @ v]
        problem = cp.Problem(objective, constraints)
        result = problem.solve()

        # c = np.ones(shape=(self.env.states_num))
        # A = np.zeros(shape = (self.env.states_num, self.env.actions_num))
        # b = np.zeros(shape = (self.env.action_num))
        #
        # for s in range(self.env.states_num):
        # 	if self.env.terminal_state == s:
        # 		constraints.append(v[s] >= self.reward[s])
        # 		continue
        #
        # 	for a in range(self.env.actions_num):
        # 		if self.env.is_valid_action(a, s):
        # 			constraints.append(v[s] >= self.reward[s] + self.env.discount_factor * transition_matrix[s, a] @ v)

        # assemble a policy

        p = np.zeros((self.env.states_num, self.env.actions_num))
        for s in range(self.env.states_num):
            q_function = self.reward[s] + self.env.discount_factor * np.dot(transition_matrix[s, :], v.value)
            for a in range(self.env.actions_num):
                if not self.env.is_valid_action(a, s):
                    q_function[a] = -1e10
            p[s, np.argwhere(q_function == np.max(q_function))] = 1. / \
                                                                  np.argwhere(q_function == np.max(q_function)).shape[0]

        return StationaryPolicy(self.env, p)
