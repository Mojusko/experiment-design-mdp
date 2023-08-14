import autograd.numpy as np

from mdpexplore.solvers.solver_base import DiscreteSolver
from mdpexplore.policies.policy_base import Policy
from mdpexplore.policies.base_policies.non_stationary_policy import NonStationaryPolicy
from mdpexplore.policies.base_policies.stationary_policy import StationaryPolicy


class DP(DiscreteSolver):
    def solve(self) -> Policy:
        '''
        Solves the MDP and returns the value function.
        '''
        transition_matrix = self.env.get_transition_matrix()
        
        actions = np.zeros((self.env.max_episode_length + 1 - self.env.h, self.env.states_num), dtype=int)
        values = np.ones((self.env.max_episode_length + 1 - self.env.h, self.env.states_num, self.env.actions_num)) * -1e20

        actions[self.env.max_episode_length - self.env.h, :] = 0 # pointing to the 'wait' action

        #TODO: make the constrained environment more general: used when a specific state is forced 
        # at a specific time step
        
        if self.env.constrained:
            values[self.env.max_episode_length - self.env.h, :, :] = -1e10
            values[self.env.max_episode_length - self.env.h, self.env.terminal_state, :] = \
                self.reward[self.env.max_episode_length - 1 - self.env.h, self.env.terminal_state, :]
        else:
            values[self.env.max_episode_length - self.env.h] = self.reward[self.env.max_episode_length - 1 - self.env.h]

        for i in range(self.env.max_episode_length - 1 - self.env.h, -1, -1):
            for state in range(self.env.states_num):
                acts = self.env.available_actions(state)
                new_values = np.array(
                    [self.reward[i, state, a] + transition_matrix[state, a] @ values[i+1].max(axis = -1) for a in acts]
                )
                optimal_actions = np.argwhere(new_values == np.max(new_values))
                idx = np.random.choice( 
                    optimal_actions.flatten()
                )
                best_act = acts[idx]
                actions[i, state] = best_act
                values[i, state, acts] = new_values

        ps = np.zeros((self.env.max_episode_length - self.env.h, self.env.states_num, self.env.actions_num))
        for i in range(self.env.max_episode_length - self.env.h):
            for s in range(self.env.states_num):
                ps[i, s, actions[i, s]] = 1.
                # check action is valid
                assert actions[i, s] in self.env.available_actions(s), 'invalid density policy'

        return NonStationaryPolicy(self.env, ps)