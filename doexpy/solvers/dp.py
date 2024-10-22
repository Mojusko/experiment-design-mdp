import numpy as np 
import torch 
from doexpy.solvers.solver_base import DiscreteSolver
from doexpy.policies.policy_base import Policy
from doexpy.policies.base_policies.non_stationary_policy import NonStationaryPolicy
from doexpy.policies.base_policies.stationary_policy import StationaryPolicy


class DP(DiscreteSolver):
    def solve(self) -> Policy:
        '''
        Solves the MDP and returns the value function.
        '''
        transition_matrix = self.env.get_transition_matrix()
        
        actions = torch.zeros(size = (self.env.max_episode_length + 1 - self.env.h, self.env.states_num), dtype=int)
        values = torch.ones(size = (self.env.max_episode_length + 1 - self.env.h, self.env.states_num, self.env.actions_num), dtype = torch.float64) * -1e20

        actions[self.env.max_episode_length - self.env.h, :] = 0 # pointing to the 'wait' action

        #TODO: make the constrained environment more general: used when a specific state is forced 
        # at a specific time step
        
        if self.env.constrained:
            values[self.env.max_episode_length - self.env.h, :, :] = -1e10
            values[self.env.max_episode_length - self.env.h, self.env.terminal_state, :] = \
                self.reward[self.env.max_episode_length - 1 - self.env.h, self.env.terminal_state, :]
        else:
            values[self.env.max_episode_length - self.env.h] = self.reward[self.env.max_episode_length - 1 - self.env.h]

        #TODO: remove for-loop to make more efficient
        for i in range(self.env.max_episode_length - 1 - self.env.h, -1, -1):
            for state in range(self.env.states_num):
                acts = self.env.available_actions(state)
                
                new_values = torch.stack([self.reward[i, state, a] + transition_matrix[state, a] @ values[i+1].max(dim = -1)[0] for a in acts])

                optimal_actions = torch.argwhere(new_values == torch.max(new_values))
                idx = np.random.choice(
                    optimal_actions.numpy().flatten()
                )

                best_act = acts[idx]
                actions[i, state] = best_act
                values[i, state, acts] = new_values

        ps = torch.zeros(size = (self.env.max_episode_length - self.env.h, self.env.states_num, self.env.actions_num), dtype = torch.float64)
        for i in range(self.env.max_episode_length - self.env.h):
            for s in range(self.env.states_num):
                ps[i, s, actions[i, s]] = 1.
                # check action is valid
                assert actions[i, s] in self.env.available_actions(s), 'invalid density policy'

        return NonStationaryPolicy(self.env, ps)