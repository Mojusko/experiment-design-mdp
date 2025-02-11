import numpy as np 
import torch 
from doexpy.solvers.solver_base import DiscreteSolver
from doexpy.policies.policy_base import Policy
from doexpy.policies.base_policies.non_stationary_policy import NonStationaryPolicy
from doexpy.policies.base_policies.stationary_policy import StationaryPolicy

class DP(DiscreteSolver):
    def solve(self) -> Policy:
        '''
        Solves the MDP and returns a policy.
        
        If self.reward is a 3d tensor (time x states x actions) the method runs a 
        finite-horizon DP (backwards induction) and returns a NonStationaryPolicy.
        
        If self.reward is a 2d tensor (states x actions) the method runs value iteration
        (using self.env.max_episode_length - self.env.h iterations) to compute a fixed
        point and returns a StationaryPolicy.
        '''
        # Get the transition matrix (assumed to be indexed as: transition_matrix[s, a] is a vector over next states)
        transition_matrix = self.env.get_transition_matrix()

        # --- NON-STATIONARY CASE (reward: time x states x actions) ---
        if self.reward.dim() == 3:
            # Allocate arrays for actions and values.
            T = self.env.max_episode_length + 1 - self.env.h  # total number of stages
            actions = torch.zeros(size=(T, self.env.states_num), dtype=int)
            values = torch.ones(size=(T, self.env.states_num, self.env.actions_num), dtype=torch.float64) * -1e20

            # Set a default terminal decision (e.g. “wait”)
            actions[self.env.max_episode_length - self.env.h, :] = 0

            if self.env.constrained:
                values[self.env.max_episode_length - self.env.h, :, :] = -1e10
                values[self.env.max_episode_length - self.env.h, self.env.terminal_state, :] = \
                    self.reward[self.env.max_episode_length - 1 - self.env.h, self.env.terminal_state, :]
            else:
                values[self.env.max_episode_length - self.env.h] = self.reward[self.env.max_episode_length - 1 - self.env.h]

            # Backward recursion (note: could be vectorized for speed)
            for i in range(self.env.max_episode_length - 1 - self.env.h, -1, -1):
                for state in range(self.env.states_num):
                    acts = self.env.available_actions(state)
                    # Compute the Q-value for each available action
                    new_values = torch.stack([
                        self.reward[i, state, a] + (transition_matrix[state, a] @ values[i+1].max(dim=-1)[0])
                        for a in acts
                    ])
                    # If several actions are optimal, choose one at random.
                    optimal_actions = torch.argwhere(new_values == torch.max(new_values))
                    idx = np.random.choice(optimal_actions.numpy().flatten())
                    best_act = acts[idx]
                    actions[i, state] = best_act
                    values[i, state, acts] = new_values

            # Build the policy tensor (only for decision epochs, not including terminal state)
            ps = torch.zeros(size=(self.env.max_episode_length - self.env.h, self.env.states_num, self.env.actions_num),
                             dtype=torch.float64)
            for i in range(self.env.max_episode_length - self.env.h):
                for s in range(self.env.states_num):
                    ps[i, s, actions[i, s]] = 1.
                    # Double-check that the chosen action is allowed.
                    assert actions[i, s] in self.env.available_actions(s), 'invalid density policy'
            return NonStationaryPolicy(self.env, ps)

        # --- STATIONARY CASE (reward: states x actions) ---
        elif self.reward.dim() == 2:
            # We use a simple value-iteration (with a fixed number of iterations)
            # to compute the (undiscounted) fixed point.
            Q = torch.zeros((self.env.states_num, self.env.actions_num), dtype=torch.float64)
            if self.env.constrained:
                # In the constrained case we “fix” the Q-values for the terminal state.
                Q.fill_(-1e20)
                Q[self.env.terminal_state, :] = self.reward[self.env.terminal_state, :]
            else:
                Q = self.reward.clone()

            num_iter = self.env.max_episode_length - self.env.h
            for _ in range(num_iter):
                Q_new = Q.clone()
                # For each state, we update the Q-value for every allowed action.
                # Here V(s) = max_a Q(s, a) is used.
                V = torch.max(Q, dim=1)[0]
                for s in range(self.env.states_num):
                    if self.env.constrained and s == self.env.terminal_state:
                        # Do not update the terminal state's Q-values if constrained.
                        continue
                    acts = self.env.available_actions(s)
                    for a in acts:
                        Q_new[s, a] = self.reward[s, a] + (transition_matrix[s, a] @ V)
                # Check for (simple) convergence.
                if torch.max(torch.abs(Q_new - Q)) < 1e-6:
                    Q = Q_new
                    break
                Q = Q_new

            # Extract a greedy (deterministic) stationary policy.
            p = torch.zeros((self.env.states_num, self.env.actions_num), dtype=torch.float64)
            for s in range(self.env.states_num):
                acts = self.env.available_actions(s)
                q_vals = Q[s, acts]
                best_idx = torch.argmax(q_vals)
                best_act = acts[best_idx]
                p[s, best_act] = 1.0
            return StationaryPolicy(self.env, p)

        else:
            raise ValueError("Reward must be either a 2d tensor (states x actions) or a 3d tensor (horizon x states x actions)")
