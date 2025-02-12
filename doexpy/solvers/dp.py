import numpy as np 
import torch 
from doexpy.solvers.solver_base import DiscreteSolver
from doexpy.policies.policy_base import Policy
from doexpy.policies.base_policies.non_stationary_policy import NonStationaryPolicy
from doexpy.policies.base_policies.stationary_policy import StationaryPolicy


class DP(DiscreteSolver):
    def solve(self) -> Policy:
        """
        Solves the MDP and returns a policy.
        
        If self.reward is a 3d tensor (time x states x actions) the method runs a 
        finite-horizon DP (backwards induction) and returns a NonStationaryPolicy.
        
        If self.reward is a 2d tensor (states x actions) the method runs value iteration
        (using self.env.max_episode_length - self.env.h iterations) to compute a fixed
        point and returns a StationaryPolicy.
        """
        # Choose device (GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move transition matrix and reward tensor to the device.
        transition_matrix = self.env.get_transition_matrix().to(device)
        reward = self.reward.to(device)
        
        num_states = self.env.states_num
        num_actions = self.env.actions_num

        # Precompute a boolean mask of allowed actions.
        # Here we assume that available_actions(s) returns an iterable (e.g. list) of allowed action indices.
        allowed_mask = torch.zeros((num_states, num_actions), dtype=torch.bool, device=device)
        for s in range(num_states):
            acts = self.env.available_actions(s)
            # Set allowed_mask[s, a] = True for every allowed action a.
            allowed_mask[s, torch.tensor(acts, device=device, dtype=torch.long)] = True

        # ------------------- NON-STATIONARY CASE -------------------
        if reward.dim() == 3:
            # Time horizon: note that T = max_episode_length + 1 - h,
            # and the terminal decision is at index T-1.
            T = self.env.max_episode_length + 1 - self.env.h  
            # actions: (time x states) with integer entries.
            actions_tensor = torch.zeros((T, num_states), dtype=torch.int64, device=device)
            # values: (time x states x actions), initialized to a very low value.
            values = torch.full((T, num_states, num_actions), -1e20, dtype=torch.float64, device=device)

            # Terminal stage index (for DP recursion)
            terminal_idx = self.env.max_episode_length - self.env.h  # equals T-1
            # Set default terminal decision to “wait” (action 0)
            actions_tensor[terminal_idx] = 0

            if self.env.constrained:
                # For the constrained case, initialize all Q-values at terminal stage to -1e10,
                # except for the terminal state, which is set to the terminal reward.
                values[terminal_idx] = -1e10
                # The reward index used for terminal stage is (max_episode_length - 1 - h)
                term_reward_idx = self.env.max_episode_length - 1 - self.env.h
                values[terminal_idx, self.env.terminal_state, :] = reward[term_reward_idx, self.env.terminal_state, :]
            else:
                # Unconstrained: terminal-stage Q-values equal the terminal reward.
                term_reward_idx = self.env.max_episode_length - 1 - self.env.h
                values[terminal_idx] = reward[term_reward_idx]

            # Backward recursion.
            # We iterate from time index terminal_idx-1 down to 0.
            for i in range(terminal_idx - 1, -1, -1):
                # For each state, compute V_next = max_a { values[i+1, state, a] }
                V_next = values[i + 1].max(dim=-1)[0]  # shape: (num_states,)
                # Compute expected next-stage value for all (state, action) pairs:
                # For each state s and action a, do: dot = transition_matrix[s,a] @ V_next.
                dot = torch.einsum('sak,k->sa', transition_matrix, V_next)
                # Q_all is the one–step return at time i.
                Q_all = reward[i] + dot  # shape: (num_states, num_actions)
                # Only allowed actions are “active”: mask out disallowed ones.
                Q_allowed = torch.where(allowed_mask, Q_all,
                                        torch.full_like(Q_all, -1e20))
                # For each state, find the maximum Q value among allowed actions.
                max_val = Q_allowed.max(dim=-1, keepdim=True)[0]  # shape: (num_states, 1)
                # Identify all actions that are tied for the maximum.
                ties = (Q_allowed == max_val)
                # For each state, sample uniformly among the tied actions.
                # (Note: multinomial expects nonnegative weights; ties.to(float) yields 1.0 where True.)
                best_actions = torch.multinomial(ties.to(torch.float64), num_samples=1).squeeze(-1)  # shape: (num_states,)
                # Record the chosen best action for this stage.
                actions_tensor[i] = best_actions
                # Update the Q-values at time i for allowed actions.
                values[i] = torch.where(allowed_mask, Q_all, values[i])

            # Build the policy tensor for the decision epochs (exclude terminal stage).
            policy_tensor = torch.zeros((terminal_idx, num_states, num_actions),
                                        dtype=torch.float64, device=device)
            # Vectorized assignment: for each time i and state s, set probability 1 for the chosen action.
            time_idx = torch.arange(terminal_idx, device=device).unsqueeze(1).expand(terminal_idx, num_states)
            state_idx = torch.arange(num_states, device=device).unsqueeze(0).expand(terminal_idx, num_states)
            policy_tensor[time_idx, state_idx, actions_tensor[:terminal_idx]] = 1.0

            return NonStationaryPolicy(self.env, policy_tensor.cpu())

        # ------------------- STATIONARY CASE -------------------
        elif reward.dim() == 2:
            if self.env.constrained:
                # For the constrained case, we “fix” Q-values for the terminal state.
                Q = torch.full((num_states, num_actions), -1e20, dtype=torch.float64, device=device)
                Q[self.env.terminal_state] = reward[self.env.terminal_state]
            else:
                Q = reward.clone()

            num_iter = self.env.max_episode_length - self.env.h
            for _ in range(num_iter):
                # V(s) = max_a Q(s, a)
                V = Q.max(dim=1)[0]  # shape: (num_states,)
                # For every (state, action), compute the expected next-stage value.
                dot = torch.einsum('sak,k->sa', transition_matrix, V)
                Q_update = reward + dot  # shape: (num_states, num_actions)
                if self.env.constrained:
                    # Do not update the terminal state's Q-values.
                    non_terminal = torch.ones(num_states, dtype=torch.bool, device=device)
                    non_terminal[self.env.terminal_state] = False
                    update_mask = non_terminal.unsqueeze(1) & allowed_mask
                    Q_new = torch.where(update_mask, Q_update, Q)
                else:
                    Q_new = torch.where(allowed_mask, Q_update, Q)
                if torch.max(torch.abs(Q_new - Q)) < 1e-6:
                    Q = Q_new
                    break
                Q = Q_new

            # Extract a greedy (deterministic) policy.
            policy_tensor = torch.zeros((num_states, num_actions), dtype=torch.float64, device=device)
            Q_allowed = torch.where(allowed_mask, Q, torch.full_like(Q, -1e20))
            max_val = Q_allowed.max(dim=1, keepdim=True)[0]
            ties = (Q_allowed == max_val)
            best_actions = torch.multinomial(ties.to(torch.float64), num_samples=1).squeeze(-1)
            policy_tensor[torch.arange(num_states, device=device), best_actions] = 1.0

            return StationaryPolicy(self.env, policy_tensor.cpu())

        else:
            raise ValueError("Reward must be either a 2d tensor (states x actions) or a 3d tensor (time x states x actions)")
