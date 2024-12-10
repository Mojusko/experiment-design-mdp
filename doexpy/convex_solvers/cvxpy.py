import numpy as np
from doexpy.policies.summary_policies.density_policy import DensityPolicy
# cvxpy imports
import cvxpy as cp 
import mosek
from doexpy.policies.summary_policies.mixture_policy import MixturePolicy

from doexpy.convex_solvers.convex_solvers_base import ConvexSolverBase

class CVXPY(ConvexSolverBase):
    def __init__(self, env, objective, verbosity = 0, accuracy = 1e-4) -> None:
        super().__init__(env, objective, verbosity = verbosity, accuracy = accuracy)
        self.type = 'cvxpy'
        self.stationary = False
        self.SummarizedPolicyType = MixturePolicy
    
    def optimize(self, emissions, visitations, episodes) -> None: 
        # initialize the objective function
        if hasattr(self.objective, "get_eval_cvxpy"):
            v = {}
            for h in range(self.env.max_episode_length):
                v[h] = cp.Variable((self.env.states_num, self.env.actions_num))
            objective = cp.Maximize(self.objective.get_eval_cvxpy(emissions, v, visitations, episodes))
            constraints = []
            # the initial marginal density should be 1 in the initial state and zero elsewhere
            constraints += [cp.sum(v[0][self.env.init_state, :]) == 1]
            for h in range(self.env.max_episode_length):
                constraints += [v[h] >= 0, v[h] <= 1, cp.sum(v[h]) == 1]
            
            # here we create a state-action visitation poly-tope 
            # TODO: this is not checked 
            P = self.env.get_transition_matrix()
            for i in range(self.env.max_episode_length-1):
                for s in range(self.env.states_num):
                    constraints += [cp.sum(v[i+1][s]) == cp.sum(cp.multiply(v[i], P[:, :, s]))]
        
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver = cp.MOSEK, mosek_params={mosek.dparam.intpnt_tol_rel_gap: 1e-6}, verbose = self.verbose)
            # v_val = v.value

            v_val = np.zeros((self.env.max_episode_length, self.env.states_num, self.env.actions_num))
            for h in range(self.env.max_episode_length):
                v_val_h = v[h].value
                # round anything below 1e-5 to 0
                v_val_h[v_val_h < 1e-5] = 0
                # normalize
                v_val_h = v_val_h / v_val_h.sum()
                # add to the array
                v_val[h] = v_val_h

            # Now create a density policy 
            # TODO: this is not compatibly with policy summarization, as it tries to summarize the policy twice
            new_policy = DensityPolicy(self.env, v_val)
            self.densities.append(v_val)
            self.policies.append(new_policy)
            self.weights = [1.0]

            self.summarize()

            return self.summarized_policy, self.policies, self.weights, self.densities

        else:
            raise NotImplementedError("The reward function does not have cvxpy interface implemented. Use different solver.")
