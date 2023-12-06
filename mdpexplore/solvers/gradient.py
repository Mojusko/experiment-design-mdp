import numpy as np
from mdpexplore.solvers.solver_base import DiscreteSolver
from mdpexplore.solvers.solver_base import ContinuousSolver
from mdpexplore.policies.policy_base import Policy
from mdpexplore.policies.base_policies.linear_policy import LinearPolicy
from mdpexplore.env.linear_system import ContinuousEnv
from typing import Callable
from autograd import grad

class GradientSolver(ContinuousSolver):
    def __init__(
        self,
        env: ContinuousEnv,
        reward: Callable,
        max_iters: int = 10,
        l2_bound: int = 1,
        eta: float = 10000,
        verbose: bool = False
    ) -> None:
        self.verbose = verbose
        self.max_iters = max_iters
        self.l2_bound = l2_bound
        self.env = env
        self.reward = reward
        self.eta = eta

    def objective(self, K):
        pi = LinearPolicy(self.env, K)
        new_density = self.env.density_oracle_single(pi)
        value = self.reward(new_density)
        return value

    def solve(self) -> Policy:
        dim = self.env.state_dim
        K = np.eye(dim)/dim
        for i in range(self.max_iters):
            gradient = grad(lambda K: self.objective(K))
            v = gradient(K)
            K = K + self.eta * v
            if np.linalg.norm(K,ord = 'fro') > self.l2_bound:
                K = K/np.linalg.norm(K,ord = 'fro')
            if self.verbose:
                print ('\t iter %d objective:%f norm:%f' % (i, self.objective(K), np.linalg.norm(v)))

        pi = LinearPolicy(self.env, K)
        return pi


