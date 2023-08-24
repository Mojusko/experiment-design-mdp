from typing import Callable, Type, Union, Tuple
import numpy as np
from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.policies.policy_base import Policy
from mdpexplore.policies.base_policies.non_stationary_policy import NonStationaryPolicy
from mdpexplore.policies.base_policies.stationary_policy import StationaryPolicy
from mdpexplore.policies.summary_policies.density_policy import DensityPolicy
from mdpexplore.solvers.solver_base import DiscreteSolver
from mdpexplore.solvers.dp import DP
from mdpexplore.densities.density_estimators import TabularDensity
from scipy.optimize import minimize_scalar
from autograd import grad, hessian
import numpy.linalg as la
# cvxpy imports
import cvxpy as cp 
import mosek

from mdpexplore.convex_solvers.convex_solvers_base import ConvexSolverBase

class FrankWolfe(ConvexSolverBase):
    def __init__(self, env, objective, verbosity : int = 0, accuracy : float = None, num_components : int = 10, initial_policy : bool = False, step: Union[float, str] = None, solver : DiscreteSolver = DP, SummarizedPolicyType : Policy = DensityPolicy) -> None:
        super().__init__(env, objective, verbosity = verbosity, accuracy = accuracy, initial_policy = initial_policy, solver = solver)
        self.SummarizedPolicyType = SummarizedPolicyType
        self.num_components = num_components
        self.step = step
        self.type = 'frank-wolfe'
        
        # density estimator
        if self.env.type == 'discrete':
            self.density_estimator = TabularDensity(self.env, self.objective)
        else:
            raise NotImplementedError
    
    def _reward_fn_gradient(self, distribution: np.ndarray, emissions, visitations, episodes) -> np.ndarray:
        """Computes the reward functional differentiated wrt to the state distribution

        Args:
            distribution (np.ndarray): state distribution to compute the reward function

        Returns:
            np.ndarray: gradient of the functional wrt to the state distribution - i.e. the reward function
        """
        grad_fn = getattr(self.objective, "gradient", None)
        if callable(grad_fn):
            return grad_fn(emissions, distribution)

        if self.objective.get_type() == "adaptive":
            grad_fn = grad(lambda d: self.objective.eval(emissions, d, visitations, episodes))
        else:
            grad_fn = grad(lambda d: self.objective.eval(emissions, d, episodes))
        return grad_fn(distribution)

    def _reward_fn_hessian(self, distribution: np.array)->np.array:
        if self.objective.get_type() == "adaptive":
            hes_fn = hessian(lambda d: self.objective.eval(self.emissions, d, self.visitations, self.episodes))
        else:
            hes_fn = hessian(lambda d: self.objective.eval(self.emissions, d, self.episodes))
        return hes_fn(distribution)

    def _planning_oracle(self, reward: np.ndarray) -> Policy:
        """Computes the optimal policy given a reward function and internal environment

        Args:
            reward (np.ndarray): reward function for each state

        Returns:
            Policy: policy solving the saved environment with the given reward function
        """
        solver = self.solver(self.env, reward)
        return solver.solve()
    
    def optimize(self, emissions, visitations, episodes) -> None:
        """Performs Frank-Wolfe algorithm to maximize the objective function

        Args:
            num_components (int): limit on number of components to be obtained by the algorithm
            gap ([type], optional): upper bound on the optimality gap. Defaults to None.
            verbose (bool, optional): if True, logs optimization progress to terminal. Defaults to True.
        """

        # this is to ensure that the first policy has probability 1
        if self.initial_policy:
            counter = 1
        else:
            counter = 0

        gap = -10e10 if self.accuracy is None else self.accuracy
        empirical_gap = 1e10

        while counter < self.num_components and empirical_gap > gap:

            # calculate the current density
            density = self.density_estimator.density_oracle(self.policies, self.weights, self.densities, self.stationary)

            # gradient of the reward
            reward = self._reward_fn_gradient(density, emissions, visitations, episodes)
            
            new_policy = self._planning_oracle(reward)
            self.policies.append(new_policy)
            #
            # hess_min = np.min(np.linalg.eigh(self._reward_fn_hessian(density))[0])
            # hess_max = np.max(np.linalg.eigh(self._reward_fn_hessian(density))[0])
            hess_min = 0
            hess_max = 0
            # new base density to be added
            new_density = self.density_estimator.density_oracle_single(new_policy)

            if self.step == "line-search" and self.num_components > 1:
                # line search to determine optimal step-size

                def fn(h):
                    if self.objective.get_type() == "adaptive":
                        return -self.objective.eval(
                            self.emissions,
                            density * (1 - h) + h * new_density,
                            self.visitations,
                            self.episodes
                        )
                    return -self.objective.eval(
                        self.emissions,
                        density * (1 - h) + h * new_density,
                        self.episodes
                    )

                res = minimize_scalar(
                    fn,
                    bounds=(1e-5, 1. - 1e-5),
                    method='bounded')
                step_size = res.x

            elif self.step is not None and isinstance(self.step, float):
                # fixed step size
                step_size = self.step
            else:
                # greedy simulation
                step_size = 1.0 / (1 + counter)

            if self.objective.get_type() == "adaptive":
                objective = self.objective.eval(emissions, density, visitations, episodes)
            else:
                objective = self.objective.eval(emissions, density, episodes)

            empirical_gap = np.minimum((reward * (new_density - density)).sum(), empirical_gap)

            if self.verbosity > 0:
                print(f'component: {counter}, gap: {empirical_gap}, objective: {objective}, stepsize: {step_size}, gradient:{la.norm(reward)}, hess_max:{hess_max}, hess_min:{hess_min}')
            self.weights = [(1 - step_size) * weight for weight in self.weights] + [step_size]

            counter += 1
        
        self.summarize()
        
        return self.summarized_policy, self.policies, self.weights, self.densities