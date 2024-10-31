from typing import Callable, Type, Union, Tuple
import torch 
import warnings
from doexpy.env.discrete_env import DiscreteEnv
from doexpy.policies.policy_base import Policy
from doexpy.policies.base_policies.non_stationary_policy import NonStationaryPolicy
from doexpy.policies.base_policies.stationary_policy import StationaryPolicy
from doexpy.policies.summary_policies.density_policy import DensityPolicy
from doexpy.policies.summary_policies.mixture_policy import MixturePolicy
from doexpy.solvers.solver_base import DiscreteSolver, ContinuousSolver
from doexpy.solvers.dp import DP
from doexpy.densities.density_estimators import TabularDensity, DeltaDensityEstimator
from doexpy.densities.continous_densities import ContinuousDensity, SimpleDeltaDensity, NonStationaryDeltaDensity
from scipy.optimize import minimize_scalar

#from autograd import grad, hessian
from torch.autograd import grad 

import torch.linalg as la

# cvxpy imports
import cvxpy as cp 
import mosek

from typing import Union

from doexpy.convex_solvers.convex_solvers_base import ConvexSolverBase

class FrankWolfe(ConvexSolverBase):
    def __init__(self, env,
                 objective,
                 verbosity : int = 0,
                 accuracy : float = None,
                 num_components : int = 10,
                 initial_policy : bool = False,
                 step: Union[float, str] = None,
                 solver : Union[DiscreteSolver, ContinuousSolver] = DP,
                 SummarizedPolicyType : Policy = DensityPolicy
                 ) -> None:
        super().__init__(env, objective, verbosity = verbosity, accuracy = accuracy, initial_policy = initial_policy, solver = solver)

        if (SummarizedPolicyType != MixturePolicy) & (self.env.type == 'continuous'):
            warnings.warn("SummarizedPolicyType is not MixturePolicy, but env is continuous. SummarizedPolicyType was automatically changed to MixturePolicy.")
            self.SummarizedPolicyType = MixturePolicy
        else:
            self.SummarizedPolicyType = SummarizedPolicyType
        self.num_components = num_components
        self.step = step
        self.type = 'frank-wolfe'
        
        # density estimator
        if self.env.type == 'discrete':
            self.density_estimator = TabularDensity(self.env, self.objective)
        elif self.env.type == 'continuous':
            self.density_estimator = DeltaDensityEstimator(self.env, self.objective)
        else:
            raise NotImplementedError
    
    def _reward_fn_gradient(self,
                            distribution: Union[torch.Tensor, ContinuousDensity],
                            emissions: torch.Tensor,
                            visitations,
                            episodes: int)-> Union[torch.Tensor, Callable]:
        """Computes the reward functional differentiated wrt to the state distribution

        Args:
            distribution (np.ndarray): state distribution to compute the reward function

        Returns:
            np.ndarray: gradient of the functional wrt to the state distribution - i.e. the reward function
        """
        grad_fn = getattr(self.objective, "gradient", None)

        if callable(grad_fn) and (self.objective.get_type() != "adaptive"):
            return grad_fn(emissions, distribution)
        
        if self.env.type == 'discrete':
            
            if callable(grad_fn):
                return grad_fn(emissions, distribution, visitations, episodes)

            if self.objective.get_type() == "adaptive":
                grad_fn = lambda d: grad(outputs=self.objective.eval(
                    emissions, d,visitations, episodes),
                      inputs=d)[0]
            else:
                grad_fn = lambda d: grad(outputs=self.objective.eval(
                    emissions, d, episodes),
                      inputs=d)[0]
                
            return grad_fn(distribution)

        elif self.env.type == 'continuous':

            # run objective pre-computations
            self.objective.pre_compute(emissions, distribution, visitations, episodes)

            if self.objective.get_type() == "adaptive":
                if self.stationary:
                    grad_fn = lambda h, s, a: self.objective.get_gradient_density(emissions, distribution, visitations, episodes, s, a)
                else:
                    grad_fn = lambda h, s, a: self.objective.get_gradient_density(emissions, distribution, visitations, episodes, s, a)
            else:
                if self.stationary:
                    grad_fn = lambda h, s, a: self.objective.get_gradient_density(emissions, distribution, visitations, episodes, s, a)
                else:
                    grad_fn = lambda h, s, a: self.objective.get_gradient_density(emissions, distribution, visitations, episodes, s, a)
            
            return grad_fn

    def _planning_oracle(self, reward: Union[torch.Tensor, Callable]) -> Policy:
        """Computes the optimal policy given a reward function and internal environment

        Args:
            reward (np.ndarray): reward function for each state

        Returns:
            Policy: policy solving the saved environment with the given reward function
        """
        # define the solver
        solver = self.solver(self.env, reward)
        # initialize the solver
        solver.initialize(self.initialization_params)
        # solve the problem, obtain the policy
        policy = solver.solve()
        # save the initialization parameters for the next iteration
        self.initialization_params = solver.initialization_params()

        return policy
    
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
        empirical_gap = torch.Tensor([1e10]).double()

        while counter < self.num_components and empirical_gap > gap:

            # calculate the current density
            density = self.density_estimator.density_oracle(self.policies, self.weights, self.densities, self.stationary)
            if self.env.type == 'discrete':
                density.requires_grad_(True)
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
            # print("new density states:", new_density.average_density().delta_states)

            if self.step == "line-search" and self.num_components > 1:
                # line search to determine optimal step-size

                def fn(h):
                    if self.objective.get_type() == "adaptive":
                        return -self.objective.eval(
                            emissions,
                            density * (1 - h) + h * new_density,
                            visitations,
                            episodes
                        ).detach().numpy()
                    return -self.objective.eval(
                        emissions,
                        density * (1 - h) + h * new_density,
                        episodes
                    ).detach().numpy()

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

            if self.env.type == 'discrete':
                empirical_gap = torch.minimum((reward * (new_density - density)).sum(), empirical_gap)

                if self.verbosity > 0:
                    print(f'component: {counter}, gap: {empirical_gap}, objective: {objective}, stepsize: {step_size}, gradient:{la.norm(reward)}, hess_max:{hess_max}, hess_min:{hess_min}')
            
            #TODO: add gradient norm and empirical gap to continuous case
            elif self.env.type == 'continuous':
                if self.verbosity > 0:
                    print(f'component: {counter}, objective: {objective}, stepsize: {step_size}, hess_max:{hess_max}, hess_min:{hess_min}')

            self.weights = [(1 - step_size) * weight for weight in self.weights] + [step_size]

            counter += 1
        
        self.summarize()
        
        return self.summarized_policy, self.policies, self.weights, self.densities