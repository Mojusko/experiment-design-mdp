import numpy as np
import torch.optim as optim
from typing import Callable, Type, Union, Tuple, List
import torch 
import warnings
from doexpy.env.discrete_env import DiscreteEnv
from doexpy.policies.policy_base import Policy
from doexpy.policies.base_policies.non_stationary_policy import NonStationaryPolicy
from doexpy.policies.base_policies.stationary_policy import StationaryPolicy
from doexpy.policies.summary_policies.density_policy import DensityPolicy
from doexpy.policies.summary_policies.mixture_policy import MixturePolicy
from doexpy.policies.summary_policies.tracking_policy import TrackingPolicy
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
                 verbosity: int = 0,
                 accuracy: float = None,
                 num_components: int = 10,
                 initial_policy: bool = False,
                 step: Union[float, str] = None,
                 solver: Union[DiscreteSolver, ContinuousSolver] = DP,
                 SummarizedPolicyType: Policy = DensityPolicy,
                 num_summarized_policies: int = 1,
                 num_rounds: int = 75,
                 stationary: bool = False
                 ) -> None:
        super().__init__(env, objective, verbosity=verbosity, accuracy=accuracy, initial_policy=initial_policy, solver=solver)
        self.stationary = stationary
    
        if (SummarizedPolicyType != MixturePolicy) & (self.env.type == 'continuous'):
            warnings.warn("SummarizedPolicyType is not MixturePolicy, but env is continuous. SummarizedPolicyType was automatically changed to MixturePolicy.")
            self.SummarizedPolicyType = MixturePolicy
        else:
            self.SummarizedPolicyType = SummarizedPolicyType
    
        self.num_components = num_components
        self.num_rounds = num_rounds
        self.step = step
        self.type = 'frank-wolfe'
        self.num_summarized_policies = num_summarized_policies
    
        # For multiple policies, replicate the base initialization
        if num_summarized_policies > 1:
            self.policies = [self.policies.copy() for _ in range(num_summarized_policies)]
            self.weights = [self.weights.copy() for _ in range(num_summarized_policies)]
            self.densities = [self.densities.copy() for _ in range(num_summarized_policies)]
    
        # density estimator
        if self.env.type == 'discrete':
            self.density_estimator = TabularDensity(self.env, self.objective)
        elif self.env.type == 'continuous':
            self.density_estimator = DeltaDensityEstimator(self.env, self.objective)
        else:
            raise NotImplementedError

    def _reward_fn_gradient(self,
                           distributions: Union[List[torch.Tensor], List[ContinuousDensity]],
                           emissions: torch.Tensor,
                           visitations,
                           episodes: int) -> List[Union[torch.Tensor, Callable]]:
        """Returns list of gradients, one per policy"""
        
        if len(distributions) == 1:
            return [super()._reward_fn_gradient(distributions[0], emissions, visitations, episodes)]
    
        grad_fn = getattr(self.objective, "gradient", None)
        
        if self.env.type == 'discrete':
            if callable(grad_fn):
                return grad_fn(emissions, distributions, visitations, episodes)
    
            if self.objective.get_type() == "adaptive":
                return list(grad(outputs=self.objective.eval(emissions, distributions, visitations, episodes),
                          inputs=distributions))
            else:
                return list(grad(outputs=self.objective.eval(emissions, distributions, episodes),
                          inputs=distributions))
    
        elif self.env.type == 'continuous':
            # Run objective pre-computations
            self.objective.pre_compute(emissions, distributions, visitations, episodes)
    
            if self.objective.get_type() == "adaptive":
                if self.stationary:
                    grad_fn = lambda h, s, a: self.objective.get_gradient_density(emissions, distributions, visitations, episodes, s, a)
                else:
                    grad_fn = lambda h, s, a: self.objective.get_gradient_density(emissions, distributions, visitations, episodes, s, a)
            else:
                if self.stationary:
                    grad_fn = lambda h, s, a: self.objective.get_gradient_density(emissions, distributions, visitations, episodes, s, a)
                else:
                    grad_fn = lambda h, s, a: self.objective.get_gradient_density(emissions, distributions, visitations, episodes, s, a)
            
            return grad_fn
    
        raise NotImplementedError(f'Environment type {self.env.type} not implemented')
    
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
    
    def _optimize_single(self, emissions, visitations, episodes) -> None:
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
            if len(reward) > 1:
                raise ValueError("_optimize_single received multiple rewards")
            reward = reward[0]
            new_policy = self._planning_oracle(reward)
            self.policies.append(new_policy)
            #
            # hess_min = np.min(np.linalg.eigh(self._reward_fn_hessian(density))[0])
            # hess_max = np.max(np.linalg.eigh(self._reward_fn_hessian(density))[0])
            hess_min = 0
            hess_max = 0
            # new base density to be added
            new_density = self.density_estimator.density_oracle_single(new_policy)
            # print("new density states:", new_density.average_density().delta_states)

            if self.step == "line-search" and self.num_components > 1:
                # line search to determine optimal step-size

                def fn(h):
                    if self.objective.get_type() == "adaptive":
                        return -self.objective.eval(
                            emissions,
                            density * (1 - h) + h * new_density,
                            visitations,
                            episodes
                        ).detach().cpu().numpy()
                    return -self.objective.eval(
                        emissions,
                        density * (1 - h) + h * new_density,
                        episodes
                    ).detach().cpu().numpy()

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

    def optimize(self, emissions, visitations, episodes):
        if self.num_summarized_policies == 1:
            return self._optimize_single(emissions, visitations, episodes)
        
        gap = -10e10 if self.accuracy is None else self.accuracy
    
        # If initial_policy is True, set the persistent counter high so that no updates occur.
        policy_counters = [
            self.num_rounds * self.num_components if self.initial_policy else 0
            for _ in range(self.num_summarized_policies)
        ]
        
        for round_idx in range(self.num_rounds):
            # For each round, optimize each policy in turn.
            for policy_idx in range(self.num_summarized_policies):
                empirical_gap = torch.Tensor([1e10]).double()
                
                # Each round allows up to (round_idx+1)*self.num_components updates.
                while (policy_counters[policy_idx] < (round_idx + 1) * self.num_components and 
                       torch.abs(empirical_gap) > gap):
                    
                    # Get current density for all policies.
                    densities = []
                    for i in range(self.num_summarized_policies):
                        density = self.density_estimator.density_oracle(
                            self.policies[i], 
                            self.weights[i], 
                            self.densities[i], 
                            self.stationary
                        ).double()
                        if self.env.type == 'discrete':
                            density.requires_grad_(True)
                        densities.append(density)

                    
                    # Get gradients for all policies.
                    rewards = self._reward_fn_gradient(densities, emissions, visitations, episodes)
                    
                    # Only update the current policy.
                    new_policy = self._planning_oracle(rewards[policy_idx])
                    self.policies[policy_idx].append(new_policy)
                    new_density = self.density_estimator.density_oracle_single(new_policy)
                    
                    # Compute step size for current policy.
                    if self.step == "line-search":
                        def compute_loss(h):
                            temp_densities = densities.copy()
                            temp_densities[policy_idx] = densities[policy_idx] * (1 - h) + h * new_density
                            if self.objective.get_type() == "adaptive":
                                return -self.objective.eval(emissions, temp_densities, visitations, episodes)
                            return -self.objective.eval(emissions, temp_densities, episodes)
                        step_size = self._gradient_line_search(compute_loss, emissions.device)                       
                    elif self.step is not None and isinstance(self.step, float):
                        step_size = self.step
                    else:
                        # Compute a step size that decreases over time using the persistent counter.
                        step_size = 1.0 / (1 + policy_counters[policy_idx])
                    
                    if self.env.type == 'discrete':
                        empirical_gap = torch.minimum(
                            (rewards[policy_idx] * (new_density - densities[policy_idx])).sum(), 
                            empirical_gap
                        )
                    
                    # Update weights for the current policy.
                    self.weights[policy_idx] = [(1 - step_size) * w for w in self.weights[policy_idx]] + [step_size]
                    
                    if self.verbosity > 0 and policy_counters[policy_idx] % 100 == 0:
                        if self.objective.get_type() == "adaptive":
                            objective = self.objective.eval(emissions, densities, visitations, episodes, should_mask=False)
                        else:
                            objective = self.objective.eval(emissions, densities, episodes)
                    
                        if self.env.type == 'discrete':
                            total_grad_norm = sum(la.norm(r) for r in rewards)
                            print(f'Round: {round_idx}, Policy: {policy_idx}, Component: {policy_counters[policy_idx]}, '
                                  f'Gap: {empirical_gap}, Objective: {objective}, '
                                  f'Stepsize: {step_size} ({self.step}), Gradient: {total_grad_norm}')
                        elif self.env.type == 'continuous':
                            print(f'Round: {round_idx}, Policy: {policy_idx}, '
                                  f'Objective: {objective}')
                    
                    # Increment the persistent counter for this policy.
                    policy_counters[policy_idx] += 1
        
        self.summarize()
        return self.summarized_policies, self.policies, self.weights, self.densities

    def _gradient_line_search(self, compute_loss, device, init=0.5, lr=0.05, n_iter=30):
        # Create a scalar tensor h with gradient tracking.
        h = torch.tensor(init, dtype=torch.float64, device=device, requires_grad=True)
        optimizer_h = optim.Adam([h], lr=lr)
        for _ in range(n_iter):
            optimizer_h.zero_grad()
            loss = compute_loss(h)
            loss.backward()
            optimizer_h.step()
            # Clamp h to be within (1e-5, 1-1e-5)
            with torch.no_grad():
                h.clamp_(1e-5, 1. - 1e-5)
        return h.item()
    def summarize(self) -> None:
        def create_policy(policies, weights, densities, empirical=None):
            if self.SummarizedPolicyType == DensityPolicy:
                return self.SummarizedPolicyType(
                    self.env, 
                    self.density_estimator.density_oracle(policies, weights, densities)
                )
            elif self.SummarizedPolicyType == MarginalDensityPolicy:
                return self.SummarizedPolicyType(
                    self.env, 
                    self.density_estimator.density_oracle(policies, weights, densities)
                )
            elif self.SummarizedPolicyType == TrackingPolicy:
                policy = self.SummarizedPolicyType(
                    self.env, policies, weights, empirical
                )
                empirical[policy.get_picked_policy_id()] += 1
                return policy
            else:
                self.density_estimator.density_oracle(policies, weights, densities)
                return self.SummarizedPolicyType(
                    self.env, policies, weights
                )

        # It would've been cleaner to always have summarize_policies list,
        # but not to mess with current API, to set summarized_policy for the singleton case
        if self.num_summarized_policies == 1:
            empirical = np.zeros(len(self.policies)) if self.SummarizedPolicyType == TrackingPolicy else None
            self.summarized_policy = create_policy(self.policies, self.weights, self.densities, empirical)
        else:
            empirical = np.zeros(len(self.policies[0])) if self.SummarizedPolicyType == TrackingPolicy else None
            self.summarized_policies = [
                create_policy(self.policies[i], self.weights[i], self.densities[i], empirical)
                for i in range(self.num_summarized_policies)
            ]

    def reset(self):
        super().reset()  # Get base class reset
        
        # Convert to nested structure for multiple policies
        if self.num_summarized_policies > 1:
            self.policies = [self.policies.copy() for _ in range(self.num_summarized_policies)]
            self.weights = [self.weights.copy() for _ in range(self.num_summarized_policies)]
            self.densities = [self.densities.copy() for _ in range(self.num_summarized_policies)]
