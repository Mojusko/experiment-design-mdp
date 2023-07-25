from typing import Callable, Type, Union, Tuple
from datetime import datetime
import os
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import wandb

from mdpexplore.solvers.solver_base import DiscreteSolver, ContinuousSolver
from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.policies.policy_base import Policy, SummarizedPolicy
from mdpexplore.policies.stationary_policy import StationaryPolicy
from mdpexplore.policies.non_stationary_policy import NonStationaryPolicy
from mdpexplore.policies.mixture_policy import MixturePolicy
from mdpexplore.policies.density_policy import DensityPolicy
from mdpexplore.policies.policy_generator import ContinuousPolicyGenerator
from mdpexplore.functionals.reward_functionals import *
from mdpexplore.mdpexplore import MdpExplore
from mdpexplore.env.linear_system import ContinuousEnv
from mdpexplore.utils.continuous_reward_functionals import ContinuousRewardFunctional


class ContinuousMdpExplore(MdpExplore):
    def __init__(
            self,
            env: ContinuousEnv,
            objective: ContinuousRewardFunctional,
            solver: Type[ContinuousSolver],
            method: str = 'frank-wolfe',
            step: Union[float, str] = None,
            initial_policy: bool = False,
            verbosity: int = 0
    ) -> None:
        """Class containing components required to run the maximum entropy exploration algorithm

        Args:
            env (DiscreteEnv): environment to be solved with max-ent
            objective (Callable[..., float]): reward functional to generate the reward functions
            solver (Type[DiscreteSolver]): MDP solver to be used as planning oracle
            step (Union[float, str], optional): step size to be used in optimization - float, 'line-search' or None (emulating greedy). Defaults to None.
            method (str, optional): optimization method. Defaults to 'frank-wolfe'.
            verbosity (int, optional): level of information logging. Defaults to 0.
        """
        self.method = method
        self.env = env
        self.objective = objective
        self.solver = solver
        self.step = step
        self.verbosity = verbosity
        self.policy_generator = ContinuousPolicyGenerator(self.env)

        self.initial_policy = initial_policy
        if self.initial_policy:
            self.policies = [self.policy_generator.uniform_policy()]
            self.weights = [1]
        else:
            self.policies = []
            self.weights = []

        self.densities = []
        self.trajectory = []
        self.objective_values_baseline = []
        self.visitations = []

    def _reset(self, reset_visitations=True) -> None:
        """Resets the max-ent solver to its initial state
        """
        self.env.reset()
        if self.initial_policy:
            self.policies = [self.policy_generator.uniform_policy()]
            self.weights = [1]
        else:
            self.policies = []
            self.weights = []

        self.densities = []
        self.trajectory = []

        if reset_visitations:
            self.visitations = []
            self.objective_values_baseline = []

    def _density_oracle_single(self, policy: Policy) -> np.ndarray:
        """Computes state distribution induced by the given policy over a specified horizon

        Args:
            policy (Policy): inducing policy
        Returns:
            param (list): parameters of the distribution
        """
        distribution_params = self.env.density_oracle_single(policy)
        return distribution_params

    def _density_oracle(self, actions: bool = False) -> np.ndarray:
        """Computes the combined state (or state-action) distribution induced by the saved policies

        Args:
            actions (bool, optional): if True, computes state-action distribution instead of state distribution. Defaults to False.

        Returns:
            np.ndarray: 1-D array with density for each state

        Raises:
            TypeError: if the saved policies are non-stationary
        """
        distribution_params = self.env.density_oracle(self.policies, self.weights)
        return distribution_params

    def _planning_oracle(self, reward: np.ndarray) -> Policy:
        """Computes the optimal policy given a reward function and internal environment

        Args:
            reward (np.ndarray): reward function for each state

        Returns:
            Policy: policy solving the saved environment with the given reward function
        """
        solver = self.solver(self.env, reward, verbose = self.verbosity>3)
        return solver.solve()

    def _reward_fn_gradient(self, distribution: np.ndarray) -> np.ndarray:

        ## TODO: This might need to change
        if self.objective.get_type() == "adaptive":
            grad_fn = lambda new_density : self.objective.gradient_at_density(distribution, self.visitations, self.episodes, new_density)
        else:
            grad_fn = lambda new_density : self.objective.gradient_at_density(distribution, self.episodes, new_density)
        return grad_fn

    def evaluate(
            self,
            SummarizedPolicyType: Type[SummarizedPolicy] = MixturePolicy,
            episodes: int = 100,
    ) -> None:

        """Evaluates the saved policies using chosen summarization method

        Args:
            SummarizedPolicyType (Type[SummarizedPolicy], optional): method of policy summarization (Mixture, Density or Average). Defaults to MixturePolicy.
            episodes (int, optional): number of policy rollouts to execute. Defaults to 100.
            plot (bool, optional): if True, plots the heatmap based on the policy rollouts. Defaults to True.
        """
        for _ in range(episodes):
            self.env.reset()

            summarized_policy = SummarizedPolicyType(self.env, self.policies, self.weights)
            self.trajectory.append(self.env.init_state)
            for _ in range(self.env.max_episode_length):
                action = summarized_policy.next_action(self.env.state)
                next_state = self.env.step(action)
                self.trajectory.append(next_state)

        self.visitations += [(1. / self.env.max_episode_length, self.env.visitations)]


    def optimize(
            self,
            num_components: int,
            method: str,
            accuracy: float = None,
    ) -> None:
        """Optimizes the objective function using the specified method

        Args:
            num_components (int): limit on number of components to be obtained by the algorithm
            method (str): method to be used for optimization (only frank-wolfe available)
            accuracy (float, optional): optimality gap to be reached. Defaults to None.
        """
        if method == 'frank-wolfe':
            self._optimize_frank_wolfe(
                num_components=num_components,
                gap=accuracy,
                verbose=self.verbosity > 2,
            )
        elif method == "direct":
            self._projected_gradient_descent(
                gap = accuracy,
                verbose = self.verbosity >2,
            )
        else:
            pass  # can't happen, handled by argparser

    def _projected_gradient_descent(self):
        pass

    def _optimize_frank_wolfe(
            self,
            num_components: int,
            gap=None,
            verbose=False,
    ) -> None:
        """Performs Frank-Wolfe algorithm to maximize the objective function

        Args:
            num_components (int): limit on number of components to be obtained by the algorithm
            gap ([type], optional): upper bound on the optimality gap. Defaults to None.
            verbose (bool, optional): if True, logs optimization progress to terminal. Defaults to True.
        """
        if self.initial_policy:
            counter = 1
        else:
            counter = 0

        gap = -10e10 if gap is None else gap
        empirical_gap = 1e10

        while counter < num_components and empirical_gap > gap:

            # calculate the current density
            density = self._density_oracle()

            # gradient of the reward
            reward_functional = self._reward_fn_gradient(density)

            # RL algorithm
            new_policy = self._planning_oracle(reward_functional)

            # new policy
            self.policies.append(new_policy)

            # new base density to be added
            new_density = self._density_oracle_single(new_policy)

            if self.step is not None and isinstance(self.step, float):
                # fixed step size
                step_size = self.step
            else:
                # greedy simulation
                step_size = 1.0 / (1 + counter)

            if self.objective.get_type() == "adaptive":
                objective = self.objective.eval(density, self.visitations, self.episodes)
            else:
                objective = self.objective.eval(density, self.episodes)

            if verbose:
                print(f'component: {counter}, objective: {objective}, stepsize: {step_size}')

            self.weights = [(1 - step_size) * weight for weight in self.weights] + [step_size]
            counter += 1

    def run(
            self,
            num_components: int,
            episodes: int = 100,
            accuracy: float = None,
            SummarizedPolicyType: Type[SummarizedPolicy] = MixturePolicy,
            save_trajectory: Union[str, None] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray, float], None]:
        """Runs the full max-ent procedure

        Args:
            num_components (int): limit on number of components to be obtained by the algorithm
            episodes (int, optional): number of policy rollouts for evaluation. Defaults to 100.
            num_runs (int, optional): number of max-ent runs. Defaults to 1.
            accuracy (float, optional): optimality gap for the optimization procedure. Defaults to None.
            SummarizedPolicyType (Type[SummarizedPolicy], optional): policy summarization mode to be used. Defaults to MixturePolicy.
            plot (bool, optional): if True, plots the resulting heatmaps. Defaults to True.

        Returns:
            np.ndarray: means of the objective values after each rollout
            np.ndarray: standard deviations of the objective values
            float: optimal objective value
        """
        if self.objective.type == "adaptive":
            self.episodes = episodes

            self._reset()
            run_objective_values = []
            aggregate_distribution = 0

            for i in range(episodes):

                if self.verbosity > 2:
                    print("Episode:", i)

                self._reset(reset_visitations=False)

                self.optimize(num_components, self.method, accuracy)

                self.evaluate(SummarizedPolicyType, 1)

                if save_trajectory is not None:
                    np.savetxt(f"{save_trajectory}{i}.txt",
                               np.array([self.env.convert(state) for state in self.trajectory]))

                print ("policies:", [policy.K for policy in self.policies])
                value = self.objective.eval_full( self.visitations, episodes)
                self.objective_values_baseline.append(value)
                run_objective_values.append(value)

            objective_values = run_objective_values

        else:
            self._reset()
            self.episodes = episodes
            self.optimize(num_components, self.method, accuracy)

            self.visitations = []
            self.evaluate(SummarizedPolicyType, episodes)

            run_objective_values = []
            aggregate_distribution = 0

            for i, d in enumerate(self.visitations):
                aggregate_distribution = (i * aggregate_distribution + d) / (i + 1)
                value = self.objective.eval_full( self.visitations, episodes)

                run_objective_values.append(
                    self.objective.eval_full(self.emissions, aggregate_distribution, self.episodes))

            objective_values = run_objective_values

        objective_values = np.array(objective_values)
        if self.objective.get_type() != "adaptive":
            opt = self.objective.eval_full(self.env, self._density_oracle(), self.episodes)
        else:
            opt = None

        return objective_values, opt
