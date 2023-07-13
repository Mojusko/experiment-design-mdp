from typing import Callable, Type, Union, Tuple
from datetime import datetime
import os
import autograd.numpy as np
from autograd import grad, hessian
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import wandb

from mdpexplore.solvers.solver_base import DiscreteSolver
from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.policies.policy_base import Policy, SummarizedPolicy
from mdpexplore.policies.tracking_policy import TrackingPolicy
from mdpexplore.policies.simple_policy import SimplePolicy
from mdpexplore.policies.non_stationary_policy import NonStationaryPolicy
from mdpexplore.policies.mixture_policy import MixturePolicy
from mdpexplore.policies.density_policy import DensityPolicy
from mdpexplore.policies.policy_generator import PolicyGenerator
from mdpexplore.utils.reward_functionals import *


class MdpExplore():
    def __init__(
            self,
            env: DiscreteEnv,
            objective: RewardFunctional,
            solver: Type[DiscreteSolver],
            step: Union[float, str] = None,
            method: str = 'frank-wolfe',
            verbosity: int = 0,
            optimize_repetitions: bool = False,
            initial_policy: bool = False,
            callback: Union[Callable, None] = None,
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
        self.env = env
        self.objective = objective
        self.solver = solver
        self.step = step
        self.method = method
        self.verbosity = verbosity
        self.callback = callback

        self.policy_generator = PolicyGenerator(self.env)
        self.initial_policy = initial_policy

        if self.initial_policy:
            self.policies = [self.policy_generator.uniform_policy()]
            self.weights = [1]
        else:
            self.policies = []
            self.weights = []

        self.densities = []
        self.objective_values_baseline = []
        self.visitations = []
        self.optimize_repetitions = optimize_repetitions
        self._precompute_emissions()

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

    def _precompute_emissions(self) -> None:
        """Precomputes an emission matrix given the saved environment
        """
        emissions = []
        for i in range(self.env.states_num):
            emissions.append(self.env.emissions[i])
        emissions = np.array(emissions)
        self.emissions = emissions

    def _density_oracle_single(self, policy: Policy) -> np.ndarray:
        """Computes state distribution induced by the given policy over a specified horizon

        Args:
            policy (Policy): inducing policy

        Returns:
            np.ndarray: 1-D array with density for each state
        """
        d0 = np.zeros(self.env.states_num)
        d0[self.env.init_state] = 1

        d = np.array(d0)
        temp = np.array(d0)

        if type(policy) is SimplePolicy:
            p_pi = (self.env.get_transition_matrix() *
                    np.expand_dims(policy.p, axis=2)).sum(axis=1)
            assert (np.allclose(p_pi.sum(axis=1), 1, rtol=1e-05, atol=1e-05))

            for _ in range(self.env.max_episode_length):
                temp = p_pi.T @ temp
                d += temp

        elif type(policy) is NonStationaryPolicy:
            for i in range(self.env.max_episode_length):
                p_pi = (self.env.get_transition_matrix() *
                        np.expand_dims(policy.ps[i], axis=2)).sum(axis=1)
                assert (np.allclose(p_pi.sum(axis=1), 1, rtol=1e-05, atol=1e-05))
                temp = p_pi.T @ temp
                d += temp

        return d / d.sum()

    def _density_oracle(self, actions: bool = False) -> np.ndarray:
        """Computes the combined state (or state-action) distribution induced by the saved policies

        Args:
            actions (bool, optional): if True, computes state-action distribution instead of state distribution. Defaults to False.

        Returns:
            np.ndarray: 1-D array with density for each state

        Raises:
            TypeError: if the saved policies are non-stationary
        """
        if actions:
            total_density = np.zeros((self.env.states_num, self.env.actions_num))
        else:
            total_density = np.zeros(self.env.states_num)

        for i, policy in enumerate(self.policies):
            if i >= len(self.densities):
                d = self._density_oracle_single(policy)
                self.densities.append(d)
            if actions:
                # TODO: doesn't work with non-stationary
                try:
                    total_density += self.weights[i] * policy.p * np.expand_dims(self.densities[i], axis=1)
                except:
                    raise TypeError('Non-stationary policies are not supported for state-action distribution')
            else:
                total_density += self.weights[i] * self.densities[i]
        return total_density

    def _planning_oracle(self, reward: np.ndarray) -> Policy:
        """Computes the optimal policy given a reward function and internal environment

        Args:
            reward (np.ndarray): reward function for each state

        Returns:
            Policy: policy solving the saved environment with the given reward function
        """
        solver = self.solver(self.env, reward)
        return solver.solve()

    def _reward_fn_gradient(self, distribution: np.ndarray) -> np.ndarray:
        """Computes the reward functional differentiated wrt to the state distribution

        Args:
            distribution (np.ndarray): state distribution to compute the reward function

        Returns:
            np.ndarray: gradient of the functional wrt to the state distribution - i.e. the reward function
        """
        grad_fn = getattr(self.objective, "gradient", None)
        if callable(grad_fn):
            return grad_fn(self.emissions, distribution)

        if self.objective.get_type() == "adaptive":
            grad_fn = grad(lambda d: self.objective.eval(self.emissions, d, self.visitations, self.episodes))
        else:
            grad_fn = grad(lambda d: self.objective.eval(self.emissions, d, self.episodes))
        return grad_fn(distribution)

    def _reward_fn_hessian(self, distribution: np.array)->np.array:
        if self.objective.get_type() == "adaptive":
            hes_fn = hessian(lambda d: self.objective.eval(self.emissions, d, self.visitations, self.episodes))
        else:
            hes_fn = hessian(lambda d: self.objective.eval(self.emissions, d, self.episodes))
        return hes_fn(distribution)

    def _update_data(self):
        if self.callback is not None:
            self.callback(self.trajectory, self.objective)

    def evaluate(
            self,
            SummarizedPolicyType: Type[SummarizedPolicy] = MixturePolicy,
            episodes: int = 100,
            plot: bool = True
    ) -> None:
        """Evaluates the saved policies using chosen summarization method

        Args:
            SummarizedPolicyType (Type[SummarizedPolicy], optional): method of policy summarization (Mixture, Density or Average). Defaults to MixturePolicy.
            episodes (int, optional): number of policy rollouts to execute. Defaults to 100.
            plot (bool, optional): if True, plots the heatmap based on the policy rollouts. Defaults to True.
        """
        empirical = np.zeros(len(self.policies))
        for _ in range(episodes):
            self.env.reset()

            if SummarizedPolicyType == DensityPolicy:
                summarized_policy = SummarizedPolicyType(
                    self.env, self._density_oracle(), self._density_oracle(actions=True)
                )
            elif SummarizedPolicyType == TrackingPolicy:
                summarized_policy = SummarizedPolicyType(
                    self.env, self.policies, self.weights, empirical
                )
                empirical[summarized_policy.get_picked_policy_id()] += 1
            else:
                summarized_policy = SummarizedPolicyType(
                    self.env, self.policies, self.weights
                )

            self.trajectory.append(self.env.init_state)
            for _ in range(self.env.max_episode_length):
                action = summarized_policy.next_action(self.env.state)
                next_state = self.env.step(action)
                self.trajectory.append(next_state)

            self._update_data()
            self.visitations.append(self.env.visitations / self.env.visitations.sum())

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

        # this is to ensure that the first policy has probability 1
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
            reward = self._reward_fn_gradient(density)
            if self.objective.get_type() != "adaptive":
                self.objective_values_baseline.append(self.objective.eval(self.emissions, density, self.episodes))
            new_policy = self._planning_oracle(reward)
            self.policies.append(new_policy)
            #
            # hess_min = np.min(np.linalg.eigh(self._reward_fn_hessian(density))[0])
            # hess_max = np.max(np.linalg.eigh(self._reward_fn_hessian(density))[0])
            hess_min = 0
            hess_max = 0
            # new base density to be added
            new_density = self._density_oracle_single(new_policy)

            if self.step == "line-search" and num_components > 1:
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
                objective = self.objective.eval(self.emissions, density, self.visitations, self.episodes)
            else:
                objective = self.objective.eval(self.emissions, density, self.episodes)


            empirical_gap = np.minimum(reward @ (new_density - density), empirical_gap)

            if verbose:
                print(f'component: {counter}, gap: {empirical_gap}, objective: {objective}, stepsize: {step_size}, gradient:{la.norm(reward)}, hess_max:{hess_max}, hess_min:{hess_min}')
            self.weights = [(1 - step_size) * weight for weight in self.weights] + [step_size]

            counter += 1

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
        else:
            pass  # can't happen, handled by argparser

    def run(
            self,
            num_components: int,
            episodes: int = 100,
            accuracy: float = None,
            SummarizedPolicyType: Type[SummarizedPolicy] = MixturePolicy,
            plot: bool = False,
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
                aggregate_distribution = (i * aggregate_distribution + self.visitations[i]) / (i + 1)

                if save_trajectory is not None:
                    np.savetxt(f"{save_trajectory}{i}.txt",
                               np.array([self.env.convert(state) for state in self.trajectory]))

                objective = self.objective.eval_full(
                    self.emissions, aggregate_distribution, episodes
                )

                #print (f'episode:{i}, value :{objective}', self.objective.eval(self.emissions, aggregate_distribution,self.visitations, episodes))
                self.objective_values_baseline.append(objective)
                run_objective_values.append(objective)

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
                run_objective_values.append(
                    self.objective.eval_full(self.emissions, aggregate_distribution, self.episodes))

            objective_values = run_objective_values
            objective_values = np.array(objective_values)

        if self.objective.get_type() != "adaptive":
            opt = self.objective.eval_full(self.emissions, self._density_oracle(), self.episodes)
        else:
            opt = None

        return objective_values, opt
        
