from typing import Callable, Type, Union, Tuple
from datetime import datetime
import os
import autograd.numpy as np
import matplotlib.pyplot as plt
from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.policies.policy_base import Policy, SummarizedPolicy
from mdpexplore.policies.tracking_policy import TrackingPolicy
from mdpexplore.policies.mixture_policy import MixturePolicy
from mdpexplore.policies.density_policy import DensityPolicy, MarginalDensityPolicy
from mdpexplore.functionals.reward_functional import RewardFunctional
from mdpexplore.convex_solvers.convex_solvers_base import ConvexSolverBase


class MdpExplore():
    def __init__(
            self,
            env: DiscreteEnv,
            objective: RewardFunctional,
            convex_solver: Type[ConvexSolverBase],
            verbosity: int = 0,
            optimize_repetitions: bool = False,
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
        self.convex_solver = convex_solver
        self.verbosity = verbosity
        self.convex_solver.verbosity = verbosity
        self.callback = callback

        self.densities = []
        self.objective_values_baseline = []
        # we keep track of state and action visitations
        self.visitations = []
        # keep track of state and action visitations per episode
        self.state_visitations = []
        self.action_visitations = []

        self.optimize_repetitions = optimize_repetitions
        self._precompute_emissions()

    def _reset(self, reset_visitations=True) -> None:
        """Resets the max-ent solver to its initial state
        """
        self.env.reset()
        self.convex_solver.reset()
        # reset the episode visitations
        self.state_visitations = []
        self.action_visitations = []

        # self.densities = []
        self.trajectory = []
        if reset_visitations:
            # reset visitations if needed
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
            np.ndarray: S x A (stationary) or H x S x A (non-stationary) array with density for each state
        """
        return self.convex_solver._density_oracle_single(policy)
    
    def _density_oracle(self, actions: bool = True) -> np.ndarray:
        """Computes the combined state (or state-action) distribution induced by the saved policies

        Args:
            actions (bool, optional): if True, computes state-action distribution instead of state distribution. Defaults to False.

        Returns:
            np.ndarray: 1-D array with density for each state

        Raises:
            TypeError: if the saved policies are non-stationary
        """
        return self.convex_solver._density_oracle(actions)

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
        # if solver is cvxpy, default to MixturePolicy
        if (self.convex_solver.type == "cvxpy") & (SummarizedPolicyType is not MixturePolicy):
            print('When using cvxpy solver, summarization method is changed to MixturePolicy')
            SummarizedPolicyType = MixturePolicy

        for _ in range(episodes):
            self.env.reset()

            if SummarizedPolicyType == DensityPolicy:
                summarized_policy = SummarizedPolicyType(
                    self.env, self._density_oracle(actions=True)
                )
            elif SummarizedPolicyType == MarginalDensityPolicy:
                summarized_policy = SummarizedPolicyType(
                    self.env, self._density_oracle(actions=True)
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
            # count episode visitations
            self.state_visitations.append(self.env.init_state)
            for h in range(self.env.max_episode_length):
                action = summarized_policy.next_action(self.env.state)
                next_state = self.env.step(action)
                self.trajectory.append(next_state)
                # count episode visitations
                self.state_visitations.append(next_state)
                self.action_visitations.append(action)
            
            self._update_data()
            # update the visitations
            self.visitations.append((self.state_visitations, self.action_visitations))
        

    def optimize(self) -> None:
        """Optimizes the objective function using the specified method, returns the optimal policies and weights
        """
        self.policies, self.weights, self.densities = self.convex_solver.optimize(self.emissions, self.visitations, self.episodes)

    def run(
            self,
            episodes: int = 100,
            SummarizedPolicyType: Type[SummarizedPolicy] = MixturePolicy,
            plot: bool = False,
            save_trajectory: Union[str, None] = None,
            return_visitations: bool = False,
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

            for i in range(episodes):
                if self.verbosity > 2:
                    print("Episode:", i)

                self._reset(reset_visitations=False)

                self.optimize()

                self.evaluate(SummarizedPolicyType, 1)
                
                # calculate the aggregate distribution
                aggregate_distribution = self.objective.build_density_from_trajectories(self.visitations)

                if save_trajectory is not None:
                    np.savetxt(f"{save_trajectory}{i}.txt",
                               np.array([self.env.convert(state) for state in self.trajectory]))

                objective = self.objective.eval_full(
                    self.emissions, aggregate_distribution, episodes
                )

                print (f'episode:{i}, value :{objective}', self.objective.eval(self.emissions, aggregate_distribution, self.visitations, episodes))
                self.objective_values_baseline.append(objective)
                run_objective_values.append(objective)

            objective_values = run_objective_values

        else:
            self._reset()
            self.episodes = episodes
            self.optimize()

            self.visitations = []
            self.evaluate(SummarizedPolicyType, episodes)

            run_objective_values = []
            aggregate_distribution = 0

            for i, d in enumerate(self.visitations):
                aggregate_distribution = (i * aggregate_distribution + self.objective.build_density_from_trajectories(d)) / (i+1)
                run_objective_values.append(
                    self.objective.eval_full(self.emissions, aggregate_distribution, self.episodes))

            objective_values = run_objective_values
            objective_values = np.array(objective_values)

        if self.objective.get_type() != "adaptive":
            opt = self.objective.eval_full(self.emissions, self._density_oracle(), self.episodes)
        else:
            opt = None

        if return_visitations:
            return objective_values, opt, self.visitations

        return objective_values, opt