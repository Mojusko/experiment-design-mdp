from typing import Callable, Type, Union, Tuple
from datetime import datetime
import os
import autograd.numpy as np
import matplotlib.pyplot as plt
from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.policies.policy_base import Policy, SummarizedPolicy
from mdpexplore.policies.summary_policies.tracking_policy import TrackingPolicy
from mdpexplore.policies.summary_policies.mixture_policy import MixturePolicy
from mdpexplore.policies.summary_policies.density_policy import DensityPolicy, MarginalDensityPolicy
from mdpexplore.functionals.reward_functional import RewardFunctional
from mdpexplore.convex_solvers.convex_solvers_base import ConvexSolverBase
from mdpexplore.feedback.feedback_base import Feedback, EmptyFeedback
from mdpexplore.densities.density_estimators import TabularDensity, DeltaDensityEstimator
from mdpexplore.policies.general_policies.markovian_policy import MarkovianPolicy
from mdpexplore.policies.general_policies.non_markovian_policy import NonMarkovianPolicy

import time

class MdpExplore():
    def __init__(
            self,
            env: DiscreteEnv,
            objective: RewardFunctional,
            convex_solver: Type[ConvexSolverBase],
            verbosity: int = 0,
            optimize_repetitions: bool = False,
            feedback: Feedback = EmptyFeedback(),
            general_policy: str = 'markovian',
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
        self.feedback = feedback

        if general_policy == 'markovian':
            self.general_policy = MarkovianPolicy(self.env, self.convex_solver)
        elif general_policy == 'non-markovian':
            self.general_policy = NonMarkovianPolicy(self.env, self.convex_solver)
        else:
            raise NotImplementedError(f'General policy {general_policy} not implemented')

        # density estimator
        if self.env.type == 'discrete':
            self.density_estimator = TabularDensity(self.env, self.objective)
        elif self.env.type == 'continuous':
            self.density_estimator = DeltaDensityEstimator(self.env, self.objective)
        else:
            raise NotImplementedError(f'Density estimator for {self.env.type} environments not implemented')

        self.densities = []
        self.objective_values_baseline = []
        # we keep track of state and action visitations
        self.visitations = []
        # keep track of state and action visitations per episode
        self.state_visitations = []
        self.action_visitations = []
        
        self.optimize_repetitions = optimize_repetitions

        # if the environment is discrete, we can precompute the emission matrix
        if self.env.type == 'discrete':
            self._precompute_emissions()
        else:
            self.emissions = None

    def _reset(self, reset_visitations=True) -> None:
        """Resets  the trajectory, state_visitations, action_visitations

        """
        self.env.reset()
        self.convex_solver.reset()
        # reset the episode visitations
        self.state_visitations = []
        self.action_visitations = []

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

    # def _density_oracle_single(self, policy: Policy) -> np.ndarray:
    #     """Computes state distribution induced by the given policy over a specified horizon

    #     Args:
    #         policy (Policy): inducing policy

    #     Returns:
    #         np.ndarray: S x A (stationary) or H x S x A (non-stationary) array with density for each state
    #     """
    #     return self.density_estimator.density_oracle_single(policy)
    
    # def _density_oracle(self) -> np.ndarray:
    #     """Computes the combined state (or state-action) distribution induced by the saved policies

    #     Args:
    #         actions (bool, optional): if True, computes state-action distribution instead of state distribution. Defaults to False.

    #     Returns:
    #         np.ndarray: 1-D array with density for each state

    #     Raises:
    #         TypeError: if the saved policies are non-stationary
    #     """
    #     return self.density_estimator.density_oracle(self.policies, self.weights, self.densities, self.convex_solver.stationary)



    def evaluate(
            self,
            episodes: int = 100,
            keep = False, 
    ) -> None:
        """Evaluates the saved policies using chosen summarization method

        Args:
            SummarizedPolicyType (Type[SummarizedPolicy], optional): method of policy summarization (Mixture, Density or Average). Defaults to MixturePolicy.
            episodes (int, optional): number of policy rollouts to execute. Defaults to 100.
        """
        
        for _ in range(episodes):

            self._reset(reset_visitations=False)
   
            self.env.reset()

            self.trajectory.append(self.env.init_state)
            
            # count episode visitations
            self.state_visitations.append(self.env.init_state)

            for h in range(self.env.max_episode_length):
                action = self.general_policy.next_action(self.env.state, self.emissions, self.visitations, self.episodes, keep = keep)

                # feedback the state and action
                self.feedback.step_single(self.env.state, action)

                next_state = self.env.step(action)
                self.trajectory.append(next_state)
                
                # count episode visitations
                self.state_visitations.append(next_state)
                self.action_visitations.append(action)
            
            # per episode feedback 
            self.feedback.step_episode()
            
            # update the visitations
            self.visitations.append((self.state_visitations, self.action_visitations))
        

    def optimize_general_policy(self) -> None:
        """Optimizes the objective function using the specified method, returns the optimal policies and weights
        """
        self.general_policy.optimize(self.emissions, self.visitations, self.episodes)


    def run(
            self,
            episodes: int = 100,
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

                # evaluates one episode of the policy
                self.evaluate(1)
                
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
            self.optimize_general_policy()
            self.visitations = []

            # keep represent that the optimization is premade, and one does not need to reoptimize at deployment
            self.evaluate(episodes, keep=True)

            run_objective_values = []
            aggregate_distribution = 0

            for i, d in enumerate(self.visitations):
                aggregate_distribution = (i * aggregate_distribution + self.objective.build_density_from_trajectories([d])) / (i+1)
                
                run_objective_values.append(
                    self.objective.eval_full(self.emissions, aggregate_distribution, self.episodes))

            objective_values = run_objective_values
            objective_values = np.array(objective_values)

        if self.objective.get_type() != "adaptive":
            # here I want to evaluate on theoretical visitations, i.e., optimal 
            opt = self.objective.eval_full(self.emissions, self.general_policy.return_density(), self.episodes)
        else:
            opt = None

        if return_visitations:
            return objective_values, opt, self.visitations

        return objective_values, opt