from typing import Callable, Type, Union, Tuple, List
from datetime import datetime
import torch 
import os
import copy
import autograd.numpy as np
import matplotlib.pyplot as plt
from doexpy.env.discrete_env import DiscreteEnv
from doexpy.policies.policy_base import Policy, SummarizedPolicy
from doexpy.policies.summary_policies.tracking_policy import TrackingPolicy
from doexpy.policies.summary_policies.mixture_policy import MixturePolicy
from doexpy.policies.summary_policies.density_policy import DensityPolicy, MarginalDensityPolicy
from doexpy.functionals.reward_functional import RewardFunctional
from doexpy.convex_solvers.convex_solvers_base import ConvexSolverBase
from doexpy.feedback.feedback_base import Feedback, EmptyFeedback
from doexpy.densities.density_estimators import TabularDensity, DeltaDensityEstimator
from doexpy.policies.general_policies.markovian_policy import MarkovianPolicy
from doexpy.policies.general_policies.non_markovian_policy import NonMarkovianPolicy

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
        for i in range(self.env.emiss_num):
            emissions.append(self.env.emissions[i].view(1,-1))
            
        self.emissions = torch.vstack(emissions)

   
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
                state = copy.copy(self.env.state)

                # feedback the state and action
                self.feedback.step_single(self.env.state, action)

                next_state = self.env.step(action)

                if self.verbosity > 3:
                    print (h,state,action,next_state)

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
            if self.verbosity > 0:
                print ("Optimizing starting with budget: ", episodes)
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
            objective_values = [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in run_objective_values]
            objective_values = np.array(objective_values)


        if self.objective.get_type() != "adaptive":
            opt = self.objective.eval_full(self.emissions, self.general_policy.return_density(), self.episodes)
            if isinstance(opt, torch.Tensor):
                opt = opt.cpu().numpy()
        else:
            opt = None
           
        if return_visitations:
            return objective_values, opt, self.visitations

        return objective_values, opt

class DummyOptimizer:
    def optimize(self, *args, **kwargs):
        raise ValueError(f'Policies belonging to MdpExploreMultiPolicy should only be jointly optimized')

class MdpExploreMultiPolicy:
    def __init__(
            self,
            env: DiscreteEnv,
            objective: RewardFunctional,
            convex_solver: Type[ConvexSolverBase],
            num_policies: int,
            verbosity: int = 0,
            optimize_repetitions: bool = False,
            feedback: Feedback = EmptyFeedback(),
            general_policy: str = 'markovian',
            adaptive_design_frequency: int = 1,
    ) -> None:
        """Class containing components required to run the maximum entropy exploration algorithm with multiple policies
        
        Args:
            env: environment to be solved with max-ent
            objective: reward functional to generate the reward functions
            convex_solver: MDP solver to be used as planning oracle
            num_policies: number of policies to maintain
            verbosity: level of information logging
            optimize_repetitions: whether to optimize repetitions
            feedback: feedback mechanism
            general_policy: type of policy ('markovian' or 'non-markovian')
        """
        self.env = env
        self.objective = objective
        self.convex_solver = convex_solver
        self.verbosity = verbosity
        self.convex_solver.verbosity = verbosity
        self.feedback = feedback
        self.num_policies = num_policies
        
        # Initialize lists to store per-policy information
        self.general_policies = []
        self.objective_values_baseline_per_policy = [[] for _ in range(num_policies)]
        self.visitations_per_policy = [[] for _ in range(num_policies)]
        self.adaptive_design_frequency = adaptive_design_frequency
        
        # Initialize policies
        policy_class = MarkovianPolicy if general_policy == 'markovian' else NonMarkovianPolicy
        for _ in range(num_policies):
            # The solver acts jointly for all policies.
            # Individual policies shouldn't have solver - set to dummy
            self.general_policies.append(policy_class(self.env, DummyOptimizer()))
            
        # density estimator
        if self.env.type == 'discrete':
            self.density_estimator = TabularDensity(self.env, self.objective)
        elif self.env.type == 'continuous':
            self.density_estimator = DeltaDensityEstimator(self.env, self.objective)
        else:
            raise NotImplementedError(f'Density estimator for {self.env.type} environments not implemented')
        
        self.optimize_repetitions = optimize_repetitions
        
        # Precompute emissions for discrete environments
        if self.env.type == 'discrete':
            self._precompute_emissions()
        else:
            self.emissions = None
            
    def _reset(self, reset_visitations=True) -> None:
        """Resets trajectories and visitations for all policies"""
        self.env.reset()
        self.convex_solver.reset()
        
        # Reset per-policy tracking
        self.state_visitations_per_policy = [[] for _ in range(self.num_policies)]
        self.action_visitations_per_policy = [[] for _ in range(self.num_policies)]
        self.trajectory_per_policy = [[] for _ in range(self.num_policies)]
        
        if reset_visitations:
            self.visitations_per_policy = [[] for _ in range(self.num_policies)]
            self.objective_values_baseline_per_policy = [[] for _ in range(self.num_policies)]
            
    def _precompute_emissions(self) -> None:
        """Precomputes emission matrix for discrete environments"""
        emissions = []
        for i in range(self.env.emiss_num):
            emissions.append(self.env.emissions[i].view(1,-1))
        self.emissions = torch.vstack(emissions)
        
    #def _copy_policy_to_others(self) -> None:
    #    """Copies the first policy to all other policies"""
    #    for i in range(1, self.num_policies):
    #        self.general_policies[i].summarized_policy = copy.deepcopy(self.general_policies[0].summarized_policy)
    #        self.general_policies[i].policies = copy.deepcopy(self.general_policies[0].policies)
    #        self.general_policies[i].weights = copy.deepcopy(self.general_policies[0].weights)
    #        self.general_policies[i].densities = copy.deepcopy(self.general_policies[0].densities)
            
    def optimize_policies(self) -> None:
        """Optimizes all policies jointly"""
        summarized_policies, policies, weights, densities = self.convex_solver.optimize(self.emissions, self.visitations_per_policy, self.episodes)
        # Assign results to each policy
        for i, policy in enumerate(self.general_policies):
            policy.summarized_policy = summarized_policies[i]
            policy.policies = policies[i]
            policy.weights = weights[i]
            policy.densities = densities[i]
        
    def evaluate(self, episodes: int = 100, keep: bool = False) -> None:
        """Evaluates all policies for given number of episodes
        
        Args:
            episodes: number of episodes to evaluate
            keep: whether to keep existing optimization or reoptimize
        """
        for _ in range(episodes):
            self._reset(reset_visitations=False)
            self.env.reset()
            
            # Initialize trajectories for all policies with initial state
            for policy_idx in range(self.num_policies):
                self.trajectory_per_policy[policy_idx].append(self.env.init_state)
                self.state_visitations_per_policy[policy_idx].append(self.env.init_state)
            
            # Step through episode
            for h in range(self.env.max_episode_length):
                if all(policy.time == 0 for policy in self.general_policies) and not keep:
                    self.optimize_policies()

                # Get actions from all policies
                for policy_idx, policy in enumerate(self.general_policies):

                    action = policy.next_action(self.env.state, self.emissions, 
                                    self.visitations_per_policy[policy_idx], self.episodes, True)
                        
                    state = copy.copy(self.env.state)
                    self.feedback.step_single(state, action)
                    next_state = self.env.step(action)
                    
                    if self.verbosity > 3:
                        print(f"Policy {policy_idx}, Step {h}: {state} -> {action} -> {next_state}")
                    
                    self.trajectory_per_policy[policy_idx].append(next_state)
                    self.state_visitations_per_policy[policy_idx].append(next_state)
                    self.action_visitations_per_policy[policy_idx].append(action)
                    
                    # Reset environment for next policy
                    if policy_idx < self.num_policies - 1:
                        self.env.state = state
                
            self.feedback.step_episode()
            
            # Update visitations for each policy
            for policy_idx in range(self.num_policies):
                self.visitations_per_policy[policy_idx].append(
                    (self.state_visitations_per_policy[policy_idx],
                     self.action_visitations_per_policy[policy_idx])
                )

    def run(
        self,
        episodes: int = 100,
        save_trajectory: Union[str, None] = None,
        return_visitations: bool = False,
        update_callback=None,
        start_ep_idx: int = 0,
    ):
        """
        Runs the maximum-entropy exploration procedure in a single pass over `episodes`.
        If an `update_callback` is provided, it will be called after each episode, allowing
        partial estimator updates, design updates, etc.
    
        Args:
            episodes (int): total number of episodes to run.
            save_trajectory (str or None): path prefix for saving trajectories (if needed).
            return_visitations (bool): if True, return the final visitations array.
            update_callback (callable or None): a function `fn(ep_idx, new_visits)` called
                after each episode, where `new_visits` is the newly collected trajectories
                for each policy. Defaults to None.
            start_ep_idx (int): starting episode index for global tracking.
    
        Returns:
            (objective_values, opt) or (objective_values, opt, visitations_per_policy)
            if `return_visitations` is True.
        """
        # Reset and set total episodes
        self._reset(reset_visitations=False)
        self.episodes = episodes
    
        # We'll track a per-episode objective
        run_objective_values = []
    
        # If objective is NOT adaptive, we typically do a one-time policy optimization before running
        if self.objective.get_type() != "adaptive":
            if self.verbosity > 0:
                print("Optimizing starting with budget:", episodes)
            self.optimize_policies()
    
        # Main loop: run exactly one new episode each iteration
        for ep_i in range(start_ep_idx, episodes):
            if self.verbosity > 2:
                print("Episode:", ep_i)

            # If user provided a callback, call it to do partial re-fitting, etc.
            if update_callback is not None and ep_i > 0:
                # Now, after evaluating, the new visit is available
                new_visits = [
                    self.visitations_per_policy[p_i][-1]
                    for p_i in range(self.num_policies)
                ]
                update_callback(ep_i, new_visits)
    

            # Handle adaptive policy optimization
            if self.objective.get_type() == "adaptive":
                if ep_i % self.adaptive_design_frequency == 0 or ep_i == start_ep_idx:
                    if self.verbosity > 1:
                        print(f"Re-optimizing policies at episode {ep_i}")
                    self.env.reset()
                    self.optimize_policies()
    
            # Evaluate exactly 1 episode for each policy
            # 'keep=True' means we keep the trajectories for callback processing
            self.evaluate(episodes=1, keep=True)
    
                        # Optionally save trajectory
            if save_trajectory is not None:
                for policy_idx in range(self.num_policies):
                    np.savetxt(
                        f"{save_trajectory}_policy{policy_idx}_{ep_i}.txt",
                        np.array([
                            self.env.convert(s)
                            for s in self.trajectory_per_policy[policy_idx]
                        ])
                    )
    
            # For logging, compute the objective so far
            if self.verbosity > 2 and (ep_i % 25 == 0 or ep_i == episodes-1):
                aggregate_distributions = []
                for policy_idx in range(self.num_policies):
                    agg_dist = self.objective.build_density_from_trajectories(
                        self.visitations_per_policy[policy_idx]
                    )
                    aggregate_distributions.append(agg_dist)
    
                val = self.objective.eval_full(self.emissions, aggregate_distributions, episodes)
                if isinstance(val, torch.Tensor):
                    val = val.detach()
                print(f"Episode {ep_i}, Objective Value: {val}")
                run_objective_values.append(val)
    
        # Convert objective values to NumPy
        objective_values = []
        for val in run_objective_values:
            if isinstance(val, torch.Tensor):
                objective_values.append(val.cpu().numpy())
            else:
                objective_values.append(val)
        objective_values = np.array(objective_values)
    
        # Handle optional "optimal" value computation if not adaptive
        if self.objective.get_type() != "adaptive":
            densities = [policy.return_density() for policy in self.general_policies]
            opt = self.objective.eval_full(self.emissions, densities, self.episodes)
            if isinstance(opt, torch.Tensor):
                opt = opt.detach().cpu().numpy()
        else:
            opt = None
    
        if return_visitations:
            return objective_values, opt, self.visitations_per_policy
    
        return objective_values, opt
                    
