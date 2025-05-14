import warnings
from doexpy.convex_solvers.frank_wolfe import FrankWolfe
from doexpy.mdpexplore import MdpExplore, MdpExploreMultiPolicy
from doexpy.solvers.dp import DP
from doexpy.feedback.feedback_base import EmptyFeedback
from doexpy.policies.summary_policies.density_policy import DensityPolicy

class SolverFactory:
    @staticmethod
    def create(cfg, env, design, feedback, same_first_action_in_episode: bool = False): # Add flag here
        num_policies = cfg.feedback.num_policies
        adaptive_estimation_start = cfg.feedback.adaptive_estimation_start if cfg.algorithm != 'random' else 0
        total_episodes = cfg.experiment.episodes

        use_random_initially = adaptive_estimation_start > 0 and total_episodes > adaptive_estimation_start

        if use_random_initially:
            random_solver = FrankWolfe(
                env=env, objective=design, num_components=1, num_summarized_policies=num_policies,
                initial_policy=True, solver=DP, SummarizedPolicyType=DensityPolicy,
                accuracy=cfg.accuracy, num_rounds=cfg.feedback.get('num_rounds', -1), stationary=True
            )
        else:
            random_solver = None

        optimized_solver = FrankWolfe(
            #step='line-search',
            env=env, objective=design,
            num_components=cfg.feedback.num_components if cfg.algorithm != 'random' else 1,
            num_summarized_policies=num_policies, initial_policy=cfg.algorithm=='random', solver=DP,
            SummarizedPolicyType=DensityPolicy, accuracy=cfg.accuracy,
            num_rounds=cfg.feedback.get('num_rounds', -1), stationary=True
        )

        explorer_cls = MdpExplore if num_policies == 1 else MdpExploreMultiPolicy

        explorer_kwargs = {
            'env': env, 'objective': design, 'convex_solver': optimized_solver, 'verbosity': 3,
            'feedback': EmptyFeedback(env, design), 'general_policy': 'markovian',
            # Pass the flag to the explorer constructor kwargs
            'same_first_action_in_episode': same_first_action_in_episode
        }
        if explorer_cls is MdpExploreMultiPolicy:
            explorer_kwargs['num_policies'] = num_policies
            explorer_kwargs['adaptive_design_frequency'] = cfg.feedback.adaptive_design_frequency
            # Remove the flag if it was added generically, as it's specific to MdpExploreMultiPolicy
            # explorer_kwargs.pop('same_first_action_in_episode', None) # Keep it, MdpExploreMultiPolicy needs it
        else:
             # Remove the flag if the explorer is not MdpExploreMultiPolicy
             explorer_kwargs.pop('same_first_action_in_episode', None)

        if use_random_initially:
            explorer = TwoPhaseExplorer(
                random_solver=random_solver, optimized_solver=optimized_solver,
                adaptive_estimation_start=adaptive_estimation_start, total_episodes=total_episodes,
                explorer_cls=explorer_cls, explorer_kwargs=explorer_kwargs
            )
        else:
            explorer = explorer_cls(**explorer_kwargs)

        return explorer

class TwoPhaseExplorer:
    def __init__(self, random_solver, optimized_solver, adaptive_estimation_start, total_episodes, explorer_cls, explorer_kwargs):
        # explorer_kwargs already contains 'same_first_action_in_episode' if applicable
        self.random_solver = random_solver
        self.optimized_solver = optimized_solver
        self.adaptive_estimation_start = adaptive_estimation_start
        self.total_episodes = total_episodes
        self.explorer_cls = explorer_cls
        self.explorer_kwargs = explorer_kwargs  # Add this line

        # Create a copy of explorer_kwargs for the random explorer and update convex_solver
        random_explorer_kwargs = explorer_kwargs.copy()
        random_explorer_kwargs['convex_solver'] = self.random_solver
        self.random_explorer = self.explorer_cls(**random_explorer_kwargs)

        # Use the original explorer_kwargs for the optimized explorer
        self.optimized_explorer = self.explorer_cls(**explorer_kwargs)


    def run(self, episodes, return_visitations=False, update_callback=None):
        all_visits = [[] for _ in range(self.random_explorer.num_policies)]
        global_ep_idx = 0
        # Run random phase
        if self.adaptive_estimation_start > 0:
            random_results = self.random_explorer.run(
                episodes=self.adaptive_estimation_start, 
                return_visitations=True,
                update_callback=update_callback
            )
            random_visits = random_results[-1]
            for policy_idx, visits in enumerate(random_visits):
                all_visits[policy_idx].extend(visits)
            # Pass visitations to optimized explorer
            self.optimized_explorer.visitations_per_policy = all_visits
        global_ep_idx += self.adaptive_estimation_start
        # Run optimized phase
        optimized_results = self.optimized_explorer.run(
            episodes=episodes,
            start_ep_idx=global_ep_idx,
            return_visitations=return_visitations,
            update_callback=update_callback
        )
        if return_visitations:
            return optimized_results
        return optimized_results[:-1]
