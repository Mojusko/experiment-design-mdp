import warnings
import logging
from doexpy.convex_solvers.frank_wolfe import FrankWolfe
from doexpy.mdpexplore import MdpExplore, MdpExploreMultiPolicy
from doexpy.solvers.dp import DP
from doexpy.feedback.feedback_base import EmptyFeedback
from doexpy.policies.summary_policies.density_policy import DensityPolicy

logger = logging.getLogger(__name__)


class SolverFactory:
    @staticmethod
    def create(cfg, env, design, feedback, same_first_action_in_episode: bool = False, embedder=None):
        # Check for REINFORCE algorithm (vocabulary-free mode)
        if cfg.algorithm == "reinforce":
            logger.info("Using REINFORCE optimizer (vocabulary-free mode)")
            from components.reinforce_optimizer import REINFORCEOptimizer

            # Get REINFORCE-specific config
            reinforce_cfg = cfg.get("reinforce", {})

            optimizer = REINFORCEOptimizer(
                num_policies=cfg.feedback.num_policies,
                model_name=reinforce_cfg.get("model_name", "gpt2"),
                lora_rank=reinforce_cfg.get("lora_rank", 8),
                lora_alpha=reinforce_cfg.get("lora_alpha", 16.0),
                lambda_reg=cfg.feedback.get("lambda", 1.0),
                learning_rate=reinforce_cfg.get("learning_rate", 1e-4),
                device="cuda",
                embedder=embedder,
            )

            # Wrap in a compatible interface
            return REINFORCEExplorerWrapper(
                optimizer=optimizer,
                cfg=cfg,
                reinforce_cfg=reinforce_cfg,
            )

        num_policies = cfg.feedback.num_policies
        adaptive_estimation_start = cfg.feedback.adaptive_estimation_start if cfg.algorithm != 'random' else 0
        total_episodes = cfg.experiment.episodes

        # Parse adaptive_design_frequency
        parsed_adaptive_design_freq = 0
        raw_adf = cfg.feedback.adaptive_design_frequency
        if isinstance(raw_adf, str) and raw_adf.startswith('/'):
            try:
                divisor = int(raw_adf[1:])
                if divisor > 0:
                    # total_episodes is already defined above
                    parsed_adaptive_design_freq = total_episodes // divisor
                else:
                    warnings.warn(f"adaptive_design_frequency divisor must be positive, got {divisor}. Defaulting to 0 (non-adaptive).")
            except ValueError:
                warnings.warn(f"Malformed adaptive_design_frequency string '{raw_adf}'. Expected format '/n'. Defaulting to 0 (non-adaptive).")
        elif isinstance(raw_adf, int):
            parsed_adaptive_design_freq = raw_adf
        else:
            warnings.warn(f"Unexpected type for adaptive_design_frequency: {type(raw_adf)}. Expected int or string like '/n'. Defaulting to 0 (non-adaptive).")

        use_random_initially = adaptive_estimation_start > 0 and total_episodes > adaptive_estimation_start

        if use_random_initially:
            random_solver = FrankWolfe(
                env=env, objective=design, num_components=1, num_summarized_policies=num_policies,
                initial_policy=True, solver=DP, SummarizedPolicyType=DensityPolicy,
                accuracy=cfg.accuracy, num_rounds=cfg.feedback.get('num_rounds', -1), stationary=True
            )
        else:
            random_solver = None

        # Determine FrankWolfe step argument based on feedback type
        fw_step_arg = {}
        if cfg.feedback.name == 'multinomial':
            fw_step_arg['step'] = 'line-search'
        # For numerical or other types, 'step' is not passed, FrankWolfe uses its default

        optimized_solver = FrankWolfe(
            env=env, objective=design,
            num_components=cfg.feedback.num_components if cfg.algorithm != 'random' else 1,
            num_summarized_policies=num_policies, initial_policy=cfg.algorithm=='random', solver=DP,
            **fw_step_arg,  # Pass step argument only if defined
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
            explorer_kwargs['adaptive_design_frequency'] = parsed_adaptive_design_freq # Use parsed value
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


class REINFORCEExplorerWrapper:
    """
    Wrapper that adapts REINFORCEOptimizer to the Explorer interface.

    This allows REINFORCE to be used in place of MdpExplore/MdpExploreMultiPolicy
    while maintaining a compatible interface for the experiment pipeline.

    Note: REINFORCE is vocabulary-free, so it doesn't use LLMGrid or produce
    traditional "visits". Instead, it optimizes GPT-2 policies and generates
    prompts that can be evaluated.
    """

    def __init__(self, optimizer, cfg, reinforce_cfg):
        """
        Initialize the wrapper.

        Args:
            optimizer: REINFORCEOptimizer instance
            cfg: Full experiment config
            reinforce_cfg: REINFORCE-specific config
        """
        self.optimizer = optimizer
        self.cfg = cfg
        self.reinforce_cfg = reinforce_cfg
        self.num_policies = cfg.feedback.num_policies

        # Store optimization history
        self.history = None
        self.generated_prompts = None

    def run(self, episodes, return_visitations=False, update_callback=None):
        """
        Run REINFORCE optimization.

        Note: "episodes" maps to "num_iterations" for REINFORCE.
        The concept is different but the config interface remains consistent.

        Args:
            episodes: Number of optimization iterations
            return_visitations: Whether to return generated prompts (mapped from visitations concept)
            update_callback: Optional callback for progress updates

        Returns:
            If return_visitations=True: (history, generated_prompts)
            Otherwise: history dict
        """
        # Map episodes to iterations
        num_iterations = self.reinforce_cfg.get("num_iterations", episodes)

        # Get generation parameters
        prompt_prefix = self.cfg.get("base_prompt", "")
        samples_per_policy = self.reinforce_cfg.get("samples_per_policy", 16)
        max_new_tokens = self.reinforce_cfg.get("max_new_tokens", 20)
        temperature = self.reinforce_cfg.get("temperature", 1.0)
        baseline = self.reinforce_cfg.get("baseline", "per_policy")

        # Run optimization
        self.history = self.optimizer.optimize(
            num_iterations=num_iterations,
            samples_per_policy=samples_per_policy,
            prompt_prefix=prompt_prefix,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            baseline=baseline,
            log_interval=max(1, num_iterations // 10),
            callback=update_callback,
        )

        # Generate evaluation prompts (analogous to visits)
        self.generated_prompts = self.optimizer.generate_evaluation_prompts(
            num_prompts_per_policy=samples_per_policy,
            prompt_prefix=prompt_prefix,
            max_new_tokens=max_new_tokens,
        )

        if return_visitations:
            # Return format compatible with existing pipeline
            # generated_prompts[q] is a list of prompts for policy q
            return self.history, self.generated_prompts

        return self.history

    def save_policies(self, path: str):
        """Save trained policies."""
        self.optimizer.save_policies(path)

    def load_policies(self, path: str):
        """Load trained policies."""
        self.optimizer.load_policies(path)
