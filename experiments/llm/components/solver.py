from doexpy.convex_solvers.frank_wolfe import FrankWolfe
from doexpy.mdpexplore import MdpExplore, MdpExploreMultiPolicy
from doexpy.solvers.dp import DP
from doexpy.policies.summary_policies.density_policy import DensityPolicy

class SolverFactory:
    @staticmethod
    def create(cfg, env, design, feedback):
        """
        Based on cfg.algorithm, create a FrankWolfe solver and 
        MdpExplore / MdpExploreMultiPolicy explorer.
        """
        # default
        if cfg.algorithm == 'random':
            initial_policy = True
            num_components = 1
        elif cfg.algorithm == 'optim':
            # numerical => ~750, multinomial => ~75, etc.
            if cfg.feedback_type == 'numerical':
                num_components = 750
            else:
                num_components = 75
            initial_policy = False
        else:
            # 'greedy' or anything else
            if cfg.feedback_type == 'numerical':
                num_components = 750
            else:
                num_components = 75
            initial_policy = False

        # If user explicitly overrides num_components in CLI:
        if cfg.get("num_components") is not None:
            num_components = cfg.num_components

        num_policies = 1 if cfg.feedback_type == 'numerical' else 2

        solver = FrankWolfe(
            env=env,
            objective=design,
            num_components=num_components,
            num_summarized_policies=num_policies,
            solver=DP,
            initial_policy=initial_policy,
            SummarizedPolicyType=DensityPolicy,
            accuracy=cfg.accuracy,
            step='line-search',
        )

        if cfg.feedback_type == 'numerical':
            explorer = MdpExplore(
                env=env,
                objective=design,
                convex_solver=solver,
                verbosity=3,
                feedback=feedback.feedback,  # or just feedback if you prefer
                general_policy='markovian'
            )
        else:
            explorer = MdpExploreMultiPolicy(
                num_policies=num_policies,
                env=env,
                objective=design,
                convex_solver=solver,
                verbosity=3,
                feedback=feedback.feedback,
                general_policy='markovian'
            )

        return explorer
