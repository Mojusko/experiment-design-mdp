from doexpy.convex_solvers.frank_wolfe import FrankWolfe
from doexpy.mdpexplore import MdpExplore, MdpExploreMultiPolicy
from doexpy.solvers.dp import DP
from doexpy.feedback.feedback_base import EmptyFeedback
from doexpy.policies.summary_policies.density_policy import DensityPolicy

class SolverFactory:
    @staticmethod
    def create(cfg, env, design, feedback):
        """
        Create a FrankWolfe solver and an appropriate explorer based on the configuration.
    
        Parameters:
            cfg (DictConfig): The Hydra configuration object.
            env: The environment object.
            design: The design object.
            feedback: The feedback object.
    
        Returns:
            explorer: An instance of MdpExplore or MdpExploreMultiPolicy.
        """
        # Determine if the algorithm initializes with a policy
        initial_policy = cfg.algorithm == 'random'
        #initial_policy = True
    
        num_components = cfg.feedback.num_components if cfg.algorithm != 'random' else 1
    
        num_policies = cfg.feedback.num_policies
    
        # Initialize the FrankWolfe solver
        solver = FrankWolfe(
            env=env,
            objective=design,
            num_components=num_components,
            num_summarized_policies=num_policies,
            solver=DP,
            initial_policy=initial_policy,
            SummarizedPolicyType=DensityPolicy,
            accuracy=cfg.accuracy,
            #step='line-search',
            num_rounds=cfg.feedback.get('num_rounds',-1),
            stationary=True # DEBUG
        )
    
        # Select the appropriate explorer based on the number of policies
        explorer_cls = MdpExplore if num_policies == 1 else MdpExploreMultiPolicy
    
        # Prepare common arguments for both explorers
        explorer_kwargs = {
            'env': env,
            'objective': design,
            'convex_solver': solver,
            'verbosity': 3,
            #'verbosity': 0,
            'feedback': EmptyFeedback(env,design),
            'general_policy': 'markovian'
        }
    
        # Add specific arguments for MdpExploreMultiPolicy
        if explorer_cls is MdpExploreMultiPolicy:
            explorer_kwargs['num_policies'] = num_policies
            explorer_kwargs['adaptive_design_frequency'] = cfg.feedback.adaptive_design_frequency
    
        # Instantiate the explorer
        explorer = explorer_cls(**explorer_kwargs)
    
        return explorer
