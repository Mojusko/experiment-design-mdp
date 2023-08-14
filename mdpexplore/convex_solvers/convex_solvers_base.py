from abc import ABC, abstractmethod
import numpy as np
from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.policies.policy_base import Policy
from mdpexplore.policies.base_policies.non_stationary_policy import NonStationaryPolicy
from mdpexplore.policies.base_policies.stationary_policy import StationaryPolicy
from mdpexplore.policies.summary_policies.density_policy import DensityPolicy, MarginalDensityPolicy
from mdpexplore.policies.summary_policies.tracking_policy import TrackingPolicy
from mdpexplore.solvers.solver_base import DiscreteSolver
from mdpexplore.functionals.reward_functional import RewardFunctional
from mdpexplore.solvers.dp import DP
from mdpexplore.policies.policy_generator import PolicyGenerator

class ConvexSolverBase(ABC):
    def __init__(self, env : DiscreteEnv, objective : RewardFunctional, verbosity : int = 0, accuracy : float = None, initial_policy : bool = False, solver : DiscreteSolver = DP, summarization : str = None) -> None:
        self.env = env
        self.objective = objective
        self.verbosity = verbosity
        self.accuracy = accuracy
        self.initial_policy = initial_policy
        self.solver = solver
        self.SummarizedPolicyType = None
        #TODO: this needs to be more exhaustive and will result in errors if the solver is not DP
        if self.solver in [DP]:
            self.stationary = False
        else:
            self.stationary = True
        
        if self.initial_policy:
            self.initial_policy_generator = PolicyGenerator(self.env)
            self.policies = [self.initial_policy_generator.uniform_policy()]
            self.weights = [1.0]
        else:
            self.policies = []
            self.weights = []
        
        self.densities = []
    
    @abstractmethod
    def optimize(self, emissions, visitations, episodes) -> None:
        ...
    
    def summarize(self) -> Policy:
        empirical = np.zeros(len(self.policies))
        if self.SummarizedPolicyType == DensityPolicy:
                self.summarized_policy = self.SummarizedPolicyType(
                    self.env, self.density_estimator.density_oracle(self.policies, self.weights, self.densities)
                )
        elif self.SummarizedPolicyType == MarginalDensityPolicy:
            self.summarized_policy = self.SummarizedPolicyType(
                self.env, self.density_estimator.density_oracle(self.policies, self.weights, self.densities)
            )
        elif self.SummarizedPolicyType == TrackingPolicy:
            self.summarized_policy = self.SummarizedPolicyType(
                self.env, self.policies, self.weights, empirical
            )
            empirical[self.summarized_policy.get_picked_policy_id()] += 1
        else:
            self.summarized_policy = self.SummarizedPolicyType(
                self.env, self.policies, self.weights
            )
    
    def reset(self):

        if self.initial_policy:
            self.policies = [self.initial_policy_generator.uniform_policy()]
            self.weights = [1.0]
        else:
            self.policies = []
            self.weights = []
        
        self.densities = []