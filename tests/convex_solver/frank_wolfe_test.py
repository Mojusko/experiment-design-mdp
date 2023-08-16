import numpy as np
import torch

from mdpexplore.env.bandits import ConstrainedMaxMovement
from mdpexplore.functionals.bandit_functionals import DesignBestArmLinearBanditNoDenominator
from mdpexplore.feedback.bandit_feedback import BanditFeedback
from mdpexplore.convex_solvers.frank_wolfe import FrankWolfe
from mdpexplore.solvers.dp import DP
from mdpexplore.solvers.lp import LP
from mdpexplore.policies.summary_policies.density_policy import DensityPolicy
from mdpexplore.solvers.solver_base import DiscreteSolver

from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.kernels import KernelFunction
from stpy.embeddings.embedding import HermiteEmbedding
from stpy.continuous_processes.nystrom_fea import NystromFeatures

import pytest

# define the problem
sigma = 0.05
delta = 1
lambd = 1
mix_objective = False

theta_star = np.array([1.0, 1.0]).reshape(-1)
action_space = np.array([[-0.5, 0.5],
                         [0.0, 0.0],
                         [0.5, -0.5],
                         [0.5, 0.5], 
                         [-0.5, -0.5]])

real_mu = theta_star @ action_space.T
optimal_action = np.argmax(real_mu)

@pytest.mark.parametrize("initial_policy", [False, True])
@pytest.mark.parametrize("solver", [DP, LP])
def test_optimize(initial_policy: bool, solver: DiscreteSolver):

    delta_mov = 1.0

    env = ConstrainedMaxMovement(
                action_space=action_space,
                action_space_pre_embedding=action_space,
                theta_star=theta_star,
                sigma=sigma,
                delta=delta_mov,
                max_episode_length = 10)

    design = DesignBestArmLinearBanditNoDenominator(
                env = env,
                lambd=lambd,
                sigma=sigma,
                mix_objectives=(mix_objective, 0.5),
                init_ucb = 3
            )
    
    convex_solver = FrankWolfe(env, objective=design, num_components = 5, solver = DP, SummarizedPolicyType = DensityPolicy, initial_policy = initial_policy)

    summarized_policy, policy, weights, densities = convex_solver.optimize(env.emissions, [], episodes = 10)

    assert len(policy) == 5, "number of policies should equal number of components"
    assert len(weights) == 5, "number of weights should equal number of components"
    
    assert len(densities) == 5, "number of densities should equal number of components"

    assert np.sum(weights) == 1, "weights should sum to 1"

    assert isinstance(summarized_policy, DensityPolicy), "summarized policy should be a density policy"