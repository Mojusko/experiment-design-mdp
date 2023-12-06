import numpy as np
import torch

from mdpexplore.env.bandits import ConstrainedMaxMovement
from mdpexplore.env.continuous_bandits import ContinuousMovementConstrainedBayesianOptimization
from mdpexplore.env.env_dummy_testing import DummyTestEnvContinuous
from mdpexplore.functionals.bandit_functionals import DesignBestArmLinearBanditNoDenominator, DesignBestArmLinearBanditNoDenominatorContinuous
from mdpexplore.feedback.bandit_feedback import BanditFeedback
from mdpexplore.convex_solvers.frank_wolfe import FrankWolfe
from mdpexplore.solvers.dp import DP
from mdpexplore.solvers.lp import LP
from mdpexplore.solvers.ddpg import DDPG
from mdpexplore.policies.summary_policies.density_policy import DensityPolicy
from mdpexplore.policies.summary_policies.mixture_policy import MixturePolicy
from mdpexplore.solvers.solver_base import DiscreteSolver

from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.kernels import KernelFunction
from stpy.embeddings.embedding import HermiteEmbedding
from stpy.continuous_processes.nystrom_fea import NystromFeatures

import pytest

torch.set_default_dtype(torch.float64)

@pytest.mark.parametrize("initial_policy", [False, True])
@pytest.mark.parametrize("solver", [DP])
def test_optimize(initial_policy: bool, solver: DiscreteSolver):

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
    
    convex_solver = FrankWolfe(env, objective=design, num_components = 5, solver = solver, SummarizedPolicyType = DensityPolicy, initial_policy = initial_policy)

    summarized_policy, policy, weights, densities = convex_solver.optimize(env.emissions, [], episodes = 10)

    assert len(policy) == 5, "number of policies should equal number of components"
    assert len(weights) == 5, "number of weights should equal number of components"
    
    assert len(densities) == 5, "number of densities should equal number of components"

    assert np.sum(weights) == 1, "weights should sum to 1"

    assert isinstance(summarized_policy, DensityPolicy), "summarized policy should be a density policy"

def test_optimize_continuous():


    # define the problem
    sigma = 0.05
    lambd = 1
    max_episode_length = 5

    theta_star = np.array([1.0]).reshape(-1)

    delta_mov = 0.2

    env = ContinuousMovementConstrainedBayesianOptimization(
        states_dim = 1,
        actions_dim = 1,
        theta_star = theta_star,
        sigma = sigma,
        max_episode_length = max_episode_length,
        min_action = -delta_mov,
        max_action = delta_mov
    )

    design = DesignBestArmLinearBanditNoDenominatorContinuous(
        env = env,
        lambd=lambd,
        sigma=sigma
    )
    
    convex_solver = FrankWolfe(env, objective=design, num_components = 3, solver = DDPG, SummarizedPolicyType = MixturePolicy, verbosity = 4)

    summarized_policy, policy, weights, densities = convex_solver.optimize(env.emissions, [], episodes = 10)

    assert len(policy) == 3, "number of policies should equal number of components"
    assert len(weights) == 3, "number of weights should equal number of components"
    
    assert len(densities) == 3, "number of densities should equal number of components"

    assert np.sum(weights) == 1, "weights should sum to 1"

    assert isinstance(summarized_policy, MixturePolicy), "summarized policy should be a density policy"