import numpy as np
import torch

from doexpy.env.bandits import ConstrainedMaxMovement
from doexpy.functionals.bandit_functionals import DesignBestArmLinearBanditNoDenominator
from doexpy.feedback.bandit_feedback import BanditFeedback
from doexpy.convex_solvers.frank_wolfe import FrankWolfe
from doexpy.solvers.dp import DP
from doexpy.policies.summary_policies.density_policy import DensityPolicy
from doexpy.mdpexplore import MdpExplore

from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.kernels import KernelFunction
from stpy.embeddings.embedding import HermiteEmbedding
from stpy.continuous_processes.nystrom_fea import NystromFeatures

import pytest

# set default tensor type
torch.set_default_tensor_type(torch.DoubleTensor)

# define the problem
sigma = 0.05
delta = 1
lambd = 1

theta_star = torch.tensor([1.0, 1.0]).reshape(-1)
action_space = torch.tensor([[-0.5, 0.5],
                         [0.0, 0.0],
                         [0.5, -0.5],
                         [0.5, 0.5], 
                         [-0.5, -0.5]])

real_mu = theta_star @ action_space.T
optimal_action = torch.argmax(real_mu).item()

@pytest.mark.parametrize("policy_type", ['markovian', 'non-markovian'])
def test_bandits(policy_type: str):
    if policy_type == 'markovian':
        markovian = True
    elif policy_type == 'non-markovian':
        markovian = False

    kernel = KernelFunction(kernel_name='linear', d = 2)
    estimator = GaussianProcess(kernel=kernel, s = sigma)

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
                init_ucb = 3
            )
    
    feedback = BanditFeedback(env, design, estimator, theta_star = theta_star, sigma = sigma, markovian = markovian)
    
    convex_solver = FrankWolfe(env, objective=design, num_components = 1, solver = DP, SummarizedPolicyType = DensityPolicy)

    me = MdpExplore(
            env,
            objective=design,
            convex_solver = convex_solver,
            verbosity=4,
            feedback=feedback,
            general_policy = policy_type
            )
    
    val, opt_val, visitations = me.run(
            episodes=5,
            return_visitations=True
        )
    
    # check the valid set
    mask = torch.max(design.lcbs) <= design.ucbs
    mask = mask.reshape(-1)
    # assert there is only one optimal action
    assert torch.sum(mask) == 1, "There should be only one optimal action"
    # assert the optimal action is the one we expect
    assert torch.all(action_space[mask] == action_space[optimal_action]), "The optimal action is incorrect"
