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
action_space = torch.tensor([[-0.5, -0.5],
                        [-0.5, 0.5],
                         [0.0, 0.0],
                         [0.5, -0.5],
                         [0.5, 0.5]])

real_mu = theta_star @ action_space.T
optimal_action = torch.argmax(real_mu).item()

@pytest.mark.parametrize("markovian", [True, False])
def test_feedback(markovian: bool):

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

    initial_ucbs = design.ucbs.clone()

    feedback = BanditFeedback(env, design, estimator, theta_star = theta_star, sigma = sigma, markovian = markovian)

    state_action_trajectory = [[[0, 1], [1, 2], [2, 3], [3, 4], [4, 3]]]

    episode = 0

    for h in range(5):
        state, action = state_action_trajectory[episode][h]
        feedback.step_single(state, action)

        assert len(feedback.state_trajectory) == h + 1, "state trajectory length is wrong"
        assert len(feedback.action_trajectory) == h + 1, "action trajectory length is wrong"
    
    if markovian:
        assert torch.all(initial_ucbs == feedback.objective.ucbs), "ucbs should not change until the end of the episode in markovian case"
    else:
        assert torch.all(initial_ucbs != feedback.objective.ucbs), "ucbs should change at each step in non-markovian case"
    
    feedback.step_episode()

    if markovian:
        assert torch.all(initial_ucbs != feedback.objective.ucbs), "ucbs should change at the end of the episode in markovian case"
    
    assert len(feedback.state_trajectory) == 0, "state trajectory should be empty after episode"
    assert len(feedback.action_trajectory) == 0, "action trajectory should be empty after episode"

    assert torch.all(torch.isclose(feedback.estimator.x, torch.tensor(np.array([action_space[act_idx] for act_idx in [1, 2, 3, 4, 3]])))), "data stored in estimator is wrong"
    assert torch.all(feedback.objective.ucbs >= feedback.objective.lcbs), "ucbs should be greater than lcbs"

test_feedback(True)