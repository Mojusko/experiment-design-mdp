import numpy as np
import torch

from mdpexplore.env.bandits import ConstrainedMaxMovement
from mdpexplore.functionals.bandit_functionals import DesignBestArmLinearBanditNoDenominator
from mdpexplore.feedback.bandit_feedback import BanditFeedback
from mdpexplore.convex_solvers.frank_wolfe import FrankWolfe
from mdpexplore.solvers.dp import DP
from mdpexplore.policies.summary_policies.density_policy import DensityPolicy
from mdpexplore.mdpexplore import MdpExplore

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
action_space = np.array([[-0.5, -0.5],
                        [-0.5, 0.5],
                         [0.0, 0.0],
                         [0.5, -0.5],
                         [0.5, 0.5]])

real_mu = theta_star @ action_space.T
optimal_action = np.argmax(real_mu)

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
                mix_objectives=(mix_objective, 0.5),
                init_ucb = 3
            )

    initial_ucbs = design.ucbs.copy()

    feedback = BanditFeedback(env, design, estimator, theta_star = theta_star, sigma = sigma, markovian = markovian)

    state_action_trajectory = [[[0, 1], [1, 2], [2, 3], [3, 4], [4, 3]]]

    episode = 0

    for h in range(5):
        state, action = state_action_trajectory[episode][h]
        feedback.step_single(state, action)

        assert len(feedback.state_trajectory) == h + 1, "state trajectory length is wrong"
        assert len(feedback.action_trajectory) == h + 1, "action trajectory length is wrong"
    
    if markovian:
        assert np.all(initial_ucbs == feedback.objective.ucbs), "ucbs should not change until the end of the episode in markovian case"
    else:
        assert np.all(initial_ucbs != feedback.objective.ucbs), "ucbs should change at each step in non-markovian case"
    
    feedback.step_episode()

    if markovian:
        assert np.all(initial_ucbs != feedback.objective.ucbs), "ucbs should change at the end of the episode in markovian case"
    
    assert len(feedback.state_trajectory) == 0, "state trajectory should be empty after episode"
    assert len(feedback.action_trajectory) == 0, "action trajectory should be empty after episode"

    assert torch.all(torch.isclose(feedback.estimator.x, torch.tensor(np.array([action_space[act_idx] for act_idx in [1, 2, 3, 4, 3]])))), "data stored in estimator is wrong"
    assert np.all(feedback.objective.ucbs >= feedback.objective.lcbs), "ucbs should be greater than lcbs"