import numpy as np
from doexpy.solvers.ddpg import DDPG
from doexpy.env.env_dummy_testing import DummyTestEnvContinuous
import torch

# set default tensor type
torch.set_default_tensor_type(torch.DoubleTensor)

env = DummyTestEnvContinuous(max_episode_length = 3, min_action = -0.2, max_action = 0.2)
# initialize reward
reward = lambda h, s, a: np.abs(np.clip(s+a, -0.5, 0.5)) * -4 + 1

solver = DDPG(env, reward, verbosity = 0, gradient_steps = 128, buffer_size = 1024, update_iterations = 5, exploration_noise = 0.2)

def test_policy_is_optimal():
    policy = solver.solve()

    state0 = torch.tensor([-0.5], requires_grad = False).reshape(1, 1)
    state1 = torch.tensor([-0.3], requires_grad = False).reshape(1, 1)
    state2 = torch.tensor([-0.1], requires_grad = False).reshape(1, 1)

    action0 = torch.clip(policy[0](state0), -0.2, 0.2)
    action1 = torch.clip(policy[1](state1), -0.2, 0.2)
    action2 = torch.clip(policy[2](state2), -0.2, 0.2)

    assert .17 <= action0 <= .20, 'action far from optimal for state 0'
    assert .17 <= action1 <= .20, 'action far from optimal for state 1'
    assert .07 <= action2 <= .13, 'action far from optimal for state 2'
