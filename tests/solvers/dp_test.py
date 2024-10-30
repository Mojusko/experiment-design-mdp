import numpy as np
import torch
from doexpy.solvers.dp import DP
from doexpy.env.env_dummy_testing import DummyTestEnv

# initialize reward
reward = torch.zeros((10, 5, 5))
# reward for staying in the rightmost state
reward[:, 4, 4] = 1.0
# reward for transitioning to the right
for i in range(4):
    reward[:, i, i+1] = 0.1

env = DummyTestEnv()
solver = DP(env, reward)
policy = solver.solve()

def test_policy_is_valid():
    for h in range(10):
        for s in range(5):
            for a in range(5):
                if policy.ps[h, s, a] > 0:
                    assert env.is_valid_action(a, s), 'policy breaks environment constraints'
                    assert policy.ps[h, s, a] <= 1, 'policy suggests a probability greater than 1'
                else:
                    assert policy.ps[h, s, a] == 0, 'policy suggests a negative probability'

def test_policy_is_optimal():
    for i in range(10):
        assert np.all(np.isclose(policy.ps[i], torch.tensor([[0., 1., 0., 0., 0.],
                                                [0., 0., 1., 0., 0.],
                                                [0., 0., 0., 1., 0.],
                                                [0., 0., 0., 0., 1.],
                                                [0., 0., 0., 0., 1.]]))), 'policy is not optimal'
