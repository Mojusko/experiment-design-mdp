import numpy as np
import torch
from doexpy.env.env_dummy_testing import DummyTestEnv
from doexpy.functionals.doe_adaptive_functionals import AdaptiveDesignD
from doexpy.densities.density_estimators import TabularDensity
from doexpy.policies.base_policies.stationary_policy import StationaryPolicy
from doexpy.policies.base_policies.non_stationary_policy import NonStationaryPolicy

import pytest

env = DummyTestEnv()
design = AdaptiveDesignD(env)
density_estimator = TabularDensity(env, design)

@pytest.mark.parametrize("policy_type", ['StationaryPolicy', 'NonStationaryPolicy'])
def test_density_oracle_single(policy_type : str):
    # test that the density oracle returns the correct density
    p = torch.tensor([[0., 1., 0., 0., 0.],
                [0., 0., 1., 0., 0.],
                [0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 1.]])

    if policy_type == 'StationaryPolicy':
        # test for stationary policy
        policy = StationaryPolicy(env, p)
        density = density_estimator.density_oracle_single(policy)

        # make sure the state-action density is valid
        for s in range(env.states_num):
            for a in range(env.actions_num):
                if density[s, a] > 0:
                    assert env.is_valid_action(a, s), 'policy breaks environment constraints'
                    assert density[s, a] <= 1, 'policy suggests a probability greater than 1'
                else:
                    density[s, a] == 0, 'policy suggests a negative probability'
    
        # test that the density is correct
        correct_density = torch.tensor([[0., 1., 0., 0., 0.],
                                    [0., 0., 1., 0., 0.],
                                    [0., 0., 0., 1., 0.],
                                    [0., 0., 0., 0., 1.],
                                    [0., 0., 0., 0., 7.]])
        correct_density = correct_density / 11
        assert np.all(np.isclose(density, correct_density)), 'density oracle returns incorrect density'
    
    elif policy_type == 'NonStationaryPolicy':
        ps = p.reshape(1, 5, 5).repeat(10, 1, 1)
        # test for non-stationary policy
        policy = NonStationaryPolicy(env, ps)
        density = density_estimator.density_oracle_single(policy)

        # test that the density is valid
        for h in range(10):
            for s in range(env.states_num):
                for a in range(env.actions_num):
                    if density[h, s, a] > 0:
                        assert env.is_valid_action(a, s), 'policy breaks environment constraints'
                        assert density[h, s, a] <= 1, 'policy suggests a probability greater than 1'
                    else:
                        density[h, s, a] == 0, 'policy suggests a negative probability'
        
        # test that the density is correct
        correct_density = torch.tensor([[[0., 1., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.]],
                                    
                                    [[0., 0., 0., 0., 0.],
                                    [0., 0., 1., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.]],
                                    
                                    [[0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.]],
                                    
                                    [[0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 1.],
                                    [0., 0., 0., 0., 0.]],
                                    
                                    [[0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 1.]],
                                    
                                    [[0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 1.]],
                                    
                                    [[0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 1.]],
                                    
                                    [[0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 1.]],
                                    
                                    [[0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 1.]],
                                    
                                    [[0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 1.]]])

        assert np.all(np.isclose(density, correct_density)), 'density oracle returns incorrect density'