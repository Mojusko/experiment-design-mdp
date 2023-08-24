import numpy as np
from mdpexplore.env.env_dummy_testing import DummyTestEnv
from mdpexplore.functionals.doe_adaptive_functionals import AdaptiveDesignD
from mdpexplore.densities.density_estimators import TabularDensity
from mdpexplore.policies.base_policies.stationary_policy import StationaryPolicy
from mdpexplore.policies.base_policies.non_stationary_policy import NonStationaryPolicy

import pytest

env = DummyTestEnv()
design = AdaptiveDesignD(env)
density_estimator = TabularDensity(env, design)

@pytest.mark.parametrize("policy_type", ['StationaryPolicy', 'NonStationaryPolicy'])
def test_density_oracle_single(policy_type : str):
    # test that the density oracle returns the correct density
    p = np.array([[0., 1., 0., 0., 0.],
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
        correct_density = np.array([[0., 1., 0., 0., 0.],
                                    [0., 0., 1., 0., 0.],
                                    [0., 0., 0., 1., 0.],
                                    [0., 0., 0., 0., 1.],
                                    [0., 0., 0., 0., 7.]])
        correct_density = correct_density / 11
        assert np.all(density == correct_density), 'density oracle returns incorrect density'
    
    elif policy_type == 'NonStationaryPolicy':
        ps = np.expand_dims(p, axis=0).repeat(10, axis=0)
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
        correct_density = np.array([[[0., 1., 0., 0., 0.],
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

        assert np.all(density == correct_density), 'density oracle returns incorrect density'