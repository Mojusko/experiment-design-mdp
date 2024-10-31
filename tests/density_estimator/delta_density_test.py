import numpy as np

from doexpy.densities.density_estimators import DeltaDensityEstimator
from doexpy.densities.continous_densities import SimpleDeltaDensity, NonStationaryDeltaDensity
from doexpy.env.env_dummy_testing import DummyTestEnvContinuous

from doexpy.policies.base_policies.stationary_policy import StationaryPolicyContinuous
from doexpy.policies.base_policies.non_stationary_policy import NonStationaryPolicyContinuous

from doexpy.functionals.doe_adaptive_functionals import AdaptiveDesignD

import torch
import torch.nn as nn

import pytest

env = DummyTestEnvContinuous(max_episode_length = 5, init_state = torch.tensor([-0.4]).reshape(1, 1))
design = None
density_estimator = DeltaDensityEstimator(env, design)

class DummyPolicy(nn.Module):
    def __init__(self, step = 0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step = step
    
    def forward(self, state):
        action = torch.tensor([self.step])
        return action

@pytest.mark.parametrize("policy_type", ['StationaryPolicy', 'NonStationaryPolicy'])
def test_density_oracle_single(policy_type):
    if policy_type == 'StationaryPolicy':
        # test for stationary policy
        policy = DummyPolicy(step = 0.1)
        policy = StationaryPolicyContinuous(env, policy)
        density = density_estimator.density_oracle_single(policy)

        assert np.isclose(density.delta_states, np.array([-0.4, -0.3, -0.2, -0.1, 0.0]).reshape(5, 1), atol = 1e-6).all(), 'density is not correct'
    
    elif policy_type == 'NonStationaryPolicy':
        policies = nn.ModuleList()
        for i in range(2):
            policy = DummyPolicy(step = 0.1)
            policies.append(policy)
        
        for i in range(2, 5):
            policy = DummyPolicy(step = 0.2)
            policies.append(policy)
        
        policy = NonStationaryPolicyContinuous(env, policies)
        density = density_estimator.density_oracle_single(policy)

        assert np.isclose(density.densities[0].delta_states, np.array([-0.4])).all(), 'density is not correct'
        assert np.isclose(density.densities[1].delta_states, np.array([-0.3])).all(), 'density is not correct'
        assert np.isclose(density.densities[2].delta_states, np.array([-0.2])).all(), 'density is not correct'
        assert np.isclose(density.densities[3].delta_states, np.array([0.0]), atol = 1e-6).all(), 'density is not correct'
        assert np.isclose(density.densities[4].delta_states, np.array([0.2])).all(), 'density is not correct'

        assert np.isclose(density.densities[0].delta_actions, np.array([0.1])).all(), 'density is not correct'
        assert np.isclose(density.densities[1].delta_actions, np.array([0.1])).all(), 'density is not correct'
        assert np.isclose(density.densities[2].delta_actions, np.array([0.2])).all(), 'density is not correct'
        assert np.isclose(density.densities[3].delta_actions, np.array([0.2])).all(), 'density is not correct'
        assert np.isclose(density.densities[4].delta_actions, np.array([0.2])).all(), 'density is not correct'

@pytest.mark.parametrize("policy_type", ['StationaryPolicy', 'NonStationaryPolicy'])
def test_density_oracle(policy_type):

    if policy_type == 'StationaryPolicy':

        policies_mix = []
        weights = [0.4, 0.3, 0.3]
        densities = []

        for step in [0.1, 0.15, 0.2]:
            policy = DummyPolicy(step = step)
            policy = StationaryPolicyContinuous(env, policy)
            policies_mix.append(policy)
        
        density = density_estimator.density_oracle(policies_mix, weights, densities, stationary = True)

        assert np.isclose(density.delta_states, np.array([-0.4, -0.3, -0.2, -0.1, 0.0,
                                                        -0.4, -0.25, -0.1, 0.05, 0.2,
                                                        -0.4, -0.2, 0.0, 0.2, 0.4]).reshape(15, 1), atol = 1e-6).all(), 'density is not correct'

        assert np.isclose(density.delta_actions, np.array([0.1, 0.1, 0.1, 0.1, 0.1,
                                                        0.15, 0.15, 0.15, 0.15, 0.15,
                                                        0.2, 0.2, 0.2, 0.2, 0.2,]).reshape(15, 1), atol = 1e-6).all(), 'density is not correct'

        assert np.isclose(density.weights, np.array([0.4, 0.4, 0.4, 0.4, 0.4,
                                                     0.3, 0.3, 0.3, 0.3, 0.3,
                                                     0.3, 0.3, 0.3, 0.3, 0.3]), atol = 1e-6).all(), 'weighting is not correct'

    if policy_type == 'NonStationaryPolicy':

        policies_mix = []
        weights = [0.4, 0.3, 0.15, 0.15]
        densities = []

        for step in [0.1, 0.15, 0.2, 0.3]:
            policies = nn.ModuleList()
            for i in range(5):
                policy = DummyPolicy(step = step)
                policies.append(policy)
            
            policy = NonStationaryPolicyContinuous(env, policies)
            policies_mix.append(policy)

        density = density_estimator.density_oracle(policies_mix, weights, densities, stationary = False)

        assert np.isclose(density.densities[0].delta_states, np.array([-0.4, -0.4, -0.4, -0.4]).reshape(-1, 1)).all(), 'density is not correct'
        assert np.isclose(density.densities[1].delta_states, np.array([-0.3, -0.25, -0.2, -0.2]).reshape(-1, 1)).all(), 'density is not correct'
        assert np.isclose(density.densities[2].delta_states, np.array([-0.2, -0.1, 0.0, 0.0]).reshape(-1, 1)).all(), 'density is not correct'
        assert np.isclose(density.densities[3].delta_states, np.array([-0.1, 0.05, 0.2, 0.2]).reshape(-1, 1)).all(), 'density is not correct'
        assert np.isclose(density.densities[4].delta_states, np.array([0.0, 0.2, 0.4, 0.4]).reshape(-1, 1), atol = 1e-6).all(), 'density is not correct'

        for h in range(5):
            assert np.isclose(density.densities[h].delta_actions, np.array([0.1, 0.15, 0.2, 0.2]).reshape(-1, 1)).all(), 'density is not correct'
            assert np.isclose(density.densities[h].weights, np.array([0.4, 0.3, 0.15, 0.15])).all(), 'weighting is not correct'