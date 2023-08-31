from abc import ABC, abstractmethod
import numpy as np

class ContinuousDensity(ABC):
    def __init__(self) -> None:
        super().__init__()
    
class SimpleDeltaDensity(ContinuousDensity):
    def __init__(self, env, initial_states = None, initial_actions = None, weights = None) -> None:
        super().__init__()
        self.env = env

        if initial_states is None:
            assert initial_actions is None, "initial_actions must be None if initial_states is None"
            self.delta_states = np.zeros((0, env.states_dim))
        else:
            assert initial_actions is not None, "initial_actions must be provided if initial_states is provided"
            assert initial_states.shape[0] == initial_actions.shape[0], "initial_states and initial_actions must have the same number of rows"
            self.delta_states = initial_states
        
        if initial_actions is None:
            self.delta_actions = np.zeros((0, env.actions_dim))
        else:
            self.delta_actions = initial_actions
        
        if weights is None:
            self.weights = np.ones(self.delta_states.shape[0])
        else:
            assert weights.shape[0] == self.delta_states.shape[0], "weights must have the same number of rows as initial_states"
            self.weights = weights

    def __add__(self, other):
        return SimpleDeltaDensity(self.env, np.vstack((self.delta_states, other.delta_states)), np.vstack((self.delta_actions, other.delta_actions)), np.hstack((self.weights, other.weights)))

    def __mul__(self, other):
        return SimpleDeltaDensity(self.env, self.delta_states, self.delta_actions, self.weights * other)
    
    def __rmul__(self, other):
        return SimpleDeltaDensity(self.env, self.delta_states, self.delta_actions, self.weights * other)
    
    def normalization_constant(self):
        return self.weights.sum()

    def average_density(self):
        normalization_constant = self.normalization_constant()

        if normalization_constant == 0:
            return self
        else:
            return self * (1 / self.normalization_constant())
    
    def is_empty(self):
        if self.average_density().delta_states.shape[0] == 0:
            return True
        else:
            return False

class NonStationaryDeltaDensity(ContinuousDensity):
    def __init__(self, env, initial_densities = None, initial_states = None, initial_actions = None) -> None:
        super().__init__()
        self.env = env

        if initial_densities is None:
            self.densities = [SimpleDeltaDensity(self.env) for _ in range(env.max_episode_length - env.h)]
            if initial_states is not None:
                assert initial_actions is not None, "initial_actions must be provided if initial_states is provided"
                self.densities[0] = SimpleDeltaDensity(self.env, initial_states=initial_states, initial_actions=initial_actions) + self.densities[0]
        else:
            # assert len(initial_densities) == env.max_episode_length - env.h, "initial_densities must have length equal to env.max_episode_length - env.h"
            assert all([type(d) is SimpleDeltaDensity for d in initial_densities]), "initial_densities must be a list of SimpleDeltaDensity"
            self.densities = initial_densities
    
    def __add__(self, other):
        return NonStationaryDeltaDensity(self.env, [self.densities[i] + other.densities[i] for i in range(self.env.max_episode_length - self.env.h)])
    
    def __mul__(self, other):
        return NonStationaryDeltaDensity(self.env, [self.densities[i] * other for i in range(self.env.max_episode_length - self.env.h)])
    
    def __rmul__(self, other):
        return NonStationaryDeltaDensity(self.env, [self.densities[i] * other for i in range(self.env.max_episode_length - self.env.h)])
    
    def average_density(self):
        average_density = SimpleDeltaDensity(self.env)
        for d in self.densities:
            average_density += d
        
        normalization_constant = average_density.normalization_constant()

        if normalization_constant == 0:
            return average_density
        else:
            return average_density * (1 / normalization_constant)
    
    def is_empty(self):
        if self.average_density().delta_states.shape[0] == 0:
            return True
        else:
            return False
