from abc import ABC, abstractmethod
import numpy as np
from mdpexplore.env.discrete_env import DiscreteEnv
from mdpexplore.policies.policy_base import Policy
from mdpexplore.policies.non_stationary_policy import NonStationaryPolicy
from mdpexplore.policies.stationary_policy import StationaryPolicy
from mdpexplore.solvers.solver_base import DiscreteSolver
from mdpexplore.functionals.reward_functional import RewardFunctional
from mdpexplore.solvers.dp import DP
from mdpexplore.policies.policy_generator import PolicyGenerator

class ConvexSolverBase(ABC):
    def __init__(self, env : DiscreteEnv, objective : RewardFunctional, verbosity : int = 0, accuracy : float = None, initial_policy : bool = False, solver : DiscreteSolver = DP) -> None:
        self.env = env
        self.objective = objective
        self.verbosity = verbosity
        self.accuracy = accuracy
        self.initial_policy = initial_policy
        self.solver = solver
        
        if self.initial_policy:
            self.initial_policy_generator = PolicyGenerator(self.env)
            self.policies = [self.initial_policy_generator.uniform_policy()]
            self.weights = [1.0]
        else:
            self.policies = []
            self.weights = []
        
        self.densities = []
    
    @abstractmethod
    def optimize(self, emissions, visitations, episodes) -> None:
        ...
    
    def _density_oracle_single(self, policy: Policy) -> np.ndarray:
        """Computes state distribution induced by the given policy over a specified horizon

        Args:
            policy (Policy): inducing policy

        Returns:
            np.ndarray: S x A (stationary) or H x S x A (non-stationary) array with density for each state
        """

        if type(policy) is StationaryPolicy:

            v0 = np.zeros((self.env.states_num, self.env.actions_num))
            # initialize with the initial state and corresponding actions
            for act in self.env.available_actions(self.env.init_state):
                v0[self.env.init_state, act] = policy.p[self.env.init_state, act]

            v = np.array(v0)
            temp = np.array(v0).sum(axis = -1)

            p_pi = (self.env.get_transition_matrix() *
                    np.expand_dims(policy.p, axis=2)).sum(axis=1)
            assert (np.allclose(p_pi.sum(axis=1), 1, rtol=1e-05, atol=1e-05))

            for _ in range(self.env.max_episode_length):
                temp = p_pi.T @ temp
                v += np.expand_dims(temp, axis=-1) * policy.p
            
            v = v / v.sum()

        elif type(policy) is NonStationaryPolicy:

            v0 = np.zeros((self.env.max_episode_length, self.env.states_num, self.env.actions_num))
            # initialize with the initial state and corresponding actions
            for act in self.env.available_actions(self.env.init_state):
                v0[0, self.env.init_state, act] = policy.ps[0, self.env.init_state, act]

            v = np.array(v0)
            # get marginal state distribution for initial temp
            temp = np.array(v0)[0].sum(axis = -1)
            
            for i in range(self.env.max_episode_length - 1):
                p_pi = (self.env.get_transition_matrix() *
                        np.expand_dims(policy.ps[i], axis=2)).sum(axis=1)
                # assert (np.allclose(p_pi.sum(axis=1), 1, rtol=1e-05, atol=1e-05))
                temp = p_pi.T @ temp
                v[i + 1] += np.expand_dims(temp, axis=-1) * policy.ps[i + 1]

        return v

    def _density_oracle(self, actions: bool = True) -> np.ndarray:
        """Computes the combined state (or state-action) distribution induced by the saved policies

        Args:
            actions (bool, optional): if True, computes state-action distribution instead of state distribution. Defaults to False.

        Returns:
            np.ndarray: 1-D array with density for each state

        Raises:
            TypeError: if the saved policies are non-stationary
        """

        # if the first policy is non-stationary, we define a total density with time index
        if self.solver is DP:
            total_density = np.zeros((self.env.max_episode_length, self.env.states_num))
        else:
            total_density = np.zeros(self.env.states_num)
        
        # add actions dimension 
        total_density = np.repeat(np.expand_dims(total_density, axis = -1), axis = -1, repeats = self.env.actions_num)

        for i, policy in enumerate(self.policies):
            if i >= len(self.densities):
                d = self._density_oracle_single(policy)
                self.densities.append(d)
            total_density += self.weights[i] * self.densities[i]
        return total_density
    
    def reset(self):

        if self.initial_policy:
            self.policies = [self.initial_policy_generator.uniform_policy()]
            self.weights = [1.0]
        else:
            self.policies = []
            self.weights = []
        
        self.densities = []