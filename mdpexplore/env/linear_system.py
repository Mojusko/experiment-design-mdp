from mdpexplore.env.discrete_env import Environment
from mdpexplore.policies.base_policies.linear_policy import LinearPolicy
import autograd.numpy as np
import autograd.numpy.linalg as la
import torch

from typing import Callable, Type, Union, Tuple, List
from mdpexplore.utils.embedding import Embedding, HermiteEmbedding
class ContinuousEnv(Environment):

    def __init__(self, init_state):
        super().__init__()


class LinearSystem(ContinuousEnv):

    def __init__(self, init_state=None, A=None, B=None, dim=3, sigma=0.01, emb: Embedding = None, max_episode_length = 10):
        super().__init__(init_state)

        # Work only with zero state so far
        assert (init_state is None)
        if init_state is None:
            self.init_state = np.zeros(dim)


        if A is None:
            self.A = np.eye(dim) * 0.99
        if B is None:
            self.B = np.eye(dim) * 1.0
        self.max_episode_length = max_episode_length

        self.emb = emb
        self.sigma = sigma
        self.state = self.init_state
        self.visitations = []

        self.state_dim = dim
        self.dim = self.emit(self.init_state).shape[0]

    def emit(self, state: np.array):
        return self.emb.embed(state.reshape(1,-1)).reshape(-1)

    def reset(self) -> None:
        self.state = self.init_state

    def convert(self, state: np.array):
        return state

    def next(self, state: np.array, action: np.array):
        new_state = self.A @ state + self.B @ action + np.random.randn(self.state_dim) * self.sigma
        return new_state

    def step(self, action: np.array ):
        self.state = self.next(self.state, action)
        self.visitations.append(self.state)
        return self.state

    def density_oracle(self, policies: List[LinearPolicy], weights : List[float]):
        mu = np.zeros(self.state_dim)
        Sigma = np.zeros(shape = (self.state_dim,self.state_dim))
        for weight,policy in zip(weights,policies):
            mu_single, Sigma_single = self.density_oracle_single(policy)
            mu += mu_single*weight
            Sigma += Sigma_single*weight
        return (mu, Sigma)

    def density_oracle_single(self, policy: LinearPolicy):
        mu = self.init_state*0
        K = policy.K
        Sigma = np.zeros(shape = (self.state_dim, self.state_dim))
        I = np.eye(self.state_dim)

        for i in range(self.max_episode_length):
            # this impelments an array
            temps = []
            temp = np.eye(self.state_dim)
            for k in range(i):
                temp = (self.A + self.B @ K) @ temp
                temps.append(temp)

            arr = [temp @ I @ temp.T for temp in temps] + [np.eye(self.state_dim)]

            Sigma_temp = self.sigma**2 * sum(arr)

            mu = mu +  temp@self.init_state
            Sigma = Sigma + Sigma_temp


        mu = mu / self.max_episode_length
        Sigma = Sigma / self.max_episode_length

        Sigma = Sigma - mu @ mu.T


        #print (mu, Sigma)
        return (mu,Sigma)

if __name__ == "__main__":

    emb = Embedding()
    env = LinearSystem(max_episode_length = 3, emb = emb)
    K = np.eye(env.state_dim)
    pi = LinearPolicy(env,K)

    d = env.density_oracle_single(pi)
    print (d)