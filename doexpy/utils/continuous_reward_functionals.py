from doexpy.env.linear_system import ContinuousEnv
import autograd.numpy as np
import autograd.numpy.linalg as la
import torch

class ContinuousRewardFunctional():

    def __init__(self, env: ContinuousEnv):
        pass

    def eval(self, density, visitations, episodes):
        pass

    def eval_full(self, visitations, episodes):
        pass

    def get_type(self):
        return self.type

class ContinuousDesignBayesD(ContinuousRewardFunctional):

    def __init__(self, env: ContinuousEnv, lambd = 0.1):
        super().__init__(env)
        self.type = "adaptive"
        self.env = env
        self.dim = env.dim
        self.lambd = lambd * np.eye(self.dim)
        self.samples = 1000

    def calculate_prior(self, visitations):
        prior = np.zeros(shape=(self.env.dim, self.env.dim))
        for p, visit in visitations:
            prior = prior +  p * sum([self.env.emit(a).reshape(-1, 1) @ (self.env.emit(a).reshape(-1, 1).T) for a in visit])
        return prior

    def eval(self, density, visitations, episodes):
        prior = self.calculate_prior(visitations)
        alpha = len(visitations)/episodes
        covariance_integral = self.covariance_integral(density)
        val = la.slogdet(alpha * prior + (1-alpha) * covariance_integral +  self.lambd/episodes )[1]
        return val

    def eval_full(self, visitations, episodes):
        prior = self.calculate_prior(visitations)
        prior = prior/episodes
        val = la.slogdet(prior + self.lambd / episodes)[1]
        return val

    def gradient(self, density, visitations, episodes):
        (mu, Sigma) = density
        # gradient evaluated at density
        prior = self.calculate_prior(visitations)
        alpha = len(visitations)/episodes
        covariance_integral = Sigma + mu@mu.T
        gradient = lambda x: x.T@la.inv(alpha * covariance_integral + (1-alpha) * prior +  self.lambd/episodes)@x
        return gradient

    def gradient_at_density(self, density, visitations, episodes, new_density):
        prior = self.calculate_prior(visitations)
        alpha = len(visitations)/episodes
        Z = self.covariance_integral(density)
        S = self.covariance_integral(new_density)

        #print (density)
        #print (new_density)
        #print ('---------')
        V = alpha * Z + (1-alpha) * prior + self.lambd/episodes
        return np.trace(la.inv(V)@( S))

    def covariance_integral(self, density):
        mu, Sigma = density
        try:
            points = np.tile(mu.reshape(1,-1),(self.samples,1)) + np.random.randn(self.samples, self.env.state_dim) @ la.cholesky(Sigma)
        except:
            points = np.tile(mu.reshape(1, -1), (self.samples, 1))
        #X = self.env.emb.embed(torch.from_numpy(points)).numpy()
        X = self.env.emb.embed(points)
        return (X.T @ X) / self.samples

class ContinuousDesignD(ContinuousRewardFunctional):

    def __init__(self, env: ContinuousEnv):
        super().__init__(env)
        self.type = "static"

