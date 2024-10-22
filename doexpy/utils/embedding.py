import autograd.numpy as np

from numpy.polynomial.hermite import hermgauss
import stpy.helpers.helper as helper

class Embedding():
    """
    Base class for Embeddings to approximate kernels with a higher dimensional linear product.
    """

    def __init__(self, gamma=0.1, nu=0.5, m=100, d=1, diameter=1.0, groups=None, kappa=1.0,
                 kernel="squared_exponential", cosine=False, approx="rff", **kwargs):
        """
        Called to calculate the embedding weights (either via sampling or deterministically)

        Args:
            gamma: (positional, 0.1) bandwidth of the squared exponential kernel
            nu: (positional, 0.5) the parameter of Matern family
            m: (positional, 1)
            d: dimension of the

        Returns:
            None
        """
        self.gamma = float(gamma)
        self.n = nu
        self.m = int(m)
        self.d = int(d)
        self.nu = nu
        self.kappa = kappa
        self.cosine = cosine
        self.diameter = diameter
        self.groups = groups
        self.kernel = kernel
        self.approx = approx
        self.gradient_avail = 0
        if self.m % 2 == 1:
            raise AssertionError("Number of random features has to be even.")

    def sample(self):
        """
        Called to calculate the embedding weights (either via sampling or deterministically)

        Args:
            None

        Returns:
            None
        """
        raise AttributeError("Only derived classes can call this method.")

    def embed(self, x):
        """
        Called to calculate the embedding weights (either via sampling or deterministically)

        Args:
            x: numpy array containing the points to be embedded in the format (n,d)

        Returns:
            y: numpy array containg the embedded points (n,m), where m is the embedding dimension
        """
        return x

    def get_m(self):
        """

        :return:

        """
        return self.m


class QuadratureEmbedding(Embedding):
    """
        General quadrature embedding
    """

    def __init__(self, scale=1.0, **kwargs):
        Embedding.__init__(self, **kwargs)
        self.scale = scale
        self.compute()

    def reorder_complexity(self, omegas, weights):
        abs_omegas = np.abs(omegas)
        order = np.argsort(abs_omegas)
        new_omegas = omegas[order]
        new_weights = weights[order]
        return new_omegas, new_weights

    def compute(self, complexity_reorder=True):
        """
            Computes the tensor grid for Fourier features
        :return:
        """

        if self.cosine == False:
            self.q = int(np.power(self.m // 2, 1. / self.d))
            self.m = self.q ** self.d
        else:
            self.q = int(np.power(self.m, 1. / self.d))
            self.m = self.q ** self.d

        (omegas, weights) = self.nodesAndWeights(self.q)

        if complexity_reorder == True:
            (omegas, weights) = self.reorder_complexity(omegas, weights)

        self.weights = helper.cartesian([weights for weight in range(self.d)])
        self.weights = np.prod(self.weights, axis=1)

        v = [omegas for omega in range(self.d)]
        self.W = helper.cartesian(v)

        if self.cosine == False:
            self.m = self.m * 2
        else:
            pass

        #self.W = torch.from_numpy(self.W)
        #self.weights = torch.from_numpy(self.weights)

    def transform(self):
        """

        :return: spectral density of a kernel
        """
        if self.kernel == "squared_exponential":
            p = lambda omega: np.exp(-np.sum(omega ** 2, axis=1).reshape(-1, 1) / 2 * (self.gamma ** 2)) * np.power(
                (self.gamma / np.sqrt(2 * np.pi)), 1.) * np.power(np.pi / 2, 1.)

        elif self.kernel == "laplace":
            p = lambda omega: np.prod(1. / ((self.gamma ** 2) * (omega ** 2) + 1.), axis=1).reshape(-1, 1) * np.power(
                self.gamma / 2., 1.)

        elif self.kernel == "modified_matern":
            if self.nu == 2:
                p = lambda omega: np.prod(1. / ((self.gamma ** 2) * (omega ** 2) + 1.) ** self.nu, axis=1).reshape(-1,
                                                                                                                   1) * np.power(
                    self.gamma * 1, 1.)
            elif self.nu == 3:
                p = lambda omega: np.prod(1. / ((self.gamma ** 2) * (omega ** 2) + 1.) ** self.nu, axis=1).reshape(-1,
                                                                                                                   1) * np.power(
                    self.gamma * 4 / 3, 1.)
            elif self.nu == 4:
                p = lambda omega: np.prod(1. / ((self.gamma ** 2) * (omega ** 2) + 1.) ** self.nu, axis=1).reshape(-1,
                                                                                                                   1) * np.power(
                    self.gamma * 8 / 5, 1.)

        return p

    def nodesAndWeights(self, q):
        """
        Compute nodes and weights of the quadrature scheme in 1D

        :param q: degree of quadrature
        :return: tuple of (nodes, weights)
        """

        # For osciallatory integrands even this has good properties.
        # weights = np.ones(self.q) * self.scale * np.pi / (self.q + 1)
        # omegas = (np.linspace(0, self.q - 1, self.q)) + 1
        # omegas = omegas * (np.pi / (self.q + 1))

        (omegas, weights) = np.polynomial.legendre.leggauss(2 * q)

        omegas = omegas[q:]
        weights = 2 * weights[q:]

        omegas = ((omegas + 1.) / 2.) * np.pi
        sine_scale = (1. / (np.sin(omegas) ** 2))
        omegas = self.scale / np.tan(omegas)
        prob = self.transform()
        weights = self.scale * sine_scale * weights * prob(omegas.reshape(-1, 1)).flatten()
        return (omegas, weights)

    def embed(self, x):
        """
        :param x: torch array
        :return: embeding of the x
        """
        (times, d) = x.shape
        q = self.W[:, 0:d] @ x.T
        if self.cosine == False:
            z = np.concatenate((np.sqrt(self.weights.reshape(-1, 1)) * np.cos(q), np.sqrt(self.weights.reshape(-1, 1)) * np.sin(q)), axis = 0)
        else:
            z = np.sqrt(self.weights.reshape(-1, 1)) * np.cos(q)

        return z.T * np.sqrt(self.kappa)


class HermiteEmbedding(QuadratureEmbedding):
    """
        Hermite Quadrature Fourier Features for squared exponential kernel
    """

    def __init__(self, ones=False, cosine=False, **kwargs):
        self.ones = ones
        self.cosine = cosine
        QuadratureEmbedding.__init__(self, **kwargs)
        if self.kernel != "squared_exponential":
            raise AssertionError("Hermite Embedding is allowed only with Squared Exponential Kernel")

    def nodesAndWeights(self, q):
        """
        Compute nodes and weights of the quadrature scheme in 1D

        :param q: degree of quadrature
        :return: tuple of (nodes, weights)
        """
        (nodes, weights) = hermgauss(2 * q)
        # print (nodes)
        nodes = nodes[q:]
        weights = 2 * weights[q:]

        if self.ones == True:
            weights = np.ones(q)

        nodes = np.sqrt(2) * nodes / self.gamma
        weights = weights / np.sqrt(np.pi)
        return (nodes, weights)

class EmptyEmbedding(Embedding):
    """
    Empty Embedding
    """

    def __init__(self, **kwargs):
        Embedding.__init__(self, **kwargs)

    def embed(self, x):
        """
        :param x: np array
        :return: embedding of the x
        """
        return x