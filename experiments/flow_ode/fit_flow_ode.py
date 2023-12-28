import numpy as np
from scipy.integrate import odeint
import torch
from stpy.kernels import KernelFunction
from stpy.continuous_processes.nystrom_fea import NystromFeatures

# define the ODE class
class ODE():
    def __init__(self) -> None:
        pass

    def f(self, t, y):
        raise "f not implemented"

    def solve(self, t_span, y0):
        raise "solve not implemented"

    def exact(self, t, y0):
        raise "exact not implemented"

# define the quadratic ODE which we will optimize
class SchreckerQuadraticODE(ODE):
    def __init__(self, k1 = 2.2, k2 = 50, k3 = 5) -> None:
        super().__init__()
        self.name = "Schrecker Quadratic ODE"
        self.latent_dim = 5
        # set the parameters
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
    
    def f(self, y, t):
        R1 = self.k1 * y[1] * y[2] - self.k2 * y[4] * y[6]
        R2 = self.k3 * y[4]

        dy1dt = R2
        dy2dt = -R1
        dy3dt = -R1
        dy5dt = R1 - R2
        dy7dt = R1 + R2

        return [dy1dt, dy2dt, dy3dt, 0, dy5dt, 0, dy7dt]
    
    def solve(self, t_span, y0):
        y = odeint(self.f, y0, t_span)

# define the full ODE which we will optimize
class SchreckerODE(ODE):
    def __init__(self, k1f = 2.23589, k1r = 0, k2f = 1753.989, k2r = 0.25614, k3f = 13760.73, k3r = 0, k4 = 0.63061, k5 = 0, k6 = 0.0399, k7 = 11.87965, k8 = 0, k9 = 3.69715) -> None:
        super().__init__()
        self.name = "Schrecker ODE"
        self.latent_dim = 7
        # set the parameters
        self.k1f = k1f
        self.k1r = k1r
        self.k2f = k2f
        self.k2r = k2r
        self.k3f = k3f
        self.k3r = k3r
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.k7 = k7
        self.k8 = k8
        self.k9 = k9
    
    def f(self, y, t):
        R1 = self.k1f * y[1] * y[2] - self.k1r * y[3] * y[6]
        R2 = self.k2f * y[3] - self.k2r * y[4]
        R3 = self.k3f * y[3] * y[1] - self.k3r * y[5] * y[6]
        R4 = self.k4 * y[4] * y[2]
        R5 = self.k5 * y[4]
        R6 = self.k6 * y[4] * y[0]
        R7 = self.k7 * y[5] * y[2]
        R8 = self.k8 * y[5]
        R9 = self.k9 * y[5] * y[0]

        dy1dt = R4 + R5 + R6 + R7 + R8 + R9
        dy2dt = -R1 - R3 + R8 + R8 + R9
        dy3dt = -R1
        dy4dt = R1 - R2 - R3
        dy5dt = R2 - R4 - R5 - R6
        dy6dt = R3 - R7 - R8 - R9
        dy7dt = R1 + R3 + R4 + R5 + R6

        return [dy1dt, dy2dt, dy3dt, dy4dt, dy5dt, dy6dt, dy7dt]
    
    def solve(self, t_span, y0):
        y = odeint(self.f, y0, t_span)
        return y

# define the kernel specified by the ODE
class ode_kernel():
    def __init__(self, k1 : float = 10, k2 : float = 874, k3 : float = 19200, alpha : float = 5, beta : float = 0.5, alpha_ode : float = 0.35, **kwargs):
        self.k1 = torch.tensor(k1).double()
        self.k2 = torch.tensor(k2).double()
        self.k3 = torch.tensor(k3).double()
        self.alpha = torch.tensor(alpha).double()
        self.beta = torch.tensor(beta).double()
        self.alpha_ode = torch.tensor(alpha_ode).double()

        # known parameters
        self.b = 0.1
        self.c = 0.0
    
    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-self.alpha * (x - self.beta)))

    def __call__(self, x1, x2, **kwargs):
        # define the different parts of the inputs
        _t1 = (x1[:, 0] + 0.5) * 40
        _t2 = (x2[:, 0] + 0.5) * 40
        _B1 = x1[:, 1] + 0.5
        _B2 = x2[:, 1] + 0.5

        # calculate the eigenvalues
        part1 = self.k1 * self.b + self.k2 * self.c + self.k3
        part2 = self.b**2 * self.k1**2 + self.c**2 * self.k2**2 + self.k3**2 + \
                2 * self.b * self.c * self.k1 * self.k2 - 2 * self.b * self.k1 * self.k3 + \
                2 * self.c * self.k2 * self.k3
        
        # define the eigenvalues
        lambda_1 = -0.5 * (part1 + torch.sqrt(part2))
        lambda_2 = -0.5 * (part1 - torch.sqrt(part2))
        
        # efficient implementation using broadcasting and outer products
        # calculate the first exponential
        y1 = lambda_2 / (lambda_1 - lambda_2) * torch.exp(lambda_1 * _t1.unsqueeze(1))
        y1 = y1 - lambda_1 / (lambda_1 - lambda_2) * torch.exp(lambda_2 * _t1.unsqueeze(1)) + 1
        # calculate the second exponential
        y2 = lambda_2 / (lambda_1 - lambda_2) * torch.exp(lambda_1 * _t1.unsqueeze(1))
        y2 = y2 - lambda_1 / (lambda_1 - lambda_2) * torch.exp(lambda_2 * _t1.unsqueeze(1)) + 1
        # calculate the first vector
        vec1 = (_B1 * (1 - self.sigmoid(_B1))).unsqueeze(1) * y1 + ((1 - _B1) * self.sigmoid(_B1)).unsqueeze(1) * y2

        # now repeat for the second vector
        y1 = lambda_2 / (lambda_1 - lambda_2) * torch.exp(lambda_1 * _t2.unsqueeze(0))
        y1 = y1 - lambda_1 / (lambda_1 - lambda_2) * torch.exp(lambda_2 * _t2.unsqueeze(0)) + 1
        # calculate the second exponential
        y2 = lambda_2 / (lambda_1 - lambda_2) * torch.exp(lambda_1 * _t2.unsqueeze(0))
        y2 = y2 - lambda_1 / (lambda_1 - lambda_2) * torch.exp(lambda_2 * _t2.unsqueeze(0)) + 1
        # calculate the second vector
        vec2 = (_B2 * (1 - self.sigmoid(_B2))).unsqueeze(0) * y1 + ((1 - _B2) * self.sigmoid(_B2)).unsqueeze(0) * y2

        # covariance matrix is the outer product of the two vectors
        covar_matrix = torch.matmul(vec1, vec2)

        return covar_matrix.T * self.alpha_ode

# define the custom embedding class
class ode_embedding():
    def __init__(self, num_features, action_space, k1 : float = 10, k2 : float = 874, k3 : float = 19200, alpha : float = 5, beta : float = 0.5, alpha_ode : float = 0.35, alpha_rbf : float = 0.01, ard = False) -> None:
        # define the parameters for the ode kernel
        self.k1 = torch.tensor(k1).double()
        self.k2 = torch.tensor(k2).double()
        self.k3 = torch.tensor(k3).double()
        self.alpha = torch.tensor(alpha).double()
        self.beta = torch.tensor(beta).double()
        self.alpha_ode = torch.tensor(alpha_ode).double()

        # define the rbf kernel
        self.alpha_rbf = torch.tensor(alpha_rbf).double()
        if ard:
            self.kernel_rbf = KernelFunction(kernel_name='ard', gamma = [0.15, 0.05], d = 2, kappa = self.alpha_rbf)
        else:
            self.kernel_rbf = KernelFunction(kernel_name='squared_exponential', gamma = 0.05, d = 2, kappa = self.alpha_rbf)
        # now define the nystrom embedding
        self.nystrom_embedding = NystromFeatures(m = torch.tensor(num_features - 1), kernel_object = self.kernel_rbf)
        self.nystrom_embedding.fit_gp(torch.tensor(action_space), None)

        # known parameters
        self.b = 0.1
        self.c = 0.0

    def embed(self, x):
        # input is a tensor of size (batch_size, 2)
        # output is a tensor of size (batch_size, num_features)

        # first calculate the nystrom rbf embedding, size (batch_size, num_features - 1)
        rbf_embedding = self.nystrom_embedding.embed(x)

        # now calculate the ode embedding, size (batch_size, 1)
        ode_embedding = self.ode_embed(x)

        # concatenate the two embeddings, size (batch_size, num_features) and return
        return torch.cat((ode_embedding, rbf_embedding), dim = 1)
    
    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-self.alpha * (x - self.beta)))

    def ode_embed(self, x):
        # return  the ode feature
        # define the different parts of the inputs
        _t1 = (x[:, 0] + 0.5) * 40
        _B1 = x[:, 1] + 0.5

        # calculate the eigenvalues
        part1 = self.k1 * self.b + self.k2 * self.c + self.k3
        part2 = self.b**2 * self.k1**2 + self.c**2 * self.k2**2 + self.k3**2 + \
                2 * self.b * self.c * self.k1 * self.k2 - 2 * self.b * self.k1 * self.k3 + \
                2 * self.c * self.k2 * self.k3
        
        # define the eigenvalues
        lambda_1 = -0.5 * (part1 + torch.sqrt(part2))
        lambda_2 = -0.5 * (part1 - torch.sqrt(part2))
        
        # efficient implementation using broadcasting and outer products
        # calculate the first exponential
        y1 = lambda_2 / (lambda_1 - lambda_2) * torch.exp(lambda_1 * _t1.unsqueeze(1))
        y1 = y1 - lambda_1 / (lambda_1 - lambda_2) * torch.exp(lambda_2 * _t1.unsqueeze(1)) + 1
        # calculate the second exponential
        y2 = lambda_2 / (lambda_1 - lambda_2) * torch.exp(lambda_1 * _t1.unsqueeze(1))
        y2 = y2 - lambda_1 / (lambda_1 - lambda_2) * torch.exp(lambda_2 * _t1.unsqueeze(1)) + 1
        # calculate the first vector
        vec1 = (_B1 * (1 - self.sigmoid(_B1))).unsqueeze(1) * y1 + ((1 - _B1) * self.sigmoid(_B1)).unsqueeze(1) * y2

        return vec1 * self.alpha_ode
