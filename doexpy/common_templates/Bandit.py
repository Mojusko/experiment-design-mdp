import matplotlib.pyplot as plt
import torch
import numpy as np
from abc import ABC, abstractmethod
class Bandit(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def step(self,x):
        pass


    def plot_conf(self, xtest, d, empirical=True, F=None, limit=None):
        '''
        Confidence
        '''
        from scipy.interpolate import griddata

        if d == 1:

            lw = 3
            ms = 12

            plt.rcParams.update({'font.size': 22})
            (mu, std) = self.GP.mean_std(xtest)

            plt.clf()
            plt.xlim([-0.51, 0.51])

            if limit is None:
                # plt.ylim([-2.5, 3.5])
                pass
            else:
                plt.ylim(limit)

            plt.title("Bayesian Optimization Example")
            plt.plot(self.x.numpy(), self.y.numpy(), "o", ms=ms, marker="o", label="evaluations", color="#1F77B4")

            if F is None:
                plt.plot(xtest.numpy(), self.F(xtest).numpy(), 'r-', label="true\nfunction", lw=lw)
            else:
                plt.plot(xtest.numpy(), F(xtest).numpy(), 'r-', label="true\nfunction", lw=lw)

            plt.plot(xtest.numpy(), self.confidence(xtest).numpy(), 'g-', label="acquisition\nfunction", lw=lw)
            value, index = torch.max(self.confidence(xtest), dim=0)
            x = xtest[index]
            plt.plot([x.numpy().flatten(), x.numpy().flatten()],
                     [value.numpy().flatten(), self.F(xtest)[index].numpy().flatten()], '--', lw=lw)
            plt.plot(x.numpy(), value.numpy(), 'g+', ms=ms, marker="o")

            plt.fill_between(xtest.numpy().flat, (mu - 2 * std).numpy().flat, (mu + 2 * std).numpy().flat,
                             color="#1F77B4",
                             label="2x std. dev.", alpha=0.1)
            plt.plot(xtest.numpy(), mu.numpy(), '--', lw=lw, label="mean\nprediction", color="#1F77B4")

            fig = plt.gcf()
            fig.set_size_inches(10, 6)
            ax = plt.subplot(111)

            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([0.08, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

            # plt.legend(loc = "upper right")
            plt.draw()

        elif d == 2:

            self.fit_gp(self.x, self.y)

            (mu, s) = self.GP.mean_std(xtest)

            plt.clf()
            ax = plt.axes(projection='3d')
            xx = xtest[:, 0].numpy()
            yy = xtest[:, 1].numpy()
            z = self.F(xtest).numpy()
            grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
            grid_z = griddata((xx, yy), z[:, 0], (grid_x, grid_y), method='linear')
            grid_z_mu = griddata((xx, yy), mu[:, 0].numpy(), (grid_x, grid_y), method='linear')
            ax.plot_surface(grid_x, grid_y, grid_z, shade=True, alpha=0.1)

            ax.scatter(self.x[:, 0].numpy(), self.x[:, 1].numpy(), self.y.numpy(), c='r', s=100, marker="o",
                       depthshade=False)
            ax.plot_surface(grid_x, grid_y, grid_z_mu, shade=True, alpha=0.2, color='r')
            # sample a couple of points
            f = self.GP.sample(xtest, 1)
            grid_z_f = griddata((xx, yy), f[:, 0].numpy(), (grid_x, grid_y), method='linear')
            ax.plot_surface(grid_x, grid_y, grid_z_f, shade=True, alpha=0.15, color='g')
            plt.draw()
        else:
            "nothing"