import warnings

from doexpy.optimization.ucb import UCB

warnings.filterwarnings("ignore",".*GUI is implemented.*")

class Greedy(UCB):

	def utility(self,xtest):
		(ymean,ystd) = self.model.mean_std(xtest)
		conf = ymean
		return conf
