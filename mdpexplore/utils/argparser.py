import argparse

from mdpexplore.functionals.reward_functional import *
from mdpexplore.policies.average_policy import AveragePolicy
from mdpexplore.policies.density_policy import DensityPolicy
from mdpexplore.policies.mixture_policy import MixturePolicy
from mdpexplore.solvers.dp import DP
from mdpexplore.solvers.lp import LP


class Parser():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description='A simple parser for the command line.')

        self.parser.add_argument('--max_episode_length', default=50, type=int, help='Maximum episode length')     
        self.parser.add_argument('--seed', default=None, type=int, help='Use this to set the seed for the random number generator')
        self.parser.add_argument('--name', default="name", type= str, help='name of the file')
        self.parser.add_argument('--cores', default=None, type=int, help='number of cores')

        self.parser.add_argument('--constrained', action='store_true', help='Use this to force the agent to get to the terminal state at the end of the episode')
        self.parser.add_argument('--terminal_state', default=None, type=int, help='Use this to set the terminal state')

        self.parser.add_argument('--width', default=7, type=int, help='Width of the Grid World')
        self.parser.add_argument('--height', default=6, type=int, help='Height of the Grid World')
        self.parser.add_argument('--max_sectors_num', default=5, type=int, help='Maximum number of sector types')
        
        self.parser.add_argument('--discount_factor', default=0.99, type=float, help='Discount factor')
        self.parser.add_argument('--teleport', default=None, type=int, help='Use this to add a teleport to initial state')
        self.parser.add_argument('--environment', type = str, default = 'dummy_grid', help='Environment type')


        self.parser.add_argument('--objective_fn', default='D', type=str, help='Reward functional (A/D/C/B)')
        self.parser.add_argument('--solver', default='LP', type=str, help='Solver (LP/DP)')

        self.parser.add_argument('--step', default='greedy', type=str, help='Step size in Frank-Wolfe (float/line-search/greedy)')
        self.parser.add_argument('--method', default='frank-wolfe', type=str, help='Solver (only frank-wolfe for now)')
        self.parser.add_argument('--verbosity', default=3, type=int, help='Use this to increase debug ouput')
        self.parser.add_argument('--accuracy', default=0.1, type=float, help='Termination criterion for optimality gap')

        self.parser.add_argument('--policy', default='density', type=str, help='Summarized policy type (mixed/average/density)')
        self.parser.add_argument('--num_components', default=1000, type=int, help='Number of MaxEnt components (basic policies)')
        self.parser.add_argument('--episodes', default=64, type=int, help='Number of evaluation policy unrolls')

        self.parser.add_argument('--repeats', default=20, type=int, help='Number of repeats')
        self.parser.add_argument('--noplot', action='store_true', help='Use this to disable plotting')

    def parse(self):
        p = self.parser.parse_args()

        if p.objective_fn == 'A':
            p.objective_fn = DesignA
        elif p.objective_fn == 'D':
            p.objective_fn = DesignD
        elif p.objective_fn == 'C':
            p.objective_fn = DesignC
        elif p.objective_fn == 'B':
            p.objective_fn = DesignBayesD
        else:
            raise ValueError('Invalid reward functional')

        if p.solver == 'LP':
            p.solver = LP
        elif p.solver == 'DP':
            p.solver = DP
        else:
            raise ValueError('Invalid solver')

        if p.step == 'greedy':
            p.step = None
        elif p.step == 'line-search':
            pass
        else:
            try:
                p.step = float(p.step)
            except ValueError:
                raise ValueError('Invalid step size')
        
        if p.policy == 'mixed':
            p.policy = MixturePolicy
        elif p.policy == 'average':
            p.policy = AveragePolicy
        elif p.policy == 'density':
            p.policy = DensityPolicy
        else:
            raise ValueError('Invalid policy type')
        
        if not p.noplot:
            p.plot = True
        else:
            p.plot = False
        
        if p.repeats < 1:
            raise ValueError('Invalid number of runs')
        
        if p.episodes < 1:
            raise ValueError('Invalid number of trajectories')

        if not p.constrained:
            p.constrained = False
        
        if p.method != 'frank-wolfe':
            raise ValueError("The requested optimizer is not implemented.")

        return p