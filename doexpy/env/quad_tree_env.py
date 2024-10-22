from typing import List
import torch
from doexpy.env.grid_world_base import DeterministicGridWorldBase
from doexpy.env.stochastic_grid_world_base import StochasticGridWorldBase
from doexpy.env.grid_worlds import ErgodicGridWorld
from stpy.borel_set import HierarchicalBorelSets
from typing import List, Union, Callable


class QuadTreeGrid(ErgodicGridWorld):
    def __init__(
            self,
            embed : Callable,
            borel_set : HierarchicalBorelSets,
            depth: int = 4,
            init_state: int = 0,
            max_episode_length: int = 50,
            discount_factor: float = 0.99,
            seed: int = None,
            constrained: bool = False,
            terminal_state: int = 1,

    ) -> None:
        self.actions = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, -1)}

        # Calculate width and height
        self.hierarchy = borel_set
        self.sensing_regions = self.hierarchy.get_sets_level(depth)
        self.embed = embed
        self.width = 2**(depth-1)
        self.height = 2**(depth-1)
        self.sets = {}

        super().__init__(
            init_state=init_state,
            width=self.width,
            height=self.height,
            max_episode_length=max_episode_length,
            discount_factor=discount_factor,
            seed=seed,
            constrained=constrained,
            terminal_state=terminal_state
        )
        if not constrained:
            self.terminal_state = None

    def get_dim(self):
        return self.dim

    def _generate_emissions(self):
        hw = 2./self.width
        hh = 2./self.height
        for i in range(self.states_num):
            x, y = self.convert_to_grid(i)
            coord_x = -1 + hw * (x+0.5)
            coord_y = -1 + hh * (y+0.5)
            x = torch.Tensor([coord_x,coord_y]).double().view(-1,2)
            for action in self.sensing_regions:
                if action.is_inside(x):
                    set = action
                    break
            self.emissions[i] = self.embed(set)
            self.sets[i] = set
        self.dim = self.emissions[0].shape[0]
        self.theta = self.rng.random(self.dim)