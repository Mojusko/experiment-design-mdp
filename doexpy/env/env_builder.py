from doexpy.env.discrete_env import DiscreteEnv
from doexpy.env.grid_worlds import (
    ErgodicGridWorld,
    NonErgodicGridWorld,
    DummyGridWorld,
    StochasticDummyGridWorld,
    TunnelGridWorld,
)

class EnvBuilder:
    def __init__(self, config) -> None:
        self.config = config
    
    def build(self) -> DiscreteEnv:
        if self.config.environment == "dummy_grid":
            return DummyGridWorld(
                max_episode_length=self.config.max_episode_length,
                teleport=self.config.teleport,
                constrained=self.config.constrained,
                terminal_state=self.config.terminal_state
            )
        
        if self.config.environment == "tunnels_grid":
            return TunnelGridWorld(
                width=self.config.width,
                max_episode_length=self.config.max_episode_length
            )

        if self.config.environment == "ergodic_grid":
            return ErgodicGridWorld(
                width=self.config.width,
                height=self.config.height,
                max_episode_length=self.config.max_episode_length,
                discount_factor=self.config.discount_factor,
                max_sectors_num=self.config.max_sectors_num,
                seed=self.config.seed,
                teleport=self.config.teleport,
                constrained=self.config.constrained,
                terminal_state=self.config.terminal_state,
            )

        if self.config.environment == "stochastic_dummy_grid":
            return StochasticDummyGridWorld(
                max_episode_length=self.config.max_episode_length,
                teleport=self.config.teleport,
                width=self.config.width,
                height=self.config.height,
                constrained=self.config.constrained,
                terminal_state=self.config.terminal_state
            )
        if self.config.environment == "nonergodic_grid":
            return NonErgodicGridWorld(
                width=self.config.width,
                height=self.config.height,
                max_episode_length=self.config.max_episode_length,
                discount_factor=self.config.discount_factor,
                max_sectors_num=self.config.max_sectors_num,
                seed=self.config.seed,
                teleport=self.config.teleport,
                constrained=self.config.constrained,
                terminal_state=self.config.terminal_state,
            )