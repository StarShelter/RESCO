from resco_benchmark.config.config import config as cfg
from resco_benchmark.agents.action_value.mplight import MPLight


class AdvancedMPLight(MPLight):
    def __init__(self, obs_act):
        super().__init__(obs_act)

        if cfg.state != "advanced_mplight":
            raise ValueError("AdvancedMPLight can only run with advanced_mplight state")

        if cfg.reward != "pressure":
            raise ValueError("AdvancedMPLight can only run with pressure reward")
