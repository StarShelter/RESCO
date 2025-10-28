import numpy as np
from resco_benchmark.agents.agent import Agent, IndependentAgent
from resco_benchmark.config.config import config as cfg


class MAXWAVE(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        for agent_id in obs_act:
            self.agents[agent_id] = WaveAgent()


class WaveAgent(Agent):
    def act(self, observation):
        all_press = []
        for pair in cfg["phase_pairs"]:
            left = cfg.directions[pair[0]]
            right = cfg.directions[pair[1]]
            all_press.append(observation[left] + observation[right])

        return np.argmax(all_press)

    def observe(self, observation, reward, done, info):
        pass
