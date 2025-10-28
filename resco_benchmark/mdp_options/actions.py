import numpy as np

from resco_benchmark.config.config import config as cfg
from resco_benchmark.agents.static.fixed import FIXED, FixedAgent


# Config values of phase or null will use direct 'set next phase' actions


#   Replaces action space with simplified set of [keep, increase, decrease] following fixed timing config
class FixedCyclePlan:
    def __init__(self, obs_act, signals):
        self.signals = signals
        self.fixed_agent = FIXED(obs_act)
        self.num_acts = 3

    def act(self, acts):
        for signal in self.signals:
            agt_act = acts[signal]
            if agt_act == 0:
                pass
            elif agt_act == 1:
                self.fixed_agent.agents[signal].increase_current_phase_length()
            elif agt_act == 2:
                self.fixed_agent.agents[signal].decrease_current_phase_length()
            else:
                raise NotImplementedError()
        acts = self.fixed_agent.act(observation=self.signals)
        return acts


# Follow config fixed time defined cycle, choose to stay in current phase or go next
class FixedCycle:
    def __init__(self, obs_act, signals):
        self.signals = signals
        self.fixed_agent = FIXED(obs_act)
        self.num_acts = 2

        # Force all phases to 1 step length
        for signal in self.signals:
            for i in range(len(self.fixed_agent.agents[signal].plan)):
                self.fixed_agent.agents[signal].plan[i] = 1
            self.fixed_agent.agents[signal].active_phase = (
                len(self.fixed_agent.agents[signal].plan) - 2
            )

    def act(self, acts):
        if cfg.algorithm == "FIXED":
            return acts
        for signal in self.signals:
            agt_act = acts[signal]
            print(signal, agt_act)
            if agt_act == 0:  # Keep same phase
                self.fixed_agent.agents[signal].active_phase_len = 0
            elif agt_act == 1:  # Go next
                self.fixed_agent.agents[signal].active_phase_len = np.inf
            else:
                raise NotImplementedError()
        acts = self.fixed_agent.act(observation=self.signals)
        print("mask", acts)
        return acts


class PlanPick:
    def __init__(self, obs_act, signals):
        self.signals = signals
        # Only valid/tested for saltlake
        cfg["equal_plan"] = dict()
        cfg["vertical_plan"] = dict()
        cfg["horizontal_plan"] = dict()
        cfg["horizontal_plan"]["fixed_phase_order_idx"] = 0
        cfg["vertical_plan"]["fixed_phase_order_idx"] = 0
        cfg["equal_plan"]["fixed_phase_order_idx"] = 0
        cfg["equal_plan"]["fixed_timings"] = [4, 0, 4, 4, 4, 4, 4, 0]
        cfg["vertical_plan"]["fixed_timings"] = [
            4,
            0,
            4,
            4,
            4 + cfg.priority_offset,
            4 + cfg.priority_offset,
            4 + cfg.priority_offset,
            0,
        ]
        cfg["horizontal_plan"]["fixed_timings"] = [
            4 + cfg.priority_offset,
            0,
            4 + cfg.priority_offset,
            4 + cfg.priority_offset,
            4,
            4,
            4,
            0,
        ]

        self.equal_plans = dict()
        obs_act = obs_act.copy()
        obs_act[1] = 8
        for signal in self.signals:
            self.equal_plans[signal] = FixedAgent(obs_act, "equal_plan")
        self.vertical_plans = dict()
        for signal in self.signals:
            self.vertical_plans[signal] = FixedAgent(obs_act, "vertical_plan")
        self.horizontal_plans = dict()
        for signal in self.signals:
            self.horizontal_plans[signal] = FixedAgent(obs_act, "horizontal_plan")
        self.num_acts = 3

    def act(self, acts):
        new_acts = dict()
        for signal in self.signals:
            agt_act = acts[signal]
            if agt_act == 0:
                new_acts[signal] = self.equal_plans[signal].act(None)
            elif agt_act == 1:
                new_acts[signal] = self.vertical_plans[signal].act(None)
            elif agt_act == 2:
                new_acts[signal] = self.horizontal_plans[signal].act(None)
            else:
                print(signal, agt_act)
                raise NotImplementedError()
        return new_acts


# Continuous or discrete, choose current phase's length
class PhaseLength:
    pass  # TODO
