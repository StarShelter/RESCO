import logging
from collections import defaultdict, deque

import numpy as np


from resco_benchmark.config.config import config as cfg
from resco_benchmark.agents.agent import IndependentAgent, Agent
from resco_benchmark.agents.action_value.pfrl_dqn import DQNAgent, build_q_network
from resco_benchmark.agents.static.maxpressure import MAXPRESSURE

imports = True
try:
    from scipy.stats import kstest
except ImportError:
    imports = False

logger = logging.getLogger(__name__)


class IMultiDQN(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        cfg.csv_metrics.append("expert_usage")
        cfg.csv_metrics.append("num_models")
        if "prob_test" in cfg:
            cfg.csv_metrics.append("context_hits")
            cfg.csv_metrics.append("context_misses")
        fallback = MAXPRESSURE(obs_act)
        self.global_buffer = list()
        self.global_active_model = None

        def fake_switch():
            return self.global_active_model

        for i, agent_id in enumerate(obs_act):
            obs_space = obs_act[agent_id][0]
            act_space = obs_act[agent_id][1]
            self.agents[agent_id] = DARTSAgent(agent_id, obs_space, act_space, fallback)
            if cfg.global_buffers:
                self.agents[agent_id].buffer = self.global_buffer
                if i != 0:
                    self.agents[agent_id].switch = fake_switch

    def act(self, observation):
        if cfg.global_buffers:
            if len(self.global_buffer) != 0:
                summed = 0
                for _ in self.agents:
                    summed += self.global_buffer.pop()
                self.global_buffer.append(summed)
            super_agent = self.agents[list(self.agents.keys())[0]]
            active = super_agent.switch()
            super_agent.active_model = active
            self.global_active_model = active
        return super().act(observation)


class DARTSAgent(Agent):
    def __init__(self, agent_id, obs_space, act_space, fallback):
        super().__init__()
        self.agent_id = agent_id
        self.obs_space = obs_space
        self.act_space = act_space
        self.fallback = fallback

        self.buffer = []  # Reset at init_size at most
        self.models = dict()
        self.model_usage = defaultdict(int)
        self.model_buffer = dict()

        self.init_size = int(cfg.init_time * 60 / cfg.step_length)
        self.sample_size = int(cfg.sample_time * 60 / cfg.step_length)

        self.total_step = 0
        self.last_obs = None
        self.last_act = 0
        self.environment = None
        self.active_model = None
        self.last_reward = 0
        self.context_miss = 0
        self.context_check = 0

        self.swoks = None
        if cfg.criteria == "swoks":
            from resco_benchmark.agents.policy.SWOKS.swoks_alg import swoks

            self.swoks = swoks()

        self.expert_usage = 0

    def act(self, observation):
        self.last_obs = observation
        cur = self.active_model
        self.active_model = self.switch()
        self.model_usage[self.active_model] += 1
        if self.active_model is None:
            if cfg.expert is None:
                if cur is None:
                    act = np.random.randint(0, self.act_space)
                else:
                    self.active_model = cur
                    act = self.models[self.active_model].act(observation)
            elif cfg.expert == "maxpressure":
                if self.environment is None:
                    act = np.random.randint(0, self.act_space)
                else:
                    from resco_benchmark.mdp_options.states import mplight

                    mp_state = mplight(self.environment.signals)
                    act = self.fallback.act(mp_state)[self.agent_id]
                    self.expert_usage = 1
            elif cfg.expert == "random":
                act = np.random.randint(0, self.act_space)
                self.expert_usage = 1
            else:
                raise NotImplementedError("Expert not implemented. Check spelling?")
        else:
            if self.active_model not in self.models:
                logger.info(f"Creating model {self.active_model}")
                self.models[self.active_model] = DQNAgent(
                    agent_id=self.agent_id + "_" + str(self.active_model),
                    act_space=self.act_space,
                    model=build_q_network(self.obs_space, self.act_space),
                )
            act = self.models[self.active_model].act(observation)

        self.last_act = act
        self.total_step += 1
        return act

    def switch(self):
        model_number = None

        if cfg.criteria == "phase":
            phase_period = cfg.phase_period * 3600 / cfg.step_length
            if self.total_step > phase_period:
                dqn_percentage = 1.0
            else:
                dqn_percentage = self.total_step / phase_period
            if np.random.rand() < dqn_percentage:
                model_number = 0
        elif cfg.criteria == "hourly":
            if self.environment is None:
                model_number = None
            else:
                model_number = self.environment.hour_of_day
                maxq = self.environment.signals[self.agent_id].observation.max_queue
                if maxq > cfg.fallback:
                    model_number = None
        elif cfg.criteria == "maxq":
            if self.environment is None:
                model_number = None
            else:
                model_number = self.environment.signals[
                    self.agent_id
                ].observation.max_queue
                if model_number > cfg.fallback:
                    model_number = None
        elif cfg.criteria == "oracle_vehicles":
            if self.environment is None:
                model_number = None
            else:
                for i in range(len(cfg.intervals) - 1):
                    if (
                        cfg.intervals[i]
                        < self.environment.episode_vehicles
                        <= cfg.intervals[i + 1]
                    ):
                        model_number = i + 1
                        break
        elif cfg.criteria in ["std", "kstest"]:
            model_number = self.stat_criteria()
        elif cfg.criteria == "swoks":
            self.swoks.step(self.last_reward, self.last_act, self.last_obs)
            if self.swoks.task_changing:
                self.swoks.task_changing = False
            if self.swoks.new_agent and self.swoks.adopt_masks:
                self.swoks.new_agent = False
            model_number = self.swoks.current_task

        return model_number

    def extend_buffer(self, model_number):
        if not cfg.fixed_buffer:
            self.model_buffer[model_number].extend(self.buffer)
        else:
            if len(self.model_buffer[model_number]) == 0:
                self.model_buffer[model_number].extend(self.buffer)

    def stat_criteria(self):
        if len(self.buffer) % self.sample_size == 0 and len(self.buffer) > 0:
            if cfg.criteria == "std":
                model_number = self.std_criteria()
            elif cfg.criteria == "kstest":
                model_number = self.ks_criteria()
            else:
                raise NotImplementedError("Criteria not implemented. Check spelling?")

            if model_number is not None:
                self.extend_buffer(model_number)
                self.buffer.clear()
            elif model_number is None and len(self.buffer) >= self.init_size:
                model_number = len(self.model_buffer)
                if model_number not in self.model_buffer:
                    self.model_buffer[model_number] = deque(
                        maxlen=cfg.model_buffer_size
                    )
                self.extend_buffer(model_number)
                self.buffer.clear()
        else:
            model_number = self.active_model
        return model_number

    def std_criteria(self):
        model_number, best_value = None, float("inf")
        np_buffer = np.asarray(self.buffer)
        buffer_mean = float(np.mean(np_buffer))
        if self.agent_id == "A3" and "prob_test" in cfg:
            logger.info(f"Buffer mean: {buffer_mean}")
        for key in self.model_buffer:
            np_model_buffer = np.asarray(self.model_buffer[key])
            dist = abs(buffer_mean - float(np.mean(np_model_buffer)))
            if self.agent_id == "A3" and "prob_test" in cfg:
                logger.info(
                    f"Model {key} mean: {np.mean(np_model_buffer)}, Dist: {dist}, Threshold: {cfg.deviations * float(np.std(np_model_buffer))}"
                )
            if (
                dist < cfg.deviations * float(np.std(np_model_buffer))
                and dist < best_value
            ):
                best_value = dist
                model_number = key
        if self.agent_id == "A3" and "prob_test" in cfg:
            context = self.environment.context
            if model_number == context:
                self.context_check += 1
            else:
                self.context_miss += 1
            logger.info(f"Best model number: {model_number} Context: {context}")
            logger.info(
                f"Context hits: {self.context_check}, Context misses: {self.context_miss}"
            )
        return model_number

    def ks_criteria(self):
        model_number, best_value = None, 0.0
        np_buffer = np.asarray(self.buffer)
        for key in self.model_buffer:
            stat, p_value = kstest(np_buffer, self.model_buffer[key])
            if p_value > cfg.p_value and p_value > best_value:
                best_value = p_value
                model_number = key
        return model_number

    def observe(self, observation, reward, done, info):
        self.environment = info["environment"]
        self.last_reward = reward
        last_metric = self.environment.metrics[-1]
        if "expert_usage" not in last_metric:
            last_metric["expert_usage"] = dict()
            last_metric["num_models"] = dict()
            last_metric["context_hits"] = dict()
            last_metric["context_misses"] = dict()
        last_metric["expert_usage"][self.agent_id] = self.expert_usage
        last_metric["num_models"][self.agent_id] = len(self.models)
        if "prob_test" in cfg and self.agent_id == "A3":
            context = self.environment.context
            if self.active_model == context:
                last_metric["context_hits"][self.agent_id] = 1
                last_metric["context_misses"][self.agent_id] = 0
            else:
                last_metric["context_hits"][self.agent_id] = 0
                last_metric["context_misses"][self.agent_id] = 1
        if done:
            logger.info(
                f"Episode finished. Agent {self.agent_id} Model usage: {self.model_usage}"
            )
            self.expert_usage = 0

        self.buffer.append(reward)

        if self.active_model is not None or cfg.criteria == "phase":
            if cfg.criteria == "phase":
                active_model = 0
            else:
                active_model = self.active_model
            # Overwrite pfrl last action buffer
            self.models[active_model].agent.batch_last_obs = [self.last_obs]
            self.models[active_model].agent.batch_last_action = [self.last_act]
            self.models[active_model].observe(observation, reward, done, info)
