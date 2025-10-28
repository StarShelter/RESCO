import os
import numpy as np

from resco_benchmark.config.config import config as cfg
from resco_benchmark.agents.agent import Agent
from numpy.linalg import norm
import tensorflow as tf
from resco_benchmark.agents.policy.ma2c import MA2CAgent



class FMA2C(Agent):
    def __init__(self, obs_act):
        super().__init__()

        tf.reset_default_graph()
        cfg_proto = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=cfg_proto)

        self.supervisors = cfg["supervisors"]  # reverse of management
        self.management_neighbors = cfg["management_neighbors"]
        management = cfg["management"]

        self.state = None
        self.acts = None

        self.managers = dict()
        self.workers = dict()

        for manager in management:
            worker_ids = management[manager]
            mgr_act_size = cfg["management_acts"]
            mgr_fingerprint_size = (
                len(self.management_neighbors[manager]) * mgr_act_size
            )
            self.managers[manager] = MA2CAgent(
                obs_act[manager][0],
                mgr_act_size,
                mgr_fingerprint_size,
                0,
                manager + str(cfg.uuid),
                self.sess,
            )

            for worker_id in worker_ids:
                # Get fingerprint size
                downstream = cfg[worker_id]["downstream"]
                neighbors = [downstream[direction] for direction in downstream]
                fp_size = 0
                for neighbor in neighbors:
                    if (
                        neighbor is not None
                        and self.supervisors[neighbor]
                        == self.supervisors[worker_id]
                    ):
                        fp_size += obs_act[neighbor][1]  # neighbor's action size

                # Get waiting size
                lane_sets = cfg[worker_id]["lane_sets"]
                lanes = []
                for direction in lane_sets:
                    for lane in lane_sets[direction]:
                        if lane not in lanes:
                            lanes.append(lane)
                waits_len = len(lanes)

                management_size = len(self.management_neighbors[manager]) + 1

                observation_shape = (obs_act[worker_id][0][0] + management_size,)
                num_actions = obs_act[worker_id][1]
                self.workers[worker_id] = MA2CAgent(
                    observation_shape,
                    num_actions,
                    fp_size,
                    waits_len,
                    worker_id + str(cfg.uuid),
                    self.sess,
                )

        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess.run(tf.global_variables_initializer())
        self.prev_states = dict()
        self.prev_acts = dict()
        for manager in self.managers:
            self.prev_states[manager] = np.zeros(obs_act[manager][0])
            self.prev_acts[manager] = np.zeros(obs_act[manager][1])
        for worker in self.workers:
            self.prev_states[worker] = np.zeros(obs_act[worker][0])
            self.prev_acts[worker] = np.zeros(obs_act[worker][1])

    def fingerprints(self, observation):
        agent_fingerprint = {}
        for agent_id in observation.keys():
            if agent_id in self.managers:
                fingerprints = []
                for neighbor in self.management_neighbors[agent_id]:
                    neighbor_fp = self.managers[neighbor].fingerprint
                    fingerprints.append(neighbor_fp)
                if len(fingerprints) > 0:
                    fp = np.concatenate(fingerprints)
                else:
                    fp = np.asarray([])
                agent_fingerprint[agent_id] = fp
            else:
                downstream = cfg[agent_id]["downstream"]
                neighbors = [downstream[direction] for direction in downstream]
                fingerprints = []
                for neighbor in neighbors:
                    if (
                        neighbor is not None
                        and self.supervisors[neighbor] == self.supervisors[agent_id]
                    ):
                        neighbor_fp = self.workers[neighbor].fingerprint
                        fingerprints.append(neighbor_fp)
                if len(fingerprints) > 0:
                    fp = np.concatenate(fingerprints)
                else:
                    fp = np.asarray([])
                agent_fingerprint[agent_id] = fp
        return agent_fingerprint

    def act(self, observation):
        acts = dict()
        full_state = dict()  # Includes fingerprints, but not manager acts
        fingerprints = self.fingerprints(observation)
        # First get management's acts, they're part of the state for workers
        for agent_id in self.managers:
            env_obs = observation[agent_id]
            neighbor_fingerprints = fingerprints[agent_id]
            combine = np.concatenate([env_obs, neighbor_fingerprints])
            acts[agent_id] = self.managers[agent_id].act(combine)
            self.prev_states[agent_id] = env_obs
            self.prev_acts[agent_id] = acts[agent_id]

        for agent_id in self.workers:
            env_obs = observation[agent_id]
            self.prev_states[agent_id] = env_obs
            neighbor_fingerprints = fingerprints[agent_id]

            combine = np.concatenate([env_obs, neighbor_fingerprints])
            full_state[agent_id] = combine

            # Get management goals
            managing_agent = self.supervisors[agent_id]
            managing_agents_acts = [acts[managing_agent]]
            for mgr_neighbor in self.management_neighbors[managing_agent]:
                managing_agents_acts.append(acts[mgr_neighbor])
            managing_agents_acts = np.asarray(managing_agents_acts)
            combine = np.concatenate([managing_agents_acts, combine])

            acts[agent_id] = self.workers[agent_id].act(combine)
        self.state = full_state
        self.acts = acts
        return acts

    def observe(self, observation, reward, done, info):
        fingerprints = self.fingerprints(observation)

        for agent_id in observation.keys():
            env_obs = observation[agent_id]
            neighbor_fingerprints = fingerprints[agent_id]
            combine = np.concatenate([env_obs, neighbor_fingerprints])

            if agent_id in self.managers:
                self.managers[agent_id].observe(
                    combine, reward[agent_id], done, info
                )
            else:
                managing_agent = self.supervisors[agent_id]
                managing_agents_acts = [self.acts[managing_agent]]
                for mgr_neighbor in self.management_neighbors[managing_agent]:
                    managing_agents_acts.append(self.acts[mgr_neighbor])
                managing_agents_acts = np.asarray(managing_agents_acts)
                combine = np.concatenate([managing_agents_acts, combine])

                diff = env_obs - self.prev_states[agent_id]
                cosine = np.dot(diff.T, self.prev_acts[managing_agent]) / (
                    norm(diff) * norm(self.prev_acts[managing_agent]) + 1e-8
                )
                self.workers[agent_id].observe(
                    combine, np.sum(cosine + reward[agent_id], -1), done, info
                )

    def save(self):
        self.saver.save(self.sess, os.path.join(cfg.run_path, cfg.uuid) + ".tf")

    def load(self):
        raise NotImplementedError()
