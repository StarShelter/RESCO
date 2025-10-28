import logging
import os
from typing import Iterable, Callable, Tuple
import numpy as np

from resco_benchmark.config.config import config as cfg
from resco_benchmark.agents.agent import IndependentAgent, Agent

import gymnasium
import tensorflow as tf
from keras import Input, Model
from keras.layers import Dense, Activation, Concatenate
from keras.layers import LayerNormalization

from COOM.env.base import BaseEnv
from CL.utils.running import create_one_hot_vec
from CL.rl.sac import SAC
from CL.replay.buffers import BufferType
from CL.utils.logging import EpochLogger
import CL.rl.models as models

logger = logging.getLogger(__name__)
from CL.methods.agem import AGEM_SAC
from CL.methods.clonex import ClonExSAC
from CL.methods.ewc import EWC_SAC
from CL.methods.l2 import L2_SAC
from CL.methods.mas import MAS_SAC
from CL.methods.owl import OWL_SAC
from CL.methods.packnet import PackNet_SAC
import CL.methods.vcl as vcl


def mlp(
    state_shape: Tuple[int],
    num_tasks: int,
    hidden_sizes: Iterable[int],
    activation: Callable,
    use_layer_norm: bool = False,
    use_lstm: bool = False,
    hide_task_id: bool = False,
) -> Model:
    task_input = Input(shape=num_tasks, name="task_input", dtype=tf.float32)
    conv_in = Input(shape=state_shape, name="conv_head_in")

    model = conv_in if hide_task_id else Concatenate()([conv_in, task_input])
    model = Dense(cfg.number_of_units)(model)
    if use_layer_norm:
        model = LayerNormalization()(model)
        model = Activation(tf.nn.tanh)(model)
    else:
        model = Activation(activation)(model)
    for _ in range(cfg.number_of_layers - 1):
        model = Dense(cfg.number_of_units, activation=activation)(model)
    inputs = conv_in if hide_task_id else [conv_in, task_input]
    model = Model(inputs=inputs, outputs=model)
    return model


# Override COOM convolutional required impl
models.mlp = mlp


def variational_mlp(
    state_shape: Tuple[int],
    num_tasks: int,
    hidden_sizes: Tuple[int],
    activation: Callable,
    use_layer_norm: bool = False,
    use_lstm: bool = False,
    hide_task_id: bool = False,
) -> Model:
    task_input = Input(shape=num_tasks, name="task_input", dtype=tf.float32)
    conv_in = Input(shape=state_shape, name="conv_head_in")

    model = conv_in if hide_task_id else Concatenate()([conv_in, task_input])
    model = vcl.BayesianDense(model.shape[-1], hidden_sizes[0])(model)
    if use_layer_norm:
        model = LayerNormalization()(model)
        model = Activation(tf.nn.tanh)(model)
    else:
        model = Activation(activation)(model)
    for layer_idx in range(1, len(hidden_sizes)):
        prev_size, next_size = hidden_sizes[layer_idx - 1], hidden_sizes[layer_idx]
        model = vcl.BayesianDense(prev_size, next_size, activation=activation)(model)
    inputs = conv_in if hide_task_id else [conv_in, task_input]
    model = Model(inputs=inputs, outputs=model)
    return model


# Override COOM variational MLP for VCL
vcl.variational_mlp = variational_mlp


class DummyGame:
    def get_episode_timeout(self):
        return 3600 / cfg.step_length

    def close(self):
        pass


class DummyEnv(BaseEnv):
    def __init__(self, obs_space, act_space):
        super().__init__()
        self.obs_space = obs_space
        self.act_space = act_space
        self.cur_seq_idx = 0
        self.n_task = 24
        self.steps_per_env = 3600 / cfg.step_length
        self.game = DummyGame()
        self.envs = [DummyGame()] * self.n_task

    @property
    def name(self):
        return "ContinualLearningEnv"

    @property
    def task(self):
        return "DummyEnv"

    @property
    def task_id(self) -> int:
        return 0

    def get_active_env(self) -> BaseEnv:
        return DummyEnv(self.obs_space, self.act_space)

    @property
    def num_tasks(self) -> int:
        return self.n_task

    @property
    def observation_space(self) -> gymnasium.Space:
        return gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.obs_space, dtype=np.float32
        )

    @property
    def action_space(self) -> gymnasium.spaces.Discrete:
        return gymnasium.spaces.Discrete(self.act_space)


class ISAC(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        for agent_id in obs_act:
            obs_space = obs_act[agent_id][0]
            act_space = obs_act[agent_id][1]
            self.agents[agent_id] = COOMSACAgent(agent_id, obs_space, act_space)


class COOMSACAgent(Agent):
    def __init__(self, agent_id, obs_space, act_space):
        super().__init__()
        self.agent_id = agent_id
        logger = EpochLogger(cfg.run_path + "coom.log", dict(), 0)

        env = DummyEnv(obs_space, act_space)

        steps_per_hour = int(3600 / cfg.step_length)
        actor_cl = vcl.VclMlpActor if cfg.cl_method == "vcl" else models.MlpActor
        sac_args = dict(
            env=env,
            actor_cl=actor_cl,
            test_envs=list(),
            logger=logger,
            scenarios=list(),
            experiment_dir=cfg.run_path,
            log_every=int(1e8),
            save_freq_epochs=int(1e8),
            steps_per_env=env.steps_per_env,
            replay_size=cfg.replay_size,
            start_steps=0,
            update_after=steps_per_hour * 5,
            update_every=int(steps_per_hour / 2),
            policy_kwargs={"hidden_sizes": (cfg.number_of_units,)},
            lr=cfg.learning_rate,
            gamma=cfg.discount,
        )
        if cfg.cl_method is None:
            self.agent = SAC(**sac_args)
        elif cfg.cl_method == "agem":
            env.n_task *= 90
            self.agent = AGEM_SAC(
                **sac_args,
                episodic_mem_per_task=int(cfg.replay_size / 24),
                episodic_batch_size=128,
            )
        elif cfg.cl_method == "clonex":
            env.n_task *= 90
            self.agent = ClonExSAC(
                **sac_args,
                episodic_mem_per_task=int(cfg.replay_size / 24),
                episodic_batch_size=128,
                cl_reg_coef=100.0,
                episodic_memory_from_buffer=True,
                exploration_kind="best_return",
            )
        elif cfg.cl_method == "ewc":
            self.agent = EWC_SAC(**sac_args, cl_reg_coef=250.0)
        elif cfg.cl_method == "l2":
            self.agent = L2_SAC(**sac_args, cl_reg_coef=10000.0)
        elif cfg.cl_method == "mas":
            self.agent = MAS_SAC(**sac_args, cl_reg_coef=10000.0)
        elif cfg.cl_method == "owl":
            self.agent = OWL_SAC(**sac_args, cl_reg_coef=0.0)
        elif cfg.cl_method == "packnet":
            self.agent = PackNet_SAC(**sac_args, retrain_steps=10000, clipnorm=2e-05)
        elif cfg.cl_method == "vcl":
            self.agent = vcl.VCL_SAC(**sac_args, cl_reg_coef=1.0, first_task_kl=False)

        self.current_task_idx = -1
        self.current_task_timestep = 0
        self.episode_len = 0
        self.global_timestep = 0
        self.learn_on_batch = self.agent.get_learn_on_batch(self.current_task_idx)
        self.one_hot_vec = create_one_hot_vec(env.num_tasks, env.task_id)
        self.num_actions = act_space
        self.action_counts = {i: 0 for i in range(self.num_actions)}
        self.exploration_head_one_hot = None

        self.last_obs = None
        self.last_act = None

    def act(self, observation):
        obs_tensor = tf.convert_to_tensor(observation)
        if (
            self.current_task_timestep > self.agent.start_steps
            or (self.agent.agent_policy_exploration and self.current_task_idx > 0)
            or self.agent.model_path
        ):
            action = self.agent.get_action(
                obs_tensor,
                tf.convert_to_tensor(self.one_hot_vec, dtype=tf.dtypes.float32),
            )
        else:
            # Exploration
            if self.agent.exploration_helper is not None:
                # Use strategy provided by exploration helper.
                if self.exploration_head_one_hot is None:
                    self.exploration_head_one_hot = (
                        self.agent.exploration_helper.get_exploration_head_one_hot()
                    )
                task_id_tensor = tf.convert_to_tensor(
                    self.exploration_head_one_hot, dtype=tf.dtypes.float32
                )

                if self.agent.exploration_actor is not None:
                    action = self.agent.get_exploration_action(
                        obs_tensor, task_id_tensor
                    )
                else:
                    action = self.agent.get_action(obs_tensor, task_id_tensor)
            else:
                # Just pure random exploration.
                action = self.agent.env.action_space.sample()

        # Environment step
        action = action.numpy()[0] if isinstance(action, tf.Tensor) else action

        self.last_obs = observation
        self.last_act = action
        return action

    def observe(self, observation, reward, done, info):
        if (
            self.agent.exploration_helper is not None
            and self.exploration_head_one_hot is not None
        ):
            self.agent.exploration_helper.update_reward(reward)
        self.episode_len += 1
        self.global_timestep += 1

        # On task change
        if self.current_task_idx != getattr(self.agent.env, "cur_seq_idx", -1):
            self.current_task_timestep = 0
            self.current_task_idx = getattr(self.agent.env, "cur_seq_idx")
            self.agent._handle_task_change(self.current_task_idx)
            self.one_hot_vec = create_one_hot_vec(
                self.agent.env.num_tasks, self.agent.env.task_id
            )

        # Consider also whether episode was truncated
        done_to_store = (
            False if self.episode_len == self.agent.max_episode_len else done
        )

        # Store experience to replay buffer
        self.agent.replay_buffer.store(
            self.last_obs,
            self.last_act,
            reward,
            observation,
            done_to_store,
            self.one_hot_vec,
        )

        # End of trajectory handling
        if done:
            self.episode_len = 0
            if self.global_timestep < self.agent.steps - 1:
                self.exploration_head_one_hot = None

            if info["environment"].cumulative_episode % 24 == 0:
                self.agent.env.cur_seq_idx = 0
            else:
                self.agent.env.cur_seq_idx += 1

        # Update handling
        if (
            self.current_task_timestep >= self.agent.update_after
            and self.current_task_timestep % self.agent.update_every == 0
        ):

            for j in range(self.agent.n_updates):

                batch = self.agent.replay_buffer.sample_batch(self.agent.batch_size)
                episodic_batch = self.agent.get_episodic_batch(self.current_task_idx)

                results = self.learn_on_batch(
                    tf.convert_to_tensor(self.current_task_idx),
                    batch,
                    episodic_batch,
                )

                # Update priority in the tree
                abs_errors = results["abs_error"].numpy()
                if (
                    self.agent.buffer_type == BufferType.PER
                    or self.agent.buffer_type == BufferType.PRIORITY
                ):
                    self.agent.replay_buffer.update_weights(
                        batch["idxs"].numpy(), abs_errors
                    )

        if self.current_task_timestep + 1 == self.agent.env.steps_per_env:
            self.agent.on_task_end(self.current_task_idx)

        self.current_task_timestep += 1

    def save(self):
        pass

    def load(self):
        pass
