from copy import deepcopy

import numpy as np


from resco_benchmark.config.config import config as cfg
from resco_benchmark.agents.agent import IndependentAgent, Agent

import torch
from torch import tensor, nn

from resco_benchmark.agents.policy.SWOKS.network_bodies import (
    DummyBody_CL,
    NatureConvBodySS,
)
from resco_benchmark.agents.policy.SWOKS.network_heads import (
    CategoricalActorCriticNet_CL_MultiHead,
)
from resco_benchmark.agents.policy.SWOKS.ssmask_utils import (
    get_mask,
    set_mask,
    set_model_task,
)
from resco_benchmark.agents.policy.SWOKS.swoks_alg import swoks


"""
Adapted from author's source https://github.com/JupiLogy/swoks/
"""


class SWOKS(IndependentAgent):

    def __init__(self, obs_act):
        super().__init__(obs_act)
        cl_num_tasks = int(cfg.num_tasks)
        network_fn = lambda state_dim, action_dim, label_dim: CategoricalActorCriticNet_CL_MultiHead(
            int(np.prod(state_dim)),
            action_dim,
            2 * cl_num_tasks,
            label_dim,
            phi_body=NatureConvBodySS(
                num_tasks=2 * cl_num_tasks, discrete=True, state_dim=state_dim
            ),
            actor_body=DummyBody_CL(16),
            critic_body=DummyBody_CL(16),
        )
        for agent_id in obs_act:
            obs_space = obs_act[agent_id][0]
            act_space = obs_act[agent_id][1]
            model = network_fn(obs_space, act_space, cl_num_tasks)
            self.agents[agent_id] = WOKSPPOAgent(
                agent_id, act_space, model, cl_num_tasks, swoks=swoks()
            )


class WOKSPPOAgent(Agent):
    def __init__(self, agent_id, act_space, model, cl_num_tasks, swoks):
        super().__init__()
        self.swoks = swoks
        self.network = model
        self.optimizer = torch.optim.RMSprop(
            self.network.parameters(), lr=cfg.learning_rate
        )
        self.total_steps = 0
        self.states = None
        self.episode_rewards = []
        self.last_episode_rewards = []
        self.iteration = 0
        self.labels_set = np.eye(cl_num_tasks)
        self.task_label = np.eye(len(self.labels_set))[swoks.current_task]
        self.backup = [None, None, 0]
        self.learn = True

        self.actions_list, self.states_info_list, self.reward_list = (
            [],
            [],
            [],
        )
        self.values_list = [[] for _ in range(len(self.labels_set))]
        self.rollout = []
        self.actions, self.log_probs, self.values, self.returns = [], [], [], []

    def act(self, observation, pair_to_act_map=None, reverse_valid=None):
        observation = torch.as_tensor(np.expand_dims(observation, 0)).float()
        self.states = observation
        for idx, each_label in enumerate(self.labels_set):
            with torch.no_grad():
                _, _, _, _, swoks_vals, _ = self.network.predict(
                    self.states, task_label=each_label.tolist()
                )
            self.values_list[idx].append(swoks_vals.detach().cpu().numpy())

        _, self.actions, self.log_probs, self.entropy, self.values, supp_info = (
            self.network.predict(self.states, task_label=self.task_label)
        )
        action = self.actions.detach().cpu().numpy()
        self.actions_list.append(action)
        self.states_info_list.append(supp_info.detach().cpu().numpy())
        return action[0]

    def observe(self, observation, reward, done, info):
        self.reward_list.append(np.expand_dims(np.array(reward), 0))

        self.rollout.append(
            [
                np.copy(self.states),
                self.actions.detach(),
                self.log_probs.detach(),
                self.values.detach(),
                self.actions,
                [reward],
                [1 - done],
                self.entropy,
            ]
        )
        observation = torch.as_tensor(np.expand_dims(observation, 0)).float()
        self.states = observation

        self.total_steps += 1
        if self.total_steps % cfg.rollout_length == 0:
            self.post_rollout()
            self.swoks_iteration()
            self.actions_list, self.states_info_list, self.reward_list = (
                [],
                [],
                [],
            )
            self.values_list = [[] for _ in range(len(self.labels_set))]
            self.rollout = []

    def post_rollout(self):
        pending_value = self.network.predict(self.states, task_label=self.task_label)[
            -2
        ]
        self.rollout.append([None, pending_value, None, None, None, None])

        processed_rollout = [None] * (len(self.rollout) - 1)
        returns = pending_value.detach()
        for i in reversed(range(len(self.rollout) - 1)):
            (
                state,
                actions,
                log_prob,
                value,
                actions,
                rewards,
                terminals,
                entropy,
            ) = self.rollout[i]
            state = tensor(state).to(cfg.device)
            actions = tensor(actions).to(cfg.device).unsqueeze(1)
            terminals = tensor(terminals).to(cfg.device).unsqueeze(1)
            rewards = tensor(rewards).to(cfg.device).unsqueeze(1)
            returns = rewards + cfg.discount * terminals * returns
            advantages = returns - value.squeeze(1).detach()
            processed_rollout[i] = [
                state,
                actions,
                log_prob,
                value,
                returns,
                advantages,
                entropy,
            ]

        state, actions, log_prob, value, returns, advantages, entropy = map(
            lambda x: torch.cat(x, dim=0), zip(*processed_rollout)
        )

        batcher = Batcher(
            state.size(0) // cfg.num_mini_batches, [np.arange(state.size(0))]
        )
        for _ in range(5):  # optimisation epochs
            batcher.shuffle()
            while not batcher.end():
                batch_indices = batcher.next_batch()[0]
                batch_indices = tensor(batch_indices).long()
                sampled_states = state[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_prob[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                _, _, logprob, entloss, vals, _ = self.network.predict(
                    sampled_states,
                    action=sampled_actions,
                    task_label=self.task_label,
                )
                ratio = (logprob - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = (
                    ratio.clamp(
                        1.0 - cfg.ppo_ratio_clip,
                        1.0 + cfg.ppo_ratio_clip,
                    )
                    * sampled_advantages
                )
                policy_loss = (
                    -torch.min(obj, obj_clipped).mean(0)
                    - cfg.entropy_weight * entloss.mean()
                )

                value_loss = 0.5 * (sampled_returns - vals).pow(2).mean()

                if self.learn:
                    self.optimizer.zero_grad()
                    (policy_loss + value_loss).sum().backward()
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), cfg.gradient_clip
                    )
                    self.optimizer.step()

        # with the block of code below, rows are workers, columns are actions/states_info for
        # each worker in rollout_length of the iteration.
        self.actions_list = np.array(self.actions_list).T
        self.states_info_list = np.swapaxes(np.array(self.states_info_list), 0, 1)
        self.reward_list = np.array(self.reward_list).T

    def swoks_iteration(self):
        # SWOKS code here
        for j in range(self.actions_list.shape[1]):  # for each timestep in rollout
            self.swoks.step(
                r=self.reward_list[0][j],
                a=self.actions_list[0][j],
                supp=self.states_info_list[0][j],
            )

        # Backing up
        if self.iteration % cfg.rollback == 0 and self.task_label is not None:
            mask = deepcopy(get_mask(self.network, self.swoks.current_task))
            if self.iteration % 100 == 0:
                self.backup = [mask, self.backup[1], 1]
                print("backup0")
            elif self.iteration % 100 == cfg.rollback:
                self.backup = [self.backup[0], mask, 0]
                print("backup1")

        # Task change stuff
        change_task = False
        if self.swoks.task_changing:
            change_task = True
            self.swoks.task_changing = False

        if change_task:
            print("changing task.")
            print(self.task_label)
            print("loading" + str(self.backup[2]))
            try:
                set_mask(
                    self.network,
                    deepcopy(self.backup[self.backup[2]]),
                    np.where(np.array(self.task_label) == 1)[0][0],
                )
            except:
                print("premature task switching. Loading first mask.")
                set_mask(
                    self.network,
                    deepcopy(self.backup[0]),
                    np.where(np.array(self.task_label) == 1)[0][0],
                )
        self.learn = True
        if self.swoks.tested_tasks != []:
            self.learn = False

        self.task_label = np.eye(len(self.labels_set))[self.swoks.current_task]
        set_model_task(self.network, self.swoks.current_task)

        if self.swoks.new_agent and self.swoks.adopt_masks:
            self.swoks.new_agent = False
            set_mask(
                self.network,
                deepcopy(get_mask(self.network, self.swoks.current_task - 1)),
                self.swoks.current_task,
            )
            # adopt the previous guy's mask!
            print(
                f"----\n ADOPTING MASK {self.swoks.current_task - 1} FOR AGENT {self.swoks.current_task}\n----"
            )

        self.iteration += 1

    def save(self):
        pass

    def load(self):
        pass

    def training(self):
        pass

    def testing(self):
        pass


class Batcher:
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        return self.batch_start >= self.num_entries

    def next_batch(self):
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start : self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
        return batch

    def shuffle(self):
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]
