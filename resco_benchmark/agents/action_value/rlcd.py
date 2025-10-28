import logging
from collections import defaultdict
import heapq

import numpy as np

from resco_benchmark.config.config import config as cfg
from resco_benchmark.agents.agent import Agent, IndependentAgent

logger = logging.getLogger(__name__)


class RLCD(IndependentAgent):
    def __init__(self, obs_act):
        super().__init__(obs_act)
        for agent_id in obs_act:
            obs_space = obs_act[agent_id][0]
            act_space = obs_act[agent_id][1]
            self.agents[agent_id] = RLCDAgent(agent_id, obs_space, act_space)


class RLCDAgent(Agent):
    def __init__(self, agent_id, obs_space, act_space):
        super().__init__()
        self.agent_id = agent_id
        self.obs_space = obs_space
        self.act_space = act_space
        self.num_states = 12 * 3
        Rmax = 23**2 * self.num_states
        Rmin = 0
        self.Z_R = (Rmax - Rmin) ** -1
        self.T_init = 1 / self.num_states
        self.s = None
        self.a = None
        self.M = dict()
        self.E = defaultdict(float)
        self.N = defaultdict(dict)
        self.R = defaultdict(dict)
        self.T = defaultdict(dict)
        self.S = set()
        self.m_cur = 0
        self.plan_steps = cfg.plan_steps - 1
        self.model_usage = defaultdict(int)

    def act(self, observation):
        observation = tuple(observation)
        self.plan_steps += 1
        if self.plan_steps < cfg.plan_steps:
            return self.a
        elif self.plan_steps == cfg.plan_steps:
            self.plan_steps = 1
        self.S.add(observation)
        if self.m_cur not in self.M:
            self.M[self.m_cur] = self.newmodel()
        self.a = self.M[self.m_cur].get_action(observation)
        self.s = observation
        self.model_usage[self.m_cur] += 1
        return self.a

    def delta_T(self, m, k, s, a, s_p):
        return (1 if k == s_p else 0 - self.T[m][(s, a, s_p)]) / (self.N[m][(s, a)] + 1)

    def delta_R(self, m, r, s, a):
        return (r - self.R[m][(s, a)]) / (self.N[m][(s, a)] + 1)

    def observe(self, observation, reward, done, info):
        observation = tuple(observation)
        if done:
            logger.info(
                f"Episode finished. Agent {self.agent_id} Model usage: {self.model_usage}"
            )
        if self.plan_steps < cfg.plan_steps - 1:
            return
        self.M[self.m_cur].update(self.s, self.a, reward, observation)
        s_a = (self.s, self.a)
        s_a_s = (self.s, self.a, observation)
        for m in self.M:
            if s_a not in self.N[m]:
                self.N[m][s_a] = 0
                self.R[m][s_a] = 0.0
            if s_a_s not in self.R[m]:
                self.T[m][s_a_s] = self.T_init

            sum_delta_T_sq = 0
            for k in self.S:
                sum_delta_T_sq += (self.delta_T(m, k, self.s, self.a, observation)) ** 2
            unseen = self.num_states - len(self.S)
            sum_delta_T_sq += unseen * (-self.T_init) ** 2

            self.E[m] = self.E[m] + cfg.rho * (
                self.quality_e(
                    m,
                    self.s,
                    self.a,
                    self.delta_R(m, reward, self.s, self.a),
                    sum_delta_T_sq,
                )
                - self.E[m]
            )
        max_E = float("-inf")
        m_prev = self.m_cur
        for m in self.M:
            if self.E[m] > max_E:
                max_E = self.E[m]
                self.m_cur = m
        if self.E[self.m_cur] < cfg.min_E:
            logger.info(f"Creating new model for agent {self.agent_id} with E={max_E}")
            self.m_cur = len(self.M)

        self.T[m_prev][s_a_s] = self.T[m_prev][s_a_s] + self.delta_T(
            m_prev, observation, self.s, self.a, observation
        )
        self.R[m_prev][s_a] = self.R[m_prev][s_a] + self.delta_R(
            m_prev, reward, self.s, self.a
        )
        self.N[m_prev][s_a] = min(self.N[m_prev][s_a] + 1, cfg.M)

    def newmodel(self):
        model = DynaQAgent(1, mdp=TSCMDP(self.act_space))
        model.reset()
        return model

    def confidence_c(self, m, s, a):
        return self.N[m][(s, a)] / cfg.M

    def quality_e(self, m, s, a, delta_R, sum_delta_T_sq):
        R = 1 - 2 * (self.Z_R * delta_R**2)
        Z_T = (self.N[m][(s, a)] + 1) ** 2 / 2
        T = 1 - 2 * (Z_T * sum_delta_T_sq)
        return self.confidence_c(m, s, a) * (cfg.omega * R + (1 - cfg.omega) * T)


""" Prioritized Sweeping and Dyna-Q agents and control algorithms from https://github.com/kamenbliznashki/sutton_barto"""


class TSCMDP:
    def __init__(self, num_acts):
        self.num_acts = num_acts

    def get_possible_actions(self, state):
        return range(self.num_acts)

    def step(self):
        pass


class BaseAgent:
    """Base class for a RL agent.
    Different state-value / state-action value algorithms overwrite run_episode and update functions
    Approximation agents overwrite the q_value function representation from dictionary enumeration to feature vec approximation using
    get_q_value and reset
    """

    def __init__(
        self,
        mdp,
        run_episode_fn,
        discount=None,
        epsilon=None,
        alpha=None,
    ):
        """
        Args
            mdp             -- class with markov decision process providing the following function calls:
                                - get_possible_actions
                                - get_state_reward_transition
            run_episode_fn  -- function specifying the sequence of agent-environment interactions and updates
                                for the specific algorithm (e.g. Sarsa, Q-learning). This will be run during training by
                                calling agent.run_episode()
            discount        -- float in [0, 1]; discount for state / state-action value calculation (gamma in Sutton&Barto)
            epsilon         -- float in [0, 1]; spec for epsilon-greedy algorithms % exploration
            alpha           -- float in [0, 1]; learning step size parameter
        """
        self.mdp = mdp
        self.run_episode = lambda: run_episode_fn(mdp, self)
        self.discount = cfg.discount
        self.epsilon = cfg.epsilon
        self.alpha = cfg.alpha

        # initialize q_values
        self.reset()

    def get_action(self, state):
        """e-greedy policy"""
        rand = np.random.rand()
        actions = self.mdp.get_possible_actions(state)
        if rand < self.epsilon:
            return actions[np.random.choice(len(actions))]
        else:
            return self.compute_best_action(state)

    def get_q_value(self, state, action):
        return self.q_values[(state, action)]

    def get_value(self, state):
        return self.compute_value(state)

    def compute_best_action(self, state):
        # several actions may have the 'best' q_value; choose among them randomly
        legal_actions = self.mdp.get_possible_actions(state)
        if legal_actions[0] is None:
            return None
        q_values = [self.get_q_value(state, a) for a in legal_actions]
        eligible_best_actions = [
            a
            for i, a in enumerate(legal_actions)
            if np.round(q_values[i], 8) == np.round(np.max(q_values), 8)
        ]
        best_action_idx = np.random.choice(len(eligible_best_actions))
        best_action = eligible_best_actions[best_action_idx]
        return best_action

    def compute_q_value(self, state, action):
        next_state, reward = self.mdp.get_state_reward_transition(state, action)
        return reward + self.discount * self.get_value(next_state)

    def compute_value(self, state):
        best_action = self.compute_best_action(state)
        if best_action is None:
            return 0
        else:
            return self.get_q_value(state, best_action)

    def update(self, state, action, reward, next_state, next_action):
        """Update to the q_values to be overwriten per the specific algorithm in sync with the run_episode function"""
        raise NotImplementedError

    def reset(self):
        self.q_values = defaultdict(float)
        self.num_updates = 0


def run_qlearning_episode(mdp, agent):
    """Execute the Q-learning off-policy algorithm per Section 6.5.
    This is paired to an agent for the agent.run_episode() call.
    """
    # record episode path and actions
    states_visited = []
    actions_performed = []
    episode_rewards = 0

    # initialize S
    state = mdp.reset_state()
    states_visited.append(state)

    # loop for each step
    while not mdp.is_goal(state):

        # choose A from S using policy derived from Q
        action = agent.get_action(state)

        # take action A, observe R, S'
        next_state, reward = mdp.get_state_reward_transition(state, action)

        # update agent
        agent.update(state, action, reward, next_state)
        # update state
        state = next_state

        # record path
        states_visited.append(state)
        actions_performed.append(action)
        episode_rewards += reward

    return states_visited, actions_performed, episode_rewards


class QLearningAgent(BaseAgent):
    def __init__(self, run_episode_fn=run_qlearning_episode, **kwargs):
        super().__init__(run_episode_fn=run_qlearning_episode, **kwargs)

    def update(self, state, action, reward, next_state):
        """Q learning update to the policy -- eq 6.8"""

        q_t0 = self.get_q_value(state, action)
        q_t1 = self.get_value(next_state)

        # q learning update per eq 6.8 -- greedy policy after the current step
        new_value = q_t0 + self.alpha * (reward + self.discount * q_t1 - q_t0)

        # perform update
        self.q_values[(state, action)] = new_value

        self.num_updates += 1

        return new_value


class DynaQAgent(QLearningAgent):
    """Tabular Dyna-Q algorithm per Section 8.2"""

    def __init__(self, n_planning_steps, **kwargs):
        super().__init__(**kwargs)
        self.n_planning_steps = n_planning_steps

    def reset(self):
        super().reset()
        self.model = {}

    def sample_model(self):
        # sample state
        past_states = [k[0] for k in self.model.keys()]
        sampled_state = past_states[np.random.choice(len(past_states))]
        # sample action, previously taken from the sampled state
        past_actions = [k[1] for k in self.model.keys() if k[0] == sampled_state]
        sampled_action = past_actions[np.random.choice(len(past_actions))]
        # model assumes deterministic environment so no need to sample from the (R,S') pair under model(S,A)
        reward, next_state = self.model[(sampled_state, sampled_action)][1]
        return sampled_state, sampled_action, reward, next_state

    def update(self, state, action, reward, next_state):
        """Execute the Q-learning off-policy algorithm in Section 6.5 with Dyna-Q model update/planning in Section 8.2"""

        # perform q-learning update (Section 8.2 - Tabular Dyna-Q algorithm line (d))
        super().update(
            state, action, reward, next_state
        )  # note this is stepping the num_updates counter

        # update model (Sec 8.2 - Dyna-Q line (e))
        # model assumes deterministic environment
        self.model[(state, action)] = self.num_updates, (reward, next_state)

        # perform planning (Sec 8.2 - Dyna-Q line(f))
        # Loop repeat n times for the n_planning_steps
        for i in range(self.n_planning_steps):
            # sample randomly previously observed state (S) and sample randomly action previously taken at S
            super().update(
                *self.sample_model()
            )  # update q_values with the planning sample

        self.mdp.step()  # keep track of mdp number of update steps to change the mdp dynamically per example 8.2 blocking maze


class PrioritizedSweepingAgent(QLearningAgent):
    """Prioritized sweeping algorithm per Section 8.4

    Proposed upates to q values are kept in a priority queue as a list with python's heapq module.
    Heapq as a min-heap returns the min entry; min comparison for a python tuple compares each element in turn to break ties.

    Here the entry in the PQ is (-abs(proposed_update), -np.sign(proposed_update), (state, action)):
        -- heap index -abs(value) returns the min of the negative absolute updates i.e. the maximum update in abolute value;
        -- heap index -np.sign(value) returns the inverted sign of the actual update; thus heapq breaks ties for the first index
            by returning the min of -sign
            i.e. the entry with with + sign since rewards in this experiment are +1 for goal and 0 otherwise -- that is we'd like
            to prioritize large positive updates to the q values

    E.g.
        proposed_update_1 = +1
        heappush((-abs(1), -sign(1), ...)) pushes entry = (-1, -1, ...)
        proposed_update_2 = -1
        heappush((-abs(-1), -sign(-1), ...)) pushes entry = (-1, 1, ...)
        min heap property returns (-1, -1, ...) first which corresponds to the update proposal +1

    """

    def __init__(self, n_planning_steps, theta, **kwargs):
        super().__init__(**kwargs)
        self.n_planning_steps = n_planning_steps
        self.theta = theta  # the minimum magnitude of q_value update to be performed

    def reset(self):
        super().reset()
        self.model = {}
        self.pq = PriorityQueue()
        self.predecessors = defaultdict(set)

    def _update_predecessors(self, state, action, next_state):
        # add predecessors as a set of (state, action) tuples
        self.predecessors[next_state].add((state, action))

    def update(self, state, action, reward, next_state):
        """Execute the Q-learning off-policy algorithm in Section 6.5 with
        Prioritized Sweeping model update/planning in Section 8.4"""

        # update model (Sec 8.4 - line (d))
        # model assumes deterministic environment
        self.model[(state, action)] = (reward, next_state)
        # keep track of predecessors for the pq loop below
        self._update_predecessors(state, action, next_state)

        # compute q value proposed update and update priority queue (Sec 8.4 - line (e-f))
        proposed_update = (
            reward
            + self.discount * self.get_value(next_state)
            - self.get_q_value(state, action)
        )
        if abs(proposed_update) > self.theta:
            self.pq.push((state, action), -abs(proposed_update))

        # loop over n_planning steps while pq is not empty (Sec 8.4 - line(g)
        for i in range(self.n_planning_steps):
            if self.pq.is_empty():
                break

            # pop best update from queue and transition from model
            state, action = self.pq.pop()
            reward, next_state = self.model[(state, action)]

            # update q values for this state-action pair
            super().update(state, action, reward, next_state)

            # loop for all S', A' predicted to lead to the above state
            for s, a in self.predecessors[state]:
                # get predicted reward from the predecessor leading to `state`
                r, _ = self.model[(s, a)]
                # calculate the proposed update to (s,a)
                proposed_update = (
                    r + self.discount * self.get_value(state) - self.get_q_value(s, a)
                )
                # add to priority queue if greater than min threshold
                if abs(proposed_update) > self.theta:
                    self.pq.push((s, a), -abs(proposed_update))


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.key_index = {}  # key to index mapping
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        _, _, item = heapq.heappop(self.heap)
        return item

    def is_empty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        for idx, (p, c, i) in enumerate(self.heap):
            if i == item:
                # item already in, so has either lower or higher priority
                # if already in with smaller priority, don't do anything
                if p <= priority:
                    break
                # if already in with larger priority, update the priority and restore min-heap property
                del self.heap[idx]
                self.heap.append((priority, c, i))
                heapq.heapify(self.heap)
                break
            else:
                # item is not in, so just add to priority queue
                self.push(item, priority)


def dijkstra(mdp):
    # init priority queue with problem start state
    init_state = mdp.reset_state()
    pq = PriorityQueue()
    pq.push(init_state, 0)

    # tracker {child: parent, cost}
    path = {init_state: (None, 0)}

    while not pq.is_empty():
        # visit a node
        state = pq.pop()
        in_cost = path[state][1]

        if mdp.is_goal(state):
            break

        # construct the successors
        actions = mdp.get_possible_actions(state)
        successors = []
        for action in actions:
            next_state, reward = mdp.get_state_reward_transition(state, action)
            # prevent loops
            if next_state == state:
                continue
            successors.append(next_state)

        # relax the successors
        for next_state in successors:
            path_cost = in_cost + 1
            # if never seen record the path with cost
            if next_state not in path:
                path[next_state] = state, path_cost
                pq.push(next_state, path_cost)
            # if visited but the path was longer, then update the pq and the path tracker
            if path_cost < path[next_state][1]:
                pq.update(next_state, path_cost)
                path[next_state] = state, path_cost

    # recontruct shortest path
    # `state` var currently refers to the goal state after the loop above exits
    states_visited = [state]
    while state is not init_state:
        state, _ = path[state]  # grab the parent path[child] = parent
        states_visited.insert(0, state)

    return states_visited
