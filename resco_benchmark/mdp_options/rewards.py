from __future__ import annotations
import logging

import numpy as np

from resco_benchmark.config.config import config as cfg
from resco_benchmark.traffic_signal import Signal


logger = logging.getLogger(__name__)


def rlcd(signals):
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        sig_rew = 0
        for i, direction in enumerate(signal.lane_sets):
            queue_sum = 0
            for lane_id in signal.lane_sets[direction]:
                lane = signal.observation.get_lane(lane_id)
                queue_sum += lane.queued
            sig_rew += queue_sum**2
        rewards[signal_id] = -sig_rew
    return rewards


def oracle_delay(signals: dict[str, Signal]) -> dict[str, float]:
    # Will include timeLoss of vehicles not even on a road with an intersection, etc.
    rewards: dict[str, float] = dict()
    total_reward: float = 0.0
    a_signal = list(signals.values())[0]

    for veh_id in a_signal.sumo.vehicle.getIDList():
        total_reward -= a_signal.sumo.vehicle.getTimeLoss(veh_id)

    for signal_id in signals:
        rewards[signal_id] = total_reward
    return rewards


def oracle_delay_depart(signals: dict[str, Signal]) -> dict[str, float]:
    # Will include timeLoss of vehicles not even on a road with an intersection, etc. + departure delay
    rewards: dict[str, float] = dict()
    total_reward: float = 0.0
    a_signal = list(signals.values())[0]

    for veh_id in a_signal.sumo.vehicle.getIDList():
        if a_signal.sumo.vehicle.getRouteIndex(veh_id) == -1:  # Not departed yet
            total_reward -= a_signal.sumo.vehicle.getDepartDelay(veh_id)
        else:
            total_reward -= a_signal.sumo.vehicle.getTimeLoss(veh_id)

    for signal_id in signals:
        rewards[signal_id] = total_reward
    return rewards


def wait(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        rewards[signal_id] = -signals[signal_id].observation.total_wait
    return rewards


def wait_norm(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        rewards[signal_id] = np.clip(
            -signals[signal_id].observation.total_wait / 224, -4, 4
        ).astype(np.float32)
    return rewards


def phase_queue(signals: dict[str, Signal]) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    phase_pairs = cfg["phase_pairs"]
    for signal_id in signals:
        phase_queues = []
        direction_queue_length = []
        signal = signals[signal_id]
        for direction in signal.lane_sets:
            # Add inbound
            queue_length = 0
            for lane_id in signal.lane_sets[direction]:
                lane = signal.observation.get_lane(lane_id)
                queue_length += lane.queued
            direction_queue_length.append(queue_length)

        for pair in phase_pairs:
            phase_queues.append(
                direction_queue_length[pair[0]] + direction_queue_length[pair[1]]
            )

        rewards[signal_id] = -max(phase_queues) / 500
    return rewards


def pressure(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        entering_queued: int = signal.observation.total_queued

        exiting_queued: int = 0
        for lane_id in signal.outbound_lanes:
            dwn_signal_id: str = signal.out_lane_to_signal_id[lane_id]
            if dwn_signal_id is not None:
                lane = signal.signals[dwn_signal_id].observation.get_lane(lane_id)
                exiting_queued += lane.queued

        pressure_ = entering_queued - exiting_queued
        rewards[signal_id] = -pressure_
    return rewards


def coslight(signals) -> dict[str, float]:
    rewards: dict[str, float] = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        entering_queued: int = signal.observation.total_queued

        exiting_queued: int = 0
        for lane_id in signal.outbound_lanes:
            dwn_signal_id: str = signal.out_lane_to_signal_id[lane_id]
            if dwn_signal_id is not None:
                lane = signal.signals[dwn_signal_id].observation.get_lane(lane_id)
                exiting_queued += lane.queued

        delay: float = 0.0
        for lane_id in signal.lanes:
            lane = signal.observation.get_lane(lane_id)
            for veh_id in lane.vehicles:
                vehicle = lane.vehicles[veh_id]
                delay += vehicle.delay

        pressure_ = entering_queued - exiting_queued
        rewards[signal_id] = (
            -(pressure_ + delay + entering_queued + signal.observation.total_wait)
            / 5000.0
        )
    return rewards


def fma2c(signals) -> dict[str, float]:
    region_fringes: dict[str, list[str]] = dict()
    fringe_arrivals: dict[str, int] = dict()
    liquidity: dict[str, int] = dict()
    for manager in cfg.management:
        region_fringes[manager] = list()
        fringe_arrivals[manager] = 0
        liquidity[manager] = 0

    for signal_id in signals:
        signal = signals[signal_id]
        down_streams = cfg[signal_id]["downstream"]
        for key in down_streams:
            neighbor = down_streams[key]
            if (
                neighbor is None
                or cfg.supervisors[neighbor] != cfg.supervisors[signal_id]
            ):
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = cfg.supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    for signal_id in signals:
        signal = signals[signal_id]
        manager = cfg.supervisors[signal_id]
        fringes = region_fringes[manager]
        liquidity[manager] += signal.observation.departed - signal.observation.arrived
        for lane_id in signal.lanes:
            if lane_id in fringes:
                lane = signal.observation.get_lane(lane_id)
                fringe_arrivals[manager] = lane.arrived

    management_neighborhood: dict[str, float] = dict()
    for manager in cfg.management:
        mgr_rew = fringe_arrivals[manager] + liquidity[manager]
        for neighbor in cfg.management_neighbors[manager]:
            mgr_rew += cfg.alpha * (fringe_arrivals[neighbor] + liquidity[neighbor])
        management_neighborhood[manager] = mgr_rew

    rewards: dict[str, float] = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        reward = 0
        for lane_id in signal.lanes:
            lane = signal.observation.get_lane(lane_id)
            reward += lane.queued
            reward += lane.max_wait * cfg.coef
        rewards[signal_id] = -reward

    neighborhood_rewards: dict[str, float] = dict()
    for signal_id in signals:
        sum_reward = rewards[signal_id]

        down_streams = cfg[signal_id]["downstream"]
        for key in down_streams:
            neighbor = down_streams[key]
            if (
                neighbor is not None
                and cfg.supervisors[neighbor] == cfg.supervisors[signal_id]
            ):
                sum_reward += cfg.alpha * rewards[neighbor]
        neighborhood_rewards[signal_id] = sum_reward

    neighborhood_rewards.update(management_neighborhood)
    return neighborhood_rewards


def reward_builder(signals):
    return cfg.reward_builder(signals)
