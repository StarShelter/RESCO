from __future__ import annotations
import typing

import numpy as np

from resco_benchmark.config.config import config as cfg
from resco_benchmark.traffic_signal import Signal, Lane
from resco_benchmark.utils.utils import one_hot_list


def wave(signals: dict[str, Signal]) -> dict[str, np.ndarray]:
    states: dict[str, typing.Any] = dict()
    for signal_id in signals:
        signal: Signal = signals[signal_id]
        state: np.ndarray = np.zeros(len(signal.lane_sets))

        for i, direction in enumerate(signal.lane_sets):
            for lane_id in signal.lane_sets[direction]:
                lane: Lane = signal.observation.get_lane(lane_id)
                state[i] += lane.queued + lane.approaching

        states[signal_id] = state
    return states


def rlcd(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        sig_obs = []
        for i, direction in enumerate(signal.lane_sets):
            queue_sum = 0
            for lane_id in signal.lane_sets[direction]:
                lane = signal.observation.get_lane(lane_id)
                queue_sum += lane.queued
            if queue_sum < cfg.regular_limit:
                sig_obs.append(0)
            elif cfg.full_limit > queue_sum >= cfg.regular_limit:
                sig_obs.append(1)
            else:
                sig_obs.append(2)
        observations[signal_id] = np.array(sig_obs)
    return observations


def drq(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.current_phase
        lane_dict = dict()
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            total_wait, total_speed = 0, 0
            vehicles = signal.observation.get_lane(lane).vehicles
            for vehicle in vehicles.values():
                total_wait += vehicle.wait
                total_speed += vehicle.average_speed

            lane_obs.append(signal.observation.get_lane(lane).approaching)
            lane_obs.append(total_wait)
            lane_obs.append(signal.observation.get_lane(lane).queued)

            lane_obs.append(total_speed)

            obs.append(lane_obs)
            lane_dict[lane] = lane_obs
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations


def drq_norm(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []
        act_index = signal.current_phase
        lane_dict = dict()
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            if i == act_index:
                lane_obs.append(1)
            else:
                lane_obs.append(0)

            total_wait, total_speed = 0, 0
            vehicles = signal.observation.get_lane(lane).vehicles
            for vehicle in vehicles.values():
                total_wait += vehicle.wait
                total_speed += vehicle.average_speed

            lane_obs.append(signal.observation.get_lane(lane).approaching / 28)
            lane_obs.append(total_wait / 28)
            lane_obs.append(signal.observation.get_lane(lane).queued / 28)

            lane_obs.append(total_speed / 20 / 28)

            obs.append(lane_obs)
            lane_dict[lane] = lane_obs
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations


def extended_state(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [[0.0] * 12]  # Extra space

        obs[0][-1] = signal.observation.arrived
        obs[0][-2] = signal.observation.departed
        for i in signal.time_since_phase:
            if i == signal.current_phase:
                obs[0][i] = 0
                obs[0][-3] = signal.time_since_phase[i]
            else:
                obs[0][i] = signal.time_since_phase[i]

        lane_dict = dict()
        for i, lane in enumerate(signal.lanes):
            lane_obs = []
            sig_lane_obs = signal.observation.get_lane(lane)

            lane_obs.append(sig_lane_obs.approaching)
            lane_obs.append(sig_lane_obs.queued)

            wait_sum, speed_sum, accel_sum, decel_sum, delay_sum = 0, 0, 0, 0, 0
            max_wait, max_speed, max_accel, max_decel, max_delay = 0, 0, 0, 0, 0
            for vehicle in sig_lane_obs.vehicles:
                vehicle = sig_lane_obs.vehicles[vehicle]
                wait_sum += vehicle.wait
                speed_sum += vehicle.average_speed
                delay_sum += vehicle.delay
                if vehicle.wait > max_wait:
                    max_wait = vehicle.wait
                if vehicle.average_speed > max_speed:
                    max_speed = vehicle.average_speed
                if vehicle.delay > max_delay:
                    max_delay = vehicle.delay

                accel = vehicle.acceleration
                if accel < 0:
                    decel = -1 * accel
                    decel_sum += decel
                    if decel > max_decel:
                        max_decel = decel
                elif accel > 0:
                    accel_sum += accel
                    if accel > max_accel:
                        max_accel = accel

            lane_vehicles_cnt = len(sig_lane_obs.vehicles)
            if lane_vehicles_cnt == 0:
                lane_vehicles_cnt = 1
            lane_obs.append(wait_sum / lane_vehicles_cnt)
            lane_obs.append(speed_sum / lane_vehicles_cnt)
            lane_obs.append(accel_sum / lane_vehicles_cnt)
            lane_obs.append(decel_sum / lane_vehicles_cnt)
            lane_obs.append(delay_sum / lane_vehicles_cnt)

            lane_obs.append(max_wait)
            lane_obs.append(max_speed)
            lane_obs.append(max_accel)
            lane_obs.append(max_decel)
            lane_obs.append(max_delay)

            obs.append(lane_obs)
            lane_dict[lane] = lane_obs
        observations[signal_id] = np.expand_dims(np.asarray(obs), axis=0)
    return observations


def coslight(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = []

        phase_pair = cfg["phase_pairs"][signal.current_phase]
        phase_pair = [cfg.directions[phase_pair[0]], cfg.directions[phase_pair[1]]]
        for i, direction in enumerate(signal.lane_sets):
            # Add inbound
            car_num = 0
            halting = 0
            queue_distance = 0
            pressure = 0
            departed = 0
            occupancy = 0

            for lane_id in signal.lane_sets[direction]:
                lane = signal.observation.get_lane(lane_id)
                occupancy_limit = (
                    lane.length if lane.length < cfg.max_distance else cfg.max_distance
                )
                car_num += lane.vehicle_count
                pressure += lane.queued
                max_distance = 0
                total_vehicle_length = 0
                departed += lane.departed
                for vehicle in lane.vehicles.values():
                    total_vehicle_length += vehicle.length + vehicle.min_gap
                    if vehicle.speed == 0:
                        if vehicle.position > max_distance:
                            max_distance = vehicle.position
                    if vehicle.speed <= 0.1:
                        halting += 1  # SUMO defines halting as speed <= 0.1 m/s
                queue_distance += max_distance
                occupancy += total_vehicle_length / occupancy_limit

                # Subtract downstream
                for lid in signal.lane_sets_outbound[direction]:
                    dwn_signal = signal.out_lane_to_signal_id[lid]
                    if dwn_signal in signal.signals:
                        lane = signal.signals[dwn_signal].observation.get_lane(lid)
                        pressure -= lane.queued

            direction_lanes = len(signal.lane_sets[direction])
            if direction_lanes != 0:
                car_num /= direction_lanes
                halting /= direction_lanes
                queue_distance /= direction_lanes
                departed /= direction_lanes
                occupancy /= direction_lanes
                pressure /= direction_lanes

            phase = 1 if i in phase_pair else 0

            obs.append(
                [phase, car_num, queue_distance, occupancy, departed, halting, pressure]
            )
        observations[signal_id] = np.asarray(obs)
    return observations


def mplight(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = [signal.current_phase]
        for direction in signal.lane_sets:
            # Add inbound
            pressure = 0
            for lane_id in signal.lane_sets[direction]:
                lane = signal.observation.get_lane(lane_id)
                pressure += lane.queued

            # Subtract downstream
            for lane_id in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signal_id[lane_id]
                if dwn_signal in signal.signals:
                    lane = signal.signals[dwn_signal].observation.get_lane(lane_id)
                    pressure -= lane.queued
            obs.append(pressure)
        observations[signal_id] = np.asarray(obs)
    return observations


def mplight_full(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = one_hot_list(signal)
        for direction in signal.lane_sets:
            # Add inbound
            queue_length, total_wait, total_speed, tot_approach = 0, 0, 0, 0
            for lane in signal.lane_sets[direction]:
                queue_length += signal.observation[lane]["queue"]
                total_wait += signal.observation[lane]["total_wait"]
                total_speed = 0
                vehicles = signal.observation[lane]["vehicles"]
                for vehicle in vehicles:
                    total_speed += vehicle["speed"]
                tot_approach += signal.observation[lane]["approach"]

            # Subtract downstream
            for lane in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signal_id[lane]
                if dwn_signal in signal.signals:
                    queue_length -= signal.signals[dwn_signal].observation[lane][
                        "queue"
                    ]
            obs.append(queue_length)
            obs.append(total_wait)
            obs.append(total_speed)
            obs.append(tot_approach)
        observations[signal_id] = np.asarray(obs)
    return observations


def advanced_mplight(signals):
    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        obs = one_hot_list(signal)
        for direction in signal.lane_sets:
            total_demand = 0
            inbound_queue_length = 0
            for lane_id in signal.lane_sets[direction]:
                lane = signal.observation.get_lane(lane_id)
                inbound_queue_length += lane.queued

                # Effective running
                vmax = signal.sumo.lane.getMaxSpeed(lane_id)
                for veh_id in lane.vehicles:
                    if lane.vehicles[veh_id].position < vmax * cfg.step_length:
                        total_demand += 1
            obs.append(total_demand)
            if len(signal.lane_sets[direction]) != 0:
                inbound_queue_length /= len(signal.lane_sets[direction])

            outbound_queue_length = 0
            for lane_id in signal.lane_sets_outbound[direction]:
                dwn_signal = signal.out_lane_to_signal_id[lane_id]
                if dwn_signal in signal.signals:
                    lane = signal.signals[dwn_signal].observation.get_lane(lane_id)
                    outbound_queue_length -= lane.queued
            if len(signal.lane_sets_outbound[direction]) != 0:
                outbound_queue_length /= len(signal.lane_sets_outbound[direction])
            obs.append(inbound_queue_length - outbound_queue_length)
        observations[signal_id] = np.asarray(obs)
    return observations


def ma2c(signals):
    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            waves.append(
                signal.observation[lane]["queue"] + signal.observation[lane]["approach"]
            )
        signal_wave[signal_id] = np.clip(
            np.asarray(waves) / cfg.norm_wave, 0, cfg.clip_wave
        )

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if neighbor is not None:
                waves.append(cfg.coop_gamma * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.observation[lane]["max_wait"]
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / cfg.norm_wait, 0, cfg.clip_wait)

        observations[signal_id] = np.concatenate([waves, waits])
    return observations


def fma2c(signals):
    region_fringes = dict()
    for manager in cfg.management:
        region_fringes[manager] = []
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
                    region_fringes[mgr].append(inbounds)

    lane_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane_id in signal.lanes:
            lane = signal.observation.get_lane(lane_id)
            lane_wave[lane_id] = lane.queued + lane.approaching

    manager_obs = dict()
    for manager in region_fringes:
        fringes = region_fringes[manager]
        waves = []
        for direction in fringes:
            summed = 0
            for lane_id in direction:
                summed += lane_wave[lane_id]
            waves.append(summed)
        manager_obs[manager] = np.clip(
            np.asarray(waves) / cfg.norm_wave, 0, cfg.clip_wave
        )

    management_neighborhood = dict()
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        for neighbor in cfg.management_neighbors[manager]:
            neighborhood.append(cfg.alpha * manager_obs[neighbor])
        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane_id in signal.lanes:
            lane = signal.observation.get_lane(lane_id)
            waves.append(lane.queued + lane.approaching)
        signal_wave[signal_id] = np.clip(
            np.asarray(waves) / cfg.norm_wave, 0, cfg.clip_wave
        )

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        down_streams = cfg[signal_id]["downstream"]
        for key in down_streams:
            neighbor = down_streams[key]
            if (
                neighbor is not None
                and cfg.supervisors[neighbor] == cfg.supervisors[signal_id]
            ):
                waves.append(cfg.alpha * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane_id in signal.lanes:
            lane = signal.observation.get_lane(lane_id)
            max_wait = lane.max_wait
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / cfg.norm_wait, 0, cfg.clip_wait)

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations


def fma2c_full(signals):
    region_fringes = dict()
    for manager in cfg.management:
        region_fringes[manager] = []
    for signal_id in signals:
        signal = signals[signal_id]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if (
                neighbor is None
                or cfg.supervisors[neighbor] != cfg.supervisors[signal_id]
            ):
                inbounds = signal.inbounds_fr_direction.get(key)
                if inbounds is not None:
                    mgr = cfg.supervisors[signal_id]
                    region_fringes[mgr] += inbounds

    lane_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        for lane in signal.lanes:
            lane_wave[lane] = (
                signal.observation[lane]["queue"] + signal.observation[lane]["approach"]
            )

    manager_obs = dict()
    for manager in region_fringes:
        lanes = region_fringes[manager]
        waves = []
        for lane in lanes:
            waves.append(lane_wave[lane])
        manager_obs[manager] = np.clip(
            np.asarray(waves) / cfg.norm_wave, 0, cfg.clip_wave
        )

    management_neighborhood = dict()
    for manager in manager_obs:
        neighborhood = [manager_obs[manager]]
        for neighbor in cfg.management_neighbors[manager]:
            neighborhood.append(cfg.alpha * manager_obs[neighbor])
        management_neighborhood[manager] = np.concatenate(neighborhood)

    signal_wave = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = []
        for lane in signal.lanes:
            waves.append(
                signal.observation[lane]["queue"] + signal.observation[lane]["approach"]
            )

            waves.append(signal.observation[lane]["total_wait"])
            total_speed = 0
            vehicles = signal.observation[lane]["vehicles"]
            for vehicle in vehicles:
                total_speed += vehicle["speed"]
            waves.append(total_speed)
        signal_wave[signal_id] = np.clip(
            np.asarray(waves) / cfg.norm_wave, 0, cfg.clip_wave
        )

    observations = dict()
    for signal_id in signals:
        signal = signals[signal_id]
        waves = [signal_wave[signal_id]]
        for key in signal.downstream:
            neighbor = signal.downstream[key]
            if (
                neighbor is not None
                and cfg.supervisors[neighbor] == cfg.supervisors[signal_id]
            ):
                waves.append(cfg.alpha * signal_wave[neighbor])
        waves = np.concatenate(waves)

        waits = []
        for lane in signal.lanes:
            max_wait = signal.observation[lane]["max_wait"]
            waits.append(max_wait)
        waits = np.clip(np.asarray(waits) / cfg.norm_wait, 0, cfg.clip_wait)

        observations[signal_id] = np.concatenate([waves, waits])
    observations.update(management_neighborhood)
    return observations


def state_builder(signals):
    return cfg.state_builder(signals)
