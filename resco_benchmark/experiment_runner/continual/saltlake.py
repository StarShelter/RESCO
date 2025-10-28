from resco_benchmark.experiment_runner.common import *

episodes = 2160  # Jan-Feb, Q1 of 2023
maps = [
    "saltlake2_stateXuniversity controlled_signals:['B3']",
    "saltlake2_stateXuniversity controlled_signals:['A3']",
    "saltlake2_400sX200w controlled_signals:['A3']",
    "saltlake2_400sX200w controlled_signals:['B3']",
    "saltlake2_stateXuniversity",
    "saltlake2_400sX200w",
]
for i in range(len(maps)):
    maps[i] = maps[i] + f" episodes:{episodes} clean_nightly:3"


decay_period = 7 * 4 * 24 / episodes

algs = [
    "FIXED",
    "MAXWAVE",
    "MAXPRESSURE",
    f"IDQN epsilon_decay_period:{decay_period}",
    f"MPLight epsilon_decay_period:{decay_period}",
    f"AdvancedMPLight epsilon_decay_period:{decay_period}",
    "IPPO",
    "CoSLight",
    "SWOKS",
    "RLCD",
    "RLCD action_set:Phase plan_length:1",
    "FMA2C",
]


commands = []
for _ in range(cfg.trials):
    for map_name in maps:
        for alg in algs:
            cmd = " ".join(
                [
                    python_cmd,
                    "main.py",
                    "@" + map_name,
                    "@" + alg,
                ]
            )
            commands.append(cmd)

if __name__ == "__main__":
    launch_command(commands)
