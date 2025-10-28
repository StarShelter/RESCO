from resco_benchmark.experiment_runner.common import *
from resco_benchmark.experiment_runner.continual.saltlake import maps, episodes

method = [
    "fasttrac:True",
    "crelu:True",
    "parameter_reset_freq:48",
    "buffer_size:6307200",
]

decay_period = 7 * 4 * 24 / episodes

commands = []
for _ in range(cfg.trials):
    for map_name in maps:
        for meth in method:

            cmd = " ".join(
                [
                    python_cmd,
                    "main.py",
                    "@" + map_name,
                    "@IDQN",
                    meth,
                    f"epsilon_decay_period:{decay_period}",
                ]
            )
            commands.append(cmd)


if __name__ == "__main__":
    launch_command(commands)
