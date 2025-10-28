from resco_benchmark.experiment_runner.common import *

maps = []

commands = []
for _ in range(cfg.trials):
    for map_name in maps:
        cmd = " ".join(
            [python_cmd, "main.py", "@" + map_name, "@IDQN", "state:drq_nolanes"]
        )
        commands.append(cmd)

if __name__ == "__main__":
    launch_command(commands)
