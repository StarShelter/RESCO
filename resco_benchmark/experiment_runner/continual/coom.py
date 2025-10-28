from resco_benchmark.experiment_runner.common import *
from resco_benchmark.experiment_runner.continual.saltlake import maps

cl_method = ["null", "agem", "clonex", "ewc", "l2", "mas", "vcl"]


commands = []
for _ in range(cfg.trials):
    for map_name in maps:
        for meth in cl_method:
            cmd = " ".join(
                [
                    python_cmd,
                    "main.py",
                    "@" + map_name,
                    "@ISAC",
                    f"cl_method:{meth}",
                ]
            )
            commands.append(cmd)


if __name__ == "__main__":
    launch_command(commands)
