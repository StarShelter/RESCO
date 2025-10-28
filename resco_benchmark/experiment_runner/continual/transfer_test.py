from resco_benchmark.experiment_runner.common import *

from resco_benchmark.experiment_runner.continual.saltlake import maps

# Put the model directories here, like below.
model_loads = [
    "63d936ff-3822-41e1-91e6-ee310220d44b",
    "036d5cb6-832a-43c3-af8a-f725a727359c",
    "a9574767-3644-4e09-b0be-d0f4559ea644",
    "a4bd6567-fe2e-495b-beb2-012c38b5bf64",
    "b229d47e-f2bd-4188-a572-ca8dc8fe2c65",
]

commands = []

for _ in range(cfg.trials):
    for model in model_loads:
        load_model = f"load_model:{model}"
        for map_name in maps:

            cmd = " ".join(
                [
                    python_cmd,
                    "main.py",
                    "@" + map_name,
                    "@IDQN",
                    "load_replay:False",
                    "state:drq_nolanes",
                    "epsilon_begin:0.02",
                    load_model,
                ]
            )
            commands.append(cmd)

if __name__ == "__main__":
    launch_command(commands)
