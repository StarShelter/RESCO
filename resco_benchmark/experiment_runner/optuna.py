from resco_benchmark.experiment_runner.common import *

algs = ["ISAC"]

commands = []
for map_name in [
    # "saltlake2_stateXuniversity run_hour:peak",
    # "saltlake2_400sX200w run_hour:peak",
    "saltlake1B_stateXuniversity peak_date:2023-1-24 peak_hour:64800"
]:
    for alg in algs:
        for _ in range(total_processes):

            if alg == "FIXED":
                obj = "fixed episodes:2 optuna_trials:" + str(
                    cfg.optuna_trials * cfg.episodes / 2
                )
            else:
                obj = "learning_rate"

            cmd = " ".join(
                [
                    python_cmd,
                    "main.py",
                    "@" + map_name,
                    "@" + alg,
                    "optuna_objective:" + obj,
                    "converged:null",  # Run a fixed number of episodes
                ]
            )

            commands.append(cmd)

if __name__ == "__main__":
    launch_command(commands)
