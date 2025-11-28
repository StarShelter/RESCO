from resco_benchmark.experiment_runner.common import *

maps = [
    "grid4x4",
    "arterial4x4",
]

algs = ["IPPO"]
# algs = ["FIXED", "MAXWAVE", "MAXPRESSURE", "IDQN", "MPLight", "IPPO"]
# algs = ["FMA2C"]  # Requires old python


commands = []
for map_name in maps:
    for alg in algs:
        for _ in range(cfg.trials):
            episodes = ""  # Defaults to cfg.episodes
            if "IPPO" in alg or "FMA2C" in alg:
                episodes = "episodes:1400"

            max_distance = None
            if "MAXWAVE" in alg:
                max_distance = 50
            elif "MPLight" in alg:
                max_distance = 9999
            elif "MAXPRESSURE" in alg:
                max_distance = 9999
            elif "FMA2C" in alg:
                max_distance = 50
            if max_distance is not None:
                max_distance = f"max_distance:{max_distance}"
            else:
                max_distance = ""

            if "FIXED" in alg:
                step_length = 5  # Config is setup for 5s steps, but will be equivalent to the original 10s step setting
            else:
                step_length = 10

            cmd = " ".join(
                [
                    python_cmd,
                    "main.py",
                    "@" + map_name,
                    "@" + alg,
                    "flat_state:False",
                    f"step_length:{step_length}",
                    episodes,
                    max_distance,
                ]
            )

            commands.append(cmd)

if __name__ == "__main__":
    launch_command(commands)
