from resco_benchmark.experiment_runner.common import *

maps = [
    "cologne1",
    "ingolstadt1",
    "cologne3",
    "ingolstadt7",
    "cologne8",
    "ingolstadt21",
]

algs = ["IDQN"]
# algs = ["FIXED", "MAXWAVE", "MAXPRESSURE", "IDQN", "MPLight", "IPPO"]
# algs = ["FMA2C"]  # Requires old python


commands = []
for _ in range(cfg.trials):
    for map_name in maps:
        for alg in algs:
            episodes = ""  # Defaults to cfg.episodes
            if "IPPO" in alg or "FMA2C" in alg:
                episodes = "episodes:1400"

            cmd = " ".join(
                [
                    python_cmd,
                    "main.py",
                    "@" + map_name,
                    "@" + alg,
                    episodes,
                ]
            )

            commands.append(cmd)

if __name__ == "__main__":
    launch_command(commands)
