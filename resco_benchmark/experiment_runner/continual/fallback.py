from resco_benchmark.experiment_runner.common import *
import datetime

maps = [
    "saltlake2_stateXuniversity controlled_signals:['B3']",
]

times = []

date = datetime.datetime.strptime("2023-1-12", "%Y-%m-%d")
for i in range(336):
    times.append(f"run_hour:peak peak_date:{date.date()} peak_hour:{date.hour*3600}")
    date += datetime.timedelta(hours=1)

commands = []
for _ in range(cfg.trials):
    for map_name in maps:
        for time in times:
            cmd = " ".join(
                [
                    python_cmd,
                    "main.py",
                    "@" + map_name,
                    f"@MAXPRESSURE episodes:1",
                    time,
                ]
            )
            commands.append(cmd)

for _ in range(cfg.trials):
    for map_name in maps:
        for time in times:
            cmd = " ".join([python_cmd, "main.py", "@" + map_name, "@IDQN", time])
            commands.append(cmd)

if __name__ == "__main__":
    launch_command(commands)
