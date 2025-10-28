from collections import defaultdict

from resco_benchmark.experiment_runner.common import *
from resco_benchmark.experiment_runner.continual.saltlake import maps

for i in range(len(maps)):
    maps[i] = maps[i].replace(" clean_nightly:3", "")

maps = [maps[-2], maps[-1]]


args = defaultdict(list)

# Phasing
# hr_wk = 168  # hours in a week
# phase_periods = [1, 2, 4, 6, 8]  # in weeks (12 weeks total in Q1)
# phase_periods = [wk * hr_wk for wk in phase_periods]
# for phase_period in phase_periods:
#     args["phase"].append(f"phase_period:{phase_period}")

args["hourly"].append("fallback:20")

args["maxq"].append("fallback:20")

args["oracle_vehicles"].append("")

args["swoks"].append("")

criterias = ["phase", "hourly", "maxq", "oracle_vehicles", "std", "kstest"]

# Common for std and kstest
sample_times = {20: [5, 10], 30: [5, 15], 60: [5, 15, 30]}
experts = ["maxpressure", "random", "null"]
fixed_buffer = [True, False]
global_buffers = [True, False]
model_buffer_sizes = [5760, 17280, 86400, "null"]  # 8 hr, 1 day, 5 days, unlimited

devs = [0.2, 0.5, 1.0]  # std
p_vals = [0.05, 0.1]  # kstest

for init_time in sample_times:
    for sample_time in sample_times[init_time]:
        for deviations in devs:
            for expert in experts:
                for fixed in fixed_buffer:
                    for global_buffer in global_buffers:
                        ag = f"expert:{expert} init_time:{init_time} sample_time:{sample_time} deviations:{deviations}"
                        if global_buffer:
                            ag += " global_buffers:True"
                        if fixed:
                            args["std"].append(ag + f" fixed_buffer:True")
                        else:
                            for size in model_buffer_sizes:
                                args["std"].append(ag + f" model_buffer_size:{size}")


for init_time in sample_times:
    for sample_time in sample_times[init_time]:
        for p_value in p_vals:
            for global_buffer in global_buffers:
                ag = (
                    f"init_time:{init_time} sample_time:{sample_time} p_value:{p_value}"
                )
                if global_buffer:
                    ag += " global_buffers:True"
                args["kstest"].append(ag)

commands = []
for _ in range(cfg.trials):
    for map_name in maps:
        for criteria in criterias:
            for arg in args[criteria]:
                cmd = (
                    " ".join(
                        [
                            python_cmd,
                            "main.py",
                            "@" + map_name,
                            "@IMultiDQN",
                            f"criteria:{criteria} ",
                        ]
                    )
                    + arg
                )
                commands.append(cmd)


if __name__ == "__main__":
    launch_command(commands)
