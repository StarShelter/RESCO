import os
from collections import defaultdict

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from resco_benchmark.config.config import config as cfg
from resco_benchmark.utils.graph import combine_results, stack_trials
from resco_benchmark.utils.logs import parse_logs

try:
    matplotlib.use("TkAgg")
    plot = True
except ImportError:
    print('Matplotlib  "TkAgg" not installed. Skipping graph.')
    plot = False


def graph_it(results):
    font = {"size": 32}
    matplotlib.rc("font", **font)

    for stat in ["minmax", "std"]:
        for metric in results.keys():
            if "timeLoss" in metric:
                map_to_plt = average_trials(results, metric)
                make_plot(map_to_plt, metric, stat)


def resolve_map_name(exp_name):
    splitted = exp_name.split("+")
    map_name = splitted[0].replace("saltlake2stateXuniversity_peak", "Univ & State")

    splitted = splitted[1].split("_")
    pretty_name = ""
    peak_hour = ""
    peak_date = ""
    for spl in splitted:
        if "peakHour@" in spl:
            peak_hour = spl.replace("peakHour@", "")
        elif "peakDate@" in spl:
            peak_date = spl.replace("peakDate@", "")
        elif "controlledSignals@B3" in spl:
            map_name += " - East Signal"
        else:
            pretty_name += spl + " "

    peak_time = f"{peak_date}{peak_hour}"

    for word in cfg.names_map.findreplace:
        pretty_name = pretty_name.replace(word, cfg.names_map.findreplace[word])
    return map_name, pretty_name, peak_time


def make_plot(map_to_plt, metric, stat):
    for map_name in map_to_plt:
        fig, ax = plt.subplots()
        fig.set_size_inches(16, 10, forward=True)
        algorithm_results = map_to_plt[map_name]

        for name in algorithm_results:
            x, y, a, b, z = [], [], [], [], []
            sorted_keys = sorted(list(algorithm_results[name].keys()))
            i = 288
            for flow in sorted_keys:
                x.append(i)
                i += 1
                y.append(algorithm_results[name][flow][0])
                z.append(algorithm_results[name][flow][1])
                a.append(algorithm_results[name][flow][2])
                b.append(algorithm_results[name][flow][3])

            print(name, "x")
            print(x)
            print(y)
            print(z)
            print(a)
            print(b)

            color_resolved = None
            for word in cfg.colors_map:
                if word in name:
                    color_resolved = cfg.colors_map[word]

            if color_resolved is None:
                ax.plot(x, y, label=name)
            else:
                ax.plot(x, y, label=name, color=color_resolved)

            if cfg.error_bars:
                if stat == "minmax":
                    if color_resolved is None:
                        ax.fill_between(x, a, b, alpha=0.4)
                    else:
                        ax.fill_between(x, a, b, alpha=0.4, color=color_resolved)
                else:
                    if color_resolved is None:
                        ax.fill_between(
                            x,
                            np.asarray(y) - z,
                            np.asarray(y) + np.asarray(z),
                            alpha=0.4,
                        )
                    else:
                        ax.fill_between(
                            x,
                            np.asarray(y) - z,
                            np.asarray(y) + np.asarray(z),
                            alpha=0.4,
                            color=color_resolved,
                        )

        map_name = f"{map_name} - Trained fresh each hour"
        ax.set_title(map_name)

        plt.legend()
        ax.set_xlim(288, 584)
        ax.set_ylim(4, 40)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Delay (s)")
        fig.tight_layout()

        img_dir = str(
            os.path.join(
                cfg.log_dir,
                "graphs",
                metric.replace(" ", "_").lower(),
                "curriculum",
                stat + "_smooth" + str(cfg.smoothing),
            )
        )
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        file_name = "{0}.png".format(map_name.replace("-", "")).replace(" ", "_")
        fig.savefig(os.path.join(img_dir, file_name))

        plt.close()


def average_trials(results, metric):
    map_to_plt = defaultdict(dict)
    for exp_name in results[metric].keys():
        stack = stack_trials(results, metric, exp_name)
        map_name, pretty_name, time = resolve_map_name(exp_name)

        if "Max Pressure" in pretty_name:
            stk_len = 0
        else:
            stk_len = -10

        exp_avg = np.mean(stack[:, stk_len:])
        exp_std = np.std(stack[:, stk_len:])
        exp_min = np.min(stack[:, stk_len:])
        exp_max = np.max(stack[:, stk_len:])
        if pretty_name not in map_to_plt[map_name]:
            map_to_plt[map_name][pretty_name] = dict()

        map_to_plt[map_name][pretty_name][time] = (
            exp_avg,
            exp_std,
            exp_min,
            exp_max,
        )
    return map_to_plt


if __name__ == "__main__":
    bot_lvl_dirs = list()
    for item in os.walk(cfg.log_dir):
        if len(item[2]) <= 3:
            continue  # Execution failed, skip empty results
        if "config.json" in item[2]:
            bot_lvl_dirs.append(item[0])

    for path in bot_lvl_dirs:
        parse_logs(path, False)

    included_logs = list()
    for log in list(os.listdir(cfg.log_dir)):
        if ".json" in log:
            included_logs.append(log)

    graph_it(combine_results(included_logs))
