import os
import json
from collections import deque, defaultdict

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from resco_benchmark.config.config import config as cfg
from resco_benchmark.utils.logs import parse_logs

try:
    matplotlib.use("TkAgg")
    plot = True
except ImportError:
    print('Matplotlib  "TkAgg" not installed. GUI unavailable.')
    plot = False


metrics = ["timeLoss", "phase_length", "max_queues", "queue_lengths"]
# metrics = ["vehicles", "timeLoss"]  # Use for graph_by_vehicles
errors = ["std"]  # , "minmax"


def get_val(js, p):
    if p in js:
        val = js[p]
        if type(val) == str and val.isnumeric():
            val = float(val)
        elif type(val) == float:
            val = round(val, 2)
        if p in cfg.names_map:
            p = cfg.names_map[p]
    else:
        val = None
    p = p.replace("_", " ")
    return p, val


def new_pretty_name(log_file, log_uuid, fp):
    try:
        hsh_fp = os.path.join(cfg.log_dir, log_file[:-13], log_uuid + ".hsh")
        with open(hsh_fp, "r") as hshf:
            hshfs = "".join(hshf.readlines()[2:])
            js = json.loads(hshfs)

        # Get map modifer, if any
        map_mod = ""
        ctrl = js["controlled_signals"]
        run_hour = js["run_hour"]
        if ctrl is not None and run_hour is not None:
            raise NotImplementedError
        elif ctrl is not None and run_hour is None:
            map_mod = f"#signals@{ctrl}"
        elif run_hour is not None:
            peak_dt = js["peak_date"]
            peak_hr = js["peak_hour"]
            map_mod = f"#run_peak@{peak_dt}:{peak_hr}"

        map_name = js["map"] + map_mod
        if map_name in cfg.names_map:
            map_name = cfg.names_map[js["map"] + map_mod]
        alg = js["algorithm"]
        if alg in cfg.names_map:
            alg = cfg.names_map[js["algorithm"]]
        better_name = f"{map_name}+{alg}"

        # Set hash back on the log unless overriden
        if "noshow_hash" not in cfg:
            better_name += f" {log_uuid[0:8]}"

        # Include relevant parameters
        relevant_params = cfg.names_map.relevant_params
        params = (
            relevant_params[js["algorithm"]]
            if js["algorithm"] in relevant_params
            else []
        ).copy()
        blank_params = (
            cfg.names_map.blank_params[js["algorithm"]]
            if js["algorithm"] in cfg.names_map.blank_params
            else []
        )
        text_values = cfg.names_map.text_values

        for p in params:
            if type(p) == list:
                param = js[p[0]]
                or_cond = False
                and_cond = False
                if "|" in p[1]:
                    or_cond = True
                    cond = p[1].split("|")
                elif "&" in p[1]:
                    and_cond = True
                    cond = p[1].split("&")
                else:
                    cond = p[1]

                passed = True
                if or_cond:
                    passed = False
                    for oc in cond:
                        if "!" in oc:
                            if param != oc[1:]:
                                passed = True
                        else:
                            if param == oc:
                                passed = True
                elif and_cond:
                    for ac in cond:
                        if "!" in ac:
                            if param == ac[1:]:
                                passed = False
                        else:
                            if param != ac:
                                passed = False
                else:
                    if "!" in cond:
                        if param == cond[1:]:
                            passed = False
                    else:
                        if param != cond:
                            passed = False

                if passed:
                    params.extend(p[2])
            else:
                old_p = p
                p, val = get_val(js, p)
                if len(p) > 1:
                    p = p[0].upper() + p[1:]
                if type(val) == str:
                    if val in text_values:
                        val = text_values[val]
                    else:
                        val = val[0].upper() + val[1:]
                        val = val.replace("_", " ")

                if val is None:
                    pass
                elif old_p in blank_params:
                    better_name += f"  {val}"
                elif type(val) == bool:
                    if val or val is None:
                        better_name += f"  {p}"
                else:
                    better_name += f"  {p}={val}"

    except FileNotFoundError:
        stat = os.stat(fp)
        print(
            f"Config not found {log_file} modification time: {stat.st_mtime}, size: {stat.st_size}"
        )
        better_name = log_file.split("/")[-1].split(".")[0]

        if "show_hash" in cfg:
            better_name = better_name[:-8]  # Cut off hashes
        else:
            better_name = better_name[:-17]  # Cut off hashes
    return better_name


def combine_results(log_files):
    results = defaultdict(dict)
    for log_file in log_files:
        fp = os.path.join(cfg.log_dir, log_file)
        try:
            with open(fp, "r") as f:
                tmp = json.load(f)
        except json.decoder.JSONDecodeError:
            print("Skipping log file, not valid json:", fp)
            continue

        for metric in tmp:
            if metric not in metrics:
                continue
            for log_uuid in tmp[metric]:
                log_entry = tmp[metric][log_uuid]
                if len(log_entry) == 1 and type(log_entry[0]) == list:
                    log_entry = log_entry[0]
                if len(log_entry) < cfg.episodes:
                    print(
                        f"WARNING: skipping {log_file} has only episodes={len(log_entry)}"
                    )
                    continue

                better_name = new_pretty_name(log_file, log_uuid, fp)
                if better_name is None:
                    continue

                if better_name not in results[metric]:
                    results[metric][better_name] = [log_entry]
                else:
                    results[metric][better_name].append(log_entry)

    return results


def stack_trials(results, metric, exp_name, truncated=True):
    stack = list()
    minlen = 0
    maxlen = 0
    for i, log in enumerate(results[metric][exp_name]):
        stack.append(log)
        if len(log) < len(stack[minlen]):
            minlen = i
        if len(log) > len(stack[maxlen]):
            maxlen = i

    # If some trials are shorter, truncate to the shortest
    if truncated:
        for i in range(len(stack)):
            stack[i] = np.asarray(stack[i])[: len(stack[minlen])]
    else:
        new_stack = list()
        for i in range(len(stack)):
            if len(stack[i]) < len(stack[maxlen]):
                stk = stack[i] + stack[maxlen][len(stack[i]) :]
                new_stack.append(stk)
            else:
                stk = stack[i]
                new_stack.append(stk)
        stack = new_stack

    # limit to cfg.episodes
    for i in range(len(stack)):
        if len(stack[i]) > cfg.episodes:
            stack[i] = np.asarray(stack[i])[: cfg.episodes]
    if len(stack) < cfg.trials and metric == metrics[0]:
        print(
            f"WARNING: fewer than {cfg.trials} trials",
            exp_name,
            len(stack),
            len(stack[minlen]),
        )
    return np.stack(stack)


def moving_average_filter(data, window):
    windowed = []
    queue = deque(maxlen=window)
    for i in range(len(data)):
        queue.append(data[i])
        windowed.append(np.mean(queue))
    return np.asarray(windowed)


def plot_line(ax, map_name, name, stat, x, y, a, b, z):
    color_resolved = None
    for word in cfg.colors_map:
        if word in name:
            color_resolved = cfg.colors_map[word]

    if ("IPPO" in name or "FMA2C" in name) and "saltlake" not in map_name.lower():
        ax.plot(110, y[-1], "o", markersize=14, label=name)
        ax.fill_between([110], y[-1], y[-1], alpha=0.0)
        return

    if color_resolved is not None and "resolve_colors" in cfg:
        ax.plot(x, y, label=name, color=color_resolved)
    else:
        ax.plot(x, y, label=name)

    if cfg.error_bars:
        if stat == "minmax":
            if color_resolved is not None and "resolve_colors" in cfg:
                ax.fill_between(x, a, b, alpha=0.4, color=color_resolved)
            else:
                ax.fill_between(x, a, b, alpha=0.4)
        else:
            if color_resolved is not None and "resolve_colors" in cfg:
                ax.fill_between(
                    x,
                    np.asarray(y) - z,
                    np.asarray(y) + np.asarray(z),
                    alpha=0.4,
                    color=color_resolved,
                )
            else:
                ax.fill_between(
                    x,
                    np.asarray(y) - z,
                    np.asarray(y) + np.asarray(z),
                    alpha=0.4,
                )


def print_summary(summary):
    table = ""
    for map_name in summary:
        cols = len(metrics) + 1
        header = r"\midrule\multicolumn{COLS}{l}{\textbf{REPLACEME}} \\\midrule"
        header = header.replace("REPLACEME", map_name.replace("&", "\\&"))
        header = header.replace("COLS", str(cols))
        table += f"\n\n{header}" + "\n"
        sorted_algs = sorted(summary[map_name].keys())
        for alg in sorted_algs:
            table += f"{alg}"
            for metric in metrics:
                avg = summary[map_name][alg][metric]
                var = summary[map_name][alg]["var_" + metric]
                table += f"\t&\t{avg:.2f}({var:.2f})"
            table += "\\\\\n"
        table += "\n"
    print(table)


def save_as_png(fig, map_name, metric, demand, stat):
    img_dir = str(
        os.path.join(
            cfg.log_dir,
            "graphs",
            metric.replace(" ", "_").lower(),
            demand,
            stat + "_smooth" + str(cfg.smoothing),
        )
    )
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    file_name = "{0}.png".format(map_name.replace("-", "")).replace(" ", "_")
    fig.savefig(os.path.join(img_dir, file_name))


def average_over_trials(stack):
    exp_avg = np.mean(stack, axis=0)
    exp_std = np.std(stack, axis=0)
    exp_min = np.min(stack, axis=0)
    exp_max = np.max(stack, axis=0)

    if cfg.smoothing is not None and stack.shape[1] >= cfg.smoothing:
        exp_avg = moving_average_filter(exp_avg, cfg.smoothing)
        exp_std = moving_average_filter(exp_std, cfg.smoothing)
        exp_min = moving_average_filter(exp_min, cfg.smoothing)
        exp_max = moving_average_filter(exp_max, cfg.smoothing)
    return exp_avg, exp_std, exp_min, exp_max


def set_axis_labels(ax, map_name, metric, max_x, max_y):
    if "peak" in map_name.lower():
        demand = "peak"

        # Use when plotting a long-running PG method
        points = np.asarray([0, 20, 40, 60, 80, 100, 110])
        labels = ("0", "20", "40", "60", "80", "100", ".." + str(max_x))
        plt.xticks(points, labels)

        ax.set_xlabel("Episode")
    else:
        demand = "year"
        ax.set_xlabel("Hour")

    ylabel = (
        cfg.names_map["yaxis_" + metric]
        if "yaxis_" + metric in cfg.names_map
        else metric
    )
    ax.set_ylabel(ylabel)

    ax.set_title(f"{map_name}")

    return demand


def graph_it(results):
    font = {"size": 24}
    matplotlib.rc("font", **font)

    summary = dict()
    for stat in errors:
        for metric in results:
            # Merge collisions
            stacks = defaultdict(list)
            for exp_name in results[metric]:
                stack = stack_trials(results, metric, exp_name)

                spl = exp_name.split("+")
                map_name, pretty_name = spl[0], spl[1]
                stacks[(map_name, pretty_name)].append(stack)
                if map_name not in summary:
                    summary[map_name] = dict()
                if pretty_name not in summary[map_name]:
                    summary[map_name][pretty_name] = dict()
            merged_stacks = dict()
            for st in stacks:
                merged_stacks[st] = np.vstack(stacks[st])

            # Stack trials and average over them
            map_to_plt = defaultdict(list)
            for map_name, pretty_name in merged_stacks:
                stack = merged_stacks[(map_name, pretty_name)]

                exp_avg, exp_std, exp_min, exp_max = average_over_trials(stack)

                x = [i + 1 for i, _ in enumerate(exp_avg)]
                map_to_plt[map_name].append(
                    (pretty_name, x, exp_avg, exp_std, exp_min, exp_max)
                )

                if "max" in metric or metric == "num_models":
                    per_ep_max = np.max(stack, axis=1)
                    summary[map_name][pretty_name][metric] = np.round(
                        np.mean(per_ep_max), 2
                    )
                    summary[map_name][pretty_name]["var_" + metric] = np.round(
                        np.std(per_ep_max), 2
                    )
                elif metric == "expert_usage":
                    steps = (
                        exp_avg * (3600 // cfg.step_length) / 2
                    )  # TODO should be div # of lights
                    summary[map_name][pretty_name][metric] = int(
                        np.round(np.sum(steps))
                    )
                    summary[map_name][pretty_name]["var_" + metric] = int(
                        np.round(np.std(steps))
                    )
                else:
                    summary[map_name][pretty_name][metric] = np.round(
                        np.mean(exp_avg), 2
                    )
                    summary[map_name][pretty_name]["var_" + metric] = np.round(
                        np.mean(exp_std), 2
                    )

            # Plot the results
            for map_name in map_to_plt:
                fig, ax = plt.subplots()
                fig.set_size_inches(16, 10, forward=True)
                algorithm_results = map_to_plt[map_name]

                max_x, max_y, boxes, box_lbls = 0, 0, [], []
                legend = list()
                for name, x, y, z, a, b in algorithm_results:
                    max_x = max(max_x, max(x))
                    max_y = max(max_y, max(y))
                    plot_line(ax, map_name, name, stat, x, y, a, b, z)
                    legend.append(name)

                demand = set_axis_labels(ax, map_name, metric, max_x, max_y)

                # Create a new figure for the legend
                if "split_leg" in cfg:
                    legend_fig = plt.figure(figsize=(15, 5))
                    legend_ax = legend_fig.add_subplot(111)
                    legend_ax.axis("off")
                    legend_ax.legend(ax.lines, legend, loc="center", ncol=2)
                else:
                    plt.legend()

                fig.tight_layout()
                if "noshow" not in cfg:
                    if "split_leg" in cfg:
                        legend_fig.show()
                    plt.show()
                save_as_png(fig, map_name, metric, demand, stat)
                if "split_leg" in cfg:
                    save_as_png(legend_fig, map_name + " Legend", metric, demand, stat)
                plt.close()

        print_summary(summary)
        if "noshow_hash" not in cfg:
            print(
                "Hashes are included by default in case relevant_parameters cfg are not unique. To remove them, set 'noshow_hash' in config."
            )


def skip_x(results, x):
    stacks = defaultdict(list)
    metric = "vehicles"
    for exp_name in results[metric]:
        stack = stack_trials(results, metric, exp_name)

        spl = exp_name.split("+")
        map_name, pretty_name = spl[0], spl[1]
        stacks[(map_name, pretty_name)].append(stack)
    merged_stacks = dict()
    for st in stacks:
        merged_stacks[st] = np.vstack(stacks[st])
    skip_x = defaultdict(list)
    for key in merged_stacks:
        map_name, pretty_name = key
        if map_name in skip_x:
            continue
        stack = merged_stacks[key]
        exp_avg, exp_std, exp_min, exp_max = average_over_trials(stack)
        for i in range(len(exp_avg)):
            if exp_avg[i] < x:
                skip_x[map_name].append(1)
            else:
                skip_x[map_name].append(0)
    for key in skip_x:
        skip_x[key] = np.asarray(skip_x[key])
    return skip_x


def graph_by_vehicles(results):
    font = {"size": 24}
    matplotlib.rc("font", **font)

    summary = dict()
    for stat in ["std"]:
        for metric in results:
            if metric == "vehicles":
                continue
            # Merge collisions
            stacks = defaultdict(list)
            for exp_name in results[metric]:
                stack = stack_trials(results, metric, exp_name)

                spl = exp_name.split("+")
                map_name, pretty_name = spl[0], spl[1]
                stacks[(map_name, pretty_name)].append(stack)
                if map_name not in summary:
                    summary[map_name] = dict()
                if pretty_name not in summary[map_name]:
                    summary[map_name][pretty_name] = dict()
            merged_stacks = dict()
            for st in stacks:
                merged_stacks[st] = np.vstack(stacks[st])

            # Stack trials and average over them
            map_to_plt = defaultdict(list)
            x = [i for i in range(100, 6000, 100)]
            skipped_avg = defaultdict(list)
            skipped_std = defaultdict(list)
            for map_name, pretty_name in merged_stacks:
                stack = merged_stacks[(map_name, pretty_name)]

                exp_avg, exp_std, exp_min, exp_max = average_over_trials(stack)
                for i in x:
                    skip = skip_x(results, i)
                    a = skip[map_name] * exp_avg
                    skipped_avg[(map_name, pretty_name)].append(np.mean(a[a != 0]))
                    a = skip[map_name] * exp_std
                    skipped_std[(map_name, pretty_name)].append(np.mean(a[a != 0]))

            for map_name, pretty_name in merged_stacks:
                exp_avg = skipped_avg[(map_name, pretty_name)]
                exp_std = skipped_std[(map_name, pretty_name)]
                map_to_plt[map_name].append((pretty_name, x, exp_avg, exp_std, [], []))

            # Plot the results
            for map_name in map_to_plt:
                fig, ax = plt.subplots()
                fig.set_size_inches(16, 10, forward=True)
                algorithm_results = map_to_plt[map_name]

                for name, x, y, z, a, b in algorithm_results:
                    plot_line(ax, map_name, name, stat, x, y, a, b, z)
                    print(map_name, name, y)

                ax.set_xlabel("Vehicles")
                ylabel = (
                    cfg.names_map["yaxis_" + metric]
                    if "yaxis_" + metric in cfg.names_map
                    else metric
                )
                ax.set_ylabel(ylabel)

                ax.set_title(f"{map_name}")

                plt.legend()
                fig.tight_layout()
                plt.show()
                save_as_png(fig, map_name, metric, "year", stat)
                plt.close()


if __name__ == "__main__":
    included_logs = list()
    for log in list(os.listdir(cfg.log_dir)):
        if ".json" in log:
            included_logs.append(log)

    graph_it(combine_results(included_logs))
    # graph_by_vehicles(combine_results(included_logs))
