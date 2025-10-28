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


metrics = ["vehicles", "timeLoss"]  # , "phase_length", "max_queues", "queue_lengths"]


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
                        "WARNING",
                        log_file,
                        "has",
                        len(log_entry),
                        "episodes, less than",
                        cfg.episodes,
                        "expected",
                    )
                    if "purge_unfinished" in cfg and len(log_entry) < int(cfg.episodes):
                        # Delete file
                        print("Skipping", log_file)
                        # os.remove(fp)
                        continue

                # TODO use json for more than just fallback exp naming
                # hsh_fp = os.path.join(cfg.log_dir, log_file[:-13], log_uuid + ".hsh")
                # with open(hsh_fp, "r") as hshf:
                #     hshfs = "".join(hshf.readlines()[2:])
                #     js = json.loads(hshfs)
                # better_name = f"{js['map']}_#{js['peak_date']}_{js['peak_hour']}+{js['algorithm']}"

                better_name = log_file.split("/")[-1].split(".")[0]
                if "show_hash" in cfg:
                    better_name = better_name[:-8]  # Cut off hashes
                else:
                    better_name = better_name[:-17]  # Cut off hashes

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
    if len(stack) < cfg.trials:
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


# TODO padding like 9.2 = 9.20
def print_summary(summary):
    best_algs = dict()
    best_avgs = dict()
    for map in summary:
        for alg in summary[map]:
            for metric in summary[map][alg]:
                if "var" in metric:
                    continue
                avg = summary[map][alg][metric]
                key = map + metric
                if (
                    key not in best_avgs
                    or avg < best_avgs[key]
                    and metric in ["timeLoss", "max_queues", "queue_lengths"]
                ):
                    best_avgs[key] = avg
                    best_algs[key] = alg

    table = ""
    for map in summary:
        cols = len(metrics) + 1
        header = r"\midrule\multicolumn{COLS}{l}{\textbf{REPLACEME}} \\\midrule"
        header = header.replace("REPLACEME", map.replace("&", "\\&"))
        header = header.replace("COLS", str(cols))
        table += f"\n\n{header}" + "\n"
        sorted_algs = sorted(summary[map].keys())
        for alg in sorted_algs:
            table += f"{alg}"
            for metric in metrics:
                avg = summary[map][alg][metric]
                var = summary[map][alg]["var_" + metric]
                key = map + metric
                if alg == best_algs[key]:
                    table += f"\t&\t$\\mathbf{{{avg:3.2f}}}\t\\pm {var:.2f}$"
                else:
                    table += f"\t&\t${avg:.2f}\t\t \\pm {var:.2f}$"
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


def prettify_names(exp_name):
    pretty_name = exp_name
    try:
        splitted = exp_name.split("+")
        map_name = splitted[0]
        pretty_name = " ".join(splitted[1:])

        if "controlled" in pretty_name or "cntrlldSgnls" in pretty_name:
            sigs = splitted[1].split("_")[0]
            map_name += sigs.split("@")[1]
            pretty_name = pretty_name.replace(sigs + "_", "")

        # print("Preresolution name", pretty_name)
        for word in cfg.names_map.findreplace:
            pretty_name = pretty_name.replace(word, cfg.names_map.findreplace[word])

        pretty_name += "_"  # Append closing _
        # pretty_name = re.sub("@[^_]*_", " ", pretty_name)
        for word in cfg.names_map.blank_parameters:
            pretty_name = pretty_name.replace(
                word, cfg.names_map.blank_parameters[word]
            )
        pretty_name = pretty_name.replace("_", " ")
        pretty_name = pretty_name.strip()
        # print("Resolved name to", pretty_name)

    except:
        map_name = "null"

    if map_name in cfg.names_map.keys():
        map_name = cfg.names_map[map_name]
    return map_name, pretty_name


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
        ax.set_xlabel("Vehicles")
        # Q1 2023
        # ax.set_xlim(24 * 60, 24 * 90)
        # ax.set_ylim(3, 1000)

    ylabel = (
        cfg.names_map["yaxis_" + metric]
        if "yaxis_" + metric in cfg.names_map
        else metric
    )
    ax.set_ylabel(ylabel)

    ax.set_title(f"{map_name}")

    return demand


def skip_x(results, x):
    stacks = defaultdict(list)
    metric = "vehicles"
    for exp_name in results[metric]:
        stack = stack_trials(results, metric, exp_name)

        map_name, pretty_name = prettify_names(exp_name)
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


def graph_it(results):
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

                map_name, pretty_name = prettify_names(exp_name)
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

                # plt.yscale("log")
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
