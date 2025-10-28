from resco_benchmark.experiment_runner.common import *


def parse_directory(log_dir):
    bot_lvl_dirs = list()
    for item in os.walk(log_dir):
        if len(item[2]) <= 3:
            continue  # Execution failed, skip empty results
        if "config.json" in item[2]:
            bot_lvl_dirs.append(item[0])

    return bot_lvl_dirs


if __name__ == "__main__":
    graph_dir = cfg.log_dir
    folders = parse_directory(graph_dir)

    commands = []
    for folder in folders:
        cmd = " ".join(
            [
                python_cmd,
                f"-c \"from resco_benchmark.utils.logs import *; parse_logs(r'{folder}', True)\"",
            ]
        )
        commands.append(cmd)

    launch_command(commands)
