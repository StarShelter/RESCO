import os
import multiprocessing as mp
import subprocess
import shutil
import sys
import time

from resco_benchmark.config.config import config as cfg

if os.getcwd().endswith("experiment_runner"):
    os.chdir("..")
if os.getcwd().endswith("config"):
    os.chdir("..")

if cfg.processors is not None:
    total_processes = min(cfg.processors, mp.cpu_count())
else:
    total_processes = mp.cpu_count() - 2

python_cmd = "python"
if shutil.which("python") is None:
    python_cmd = "python3"


def _fn(x):
    subprocess.call(x, shell=True)


def launch_command(commands):
    if cfg.script_launcher or cfg.log_level == "DEBUG" or cfg.log_level == "NOTSET":
        if len(commands) > 500:
            print(
                "WARNING: TAMU HPRC caps the number of simultaneous running jobs at 500."
            )
        cur_script_name = sys.argv[0].split(os.sep)[-1].replace(".py", "")
        print("Creating script: {0}_commands.sh".format(cur_script_name))
        with open("{0}_commands.sh".format(cur_script_name), "w") as f:
            cp_loc = ";mv \$TMPDIR/RESCO/results \$SCRATCH/results/SLURM\$SLURM_JOBID"
            for command in commands:
                f.write("sed -i '$d' resco.slurm\n")
                f.write(
                    'echo "'
                    + command
                    + " home_log:True"
                    + cp_loc
                    + '" | tee -a resco.slurm > log.txt\n'
                )
                f.write(
                    f"echo {command}\n"
                )  # Write command to out so slurm ID can be matched to command
                f.write("sbatch ./resco.slurm\n")
                f.write(
                    "sleep 10\n"
                )  # Long sleep, to avoid overloading the scheduler and having a job get rejected
    if not cfg.script_launcher:
        pool = mp.Pool(processes=int(total_processes))
        for command in commands:
            pool.apply_async(_fn, args=(command,))
            time.sleep(10)  # Required for optuna to work w/many processes
        pool.close()
        pool.join()
