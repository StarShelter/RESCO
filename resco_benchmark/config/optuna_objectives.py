import numpy as np

from resco_benchmark.config.config import config as cfg, reload_config


def fixed(trial, trial_fn):
    for signal in ["A3", "B3"]:
        # zeros = list(np.where(np.asarray(cfg[signal].fixed_timings) == 0)[0])
        phase_lengths = list()
        for i in range(8):
            # p = 0
            # if i not in zeros: # Can be used to limit search space
            p = trial.suggest_int(signal + "p" + str(i), 0, 20)
            phase_lengths.append(p)
        cfg[signal].fixed_timings = phase_lengths

    cfg["B3"].fixed_offset = trial.suggest_int("offset", 0, 10)

    results = trial_fn()
    reload_config()
    loss = results["timeLoss"]
    avg = np.mean(loss)
    return avg


def learning_rate(trial, trial_fn):
    lr = trial.suggest_float("lr", 1e-7, 5e-1, log=True)
    cfg.learning_rate = lr

    results = trial_fn()
    reload_config()
    loss = results["timeLoss"]
    avg = np.mean(loss[-5:])
    return avg
