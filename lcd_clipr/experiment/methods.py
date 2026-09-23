import argparse
import time
import wandb
from pathlib import Path
import numpy as onp
from jax import random

from lcd_clipr.definitions import (
    BASELINE_CTRL,
    BASELINE_PERTURBED,
    BASELINE_SALT, BASELINE_PEPER,
    BASELINE_CPA,
    BASELINE_GEARS,
    BASELINE_OURS,
)
from lcd_clipr.utils.parse import load_methods_config, timer, load_data, save_json_zip, print_timer


if __name__ == "__main__":
    """
    Runs methods on a data instance and creates predictions 
    """


    parser = argparse.ArgumentParser()
    parser.add_argument("--descr", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--path_results", type=Path, required=True)
    parser.add_argument("--path_data", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--path_methods_config", type=Path, required=True)
    parser.add_argument("--n_samples", type=int, default=1000)
    kwargs = parser.parse_args()

    # generate directory if it doesn't exist
    kwargs.path_results.mkdir(exist_ok=True, parents=True)
    (kwargs.path_results / "logs").mkdir(exist_ok=True, parents=True)

    # load method kwargs
    assert kwargs.fold is not None
    kwargs.fold = "val" if kwargs.fold == 0 else kwargs.fold

    methods_config = load_methods_config(kwargs.path_methods_config, abspath=True)
    assert kwargs.method in methods_config, f"{kwargs.method} not in config with keys {list(methods_config.keys())}"
    config = methods_config[kwargs.method]

    # load data
    train_datasets, test_intv, remaining_intv, data_config, data_seeding = \
        load_data(kwargs.path_data / f"{kwargs.fold}", eval_mode=True)

    train_intv = train_datasets.intv

    n_train_intv = train_intv.shape[0]
    n_test_intv = 0 if test_intv is None else test_intv.shape[0]
    print(f"Loaded envs\n"
          f"train: {n_train_intv}\n"
          f"test: {n_test_intv}")

    """
    Run algorithm
    """
    base_run_name = f"{kwargs.method}_{kwargs.fold}"
    base_method = kwargs.method.split("__")[0] # catch hyperparameter calibration case where name differs

    # prepare envs to predicted
    envs = [train_intv]
    if test_intv is not None:
        envs.append(test_intv)
    if remaining_intv is not None:
        envs.append(remaining_intv)
    envs = onp.concatenate(envs, axis=0)

    split = lambda arr: (arr[:n_train_intv],
                         arr[n_train_intv:n_train_intv + n_test_intv],
                         arr[n_train_intv + n_test_intv:])

    # run method
    pred = dict()
    if base_method == BASELINE_CTRL:
        from lcd_clipr.metrics import ctrl_dist

        with timer(verbose=True) as walltime:
            key = random.PRNGKey(kwargs.seed)
            ctrl = ctrl_dist(key, train_datasets.data_normalized, n=kwargs.n_samples)

        pred["norm_train"] = pred["norm_test"] = pred["norm_remaining"] = ctrl

    elif base_method == BASELINE_PERTURBED:
        from lcd_clipr.metrics import perturbed_dist

        with timer(verbose=True) as walltime:
            key = random.PRNGKey(kwargs.seed)
            perturbed = perturbed_dist(key, train_datasets.data_normalized, n=kwargs.n_samples)

        pred["norm_train"] = pred["norm_test"] = pred["norm_remaining"] = perturbed

    elif base_method == BASELINE_SALT:
        from lcd_clipr.metrics import salt_dist

        with timer(verbose=True) as walltime:
            key = random.PRNGKey(kwargs.seed)
            salt_samples = salt_dist(
                key, train_datasets.data_normalized, train_datasets.intv, envs, n=kwargs.n_samples,
            )

        pred["norm_train"], pred["norm_test"], pred["norm_remaining"] = split(salt_samples)

    elif base_method == BASELINE_PEPER:
        from lcd_clipr.baselines.peper import run_peper

        with timer(verbose=True) as walltime:
            peper_samples = run_peper(
                kwargs.seed, config, train_datasets.data_normalized, train_datasets.intv, envs, n=kwargs.n_samples,
            )

        pred["norm_train"], pred["norm_test"], pred["norm_remaining"] = split(peper_samples)

    elif base_method == BASELINE_CPA:
        from lcd_clipr.baselines.cpa import run_cpa

        with timer(verbose=True) as walltime:
            cpa_samples = run_cpa( # uses count data
                kwargs.seed, config, train_datasets.data_noisy, train_datasets.intv, envs,
                data_seeding["var_names"], n=kwargs.n_samples,
            )

        pred["norm_train"], pred["norm_test"], pred["norm_remaining"] = split(cpa_samples)

    elif base_method == BASELINE_GEARS:
        from lcd_clipr.baselines.gears import run_gears

        with timer(verbose=True) as walltime:
            gears_samples = run_gears(
                kwargs.seed, config, train_datasets.data_normalized, train_datasets.intv, envs,
                data_seeding["var_names"], data_seeding["id"], n=kwargs.n_samples,
            )

        pred["norm_train"], pred["norm_test"], pred["norm_remaining"] = split(gears_samples)

    elif BASELINE_OURS in base_method:
        from lcd_clipr.run import run_algo as run_ours

        with timer(verbose=True) as walltime:
            with wandb.init(config=config, mode="disabled"):
                wandb_config = wandb.config
                wandb_config.update(dict(seed=kwargs.seed), allow_val_change=True)

                with print_timer("run_ours"):
                    our_sampler, ours_causal, ours_aux = run_ours(
                        train_datasets, None, None, config=wandb_config, eval_mode=True, t_init=time.time(),
                    )

                with print_timer("our_sampler"):
                    return_mean = kwargs.n_samples < 0
                    n_samples = abs(kwargs.n_samples)
                    pred_ours = our_sampler(
                        random.PRNGKey(kwargs.seed),
                        envs,
                        n_samples=n_samples,
                        return_mean=return_mean,
                    )
                    print("ours_samples:", pred_ours["counts"].shape)

        pred["state_train"], pred["state_test"], pred["state_remaining"] = split(pred_ours["state"])
        pred["counts_train"], pred["counts_test"], pred["counts_remaining"] = split(pred_ours["counts"])
        if pred_ours["norm"] is not None:
            pred["norm_train"], pred["norm_test"], pred["norm_remaining"] = split(pred_ours["norm"])

        pred["causal"] = ours_causal
        for k, v in ours_aux.items():
            pred[k] = v

    else:
        raise KeyError(f"Unknown method `{kwargs.method}`")

    t_finish = walltime() / 60.0 # mins

    """Save all predictions"""
    pred["method_base"] = base_method
    pred["method"] = kwargs.method
    pred["config"] = config
    pred["data_seeding"] = data_seeding
    pred["walltime"] = t_finish

    pred["intv_train"] = train_datasets.intv
    pred["intv_test"] = test_intv
    pred["intv_remaining"] = remaining_intv
    pred["n_samples"] = kwargs.n_samples

    # save
    save_json_zip(pred, kwargs.path_results / base_run_name)
    # loaded = load_json_zip(kwargs.path_results / filename)

    print(f"{kwargs.descr}: {kwargs.method} seed={kwargs.seed} fold={kwargs.fold} finished successfully.")