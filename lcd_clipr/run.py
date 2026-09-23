import warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal for")
warnings.filterwarnings("ignore", message="entering retry loop")
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import shutil
import gc
import subprocess
from collections import defaultdict
import time
from pprint import pprint
import datetime
import traceback
import cmcrameri
from argparse import Namespace
from lcd_clipr.definitions import cpu_count, PROJECT_DIR, SUBDIR_EXPERIMENTS, SYNC_TIME_MIN, \
    MAX_ENVS_SAVE_INFERRED_STATES, MAX_SAMPLES_SAVE_INFERRED_STATES

# simulate 2 devices locally to test multi-device code
# os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count=2"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # hide gpus to tf to avoid OOM and cuda errors by conflicting with jax

import wandb
import jax
from jax import random
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import optax
from optax._src import linear_algebra
import numpy as onp
import matplotlib.pyplot as plt

from lcd_clipr.data import make_data
from lcd_clipr.dataloader import make_dataloader
from lcd_clipr.dsm import make_dsm_loss_fun
from lcd_clipr.optim import init_optimizer

from lcd_clipr.metrics import run_all_metrics, is_stable
from lcd_clipr.metrics import salt_dist

from lcd_clipr.models.main import Model

from lcd_clipr.utils.plot import plot, plot_sim_stats
from lcd_clipr.utils.tree import tree_isnan
from lcd_clipr.utils.opt import update_ave, retrieve_ave, at_t, update_theta_config
from lcd_clipr.utils.parse import load_config, print_timer, update_config
from lcd_clipr.utils.version_control import get_gpu_info2
from lcd_clipr.utils.cache import init_cache
from lcd_clipr.utils.partial import device_put_replicate
from lcd_clipr.utils.gi import assemble_gi_dataset, run_all_gi_score_metrics, mask_from_seeding

from lcd_clipr.experiment.analyze_causal import causal_mats, compute_causal_mat_metrics, \
    linear_causal_mean, empirical_differential_expressions

from lcd_clipr.experiment.plot_causal import plot_two_matrices

from lcd_clipr.experiment.plot_config import MATPLOTLIB_RCPARAMS
plt.rcParams.update(MATPLOTLIB_RCPARAMS)
plt.rcParams['figure.facecolor'] = 'white'  # to avoid transparent colors in PyCharm plot viewer


# function caching for fast prototyping
memory = init_cache()
make_data_cached = memory.cache(make_data)


def run_algo_wandb(wandb_config=None, eval_mode=False):
    """Function run by wandb.agent()"""

    # job setup
    t_init = time.time()
    exception_after_termination = None

    # wandb setup
    with wandb.init(**wandb_config):

        try:
            # this config will be set by wandb
            config = wandb.config

            # summary metrics we are interested in
            wandb.define_metric("loss", summary="min")
            wandb_run_dir = os.path.abspath(os.path.join(wandb.run.dir, os.pardir))
            print("wandb directory: ", wandb_run_dir, flush=True)

            """++++++++++++++  Load data   ++++++++++++++"""

            onp.set_printoptions(precision=2, suppress=True, linewidth=200)

            print("\nLoading data ...", flush=True)

            # load or sample data
            data_config_path = PROJECT_DIR / SUBDIR_EXPERIMENTS / config.data_config
            data_config = load_config(data_config_path, abspath=True)
            assert data_config is not None, "data_config could not be loaded; double check the file name and path: " \
                                            f"{data_config_path}"


            with print_timer("make_data"):
                _make_data = make_data if data_config["id"] == "linear" else make_data_cached
                train_datasets, test_datasets, meta_data = _make_data(seed=config.seed, config=data_config)

            print("done.\n", flush=True)

            """++++++++++++++   Run algorithm   ++++++++++++++"""
            with print_timer("run_algo"):
                _ = run_algo(train_datasets, test_datasets, meta_data, config=config, eval_mode=eval_mode, t_init=t_init)

        except Exception as e:
            print("\n\n" + "-" * 30 + "\nwandb sweep exception traceback caught:\n")
            print(traceback.print_exc(), flush=True)
            exception_after_termination = e

        print("End of wandb.init context.\n", flush=True)

    # manual sync of offline wandb run -- this is for some reason much faster than the online mode of wandb
    if wandb_config["mode"] == "offline":
        print("Starting wandb sync ...", flush=True)
        subprocess.call(f"wandb sync {wandb_run_dir}", shell=True)

    # clean up
    try:
        shutil.rmtree(wandb_run_dir)
        print(f"deleted wandb directory after successful sync: `{wandb_run_dir}`", flush=True)
    except OSError as e:
        print("wandb dir not deleted.", flush=True)

    if exception_after_termination is not None:
        raise exception_after_termination

    print(f"End of run_algo_wandb after total walltime: "
          f"{str(datetime.timedelta(seconds=round(time.time() - t_init)))}",
          flush=True)



def run_algo(train_datasets, test_datasets, meta_data, config=None, eval_mode=False, t_init=None):

    if config.train_only:
        test_datasets = None

    onp.set_printoptions(precision=2, suppress=True, linewidth=200)
    key = random.PRNGKey(config.seed)

    pprint(dict(config))

    device_count = jax.device_count()
    devices = jax.devices()
    mesh = mesh_utils.create_device_mesh((device_count,), devices)
    sharding = PositionalSharding(mesh)

    print(f"jax backend:   {jax.default_backend()} ")
    print(f"devices:       {device_count}")
    print(f"sharding:      {sharding}")
    print(f"cpu_count:     {cpu_count}", flush=True)
    print(f"gpu_info:      {get_gpu_info2()}", flush=True)
    print()

    update_config(config.noise_model_kwargs, "t_init", t_init)
    update_config(config.noise_model_kwargs, "cluster_t_max", getattr(config, "cluster_t_max", None))

    wandb_steps_offset = 0

    """++++++++++++++  Print data info  ++++++++++++++"""

    if meta_data is not None and train_datasets.data_normalized is not None:
        var_names = meta_data.get("var_names")
        if var_names is not None:
            envs = defaultdict(list)
            for env_name, env_data in [("train", train_datasets), ("test", test_datasets)]\
                                      if test_datasets is not None else \
                                      [("train", train_datasets)]:
                for j, intv in enumerate(env_data.intv):
                    if (intv := onp.where(intv)[0].tolist()):
                        envs[env_name].append((",".join([var_names[i] for i in intv]) if intv else "ctrl",
                                               env_data.data_normalized[j].shape[0]))
                envs[env_name] = sorted(envs[env_name])
                print(f"{env_name} perturbations: {len(envs[env_name]):3d} "
                      f"(double: {sum([intv.sum() == 2 for intv in env_data.intv]):3d})")
                if env_name == "train":
                    print(f"     {'ctrl':>15s}  {len(train_datasets.data_normalized[0]):5d}")
                for l, env in enumerate(envs[env_name]):
                    print(f"{f'{l + 1:3d}' if not (l + 1) % 5 else '   '}  {env[0]:>15s}  {env[1]:5d}")
                print()

            assert all([env not in set(envs["test"]) for env in envs["train"]]), "train and test environments overlap"


    """++++++++++++++  Fit noise model and store denoised data   ++++++++++++++"""

    model = Model(
        config,
        n_envs=train_datasets.intv.shape[1], # each gene perturbation gets own environment
        d=train_datasets.intv.shape[1],
    )

    print("Fitting noise model ...", flush=True)
    with print_timer("model.fit_noise_model", wandb=True) as wandb_timer:
        train_datasets, noise_fitting_logs = model.fit_noise_model(
            train_datasets,
            sharding=sharding,
            eval_mode=eval_mode,
            mem=None,
        )
        wandb_steps_offset += wandb.run.step + 1 # add noise model fitting steps

    wandb.log(wandb_timer, step=wandb_steps_offset + 0)

    key, subk = random.split(key) # legacy to leave RNG stream unchanged after refactoring


    """++++++++++++++   Model and parameter initialization   ++++++++++++++"""

    # init parameters
    key, subk = random.split(key)
    theta = model.init_params(subk)

    # load parameters to device (replicate across devices)
    theta = device_put_replicate(theta, sharding)
    train_intv_mask = device_put_replicate(train_datasets.intv, sharding)

    assert onp.isclose(train_intv_mask[0].sum(), 0), "First env should be control."

    if not eval_mode and config.get("exit_after_denoising", False):
        print("Exiting after denoising.", flush=True)
        return

    """++++++++++++++  Loss function   ++++++++++++++"""

    loss_and_grad = make_dsm_loss_fun(
        drift=model.drift,
        anneal_scales=model.anneal_scales,
        config=config,
    )

    """++++++++++++++   Optimizer and update step ++++++++++++++"""

    optimizer, learning_rate_schedule = init_optimizer(
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        steps=config.steps,
        grad_clip=config.grad_clip,
        weight_decay=config.weight_decay,
    )
    opt_state = optimizer.init(theta)


    @jax.jit
    def update_step(step, step_key, step_batch, param, mask_train, opt_state_in):
        """
        A single update step of the optimizer
        """
        # update configs for gradient step
        param = update_theta_config(param, step, config.components)

        # compute gradient of objective
        (loss, loss_aux), dparam = loss_and_grad(param, step, step_key, step_batch, mask_train)

        # update parameters
        param_update, opt_state_in = optimizer.update(dparam, opt_state_in, param)
        param = optax.apply_updates(param, param_update)

        # logging
        grad_norm = linear_algebra.global_norm(dparam)
        update_norm = linear_algebra.global_norm(param_update)
        update_norms = dict()
        nan_occurred = tree_isnan(dparam) | tree_isnan(param)
        aux = dict(loss=loss,
                   **loss_aux,
                   grad_norm=grad_norm,
                   update_norm=update_norm,
                   **update_norms,
                   nan_occurred=nan_occurred)

        return (param, opt_state_in), aux

    # print anneal scales
    onp.set_printoptions(precision=6, suppress=True, linewidth=200)

    model_anneal_scales = model.anneal_scales
    sim_stepsizes = model.anneal_dt * (model_anneal_scales / model_anneal_scales[-1])
    print(f"model.anneal_scales:\n{model_anneal_scales[:5]} ... {model_anneal_scales[-5:]}", flush=True)
    print(f"anneal_dt effective:\n{sim_stepsizes[:5]} ... {sim_stepsizes[-5:]}", flush=True)

    onp.set_printoptions(precision=2, suppress=True, linewidth=200)


    """++++++++++++++ Causal inference setup ++++++++++++++"""

    assert config.scale_standardization == "all-std"
    standardization_shift = model.standardize.aux["ctrl_mean"]
    standardization_scale = model.standardize.aux["all_std"]


    """++++++++++++++   Optimization ++++++++++++++"""
    print("Starting inference...")

    train_loader, _ = make_dataloader(seed=config.seed,
                                      datasets_arrays=train_datasets.data,
                                      batch_size=config.batch_size,
                                      batch_size_split=1,
                                      sharding=sharding,
                                      datasets_intv=train_datasets.intv)


    # training loop
    logs = defaultdict(float)
    t_loop = time.time()

    train_metrics_cache = None
    train_fit_metrics_cache = None
    test_metrics_cache = None
    gi_score_metrics_cache = None

    with (print_timer("SDE inference", wandb=True) as wandb_timer):
        for t in range(config.steps):

            # sample data batch
            batch = next(train_loader)

            # update step
            key, subk = random.split(key)
            (theta, opt_state), logs_t = update_step(t, subk, batch, theta, train_intv_mask, opt_state)

            # update average of training metrics
            logs = update_ave(logs, logs_t)

            if eval_mode and at_t(t, config.log_every):
                t_elapsed = time.time() - t_loop
                t_loop = time.time()
                ave_logs = retrieve_ave(logs)
                logs = defaultdict(float)
                print_str = f"t: {t: >5d}  loss: {ave_logs['loss']: >12.6f}  " \
                            f"min: {(config.steps - t) * t_elapsed / config.log_every / 60.0: >4.1f}  " \
                            f"sec/step: {t_elapsed / config.log_every: >4.2f}"
                print(print_str, flush=True)
                continue

            # print device sharding
            if t == 0:
                print("batch sharding:")
                jax.debug.visualize_array_sharding(batch["x"])

            if eval_mode:
                continue

            """
            ------------------------------------
            Evaluation and logging 
            Not done when in `eval_mode`
            ------------------------------------
            """
            log_dict = {}
            plot_eval = []

            # check for early termination due to cluster job time
            out_of_time = t_init is not None and (time.time() - t_init) > (config.cluster_t_max - SYNC_TIME_MIN) * 60
            terminate_on_this_one = t == config.steps - 1 or out_of_time

            # train data metrics
            if at_t(t, config.log_every) or at_t(t, config.eval_train_every) or at_t(t, config.plot_every) or terminate_on_this_one:
                log_extensive = at_t(t, config.eval_train_every) or at_t(t, config.plot_every) or terminate_on_this_one

                with print_timer(f"log step={t}", wandb=terminate_on_this_one, verbose=True) as wandb_timer_eval:

                    ave_logs_full = retrieve_ave(logs, mean_only=False)
                    ave_logs = dict(filter(lambda kv: "__" not in kv[0], ave_logs_full["mean"].items()))
                    logs = defaultdict(float)

                    # loss metrics
                    t_elapsed = time.time() - t_loop
                    t_loop = time.time()

                    loss_logs = {
                        **ave_logs,
                        "t_step": t_elapsed / config.log_every,
                    }
                    log_dict.update(loss_logs)

                    log_dict.update(dict(
                        # learning_rate=learning_rate_schedule(opt_state[1][2].count.item()),
                        learning_rate=learning_rate_schedule(t).item(),
                    ))

                    if log_extensive:

                        # simulate rollouts from model as predictions
                        with print_timer(f"log step={t} model.predict", wandb=terminate_on_this_one, verbose=True) as wandb_timer_eval_inner:
                            key, subk = random.split(key)
                            pred_train = model.predict(
                                param=theta,
                                key=subk,
                                n_samples_diffuse=config.n_metric_diffuse,
                                n_samples=config.n_metric,
                                mask_train=train_datasets.intv,
                                mask_query=train_datasets.intv,
                            )
                            log_dict["train/metrics/unstable"] = onp.any(~is_stable(pred_train["state"])).astype(float).item()
                            if log_dict["train/metrics/unstable"]:
                                error_msg = "Train samples are unstable. Try reducing dt in simulation."
                                # raise ValueError(error_msg)
                                print(error_msg, flush=True)

                            plot_eval.append(("train", train_datasets, pred_train))

                        log_dict.update(wandb_timer_eval_inner)

                        # log simulation statistics
                        if terminate_on_this_one:
                            wandb_images = plot_sim_stats(
                                t,
                                pred_train["log"],
                                train_datasets.data,
                                model.anneal_scales,
                                to_wandb=False,
                            )
                            if wandb_images:
                                log_dict.update(wandb_images)

                        # eval
                        key, *subk = random.split(key, 4)

                        with print_timer(f"log step={t} run_all_metrics", wandb=terminate_on_this_one, verbose=True) as wandb_timer_eval_inner:

                            train_fit_metrics, _, train_fit_metrics_cache = run_all_metrics(
                                "train_fit",
                                pred_train["state"],
                                {},
                                train_datasets.data,
                                train_datasets.data[0],
                                skip_first=True,
                                short=True,
                                mask=train_datasets.differential_expression_mask,
                                cache=train_fit_metrics_cache,
                            )
                            log_dict.update(**train_fit_metrics)

                            if train_datasets.data_normalized is not None:
                                baselines_norm_train = dict(
                                    # ctrl=ctrl_dist(subk[0], train_datasets.data_normalized, n=config.n_metric),
                                    # perturbed=perturbed_dist(subk[1], train_datasets.data_normalized, n=config.n_metric),
                                    salt=salt_dist(subk[2], train_datasets.data_normalized, train_datasets.intv, train_datasets.intv, n=config.n_metric),
                                )

                                train_metrics, _, train_metrics_cache = run_all_metrics(
                                    "train",
                                    pred_train["norm"],
                                    baselines_norm_train,
                                    train_datasets.data_normalized,
                                    train_datasets.data_normalized[0],
                                    skip_first=True,
                                    short=True,
                                    mask=train_datasets.differential_expression_mask,
                                    cache=train_metrics_cache,
                                )
                                log_dict.update(**train_metrics)

                        log_dict.update(wandb_timer_eval_inner)

                    # print to terminal
                    print_str = f"t: {t: >5d}  loss: {ave_logs['loss']: >12.6f}  " \
                                f"min: {(config.steps - t) * t_elapsed / config.log_every / 60.0: >4.1f} " \
                                f"sec/step: {log_dict['t_step']: >4.2f} "
                    print(print_str, flush=True)

                log_dict.update(wandb_timer_eval)

            # test data metrics
            if at_t(t, config.eval_test_every) or at_t(t, config.plot_every) or (terminate_on_this_one and config.eval_test_every is not None):

                if test_datasets is not None and test_datasets.data_normalized is not None and test_datasets.intv is not None:

                    with print_timer(f"eval step={t}", wandb=terminate_on_this_one) as wandb_timer_eval:

                        # simulate rollouts from model as predictions
                        key, subk = random.split(key)
                        pred_test = model.predict(
                            param=theta,
                            key=subk,
                            n_samples_diffuse=config.n_metric_diffuse,
                            n_samples=config.n_metric,
                            mask_train=train_datasets.intv,
                            mask_query=test_datasets.intv,
                        )
                        log_dict["test/metrics/unstable"] = onp.any(~is_stable(pred_test["state"])).astype(float).item()
                        if log_dict["test/metrics/unstable"]:
                            error_msg = "Test samples are unstable. Try reducing dt in simulation."
                            # raise ValueError(error_msg)
                            print(error_msg, flush=True)

                        plot_eval.append(("test", test_datasets, pred_test))

                        # eval
                        key, *subk = random.split(key, 4)
                        baselines_norm_test = dict(
                            # ctrl=ctrl_dist(subk[0], train_datasets.data_normalized, n=config.n_metric),
                            # perturbed=perturbed_dist(subk[1], train_datasets.data_normalized, n=config.n_metric),
                            salt=salt_dist(subk[2], train_datasets.data_normalized, train_datasets.intv, test_datasets.intv, n=config.n_metric),
                        )

                        test_metrics, _, test_metrics_cache = run_all_metrics(
                            "test",
                            pred_test["norm"],
                            baselines_norm_test,
                            test_datasets.data_normalized,
                            train_datasets.data_normalized[0],
                            skip_first=False,
                            mask=test_datasets.differential_expression_mask,
                            cache=test_metrics_cache,
                        )
                        log_dict.update(**test_metrics)

                    log_dict.update(wandb_timer_eval)

            # test gi data metrics
            if at_t(t, config.eval_gi_every) or (terminate_on_this_one and config.eval_gi_every is not None):

                with print_timer(f"eval gi step={t}", wandb=terminate_on_this_one) as wandb_timer_eval:

                    gi_datasets = assemble_gi_dataset(test_datasets, train_datasets)

                    if gi_datasets is not None and gi_datasets.data_normalized is not None:

                        with print_timer(f"eval gi step={t}.predict", wandb=terminate_on_this_one) as wandb_timer_eval_inner:
                            # assemble reusable rollouts from model
                            reusable_intv = onp.concatenate([sampled[1].intv for sampled in plot_eval], axis=0)
                            reusable_pred = {
                                k: onp.concatenate([sampled[2][k] for sampled in plot_eval], axis=0)
                                for k in ["state", "counts", "norm"]
                            }

                            # simulate rollouts from model as predictions
                            key, subk = random.split(key)
                            pred_gi = model.predict(
                                param=theta,
                                key=subk,
                                n_samples_diffuse=config.n_metric_diffuse,
                                n_samples=config.n_metric,
                                mask_train=train_datasets.intv,
                                mask_query=gi_datasets.intv,
                                reuse=(reusable_pred, reusable_intv),
                            )

                        log_dict.update(wandb_timer_eval_inner)

                        # eval gi scores
                        key, *subk = random.split(key, 4)
                        baselines_norm_gi = dict(
                            # ctrl=ctrl_dist(subk[0], train_datasets.data_normalized, n=config.n_metric),
                            # perturbed=perturbed_dist(subk[1], train_datasets.data_normalized, n=config.n_metric),
                            salt=salt_dist(subk[2], train_datasets.data_normalized, train_datasets.intv, gi_datasets.intv, n=config.n_metric),
                        )

                        with print_timer(f"eval gi step={t}.run_all_gi_score_metrics", wandb=terminate_on_this_one) as wandb_timer_eval_inner:
                            gi_score_metrics, _, gi_score_metrics_cache = run_all_gi_score_metrics(
                                gi_datasets.intv,
                                pred_gi["norm"],
                                baselines_norm_gi,
                                gi_datasets.data_normalized,
                                train_datasets.data_normalized[0],
                                gi_mask=mask_from_seeding(meta_data["gi_genes"], meta_data["var_names"]),
                                thresholds=meta_data["gi_thresholds"],
                                differential_expression_mask=gi_datasets.differential_expression_mask,
                                cache=gi_score_metrics_cache,
                            )
                            log_dict.update(**gi_score_metrics)

                        log_dict.update(wandb_timer_eval_inner)

                log_dict.update(wandb_timer_eval)

            # plotting
            if at_t(t, config.plot_every) or terminate_on_this_one:

                with print_timer(f"plotting step={t}", wandb=terminate_on_this_one) as wandb_timer_plot:

                    for plot_suffix, plot_tars, plot_pred in plot_eval:

                        plot_rows = [
                            (r"$\bf{state}$ space", "state", plot_pred["state"], plot_tars.data, train_datasets.data[0], plot_tars.differential_expression_mask),
                        ]

                        if train_datasets.data_normalized is not None:
                            plot_rows.append(
                                (r"$\bf{norm}$ space",  "norm",  plot_pred["norm"],  plot_tars.data_normalized, train_datasets.data_normalized[0], plot_tars.differential_expression_mask),
                            )

                        for jj, (plot_suptitle, plot_suffix_data, pred_samples, target_samples, ref_samples, ref_mask) in enumerate(plot_rows):
                            if pred_samples is None or target_samples is None:
                                continue

                            if ref_mask is None and ref_samples is not None:
                                ref_mask = onp.zeros((pred_samples.shape[0], pred_samples.shape[-1]), dtype=bool)
                                for i, target_s in enumerate(target_samples):
                                    ref_mask[i, onp.argsort(target_s.mean(0))[-8:]] = True

                            wandb_images = plot(
                                pred_samples,
                                target_samples,
                                plot_tars.intv,
                                title_prefix=f"{plot_suffix}",
                                subfolder_name=f"{plot_suffix}/plots/{plot_suffix_data}",
                                var_names=meta_data.get("var_names"),
                                theta=theta,
                                ref_data=ref_samples,
                                ref_mask=ref_mask,
                                fig_suptitle=plot_suptitle,
                                cmain="#1976D2" if plot_suffix_data == "state" else "#43A047",
                                ctarget="grey",
                                cref="grey",
                                method_name="LCDCLIPR",
                                t_current=t,
                                # plot_params=plot_suffix_data == "state" and plot_suffix == "train",
                                matrix_percentile_cutoff_limits=0.0,
                                plot_differential_marginals=plot_suffix == "train",
                                plot_pairwise_grid=plot_suffix == "train",
                                grid_type="hist-scatter",
                                differential_envs=2,
                                grid_envs=2,
                                grid_per_env=1,
                                grid_cols=7,
                                proj_kde=False,
                                share_scatter_axis_limits=False,
                                prioritize_marker_genes=False,
                                config=config,
                                size_per_var=0.4,
                                to_wandb=False,
                            )

                            if wandb_images:
                                wandb_images = {f"{k}-{plot_suffix}-{plot_suffix_data}" if "matrix" not in k else k: v
                                                for k, v in wandb_images.items()}
                                log_dict.update(wandb_images)

                log_dict.update(wandb_timer_plot)

            # train data metrics
            if hasattr(config, "eval_causal_every") and (at_t(t, config.eval_causal_every) or terminate_on_this_one):

                state_sig, tfm_sig = causal_mats(
                    model,
                    train_intv_mask,
                    theta,
                    shift=standardization_shift,
                    scale=standardization_scale,
                    anneal_sparsify=getattr(config, "causal_anneal_sparsify", 1),
                    causal_mat_kwargs=config.causal_mat_kwargs,
                )

                # compute
                for anneal_time in state_sig.keys():

                    state_params = state_sig[anneal_time]
                    tfm_params = tfm_sig[anneal_time]

                    """
                    Plotting
                    """
                    a_true = meta_data.get("a")
                    b_true = meta_data.get("b")
                    c_true = meta_data.get("c")

                    if a_true is None or b_true is None or c_true is None:
                        de_true = empirical_differential_expressions(
                            train_datasets.data_normalized,
                            train_datasets.intv,
                            ctrl=train_datasets.data_normalized[0],
                        )
                    else:
                        de_true = linear_causal_mean(a=a_true, b=b_true, c=c_true, de=True)

                    if terminate_on_this_one:

                        if config.noise_model is not None:
                            log_dict.update(plot_two_matrices(
                                state_params["a_causal"],
                                meta_data.get("a_causal"),
                                wandb_path=f"causal_t={anneal_time}/state_a_causal",
                                title=f"a (anneal_time = {anneal_time}, t = {t})",
                                cmap=cmcrameri.cm.berlin_r,
                            ))

                        log_dict.update(plot_two_matrices(
                            tfm_params["a_causal"],
                            meta_data.get("a_causal"),
                            wandb_path=f"causal_t={anneal_time}/a_causal",
                            title=f"a_causal transformed (anneal_time = {anneal_time}, t = {t})",
                            cmap=cmcrameri.cm.berlin_r,
                        ))

                        log_dict.update(plot_two_matrices(
                            tfm_params["b"][:, None],
                            meta_data["b"][:, None] if "b" in meta_data else None,
                            wandb_path=f"causal_t={anneal_time}/b",
                            title=f"b transformed (anneal_time = {anneal_time}, t = {t})",
                            cmap=cmcrameri.cm.vanimo,
                        ))

                        log_dict.update(plot_two_matrices(
                            tfm_params["c"],
                            meta_data.get("c"),
                            wandb_path=f"causal_t={anneal_time}/c",
                            title=f"c (anneal_time = {anneal_time}, t = {t})",
                            cmap=cmcrameri.cm.lisbon_r,
                        ))

                        log_dict.update(plot_two_matrices(
                            linear_causal_mean(a=tfm_params["a"], b=tfm_params["b"], c=tfm_params["c"], de=True),
                            de_true,
                            wandb_path=f"causal_t={anneal_time}/de",
                            title=f"DE (anneal_time = {anneal_time}, t = {t})",
                            cmap=cmcrameri.cm.lisbon_r,
                        ))

                    """
                    Metrics
                    """
                    if a_true is not None:
                        a_tfm = tfm_params["a"]
                        causal_metrics = compute_causal_mat_metrics(true=a_true, pred=a_tfm)

                        log_dict.update(**{f"causal_metrics_t={anneal_time}/{k}": v
                                           for k, v in causal_metrics.items() if v is not None})

            if terminate_on_this_one:
                # pprint(log_dict)
                pass

            if log_dict:
                if t == (config.steps - 1):
                    log_dict["nan_occurred"] = int(False)

                assert not eval_mode
                wandb.log(log_dict, step=wandb_steps_offset + t + 1)

            if out_of_time:
                print(f"Terminating due to cluster job time limit at step {t}.", flush=True)
                break

    wandb.log(wandb_timer, step=wandb_steps_offset + t + 1)

    # avoid oom post training when storing data
    del train_loader
    gc.collect()

    # return prediction of causal matrix
    pred_causal_state, pred_causal_tfm = causal_mats(
        model,
        train_intv_mask,
        theta,
        shift=standardization_shift,
        scale=standardization_scale,
        anneal_sparsify=getattr(config, "causal_anneal_sparsify", 1),
        causal_mat_kwargs=config.causal_mat_kwargs,
    )
    pred_causal = dict(
        state=pred_causal_state,
        tfm=pred_causal_tfm,
    )

    if not eval_mode and meta_data.get("a") is not None:
        # sanity check causal metrics computation before return
        wandb.log({
            f"causal_metrics_t={anneal_time}/{k}": v
            for k, v in compute_causal_mat_metrics(
                true=meta_data.get("a"),
                pred=pred_causal["tfm"][anneal_time]["a"],
            ).items()
            if v is not None
        }, step=wandb_steps_offset + t + 2)

    # return sampler for test time interventions
    def eval_time_sampler(eval_key, eval_intv, *, n_samples, n_samples_diffuse=None, return_mean=False):
        return model.predict(
            param=theta,
            key=eval_key,
            n_samples_diffuse=n_samples_diffuse,
            n_samples=n_samples,
            mask_train=train_datasets.intv,
            mask_query=eval_intv,
            return_mean=return_mean,
        )

    return_aux = dict(
        theta=theta,
        noise_param=model.noise_param,
        standardization_aux=model.standardize.aux,
    )
    if getattr(config, "save_inferred_states", False):
        subset = train_datasets.data[
            :MAX_ENVS_SAVE_INFERRED_STATES,
            :MAX_SAMPLES_SAVE_INFERRED_STATES,
        ]
        subset_inferred_states = model.invert_standardize(subset)
        assert subset_inferred_states.ndim == 3
        return_aux["inferred_states"] = subset_inferred_states

    return eval_time_sampler, pred_causal, return_aux


if __name__ == "__main__":

    debug_config = Namespace()
    debug_config.seed = 1234
    debug_config.data_config = "debug_data.yaml"
    debug_config.model = "main"
    debug_config.noise_model = "generative"

    debug_config.save_inferred_states = True
    debug_config.scale_standardization = "all-std"

    debug_config.noise_model_kwargs = dict(
        mlp_generator=True,
        zero_inflation=True,
        share_pi=True,
        n_samples=1000,
        optimizer="adam",
        steps=5000,
        lr="warmup_cos__3e-3__500",
        grad_clip=1.0,
        generator_hidden=32,
        generator_layers=2,
        generator_hidden_wide=64,
        generator_layers_wide=1,
        generator_nonlinearity="silu",
        adjust_unit_scaling=True,
        return_unit_scale_states=False,
        min_unit_scaling=1.0,
        debias=False,
        logsumexp_temp="const__1",
        mc_samples=512,
        batch_size=64,
        batch_size_split=1,
        verbose=5,
    )
    debug_config.exit_after_denoising = False
    debug_config.train_only = False


    # intervention
    debug_config.interv = dict(
        share_params=True,
        linear=dict(
            indiv_downstream_shift=False,
            target_shift_scaling=1.0,
            target_scale_scaling=0.0,
            downstream_shift_scaling=10.0,
        ),
        mlp=dict(
            mode="inject_linear",
            observ_as_env=False,
            all_layers=True,
            agg="sum",
            embed_dim=128,
        ),
    )

    # architecture
    debug_config.full_sigma = False
    debug_config.wiener_procs = None

    debug_config.mlp_hidden = 128
    debug_config.mlp_layers = 2
    debug_config.mlp_activation = "tanh"
    debug_config.batch_size = 128


    debug_config.dsm = dict(
        model="cond",
        param="song",
        loss_weighting=2.0,
    )

    debug_config.noise_annealing = dict(
        sigma_min=0.01,
        sigma_max=10.0,
        sigma_steps=30,
    )

    # SDE simulation
    debug_config.n_metric_diffuse = 300
    debug_config.n_metric = 5000

    debug_config.sde_kwargs = dict(
        anneal_dt=1e-5,
        anneal_steps=100,
        sim_env_chunk_size=None,
    )

    # CLIPR
    debug_config.causal_anneal_sparsify = 5
    debug_config.causal_mat_kwargs = dict(
        tikhonov=1e-2,
    )

    debug_config.steps = 2000
    debug_config.optimizer = "adam"
    debug_config.learning_rate = "cos__1e-3"
    debug_config.grad_clip = 1.0
    debug_config.weight_decay = 0.0

    debug_config.components = dict(
        mlp=dict(active="-", stopgrad="x"), interv=dict(stopgrad="x"),
    )

    # logging
    debug_config.log_every = 1000
    debug_config.eval_train_every = None
    debug_config.eval_test_every = None
    debug_config.eval_gi_every = None
    debug_config.plot_every = None
    debug_config.eval_causal_every = 500

    debug_config.plot_proj = False
    debug_config.proj_umap = False
    debug_config.cluster_t_max = 9999 # is set automatically on cluster to log job results before time runs out

    debug_wandb_config = dict(config=debug_config, mode="disabled")
    run_algo_wandb(wandb_config=debug_wandb_config, eval_mode=False)

