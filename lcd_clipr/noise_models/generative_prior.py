from functools import partial
from collections import defaultdict
import time
import warnings

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # hide gpus to tf to avoid OOM and cuda errors by conflicting with jax

import numpy as onp
import functools
import wandb

import jax
from jax import random
import jax.numpy as jnp
from jax import jit
import optax

from jax.scipy.special import logsumexp

from lcd_clipr.core import normalizer
from lcd_clipr.definitions import SYNC_TIME_MIN
from lcd_clipr.dataloader import make_dataloader

from lcd_clipr.noise_models.generator import GaussianGenerator, MLPGenerator
from lcd_clipr.noise_models.likelihood import zipoisson_generative_model, log_likelihood_given_distribution_params, \
    state_to_warped_poisson_rate, compute_unit_scaling, pull_unit_scaling_into_state_space

from lcd_clipr.optim import init_scheduler, init_optimizer
from lcd_clipr.utils.tree import tree_global_norm

from lcd_clipr.metrics import ed

from lcd_clipr.utils.opt import update_ave, retrieve_ave, at_t
from lcd_clipr.utils.partial import vmap_scan, vmap_chunk
from lcd_clipr.utils.parse import print_timer


def log_domain_mean(logx, axis=None):
    """https://justindomke.wordpress.com/2020/08/14/calculating-variance-in-the-log-domain/ """
    "np.log(np.mean(np.exp(x))) but more stable"
    numpy = onp if type(logx) == onp.ndarray else jnp
    assert isinstance(axis, int) or axis is None
    n = logx.shape[axis] if axis is not None else numpy.prod(logx.shape)
    damax = numpy.max(logx, axis=axis)
    damax_expanded = numpy.expand_dims(damax, axis=axis)
    return numpy.log(numpy.sum(numpy.exp(logx - damax_expanded), axis=axis)) + damax - numpy.log(n)


def log_domain_var(logx, axis=None):
    """like np.log(np.var(np.exp(logx)))
    except more stable"""
    numpy = onp if type(logx) == onp.ndarray else jnp
    assert isinstance(axis, int) or axis is None
    n = logx.shape[axis] if axis is not None else numpy.prod(logx.shape)
    log_xmean = log_domain_mean(logx, axis=axis)
    log_xmean_expanded = numpy.expand_dims(log_xmean, axis=axis)
    return numpy.log(numpy.sum(numpy.expm1(logx - log_xmean_expanded) ** 2, axis=axis)) + 2 * log_xmean - numpy.log(n - 1)


def _generator_loglik_sample(
        key,
        *,
        mc_samples,
        x,
        generator,
        param,
        training,
        zero_inflation,
        unit_scaling,
        link_temp,
        link_shift,
        warp_temp,
        warp_scaling,
):
    # sample from learned state distribution
    key, subk = random.split(key)
    s = generator(subk, param, samples=mc_samples, training=training)

    # apply link function and then warp distribution to obtain poisson rate
    warped_poisson_rate = state_to_warped_poisson_rate(
        s=s,
        unit_scaling=unit_scaling,
        link_temp=link_temp,
        link_shift=link_shift,
        warp_temp=warp_temp,
        warp_scaling=warp_scaling,
    )
    assert warped_poisson_rate.shape == (mc_samples, s.shape[-1])

    # compute log likelihood based on s
    logit_pi = param["logit_pi"] if zero_inflation else None
    assert not zero_inflation or logit_pi.shape == s.shape[-1:]

    # [batch_size, mc_samples]
    loglik = log_likelihood_given_distribution_params(x, warped_poisson_rate, logit_pi)
    assert loglik.shape == (x.shape[-2], mc_samples)
    return loglik


def loss_function(
        generator,
        key,
        step,
        batch,
        param,
        *,
        logsumexp_sched,
        unit_scaling,
        debias,
        link_temp,
        link_shift,
        warp_temp,
        warp_scaling,
        mc_samples,
        zero_inflation,
    ):

    # select parameters corresponding to environment of data
    x = batch["x"]
    param["indiv"] = jax.tree.map(lambda p: p[batch["env"]], param["indiv"])
    param = dict(**param["indiv"], **param["shared"])

    assert x.ndim == 2
    batch_size, d = x.shape

    # sample from learned p(s) and compute log likelihood log p(x|s)
    key, subk = random.split(key)
    loglik = _generator_loglik_sample(
        subk,
        mc_samples=mc_samples,
        x=x,
        generator=generator,
        param=param,
        training=True,
        zero_inflation=zero_inflation,
        unit_scaling=unit_scaling,
        link_temp=link_temp,
        link_shift=link_shift,
        warp_temp=warp_temp,
        warp_scaling=warp_scaling,
    )
    assert loglik.shape == (batch_size, mc_samples)

    # mean over logsumexp schedules if there are multiple
    if not (type(logsumexp_sched) == tuple or type(logsumexp_sched) == list):
        logsumexp_sched = [logsumexp_sched]

    loss = 0.0

    for lse_sched in logsumexp_sched:
        # compute log evidence with logsumexp
        # logsumexp_temp == d corresponds to taking mean in the joint likelihood instead of sum
        lse_temp = lse_sched(step)

        loglik_scaled = loglik / lse_temp
        log_evidence_batch = logsumexp(loglik_scaled, axis=1) - jnp.log(mc_samples)

        if debias:
            log_evidence_batch += jnp.exp(log_domain_var(loglik_scaled, axis=-1) - 2 * log_evidence_batch - onp.log(2) - onp.log(mc_samples))

        log_evidence_batch *= lse_temp

        # mean over batch datapoints (rather than sum) to have magnitude of loss be invariant to batch_size
        log_evidence = jnp.mean(log_evidence_batch, axis=0)
        loss += -log_evidence

    loss /= len(logsumexp_sched)

    return loss, dict(
        evidence=log_evidence,
    )


def _fit_noise_model(
        seed,
        xs,
        zero_inflation=True,
        adjust_unit_scaling=True,
        fixed_unit_scaling=None,
        return_unit_scale_states=False,
        min_unit_scaling=1.0,
        mlp_generator=True,
        logsumexp_temp=None,
        generator_layers=4,
        generator_layers_wide=0,
        generator_hidden=256,
        generator_hidden_wide=0,
        generator_nonlinearity="silu",
        debias=False,
        link_temp=1.0,
        link_shift=0.0,
        warp_temp=0.0,
        warp_scaling=1.0,
        share_pi=True,
        steps=50000,
        optimizer="adam",
        lr="cos__1e-3",
        grad_clip=None,
        weight_decay=0.0,
        batch_size=128,
        batch_size_split=1,
        mc_samples=128,
        n_samples=10000,
        verbose=10,
        eval_mode=False,
        sharding=None,
        t_init=None,
        cluster_t_max=None,
        entropy_regularizer=None,
    ):

    assert entropy_regularizer is None, "`entropy_regularizer` was removed, only kept for legacy configs"
    assert isinstance(xs, (list, tuple))

    n_envs = len(xs)
    d = xs[0].shape[-1]

    # initialize generative prior distribution
    if mlp_generator:
        dist = MLPGenerator(d=d,
                            layers=generator_layers,
                            layers_wide=generator_layers_wide,
                            hidden=generator_hidden,
                            hidden_wide=generator_hidden_wide,
                            nonlinearity=generator_nonlinearity)
    else:
        dist = GaussianGenerator(d=d)

    def generator(k, params, *, samples, training):
        return dist.apply(params["dist"], k, samples, training=training)

    # initialize generator params (one for each dataset) and the shared noise params
    # separate parameters for each dataset
    key = random.PRNGKey(seed)
    key, *subk_gens = random.split(key, n_envs + 1)
    key, *subk = random.split(key, 5)

    param_indiv = dict(
        dist=vmap_scan(partial(dist.init, key=key, mc_samples=1, training=True), to_numpy=True)(jnp.array(subk_gens)),
        logit_pi=random.normal(subk[1], shape=(n_envs, d)),
    )

    # shared parameters for all dataset
    key, *subk = random.split(key, 5)
    param_shared = dict(
        logit_pi=random.normal(subk[1], shape=(d,)),
    )

    if share_pi or not zero_inflation:
        del param_indiv["logit_pi"]

    if not share_pi or not zero_inflation:
        del param_shared["logit_pi"]

    param = dict(indiv=param_indiv, shared=param_shared)

    # scaling state distribution units
    if adjust_unit_scaling:
        if fixed_unit_scaling is not None:
            assert fixed_unit_scaling.shape == (d,)
            print(f"fit_noise_model: Fixing unit_scaling (d={fixed_unit_scaling.shape[0]}, "
                  f"min={fixed_unit_scaling.min():.2f}, median={onp.median(fixed_unit_scaling):.2f}, "
                  f"max={fixed_unit_scaling.max():.2f})", flush=True)
            unit_scaling = fixed_unit_scaling
        else:
            unit_scaling = compute_unit_scaling(xs[0], min_scaling=min_unit_scaling)
    else:
        unit_scaling = onp.ones(d, dtype=jnp.float32)

    # initialize optimizer
    # see https://optax.readthedocs.io/en/latest/_collections/examples/gradient_accumulation.html#interaction-of-optax-multistep-with-schedules
    # lr schedule steps need to account for actual number of steps done by the inner optimizer (passed into MultiSteps)
    # since MultiSteps causes optimizer to only be stepped every `batch_size_split` steps, we have to
    # multiply the number of gradient computations (`steps`) done in training by `batch_size_split` to have
    # `update_steps` be the indended number of update steps
    # otherwise decay does not reach 0 as expected
    update_steps = steps
    steps *= batch_size_split
    opt, _ = init_optimizer(
        optimizer=optimizer,
        learning_rate=lr,
        steps=update_steps,
        grad_clip=grad_clip,
        weight_decay=weight_decay,
    )
    opt = optax.MultiSteps(opt, every_k_schedule=batch_size_split)

    opt_state = opt.init(param)

    # initialize logsumexp temp schedule
    # since we pass the true inner optimizer step counter to the logsumexp temp schedule during training
    # the steps of the logsumexp schedule are intended total number of optimizer steps (`update_steps`)
    assert logsumexp_temp is not None
    logsumexp_sched = init_scheduler(
        config=logsumexp_temp,
        steps=update_steps,
        minimum_value=1.0,
    )

    # define update step
    grad_loss = partial(
        loss_function,
        logsumexp_sched=logsumexp_sched,
        mc_samples=mc_samples,
        unit_scaling=unit_scaling,
        debias=debias,
        link_temp=link_temp,
        link_shift=link_shift,
        warp_temp=warp_temp,
        warp_scaling=warp_scaling,
        zero_inflation=zero_inflation,
    )
    grad_loss = jax.value_and_grad(grad_loss, argnums=4, has_aux=True)

    @functools.partial(jit, donate_argnums=(0, 2, 3, 4))
    def optimizer_step(_key, _step, _batch, _param, _opt_state):
        # compute grad
        _key, _subk = random.split(_key)
        (loss, aux), _dparam = grad_loss(generator, _subk, _step, _batch, _param)

        # update
        _p_update, _opt_state = opt.update(_dparam, _opt_state, _param)
        _param = optax.apply_updates(_param, _p_update)
        step_aux = dict(
            loss=loss,
            dparam=tree_global_norm(_dparam),
            dupdate=tree_global_norm(_p_update),
            **aux,
        )
        return _key, _param, _opt_state, step_aux

    # optimization loop
    dataloader, batch_sharding = make_dataloader(seed, xs, batch_size, batch_size_split, sharding)

    logs = defaultdict(float)
    t_loop = time.time()
    never_printed = True

    for t in range(steps):
        # `step` corresponds to the number of effective optimizer steps done and defines the logsumexp temperature
        # `t` corresponds to the number of gradient computations done
        step = t // batch_size_split

        batch = next(dataloader)
        key, param, opt_state, logs_t = optimizer_step(key, step, batch, param, opt_state)

        logs = update_ave(logs, logs_t)
        assert not onp.isnan(logs_t["loss"]) and not onp.isnan(logs_t["dparam"]), \
            f"NaN detected in loss or dparam at iteration {t}.\n{logs_t}"

        # check for early termination due to cluster job time
        out_of_time = t_init is not None and cluster_t_max is not None and (time.time() - t_init) > (cluster_t_max - SYNC_TIME_MIN) * 60
        terminate_on_this_one = t == steps - 1 or out_of_time

        if not verbose == 0 and (steps < verbose or at_t(t, steps / verbose) or t == steps - 1 or terminate_on_this_one):
            t_elapsed = time.time() - t_loop
            t_loop = time.time()
            ave_logs = retrieve_ave(logs)
            logs = defaultdict(float)

            print_str = (
                f"{step:9d} "
                f"| loss: {ave_logs['loss']:.4f}  "
                f"| evid: {ave_logs['evidence']:.4f}  "
                f"| dupdate: {ave_logs['dupdate']:.2f} "
                f"| min: {(steps - t) * t_elapsed * verbose / steps / 60.0: >4.1f}  "
                f"sec/step: {t_elapsed * verbose / steps : >4.2f}"
            )
            print(print_str, flush=True)

            # if eval_mode:
            #     continue

            if type(logsumexp_sched) == tuple or type(logsumexp_sched) == list:
                ave_logs["logsumexp_temp"] = onp.mean(list(map(lambda sched: sched(step), logsumexp_sched))).item()
            else:
                ave_logs["logsumexp_temp"] = logsumexp_sched(step)

            # evaluate checkpoint
            with print_timer("_fit_noise_model: _callback_metrics", verbose=never_printed):
                key, subk = random.split(key)
                ave_logs.update(**_evaluate_checkpoint(
                    subk,
                    generator=generator,
                    param=param,
                    noise_model=partial(
                        zipoisson_generative_model,
                        unit_scaling=unit_scaling,
                        link_temp=link_temp,
                        link_shift=link_shift,
                        warp_temp=warp_temp,
                        warp_scaling=warp_scaling,
                    ),
                    xs=xs,
                    n_samples=1000, # should be smaller than `n_samples` in `fit_noise_model` to match outside eval
                    chunk_size=16,
                    verbose=never_printed,
                ))
                never_printed = False

            print_str = " " * 10
            if "ed" in ave_logs:
                print_str += f"| ed:    {ave_logs['ed']:.4f}"
            print(print_str, flush=True)

            if eval_mode:
                continue

            if ave_logs:
                # log with `step` as x-axis to have the same number of steps irrespective of batch_size_split
                # and corresponding to true update steps of the parameters
                wandb.log({f"noise_model_metrics/{k}": v for k, v in ave_logs.items()}, step=step)


        if out_of_time:
            print("Exiting _fit_noise_model early due to time constraint.", flush=True)
            break

    # draw samples from p(distribution_params) for each dataset and then convert to p(s)
    subkeys = jnp.array(random.split(key, n_envs))
    param_per_env = dict(
        **param["indiv"],
        **jax.tree.map(lambda a: jnp.broadcast_to(a, (n_envs,) + a.shape), param["shared"]),
    )

    # chunked vmap returning numpy (CPU) arrays to avoid initializing large GPU buffer
    not_training = jnp.array(False, dtype=bool)
    state = vmap_scan(
        partial(generator, samples=n_samples, training=not_training),
        to_numpy=True
    )(subkeys, param_per_env)

    # if we want to return the unit-scaled states, "pull-in" unit scaling into states and return unit scales of one
    if return_unit_scale_states:
        state = pull_unit_scaling_into_state_space(state, unit_scaling)
        unit_scaling = onp.ones_like(unit_scaling)

    kwargs_generative_model = dict(
        unit_scaling=unit_scaling,
        link_temp=link_temp,
        link_shift=link_shift,
        warp_temp=warp_temp,
        warp_scaling=warp_scaling,
    )

    returned_logs = dict()
    del dataloader

    return state, param_per_env, kwargs_generative_model, returned_logs


def _evaluate_checkpoint(key, *, generator, param, noise_model, xs, n_samples, chunk_size=32, verbose=False):

    n_envs = len(xs)

    def _generative_model(k, param_indiv):
        k, subkk = random.split(k)
        s = generator(subkk, dict(**param_indiv, **param["shared"]), samples=n_samples, training=False)
        k, subkk = random.split(k)
        return noise_model(subkk, s, dict(**param_indiv, **param["shared"]))

    # sample from generative model
    # [envs, n_samples, d]
    with print_timer("_fit_noise_model: _callback_metrics generator", verbose=verbose):
        key, subk = random.split(key)
        xs_pred = vmap_chunk(
            _generative_model,
            chunk_size=chunk_size,
            to_numpy=True,
        )(random.split(subk, n_envs), param["indiv"])

    # compute metrics
    with print_timer("_fit_noise_model: _callback_metrics metrics", verbose=verbose):
        metrics = dict()

        key, subk = random.split(key)
        xs_slim = [x[random.permutation(subkk, len(x))[:n_samples].tolist()]
                   for subkk, x in zip(random.split(subk, n_envs), xs)]

        metrics["ed_noisy"] = onp.mean(ed(xs_slim, xs_pred, mask=None))
        metrics["ed_norm"] = onp.mean(ed(normalizer(xs_slim), normalizer(xs_pred), mask=None))
        metrics["ed"] = onp.mean(ed([onp.log1p(x) for x in xs_slim], [onp.log1p(x) for x in xs_pred], mask=None))
        metrics["x_mean_gap"] = onp.mean([(pred.mean(-2) - x.mean(-2)) for pred, x in zip(xs_pred, xs)]).item()
        metrics["x_var_gap"] = onp.mean([(pred.var(-2) - x.var(-2)) for pred, x in zip(xs_pred, xs)]).item()

    return metrics


def fit_noise_model(
        seed,
        datasets,
        sharding,
        eval_mode=False,
        t_init=None,
        cluster_t_max=None,
        **kwargs
    ):
    """
    Fit noise model to data using our generative approach

    Args:
        seed: int
        sharding
        datasets: Dataset
        eval_mode: bool
        t_init
        cluster_t_max
        **kwargs

    Returns:
        Dataset with updated `data` field representing the denoised states
    """
    noise_param = dict()

    # infer state distribution of each dataset
    states_est, param, kwargs_generative_model, logs = _fit_noise_model(
        seed, datasets.data_noisy, sharding=sharding, eval_mode=eval_mode, t_init=t_init, cluster_t_max=cluster_t_max,
        fixed_unit_scaling=datasets.unit_scaling, **kwargs,
    )

    # remember hyperparameters passed to other functions
    if "logit_pi" in param:
        noise_param["logit_pi"] = onp.array(param["logit_pi"])
    noise_param["__train_intv_mask"] = datasets.intv
    noise_param["__kwargs_generative_model"] = kwargs_generative_model

    # generator model parameters no longer needed, so we explicitly free its GPU buffer (won't be freed otherwise)
    del param

    return datasets._replace(data=states_est), noise_param, logs


def _heuristic_test_noise_param(
        noise_param,
        train_intv_mask,
        test_intv_mask,
    ):

    assert train_intv_mask is not None, "In current implementation, train_intv_mask should be remembered."
    assert test_intv_mask is not None, "In current implementation, test_intv_mask should be known."

    if onp.array_equal(train_intv_mask, test_intv_mask):
        return noise_param

    else:

        # take default parameters from observational distribution
        test_param = {k: onp.broadcast_to(v[0], (test_intv_mask.shape[0],) + v[0].shape) for k, v in noise_param.items()}

        # use average parameters across all environments that a given test target was involved in
        # train_in_test[i, j] is true iff any target in train environment i is targeted in test environment j
        train_in_test = onp.einsum("md,nd->mnd", train_intv_mask, test_intv_mask).any(-1)
        train_in_test = onp.where(train_in_test, train_in_test, onp.nan)

        for k, v in noise_param.items():
            env_param = onp.einsum("mk,m...->mk...", train_in_test, v)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice')
                env_param = onp.nanmean(env_param, axis=0)
            if not onp.isnan(env_param).any():
                test_param[k] = env_param

        return test_param


def decode(key, s, noise_param, *, mask_query):
    """
    Sample data `x` from noise model given state `s`.
    Intervention masks are used for smart aggregation of non-shared noise parameters for test distributions.

    Args:
        key: PRNGKey
        s: ndarray of shape [..., n, d]
        noise_param: noise parameters as returned by `fit_noise_model`
        mask_query: ndarray of shape [..., n, d] representing the intervention mask at test time

    Returns:
        ndarray of shape [..., n, d] representing the decoded noisy counts under the noise model
    """
     # construct noise params for test environments in case they are not shared across datasets
    decode_noise_param = {k: v for k, v in noise_param.items() if not k.startswith("__")}
    decode_noise_param = _heuristic_test_noise_param(decode_noise_param, noise_param["__train_intv_mask"], mask_query)

    # sample observations from noise model given state
    subkeys = onp.array(random.split(key, s.shape[0]))
    warped_zipoisson = partial(zipoisson_generative_model, **noise_param["__kwargs_generative_model"])
    x = vmap_scan(warped_zipoisson, to_numpy=True)(subkeys, s, decode_noise_param)

    assert x.ndim == s.ndim
    return x
