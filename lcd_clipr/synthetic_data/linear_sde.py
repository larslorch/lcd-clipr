from collections import defaultdict

import numpy as onp
from scipy.linalg import solve_continuous_lyapunov

from lcd_clipr.core import Dataset
from lcd_clipr.synthetic_data.graph import ErdosRenyi, ScaleFree, ScaleFreeOutgoing
from lcd_clipr.definitions import MAX_ENVS_SAVE_INFERRED_STATES, MAX_SAMPLES_SAVE_INFERRED_STATES
from tqdm import tqdm


def sample_linear_params(rng, d, graph, *, param_low, param_high, matrix_eps, degree):

    if graph == "erdos_renyi":
        g = ErdosRenyi(degree)(rng, d)
    elif graph == "scale_free":
        g = ScaleFree(degree)(rng, d)
    elif graph == "scale_free_outgoing":
        g = ScaleFreeOutgoing(degree)(rng, d)
    else:
        raise ValueError(f"Unknown graph type {graph}")

    # sample mask
    w = rng.uniform(low=param_low, high=param_high, size=(d, d)) * rng.choice([-1, 1], size=(d, d))
    w *= (1 - onp.eye(d)) * g

    # make sure the matrix is stable
    max_eval_real = onp.real(onp.linalg.eigvals(w)).max()
    w -= onp.eye(d) * (max_eval_real + matrix_eps)
    assert onp.real(onp.linalg.eigvals(w)).max() < 0
    return w


def sample_perturbation_shifts(rng, d, mode, *, shift_low, shift_high, both_sides, p_off_target=0.05):
    if mode == "on_target":
        c = rng.uniform(low=shift_low, high=shift_high, size=(d, d)) * onp.eye(d)
    elif mode == "off_target":
        c = rng.uniform(low=shift_low, high=shift_high, size=(d, d)) * onp.eye(d)
        off = rng.uniform(low=shift_low, high=shift_high, size=(d, d))
        off *= rng.choice([0, 1], p=[1 - p_off_target, p_off_target], size=(d, d))
        off *= rng.choice([-1, 1], p=[0.5, 0.5], size=(d, d))
        c += off
    elif mode == "random":
        c = rng.uniform(low=shift_low, high=shift_high, size=(d, d))
        c *= rng.choice([0, 1], p=[0.95, 0.05], size=(d, d))
    else:
        raise ValueError(f"Unknown mode {mode}")

    if both_sides:
        c *= rng.choice([-1, 1], size=(d, d))

    return c


def compute_sde_covariance(a, *, sigma=1.0):
    assert a.ndim == 2 and a.shape[0] == a.shape[1]
    d = a.shape[0]
    return solve_continuous_lyapunov(a, -onp.eye(d) * (sigma ** 2))


def simulate_linear_sde(rng, b, c, *, n, a=None, inv_a=None, cov=None, sigma=1.0):
    assert b.ndim == 1 and c.ndim == 1
    assert a is None or (a.ndim == 2 and a.shape[0] == a.shape[1] == b.shape[0])
    assert inv_a is None or (inv_a.ndim == 2 and inv_a.shape[0] == inv_a.shape[1] == b.shape[0])
    if inv_a is None:
        assert a is not None
        inv_a = onp.linalg.inv(a)
    mean = - inv_a @ (b + c)
    if cov is None:
        assert a is not None
        cov = compute_sde_covariance(a, sigma=sigma)
    s = rng.multivariate_normal(mean, cov, size=n)
    return s


def simulate(seed, config):

    if type(seed) == dict:
        rng_seed = seed["rng_seed"]
        a_sde = seed["a"]
        b = seed["b"]
        c = seed["c"]
        train_perturbs = seed["train_perturbs"]
        test_perturbs = seed["test_perturbs"]
        cells_per_perturb = config["cells_per_perturb"]

        if isinstance(train_perturbs, onp.ndarray):
            train_perturbs = train_perturbs.tolist()
        if isinstance(test_perturbs, onp.ndarray):
            test_perturbs = test_perturbs.tolist()

        seeding = seed

    else:
        if "fold" in config:
            if config["fold"] == "val":
                fold_seed = (1, 0)
            else:
                fold_seed = (1, config["fold"])
        else:
            fold_seed = (0, 0)

        rng_seed = (seed, *fold_seed)
        rng = onp.random.default_rng((0, *rng_seed))

        # weights matrix
        d = config["genes"]
        a_causal = sample_linear_params(
            rng,
            d,
            config["graph"],
            param_low=config["param_low"],
            param_high=config["param_high"],
            matrix_eps=config["matrix_eps"],
            degree=config["degree"],
        )
        a_sde = a_causal.T

        if config["bias"]:
            b = rng.uniform(low=-config["param_high"], high=config["param_high"], size=(d,))
        else:
            b = onp.zeros(d, dtype=a_sde.dtype)

        # intervention vectors
        c = sample_perturbation_shifts(
            rng,
            d,
            config["perturbation_mode"],
            shift_low=config["shift_low"],
            shift_high=config["shift_high"],
            both_sides=config["shift_both_sides"],
        )

        # sample envs
        n_test = config["n_perturbations_test"]
        n_perturbations_train = config.get("n_perturbations_train")
        n_train = min(float("inf") if n_perturbations_train is None else n_perturbations_train, d - n_test)

        perturbs = onp.arange(d)[rng.permutation(d)]
        train_perturbs = [None] + perturbs[:n_train].tolist()
        test_perturbs = perturbs[n_train:n_train + n_test].tolist()
        assert len(set(train_perturbs) & set(test_perturbs)) == 0

        cells_per_perturb = config["cells_per_perturb"]

        seeding = dict(
            rng_seed=rng_seed,
            a=a_sde,
            a_causal=a_causal,
            b=b,
            c=c,
            train_perturbs=train_perturbs,
            test_perturbs=test_perturbs,
            cells_per_perturb=cells_per_perturb,
            var_names=[f"g_{i:04d}" for i in range(d)],
        )

    # re-init rng with new seed here to have data sampling reproducible after loading from seed
    perturb_seed = (1, *rng_seed)
    rng = onp.random.default_rng(perturb_seed)

    dataset_train = defaultdict(list)
    dataset_test = defaultdict(list)
    perturbs = train_perturbs + test_perturbs

    inv_a = onp.linalg.inv(a_sde)
    cov = compute_sde_covariance(a_sde)

    # sample all perturbation envs
    for j, env_idx in tqdm(enumerate(perturbs), total=len(perturbs)):

        if env_idx is None:
            c_env = onp.zeros(a_sde.shape[0])
            intv = onp.zeros(a_sde.shape[0]).astype(onp.int32)
        else:
            c_env = c[env_idx]
            intv = onp.eye(a_sde.shape[0])[env_idx].astype(onp.int32)

        # simulate state distribution
        data = simulate_linear_sde(rng, inv_a=inv_a, b=b, c=c_env, cov=cov, n=cells_per_perturb)

        # cap number of true data stored to avoid memory explosion
        if env_idx in train_perturbs and j < MAX_ENVS_SAVE_INFERRED_STATES:
            data_stored = data[:MAX_SAMPLES_SAVE_INFERRED_STATES]
        else:
            data_stored = None

        data_noisy = data

        if env_idx in train_perturbs:
            dataset_train["data_noisy"].append(data_noisy)
            dataset_train["data_true"].append(data_stored)
            dataset_train["intv"].append(intv)

        elif env_idx in test_perturbs:
            dataset_test["data_noisy"].append(data_noisy)
            dataset_test["data_true"].append(data_stored)
            dataset_test["intv"].append(intv)
        else:
            assert False

    dataset_train = dict(**dataset_train)
    dataset_test = dict(**dataset_test)

    dataset_train['intv'] = onp.array(dataset_train['intv'])
    if dataset_test:
        dataset_test['intv'] = onp.array(dataset_test['intv'])

    return Dataset(**dataset_train), \
           Dataset(**dataset_test) if dataset_test else None, \
           None, \
           seeding
