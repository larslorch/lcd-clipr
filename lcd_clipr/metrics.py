from functools import partial
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import jax
import jax.random as random
from jax import vmap, numpy as jnp
from scipy import stats
import numpy as onp


MMD_LOGSPACE = (-2, 1, 10)
DE_RANK_BY = ("delta", "pval")


def squared_norm(x, y):
    # handle singular dims
    x_single = x.ndim == 1
    y_single = y.ndim == 1
    if x_single:
        x = x[..., None, :]
    if y_single:
        y = y[..., None, :]

    # kernel
    k = jnp.power(x[..., None, :] - y[..., None, :, :], 2).sum(-1)

    # handle singular dims
    if x_single and y_single:
        k = k.squeeze((-1, -2))
    elif x_single:
        k = k.squeeze(-2)
    elif y_single:
        k = k.squeeze(-1)
    return k


def rbf_kernel(x, y, *, ls):
    assert type(ls) == float or ls.ndim == 0
    return jnp.exp(- squared_norm(x, y) / (2.0 * (ls ** 2)))


def is_stable(samples):
    nan = onp.isnan(samples)
    inf = onp.abs(samples) > 1e12
    return ~jnp.any(nan | inf, axis=(-1, -2))


@jax.jit
def mmd_fun(x, y, *, ls):
    n_x, n_y = x.shape[-2], y.shape[-2]

    k_xx = rbf_kernel(x, x, ls=ls)
    k_xy = rbf_kernel(x, y, ls=ls)
    k_yy = rbf_kernel(y, y, ls=ls)

    dis =  (1. / (n_x * (n_x - 1))) * (k_xx.sum((-1, -2)) - jnp.einsum("...ii->...", k_xx))
    dis -= (2. / (n_x * n_y)) * k_xy.sum((-1, -2))
    dis += (1. / (n_y * (n_y - 1))) * (k_yy.sum((-1, -2)) - jnp.einsum("...ii->...", k_yy))
    return dis


@jax.jit
def mmd_fun_ave(x, y):
    return vmap(lambda ls: mmd_fun(x, y, ls=ls))(jnp.logspace(*MMD_LOGSPACE) * jnp.sqrt(x.shape[-1])).mean()


@jax.jit
def rmse_m_fun(x, y):
    mu_x = jnp.mean(x, axis=-2)
    mu_y = jnp.mean(y, axis=-2)
    return jnp.sqrt(jnp.square(mu_x - mu_y).mean(-1))


@jax.jit
def rmse_s_fun(x, y):
    std_x = jnp.std(x, axis=-2)
    std_y = jnp.std(y, axis=-2)
    return jnp.sqrt(jnp.square(std_x - std_y).mean(-1))


@jax.jit
def bias_s_fun(x, y):
    std_x = jnp.std(x, axis=-2)
    std_y = jnp.std(y, axis=-2)
    return (std_x - std_y).mean(-1)


@partial(jax.jit, static_argnums=(0,))
def _cdf_distance_fun(p, u, v):
    """
    Based on https://github.com/scipy/scipy/blob/v1.13.1/scipy/stats/_stats_py.py#L10520
    """
    assert u.ndim == v.ndim == 1, f"u and v must be 1D arrays. Got {u.ndim} and {v.ndim}."

    u_sorter = jnp.argsort(u)
    v_sorter = jnp.argsort(v)

    all_values = jnp.concatenate((u, v))
    all_values = jnp.sort(all_values)

    # Compute the differences between pairs of successive values of u and v.
    deltas = jnp.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = jnp.searchsorted(u[u_sorter], all_values[:-1], 'right')
    v_cdf_indices = jnp.searchsorted(v[v_sorter], all_values[:-1], 'right')

    # Calculate the CDFs of u and v assuming equal weights of each particle
    u_cdf = u_cdf_indices / u.size
    v_cdf = v_cdf_indices / v.size

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    if p == 1:
        return jnp.sum(jnp.multiply(jnp.abs(u_cdf - v_cdf), deltas))
    elif p == 2:
        return jnp.sqrt(jnp.sum(jnp.multiply(jnp.square(u_cdf - v_cdf), deltas)))
    else:
        return jnp.power(jnp.sum(jnp.multiply(jnp.power(jnp.abs(u_cdf - v_cdf), p),  deltas)), 1/p)


def earthmover_eltwise_fun(x, y):
    # Wasserstein-1 distance over 1D distributions (marginals)
    return vmap(_cdf_distance_fun, in_axes=(None, -1, -1), out_axes=-1)(1, x, y).mean(-1)


@partial(jax.vmap, in_axes=(None, 0), out_axes=0)
@partial(jax.vmap, in_axes=(0, None), out_axes=0)
def _sum_pairwise_norm(x, y):
    return jnp.linalg.norm(x - y, axis=-1)


def energy_distance_fun(x, y):
    a = _sum_pairwise_norm(x, y).sum((0, 1)) / (x.shape[0] * y.shape[0])
    b = _sum_pairwise_norm(x, x).sum((0, 1)) / (x.shape[0] * (x.shape[0] - 1))
    c = _sum_pairwise_norm(y, y).sum((0, 1)) / (y.shape[0] * (y.shape[0] - 1))
    return 2 * a - b - c


def _mean_de(x, y, *, ctrl):
    ctrl_mean = jnp.mean(ctrl, axis=0)
    de_x = jnp.mean(x, axis=0) - ctrl_mean
    de_y = jnp.mean(y, axis=0) - ctrl_mean
    return de_x, de_y


def pearson_de_fun(x, y, *, ctrl):
    de_x, de_y = _mean_de(x, y, ctrl=ctrl)
    return jnp.corrcoef(de_x, de_y)[0, 1]


def _binary_prec_rec_f1(*, pred, true):
    # precision, recall, and f1 for the positive class assuming binary labels (0 or 1)
    pred_is_pos = jnp.isclose(pred, 1)
    true_is_pos = jnp.isclose(true, 1)
    tp = jnp.sum(pred_is_pos & true_is_pos)
    pred_pos = jnp.sum(pred_is_pos)
    true_pos = jnp.sum(true_is_pos)

    prec_val = jnp.where(jnp.isclose(pred_pos, 0), jnp.nan, tp / pred_pos)
    rec_val = jnp.where(jnp.isclose(true_pos, 0), jnp.nan, tp / true_pos)

    # f1 only undefined when there are no positives at all
    f1_denom = pred_pos + true_pos
    f1_val = jnp.where(jnp.isclose(f1_denom, 0), jnp.nan, 2 * tp / f1_denom)
    return prec_val, rec_val, f1_val


def prec_fun(pred, true):
    return _binary_prec_rec_f1(true=true, pred=pred)[0]


def recall_fun(pred, true):
    return _binary_prec_rec_f1(true=true, pred=pred)[1]


def f1_fun(pred, true):
    return _binary_prec_rec_f1(true=true, pred=pred)[2]


def _binary_roc_auc(*, score, true):
    # AUROC for binary labels (0 or 1) given a continuous score (where higher score means more likely positive)
    true_is_pos = jnp.isclose(true, 1)
    n_pos = jnp.sum(true_is_pos)
    n_neg = jnp.sum(~true_is_pos)

    # average ranks (1-indexed) of the scores, splitting ties evenly
    sorted_score = jnp.sort(score)
    n_less = jnp.searchsorted(sorted_score, score, side="left")
    n_leq = jnp.searchsorted(sorted_score, score, side="right")
    ranks = (n_less + n_leq + 1) / 2.0

    rank_sum_pos = jnp.sum(jnp.where(true_is_pos, ranks, 0.0))
    auroc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    # undefined when only one class is present
    return jnp.where((n_pos == 0) | (n_neg == 0), jnp.nan, auroc)


def _binary_average_precision(*, score, true):
    # average precision for binary labels (0 or 1)
    # given a continuous score (where higher score means more likely positive)
    true_is_pos = jnp.isclose(true, 1).astype(jnp.float32)
    n_pos = jnp.sum(true_is_pos)

    order = jnp.argsort(-score)
    y_sorted = true_is_pos[order]
    tp = jnp.cumsum(y_sorted)
    precision = tp / jnp.arange(1, score.shape[0] + 1)

    # AP = sum_k precision_k * delta_recall_k, where delta_recall_k = y_sorted[k] / n_pos
    ap = jnp.sum(precision * y_sorted) / n_pos
    return jnp.where(n_pos == 0, jnp.nan, ap)


def roc_auc_fun(pred, true):
    return _binary_roc_auc(score=pred, true=true)


def average_precision_fun(pred, true):
    return _binary_average_precision(score=pred, true=true)


def top_hvg_mask(hv_genes, var_names, quantiles=(0, 1), tile=None):
    n = len(var_names)
    assert set(hv_genes.keys()) == set(var_names), "hv_genes keys should match var_names"
    assert sorted(set(hv_genes.values())) == list(range(n)), "hv_genes should be a permutation of var_names"
    lo = (quantiles[0] * n) if quantiles[0] is not None and quantiles[0] > 0 else 0
    hi = (quantiles[1] * n) if quantiles[1] is not None and quantiles[1] < 1 else n
    hv_gene_set = set([g for g, rank in hv_genes.items() if lo <= rank < hi])
    assert len(hv_gene_set) > 0, f"quantiles {quantiles} select no genes (rank range [{lo}, {hi}) of {n})"
    mask = onp.array([v in hv_gene_set for v in var_names], dtype=bool)
    if tile is not None:
        mask = onp.tile(mask, (tile, 1))
    return mask


@jax.jit
def leftsel(mat, mask, maskval=0.0):
    """
    jit/vmap helper function

    Args:
        mat: [N, d]
        mask: [d, ]  boolean

    Returns:
        [N, d] [N, d] with columns of `mat` with `mask` == 1 non-zero a
        and pushed leftmost; the columns with `mask` == 0 are zero

    Example:
        mat
        1 2 3
        4 5 6
        7 8 9

        mask
        1 0 1

        out
        1 3 0
        4 6 0
        7 9 0
    """
    valid_indices = jnp.where(mask, jnp.arange(mask.shape[0]), mask.shape[0])
    padded_mat = jnp.concatenate([mat, maskval * jnp.ones((mat.shape[0], 1))], axis=1)
    padded_valid_mat = padded_mat[:, jnp.sort(valid_indices)]
    return padded_valid_mat


def make_metric(metric):
    """
    Helper function for evaluating a metric for multiple environments with possible unequal shapes
    """

    @partial(jax.jit, static_argnums=(0,))
    def _compute_metric(n_masked, mask, pred, true, n=None, **kwargs):

        assert mask is None or mask.ndim == 2, "mask must be given per environment and gene"

        def _obs_subset(arr):
            if n is not None:
                if arr.ndim == 1:
                    pass
                elif arr.ndim == 2:
                    arr = arr[:n]
                else:
                    raise ValueError(f"Array must be 1D or 2D. Got {arr.ndim}")
            return arr

        def _var_subset(arr, *m):
            # masks array `arr` if `m` is provided and returns the first `n_masked` elements
            if m:
                assert n_masked is not None
                is_1d = arr.ndim == 1
                if is_1d:
                    arr = arr[None]
                if arr.ndim == 2:
                    arr = leftsel(arr, *m)[..., :n_masked]
                if is_1d:
                    arr = arr[0]
            return arr

        def metric_scanner(xs, ys):
            if len(xs) == 1 and len(ys) > 1:
                scanned = [ys]
                def metric_with_kwargs(y, *m):
                    xm = _var_subset(_obs_subset(xs[0]), *m)
                    ym = _var_subset(_obs_subset(y), *m)
                    kwargsm = jax.tree.map(lambda kwarg: _var_subset(kwarg, *m), kwargs)
                    assert xm.ndim == ym.ndim
                    return metric(xm, ym, **kwargsm)

            elif len(xs) > 1 and len(ys) == 1:
                scanned = [xs]
                def metric_with_kwargs(x, *m):
                    xm = _var_subset(_obs_subset(x), *m)
                    ym = _var_subset(_obs_subset(ys[0]), *m)
                    kwargsm = jax.tree.map(lambda kwarg: _var_subset(kwarg, *m), kwargs)
                    assert xm.ndim == ym.ndim
                    return metric(xm, ym, **kwargsm)

            else:
                assert len(xs) == len(ys), f"Number of environments must be the same. Got {len(xs)} and {len(ys)}"
                scanned = [xs, ys]
                def metric_with_kwargs(x, y, *m):
                    xm = _var_subset(_obs_subset(x), *m)
                    ym = _var_subset(_obs_subset(y), *m)
                    kwargsm = jax.tree.map(lambda kwarg: _var_subset(kwarg, *m), kwargs)
                    assert xm.ndim == ym.ndim
                    return metric(xm, ym, **kwargsm)

            # scan over mask if provided, provided as `m` to metric_with_kwargs
            if mask is not None:
                scanned.append(mask)

            return jnp.stack(list(map(metric_with_kwargs, *scanned)))

        metrics = metric_scanner(pred, true)
        return metrics

    def compute_metric(*args, mask, **kwargs):
        if mask is not None:
            n_masked = mask.sum(1).astype(jnp.int32)
            assert jnp.isclose(n_masked, n_masked[0]).all(), \
                f"metric mask must have the same number of ones per environment. "\
                f"Got: {n_masked}\nmask: {mask}"

            n_masked = n_masked[0]
        else:
            n_masked = None

        return onp.array(_compute_metric(n_masked, mask, *args, **kwargs)).tolist()

    return compute_metric


mmd_ave = make_metric(mmd_fun_ave)
ed = make_metric(energy_distance_fun)
w1_eltwise = make_metric(earthmover_eltwise_fun)
rmse_m = make_metric(rmse_m_fun)
rmse_s = make_metric(rmse_s_fun)
bias_s = make_metric(bias_s_fun)
pearson_de = make_metric(pearson_de_fun)

precision = make_metric(prec_fun)
recall = make_metric(recall_fun)
f1 = make_metric(f1_fun)

roc_auc = make_metric(roc_auc_fun)
average_precision = make_metric(average_precision_fun)


def abs_change(x, ref):
    return x - ref


def rel_change(x, ref, absolute_zero=0.0):
    x -= absolute_zero
    ref -= absolute_zero
    if onp.isclose(ref, 0):
        return onp.inf if not onp.isclose(x, 0) else 0.0
    return abs_change(x, ref) / ref


def ctrl_dist(key, data, *, n):
    ctrl = data[0]

    key, subk = random.split(key)
    idx = onp.array(random.permutation(subk, ctrl.shape[0])[:n])

    ctrl = ctrl[None, idx]
    return ctrl


def perturbed_dist(key, data, *, n):
    perturbed = onp.concatenate(data[1:], axis=-2)

    key, subk = random.split(key)
    idx = onp.array(random.permutation(subk, perturbed.shape[0])[:n])

    perturbed = perturbed[None, idx]
    return perturbed


def salt_dist(key, train_data, train_intv, test_intv, *, n, single_gene_only=True, return_single_components=False):
    """
    SALT

    single_gene_only:
        True indicates that only single-gene perturbations are considered to estimate the single gene effect
        False indicates that all perturbations involving a gene are considered to estimate the single gene effect
        In both cases, if a test gene is not observed, the mean perturbation effect is used as the estimate
    """
    ctrl = train_data[0]
    d = ctrl.shape[-1]

    # compute means of single-gene perturbations (or all perturbations involving a gene) for each training environment
    ctrl_mean = ctrl.mean(0)
    if len(train_data) == 1:
        mu = onp.zeros((d, d))
    else:
        perturbed_mean = onp.concatenate(train_data[1:]).mean(0)

        means = defaultdict(list)
        for data, intv in zip(train_data[1:], train_intv[1:]):
            if (not single_gene_only) or intv.sum() == 1:
                for tar in onp.where(intv == 1)[0]:
                    means[tar].append(data.mean(0))

        mu = onp.array([(onp.stack(means[j]).mean(0) if j in means else perturbed_mean) for j in range(d)])
        mu -= ctrl_mean

    # estimate distribution of each test environment
    salt = []
    salt_components = []
    for i, intv in enumerate(test_intv):
        # estimate mean shift
        components = []
        salt_mean = onp.zeros(d)
        tars = onp.where(intv == 1)[0]
        for tar in tars:
            salt_mean += mu[tar]
            components.append(mu[tar])

        # return only the mean
        if n == 0:
            salt.append(ctrl_mean + salt_mean)

        # sample from ctrl and shift each cell
        else:
            key, subk = random.split(key)
            idx = onp.array(random.permutation(subk, ctrl.shape[0])[:n])
            ctrl_sample = ctrl[idx]
            salt.append(ctrl_sample + salt_mean)

        salt_components.append(components)

    salt = onp.stack(salt)

    if return_single_components:
        return salt, salt_components
    else:
        return salt


def eval_metric(
    header,
    name,
    metric,
    pred,
    baselines,
    true,
    smaller_is_better,
    skip_first,
    relative_to_zero=None,
    mask=None,
    **metric_kwargs,
):
    if skip_first:
        true = true[1:]
        if len(pred) > 1:
            pred = pred[1:]
        if mask is not None and len(pred) > 1:
            mask = mask[1:]

    results = dict()
    results_per_env = dict()

    pred_score = metric(pred, true, mask=mask, **metric_kwargs)
    results[f"{header}/metrics/{name}"] = onp.mean(pred_score)
    results_per_env[f"{header}/metrics/{name}"] = pred_score

    sign = -1.0 if smaller_is_better else 1.0

    for method, baseline in baselines.items():
        if skip_first and len(baseline) > 1:
            baseline = baseline[1:]

        baseline_score = metric(baseline, true, mask=mask, **metric_kwargs)

        if relative_to_zero is not None:
            results[f"{header}/metrics_rel_to_{method}/{name}"] = sign * rel_change(onp.mean(pred_score),
                                                                                    onp.mean(baseline_score),
                                                                                    absolute_zero=relative_to_zero)
        else:
            results[f"{header}/metrics_abs_to_{method}/{name}"] = sign * abs_change(onp.mean(pred_score),
                                                                                    onp.mean(baseline_score))

    return results, results_per_env


def _significance_ranks(res, abs_change_j, rank_by):
    if rank_by == "delta":
        # larger absolute mean change is more significant
        ranks = stats.rankdata(-abs_change_j, method="max") - 1

    elif rank_by == "pval":
        # lower p-value is more significant, tie broken by larger absolute mean change
        # (vars with zero mean (atol-)effect are not DE regardless of p-value, which is sensitive to float-imprecision)
        sig_key = onp.where(abs_change_j > 0, res.pvalue, onp.inf)
        pval_code = stats.rankdata(sig_key, method="dense")
        eff_code = stats.rankdata(-abs_change_j, method="dense")
        score = pval_code * (eff_code.max() + 1) + eff_code
        ranks = stats.rankdata(score, method="max") - 1

    else:
        raise ValueError(f"Unknown rank_by {rank_by}")

    return ranks.astype(int)


def _compute_de_worker(pr, *, ctrl, fdr_threshold, atol):
    """Worker function for parallel differential expression computation."""
    try:
        pr = pr.todense()
    except AttributeError:
        pass

    # two-sample (unpaired) Wilcoxon test with BH correction
    res = stats.mannwhitneyu(pr, ctrl, method="asymptotic", alternative="two-sided", axis=0)
    adjusted_pvals_j = stats.false_discovery_control(res.pvalue, method="bh")

    # sign: which group has higher mean (in float64 to avoid spurious atol ties)
    mean_change_j = onp.mean(pr.astype(onp.float64), axis=0) - onp.mean(ctrl.astype(onp.float64), axis=0)
    abs_change_j = onp.where(onp.abs(mean_change_j) < atol, 0.0, onp.abs(mean_change_j))
    signs_j = onp.where(adjusted_pvals_j < fdr_threshold, onp.where(mean_change_j > 0, 1, -1), 0)

    # rank: most significant first (0-indexed), equal significance share a tied rank (so AUROC correctly computes ties)
    ranks_j = {rb: _significance_ranks(res, abs_change_j, rb) for rb in DE_RANK_BY}

    return signs_j, adjusted_pvals_j, ranks_j


def compute_differential_expression_sign(
        pred,
        ctrl,
        *,
        return_details=False,
        fdr_threshold=0.05,
        single_gene_only=False,
        targets=None,
        atol=1e-6,
):
    """
    Perform differential expression sign test using two-sample Wilcoxon rank-sum test (Mann-Whitney U test)
    and Benjamini-Hochberg FDR correction.

    Note: the (unadjusted) pvalues of these two functions are the same:
        sc.tl.rank_genes_groups(..., method="wilcoxon")
        stats.mannwhitneyu(method="asymptotic", alternative="two-sided")
    """
    if single_gene_only:
        assert targets is not None
        assert len(targets) == len(pred)

    try:
        ctrl = ctrl.todense()
    except AttributeError:
        pass
    assert ctrl.ndim == 2
    assert not onp.isnan(ctrl).any()

    if len(pred) > 0:
        assert pred[0].ndim == 2
        if pred[0].shape[0] == 1:
            raise NotImplementedError(
                "Need two sets of samples to do a differential expression sign test."
                "Pass a set of shifted control samples if a method only predicts the mean."
            )

    # Process in parallel
    with Pool(cpu_count()) as pool:
        results = pool.map(
            partial(_compute_de_worker, ctrl=ctrl, fdr_threshold=fdr_threshold, atol=atol),
            pred,
        )

    # Collect results
    signs = []
    adjusted_pvalues = []
    intv = []
    ranks = []

    for j, (signs_j, adjusted_pvals_j, ranks_j) in enumerate(results):
        if single_gene_only and not onp.isclose(targets[j].sum(), 1):
            continue

        signs.append(signs_j)
        adjusted_pvalues.append(adjusted_pvals_j)
        ranks.append(ranks_j)
        if targets is not None:
            intv.append(targets[j])

    signs = onp.array(signs, dtype=int)
    adjusted_pvalues = onp.array(adjusted_pvalues, dtype=float)
    intv = onp.array(intv, dtype=int)
    ranks = {rb: onp.array([r[rb] for r in ranks], dtype=int) for rb in DE_RANK_BY}

    if return_details:
        return signs, adjusted_pvalues, ranks, intv
    return signs


def run_all_metrics(
    header,
    pred,
    baselines,
    true,
    ctrl,
    skip_first,
    short=False,
    mask=None,
    absolute=False,
    cache=None,
):
    cache = cache or {}

    results = dict()
    results_per_env = dict()

    metric_modes = [(None, ""), (mask, "_20")] if mask is not None else [(None, "")]
    if len(true) == 1 and skip_first:
        return results, cache

    for gene_mask, suffix in metric_modes:

        res, res_per_env = eval_metric(header, f"rmse_m{suffix}", rmse_m,
                                       pred,
                                       baselines,
                                       true,
                                       smaller_is_better=True,
                                       skip_first=skip_first,
                                       relative_to_zero=None if absolute else 0.0,
                                       mask=gene_mask)
        results.update(res)
        results_per_env.update(res_per_env)

        if not short:
            res, res_per_env = eval_metric(header, f"pearson_de{suffix}", pearson_de,
                                           pred,
                                           baselines,
                                           true,
                                           smaller_is_better=False,
                                           skip_first=skip_first,
                                           relative_to_zero=None if absolute else -1.0,
                                           mask=gene_mask,
                                           ctrl=ctrl)
            results.update(res)
            results_per_env.update(res_per_env)

            res, res_per_env = eval_metric(header, f"ed{suffix}", ed,
                                           pred,
                                           baselines,
                                           true,
                                           smaller_is_better=True,
                                           skip_first=skip_first,
                                           relative_to_zero=None if absolute else -1.0, # technically 0.0 but approx can be noisy and negative
                                           mask=gene_mask)
            results.update(res)
            results_per_env.update(res_per_env)

            res, res_per_env = eval_metric(header, f"mmd{suffix}", mmd_ave,
                                           pred,
                                           baselines,
                                           true,
                                           smaller_is_better=True,
                                           skip_first=skip_first,
                                           relative_to_zero=None if absolute else -1.0, # technically 0.0 but approx can be noisy and negative
                                           mask=gene_mask)
            results.update(res)
            results_per_env.update(res_per_env)

    return results, results_per_env, cache
