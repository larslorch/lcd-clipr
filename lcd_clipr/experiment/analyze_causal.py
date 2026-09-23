import math

import numpy as onp
from sklearn import metrics as skmetrics

import functools

import jax.numpy as jnp
import numpy as onp
from jax import random, vmap


def maxeig(mat):
    return onp.real(onp.linalg.eigvals(mat)).max()


def linear_causal_mean(*, a, b, c, de=False):
    """
    Returns mu = -A^{-1} (b + c) implied by f(x) = Ax + b + c
    """
    assert a.ndim == 2 and c.ndim == 2 and b.ndim == 1
    assert c.shape[-1] == a.shape[0] == a.shape[1]
    if maxeig(a) >= 0:
        inv_a = onp.linalg.pinv(a)
    else:
        inv_a = onp.linalg.inv(a)
    mu = - onp.einsum("oi,ei->eo", inv_a, b + c)
    if de:
        mu -= - onp.einsum("oi,i->o", inv_a, b)
    return mu


def tikhonov_pinv(mat, tikhonov):
    """
    Computes M^T.(M.M^T + tikh * I)^(-1)

    Args:
        mat: [d, k]
        tikhonov: regularization strength
    Returns:
        pinv: [k, d]
    """
    d, k = mat.shape
    assert d >= k, f"tikhonov_pinv expects [d, k] shape (as in theorem). Got: {mat.shape}"
    reg = tikhonov * onp.eye(d)
    pinv_mat = mat.T @ onp.linalg.inv(mat @ mat.T + reg)
    return pinv_mat


def linear_causal_sde_mat(*, f0, finf, **causal_mat_kwargs):
    """
    Returns A from f(x) = Ax + b + c
    """
    mask = ~onp.isnan(finf).all(1)
    assert onp.allclose(mask, ~onp.isnan(f0).all(1))
    f0 = f0[mask].T
    finf = finf[mask].T
    assert f0.shape == finf.shape
    d = finf.shape[0]

    if "tikhonov" in causal_mat_kwargs and causal_mat_kwargs["tikhonov"] > 0:
        # Tikhonov regularization of least squares problem
        # finf: [d, k] here
        pinv_finf = tikhonov_pinv(finf, causal_mat_kwargs["tikhonov"])

    elif "tsvd" in causal_mat_kwargs and 0 < causal_mat_kwargs["tsvd"] < 1:
        # make lower rank by keeping only top `tsvd` percent of singular values
        tsvd = math.floor(d * causal_mat_kwargs["tsvd"])
        tsvd = min(tsvd, finf.shape[0], finf.shape[1])
        u, s, vh = onp.linalg.svd(finf)
        u = u[:, :tsvd]
        s = s[:tsvd]
        vh = vh[:tsvd, :]
        pinv_finf = vh.T @ onp.diag(1 / s) @ u.T

    else:
        pinv_finf = onp.linalg.pinv(finf)

    a_sde = - f0 @ pinv_finf
    assert not onp.isnan(a_sde).any()
    return a_sde


def linear_causal_sde_bias(*, f0, finf, finf_ctrl, **causal_mat_kwargs):
    """
    Returns b from f(x) = Ax + b + c
    """
    assert finf_ctrl.ndim == 1
    a_sde = linear_causal_sde_mat(f0=f0, finf=finf, **causal_mat_kwargs)
    assert not onp.isnan(a_sde).any()
    bias = - a_sde @ finf_ctrl
    return bias.T


def linear_causal_sde_shifts(*, f0, finf, finf_ctrl, **causal_mat_kwargs):
    """
    Returns c from f(x) = Ax + b + c
    """
    assert finf_ctrl.ndim == 1
    a_sde = linear_causal_sde_mat(f0=f0, finf=finf, **causal_mat_kwargs)
    assert not onp.isnan(a_sde).any()
    mask = ~onp.isnan(finf).all(1)
    finf = finf[mask].T
    shifts = - a_sde @ finf + a_sde @ finf_ctrl[:, None]
    return shifts.T


def linear_causal_params(*, f0, finf, finf_ctrl, **causal_mat_kwargs):
    """
    Returns:
        a: [d, d]
        b: [d,]
        c: [n_envs, d]
        f0: [n_envs, d]
        finf: [n_envs, d]
        finf_ctrl: [d,]
    """
    params = dict(
        a=linear_causal_sde_mat(f0=f0, finf=finf, **causal_mat_kwargs),
        b=linear_causal_sde_bias(f0=f0, finf=finf, finf_ctrl=finf_ctrl, **causal_mat_kwargs),
        c=linear_causal_sde_shifts(f0=f0, finf=finf, finf_ctrl=finf_ctrl, **causal_mat_kwargs),
        f0=f0,
        finf=finf,
        finf_ctrl=finf_ctrl,
        causal_mat_kwargs=causal_mat_kwargs,
    )
    params["a_causal"] = params["a"].T
    return params


def transform_initial_drift(*, f, shift, scale):
    return f(-shift / scale) / scale


def transform_fixed_point(*, x, shift, scale):
    return shift + scale * x


def causal_mats(
        model,
        mask_train,
        param,
        *,
        shift,
        scale,
        seed=0,
        anneal_sparsify=1,
        causal_mat_kwargs=None,
):
    """
    Infers causal matrix via CLIPS by inducing single-gene perturbations for all genes that were perturbed in training
    (either as a single or as part of a double)

    """
    if causal_mat_kwargs is None:
        causal_mat_kwargs = dict()

    # assemble sorted mask of control and the single-gene perturbations for all genes perturbed in training
    # i.e. all settings for which we learned a perturbation embedding
    # [envs, d]
    train_genes = jnp.array(sorted(list(set(intv for intvs in mask_train for intv in onp.where(intvs)[0]))))
    env_mask = jnp.concatenate([
        jnp.zeros((1, model.d,)),
        jnp.eye(model.d)[train_genes],
    ])
    zero_point = onp.zeros((model.d,))

    """
    Compute f(infinity)
    """
    simulated_anneal_indices = onp.arange(len(model.anneal_scales))[::anneal_sparsify]
    anneal_time = int(simulated_anneal_indices[-1])

    # [envs, 1, d]
    key, subk = random.split(random.PRNGKey(seed))
    finf_samples, _ = vmap(
        functools.partial(model._simulate_dynamical_system,
                          probability_flow=True,
                          n_samples=1,
                          x_init=zero_point,
                          simulate_anneal_scale_indices=simulated_anneal_indices,
                          return_traj=False),
        in_axes=(None, 0, None, 0),
        out_axes=0,
    )(param, random.split(subk, len(env_mask)), mask_train, env_mask)

    # [envs, d]
    finf = onp.array(finf_samples.mean(-2))

    finf_ctrl = finf[0]
    finf = finf[1:]
    env_mask = env_mask[1:]

    """
    Compute f(0) for each anneal time
    """
    # drift function vmapped over perturbations
    # [d,], [..., d], [] -> [..., d]
    def f(x, *, query, time):
        return vmap(lambda mask_query: model.drift(
            x=x,
            param=param,
            diffusion_time=jnp.array(time),
            noise_scale=jnp.array(model.config.sde_kwargs["anneal_scales"][time]),
            mask_train=mask_train,
            mask_query=mask_query,
        ))(query)

    f0_sig = dict()
    f0_sig[anneal_time] = f(zero_point, time=anneal_time, query=env_mask)

    """
    Compute causal SDE matrices in transformed spaces
    """
    params = dict()
    for anneal_time, f0 in f0_sig.items():
        params[anneal_time] = linear_causal_params(
            f0=f0,
            finf=finf,
            finf_ctrl=finf_ctrl,
            **causal_mat_kwargs,
        )

    tfm_params = dict()
    for anneal_time, f0 in f0_sig.items():
        f_sig = functools.partial(f, time=anneal_time, query=env_mask)
        tfm_f0 = transform_initial_drift(f=f_sig, shift=shift, scale=scale)
        tfm_finf = transform_fixed_point(x=finf, shift=shift, scale=scale)
        tfm_finf_ctrl = transform_fixed_point(x=finf_ctrl, shift=shift, scale=scale)
        tfm_params[anneal_time] = linear_causal_params(
            f0=tfm_f0,
            finf=tfm_finf,
            finf_ctrl=tfm_finf_ctrl,
            **causal_mat_kwargs,
        )

    for anneal_time in params.keys():
        params[anneal_time]["env_mask"] = env_mask
        tfm_params[anneal_time]["env_mask"] = env_mask

    return params, tfm_params


def dropna(arr):
    return arr[~onp.isnan(arr)].flatten()


def empirical_differential_expressions(
        data,
        mask,
        *,
        ctrl,
        single_gene_only=False,
):
    data_means = onp.array([onp.mean(x, axis=-2) for x in data])
    assert data_means.shape == mask.shape, f"{data_means.shape} != {mask.shape} Check joblib cache"

    # assemble sorted mask of all single gene perturbations we are looking for
    train_genes = jnp.array(sorted(list(set(intv for intvs in mask for intv in onp.where(intvs)[0]))))
    env_mask = jnp.eye(mask.shape[-1])[train_genes]
    assert jnp.allclose(env_mask.sum(1), 1)

    # for each perturbed gene, compute the differential expression observed in the dataset
    # if single_gene_only is True, skip gene if it was never perturbed alone
    # if False, use the average of all means of perturbations where the gene was perturbed
    d = data_means.shape[-1]
    des = []
    for intv in env_mask:
        means = []
        perturbed_gene = onp.where(intv)[0]
        assert perturbed_gene.size == 1
        if single_gene_only:
            train_intv_matches = onp.where(onp.isclose(mask, intv).all(1))[0]
        else:
            train_intv_matches = onp.where(mask[:, perturbed_gene])[0]

        for env_idx in train_intv_matches:
            means.append(data_means[env_idx])

        if single_gene_only:
            assert len(means) <= 1

        if means:
            des.append(onp.mean(means, axis=0))
        else:
            des.append(onp.nan * onp.zeros((d,)))

    des = onp.array(des)

    # subtract control mean to obtain differential expression
    assert ctrl.ndim == 2 and ctrl.shape[1] == d
    des -= onp.mean(ctrl, 0)

    return des


CAUSAL_EPS = 0.5
CLASSES = (-1, 0, 1)


def _drop_diag(mat):
    return mat[~onp.eye(mat.shape[0], dtype=bool)].flatten()


def thres_sgn(mat, eps=CAUSAL_EPS):
    return onp.sign(onp.where(onp.abs(mat) < eps, 0, mat)).flatten().astype(int)


def auroc_ovo(*, true, pred, eps=CAUSAL_EPS, drop_diag=True):
    if drop_diag:
        assert true.shape == pred.shape and true.shape[0] == true.shape[1]
        true, pred = _drop_diag(true), _drop_diag(pred)

    true, pred = true.flatten(), pred.flatten()
    true = thres_sgn(true, eps=eps)
    if onp.allclose(true, true[0]):
        # if all classes are the same, AUROC not defined
        return None

    true_classes = onp.unique(true)
    assert onp.array_equal(onp.union1d(true_classes, CLASSES), CLASSES)

    # one-vs-one AUROC multiclass extension insensitive to class imbalance
    # https://link.springer.com/article/10.1023/A:1010920819831
    # see also: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.htm
    ave_ovo_auroc = []

    for class_pair in [
        (-1, 0),
        (-1, 1),
        ( 0, 1),
    ]:
        if onp.isin(class_pair, true_classes).all():
            sel = onp.isin(true, class_pair)

            # assumes higher pred is higher score for class_pair[1]
            ave_ovo_auroc.append(skmetrics.roc_auc_score(
                onp.isclose(true[sel], class_pair[1]),
                pred[sel],
            ))

    assert ave_ovo_auroc
    return onp.mean(ave_ovo_auroc)


def f1_topk(*, true, pred, k=None, eps=CAUSAL_EPS, drop_diag=True):
    if drop_diag:
        assert true.shape == pred.shape and true.shape[0] == true.shape[1]
        true, pred = _drop_diag(true), _drop_diag(pred)

    true, pred = true.flatten(), pred.flatten()
    if k is None:
        k = true.size
    topk = onp.argsort(onp.abs(pred))[-k:]
    return skmetrics.f1_score(
        thres_sgn(true[topk], eps=eps),
        thres_sgn(pred[topk], eps=eps),
        average="macro",
    )


def pearson_topk(*, true, pred, k=None, drop_diag=True):
    if drop_diag:
        assert true.shape == pred.shape and true.shape[0] == true.shape[1]
        true, pred = _drop_diag(true), _drop_diag(pred)

    true, pred = true.flatten(), pred.flatten()
    if k is None:
        k = pred.size
    topk = onp.argsort(onp.abs(pred))[-k:]
    return onp.corrcoef(true[topk], pred[topk])[0, 1]


def compute_causal_mat_metrics(*, true, pred):
    return {
        f"auroc": auroc_ovo(true=true, pred=pred),
        f"pearson": pearson_topk(true=true, pred=pred),
        f"f1": f1_topk(true=true, pred=pred),
    }
