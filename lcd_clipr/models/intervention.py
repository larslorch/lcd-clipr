from typing import Any
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as onp


def reduce_given_mask(
        param,
        mask_train,
        mask_query,
        *,
        agg,
        axis=0,
        np="jnp"
    ):

    numpy = onp if np == "onp" else jnp

    assert mask_train.ndim == 2
    assert mask_query.ndim == 1
    assert mask_train.shape[-1] == mask_query.shape[-1]
    for par in jax.tree.leaves(param):
        assert par.shape[axis] == mask_train.shape[-1], \
            (f"Leading dim of param ({par.shape}) must match number of "
             f"perturbations ({mask_train.shape[-1]}) to allow reduction.")

    param, mask_train, mask_query = jax.tree.map(numpy.array, (param, mask_train, mask_query))

    # swap parameter axes so that reduction can be done over axis 0
    param = jax.tree.map(partial(numpy.swapaxes, axis1=axis, axis2=0), param)

    mask_train_any = mask_train.any(0).astype(float)
    def _nan_unseen(p):
        return numpy.where(numpy.expand_dims(mask_train_any, list(range(1, param.ndim))), p, numpy.nan)

    # nan all parameters not seen during training
    # [n_envs, ...] -> [n_envs, ...]
    param_trained = jax.tree.map(_nan_unseen, param)

    # compute average perturbation effect across all seen perturbations
    # [n_envs, ...] -> [...]
    ave_param = jax.tree.map(partial(numpy.nanmean, axis=0), param_trained)

    # impute average effect of all learned perturbations for unseen perturbation indices
    def _impute(indiv, shared):
        return numpy.where(numpy.isnan(indiv), shared[None], indiv)

    param_imputed = jax.tree.map(_impute, param_trained, ave_param)

    # if no perturbations were seen, replace nans by zero
    seen_perturbs = mask_train.any()
    def _observational(p):
        return numpy.where(seen_perturbs, p, numpy.nan_to_num(p))
    param_imputed = jax.tree.map(_observational, param_imputed)

    # for the query, aggregate the effects of all perturbation indices
    if agg == "mean":
        n_intv = mask_query.sum(-1, keepdims=True)
        n_intv = numpy.where(numpy.isclose(n_intv, 0), 1, n_intv)
        mask_query /= n_intv
    elif agg == "sum":
        pass
    else:
        raise NotImplementedError()

    def _aggregate(p):
        mask_query_broadcast = numpy.expand_dims(mask_query, list(range(1, param.ndim)))
        return numpy.sum(mask_query_broadcast * p, axis=0)

    return jax.tree.map(_aggregate, param_imputed)


class Embedding__interv__(nn.Module):

    n_envs: int
    dim: int
    kwargs: Any

    def setup(self):
        self.embed = self.param('embed', nn.initializers.normal(), (self.n_envs, self.kwargs["embed_dim"]), jnp.float32)

    @nn.compact
    def __call__(self, z, mask_train, mask_query):
        final_embed = reduce_given_mask(self.embed, mask_train, mask_query, agg=self.kwargs["agg"])
        return final_embed

