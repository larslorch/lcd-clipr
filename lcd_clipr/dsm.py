from functools import partial

import jax
from jax import vmap, lax, random
import jax.numpy as jnp

from lcd_clipr.utils.partial import wrapped_mean


def make_dsm_loss_fun(drift, anneal_scales, config):
    """
    Args:
        drift: [..., d], {param tree}, {intv param tree}, [..., d] -> [..., d]
        anneal_scales
        config

    Returns:
        objective function: ({param tree}, {intv param tree}), Batch ->  []
    """
    @partial(wrapped_mean, axis=(0,)) # mean over flattened batch size dimension
    @partial(vmap, in_axes=(None, None, 0, dict(x=0, env=None, intv=None), None), out_axes=0)
    def dsm_loss(param, step, key, batch, mask_train):
        """
        Computes denoising score matching loss function for each x in batch
        Then takes mean over shape [batch_size, ...]
        """
        assert batch["x"].shape == batch["intv"].shape

        # sample noise level
        key, subk = random.split(key)
        time = random.choice(subk, jnp.arange(len(anneal_scales))).astype(jnp.int32)
        noise_scale = anneal_scales[time]
        x = batch["x"]

        # compute loss
        key, subk = random.split(key)
        x_tilde = x + noise_scale * random.normal(subk, x.shape)

        score_tilde = drift(
            param=param,
            x=x_tilde,
            diffusion_time=time,
            noise_scale=noise_scale,
            mask_train=mask_train,
            mask_query=batch["intv"],
        )

        loss = score_tilde + ((x_tilde - x) / (noise_scale ** 2))
        loss = 0.5 * (loss ** 2)

        if config.dsm["loss_weighting"] == 0.0:
            pass
        else:
            loss *= (noise_scale ** config.dsm["loss_weighting"])

        loss = loss.mean(-1)

        noise_scale_ctr = jnp.eye(len(anneal_scales))[time]
        loss_by_noise_scale = noise_scale_ctr * loss

        return loss, loss_by_noise_scale, noise_scale_ctr


    def objective_fun(param, step, key, batch, mask_train):
        """
        Computes mean DSM loss over environments
        """

        # handle parameter configurations for fine-tuning (e.g. stopgrads)
        is_interv_param = lambda path: '__interv__' in jax.tree_util.keystr(path)
        if "__config__" in param:
            param["__config__"] = jax.tree.map(lambda p: lax.stop_gradient(p), param["__config__"])
            for k, stop in param["__config__"]["stopgrad"].items():
                if k == "interv":
                    param = jax.tree_util.tree_map_with_path(
                        lambda path, p: (p * (1 - stop) + stop * lax.stop_gradient(p)) if is_interv_param(path) else p,
                        param,
                    )
                elif k in param:
                    param[k] = jax.tree.map(lambda p: p * (1 - stop) + stop * lax.stop_gradient(p), param[k])

        # compute mean DSM loss
        key, subk = random.split(key)
        batch_size = batch["x"].shape[0]
        subks = random.split(subk, batch_size)

        loss, _loss_by_noise_scale, _noise_scale_ctr = dsm_loss(param, step, subks, batch, mask_train)
        assert loss.ndim == 0
        assert _loss_by_noise_scale.shape == _noise_scale_ctr.shape == (len(anneal_scales),)

        # compute correct mean losses by _noise_scale_ctr (the mean over batch size factor cancels out)
        loss_by_noise_scale = jnp.where(_noise_scale_ctr, _loss_by_noise_scale / _noise_scale_ctr, jnp.nan)

        # return loss, aux info dict
        aux = dict(dsm_loss=loss)
        for i, _loss in enumerate(loss_by_noise_scale):
            aux[f"dsm_loss__{i}"] = _loss

        return loss, aux


    return jax.value_and_grad(objective_fun, 0, has_aux=True)


