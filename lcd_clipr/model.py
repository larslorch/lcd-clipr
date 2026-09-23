from abc import ABC
import math

from jax import random, numpy as jnp
import numpy as onp

from lcd_clipr.sde import SDE
from lcd_clipr.utils.gi import itup
from lcd_clipr.core import normalizer


class LCDCLIPR(SDE, ABC):
    """
    Class wrapping around SDE that handles the noise model (including fitting and sampling)

    Args:
        sde_kwargs: hyperparameters for SDE simulation
    """
    def __init__(
        self,
        noise_model=None,
        noise_model_kwargs=None,
        scale_standardization=False,
        sde_kwargs=None,
    ):
        noise_model_kwargs = noise_model_kwargs or {}
        sde_kwargs = sde_kwargs or {}

        super().__init__(**sde_kwargs)

        self.noise_model = noise_model
        self.noise_model_kwargs = noise_model_kwargs
        self.scale_standardization = scale_standardization
        self.noise_param = None
        self.standardize = None
        self.invert_standardize = None


    def fit_noise_model(self, datasets, sharding=None, eval_mode=False, mem=None):
        """
        Fit noise model

        Args:
            datasets: Dataset
            sharding
            eval_mode: whether to fit the noise model in evaluation mode
            mem

        Returns:
            Dataset object the same shape as the input dataset, but with the data replaced by the denoised data
        """
        assert datasets.data_noisy is not None and datasets.data is None

        if self.noise_model is None:
            # data is expected to be a numpy array of ndim=3: [envs, n, d] since the samples n match
            # for simulated data, this list stacking works
            datasets = datasets._replace(data=onp.array(datasets.data_noisy))
            logs = {}

        elif self.noise_model == "generative":
            if self.is_nonnegative:
                raise NotImplementedError

            from lcd_clipr.noise_models.generative_prior import fit_noise_model as _fit_noise_model_generative
            if mem is None:
                fit_noise_model_generative = _fit_noise_model_generative
            else:
                fit_noise_model_generative = mem.cache(
                    _fit_noise_model_generative,
                    ignore=("sharding", "t_init", "cluster_t_max"),
                )

            datasets, self.noise_param, logs = fit_noise_model_generative(
                self.config.seed, datasets, sharding=sharding, eval_mode=eval_mode, **self.noise_model_kwargs,
            )

        else:
            raise NotImplementedError(f"noise model {self.noise_model} not implemented")

        # apply elementwise scaling if specified
        datasets, self.standardize, self.invert_standardize = self.compute_elementwise_scaling(datasets)

        assert datasets.data is not None
        assert len(datasets.data) == len(datasets.data_noisy)
        assert isinstance(logs, dict)

        return datasets, logs


    def compute_elementwise_scaling(self, datasets):

        numpy = onp if type(datasets.data[0]) == onp.ndarray else jnp

        ctrl_mean = datasets.data[0].mean(0)
        ctrl_std = datasets.data[0].std(0)

        all_mean = numpy.concatenate(datasets.data).mean(0)
        all_std = numpy.concatenate(datasets.data).std(0)

        if self.scale_standardization == "observ":
            if self.is_nonnegative:
                standardize = lambda x: x / ctrl_mean
                inverse = lambda x: x * ctrl_mean
            else:
                standardize = lambda x: (x - ctrl_mean) / ctrl_std
                inverse = lambda x: x * ctrl_std + ctrl_mean

        elif self.scale_standardization == "all-std":
            if self.is_nonnegative:
                standardize = lambda x: x / ctrl_mean
                inverse = lambda x: x * ctrl_mean
            else:
                standardize = lambda x: (x - ctrl_mean) / all_std
                inverse = lambda x: x * all_std + ctrl_mean

        elif self.scale_standardization == "all":
            if self.is_nonnegative:
                standardize = lambda x: x / all_mean
                inverse = lambda x: x * all_mean
            else:
                standardize = lambda x: (x - all_mean) / all_std
                inverse = lambda x: x * all_std + all_mean

        elif self.scale_standardization == "none":
            standardize = inverse = lambda x: x

        else:
            raise NotImplementedError(f"scale_standardization {self.scale_standardization} not implemented")

        standardization_aux = dict(
            ctrl_mean=ctrl_mean,
            ctrl_std=ctrl_std,
            all_mean=all_mean,
            all_std=all_std,
        )
        standardize.aux = inverse.aux = standardization_aux

        assert not isinstance(datasets.data, list), (
            "datasets.data should be a numpy array with dim=3. "
            "If this is a list of numpy arrays instead, it could be that "
            "you are trying to standardize the count data (not the denoised data by the generative prior) "
            "and forgot to set a noise_model != `None` in the training config. "
            "Only the denoised data should be standardized."
        )

        return datasets._replace(data=standardize(datasets.data)), standardize, inverse


    def decode(self, key, s, **kwargs):
        """
        Sample data `x` from noise model given state `s`

        Args:
            key: PRNGKey
            s: ndarray of shape [..., n, d]

        Returns:
            ndarray of shape [..., n, d] representing the decoded noisy counts under the fitted noise model
        """
        assert self.noise_param is not None or self.noise_model is None
        if not s.size:
            return s

        # undo elementwise standardization
        s = self.invert_standardize(s)

        # decode noise model
        if self.noise_model is None:
            x = s

        elif self.noise_model == "generative":
            from lcd_clipr.noise_models.generative_prior import decode as decode_generative
            x = decode_generative(key, s, self.noise_param, **kwargs)

        else:
            raise NotImplementedError(f"noise model {self.noise_model} not implemented")

        x = onp.where(onp.isnan(s), onp.nan, x.astype(onp.float32)) # casts x.dtype from int to float since nan is float
        return x


    def predict(self, *, param, key, mask_train, mask_query, n_samples, n_samples_diffuse=None, return_mean=False, reuse=None):
        """
        Args:
            key: PRNGKey
            param: parameters of the SDE drift and diffusion functions
            mask_train: binary indicator of shape [m, d] indicating which variables are intervened on in train environments
            mask_query: binary indicator of shape [q, d] indicating which variables are intervened on in each environment
            n_samples: number of samples to generate
            n_samples_diffuse: number of samples to generate for the diffusion process (if None, defaults to `n_samples`)
            return_mean: whether to return the mean of the samples (to save memory)
            reuse (optional): tuple of (pred_dict, interventions) where pred_dict contains 'state', 'counts', 'norm'
                              arrays with leading dimension matching interventions to reuse from previous calls

        Returns:
            dict with keys 'log', 'state', 'counts', 'norm' representing prediction results
        """
        if n_samples_diffuse is None or n_samples_diffuse == -1 or n_samples_diffuse > n_samples:
            n_samples_diffuse = n_samples

        # check if reuse is provided and if the perturbations match
        intv_requested = None
        if reuse is not None:
            reuse_pred_dict, reuse_mask_query = reuse
            requested = list(map(itup, mask_query))
            reusable = set(map(itup, reuse_mask_query))
            needed = onp.array([intv not in reusable for intv in requested])
            intv_requested = mask_query
            mask_query = mask_query[needed]

        # generate samples from SDE
        key, subk = random.split(key)
        s, log = self.sample_dynamical_system(
            param,
            subk,
            mask_train,
            mask_query,
            n_samples=n_samples_diffuse,
            return_log=True,
        )

        # decode samples from SDE under noise model
        # (if n_samples_diffuse < n_samples, decode multiple samples per diffusion sample by repeating samples)
        if n_samples_diffuse == n_samples:
            s_decoded = s
        else:
            decoded_per_diffusion = math.ceil(n_samples / n_samples_diffuse)
            s_decoded = onp.repeat(s, decoded_per_diffusion, axis=-2)
            s_decoded = s_decoded[..., :n_samples, :]

        key, subk = random.split(key)
        x = self.decode(subk, s_decoded, mask_query=mask_query)
        if self.noise_model is None:
            x_norm = None
        else:
            x_norm = normalizer(x)

        pred = dict(
            state=s,
            counts=x,
            norm=x_norm,
        )

        # assemble output based on reuse
        if reuse is not None:
            reuse_pred_dict, reuse_mask_query = reuse
            reusable = {itup(intv): idx for idx, intv in enumerate(reuse_mask_query)}
            simulated = {itup(intv): idx for idx, intv in enumerate(mask_query)}
            result = {}
            for k in pred.keys():
                reused_values = []
                for intv in list(map(itup, intv_requested)):
                    assert intv in reusable or intv in simulated
                    if intv in reusable:
                        reused_values.append(reuse_pred_dict[k][reusable[intv]])
                    else:
                        reused_values.append(pred[k][simulated[intv]])
                result[k] = onp.array(reused_values)

            return result

        if return_mean:
            for k in pred.keys():
                if pred[k] is not None:
                    pred[k] = pred[k].mean(axis=-2, keepdims=True)

        pred["log"] = log
        return pred


