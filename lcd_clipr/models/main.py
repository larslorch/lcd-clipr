from jax import numpy as jnp

from lcd_clipr.utils.sde import to_diag

from lcd_clipr.model import LCDCLIPR
from lcd_clipr.models.mlp import MLP


class Model(LCDCLIPR):

    def __init__(
        self,
        config,
        n_envs,
        d,
    ):
        self.config = config
        self.n_envs = n_envs
        self.d = d

        unknown_components = set(config.components) - {"mlp", "interv"}
        if unknown_components:
            raise NotImplementedError(f"Unknown model components {sorted(unknown_components)}; only `mlp` is implemented")

        # init SDE superclass
        sde_kwargs = config.get("sde_kwargs", {})
        assert not sde_kwargs.get("is_nonnegative", False)
        sde_kwargs["is_nonnegative"] = False

        self.n_wiener_procs = self.config.get("wiener_procs", d) or d
        sde_kwargs["wiener_procs"] = self.n_wiener_procs

        sde_kwargs["anneal_scales"] = jnp.sort(jnp.geomspace(
            self.config.noise_annealing["sigma_min"],
            self.config.noise_annealing["sigma_max"],
            self.config.noise_annealing["sigma_steps"],
            endpoint=True,
        ), descending=True)

        noise_model_kwargs = config.get("noise_model_kwargs")

        super().__init__(
            scale_standardization=self.config.scale_standardization,
            noise_model=self.config.noise_model,
            noise_model_kwargs=noise_model_kwargs,
            sde_kwargs=sde_kwargs
        )

        self.mlp = MLP(n_envs, d, self.config)


    def init_params(self, key):

        theta = dict()

        # dummy arg matches signature of __call__ of drift and diffusion functions
        dummy_arg = (
            jnp.zeros((self.d,)),
            jnp.array(0),
            jnp.array(0.0),
            jnp.zeros((self.n_envs, self.d)),
            jnp.zeros((self.d,)),
        )

        if "mlp" in self.config.components:
            theta["mlp"] = self.mlp.init(key, *dummy_arg)["params"]

        # switches for controlling functions and loss
        theta["__config__"] = {
            "active": {},
            "stopgrad": {},
        }
        for part, config in self.config.components.items():
            if "active" in config.keys():
                theta["__config__"]["active"][part] = jnp.array(1.0)
            if "stopgrad" in config.keys():
                theta["__config__"]["stopgrad"][part] = jnp.array(0.0)

        return theta

    """
    SDE drift and diffusion functions
    """

    def f(self, param, x, diffusion_time, noise_scale, mask_train, mask_query):
        active = param["__config__"]["active"]
        f = jnp.zeros(x.shape)

        if "mlp" in active:
            f += active["mlp"] * self.mlp.apply(dict(params=param["mlp"]), x, diffusion_time, noise_scale, mask_train, mask_query)

        assert x.shape == f.shape, (x.shape, f.shape)
        assert mask_query.shape == x.shape[-1:]
        return f


    def drift(self, *, param, x, diffusion_time, noise_scale, mask_train, mask_query):
        if self.config.dsm["model"] == "uncond":
            f = self.f(param, x, diffusion_time, noise_scale, mask_train, mask_query)
            drift = f / (1 + (noise_scale ** 2))
        elif self.config.dsm["model"] == "cond":
            f = self.f(param, x / jnp.sqrt(1 + noise_scale ** 2), diffusion_time, noise_scale, mask_train, mask_query)
            drift = (f - x) / (noise_scale ** 2)
        elif self.config.dsm["model"] == "simple":
            drift = self.f(param, x, diffusion_time, noise_scale, mask_train, mask_query)
        else:
            raise NotImplementedError(f"Unknown score function model {self.config.dsm['model']}")

        return drift


    def diffusion(self, *, param, x, diffusion_time, noise_scale, mask_train, mask_query):
        return to_diag(jnp.ones_like(x)) * jnp.sqrt(2.0)

