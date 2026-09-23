from lcd_clipr.models.intervention import *


def time_embedding(pos, embedding_dim, max_positions=10000):
    half_dim = embedding_dim // 2
    emb = jnp.log(max_positions) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = pos[..., None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    if embedding_dim % 2 == 1:  # zero pad last dimension
        emb = jnp.pad(emb, [[0, 1] if l == emb.ndim - 1 else [0, 0] for l in range(emb.ndim)])
    if pos.ndim == 0:
        emb = emb.squeeze(0)
    assert emb.shape == pos.shape + (embedding_dim,)
    return emb


class MLP(nn.Module):

    n_envs: int
    d: int
    config: Any

    def setup(self):
        self.nonlin = dict(
            tanh=nn.tanh,
            sigmoid=nn.sigmoid,
            sin=jnp.sin,
            relu=nn.relu,
            silu=nn.silu,
            mish= lambda arr: arr * jnp.tanh(jnp.logaddexp(arr, 0)),
            rbf=lambda arr: jnp.exp(- jnp.power(arr, 2)),
        )[self.config.mlp_activation]
        
        self.interv_kwargs = self.n_envs, self.config.mlp_hidden, self.config.interv["mlp"]

    @nn.compact
    def __call__(self, x, diffusion_time, noise_scale, mask_train, mask_query):
        z = x
        intv_mode = self.config.interv["mlp"].get("mode")

        # check legacy configs
        time_embedding_mode = self.config.dsm.get("time_embedding", "sin")
        if time_embedding_mode != "sin":
            raise NotImplementedError(f"Unknown time embedding {time_embedding_mode}")

        if intv_mode in [
            "inject_linear",
            "mult_direct_linear",
        ]:
            embed = Embedding__interv__(*self.interv_kwargs)(z, mask_train, mask_query)

        else:
            raise NotImplementedError(f"Unknown intervention mode {intv_mode}")


        for l in range(self.config.mlp_layers + 1):
            fan_out = self.config.mlp_hidden if l < self.config.mlp_layers else self.d
            f = nn.Dense(fan_out)(z)
            # f = nn.Dense(fan_out, kernel_init=nn.linear.default_kernel_init if l < self.config.mlp_layers else
            #                                   nn.initializers.zeros_init())(z)

            # timestamp embedding
            if l == 0 and self.config.dsm["model"] == "cond":

                t_embed = time_embedding(diffusion_time,
                                         self.config.mlp_hidden,
                                         self.config.noise_annealing["sigma_steps"])
                t_embed = nn.Dense(self.config.mlp_hidden)(t_embed)
                t_embed = self.nonlin(t_embed)
                t_embed = nn.Dense(self.config.mlp_hidden)(t_embed)
                f += t_embed

            # intervention
            if (self.config.interv["mlp"]["all_layers"] and 0 <= l < self.config.mlp_layers) or \
               (not self.config.interv["mlp"]["all_layers"] and l == 1):

                # inject
                if intv_mode == "inject_linear":
                    a = nn.Dense(self.config.mlp_hidden)(embed)
                    f += a

                # mult
                elif intv_mode == "mult_direct_linear":
                    b = nn.Dense(self.config.mlp_hidden)(embed)
                    f *= b

            if l >= self.config.mlp_layers - 1:
                z = nn.tanh(f)
            else:
                z = self.nonlin(f)


        if self.config.dsm["model"] == "uncond":
            return f - x
        elif self.config.dsm["model"] == "cond":
            if self.config.dsm["param"] == "song":
                # Song et al 2020, Score-based generative modeling through stochastic differential equations
                return x - noise_scale * f

            elif self.config.dsm["param"] == "karras":
                # Karras et al 2022, Elucidating the Design Space of Diffusion-Based Generative Models
                return x / (1 + noise_scale ** 2) + (noise_scale * f) / jnp.sqrt(1 + noise_scale ** 2)

            elif self.config.dsm["param"] == "ours":
                return x / (1 + noise_scale ** 2) + (noise_scale * noise_scale * f) / jnp.sqrt(1 + noise_scale ** 2)

            elif self.config.dsm["param"] == "ours_2":
                return x / jnp.sqrt(1 + noise_scale ** 2) + (noise_scale * noise_scale * f) / (1 + noise_scale ** 2)

            else:
                raise NotImplementedError(f"Unknown score function parametrization {self.config.dsm['param']}")

        elif self.config.dsm["model"] == "simple":
            return f
        else:
            raise NotImplementedError(f"Unknown score function model {self.config.dsm['model']}")

