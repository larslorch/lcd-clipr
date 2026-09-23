from flax import linen as nn
from jax import random, numpy as jnp


class GaussianGenerator(nn.Module):

    d: int

    @nn.compact
    def __call__(self, key, mc_samples, training: bool):

        key, subk = random.split(key)
        s = random.normal(subk, shape=(mc_samples, self.d))
        s += nn.Dense(self.d)(s)
        return s


class MLPGenerator(nn.Module):

    d: int
    hidden: int = 64
    hidden_wide: int = 256
    layers: int = 2
    layers_wide: int = 0
    dropout: float = 0.0
    nonlinearity: str = "silu"

    @nn.compact
    def __call__(self, key, mc_samples, training: bool):
        if self.nonlinearity == "tanh":
            g = nn.tanh
        elif self.nonlinearity == "relu":
            g = nn.relu
        elif self.nonlinearity == "silu":
            g = nn.silu
        elif self.nonlinearity == "sigmoid":
            g = nn.sigmoid
        elif self.nonlinearity == "rbf":
            g = lambda arr: jnp.exp(- jnp.power(arr, 2))
        else:
            raise KeyError(f"Unknown nonlinearity {self.nonlinearity}")

        key, subk = random.split(key)
        s = random.normal(subk, shape=(mc_samples, self.hidden))

        s = nn.Dense(self.hidden)(s)

        for _ in range(self.layers):
            z = nn.Dense(self.hidden)(s)
            z = g(z)
            z = nn.Dense(self.hidden)(z)
            s += z

        s = nn.Dense(self.d)(s)

        for _ in range(self.layers_wide):
            z = nn.Dense(self.hidden_wide)(s)
            z = g(z)
            z = nn.Dense(self.d)(z)
            s += z

        return s
