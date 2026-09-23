import jax.numpy as jnp


def to_diag(x):
    """
    Args:
        x: [..., d]

    Returns:
        [..., d, d] where last two dimensions are jnp.diag() of last dimension in x
    """
    assert x.ndim >= 1
    return jnp.einsum("...i,...ij->...ij", x, jnp.eye(x.shape[-1]))

