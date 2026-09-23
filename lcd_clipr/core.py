from typing import NamedTuple, Any
import numpy as onp
import jax.numpy as jnp


class Dataset(NamedTuple):
    """
    Args:
        data_noisy: Python list of length `m` of (possibly unequally shaped) data matrices [n_i, d]
            that correspond to the different data environments.
            By convention, the first dataset in the list is *unperturbed/observational* data.
        data_normalized: Same data type as `data_noisy`. Contains `data_noisy` with log1p and library size normalization.
        data: Same data type as `data_noisy`. Contains inferred noise-free data.
        data_true: Same data type as `data_noisy`. Contains ground-truth noise-free data (if available)
        intv: Binary indicator matrix of shape [m, d] encoding which variables were intervened upon in each environment
        differential_expression_mask: Binary indicator matrix of shape [m, d] encoding which variables were marker
            genes in each environment
        unit_scaling: Vector of shape [d] fixing the unit scales of the genes (if available)
    """
    data_noisy: Any = None
    data_normalized: Any = None
    data: Any = None
    data_true: Any = None
    intv: Any = None
    differential_expression_mask: Any = None
    unit_scaling: Any = None

    @property
    def size(self):
        sizes = [
            len(prop) for prop in [
                self.data_noisy,
                self.data_normalized,
                self.data,
                self.data_true,
                self.intv,
                self.differential_expression_mask,
            ]
            if prop is not None
        ]
        if not sizes:
            return 0
        else:
            assert all(s == sizes[0] for s in sizes)
            return sizes[0]


# fields that are shared across environments
_SHARED_DATASET_FIELDS = frozenset({"unit_scaling"})


def iter_dataset(datasets):
    """
    Yields `Dataset` objects with single leaves of a multi-env `Dataset` object
    """
    n_envs = datasets.size
    dct = datasets._asdict()
    for i in range(n_envs):
        yield Dataset(**{
            k: v if k in _SHARED_DATASET_FIELDS else (v[i] if v is not None else None)
            for k, v in dct.items()
        })


def index_dataset(datasets, idx):
    """
    Indexes into first axis of a multi-env `Dataset` object
    """
    dct = datasets._asdict()
    new_dct = {}
    for k, v in dct.items():
        if k in _SHARED_DATASET_FIELDS:
            new_dct[k] = v
        elif type(v) == list:
            new_dct[k] = [v[ii] for ii in idx]
        elif v is None:
            new_dct[k] = v
        elif isinstance(v, (float, int)):
            new_dct[k] = v
        else:
            new_dct[k] = v[idx]

    return Dataset(**new_dct)


def normalizer(xs, *, ref_libsize=1e4):
    def _normalize(x):
        numpy = onp if type(x) == onp.ndarray else jnp
        libsize = x.sum(-1, keepdims=True)
        size_factors = libsize / ref_libsize
        x_adjusted = x / numpy.where(numpy.isclose(size_factors, 0.0), 1.0, size_factors)
        return numpy.log1p(x_adjusted)

    if isinstance(xs, list):
        return [_normalize(x) for x in xs]
    elif isinstance(xs, tuple):
        return tuple(_normalize(x) for x in xs)
    else:
        return _normalize(xs)
