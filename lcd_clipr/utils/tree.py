import jax
import numpy as onp
from jax import numpy as jnp, random


def print_shapes(tree):
    jax.tree_util.tree_map_with_path(lambda p, l: print(p, l.shape, type(l)), tree)
    return


def tree_sum_reduce(tree):
    return jax.tree_util.tree_reduce(lambda acc, leaf: acc + jnp.sum(leaf), tree, initializer=0)


def tree_global_norm(tree, p=2.0, eps=0.0):
    sums = jax.tree_util.tree_reduce(lambda acc, leaf: acc + jnp.sum(jnp.abs(leaf) ** p), tree, initializer=0)
    return (sums + eps) ** (1.0 / p)


def tree_grad_clip(tree, norm=1.0):
    tree_nrm = tree_global_norm(tree)
    return jax.tree.map(lambda l: l / jnp.maximum(tree_nrm, norm), tree)


def tree_isnan(tree):
    return jax.tree_util.tree_reduce(lambda acc, leaf: acc | jnp.isnan(leaf).any(), tree, initializer=False)


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def tree_init_normal(rng_key, target, scale=1.0):
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree.map(
        lambda l, k: l + scale * jax.random.normal(k, l.shape, l.dtype),
        target,
        keys_tree,
    )


def iter_tree(tree):
    flat, treedef = jax.tree_util.tree_flatten(tree)
    for vals in zip(*flat):
        yield jax.tree_util.tree_unflatten(treedef, vals)


def tree_reshape_leading(tree, shape_prefix, overlap=0):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return jax.tree_util.tree_unflatten(treedef, [leaf.reshape(shape_prefix + leaf.shape[overlap:])
                                                  for leaf in leaves])


def tree_expand_leading_by(tree, n):
    """
    Converts tree with leading trees with additional `n` leading dimensions
    """
    return jax.tree.map(lambda leaf: jnp.expand_dims(leaf, axis=tuple(range(n))), tree)


def tree_zip_leading(tree_list):
    """
    Converts n trees without leading dimension into one tree with leading dim [n, ...]
    """
    return jax.tree.map(lambda *a: jnp.stack([*a]) if len(a) > 1 else tree_expand_leading_by(*a, 1), *tree_list)


def tree_unzip_leading(tree, n):
    """
    Converts tree with leading dim [n, ...] into n trees without the leading dimension
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return [jax.tree_util.tree_unflatten(treedef, [leaf[i] for leaf in leaves]) for i in range(n)]


def variance_initialization(key, shape, scale=1.0, mode='fan_in', distribution='uniform'):
    """
    Taken from dm-haiku initializers.py
    """
    # compute fans
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape
    else:
        raise ValueError(shape)

    # compute scale
    if mode == 'fan_in':
      scale /= max(1.0, fan_in)
    elif mode == 'fan_out':
      scale /= max(1.0, fan_out)
    else:
      scale /= max(1.0, (fan_in + fan_out) / 2.0)

    if distribution == 'truncated_normal':
      stddev = onp.sqrt(scale)
      # Adjust stddev for truncation.
      # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      distribution_stddev = onp.asarray(.87962566103423978)
      stddev = stddev / distribution_stddev
      return stddev * random.truncated_normal(key, -2., 2., shape)

    elif distribution == 'normal':
      stddev = onp.sqrt(scale)
      return stddev * random.normal(key, shape)

    elif distribution == 'uniform':
      limit = onp.sqrt(3.0 * scale)
      return random.uniform(key, shape, minval=-limit, maxval=limit)

    else:
        raise KeyError(f"Unknown distribution initialization: {distribution}")


def tree_variance_initialization(key, target, scale=1.0, mode='fan_in', distribution='uniform'):
    keys_tree = random_split_like_tree(key, target)
    return jax.tree.map(
        lambda l, k: l + variance_initialization(k, l.shape, scale=scale, mode=mode, distribution=distribution),
        target,
        keys_tree,
    )


def filter_nested_dict(d, keys, complement=False):
    """
    Recursively filters a nested dictionary by a list of keys.

    Parameters:
        d (dict): The nested dictionary to filter.
        keys (dict): A nested dictionary representing the structure of key paths.
        complement (bool): If True, retains only keys not in the list.

    Returns:
        dict: A new dictionary containing only the specified key paths.
    """
    if not isinstance(d, dict):
        return d

    filtered_dict = {}
    for key in d.keys():

        if not complement:
            if key in keys:
                sub_keys = keys[key]
                if isinstance(sub_keys, dict) and sub_keys:
                    filtered_dict[key] = filter_nested_dict(d[key], sub_keys, complement=complement)
                else:
                    filtered_dict[key] = d[key]
        else:
            if key in keys:
                sub_keys = keys[key]
                if isinstance(sub_keys, dict) and sub_keys:
                    filtered_dict[key] = filter_nested_dict(d[key], sub_keys, complement=complement)
                else:
                    pass
            else:
                filtered_dict[key] = d[key]

    return filtered_dict


def build_key_tree(keys):
    """
    Converts a list of key paths into a nested dictionary structure.

    Parameters:
        keys (list): List of tuples, where each tuple represents a key path.

    Returns:
        dict: A nested dictionary representing the structure of key paths.
    """
    tree = {}
    for key_path in keys:
        current_level = tree
        for key in key_path:
            current_level = current_level.setdefault(key, {})
    return tree


def filter_dict_by_keys(d, key_paths, complement=False):
    """
    Filters a nested dictionary by key paths of varying lengths.

    Parameters:
        d (dict): The nested dictionary to filter.
        key_paths (list): List of tuples, where each tuple represents a key path.
        complement (bool): If True, retains only keys not in the list.

    Returns:
        dict: A new dictionary containing only the specified key paths.
    """
    key_tree = build_key_tree(key_paths)
    return filter_nested_dict(d, key_tree, complement=complement)


def dict_get_nested(d, keys):
    """
    Get a value in a nested dictionary using a list or tuple of keys.

    Parameters:
        d: The nested dictionary.
        keys: A list or tuple of keys representing the path.

    Returns:
        The value at the path specified by the keys.
    """
    for key in keys:
        d = d[key]
    return d


def dict_set_nested(d, keys, value):
    """
    Set a value in a nested dictionary using a list or tuple of keys.

    Parameters:
        d: The nested dictionary.
        keys: A list or tuple of keys representing the path.
        value: The value to set at the path specified by the keys.
    """
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value
