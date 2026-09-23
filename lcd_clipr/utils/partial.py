import math
import numpy as onp
import jax
import jax.numpy as jnp
from jax import tree, vmap
from functools import partial


NOT_JAX_TYPES = [str,]


def device_put(pytree, device):
    # puts a pytree onto a device
    return tree.map(lambda arr: arr if type(arr) in NOT_JAX_TYPES else jax.device_put(arr, device), pytree)


def device_put_replicate(pytree, sharding):
    # replicates a pytree across a sharding
    return tree.map(lambda arr: jax.device_put(arr, sharding.replicate().reshape([1] * arr.ndim)), pytree)


def wrapped_mean(func, axis=None):
    def wrapped(*args):
        return tree.map(partial(jnp.mean, axis=axis), func(*args))
    return wrapped


def onp_dynamic_slice(arr, start_indices, slice_sizes):
    adj_start_indices = []
    for start, size, dim_size in zip(start_indices, slice_sizes, arr.shape):
        # ensure we don't overrun the array bounds as in jax.lax.dynamic_slice
        if start + size > dim_size:
            adjusted_start = max(0, dim_size - size)
        else:
            adjusted_start = start
        adj_start_indices.append(adjusted_start)

    # slice array
    slices = tuple(slice(start, start + size) for start, size in zip(adj_start_indices, slice_sizes))
    return arr[slices]


def onp_jax_scan(f, init, xs, length=None):
    """
    Equivalent to jax.lax.scan but aggregates result in a numpy array rather than jax array to avoid instantiating
    large buffers on the jax accelerator. Inner loop is jitted and runs with jax on the accelerator.
    """
    # flatten input tree
    if xs is None:
        assert length is not None
        n = length
    else:
        flats, treedef = jax.tree.flatten(xs)
        n = flats[0].shape[0]
        assert all([flat.shape[0] == n for flat in flats])
        assert length is None or n == length

    # scan
    y_treedef = None
    carry = init
    ys_flat = []
    for j in range(n):
        if xs is None:
            x = None
        else:
            flat = [elt[j] for elt in flats]
            x = jax.tree.unflatten(treedef, flat)

        carry, y = f(carry, x)

        y_flat, y_treedef = jax.tree.flatten(y)
        y_flat = tuple([onp.array(elt) for elt in y_flat])
        ys_flat.append(y_flat)

    # unflatten output tree
    assert y_treedef is not None
    ys = [onp.stack(y_flat) for y_flat in zip(*ys_flat)]
    ys = jax.tree.unflatten(y_treedef, ys)
    return carry, ys


def vmap_scan(func, in_axes=None, out_axes=None, to_numpy=False):
    """
    Equivalent to jax.vmap(func) but is computed iteratively using jax.lax.scan, for example, to avoid memory issues.
    Like jax.lax.map, but supports specified vmap axes.
    If `as_numpy` is True, the output is converted to numpy arrays before returning to avoid instantiating large buffers.
    The inner loop is still jitted and performed with JAX and thus run on the accelerator.
    """
    _scan = onp_jax_scan if to_numpy else jax.lax.scan
    _moveaxis = onp.moveaxis if to_numpy else jnp.moveaxis

    def wrapped(*args):
        _in_axes = in_axes or 0
        if isinstance(_in_axes, int):
            _in_axes = (_in_axes,) * len(args)
        assert len(_in_axes) == len(args)

        # split args into vmap and non-vmap and transpose vmap axes to position 0
        v_args = dict((j, jax.tree.map(lambda _arg: _moveaxis(_arg, ax, 0), arg))
                       for j, (ax, arg) in enumerate(zip(_in_axes, args)) if ax is not None)
        nonv_args = dict((j, arg) for j, (ax, arg) in enumerate(zip(_in_axes, args)) if ax is None)

        def scanned_func(_nonv_args, _sliced_v_args):
            # re-assemble args
            _args = tuple(_sliced_v_args[j] if j in v_args else _nonv_args[j] for j in range(len(args)))
            return _nonv_args, func(*_args)

        if not to_numpy:
            scanned_func = jax.checkpoint(scanned_func, prevent_cse=False)

        # scan over vmap axes
        stacked_result = _scan(scanned_func, nonv_args, v_args)[1]

        # transpose output vmap axis to requested position
        _out_axes = out_axes or 0
        stacked_result = jax.tree.map(lambda res: _moveaxis(res, 0, _out_axes), stacked_result)

        return stacked_result

    if not to_numpy:
        wrapped = jax.jit(wrapped)

    return wrapped


def vmap_chunk(func, in_axes=None, out_axes=None, chunk_size=32, donate_argnums=None, to_numpy=False):
    """
    Equivalent to jax.vmap(func) but computed iteratively using jax.lax.scan in chunks to avoid memory issues.
    If `as_numpy` is True, the output of func is converted to numpy arrays before returning to avoid
    instantiating large buffers.
    The inner loop is still jitted and performed with JAX and thus run on the accelerator.
    """
    _scan = onp_jax_scan if to_numpy else jax.lax.scan
    _moveaxis = onp.moveaxis if to_numpy else jnp.moveaxis
    _concatenate = onp.concatenate if to_numpy else jnp.concatenate

    def wrapped(*args):
        _in_axes = in_axes or 0
        if isinstance(_in_axes, int):
            _in_axes = (_in_axes,) * len(args)
        assert len(_in_axes) == len(args)

        vmapped_func = vmap(func, _in_axes, 0)

        # split args into vmap and non-vmap and transpose vmap axes to position 0
        v_args = dict((j, jax.tree.map(lambda _arg: _moveaxis(_arg, ax, 0), arg))
                      for j, (ax, arg) in enumerate(zip(_in_axes, args)) if ax is not None)
        nonv_args = dict((j, arg) for j, (ax, arg) in enumerate(zip(_in_axes, args)) if ax is None)

        # compute number of chunks
        v_args_num = jax.tree.flatten([
            jax.tree.map(lambda _arg: _arg.shape[ax], arg)
            for j, (ax, arg) in enumerate(zip(_in_axes, args)) if ax is not None
        ])[0]
        num_arg = v_args_num[0]
        _chunk_size = chunk_size or num_arg
        scan_length = math.ceil(num_arg / _chunk_size) if _chunk_size < num_arg else 1

        assert all([num_arg == num for num in v_args_num])

        if to_numpy:
            jit_vmapped_func = jax.jit(vmapped_func, donate_argnums=donate_argnums)

            # to avoid instantiating the full args in buffer, we need to first emulate jax.lax.dynamic_slice in numpy
            # and then apply vmapped_func to the slices rather than slicing inside the chunk_scanner as below in jax
            def chunk_scanner(_, _chunked_v_args):
                _args = tuple(_chunked_v_args[j] if j in v_args else nonv_args[j] for j in range(len(args)))
                return _, jit_vmapped_func(*_args)

            def _slice_assembler(_arg):
                sliced = onp.stack([
                    onp_dynamic_slice(_arg, (chunk_idx, *[0 for _ in range(_arg.ndim - 1)]),
                                      (min(_chunk_size, num_arg), *_arg.shape[1:]))
                    for chunk_idx in range(0, _arg.shape[0], _chunk_size)
                ], axis=0)
                assert sliced.shape[0] == scan_length
                return sliced

            chunked_v_args = dict((j, jax.tree.map(_slice_assembler, arg)) for j, arg in v_args.items())

            # scan over vmap axes
            # [scan_length, chunk_size, ...]
            res = _scan(chunk_scanner, init=0, xs=chunked_v_args, length=None)[1]

        else:
            @partial(jax.checkpoint, prevent_cse=False)
            def chunk_scanner(chunk_idx, _):
                def _slicer(_arg):
                    return jax.lax.dynamic_slice(_arg, (chunk_idx, *[0 for _ in range(_arg.ndim - 1)]),
                                                 (min(_chunk_size, num_arg), *_arg.shape[1:]))

                _chunked_v_args = dict((j, jax.tree.map(_slicer, arg)) for j, arg in v_args.items())
                _args = tuple(_chunked_v_args[j] if j in v_args else nonv_args[j] for j in range(len(args)))
                return chunk_idx + _chunk_size, vmapped_func(*_args)

                # scan over vmap axes
                # [scan_length, chunk_size, ...]

            res = _scan(chunk_scanner, init=0, xs=None, length=scan_length)[1]

        # [scan_length * chunk_size, ...]
        res = jax.tree.map(lambda result: _concatenate(result, axis=0), res)

        # adjustment for num_arg % chunk_size != 0
        # in this case: in last batch of lax.scan, only keep last `num_arg % chunk_size` elements
        # the other ones are repeated (see docs for jax.lax.dynamic_slice)
        if num_arg % _chunk_size != 0 and _chunk_size < num_arg:
            lo = scan_length * _chunk_size - _chunk_size
            hi = scan_length * _chunk_size - (num_arg % _chunk_size)
            res = jax.tree.map(lambda result: _concatenate([result[:lo], result[hi:]], axis=0), res)

        # transpose output vmap axis to requested position
        _out_axes = out_axes or 0
        res = jax.tree.map(lambda result: _moveaxis(result, 0, _out_axes), res)

        return res

    if not to_numpy:
        wrapped = jax.jit(wrapped)

    return wrapped
