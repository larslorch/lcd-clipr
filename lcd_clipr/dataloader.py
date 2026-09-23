import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # hide gpus to tf to avoid OOM and cuda errors by conflicting with jax

import collections
import itertools

import numpy as onp
import jax
import jax.numpy as jnp


def prepare_tf_data(xs):
    """
    Converts tree of tf tensors to numpy arrays.
    Taken from
    https://github.com/google/flax/blob/6ae22681ef6f6c004140c3759e7175533bda55bd/examples/imagenet/train.py#L169
    """
    # # use _numpy() for zero-copy conversion between TF and NumPy.
    return jax.tree_util.tree_map(lambda x: x._numpy(), xs)


def prefetch_to_device(iterator, sharding, size=2):
    queue = collections.deque()

    def _prefetch(x):
        return jax.tree.map(jax.device_put, x, sharding)

    def enqueue(n):
        # enqueues *up to* `n` elements from the iterator
        for data in itertools.islice(iterator, n):
            queue.append(_prefetch(data))

    # fill up the buffer
    enqueue(size)
    while queue:
        yield queue.popleft()
        enqueue(1)


def make_dataloader(seed, datasets_arrays, batch_size, batch_size_split, sharding, datasets_intv=None):

    batch_size_step = batch_size // batch_size_split

    # create dataset for each environment
    env_datasets = []
    for j, data in enumerate(datasets_arrays):
        data = data.astype(jnp.float32)
        ds = tf.data.Dataset.from_tensor_slices(data)
        # first repeat, then batch, so that we can have batch size greater than the number of elements
        # see https://www.tensorflow.org/guide/data#batching_dataset_elements
        # shuffle after repeat mixies epoch boundaries together
        ds = ds.repeat()
        ds = ds.shuffle(buffer_size=128, reshuffle_each_iteration=True, seed=seed)
        ds = ds.batch(batch_size_step)

        # remember environment index and mask for each batch
        if datasets_intv is None:
            ds = ds.map(lambda x: dict(x=x, env=j))
        else:
            ds = ds.map(lambda x: dict(x=x, env=j, intv=datasets_intv[j]))

        ds = ds.batch(batch_size_split)
        env_datasets.append(ds)

    # sample random environments
    # .unbatch() ensures that the same envs is yielded `batch_size_split` times sequentially
    # in combination with .batch(batch_size_split) above
    final = tf.data.Dataset.sample_from_datasets(env_datasets, seed=seed)
    final = final.unbatch()
    final = final.prefetch(10)

    # convert to numpy and load to device
    # debug device placement with
    # `jax.debug.visualize_array_sharding(arr)`
    # `sharding.devices_indices_map(arr.shape)`

    final = map(prepare_tf_data, final)

    n = onp.prod(sharding.shape)
    assert batch_size_step % n == 0, "batch_size_step must be divisible by the number of shards"

    batch_sharding = dict(
        x=sharding.reshape(n, 1),  # shard `x` (axis=0) across devices
        env=sharding.replicate().reshape(tuple()),  # replicate env information across devices
    )
    if datasets_intv is not None:
        batch_sharding["intv"] = sharding.replicate()  # replicate env information across devices

    final = prefetch_to_device(final, batch_sharding)
    return final, batch_sharding
