import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # hide gpus to tf to avoid OOM and cuda errors by conflicting with jax

from lcd_clipr.synthetic_data.linear_sde import simulate as linear_sampler
from lcd_clipr.realworld_data.replogle import load as replogle_loader
from lcd_clipr.realworld_data.norman import load as norman_loader
from lcd_clipr.realworld_data.wessels import load as wessels_loader

def make_data(*, seed, config):

    if config["id"] == "linear":
        train_datasets, test_datasets, remaining_datasets, meta_data = linear_sampler(seed, config=config)

    elif config["id"] == "replogle":
        train_datasets, test_datasets, remaining_datasets, meta_data = replogle_loader(seed, config=config)

    elif config["id"] == "norman":
        train_datasets, test_datasets, remaining_datasets, meta_data = norman_loader(seed, config=config)

    elif config["id"] == "wessels":
        train_datasets, test_datasets, remaining_datasets, meta_data = wessels_loader(seed, config=config)

    else:
        raise KeyError(f"Invalid dataset id `{config['id']}`")


    return train_datasets, test_datasets, meta_data