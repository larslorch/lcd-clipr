import warnings
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import time
import itertools
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import math
import pprint
import json
import zipfile
import tarfile
import wandb
import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as onp
import pandas as pd
import torch

import yaml
import re

# fixes float parsing in yaml
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


from lcd_clipr.definitions import PROJECT_DIR, EXPERIMENT_CONFIG_DATA

types_list = [list]
types_dict = [dict, defaultdict, OrderedDict]
keys_special = dict(
    __list__=list,
    __tuple__=tuple,
    __dict__=dict,
)


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, jnp.ndarray):
            obj = onp.array(obj)
        if isinstance(obj, onp.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        return json.JSONEncoder.default(self, obj)


class NumpyJSONDecoder(json.JSONDecoder):
    def _postprocess(self, obj):
        if isinstance(obj, dict):
            return {k: self._postprocess(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            out = [self._postprocess(v) for v in obj]
            # accounts for the fact that lists of non-equal shaped arrays don't get nicely formatted as np.arrays
            if all([type(v) == onp.ndarray for v in out]) and not all([v.shape == out[0].shape for v in out]):
                return out
            elif all([type(v) == str for v in out]):
                return out
            else:
                return onp.array(out)
        else:
            return obj

    def decode(self, obj, recurse=False):
        decoded = json.JSONDecoder.decode(self, obj)
        return self._postprocess(decoded)



def option_to_descr(d):
    if type(d) in types_list:
        raise ValueError("No lists should occur in option_to_descr")
    elif type(d) in types_dict:
        l = []
        for k, v in d.items():
            s = option_to_descr(v)
            if s[0] == "=":
                l.append(f"{k}{s}")
            else:
                l.append(f"{k}_{s}")
        return '-'.join(l)
    else:
        s = str(d)
        s = s.replace("[", "").replace("]", "")
        s = s.replace("(", "").replace(")", "")
        s = s.replace("'", "").replace(" ", "")
        # s = s.replace(",", "_").replace(".", "_")
        return f"={s}"


@contextmanager
def timer(verbose=True) -> float:
    start = time.time()
    yield lambda: time.time() - start
    duration = time.time() - start
    if verbose:
        print(f"Duration: {time.strftime('%H:%M:%S', time.gmtime(duration))}s", flush=True)


@contextmanager
def print_timer(text, *, verbose=True, wandb=False, key="timer"):
    start = time.time()
    wandb_log = {}
    yield wandb_log
    duration = time.time() - start
    if verbose:
        print(f"{text}: {time.strftime('%H:%M:%S', time.gmtime(duration))}s", flush=True)
    if wandb:
        wandb_log[f"{key}/{text}"] = duration / 60.0 # in minutes


safe_int = lambda x: int(x) if x is not None and not x == "val" else x


def update_config(config, key, value):
    # handle cases with wandb.Config and argparse.Namespace
    if isinstance(config, dict):
        config[key] = value
    elif isinstance(config, wandb.Config):
        config.update({key: value}, allow_val_change=True)
    elif isinstance(config, argparse.Namespace):
        setattr(config, key, value)
    else:
        raise ValueError(f"Unknown config type: {type(config)}")


def get_fold(path):
    return safe_int(path.stem.rsplit("_", 1)[1])


def cartesian_dict(d):
    """
    Cartesian product of nested dict/defaultdicts of lists
    Example:

    d = {'s1': {'a': 0,
                'b': [0, 1, 2]},
         's2': {'c': [0, 1],
                'd': [0, 1]}}

    yields
        {'s1': {'a': 0, 'b': 0}, 's2': {'c': 0, 'd': 0}}
        {'s1': {'a': 0, 'b': 0}, 's2': {'c': 0, 'd': 1}}
        {'s1': {'a': 0, 'b': 0}, 's2': {'c': 1, 'd': 0}}
        {'s1': {'a': 0, 'b': 0}, 's2': {'c': 1, 'd': 1}}
        {'s1': {'a': 0, 'b': 1}, 's2': {'c': 0, 'd': 0}}
        ...
    """
    if type(d) in types_dict:
        # check for special keys that should not be expanded
        for special_key, special_type in keys_special.items():
            if special_key in d:
                yield special_type(d[special_key])
                return

        # cartesion dict expansion
        keys, values = d.keys(), d.values()
        for c in itertools.product(*(cartesian_dict(v) for v in values)):
            yield dict(zip(keys, c))
        return
    elif type(d) in types_list:
        # cartesion dict expansion
        for c in d:
            yield from cartesian_dict(c)
        return
    else:
        yield d
        return


def _get_list_leaves(d):
    if type(d) in types_list:
        out = d
        grid = True
    elif type(d) in types_dict:
        out = {}
        grid = False
        for k, v in d.items():
            if k not in keys_special:
                out_k, grid_k = _get_list_leaves(v)
                grid = grid or grid_k
                if grid_k:
                    out[k] = out_k
    else:
        out = d
        grid = False
    return out, grid


def get_list_leaves(d):
    """
    Filter nested dict of lists/dicts for leaves with lists and their siblings
    and return cartesian product
    """
    leaf_dict, _ = _get_list_leaves(d)
    return cartesian_dict(leaf_dict)


def _comparer(x):
    if x is None:
        return -math.inf
    elif type(x) in types_dict:
        return tuple((k, _comparer(v)) for k, v in x.items())
    else:
        return x


def _ddicts_to_dicts(ddict):
    if type(ddict) == defaultdict:
        return {key: _ddicts_to_dicts(val) for key, val in ddict.items()}
    elif type(ddict) == list and not all([type(v) == str for v in ddict]):
        return onp.array(ddict)
    else:
        return ddict


def dict_tree_to_ordered(d, to_dict=False):
    if type(d) in types_dict:
        out = sorted({k: dict_tree_to_ordered(v, to_dict=to_dict) for k, v in d.items()}.items(), key=_comparer)
        if to_dict:
            return dict(out)
        else:
            return OrderedDict(out)

    elif type(d) == list:
        l = [dict_tree_to_ordered(k, to_dict=to_dict) for k in d]
        try:
            return sorted(l, key=_comparer)
        except TypeError as e:
            print(f"Sorting error for:\n{pprint.pformat(l)}\n"
                  f"Did all leaves have the same type? (list or nonlist). "
                  f"Try using a singleton list when there is only one option.\n")
            raise e
    else:
        return d


def load_config(path, abspath=False, warn=True):
    """Load plain yaml config"""

    load_path = path if abspath else (PROJECT_DIR / path)

    try:
        with open(load_path, "r") as stream:
            try:
                config = yaml.load(stream, Loader=loader)
                return config

            except yaml.YAMLError as exc:
                warnings.warn(f"YAML parsing error. Returning `None` for config.\n")
                print(exc, flush=True)
                return None

    except FileNotFoundError as exc_outside:
        if warn:
            warnings.warn(f"Returning `None` for config. FileNotFoundError error for path: {load_path}")
            print(exc_outside, flush=True)
        return None


def is_consistent_subdict(suboption, option):
    matched = True
    for k, v in suboption.items():
        if type(v) in types_dict:
            matched = matched and is_consistent_subdict(v, option[k])
        else:
            matched = matched and (v == option[k])

    return matched


def load_data_config(path, abspath=False, warn_not_found=True, warn_if_grid=False, warn_if_not_grid=False):
    """Load yaml config for data specification"""

    config = load_config(path, abspath=abspath, warn=warn_not_found)
    if config is None:
        return None

    # sanity checks
    if warn_if_grid:
        if "__grid__" in config:
            warnings.warn(f"__grid__ found in config at:\n{path}")

    if warn_if_not_grid:
        if "__grid__" not in config:
            warnings.warn(f"__grid__ not found in config at:\n{path}")


    # make dict tree ordered
    config = dict_tree_to_ordered(config)
    return config


def load_methods_config(path, abspath=False, warn_not_found=True, warn_if_grid=False, warn_if_not_grid=False):
    """Load yaml config for method specification"""

    config = load_config(path, abspath=abspath, warn=warn_not_found)
    if config is None:
        return None

    # sanity checks
    if warn_if_grid:
        if "__grid__" in config:
            warnings.warn(f"__grid__ found in config at:\n{path}")

    if warn_if_not_grid:
        if "__grid__" not in config:
            warnings.warn(f"__grid__ not found in config at:\n{path}")

    # make dict tree ordered
    config = dict_tree_to_ordered(config)

    # if grid validation, expand options and update name of resulting methods
    if "__grid__" in config:
        del config["__grid__"]
        expanded_config = {}
        for method, hparams in config.items():
            all_full_settings = list(cartesian_dict(hparams))
            all_options = list(get_list_leaves(hparams))

            # match expanded set of full settings with differences in leaves for unique naming
            if not all_options:
                # only one setting
                assert len(all_full_settings) == 1
                expanded_config[method] = next(iter(all_full_settings))

            else:
                for option in all_options:
                    match = list(filter(lambda setting: is_consistent_subdict(option, setting), all_full_settings))
                    assert len(match) == 1, (f"\n{pprint.pformat(option)}"
                                             f"\n{pprint.pformat(all_options)}"
                                             f"\n{pprint.pformat(all_full_settings)}")

                    option_descr = option_to_descr(option)
                    expanded_config[f"{method}__{option_descr}"] = next(iter(match))

        return dict_tree_to_ordered(expanded_config)
    else:
        # apply cartesian_dict function nonetheless because it parses special keys
        parsed_config = dict_tree_to_ordered(list(cartesian_dict(config))[0])
        return parsed_config


def save_zip(data, path):
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        zip_file.writestr(path.name, data=data)
        zip_file.testzip()


def compress_tar_gz(source_path, archive_path):
    """
    Compresses the specified source folder into a tar.gz file.

    Args:
        source_path: The path to the folder that should be compressed.
        archive_path: The name of the output tar.gz file.
    """
    source_path = Path(source_path)
    archive_path = Path(archive_path).with_suffix(".tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        # Loop through all files and directories in the source folder
        for item in source_path.rglob('*'):
            # Add the item to the tar file, with arcname stripping the root path
            # arcname = item.relative_to(source_folder.parent) # includes parent folder
            arcname = item.relative_to(source_path)
            tar.add(item, arcname=arcname)


def print_to_txt_file(content, filename=None, verbose=True):
    if verbose:
        print(content)
    if filename is None:
        return
    with open(filename, 'a') as file:
        print(content, file=file)


def save_json(tree, path):
    with open(path, "w") as file:
        json.dump(tree, file, indent=4, sort_keys=True, cls=NumpyJSONEncoder)


def save_json_zip(tree, path):
    # do not use .with_suffix because path may contain decimal points
    if not path.name.endswith(".json"):
        path = path.parent / (path.name + ".json")
    zip_path = path.with_suffix(".zip")

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        dumped_json = json.dumps(tree, indent=4, sort_keys=True, cls=NumpyJSONEncoder)
        # write the JSON data into `.json` inside the zip file
        zip_file.writestr(path.name, data=dumped_json)
        zip_file.testzip()


def load_json(path):
    try:
        with open(path, "r") as file:
            pred = json.load(file, cls=NumpyJSONDecoder)
            return pred
    except (json.JSONDecodeError, json.decoder.JSONDecodeError) as err:
        print(f"\n\nJSONDecodeError for path: {path}", flush=True)
        raise err


def load_json_zip(zip_path):
    # do not use .with_suffix because path may contain decimal points
    zip_path = Path(zip_path)
    if not zip_path.name.endswith(".zip"):
        zip_path = zip_path.parent / (zip_path.name + ".zip")
    path = zip_path.with_suffix(".json")

    with zipfile.ZipFile(zip_path,  mode="r", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        try:
            with zip_file.open(path.name, "r") as file:
                pred = json.load(file, cls=NumpyJSONDecoder)
                return pred

        except (json.JSONDecodeError, json.decoder.JSONDecodeError) as err:
            print(f"\n\nJSONDecodeError for path: {path}", flush=True)
            raise err


def save_csv(arr, path, columns=None, dtype=onp.float32):
    if arr.size > 0:
        pd.DataFrame(arr.astype(dtype), columns=columns).to_csv(
            path,
            index=False,
            header=columns is not None,
            # float_format="%.8f",
        )


def load_csv(path, header=None, dtype=onp.float32):
    return onp.array(pd.read_csv(path, index_col=False, header=header), dtype=dtype)


def save_data(seeding, path):
    path.parent.mkdir(exist_ok=True, parents=True)
    save_json(seeding, path.with_suffix(".json"))
    return


def load_data(path, adata_raw=None, verbose=False, eval_mode=False):

    # load data config and seeding
    config = load_data_config(path.parent / EXPERIMENT_CONFIG_DATA)
    seeding = load_json(path.with_suffix(".json"))

    if config["id"] == "linear":
        from lcd_clipr.synthetic_data.linear_sde import simulate as linear_sampler
        train_datasets, test_datasets, remaining_datasets, seeding_loaded = \
            linear_sampler(seeding, config)

    elif config["id"] == "replogle":
        from lcd_clipr.realworld_data.replogle import load as replogle_loader
        train_datasets, test_datasets, remaining_datasets, seeding_loaded = \
            replogle_loader(seeding, config, adata_raw=adata_raw, verbose=verbose)

    elif config["id"] == "norman":
        from lcd_clipr.realworld_data.norman import load as norman_loader
        train_datasets, test_datasets, remaining_datasets, seeding_loaded = \
            norman_loader(seeding, config, adata_raw=adata_raw, verbose=verbose)

    elif config["id"] == "wessels":
        from lcd_clipr.realworld_data.wessels import load as wessels_loader
        train_datasets, test_datasets, remaining_datasets, seeding_loaded = \
            wessels_loader(seeding, config, adata_raw=adata_raw, verbose=verbose)

    else:
        raise ValueError(f"Unknown data config id in load_data: {config['id']}")

    for key, value in seeding_loaded.items():
        if key not in seeding:
            seeding[key] = value
            print(f"Adding missing key `{key}` to seeding (load_data).", flush=True)

    if eval_mode:
        test_intv = None if test_datasets is None else test_datasets.intv
        remaining_intv = None if remaining_datasets is None else remaining_datasets.intv
        return train_datasets, test_intv, remaining_intv, config, seeding
    else:
        return train_datasets, test_datasets, remaining_datasets, config, seeding
