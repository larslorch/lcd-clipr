import argparse
from pathlib import Path

from lcd_clipr.definitions import EXPERIMENT_CONFIG_DATA
from lcd_clipr.manager import ExperimentManager

from lcd_clipr.utils.parse import load_data_config, save_data, timer

from lcd_clipr.synthetic_data.linear_sde import simulate as linear_sampler
from lcd_clipr.realworld_data.replogle import load_adata as replogle_loader
from lcd_clipr.realworld_data.norman import load_adata as norman_loader
from lcd_clipr.realworld_data.wessels import load_adata as wessels_loader

if __name__ == "__main__":
    """
    Generates data for the experiments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--descr", type=str, required=True)
    parser.add_argument("--data_config_path", type=Path, required=True)
    parser.add_argument("--path_data", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    kwargs = parser.parse_args()

    # load data config
    kwargs.fold = "val" if kwargs.fold == 0 else kwargs.fold
    data_config = load_data_config(kwargs.data_config_path, warn_if_grid=True)
    data_path = kwargs.path_data / f"{kwargs.fold}"
    config_file_name = EXPERIMENT_CONFIG_DATA

    # generate dataset seeding (i.e. perturbation train/test splits, gene selection, etc.)
    data_config["save_path"] = data_path
    data_config["fold"] = kwargs.fold

    with timer() as walltime:

        if data_config["id"] == "linear":
            *_, seeding  = linear_sampler(kwargs.seed, config=data_config)

        elif data_config["id"] == "replogle":
            _, seeding = replogle_loader(kwargs.seed, **data_config)

        elif data_config["id"] == "norman":
            _, seeding = norman_loader(kwargs.seed, **data_config)

        elif data_config["id"] == "wessels":
            _, seeding = wessels_loader(kwargs.seed, **data_config)

        else:
            raise ValueError(f"Unknown data config id: {data_config['id']}")

    seeding["seed"] = kwargs.seed
    seeding["fold"] = data_config["fold"]
    seeding["id"] = data_config["id"]

    # write seeding to file
    save_data(seeding, data_path)

    # copy data config if not copied yet
    config_path = data_path.parent / config_file_name
    if not config_path.exists():
        ExperimentManager._copy_file(kwargs.data_config_path, config_path)

    print(f"{kwargs.descr}: seed {kwargs.seed} fold {kwargs.fold} finished successfully.")