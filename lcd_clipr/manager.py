import warnings

warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import argparse
import shutil
from pathlib import Path

from lcd_clipr.utils.launch import generate_run_commands
from lcd_clipr.utils.parse import load_data_config, load_methods_config, safe_int

from lcd_clipr.definitions import (
    PROJECT_DIR,
    SUBDIR_EXPERIMENTS,
    SUBDIR_RESULTS,
    EXPERIMENT_CONFIG_DATA,
    EXPERIMENT_CONFIG_METHODS,
    EXPERIMENT_CONFIG_METHODS_VALIDATION,
    EXPERIMENT_DATA,
    EXPERIMENT_PREDS,
    EXPERIMENT_SUMMARY,
    YAML_RUN,
    DEFAULT_RUN_KWARGS,
)


# need to set matplotlib rcparams here, otherwise it will not work in the slurm job for summary.py
import matplotlib.pyplot as plt
from lcd_clipr.experiment.plot_config import MATPLOTLIB_RCPARAMS
plt.rcParams.update(MATPLOTLIB_RCPARAMS)


class ExperimentManager:
    """Tool for clean and reproducible experiment handling via folders"""

    def __init__(self, experiment, seed=0, verbose=True, compute="local", dry=True, n_folds=None, only_methods=None,
                 only_folds=None, scratch=False, append="", delay=None):

        self.experiment = experiment
        self.config_path = (PROJECT_DIR if scratch else PROJECT_DIR) / SUBDIR_EXPERIMENTS / self.experiment
        self.store_path_root = PROJECT_DIR
        self.store_path = self.store_path_root / SUBDIR_RESULTS / self.experiment
        self.seed = seed
        self.compute = compute
        self.verbose = verbose
        self.dry = dry
        self.append = append
        self.delay = delay

        self.logs_dir = f"{PROJECT_DIR}/logs/"
        Path(self.logs_dir).mkdir(exist_ok=True)

        self.data_config_path = self.config_path / EXPERIMENT_CONFIG_DATA
        self.methods_config_path = self.config_path / EXPERIMENT_CONFIG_METHODS
        self.methods_validation_config_path = self.config_path / EXPERIMENT_CONFIG_METHODS_VALIDATION

        if self.verbose:
            if self.config_path.exists() and self.config_path.is_dir():
                print("experiment: ", self.experiment, flush=True)
                print("store at:   ", self.store_path, flush=True, end="\n\n")
            else:
                print(f"experiment `{self.experiment}` not specified in `{self.config_path}`."
                      f"check spelling and files")
                exit(1)

        # parse configs
        self.data_config = load_data_config(self.data_config_path, abspath=True, warn_if_grid=True)

        self.methods_config = load_methods_config(self.methods_config_path, abspath=True, warn_if_grid=True)
        self.methods_validation_config = load_methods_config(self.methods_validation_config_path, abspath=True,
                                                             warn_if_not_grid=True)

        # adjust configs based on only_methods
        self.only_folds = only_folds
        self.only_methods = only_methods
        if self.only_methods is not None:
            for k in list(self.methods_config.keys()):
                if not any([m in k for m in self.only_methods]):
                    del self.methods_config[k]

            if self.methods_validation_config is not None:
                for k in list(self.methods_validation_config.keys()):
                    if not any([m in k for m in self.only_methods]):
                        del self.methods_validation_config[k]

        self.n_folds = n_folds
        if n_folds is None and self.data_config is not None:
            self.n_folds = self.data_config["n_folds"]


    @staticmethod
    def _copy_file(from_path, to_path):
        shutil.copy(from_path, to_path)


    @staticmethod
    def _inherit_specification(subdir, inherit_from):
        if inherit_from is not None:
            v = str(inherit_from.name).split("_")[1:]
            return subdir + "_" + "_".join(v)
        else:
            return subdir


    @staticmethod
    def _dir_name_without_version(p):
        return p.name.rsplit("_", 1)[0]


    def _list_main_folders(self, subdir, root_path=None, inherit_from=None):
        if root_path is None:
            root_path = self.store_path
        subdir = ExperimentManager._inherit_specification(subdir, inherit_from)
        if root_path.is_dir():
            return sorted([
                p for p in root_path.iterdir()
                if (p.is_dir() and subdir == ExperimentManager._dir_name_without_version(p))
            ])
        else:
            return []


    def _init_folder(self, subdir, root_path=None, inherit_from=None, dry=False):
        if root_path is None:
            root_path = self.store_path
        subdir = ExperimentManager._inherit_specification(subdir, inherit_from)
        existing = self._list_main_folders(subdir, root_path=root_path)
        while existing and existing[-1].name.rsplit("_", 1)[-1] == "final":
            existing.pop()
        if existing:
            latest_existing = sorted(existing)[-1]
            suff = int(latest_existing.stem.rsplit("_", 1)[-1]) + 1
        else:
            suff = 0
        folder = root_path / (subdir + f"_{suff:02d}")
        assert not folder.exists(), "Something went wrong. The data foler we initialize should not exist."
        if not dry:
            folder.mkdir(exist_ok=False, parents=True)
        return folder

    @staticmethod
    def list_data_found_and_expected(folder, *, n_folds, val):
        paths_found = sorted([p for p in folder.iterdir() if p.suffix == ".json"])
        if val:
            path_filter = lambda p: p.stem == "val"
        else:
            path_filter = lambda p: p.stem != "val"

        paths_found = sorted(list(filter(path_filter, paths_found)), key=lambda p: safe_int(p.stem))
        seeds_found = sorted([safe_int(p.stem) for p in paths_found])
        if n_folds is not None:
            seeds_expected = list(range(1, n_folds + 1)) if not val else ["val"]
        else:
            seeds_expected = None
        return paths_found, seeds_found, seeds_expected


    @staticmethod
    def list_results_found_and_expected(data_folder, results_folder, methods_config,  *, n_folds, val,
                                        results_selection=False, verbose=True):

        _, data_found, data_expected = ExperimentManager.list_data_found_and_expected(data_folder, n_folds=n_folds, val=val)
        results = dict()

        if not results_selection:
            # if no results elected, we return all data seeds for all methods
            runs = [(method, hparams, data_found) for method, hparams in methods_config.items()]
        else:
            # if a results folder is selected, we only return the missing seeds for each method
            results_found = sorted([p for p in results_folder.iterdir() if p.suffix == ".zip"])
            runs = []
            for method, hparams in methods_config.items():
                # filter runs that have already been done and are expected based on the data
                if val:
                    results_filter = lambda p: p.stem.rsplit("_", 1)[0] == method and p.stem.rsplit("_", 1)[1] == "val"
                else:
                    results_filter = lambda p: p.stem.rsplit("_", 1)[0] == method and p.stem.rsplit("_", 1)[1] != "val"

                results_method_all = list(filter(results_filter, results_found))

                # if fewer results are expected, filter them among all found results for this method
                if data_expected is not None:
                    results_method_all = list(filter(lambda p: safe_int(p.stem.rsplit("_", 1)[1]) in data_expected,
                                                     results_method_all))

                results[method] = results_method_all
                results_method_seeds = sorted([safe_int(p.stem.rsplit("_", 1)[1]) for p in results[method]])

                # check if some are missing
                if data_expected is not None:
                    results_missing = sorted(list(set(data_expected) - set(results_method_seeds)))
                    if results_missing:
                        runs.append((method, hparams, results_missing))

                    if verbose:
                        print_str = f"{method + ':':100s}"
                        print_str += f"{len(results_method_seeds):3d}/{len(data_expected)}"
                        if results_missing:
                            print_str += f"\t\t(!)\t\tMissing: {results_missing}"

                        print(print_str)

                if not results[method]:
                    del results[method]

            if verbose:
                print()

        return runs, results


    def launch_data(self, check=False, select_results=None, verbose=True):
        if check:
            assert self.store_path.exists(), f"folder doesn't exist at: {self.store_path}\nrun `--data` first"
            paths_data = self._list_main_folders(EXPERIMENT_DATA)
            assert len(paths_data) > 0, "data not created yet at; run `--data` first"
            selected = "final" if select_results is None else select_results.split("_", 1)[0]
            final_data = list(filter(lambda p: p.stem.split("_", 1)[1] == selected, paths_data))
            if final_data:
                assert len(final_data) == 1
                path_data = final_data[0]
            else:
                path_data = paths_data[-1]
            if verbose:
                print(f"data at:     {path_data}")
            return path_data

        assert self.data_config is not None, \
            f"Error when loading or file not found for data.yaml at path:\n" \
            f"{self.data_config_path}"
        data_config_path = self.data_config_path
        config_file_name = EXPERIMENT_CONFIG_DATA
        n_folds = self.n_folds

        # init results folder
        if not self.store_path.exists():
            self.store_path.mkdir(exist_ok=False, parents=True)

        # init data folder
        path_data = self._init_folder(EXPERIMENT_DATA)
        ExperimentManager._copy_file(data_config_path, path_data / config_file_name)
        if self.dry:
            shutil.rmtree(path_data)

        # launch runs that generate data
        experiment_name = kwargs.experiment.replace("/", "--")
        cmd = f"python '{PROJECT_DIR}/lcd_clipr/experiment/data.py' " \
              f"--seed {self.seed} " \
              f"--fold \$SLURM_ARRAY_TASK_ID " \
              f"--data_config_path '{data_config_path}' " \
              f"--path_data '{path_data}' "
        if self.append is not None:
            cmd += f"{self.append}"

        array_indices = range(0, n_folds + 1)

        cmd_final = cmd
        cmd_final += f"--descr '{experiment_name}-data-\$SLURM_ARRAY_TASK_ID' "

        generate_run_commands(
            array_command=cmd_final,
            array_indices=array_indices, # fold=0 initializes validation set
            mode=self.compute,
            hours=3,
            mins=59,
            n_cpus=2,
            n_gpus=0,
            # mem=30000,
            mem=50000, # for replogle we need more memory
            prompt=False,
            dry=self.dry,
            delay=self.delay,
            output_path_prefix=f"{path_data}/logs/",
            output_filename_prefix="experiment-",
        )
        # create log directory here already in case there is a failure before folder creation in script
        if not self.dry:
            (path_data / "logs").mkdir(exist_ok=True, parents=True)

        print(f"\nLaunched {len(array_indices)} runs total")
        return path_data


    def launch_methods(self, train_validation=False, check=False, select_results=None, rerun_missing=None,
                       n_samples=None, verbose=True):

        # check data has been generated
        path_data = self.launch_data(check=True, select_results=select_results, verbose=False)

        if check:
            paths_results = self._list_main_folders(EXPERIMENT_PREDS, inherit_from=path_data)
            assert len(paths_results) > 0, "results not created yet; run `--methods` first"
            selected = "final" if select_results is None else select_results
            final_results = list(filter(lambda p: p.stem.split("_", 1)[1] == selected, paths_results))
            if final_results:
                assert len(final_results) == 1
                path_results = final_results[0]
            else:
                path_results = paths_results[-1]
            if verbose:
                print(f"results at:  {path_results}")
            return path_results

        # select method config depending on whether we do train_validation or testing
        if train_validation:
            assert self.methods_validation_config is not None, \
                f"Error when loading or file not found for methods_validation.yaml at path:\n" \
                f"{self.methods_validation_config_path}"
            methods_config = self.methods_validation_config
            methods_config_path = self.methods_validation_config_path
            config_file_name = EXPERIMENT_CONFIG_METHODS_VALIDATION

        else:
            methods_config = self.methods_config
            methods_config_path = self.methods_config_path
            config_file_name = EXPERIMENT_CONFIG_METHODS

        # init results folder
        if rerun_missing is None:
            # if not rerunning missing, we create a new results folder
            path_results = self._init_folder(EXPERIMENT_PREDS, inherit_from=path_data)
            ExperimentManager._copy_file(methods_config_path, path_results / config_file_name)
            if self.dry:
                shutil.rmtree(path_results)
        else:
            # if rerunning missing, we use the existing results folder that is specified
            assert EXPERIMENT_PREDS not in rerun_missing, f"Only specify, e.g., `00_00`, not including `{EXPERIMENT_PREDS}`"
            path_results = path_data.parent / f"{EXPERIMENT_PREDS}_{rerun_missing}"

            # we copy the methods config again (in case hparams changed) but we append a suffix to avoid overwriting
            if not self.dry:
                suffix = 0
                stem, extension = config_file_name.rsplit(".", 1)
                while (path_results / config_file_name).exists():
                    suffix += 1
                    config_file_name = f"{stem}_{suffix}.{extension}"
                ExperimentManager._copy_file(methods_config_path, path_results / config_file_name)

        # print data sets expected and found
        _, data_found, data_expected = self.list_data_found_and_expected(path_data, n_folds=self.n_folds, val=train_validation)

        print(f"Found data seeds: {data_found}")
        data_missing = sorted(list(set(data_expected) - set(data_found)))
        if data_missing:
            print(f"Missing data seeds: {data_missing}")
            print(f"Expected data seeds: {data_expected}")
            warnings.warn(f"\nData sets do not match data config "
                          f"(got: `{len(data_found)}`, expected `{len(data_expected)}`).\n"
                          f"data path: {path_data}\n")
            print("Exiting.")
            return
        elif self.verbose:
            print(f"\nLaunching experiments for {len(data_found)} data set(s).")


        runs, _ = self.list_results_found_and_expected(path_data, path_results, methods_config, n_folds=self.n_folds,
                                                       val=train_validation, results_selection=rerun_missing is not None)

        if not runs:
            print("No method results are missing. Exiting.")
            return

        # launch runs that execute methods
        print("methods:\n")
        experiment_name = kwargs.experiment.replace("/", "--")
        n_launched, n_methods = 0, 0
        for method, hparams, seed_indices in runs:

            n_methods += 1

            # convert val to 0 for slurm
            seed_indices = list(map(lambda sd: 0 if sd == "val" else sd, seed_indices))

            # restrict to a subset of folds (slurm array indices) if requested
            if self.only_folds is not None:
                seed_indices = [sd for sd in seed_indices if sd in self.only_folds]
                if not seed_indices:
                    continue

            # if possible convert to range for shorter slurm command
            if len(seed_indices) > 1 and seed_indices == list(range(seed_indices[0], seed_indices[-1] + 1)):
                seed_indices = range(seed_indices[0], seed_indices[-1] + 1)

            fold = 0 if train_validation else '\$SLURM_ARRAY_TASK_ID'
            cmd = f"python '{PROJECT_DIR}/lcd_clipr/experiment/methods.py' " \
                  f"--method {method} " \
                  f"--seed {self.seed} " \
                  f"--fold {fold} " \
                  f"--path_results '{path_results}' " \
                  f"--path_data '{path_data}' " \
                  f"--path_methods_config '{methods_config_path}' " \
                  f"--descr '{experiment_name}-{method}-\$SLURM_ARRAY_TASK_ID' "

            if n_samples is not None:
                cmd += f"--n_samples {n_samples} "
            if self.append is not None:
                cmd += f"{self.append}"

            print()
            assert YAML_RUN in hparams or hparams is None, f"Add `__run__` specification of `{method}` method in yaml"
            run_kwargs = hparams[YAML_RUN] if hparams is not None else DEFAULT_RUN_KWARGS
            cmd_args = dict(
                array_indices=seed_indices,
                mode=self.compute,
                dry=self.dry,
                prompt=False,
                delay=self.delay,
                output_path_prefix=f"{path_results}/logs/",
                output_filename_prefix="experiment-",
                **run_kwargs,
            )
            # create log directory here already in case there is a failure before folder creation in script
            if not self.dry:
                (path_results / "logs").mkdir(exist_ok=True, parents=True)

            # 1 job for each dataset
            n_launched += len(seed_indices)
            generate_run_commands(array_command=cmd, **cmd_args)

        print(f"\nLaunched {n_launched} runs total ({n_methods} methods)")
        return path_results


    def launch_summary(self, train_validation=False, select_results=None):
        # check results have been generated
        path_data = self.launch_data(check=True, select_results=select_results)
        path_results = self.launch_methods(check=True, train_validation=train_validation, select_results=select_results)
        print()

        # init results folder
        path_summary = self._init_folder(EXPERIMENT_SUMMARY, inherit_from=path_results)
        if self.dry:
            shutil.rmtree(path_summary)

        # select method config depending on whether we do train_validation or testing
        if train_validation:
            assert self.methods_validation_config is not None, \
                f"Error when loading or file not found for methods_validation.yaml at path:\n" \
                f"{self.methods_validation_config_path}"
            methods_config = self.methods_validation_config
            methods_config_path = self.methods_validation_config_path

        else:
            methods_config = self.methods_config
            methods_config_path = self.methods_config_path

        # print results expected and found
        _ = self.list_results_found_and_expected(path_data, path_results, methods_config, n_folds=self.n_folds,
                                                 val=train_validation, results_selection=True)

        # create summary
        experiment_name = kwargs.experiment.replace("/", "--")
        cmd = f"JAX_PLATFORMS=cpu python '{PROJECT_DIR}/lcd_clipr/experiment/summary.py' " \
              f"--methods_config_path {methods_config_path} " \
              f"--path_data {path_data} " \
              f"--path_summary '{path_summary}' " \
              f"--path_results '{path_results}' " \
              f"--n_folds {self.n_folds} " \
              f"--descr '{experiment_name}-{path_summary.parts[-1]}' "
        if train_validation:
            cmd += f"--train_validation "
        if self.only_methods is not None:
            cmd += f"--only_methods " + " ".join(self.only_methods) + " "
        if self.append is not None:
            cmd += f"{self.append}"

        generate_run_commands(
            command_list=[cmd],
            mode=self.compute,
            hours=23,
            mins=59,
            n_cpus=12,
            n_gpus=0,
            mem=10000,
            prompt=False,
            dry=self.dry,
            delay=self.delay,
            output_path_prefix=f"{path_summary}/",
        )
        # create log directory here already in case there is a failure before folder creation in script
        if not self.dry:
            path_summary.mkdir(exist_ok=True, parents=True)

        return path_summary


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, nargs="?", default="test", help="experiment config folder")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--compute", type=str, default="local")

    parser.add_argument("--data", action="store_true")
    parser.add_argument("--methods_train_validation", action="store_true")
    parser.add_argument("--methods", action="store_true")
    parser.add_argument("--summary_train_validation", action="store_true")
    parser.add_argument("--summary", action="store_true")

    parser.add_argument("--scratch", action="store_true")

    parser.add_argument("--n_folds", type=int, help="overwrites default specified in config")
    parser.add_argument("--n_samples_method", type=int, help="overwrites default n_samples in methods.py")
    parser.add_argument("--only_methods", nargs="+", type=str)
    parser.add_argument("--only_folds", nargs="+", type=int,
                        help="restrict launched slurm array (folds) to these indices, e.g. `--only_folds 1`")
    parser.add_argument("--select_results", type=str)
    parser.add_argument("--rerun_missing", type=str,
                        help='reruns missing methods in specified results folder number, e.g. "00_00"')

    parser.add_argument("--append", type=str, default="")
    parser.add_argument("--delay", type=str, help="delay job start using slurm syntax, e.g. '2hours', '90minutes', or '2hours30minutes' (sbatch --begin=now+<delay>)")

    kwargs = parser.parse_args()

    kwargs_sum = sum([
        kwargs.data,
        kwargs.methods_train_validation,
        kwargs.methods,
        kwargs.summary_train_validation,
        kwargs.summary,
    ])
    assert kwargs_sum == 1, f"pass 1 option, got `{kwargs_sum}`"

    exp = ExperimentManager(experiment=kwargs.experiment, compute=kwargs.compute, n_folds=kwargs.n_folds,
                            dry=not kwargs.submit, only_methods=kwargs.only_methods, only_folds=kwargs.only_folds,
                            scratch=kwargs.scratch, append=kwargs.append, delay=kwargs.delay)

    if kwargs.data:
        _ = exp.launch_data()

    elif kwargs.methods or kwargs.methods_train_validation:
        _ = exp.launch_methods(train_validation=kwargs.methods_train_validation, rerun_missing=kwargs.rerun_missing,
                               n_samples=kwargs.n_samples_method)

    elif kwargs.summary or kwargs.summary_train_validation:
        _ = exp.launch_summary(train_validation=kwargs.summary_train_validation, select_results=kwargs.select_results)

    else:
        raise ValueError("Unknown option passed")

