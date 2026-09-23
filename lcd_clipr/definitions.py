from pathlib import Path
import psutil

try:
    cpu_count = len(psutil.Process().cpu_affinity())
except AttributeError:
    cpu_count = psutil.cpu_count(logical=True)


# directories
ROOT_DIR = Path(__file__).parents[0]
PROJECT_DIR = Path(__file__).parents[1]
STORE_DIR = PROJECT_DIR
SCRATCH_STORE_DIR = STORE_DIR

# subdirectories
SUBDIR_DATA = "data"
SUBDIR_ASSETS = "assets"
SUBDIR_JOBLIB = "joblib_cache"
SUBDIR_EXPERIMENTS = "experiments"
SUBDIR_PLOTS = "plots"
SUBDIR_RESULTS = "results"
SUBDIR_CLIPR_DUMP = "clipr_dump"
SUBDIR_RESULTS_DUMP = "dumps"

# data
PATH_REPLOGLE = STORE_DIR / SUBDIR_DATA / "K562_essential_raw_singlecell_01.h5ad"
PATH_NORMAN = STORE_DIR / SUBDIR_DATA / "Norman_2019_raw.h5ad"
PATH_WESSELS = STORE_DIR / SUBDIR_DATA / "Wessels_2023"

REPLOGLE_CELL_TYPE = "K562"
NORMAN_CELL_TYPE = "K562"
WESSELS_CELL_TYPE = "THP1"

# cluster
YAML_RUN = "__run__"
DEFAULT_RUN_KWARGS = {"n_cpus": 1, "n_gpus": 0, "length": "short"}


# experiments
EXPERIMENT_DATA = "data"
EXPERIMENT_PREDS = "predictions"
EXPERIMENT_SUMMARY = "summary"

EXPERIMENT_CONFIG_DATA = "data.yaml"
EXPERIMENT_CONFIG_METHODS = "methods.yaml"
EXPERIMENT_CONFIG_METHODS_VALIDATION = "methods_validation.yaml"

BASELINE_CTRL = "ctrl"
BASELINE_PERTURBED = "perturbed"
BASELINE_SALT = "salt"
BASELINE_PEPER = "peper"
BASELINE_CPA = "cpa"
BASELINE_GEARS = "gears"
BASELINE_OURS = "ours"

MAX_ENVS_SAVE_INFERRED_STATES = 32
MAX_SAMPLES_SAVE_INFERRED_STATES = 1000

# time in min for syncing and wrapping up if job ends
SYNC_TIME_MIN = 20