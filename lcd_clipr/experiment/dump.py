from pathlib import Path

from collections import defaultdict
import pandas as pd
from lcd_clipr.experiment.plot_config import METRICS_ORDERED, METRICS_CONFIG, METHODS_CONFIG


def dict_to_ordered_list(d, key_order):
    return sorted(list(d.items()),
                  key=lambda tup: ((key_order.index(tup[0]) if tup[0] in key_order else (len(key_order) + 1)), tup[0]))


def dump_metrics_to_df(save_path, results):

    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    save_path.parent.mkdir(exist_ok=True, parents=True)

    # preprocess results
    metric_results_dict = defaultdict(dict)
    for method, d in results.items():
        for metr, l in d.items():
            # flatten
            # catch case where sublist is a scalar and not a list
            flattened = []
            for sublist in l:
                if not isinstance(sublist, list):
                    sublist = [sublist]
                flattened += sublist
            metric_results_dict[metr][method] = flattened

    # impose metric ordering
    metric_results = dict_to_ordered_list(metric_results_dict, METRICS_ORDERED)

    # create long list of (metric, method, value) tuples which are then aggregated into a df
    df = []
    for metric, res in metric_results:
        for m, l in res.items():
            for j, v in enumerate(l):
                df.append((metric, m, v, j))

    df = pd.DataFrame(df, columns=["metric", "method", "val", "env_id"])

    # dump table of metrics
    df.to_csv(path_or_buf=save_path.with_suffix(".csv"))
    return df


def load_metrics_from_df(df_or_df_path):
    if isinstance(df_or_df_path, str) or isinstance(df_or_df_path, Path):
        df = pd.read_csv(df_or_df_path, index_col=0)
    else:
        df = df_or_df_path
    return df


def sort_methods_and_metrics(df):
    # metrics
    metrics_order = METRICS_ORDERED
    for metric in sorted(df["metric"].unique().tolist()):
        if metric not in metrics_order:
            metrics_order.append(metric)
    metrics = [metric for metric in metrics_order if metric in df["metric"].unique()]
    metrics_names = {tup[0]: tup[1] for tup in METRICS_CONFIG if not isinstance(tup, str)}

    # methods
    method_order = [method for method, *_ in METHODS_CONFIG]
    for method in sorted(df["method"].unique().tolist()):
        if method not in method_order:
            method_order.append(method)
    methods = [method for method in method_order if method in df["method"].unique()]
    methods_names = {k: tup[1] for k, *tup in METHODS_CONFIG}
    methods_colors = {k: tup[0] for k, *tup in METHODS_CONFIG}

    return metrics, metrics_names, methods, methods_names, methods_colors