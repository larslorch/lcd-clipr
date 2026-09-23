import warnings

# in t-testing when running t-test between best method and itself
warnings.filterwarnings("ignore", message=
    "Precision loss occurred in moment calculation due to catastrophic cancellation. "
    "This occurs when the data are nearly identical. ")
warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools
import numpy as onp
import scipy
import pandas as pd

from lcd_clipr.experiment.plot_config import *
from lcd_clipr.experiment.dump import load_metrics_from_df, sort_methods_and_metrics


mean_func = onp.nanmean
mean_func.__name__ = "__mean"

median_func = onp.nanmedian
median_func.__name__ = "__median"

sem_func = lambda a: scipy.stats.sem(a, nan_policy="omit", ddof=0) if not a.isna().all() else onp.nan
sem_func.__name__ = "_sem"

mad_func = lambda a: scipy.stats.median_abs_deviation(a, nan_policy="omit") if not a.isna().all() else onp.nan
mad_func.__name__ = "_mad"


def _add_best_markers(df, table, best_aggmetr, test_threshold=0.05, t_test=False, mark_only_best=False):

    # for each metric, find best method and mark it
    for metr in df.metric.unique():

        # add marker field indicating whether a method is highlighted for this metric
        table[(metr, "_marked_")] = onp.zeros_like(table.index).astype(bool)

        vals = table.loc[:, (metr, best_aggmetr)]

        if not vals.isna().all():
            # find best method of the metric
            lo = vals == vals.min()
            hi = vals == vals.max()
            table[(metr, "best")] = hi if metr in METRICS_HIGHER_BETTER else lo

            # do t-test or overlap test with best method
            best_method = table.loc[table[(metr, "best")] == True].index.values
            assert best_method.size # should be guaranteed by the nan check of vals
            best_method = best_method[0]
            best_values = df.loc[(df.method == best_method) & (df.metric == metr)].val.values

            if t_test:
                ttest_func = lambda arr: scipy.stats.ttest_ind(arr, best_values, equal_var=False, alternative="two-sided")[1]

                table[(metr, "_pval_best_")] = df.pivot_table(
                    index=["method"],
                    columns="metric",
                    values="val",
                    aggfunc=ttest_func,
                    dropna=False) \
                    .loc[:, metr]

            else:

                # IQR overlap
                def overlap_func(arr):
                    arr_lo, arr_hi = onp.percentile(arr, [25, 75])
                    best_lo, best_hi = onp.percentile(arr, [25, 75])
                    return best_lo < arr_hi and arr_lo < best_hi


                table[(metr, "_has_overlap_")] = df.pivot_table(
                    index=["method"],
                    columns="metric",
                    values="val",
                    aggfunc=overlap_func,
                    dropna=False)\
                    .loc[:, metr]

            # mark best
            table.loc[(table[(metr, "best")] == True), (metr, "_marked_")] = True

            # mark for overlaps or p-val > test_threshold (inside (1-test_threshold)% confidence interval of t-distribution)
            if t_test:
                if not mark_only_best:
                    table.loc[(table[(metr, "_pval_best_")] >= test_threshold), (metr, "_marked_")] = True

                # drop helpers
                table = table.drop([(metr, "best"), (metr, "_pval_best_")], axis=1, errors="ignore")

            else:
                if not mark_only_best:
                    table.loc[table[(metr, "_has_overlap_")], (metr, "_marked_")] = True

                # drop helpers
                table = table.drop([(metr, "best"), (metr, "_has_overlap_")], axis=1, errors="ignore")

    table = table.sort_index(axis="columns", level="metric")

    return table


def _format_table_str(full_table, metr, metrname, latex_highlight=False):

    # format str
    mean_str_formatter = (lambda val: f"{val:.6f}") if metr in METRICS_HIGHER_PREC else \
                         (lambda val: f"{val:.3f}")

    rounding = 6 if metr in METRICS_HIGHER_PREC else 3
    mean_col = full_table.loc[:, (metr, metrname[0])].round(rounding).apply(mean_str_formatter)
    full_table.loc[:, (metr, "str")] = mean_col
    full_table.drop(columns=(metr, metrname[0]), inplace=True)
    full_table.drop(columns=(metr, metrname[1]), inplace=True)

    # add marker str
    if latex_highlight:
        marked = "\\highlight{" + full_table.loc[:, (metr, "str")] + "}"
    else:
        marked = full_table.loc[:, (metr, "str")] + " *"

    is_marked = full_table.loc[:, (metr, "_marked_")]
    full_table.loc[is_marked, (metr, "str")] = marked.loc[is_marked]
    full_table.drop(columns=(metr, "_marked_"), inplace=True)
    return full_table


def table_summary(
    kwargs,
    save_path,
    df_or_df_path,
    median_mode=False,
    only_metrics=None,
    method_summaries=False,
    latex=False,
):

    save_path.mkdir(exist_ok=True, parents=True)

    # load df if path
    df = load_metrics_from_df(df_or_df_path)

    if only_metrics is not None:
        df = df[df["metric"].isin(only_metrics)]

    # sort methods and metrics
    metrics, metrics_names, methods, methods_names, methods_colors = sort_methods_and_metrics(df)

    if df.empty:
        warn_str = (f"\nNo results reported for metrics: {metrics}\n"
                    f"only_metrics: {only_metrics}\n"
                    f"methods: `{methods}\n"
                    f"metrics: `{metrics}\n"
                    f"Skipping this table_summary call.\n")
        print(warn_str, flush=True)
        return

    # check whether we need to aggregate at all (not if we only run 1 seed)
    is_singular = all([df.loc[(df.metric == metr) & (df.method == meth)].val.size == 1 for metr, meth in itertools.product(metrics, methods)])
    if is_singular:
        print("*" * 30 + "\nWarning: single-seed measurements; ignore std dev, std err or t-tests." + "*" * 30)
        df = pd.concat([df, df]).reset_index(drop=True)

    # create table
    if median_mode:
        table_agg_metrics = [median_func, mad_func]
    else:
        table_agg_metrics = [mean_func, sem_func]

    best_aggmetr = table_agg_metrics[0].__name__
    table = df.pivot_table(
        index=["method"],
        columns=["metric"],
        values="val",
        aggfunc=table_agg_metrics,
        dropna=False) \
        .swaplevel(-1, -2, axis=1)\
        .sort_index(axis="columns", level="metric")\
        .sort_index()

    # add marker for best
    table = _add_best_markers(df, table, best_aggmetr, t_test=True, mark_only_best=median_mode)

    # reorder and rename rows
    full_table = table.reindex(methods)
    # full_table.index = full_table.index.map(methods_names)

    # convert full table to string in correct float format
    for metric in metrics:
        full_table = _format_table_str(full_table, metric, [metric.__name__ for metric in table_agg_metrics],
                                       latex_highlight=latex)

    # method summaries for train validation
    if method_summaries:
        # save individual table for each method
        save_path.mkdir(exist_ok=True, parents=True)
        base_methods = set(meth.split("__")[0] for meth in table.index)
        sort_crit = only_metrics[0]
        ascending = sort_crit not in METRICS_HIGHER_BETTER
        for base_method in base_methods:
            method_table = table[[base_method in s for s in table.index]].copy(deep=True)
            method_table = method_table.reindex(method_table[sort_crit].sort_values(by=best_aggmetr, ascending=ascending).index)

            # drop _marked_ column for method_summaries
            method_table = method_table.filter(regex='^(?!.*_marked_).*')

            # only keep mean or median columns in the summary dataframe for readability
            # if we want _sem or _mad, can add it to the condition check below
            # (in this case, should comment out .droplevel call below
            filtered_columns = [col for col in method_table.columns
                                if col[1] == '__mean'
                                or col[1] == '__median']
            method_table = method_table[filtered_columns]

            # drop 2nd level of columns (only makes sense if we don't want to keep _sem or _mad)
            method_table = method_table.droplevel(1, axis=1)

            # order table
            method_table = method_table[[metric for metric in only_metrics if metric in method_table.columns]]

            # save
            method_table.to_csv(path_or_buf=(save_path / base_method).with_suffix(".csv"))

        # table with all methods
        table_all = table.reindex(table[sort_crit].sort_values(by=best_aggmetr, ascending=ascending).index)
        table_all = table_all.filter(regex='^(?!.*_marked_).*')
        table_all = table_all[[metric for metric in only_metrics if metric in [col[0] for col in table_all.columns]]]
        table_all.to_csv(path_or_buf=(save_path / "all-sem").with_suffix(".csv"))

        # table with all methods but no _sem or _mad column
        filtered_columns = [col for col in table_all.columns
                            if col[1] == '__mean'
                            or col[1] == '__median']
        table_all = table_all[filtered_columns]
        table_all = table_all.droplevel(1, axis=1)
        table_all.to_csv(path_or_buf=(save_path / "all").with_suffix(".csv"))

        return

    else:
        full_table = full_table.droplevel(1, axis=1) # drop the dummy string level

    table_path = save_path / f"table-{save_path.name}"
    full_table.to_csv(path_or_buf=table_path.with_suffix(".csv"))
    if latex:
        full_table.style.to_latex(buf=table_path.with_suffix(".tex"))

    return df








