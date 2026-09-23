import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch, PathPatch
import seaborn as sns
from pathlib import Path
import copy

import scipy
import numpy as onp
import pandas as pd

from lcd_clipr.experiment.plot_config import *
from lcd_clipr.experiment.dump import load_metrics_from_df, sort_methods_and_metrics
from lcd_clipr.utils.plot import plot
from lcd_clipr.experiment.plot_causal import plot_two_matrices
from lcd_clipr.experiment.plot_utils import save_legend
from lcd_clipr.experiment.analyze_causal import CAUSAL_EPS, tikhonov_pinv


nanmean = lambda arr: onp.nanmean(arr) if onp.issubdtype(arr.dtype, onp.number) else arr.iloc[0]
nanmean.__name__ = "mean"

nanmedian = lambda arr: onp.nanmedian(arr) if onp.issubdtype(arr.dtype, onp.number) else arr.iloc[0]
nanmedian.__name__ = "median"

nanstd = lambda arr: onp.nanstd(arr) if onp.issubdtype(arr.dtype, onp.number) else onp.nan
nanstd.__name__ = "std"

nansem = lambda arr: scipy.stats.sem(arr, nan_policy="omit") if onp.issubdtype(arr.dtype, onp.number) else onp.nan
nansem.__name__ = "sem"

naniqr = lambda arr: scipy.stats.iqr(arr, nan_policy="omit") if onp.issubdtype(arr.dtype, onp.number) else onp.nan
naniqr.__name__ = "iqr"

nanq025 = lambda arr: onp.nanquantile(arr, 0.025) if onp.issubdtype(arr.dtype, onp.number) else onp.nan
nanq025.__name__ = "q025"

nanq05 = lambda arr: onp.nanquantile(arr, 0.05) if onp.issubdtype(arr.dtype, onp.number) else onp.nan
nanq05.__name__ = "q05"

nanq10 = lambda arr: onp.nanquantile(arr, 0.10) if onp.issubdtype(arr.dtype, onp.number) else onp.nan
nanq10.__name__ = "q10"

nanq25 = lambda arr: onp.nanquantile(arr, 0.25) if onp.issubdtype(arr.dtype, onp.number) else onp.nan
nanq25.__name__ = "q25"

nanq75 = lambda arr: onp.nanquantile(arr, 0.75) if onp.issubdtype(arr.dtype, onp.number) else onp.nan
nanq75.__name__ = "q75"

nanq90 = lambda arr: onp.nanquantile(arr, 0.90) if onp.issubdtype(arr.dtype, onp.number) else onp.nan
nanq90.__name__ = "q90"

nanq95 = lambda arr: onp.nanquantile(arr, 0.95) if onp.issubdtype(arr.dtype, onp.number) else onp.nan
nanq95.__name__ = "q95"

nanq975 = lambda arr: onp.nanquantile(arr, 0.975) if onp.issubdtype(arr.dtype, onp.number) else onp.nan
nanq975.__name__ = "q975"


def figure_summary(
        save_path,
        df_or_df_path,
        samples_all,
        samples_aux,
        train_validation=False,
        plot_legend=False,
        test_threshold=0.05,
        only_metrics=None,
        figsize_key_boxplot="metrics_boxplot",
        rcparams=MATPLOTLIB_RCPARAMS,
):
    plt.rcParams.update(rcparams)

    if not isinstance(save_path, Path):
        save_path = Path(save_path)

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
                    f"Skipping this figure_summary call.\n")
        print(warn_str, flush=True)
        return

    # normalize values from 0 (worst) to 1 (best)
    normalized_df = df.copy()
    normalized_log_df = df.copy()

    for metric in df["metric"].unique():
        if metric not in METRICS_PAIRWISE_COMPARISONS:
            continue
        for log_space in [False, True]:
            df_metric = df.loc[df["metric"] == metric].drop("metric", axis=1)
            assert not df_metric.isna().any().any()

            # if log_scale is True, set any measurements below 0 to the best (or worst) positive value in the dataset
            # for the purpose of the pairwise comparison plot, depending on whether metric is lower or higher better
            neg_values = df_metric["val"] <= 0
            if log_space:
                df_metric.loc[~neg_values, "val"] = onp.log(df_metric.loc[~neg_values, "val"])
                df_metric.loc[neg_values, "val"] = onp.nan

            # filter out methods that shouldn't be considered for normalization
            df_stats = df_metric[~df_metric["method"].isin(METHODS_EXCLUDED_FROM_NORMALIZATION)]

            if df_stats.empty or df_stats.shape[0] == 1:
                continue

            # normalize values
            if metric in METRICS_HIGHER_BETTER:
                metric_best = df_stats["val"].max()
                metric_worst = df_stats["val"].min()
            else:
                metric_best = df_stats["val"].min()
                metric_worst = df_stats["val"].max()

            if log_space and neg_values.any():
                df_metric.loc[neg_values, "val"] = df_stats["val"].min()

            assert not df_metric.isna().any().any()

            values_normalized = (df_metric["val"] - metric_worst) / (metric_best - metric_worst)

            values_normalized = values_normalized.clip(0.0, 1.0)

            if log_space:
                normalized_log_df.loc[df["metric"] == metric, "val"] = values_normalized
            else:
                normalized_df.loc[df["metric"] == metric, "val"] = values_normalized



    """ 
    Individual boxplot panel for each metric 
    """

    for metric, log_scale in itertools.product(metrics, [True, False]):

        df_metric = df.loc[df["metric"] == metric].drop("metric", axis=1)

        # if log_scale is True, discard any measurements below 0
        if log_scale:
            df_metric = df_metric[df_metric["val"] > 0]

        # catch case where all values are nan
        if df_metric["val"].isna().all():
            continue

        if df_metric.empty:
            continue

        # add placeholder if method not measured for this metric
        for method in methods:
            if method not in df_metric["method"].unique():
                df_metric = pd.concat([df_metric, pd.DataFrame({"method": [method], "val": [onp.nan]})], ignore_index=True)

        df_metric = df_metric.reset_index(drop=True)

        # formatting of methods
        unique = df_metric["method"].unique()
        config = []
        seen = set()
        for k, *tup in METHODS_CONFIG:
            matches = list(filter(lambda w: k == w.split("__")[0], unique))
            if matches:
                assert not any([w in seen for w in matches]), f"matches `{matches}` already seen"
                seen.update(matches)
                for w in matches:
                    config.append((w, tup[0]))

        for w in filter(lambda w: w not in seen, unique):
            # unknown method substring
            config.append((w, METHODS_DEFAULT_CONFIG[0]))


        # plot
        detailed_boxplot = figsize_key_boxplot == "metrics_boxplot"

        width = FIGSIZE[figsize_key_boxplot]["ax_width"]
        height = FIGSIZE[figsize_key_boxplot]["ax_height"]
        if train_validation:
            width = width * 0.66 * len(config) / 6

        fig, ax = plt.subplots(1, 1, figsize=(width, height))

        plot_kwargs = dict(
            ax=ax,
            x="method",
            y="val",
            hue="method",
            dodge=False,  # violin positions do not change position or width according to hue
            data=df_metric,
            order=list(w for w, *_ in config),
            hue_order=list(w for w, *_ in config),
        )

        sns_plot = sns.boxplot(
            **plot_kwargs,
            showfliers=False,
            palette=dict(config),
            legend="full",
        )
        # modify transparency of the boxes
        for patch in sns_plot.patches:
            r, g, b, _ = patch.get_facecolor()
            if not detailed_boxplot:
                patch.set_facecolor((r, g, b, BOXPLOT_TRANSPARENCY))  # last element is alpha
            else:
                patch.set_facecolor((r, g, b, BOXPLOT_TRANSPARENCY_SWARM)) # last element is alpha

        # overlay swarmplot
        if detailed_boxplot:
            onp.random.seed(0)  # seed jitter
            swarm_plot = sns.stripplot(
                **plot_kwargs,
                palette=dict(config),
                legend=False,
                size=1.6,
            )

        if log_scale:
            sns_plot.set_yscale("log")
            if detailed_boxplot:
                swarm_plot.set_yscale("log")

        else:
            # # formatting
            # ax.yaxis.set_major_locator(plt.MaxNLocator(8))
            pass

        if train_validation:
            # show ticklabels to see run names
            ax.set_xticklabels([m for m, _ in config], rotation=90)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

        title = f"{metrics_names.get(metric, metric)}"
        # ax.set_title(title)

        ax.set_xlabel("")
        ax.set_ylabel(title.split("(", 1)[0]) # drop any additional info in title like "(top 20)"

        # dataset-specific y-limits
        dataset = next((d for d in METRICS_YLIMS if d in str(save_path)), None)
        ylims = (METRICS_LOG_YLIMS if log_scale else METRICS_YLIMS).get(dataset, {})
        ylim = next((lim for key, lim in ylims.items() if key in metric), None)
        if ylim is not None:
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
        elif not log_scale:
            ax.set_ylim(bottom=0.0)

        folder_name = "metrics-boxplots-log" if log_scale else "metrics-boxplots"
        file_path = save_path / folder_name / f"{metric}"

        file_path.parent.mkdir(exist_ok=True, parents=True)

        # save legend separately
        handles, labels = ax.get_legend_handles_labels()
        ax.legend().set_visible(False)
        labels_renamed = [methods_names.get(label, label) for label in labels]
        if not detailed_boxplot:
            labels_renamed = [label.replace("(ours)", "") for label in labels_renamed]

        if plot_legend:
            save_legend((file_path.parent / f"legend-{file_path.stem}"), handles, labels_renamed, rcparams=rcparams)

        plt.savefig(file_path.with_suffix(".pdf"), format="pdf")
        plt.close()

    """
    Pairwise comparison for each metric with ours
    """

    ours_methods = list(filter(lambda w: "ours" in w, methods))
    baseline_methods = list(set(methods) - set(ours_methods))

    for metric, log_space, ours_method, baseline_method in \
            itertools.product(metrics, [False, True], ours_methods, baseline_methods):

        if train_validation:
            continue

        if metric not in METRICS_PAIRWISE_COMPARISONS:
            continue

        if baseline_method in METHODS_EXCLUDED_FROM_NORMALIZATION:
            continue

        if log_space:
            df_metric = normalized_log_df.loc[df["metric"] == metric].drop(["metric"], axis=1)
        else:
            df_metric = normalized_df.loc[df["metric"] == metric].drop(["metric"], axis=1)

        # catch case where all values are nan
        if df_metric["val"].isna().all():
            continue

        if df_metric.empty:
            continue

        df_ours = df_metric.loc[df_metric["method"] == ours_method].drop(["method"], axis=1)
        df_baseline = df_metric.loc[df_metric["method"] == baseline_method].drop(["method"], axis=1)

        # collect score pairs with same id
        shared_ids = sorted(list(set(df_ours["env_id"]).intersection(set(df_baseline["env_id"]))))
        df_shared = []
        for _id in shared_ids:
            df_shared.append(pd.DataFrame({"ours": [df_ours.loc[df_ours["env_id"] == _id, "val"].values[0]],
                                           "baseline": [df_baseline.loc[df_baseline["env_id"] == _id, "val"].values[0]]}))
        df_shared = pd.concat(df_shared)



        # plot
        fig, ax = plt.subplots(1, 1, figsize=(FIGSIZE["metric_pairwise_comparison"]["ax_width"],
                                              FIGSIZE["metric_pairwise_comparison"]["ax_height"]))

        vals_baseline = df_shared["baseline"].values
        vals_ours = df_shared["ours"].values
        assert vals_baseline.shape == vals_ours.shape

        color_baseline = methods_colors.get(baseline_method, METHODS_DEFAULT_CONFIG[0])
        color_ours = methods_colors.get(ours_method, METHODS_DEFAULT_CONFIG[0])

        colors = [(color_ours if b > a else color_baseline) for a, b in zip(vals_baseline, vals_ours)]

        scatter = ax.scatter(
            vals_ours,
            vals_baseline,
            color=colors,
            s=4.0,
            linewidths=0.5,
            alpha=0.5,
        )
        scatter.set_clip_on(False) # remove blocking by axis if out of bounds

        ax.set_xlabel(methods_names.get(ours_method, ours_method))
        ax.set_ylabel(methods_names.get(baseline_method, baseline_method))

        if log_space:
            title = f"Normalized log {metrics_names.get(metric, metric)}"
        else:
            title = f"Normalized {metrics_names.get(metric, metric)}"

        suptitle = fig.suptitle(title, y=0.96)

        # significance test
        p_value = scipy.stats.wilcoxon(
            vals_ours - vals_baseline,
            # Alternative hypothesis 'greater' ('less'):
            #   the distribution underlying d = x - y = tested_values - ref_values
            #   is stochastically greater (less) than a distribution symmetric about zero.
            alternative="greater",
            nan_policy="omit",
        )[1]

        win_rate = (vals_ours > vals_baseline).mean() * 100

        sub_title = ""
        if p_value < test_threshold:
            sub_title += "WR = " + r"$\mathbf{" + f"{win_rate:.1f}" + r"}$%"
            # sub_title += "\n" + r"Wilcoxon $P$ < " + f"{test_threshold}"
            # sub_title += "\nWR =" + r"$\mathbf{" + f"{win_rate:.1f}" + r"\%}$"
        else:
            # sub_title = f"(n.s.)"
            sub_title = f"n.s."
            # sub_title += "\n" + r"Wilcoxon $P$ = " + f"{p_value:.3f}"

        ax.set_title(sub_title)

        # make x and y limits the same and add a 45 degree line and make plot quadratic
        # lims = min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1]),
        lims = 0, 1

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=2)

        ax.set_xticks(onp.linspace(0, 1, num=3, endpoint=True))
        ax.set_yticks(onp.linspace(0, 1, num=3, endpoint=True))
        ax.set_xlim(lims) # sync limits again needed after tick change
        ax.set_ylim(lims)

        ax.set_aspect('equal', adjustable='box')

        """
        Plot versions
        """
        suptitle.set_visible(False)
        folder_name = "pairwise-comparisons-log" if log_space else "pairwise-comparisons"
        file_path = save_path / folder_name / f"{metric}__{ours_method}__vs__{baseline_method}"

        # full
        file_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(file_path.with_suffix(".pdf"), format="pdf", bbox_inches='tight')

        # no xlabel
        ax.xaxis.label.set_visible(False)
        alternative_path = file_path.parent / f"{file_path.name}_no_xlabel"
        plt.savefig(alternative_path.with_suffix(".pdf"), format="pdf", bbox_inches='tight')

        plt.close()


    """ 
    Sample plots 
    """

    if samples_all is not None and samples_aux is not None:

        ###### differential expression plots
        for id, samples in samples_all.items():
            assert TRUE in samples
            ctrl_samples = samples[REF]
            target_samples = samples[TRUE]
            aux = samples_aux[id]

            if aux.get("differential_expression_mask") is not None:

                assert aux["intv_mask"].shape == aux["differential_expression_mask"].shape
                assert aux["intv_mask"].shape[0] == len(target_samples)
                assert all(aux["intv_mask"].shape[1] == data.shape[1] for data in target_samples)

                for method, pred_samples in samples.items():
                    if method == TRUE or method == REF:
                        continue

                    # handle trivial prediction case
                    if len(pred_samples) == 1 and len(target_samples) > 1:
                        pred_samples = [pred_samples[0] for _ in range(len(target_samples))]

                    plot(
                        pred_samples,
                        target_samples,
                        aux["intv_mask"],
                        filename_prefix="",
                        subfolder_name=save_path / method,
                        var_names=aux["var_names"],
                        ref_data=ctrl_samples,
                        ref_mask=aux["differential_expression_mask"],
                        cmain=METHODS_COLORS.get(method, "#43A047"),
                        ctarget="grey",
                        cref="grey",
                        method_name=METHODS_NAMES.get(method),
                        plot_params=False,
                        plot_intv_marginals=True,
                        plot_differential_marginals=True,
                        plot_marginals_violin=False,
                        plot_pairwise_grid=True,
                        grid_type="hist-scatter",
                        differential_envs=10,
                        grid_envs=FIGSIZE["grid"]["grid_envs"],
                        grid_per_env=FIGSIZE["grid"]["grid_per_env"],
                        grid_cols=FIGSIZE["grid"]["grid_cols"],
                        proj_kde=False,
                        share_scatter_axis_limits=False,
                        prioritize_marker_genes=False,
                        size_per_var=FIGSIZE["grid"]["size_per_var"],
                        to_wandb=False,
                        to_file=True,
                    )

            break # only do one data split since these are lots of plots

    return



def figure_summary_side_by_side(
        save_path,
        df_or_df_path,
        side_by_side_metrics,
        ytitle="",
        plot_legend=False,
        method_spacing=0.3,
        box_alphas=None,
        box_hatches=None,
        metric_labels=None,
        swarm_size=1.6,
        rasterize=True,
        figsize_key_boxplot="metrics_boxplot_side_by_side",
        rcparams=MATPLOTLIB_RCPARAMS,
):
    rcparams = {**copy.deepcopy(rcparams), 'hatch.linewidth': 0.5}
    plt.rcParams.update(rcparams)

    for kw in [box_alphas, box_hatches, metric_labels]:
        if kw is not None:
            assert len(kw) == len(side_by_side_metrics), \
                f"`{kw}` must have the same length as `side_by_side_metrics`"

    # map per-metric styles by metric name (robust to metrics missing from the df)
    alpha_by_metric = dict(zip(side_by_side_metrics, box_alphas)) if box_alphas is not None else {}
    hatch_by_metric = dict(zip(side_by_side_metrics, box_hatches)) if box_hatches is not None else {}
    label_by_metric = dict(zip(side_by_side_metrics, metric_labels)) if metric_labels is not None else {}

    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    save_path.mkdir(exist_ok=True, parents=True)

    # load df if path
    df = load_metrics_from_df(df_or_df_path)
    df = df[df["metric"].isin(side_by_side_metrics)]

    # sort methods and metrics
    metrics, metrics_names, methods, methods_names, methods_colors = sort_methods_and_metrics(df)

    if df.empty:
        warn_str = (f"\nNo results reported for metrics: {metrics}\n"
                    f"side_by_side_metrics: {side_by_side_metrics}\n"
                    f"methods: `{methods}\n"
                    f"metrics: `{metrics}\n"
                    f"Skipping this figure_summary_side_by_side call.\n")
        print(warn_str, flush=True)
        return

    # order metrics according to `side_by_side_metrics`
    metrics_ordered = [m for m in side_by_side_metrics if m in metrics]

    """
    Side-by-side (nested) boxplot panel: all metrics per method, sharing the same y-axis
    """
    for log_scale in [True, False]:

        df_plot = df.copy()

        # if log_scale is True, discard any measurements below 0
        if log_scale:
            df_plot = df_plot[df_plot["val"] > 0]

        # catch case where all values are nan
        if df_plot["val"].isna().all() or df_plot.empty:
            continue

        # add placeholder if a (method, metric) combination is not measured so that
        # every box position is reserved and patch <-> method mapping stays consistent
        for method, metric in itertools.product(methods, metrics_ordered):
            mask = (df_plot["method"] == method) & (df_plot["metric"] == metric)
            if not mask.any():
                df_plot = pd.concat(
                    [df_plot, pd.DataFrame({"method": [method], "metric": [metric], "val": [onp.nan]})],
                    ignore_index=True,
                )

        df_plot = df_plot.reset_index(drop=True)

        # formatting of methods (color config), as in figure_summary
        unique = df_plot["method"].unique()
        config = []
        seen = set()
        for k, *tup in METHODS_CONFIG:
            matches = list(filter(lambda w: k == w.split("__")[0], unique))
            if matches:
                assert not any([w in seen for w in matches]), f"matches `{matches}` already seen"
                seen.update(matches)
                for w in matches:
                    config.append((w, tup[0]))

        for w in filter(lambda w: w not in seen, unique):
            # unknown method substring
            config.append((w, METHODS_DEFAULT_CONFIG[0]))

        method_order = [w for w, *_ in config]
        method_color = dict(config)
        n_metrics = len(metrics_ordered)

        # plot
        figsize = FIGSIZE[figsize_key_boxplot]
        fig, ax = plt.subplots(1, 1, figsize=(figsize["ax_width"], figsize["ax_height"]))

        group_width = max(0.0, 1.0 - method_spacing)

        plot_kwargs = dict(
            ax=ax,
            x="method",
            y="val",
            hue="metric",
            dodge=True,
            data=df_plot,
            order=method_order,
            hue_order=metrics_ordered,
        )

        sns_plot = sns.boxplot(
            **plot_kwargs,
            width=group_width,
            gap=0.0,  # metric boxes within a method touch
            showfliers=False,
            legend="full",
            # draw box outlines / whiskers / caps / medians in black (not seaborn's
            # desaturated grey), matching plot_differential_expression
            boxprops={"edgecolor": "black", "linewidth": 0.5},
            medianprops={"color": "black", "linewidth": 1},
            whiskerprops={"color": "black", "linewidth": 0.5},
            capprops={"color": "black", "linewidth": 0.5},
        )

        # map a box/point center x-position back to its (method index, metric index) by
        # matching against the analytically expected dodged box centers; this is robust to
        # missing boxes and seaborn's patch ordering varying across versions
        def nearest_method_metric(center_x):
            best, best_d = (0, 0), float("inf")
            for mi in range(len(method_order)):
                for j in range(n_metrics):
                    c = mi + group_width * ((j + 0.5) / n_metrics - 0.5)
                    d = abs(center_x - c)
                    if d < best_d:
                        best_d, best = d, (mi, j)
            return best

        # default fill alpha matches figure_summary's detailed (swarm) boxplot style
        default_alpha = BOXPLOT_TRANSPARENCY_SWARM
        raster_zorder = 0.5

        # color each box by its *method*; distinguish metrics by per-metric alpha + hatch
        for patch in sns_plot.patches:
            try:
                verts = patch.get_path().vertices
                center_x = (verts[:, 0].min() + verts[:, 0].max()) / 2.0
            except Exception:
                continue
            mi, j = nearest_method_metric(center_x)
            metric = metrics_ordered[j]
            color = method_color[method_order[mi]]
            patch.set_facecolor(plt.matplotlib.colors.to_rgba(color, alpha_by_metric.get(metric, default_alpha)))

            hatch = hatch_by_metric.get(metric)
            if not hatch:
                continue  # non-hatched boxes stay fully vector with their default edge
            patch.set_hatch(hatch)
            patch.set_edgecolor("black")  # hatch line color

            if rasterize:
                # rasterize *only* the hatch (together with the flat fill it sits on) and
                # redraw a crisp vector outline on top so edges stay sharp
                ax.add_patch(PathPatch(
                    patch.get_path(),
                    transform=patch.get_transform(),
                    facecolor="none",
                    edgecolor="black",
                    linewidth=patch.get_linewidth(),
                    zorder=patch.get_zorder(),
                ))
                patch.set_linewidth(0.0)  # outline is redrawn separately as vector
                patch.set_zorder(raster_zorder - 0.1)

        # rasterize everything below `raster_zorder` (i.e. only the hatched box fills) into
        # a single image layer so the hatch line segments don't make the PDF laggy to
        # scroll (same convention as plot_differential_expression)
        if rasterize:
            ax.set_rasterization_zorder(raster_zorder)

        # overlay swarmplot
        onp.random.seed(0)  # seed jitter
        swarm_plot = sns.stripplot(
            **plot_kwargs,
            legend=False,
            size=swarm_size,
        )

        # color each point by its *method* (matching the boxes)
        for coll in swarm_plot.collections:
            colors = [method_color[method_order[nearest_method_metric(x)[0]]] for x, _ in coll.get_offsets()]
            coll.set_facecolor(colors)
            coll.set_edgecolor(colors)

        if log_scale:
            sns_plot.set_yscale("log")
            swarm_plot.set_yscale("log")

        ax.set_xticks([])
        ax.set_xticklabels([])

        ax.set_xlabel("")
        ax.set_ylabel(ytitle)

        # dataset-specific y-limits
        dataset = next((d for d in METRICS_YLIMS if d in str(save_path)), None)
        ylims = (METRICS_LOG_YLIMS if log_scale else METRICS_YLIMS).get(dataset, {})
        ylim = None
        for metric in metrics_ordered:
            ylim = next((lim for key, lim in ylims.items() if key in metric), None)
            if ylim is not None:
                break
        if ylim is not None:
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
        elif not log_scale:
            ax.set_ylim(bottom=0.0)

        folder_name = "metrics-side-by-side-log" if log_scale else "metrics-side-by-side"
        file_path = save_path / folder_name / ("__".join(metrics_ordered))
        file_path.parent.mkdir(exist_ok=True, parents=True)

        if ax.get_legend() is not None:
            ax.get_legend().set_visible(False)

        if plot_legend:
            # legend 1: methods
            save_legend(
                (file_path.parent / f"legend-{file_path.stem}"),
                [Patch(facecolor=method_color[m]) for m in method_order],
                [methods_names.get(m, m) for m in method_order],
                rcparams=rcparams,
            )

            # legend 2: metrics
            if hatch_by_metric or label_by_metric or alpha_by_metric:
                save_legend(
                    (file_path.parent / f"legend-types-{file_path.stem}"),
                    [Patch(facecolor="white", edgecolor="black", linewidth=0.5, hatch=hatch_by_metric.get(m) or None) for m in metrics_ordered],
                    [label_by_metric.get(m, metrics_names.get(m, m)) for m in metrics_ordered],
                    edgecolor="black",
                    labelspacing=0.7,
                    rcparams=rcparams,
                )

        plt.savefig(file_path.with_suffix(".pdf"), format="pdf")
        plt.close()

    return


def ablation_figure_summary(
        save_path,
        df_or_df_path,
        methods,
        methods_config,
        plot_legend=False,
        show_xticks=False,
        only_metrics=None,
        group_spacing=1,
        detailed_boxplot=True,
        rasterize=True,
        figsize_key_boxplot="metrics_boxplot_ablation",
        ylim_override=None,
        rcparams=MATPLOTLIB_RCPARAMS,
):
    rcparams = {**copy.deepcopy(rcparams), 'hatch.linewidth': 0.5}
    plt.rcParams.update(rcparams)

    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    save_path.mkdir(exist_ok=True, parents=True)

    # flatten the `methods` spec to determine plotted methods and ordering
    method_keys = [k for group in methods for k in group]
    label_map = {k: methods_config[k][0] or k for k in method_keys}
    color_map = {k: methods_config[k][1] for k in method_keys}
    hatch_map = {k: methods_config[k][2] for k in method_keys}

    # load df if path
    df = load_metrics_from_df(df_or_df_path)

    if only_metrics is not None:
        df = df[df["metric"].isin(only_metrics)]

    # restrict to the requested methods
    df = df[df["method"].isin(method_keys)]

    # sort metrics (method ordering is given by `methods`)
    metrics, metrics_names, methods_all, methods_names, methods_colors = sort_methods_and_metrics(df)

    if df.empty:
        warn_str = (f"\nNo results reported for metrics: {metrics}\n"
                    f"only_metrics: {only_metrics}\n"
                    f"methods: `{method_keys}\n"
                    f"metrics: `{metrics}\n"
                    f"Skipping this ablation_figure_summary call.\n")
        print(warn_str, flush=True)
        return

    """
    Individual boxplot panel for each metric
    """

    for metric, log_scale in itertools.product(metrics, [True, False]):

        df_metric = df.loc[df["metric"] == metric].drop("metric", axis=1)

        # if log_scale is True, discard any measurements below 0
        if log_scale:
            df_metric = df_metric[df_metric["val"] > 0]

        # catch case where all values are nan
        if df_metric["val"].isna().all():
            continue

        if df_metric.empty:
            continue

        # add placeholder if method not measured for this metric
        for method in method_keys:
            if method not in df_metric["method"].unique():
                df_metric = pd.concat([df_metric, pd.DataFrame({"method": [method], "val": [onp.nan]})], ignore_index=True)

        df_metric = df_metric.reset_index(drop=True)

        # formatting of methods with empty spacer slots inserted between groups
        config = []
        for gi, group in enumerate(methods):
            if gi > 0:
                for s in range(group_spacing):
                    config.append((f"__spacer_{gi}_{s}__", (1.0, 1.0, 1.0, 0.0)))
            for k in group:
                config.append((k, color_map[k]))

        # method key for each plotted box in order
        real_order = [w for w, _ in config if w in color_map]

        # plot
        width = FIGSIZE[figsize_key_boxplot]["ax_width"]
        height = FIGSIZE[figsize_key_boxplot]["ax_height"]

        fig, ax = plt.subplots(1, 1, figsize=(width, height))

        plot_kwargs = dict(
            ax=ax,
            x="method",
            y="val",
            hue="method",
            dodge=False,  # violin positions do not change position or width according to hue
            data=df_metric,
            order=list(w for w, *_ in config),
            hue_order=list(w for w, *_ in config),
        )

        sns_plot = sns.boxplot(
            **plot_kwargs,
            showfliers=False,
            palette=dict(config),
            legend="full",
            # draw box outlines / whiskers / caps / medians in black (not seaborn's
            # desaturated grey), matching the other figures
            boxprops={"edgecolor": "black", "linewidth": 0.5},
            medianprops={"color": "black", "linewidth": 1},
            whiskerprops={"color": "black", "linewidth": 0.5},
            capprops={"color": "black", "linewidth": 0.5},
        )

        # rasterize hatched box fills below this zorder so the hatch line segments don't
        # make the PDF laggy to scroll (same convention as figure_summary_side_by_side)
        raster_zorder = 0.5

        # modify transparency and hatching of the boxes
        drawn_order = [m for m in real_order if df_metric.loc[df_metric["method"] == m, "val"].notna().any()]
        box_patches = sns_plot.patches[:len(drawn_order)]
        for patch, method in zip(box_patches, drawn_order):
            r, g, b, _ = patch.get_facecolor()
            if not detailed_boxplot:
                patch.set_facecolor((r, g, b, BOXPLOT_TRANSPARENCY))  # last element is alpha
            else:
                patch.set_facecolor((r, g, b, BOXPLOT_TRANSPARENCY_SWARM)) # last element is alpha

            hatch = hatch_map.get(method)
            if not hatch:
                continue
            patch.set_hatch(hatch)
            patch.set_edgecolor("black")  # hatches always black

            if rasterize:
                # rasterize *only* the hatch (together with the flat fill it sits on) and
                # redraw a crisp vector outline on top so edges stay sharp
                ax.add_patch(PathPatch(
                    patch.get_path(),
                    transform=patch.get_transform(),
                    facecolor="none",
                    edgecolor="black",
                    linewidth=patch.get_linewidth(),
                    zorder=patch.get_zorder(),
                ))
                patch.set_linewidth(0.0)
                patch.set_zorder(raster_zorder - 0.1)

        # rasterize everything below `raster_zorder` (i.e. only the hatched box fills)
        if rasterize:
            ax.set_rasterization_zorder(raster_zorder)

        # overlay swarmplot
        if detailed_boxplot:
            onp.random.seed(0)  # seed jitter
            swarm_plot = sns.stripplot(
                **plot_kwargs,
                palette=dict(config),
                legend=False,
                size=1.6,
            )

        if log_scale:
            sns_plot.set_yscale("log")
            if detailed_boxplot:
                swarm_plot.set_yscale("log")

        # horizontal reference line at the median of the first method
        first_method = method_keys[0]
        first_vals = df_metric.loc[df_metric["method"] == first_method, "val"]
        if not first_vals.dropna().empty:
            ax.axhline(onp.nanmedian(first_vals), color="black", linewidth=0.5, linestyle="--", zorder=0)

        # x tick labels (overwritten labels for real methods, empty for spacers)
        if show_xticks:
            ax.set_xticks(range(len(config)))
            ax.set_xticklabels([label_map.get(m, "") for m, _ in config], rotation=90)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

        title = f"{metrics_names.get(metric, metric)}"

        ax.set_xlabel("")
        ax.set_ylabel(title.split("(", 1)[0]) # drop any additional info in title like "(top 20)"

        # dataset-specific y-limits
        dataset = next((d for d in METRICS_YLIMS if d in str(save_path)), None)
        ylims = (METRICS_LOG_YLIMS if log_scale else METRICS_YLIMS).get(dataset, {})
        ylim = next((lim for key, lim in ylims.items() if key in metric), None)
        if ylim_override is not None:
            override = next((lim for key, lim in ylim_override.items() if key in metric), None)
            if override is not None:
                ylim = (None, override[1]) if log_scale else override
        if ylim is not None:
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
        elif not log_scale:
            ax.set_ylim(bottom=0.0)

        folder_name = "metrics-boxplots-log" if log_scale else "metrics-boxplots"
        file_path = save_path / folder_name / f"{metric}"

        file_path.parent.mkdir(exist_ok=True, parents=True)

        # use the handles of plot so legend colors match
        handles, labels = ax.get_legend_handles_labels()
        handle_by_method = dict(zip(labels, handles))
        for k, h in handle_by_method.items():
            if hatch_map.get(k):
                h.set_hatch(hatch_map[k])
        ax.get_legend().set_visible(False)

        # save one legend per group of `methods`
        if plot_legend and metric == metrics[0]:
            for gi, group in enumerate(methods):
                group_handles = [handle_by_method[k] for k in group if k in handle_by_method]
                group_labels = [label_map[k] for k in group if k in handle_by_method]
                save_legend(
                    (file_path.parent / f"legend-{file_path.stem}-group{gi}"),
                    group_handles, group_labels, labelspacing=0.7, rcparams=rcparams,
                )

        fig.subplots_adjust(left=0.25, right=0.97, bottom=0.06, top=0.96)
        plt.savefig(file_path.with_suffix(".pdf"), format="pdf")
        plt.close()

    return


def comparison_figure_summary(
        save_path,
        df_or_df_path,
        only_metrics,
        only_methods,
        plot_legend=False,
        log_scale=True,
        title=None,
        ylabel=None,
        test_threshold=0.05,
        rcparams=MATPLOTLIB_RCPARAMS,
):
    plt.rcParams.update(rcparams)

    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    save_path.mkdir(exist_ok=True, parents=True)

    # load df if path
    df = load_metrics_from_df(df_or_df_path)

    df = df[df["metric"].isin(only_metrics)]
    df = df[df["method"].isin(only_methods)]

    # sort methods and metrics
    metrics, metrics_names, methods, methods_names, methods_colors = sort_methods_and_metrics(df)

    if df.empty:
        warn_str = (f"\nNo results reported for metrics: {metrics}\n"
                    f"only_metrics: {only_metrics}\n"
                    f"methods: `{methods}\n"
                    f"metrics: `{metrics}\n"
                    f"Skipping this figure_summary call.\n")
        print(warn_str, flush=True)
        return



    # if log_scale is True, discard any measurements below 0
    if log_scale:
        df = df[df["val"] > 0]

    # catch case where all values are nan
    if df["val"].isna().all():
        return

    if df.empty:
        return

    df = df.reset_index(drop=True)

    # formatting of metrics
    name_map = dict()
    unique = df["metric"].unique()
    order = []
    seen = set()
    for k, *tup in METRICS_CONFIG:
        matches = list(filter(lambda w: k == w.split("__")[0], unique))
        if matches:
            assert not any([w in seen for w in matches]), f"matches `{matches}` already seen"
            seen.update(matches)
            for w in matches:
                order.append(w)
                name_map[w] = tup[0]

    for w in filter(lambda w: w not in seen, unique):
        # unknown method substring
        order.append(w)
        name_map[w] = w


    # formatting of methods
    name_map_methods = dict()
    unique_methods = df["method"].unique()
    config_methods = []
    seen_methods = set()
    for k, *tup in METHODS_CONFIG:
        matches = list(filter(lambda w: k == w.split("__")[0], unique_methods))
        if matches:
            assert not any([w in seen_methods for w in matches]), f"matches `{matches}` already seen"
            seen_methods.update(matches)
            for w in matches:
                config_methods.append((w, tup[0]))
                name_map_methods[w] = tup[1]

    for w in filter(lambda w: w not in seen_methods, unique_methods):
        # unknown method substring
        config_methods.append((w, METHODS_DEFAULT_CONFIG[0]))
        name_map_methods[w] = w


    # rename methods and metrics
    df["metric"] = df["metric"].map(name_map)
    df["method"] = df["method"].map(name_map_methods)
    order = [name_map[w] for w in order]
    config_methods = [(name_map_methods[w], c) for w, c in config_methods]
    only_methods = [name_map_methods[w] for w in only_methods]

    # plot
    width = FIGSIZE["metrics_boxplot_comparison"]["ax_width"]
    height = FIGSIZE["metrics_boxplot_comparison"]["ax_height"]
    fig, ax = plt.subplots(1, 1, figsize=(width, height))

    plot_kwargs = dict(
        ax=ax,
        x="metric",
        y="val",
        hue="method",
        dodge=True,
        data=df,
        order=order,
        hue_order=only_methods,
    )

    sns_plot = sns.boxplot(
        **plot_kwargs,
        showfliers=False,
        palette=dict(config_methods),
        gap=0.1,
        boxprops=dict(edgecolor='black', linewidth=0.5),
        medianprops=dict(color='black', linewidth=1.0),
        whiskerprops=dict(color='black', linewidth=0.5),
        capprops=dict(color='black', linewidth=0.5),
    )
    # modify transparency of the boxes
    for patch in sns_plot.patches:
        r, g, b, _ = patch.get_facecolor()
        patch.set_facecolor((r, g, b, BOXPLOT_TRANSPARENCY_SWARM))  # last element is alpha

    # # overlay swarmplot
    onp.random.seed(0)  # seed jitter
    swarm_plot = sns.stripplot(
        **plot_kwargs,
        palette=dict(config_methods),
        legend=False,
        size=1.6,
    )

    if log_scale:
        sns_plot.set_yscale("log")
        swarm_plot.set_yscale("log")
    else:
        pass

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.tick_params(axis='x', which='both', length=0)
    ax.set_xlabel("")

    if title is not None:
        ax.set_title(title)

    # compute median comparison values
    median_values = df.groupby(["metric", "method"])["val"].median().unstack().T
    perc_changes = ((median_values.loc[only_methods[1]] - median_values.loc[only_methods[0]])
                      / median_values.loc[only_methods[0]] * 100)

    # append changes to labels
    def formatter(label):
        # signficance test
        vals_0 = df.loc[(df["metric"] == label) & (df["method"] == only_methods[0]), ["val", "env_id"]].set_index("env_id")
        vals_1 = df.loc[(df["metric"] == label) & (df["method"] == only_methods[1]), ["val", "env_id"]].set_index("env_id")
        p_value = scipy.stats.wilcoxon(
            vals_1 - vals_0,
            # Alternative hypothesis 'greater' ('less'):
            #   the distribution underlying d = x - y = tested_values - ref_values
            #   is stochastically greater (less) than a distribution symmetric about zero.
            alternative="less",
            nan_policy="omit",
        )[1]
        change = perc_changes[label]
        new_label = f"{label}"
        if p_value < test_threshold:
            # new_label += f"\n{change:.1f}%"
            new_label += "\n" + r"$\mathbf{" + f"{change:.1f}" + r"\%}$"
        else:
            new_label += f"\n(n.s.)"
        return new_label

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([
        formatter(label.get_text())
        for label in ax.get_xticklabels()
    ])

    # fig.tight_layout()

    # plt.show()

    folder_name = "comparison-boxplots-log" if log_scale else "comparison-boxplots"
    file_path = save_path.parent / folder_name / save_path.stem

    file_path.parent.mkdir(exist_ok=True, parents=True)

    # save legend separately
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().set_visible(False)
    labels_renamed = [methods_names.get(label, label) for label in labels]
    if plot_legend:
        save_legend((file_path.parent / f"legend-{file_path.stem}"), handles, labels_renamed, ncols=1, rcparams=rcparams)

    plt.savefig(file_path.with_suffix(".pdf"), format="pdf", bbox_inches='tight')
    plt.close()

    return


def causal_figure_summary(
        kwargs,
        save_path,
        df_dict,
        only_metrics,
        method="ours-mlp",
        plot_legend=False,
        rcparams=MATPLOTLIB_RCPARAMS,
        figsize_key_scaling="metric_causal_scaling",
        figsize_key_barplot="metric_causal_barplot",
):
    plt.rcParams.update(rcparams)

    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    save_path.mkdir(exist_ok=True, parents=True)

    print(save_path)

    # filter dfs based on only_metrics and method
    df = []
    for d, df_d in df_dict.items():
        for p, df_d_p in df_d.items():
            # preprocess metric@topk into metric and topk separately
            df_d_p["topk"] = df_d_p["metric"].str.split("@").str[1].str.zfill(8)
            df_d_p["metric"] = df_d_p["metric"].str.split("@").str[0]

            # filter
            df_d_p.drop(df_d_p[~df_d_p["metric"].isin(only_metrics)].index, inplace=True)
            df_d_p.drop(df_d_p[~df_d_p["method"].isin([method])].index, inplace=True)

            # remember d and p
            df_d_p["d"] = d
            df_d_p["p"] = p

            df.append(df_d_p)

    df = pd.concat(df, ignore_index=True)

    # save the concatenated dataframe
    csv_path = save_path / "data"
    csv_path.mkdir(exist_ok=True, parents=True)
    df.to_csv(csv_path / "causal_metrics_raw.csv", index=False)

    # sort methods and metrics
    metrics, metrics_names, _, _, _ = sort_methods_and_metrics(df)

    assert len(df["method"].unique()) == 1
    df.drop(columns=["method"], inplace=True)

    if df.empty:
        warn_str = (f"\nNo results reported for metrics: {metrics}\n"
                    f"only_metrics: {only_metrics}\n"
                    f"method: `{method}\n"
                    f"metrics: `{metrics}\n"
                    f"Skipping this causal_figure_summary call.\n")
        print(warn_str, flush=True)
        return

    # catch case where all values are nan
    if df["val"].isna().all():
        return

    if df.empty:
        return

    df = df.reset_index(drop=True)

    # formatting of metrics
    name_map = dict()
    inv_name_map = dict()
    unique = df["metric"].unique()
    order = []
    seen = set()
    for k, *tup in METRICS_CONFIG:
        matches = list(filter(lambda w: k == w.split("__")[0], unique))
        if matches:
            assert not any([w in seen for w in matches]), f"matches `{matches}` already seen"
            seen.update(matches)
            for w in matches:
                order.append(w)
                name_map[w] = tup[0]
                inv_name_map[tup[0]] = w

    for w in filter(lambda w: w not in seen, unique):
        # unknown method substring
        order.append(w)
        name_map[w] = w
        inv_name_map[w] = w

    # rename methods and metrics
    df["metric"] = df["metric"].map(name_map)
    order = [name_map[w] for w in order]

    # sort df so that groupy order is nice and we can print to stdout
    df = df.sort_values(by=["d", "p", "topk"])
    df["topk"] = df['topk'].apply(lambda ser: 'all' if pd.isna(ser) else str(int(ser)))

    for metric in order:

        df_metric = df.loc[df["metric"] == metric].drop("metric", axis=1)

        agg_funs = [
            nanmean,
            nanmedian,
            nanstd,
            nansem,
            naniqr,
            # nanq025,
            # nanq05,
            nanq10,
            nanq25,
            nanq75,
            nanq90,
            # nanq95,
            # nanq975,
        ]

        df_agg = df_metric.groupby(["d", "p", "topk"], dropna=False, sort=False)["val"].agg(agg_funs)

        # save the aggregated dataframe for this metric
        df_agg_reset = df_agg.reset_index()
        df_agg_reset.to_csv(csv_path / f"causal_metrics_agg_{inv_name_map[metric]}.csv", index=False)

        print(f"\n{metric}:\n")
        print(df_agg.unstack(level="topk")["mean"])

        """
        v1: multiple d and perturbs
        """

        width = FIGSIZE[figsize_key_scaling]["ax_width"]
        height = FIGSIZE[figsize_key_scaling]["ax_height"]
        fig, ax = plt.subplots(1, 1, figsize=(width, height))

        indices = list(zip(*df_agg.index))
        ds = sorted(list(set(indices[0])))
        ps = sorted(list(set(indices[1])))
        # topks = sorted(list(set(indices[2])))
        topks = sorted(list(set(indices[2])), key=lambda top: int(top) if top != "all" else onp.inf)

        for topk, topk_style, topk_style_err in [
            ("20", ":", "-"),
            # ("50", "-", "-"),
            # ("300", "--", "-"),
            # ("1000", "--", "--"),
            ("all", ":", "-"),
        ]:
            if topk not in topks:
                continue
            for j, d in enumerate(ds):
                df_sel = df_agg.loc[(d, slice(None), topk)]

                x = df_sel.index.to_numpy()

                y = df_sel["median"].values
                yerr = onp.vstack([
                    y - df_sel["q10"].values,
                    df_sel["q90"].values - y,
                ])
                if topk == "all":
                    label = f"$d={d}$"
                else:
                    label = f"$d={d}$ (@${topk})$"

                line = ax.errorbar(
                    x, y,
                    yerr=yerr,
                    fmt="d:", capsize=2, elinewidth=0.5, markersize=2.5, capthick=0.5,
                    color=COLORS[j], ecolor=COLORS[j], label=label,
                )
                line[-1][0].set_linestyle(topk_style_err)
                line[0].set_clip_on(False)  # Allow markers to extend outside axis
                line[-1][0].set_clip_on(False)  # Allow error bars to extend outside axis
                line[1][0].set_clip_on(False)
                line[1][1].set_clip_on(False)

        ax.set_xlabel("Perturbations")
        ax.set_ylabel(metric)
        ax.set_xscale("log")
        ax.set_ylim((0.0, 1.0))

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.20))

        tickpositions = [30, 100, 300, 1000]
        # tickpositions = ps
        ticklabels = [str(tick) for tick in tickpositions]  # Or custom labels if needed
        plt.xticks(tickpositions, ticklabels)
        ax.set_xlim(left=22)

        ax.legend().set_visible(False)

        # fig.tight_layout()

        # plt.show()

        # save legend separately
        file_path = save_path / f"{inv_name_map[metric]}"
        file_path.parent.mkdir(exist_ok=True, parents=True)

        if plot_legend:
            handles, labels = ax.get_legend_handles_labels()
            # labels_renamed = [methods_names.get(label, label) for label in labels]
            labels_renamed = labels

            save_legend((file_path.parent / f"legend-{file_path.stem}"), handles, labels_renamed,
                        # ncols=1,
                        ncols=len(ds),
                        # add_width=100 * PTS_TO_INCH
                        handletextpad=0.6,  # pad between the legend handle and text.
                        # handlelength=2.0, # length of the legend handles
                        columnspacing=1.0,  # spacing between columns
                        rcparams=rcparams,
                        )

        plt.savefig(file_path.with_suffix(".pdf"), format="pdf", bbox_inches='tight')
        plt.close()

        """
        v2: multiple topk across p for a given d
        """

        if inv_name_map[metric] not in [
            "causal_auroc",
        ]:

            for j, d in enumerate(ds):

                width = FIGSIZE[figsize_key_barplot]["ax_width"]
                height = FIGSIZE[figsize_key_barplot]["ax_height"]
                fig, ax = plt.subplots(1, 1, figsize=(width, height))

                df_sel = df_agg.loc[(d, slice(None), slice(None))]
                df_sel = df_sel.reorder_levels(["topk", "p"], axis=0)

                topk, psd = zip(*df_sel.index)
                psd = sorted(list(set(psd)))
                topk = list(reversed(sorted(list(set(topk)), key=lambda top: int(top) if top != "all" else onp.inf)))

                x = onp.arange(len(psd))  # the label locations
                width = 1 / (len(psd) + 1)  # the width of the bars
                multiplier = 0

                for ii, topk_val in enumerate(topk):
                    df_sel_topk = df_sel.loc[(topk_val, slice(None))]
                    offset = width * multiplier

                    y = df_sel_topk["median"].values
                    yerr = onp.vstack([
                        y - df_sel_topk["q10"].values,
                        df_sel_topk["q90"].values - y,
                    ])

                    if topk_val == "all":
                        label = f"All"
                        # label = f""
                    else:
                        label = f"@${topk_val}$"

                    cmap = cmcrameri.cm.lapaz_r
                    color = cmap((ii + 1) / (len(topk)))

                    rects = ax.bar(x + offset, y, width,
                                   yerr=yerr,
                                   color=color,
                                   alpha=0.85,
                                   error_kw=dict(capsize=0.0, elinewidth=0.5),
                                   label=label, align="edge")
                    # ax.bar_label(rects, padding=3)
                    multiplier += 1

                ax.set_xticks(x + (len(psd) / (2 * (len(psd) + 1))), psd)

                # axes and ticks
                ax.set_xlabel("Perturbations")
                ax.set_ylabel(metric)
                ax.set_ylim(bottom=0)

                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
                if ax.get_ylim()[1] - ax.get_ylim()[0] < 0.3:
                    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.10))
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.20))
                else:
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.20))


                ax.set_title(f"$d={d}$")

                ax.legend().set_visible(False)

                # fig.tight_layout()

                # plt.show()

                # save legend separately
                file_path = save_path / f"topk_d={d}_{inv_name_map[metric]}"
                file_path.parent.mkdir(exist_ok=True, parents=True)

                if plot_legend:
                    handles, labels = ax.get_legend_handles_labels()
                    # labels_renamed = [methods_names.get(label, label) for label in labels]
                    labels_renamed = labels

                    save_legend((file_path.parent / f"legend-{file_path.stem}"), handles, labels_renamed,
                                # ncols=1,
                                ncols=len(topk),
                                handletextpad=0.6, # pad between the legend handle and text.
                                # handlelength=2.0, # length of the legend handles
                                columnspacing=1.0, # spacing between columns
                                rcparams=rcparams,
                                )

                plt.savefig(file_path.with_suffix(".pdf"), format="pdf", bbox_inches='tight')
                plt.close()


    return


def causal_visualization(
        kwargs,
        save_path,
        aux,
    ):
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    save_path.mkdir(exist_ok=True, parents=True)

    """
    Visualizing causal matrices
    """
    rng = onp.random.default_rng(0)

    for data_path_id, _aux_method in aux.items():
        aux_method = _aux_method["causal"]
        for method, mats in aux_method.items():

            # make subselections
            out_degree = onp.abs(mats["a_causal_true"]).sum(1)
            top_by_out_degree = onp.argsort(-out_degree)

            for _mask, suffix in [
                (top_by_out_degree, ""),
                (top_by_out_degree[:100], "_top_100"),
                (top_by_out_degree[:200], "_top_200"),
                (top_by_out_degree[:300], "_top_300"),
            ]:
                mask = onp.sort(_mask)
                c_mask = onp.arange(len(mats["c_pred"]))

                # a_causal
                file_path = save_path / f"a_causal{suffix}_{method}_{data_path_id}"
                file_path.parent.mkdir(exist_ok=True, parents=True)
                plot_two_matrices(
                    mats["a_causal_pred"][mask][:, mask],
                    mats["a_causal_true"][mask][:, mask],
                    save_path=file_path,
                    title=f"{method} vs. true causal matrix (a_causal)",
                    ignore_diagonal_for_cmap=True,
                    cmap=cmcrameri.cm.berlin_r,
                    colorbar_label="Causal effect $\widehat{A}_{j,i}$",
                )
                a_causal_thres = CAUSAL_EPS
                plot_two_matrices(
                    mats["a_causal_pred"][mask][:, mask],
                    mats["a_causal_true"][mask][:, mask],
                    save_path=Path(f"{file_path}_thres={str(a_causal_thres).replace('.', '_')}"),
                    title=f"{method} vs. true causal matrix (a_causal) thresholded at {a_causal_thres}",
                    ignore_diagonal_for_cmap=True,
                    cmap=cmcrameri.cm.berlin_r,
                    colorbar_label="Causal effect $\widehat{A}_{j,i}$",
                    threshold_eps=a_causal_thres,
                )

                # f0 and finf
                file_path = save_path / f"f0_{suffix}_{method}_{data_path_id}"
                file_path.parent.mkdir(exist_ok=True, parents=True)
                plot_two_matrices(
                    mats["f0_pred"][c_mask][:, mask].T,
                    None,
                    save_path=file_path,
                    title=f"f0 mat (transpose)",
                    cmap=cmcrameri.cm.vanimo,
                    colorbar_label="Initial perturbation effect $\widehat{v}_{f_{q_i},j}$",
                )

                file_path = save_path / f"finf_{suffix}_{method}_{data_path_id}"
                f_inf_pred = mats["finf_pred"][c_mask][:, mask]
                plot_two_matrices(
                    f_inf_pred,
                    None,
                    save_path=file_path,
                    title=f"finf mat",
                    cmap=cmcrameri.cm.vanimo,
                    colorbar_label="Limit perturbation effect $\widehat{l}_{f_{q_j},i}$",
                )
                plot_two_matrices(
                    onp.linalg.pinv(f_inf_pred),
                    None,
                    save_path=Path(f"{file_path}_pinv"),
                    title=f"finf mat (pinv)",
                    cmap=cmcrameri.cm.vanimo,
                    colorbar_label="Pseudoinv. limit perturbation effect $\widehat{l}_{f_{q_j},i}$",
                )
                plot_two_matrices(
                    tikhonov_pinv(f_inf_pred, tikhonov=mats['causal_mat_kwargs']['tikhonov']),
                    None,
                    save_path=Path(f"{file_path}_tikhonov"),
                    title=f"finf mat (tikhonov pinv)",
                    cmap=cmcrameri.cm.vanimo,
                    colorbar_label="Tikhonov-pseudinv. limit perturbation effect $\widehat{l}_{f_{q_j},i}$",
                )

                # c
                file_path = save_path / f"c{suffix}_{method}_{data_path_id}"
                file_path.parent.mkdir(exist_ok=True, parents=True)
                plot_two_matrices(
                    mats["c_pred"][c_mask][:, mask],
                    mats["c_true"][c_mask][:, mask],
                    save_path=file_path,
                    title=f"{method} vs. true perturbations (c)",
                    cmap=cmcrameri.cm.lisbon_r,
                    colorbar_label="Perturbation $\widehat{c}_{q_i,j}$"
                )

    return
