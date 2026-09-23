import warnings
warnings.filterwarnings("ignore", message="Falling back to uncompiled AVL fast distance covariance terms")
warnings.filterwarnings("ignore", category=UserWarning)

import os
from collections import defaultdict
import numpy as onp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from joblib import Parallel, delayed, parallel_config
import pprint

from sklearn.linear_model import TheilSenRegressor
from statsmodels.stats.dist_dependence_measures import distance_correlation

from lcd_clipr.core import Dataset, normalizer
from lcd_clipr.metrics import rel_change
from lcd_clipr.definitions import ROOT_DIR, SUBDIR_ASSETS
from lcd_clipr.utils.parse import print_to_txt_file
from lcd_clipr.assets.norman_classifications import gears_classifications
from lcd_clipr.metrics import eval_metric
from lcd_clipr.metrics import pearson_de, ed, mmd_ave
from lcd_clipr.experiment.plot_config import GI_DEF_METRIC, GI_SCORE_TO_LABEL, GI_METRICS_MIN, GI_METRICS_MAX, GI_NAMES, \
    MATPLOTLIB_RCPARAMS, METHODS_CONFIG, FIGSIZE, COLORS_2


def itup(intv):
    if type(intv) == tuple:
        return intv
    else:
        return tuple(onp.where(intv)[0])


def assemble_gi_dataset(test_datasets, *other_datasets):
    """
    Takes one or more `Dataset` objects and assembles a new `Dataset` object containing all doubles of `test_datasets`
    and the singles it consists of in the case that both singles are available across all available datasets.
    """

    # handle Dataset vs tuple input
    all_datasets = []
    for dataset in [test_datasets, *other_datasets]:
        if dataset is not None:
            if type(dataset) == Dataset:
                dataset = dataset.data_normalized, dataset.intv, dataset.differential_expression_mask
            all_datasets.append(dataset)

    avail = dict()
    for k, d in enumerate(all_datasets):
        for j, mask in enumerate(d[1]):
            t = itup(mask)
            assert t not in avail
            avail[t] = (k, j)

    l = []
    for t, (k, _) in avail.items():
        # filter doubles of the test dataset that have full coverage
        if len(t) == 2 and k == 0:
            if all((tt,) in avail for tt in t):
                for env in [(t[0],), (t[1],), t]:
                    if env not in l:
                        l.append(env)

    if not l:
        print("No doubles with full coverage found. Unable to assemble GI dataset. ")
        return None

    # if prediction is same for all envs, we return only a single prediction
    constant_pred_in_dataset = [len(dataset) == 1 and intv.shape[0] > 1 for dataset, intv, _ in all_datasets]
    assert all(constant_pred_in_dataset) or not any(constant_pred_in_dataset), \
        "Inconsistent dataset shapes. Either we have a constant prediction in all cases or in none."

    if all(constant_pred_in_dataset):
        return Dataset(
            data_normalized=all_datasets[0][0],
            intv=None,
        )

    intv = []
    data_normalized = []
    masks = []
    for t in l:
        k, j = avail[t]
        data_normalized.append(all_datasets[k][0][j])
        intv.append(all_datasets[k][1][j])
        if all_datasets[k][2] is not None:
            masks.append(all_datasets[k][2][j])

    return Dataset(
        data_normalized=data_normalized,
        intv=onp.array(intv),
        differential_expression_mask=onp.array(masks) if masks else None,
    )


def gi_scores(*, a, b, double):
    """
    Compute gene interaction scores from differential expression vectors as proposed in Norman et al. 2019

    Args:
        a: [n_genes,] vector of mean differential expression after perturbation A
        b: [n_genes,] vector of mean differential expression after perturbation B
        double: [n_genes,] vector of mean differential expression after perturbation A and B

    """

    assert not onp.any(onp.isnan(a)), f"NaN values in a: {a}"
    assert not onp.any(onp.isnan(b)), f"NaN values in b: {b}"
    assert not onp.any(onp.isnan(double)), f"NaN values in double: {double}"

    # [n_genes, 2]
    singles = onp.array([a, b]).T

    reg = TheilSenRegressor(random_state=1000,
                            fit_intercept=False,
                            max_subpopulation=1e5,
                            max_iter=1000)
    reg.fit(singles, double)

    # [n_genes,]
    results = dict()
    double_pred = reg.predict(singles)

    results['ts_coef_first'] = reg.coef_[0]
    results['ts_coef_second'] = reg.coef_[1]
    results['mag'] = onp.sqrt((results['ts_coef_first'] ** 2 + results['ts_coef_second'] ** 2))

    results['dcor'] = distance_correlation(singles, double)
    results['dcor_singles'] = distance_correlation(a, b)
    results['dcor_a'] = distance_correlation(a, double)
    results['dcor_b'] = distance_correlation(b, double)
    results['ts_linear_dcor'] = distance_correlation(double_pred, double)


    results['corr_fit'] = onp.corrcoef(double_pred, double)[0, 1]
    results['dominance'] = onp.log10(onp.abs(results['ts_coef_first']) / onp.abs(results['ts_coef_second']))

    results['dcor_ratio'] = onp.minimum(results['dcor_a'], results['dcor_b']) / \
                            onp.maximum(results['dcor_a'], results['dcor_b'])


    return results


def gi_scores_parallel(delta, doubles, workers=-1, verbose=10, parallel=True):
    if parallel:
        # inner_max_num_threads=2 ensures that nested parallelism does not lead to oversubscription of threads
        with parallel_config(backend="loky", inner_max_num_threads=2):
            results = Parallel(n_jobs=workers, verbose=verbose)(delayed(gi_scores)(
                a=delta[(a,)],
                b=delta[(b,)],
                double=delta[(a, b)],
            ) for a, b in doubles)
    else:
        results = [gi_scores(
            a=delta[(a,)],
            b=delta[(b,)],
            double=delta[(a, b)],
        ) for a, b in doubles]

    return {tup: res for tup, res in zip(doubles, results)}


def compute_gi_gene_mask_from_raw(pred, *, min_umi_count):
    # library size standardized data without log1p by undoing the log1p part
    pred = [onp.expm1(p) for p in pred]

    # select genes based on UMI threshold
    gene_mask = onp.vstack(pred).mean(0) > min_umi_count
    if gene_mask.sum() < gene_mask.shape[-1] * 0.2:
        print(f"Less than 20% genes ({gene_mask.sum()}) with min. mean UMI count found. "
              f"Using all genes for GI metrics computation.", flush=True)
        return onp.ones_like(gene_mask, dtype=bool)
    else:
        print(f"Selecting {gene_mask.sum()} of {gene_mask.shape[-1]} genes for GI metrics computation.", flush=True)
        return gene_mask


def compute_gi_scores_from_raw(pred, intv, ctrl, gene_mask):
    """
    Compute gene interaction scores from raw datasets as proposed in Norman et al. 2019
    """
    # library size standardized data without log1p by undoing the log1p part
    pred = [onp.expm1(p) for p in pred]
    ctrl = onp.expm1(ctrl)

    assert not onp.any(onp.isnan(ctrl)), f"NaN values in ctrl: {ctrl}"
    for j, p in enumerate(pred):
        assert not onp.any(onp.isnan(p)), f"NaN values in pred {j}: {p}"

    # convert to z-scores
    assert ctrl.ndim == 2
    ctrl_mean = onp.mean(ctrl, axis=0)
    ctrl_std = onp.std(ctrl, axis=0)
    ctrl_std = onp.where(onp.isclose(ctrl_std, 0.0), 1.0, ctrl_std)

    pred_z = [(p - ctrl_mean) / ctrl_std for p in pred]
    ctrl_z = (ctrl - ctrl_mean) / ctrl_std

    # compute differential expression vectors and mask genes
    guides = [itup(g) for g in intv]
    delta = dict()
    assert gene_mask is not None
    for j, g in enumerate(guides):
        if len(pred) == 1:
            # handles case where prediction is same for all guides
            delta[g] = onp.array(pred_z[0].mean(0) - ctrl_z.mean(0))[gene_mask]
        else:
            delta[g] = onp.array(pred_z[j].mean(0) - ctrl_z.mean(0))[gene_mask]

    # assemble double perturbations and check that both singles available
    doubles = []
    for g in guides:
        if len(g) == 2:
            assert (g[0],) in guides
            assert (g[1],) in guides
            doubles.append(g)

    assert not any(onp.isnan(x).any() for x in delta.values()), "NaN values in delta"

    # compute GI scores for all doubles
    scores = gi_scores_parallel(delta, doubles, verbose=0)
    return scores


def gi_score_metrics(true_gis, pred, intv, ctrl, gene_mask, thresholds):
    # compute GI scores from predictions
    pred_gis = compute_gi_scores_from_raw(pred, intv, ctrl, gene_mask)
    assert set(true_gis.keys()) == set(pred_gis.keys())

    # accuracy metrics
    metrics = defaultdict(list)
    for guide in true_gis.keys():
        true_gi = true_gis[guide]
        pred_gi = pred_gis[guide]

        # rmse
        metrics["se_mag"].append((true_gi["mag"] - pred_gi["mag"]) ** 2)
        metrics["se_linear_dcor"].append((true_gi["ts_linear_dcor"] - pred_gi["ts_linear_dcor"]) ** 2)
        metrics["se_dcor_ratio"].append((true_gi["dcor_ratio"] - pred_gi["dcor_ratio"]) ** 2)
        metrics["se_dcor"].append((true_gi["dcor"] - pred_gi["dcor"]) ** 2)

    # classification metrics
    if thresholds is not None:
        true_cls = defaultdict(list)
        pred_cls = defaultdict(list)

        for guide in true_gis.keys():
            gi = classify_interaction_types(pred_gis[guide], thresholds)
            true_gi = classify_interaction_types(true_gis[guide], thresholds)
            for k in gi.keys():
                true_cls[k].append(true_gi[k])
                pred_cls[k].append(gi[k])

        def safe_div(a, b):
            if not b:
                if not a:
                    return 1.0
                return 0.0
            return a / b

        for k in true_cls.keys():
            true_cls[k] = onp.array(true_cls[k])
            pred_cls[k] = onp.array(pred_cls[k])

            tp = onp.sum(true_cls[k] & pred_cls[k])
            fp = onp.sum((~true_cls[k]) & pred_cls[k])
            tn = onp.sum((~true_cls[k]) & (~pred_cls[k]))
            fn = onp.sum(true_cls[k] & (~pred_cls[k]))

            metrics[f"cls_acc_{k[:3]}"] = safe_div(tp + tn, tp + fp + tn + fn)
            metrics[f"cls_prec_{k[:3]}"] = safe_div(tp, tp + fp)
            metrics[f"cls_rec_{k[:3]}"] = safe_div(tp, tp + fn)
            metrics[f"cls_f1_{k[:3]}"] = safe_div(2 * tp, 2 * tp + fp + fn)

    return dict(**metrics), pred_gis


def run_all_gi_score_metrics(
        intv,
        pred,
        baselines,
        true,
        ctrl,
        *,
        gi_mask,
        thresholds,
        differential_expression_mask=None,
        cache=None,
    ):

    cache = cache or {}
    if any([onp.isnan(arr).any() for arr in pred]):
        warnings.warn("NaN values in pred. Skipping GI metrics computation.")
        return {}, cache

    # compute true GI scores
    true_gis = compute_gi_scores_from_raw(true, intv, ctrl, gi_mask)

    """
    GI score metrics
    """
    pred_metrics, pred_gis = gi_score_metrics(true_gis, pred, intv, ctrl, gi_mask, thresholds)
    baseline_metrics = {
        k: gi_score_metrics(true_gis, baseline, intv, ctrl, gi_mask, thresholds)
        for k, baseline in baselines.items()
    }

    results = dict()
    results_per_env = dict()

    for metric, p in pred_metrics.items():
        results[f"gi/metrics/{metric}"] = onp.mean(p)
        results_per_env[f"gi/metrics/{metric}"] = p

        for method, (baseline, _) in baseline_metrics.items():
            b = baseline[metric]
            if "se_" in metric:
                rel = -1.0 * rel_change(onp.mean(p), onp.mean(b), absolute_zero=0.0)
            elif "cls_" in metric:
                rel = rel_change(p, b, absolute_zero=0.0)
            else:
                raise ValueError(f"Unknown gi metric: {metric}")

            results[f"gi/metrics_rel_to_{method}/{metric}"] = rel


    """
    Metrics stratified by true GI type
    """
    if thresholds is not None:
        # assign doubles to their *true* respective interaction types based on thresholds
        guides_to_intv_idx = {itup(i): j for j, i in enumerate(intv)}
        assert all(guide in guides_to_intv_idx for guide in true_gis.keys())

        gi_types = defaultdict(list)
        for guide in true_gis.keys():
            gi_cls = classify_interaction_types(true_gis[guide], thresholds)
            idx = guides_to_intv_idx[guide]
            for gi, is_type in gi_cls.items():
                if is_type:
                    gi_types[gi].append(idx)

        metric_modes = [(None, ""), (differential_expression_mask, "_20")] if differential_expression_mask is not None else [(None, "")]

        for gi_type, idxs in gi_types.items():
            # subset predictions for doubles of GI type `gi_type`
            gin = gi_type[:3]
            idxs = onp.array(idxs)

            pred_gi = [pred[i] for i in idxs] if len(pred) > 1 else pred
            true_gi = [true[i] for i in idxs]
            baselines_gi = {b: [baseline[i] for i in idxs] if len(baseline) > 1 else baseline
                            for b, baseline in baselines.items()}

            # compute metrics
            for de_mask, suffix in metric_modes:
                res, res_per_env = eval_metric("gim", f"{gin}/pearson_de{suffix}", pearson_de,
                                               pred_gi,
                                               baselines_gi,
                                               true_gi,
                                               smaller_is_better=False,
                                               skip_first=False,
                                               relative_to_zero=-1.0,
                                               mask=de_mask,
                                               ctrl=ctrl)
                results.update(res)
                results_per_env.update(res_per_env)

                res, res_per_env = eval_metric("gim", f"{gin}/ed{suffix}", ed,
                                               pred_gi,
                                               baselines_gi,
                                               true_gi,
                                               smaller_is_better=True,
                                               skip_first=False,
                                               relative_to_zero=-1.0,
                                               mask=de_mask)
                results.update(res)
                results_per_env.update(res_per_env)

                res, res_per_env = eval_metric("gim", f"{gin}/mmd{suffix}", mmd_ave,
                                               pred_gi,
                                               baselines_gi,
                                               true_gi,
                                               smaller_is_better=True,
                                               skip_first=False,
                                               relative_to_zero=-1.0,
                                               mask=de_mask)
                results.update(res)
                results_per_env.update(res_per_env)

    cache["gi_scores_true"] = true_gis
    cache["gi_scores_pred"] = pred_gis

    return results, results_per_env, cache


def classify_interaction_types(scores, thresholds):
    gi = dict()
    for gi_type, gi_metric in GI_DEF_METRIC:
        score = scores[gi_metric]

        cat_thresholds = thresholds[gi_type]
        if "min" in cat_thresholds:
            lo = cat_thresholds["min"] or onp.inf # handle missing threshold as not classified ever
        else:
            lo = -onp.inf

        if "max" in cat_thresholds:
            hi = cat_thresholds["max"] or -onp.inf # handle missing threshold as not classified ever
        else:
            hi = onp.inf

        is_type = lo <= score <= hi

        gi[gi_type] = is_type

    return gi


def mask_from_seeding(gi_genes, var_names):
    gi_mask_indices = onp.array([var_names.index(g) for g in gi_genes])
    gi_mask = onp.eye(len(var_names))[gi_mask_indices].any(0)
    return gi_mask


def add_hline(axis, y, color="black", linestyle="--", pos=1.02, yoffset=0.0):
    axis.axhline([y], color=color, linestyle=linestyle)
    axis.text(pos, y + yoffset, f'{y:.2f}', color=color, transform=axis.get_yaxis_transform(), verticalalignment="center")


def add_vline(axis, x, color="black", linestyle="--", pos=0.05):
    axis.axvline([x], color=color, linestyle=linestyle)
    axis.text(x, pos, f'{x:.2f}', color=color, transform=axis.get_xaxis_transform(), verticalalignment="center")


def quantile_of_x(data, x):
    sorted_data = onp.sort(data)
    rank = onp.searchsorted(sorted_data, x, side='left')
    quantile = rank / len(data)
    return quantile


def plot_gi_scores_comparison(true_gis, pred_gis, title=None, save_path=None, filename=None,
                              label_true="true", label_pred="pred", thresholds=None):

    shared_guides = sorted(list(set(true_gis.keys()).intersection(set(pred_gis.keys()))))

    # comparison adata vs norman
    metric_plots = ["mag", "ts_linear_dcor", "dominance", "dcor", "dcor_singles", "dcor_ratio"]
    plt.rcParams.update(MATPLOTLIB_RCPARAMS)
    fig, axes = plt.subplots(2, 3, figsize=(
        FIGSIZE["si_comparison_gi_metrics"]["ax_width"],
        FIGSIZE["si_comparison_gi_metrics"]["ax_height"],
    ))

    axes = axes.flatten()
    scatter_kwargs = dict(s=5, linewidth=0.0, edgecolor=None, color=COLORS_2[5])

    for j, metric in enumerate(metric_plots):
        ax = axes[j]
        label = GI_SCORE_TO_LABEL.get(metric, metric)

        shared_true_values = [true_gis[guide][metric] for guide in shared_guides]
        shared_pred_values = [pred_gis[guide][metric] for guide in shared_guides]

        ax.scatter(shared_true_values, shared_pred_values, **scatter_kwargs)

        ax.set_xlabel(label_true)
        ax.set_ylabel(label_pred)
        ax.set_title(label)

        axmin = min(min(shared_true_values), min(shared_pred_values))
        axmax = max(max(shared_true_values), max(shared_pred_values))
        padding = (axmax - axmin + 0.05) * 0.05
        axmin -= padding
        axmax += padding

        ax.set_aspect('equal', 'box')

        ax.plot([axmin, axmax], [axmin, axmax], color="black", linestyle="--", linewidth=0.5)

        ax.set_xlim(axmin, axmax)
        ax.set_ylim(axmin, axmax)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2, min_n_ticks=3))
        ax.yaxis.set_major_locator(ticker.FixedLocator(ax.get_xticks()))

        # add thresholds for additional context
        if thresholds is not None:
            lines = set()
            for k, v in GI_DEF_METRIC:
                if v == metric:
                    for _, t in thresholds[k].items():
                        lines.add(t)

            for line in lines:
                if line is not None and axmin < line < axmax:
                    add_hline(ax, line, color=COLORS_2[6], linestyle="--")
                    add_vline(ax, line, color=COLORS_2[6], linestyle="--")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).mkdir(exist_ok=True, parents=True)
        plt.savefig((Path(save_path) / (filename or "gi_vs_true")).with_suffix(".pdf"),
                    format="pdf", bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return fig, axes


def plot_gi_scores_comparison_by_methods(method_gis, thresholds, title=None, save_path=None, filename=None,
                                         width_per_method=2.0):

    assert all(set(method_gis["true"].keys()) == set(method_gis[method].keys()) for method in method_gis.keys())

    for cat, metric in GI_DEF_METRIC:
        thres = thresholds[cat]

        # find true doubles of GI type
        true = set(guide for guide, score in method_gis["true"].items()
                   if classify_interaction_types(method_gis["true"][guide], thresholds)[cat])

        # for each method, collect GI scores predicted for the true doubles of this GI type
        method_scores = {method: [gis[guide][metric] for guide in true] for method, gis in method_gis.items()}
        all_true_scores = [score[metric] for guide, score in method_gis["true"].items()]


        # plot
        fig, ax = plt.subplots(1, 1, figsize=(width_per_method * (len(method_gis) + 1), 4))

        method_to_color = {m: c for m, c, _ in METHODS_CONFIG}
        method_to_name = {m: n for m, _, n in METHODS_CONFIG}

        methods_core = set()
        plot_data = dict()
        plot_palette = dict()
        for method, scores in method_scores.items():
            if method == "true":
                plot_data["All True"] = all_true_scores
                plot_palette["All True"] = "black"
                plot_data["GI True"] = method_scores["true"]
                plot_palette["GI True"] = "red"
            else:
                method_core = method.split("__")[0]
                methods_core.add(method_core)
                descr = method_to_name.get(method_core, method_core)
                plot_data[descr] = scores
                plot_palette[descr] = method_to_color.get(method_core, "grey")

        order = [
            "All True",
            "GI True",
            *[method_to_name.get(n, n) for n, *_ in METHODS_CONFIG if n in methods_core],
        ]
        order += [n for n in plot_data.keys() if n not in order]


        sns.swarmplot(
            ax=ax,
            data=plot_data,
            palette=plot_palette,
            order=order,
            size=5,
        )

        ax.set_title(GI_NAMES.get(cat, cat))
        ax.set_ylabel(GI_SCORE_TO_LABEL[metric])
        if metric == "mag":
            ax.set_ylim(0, 2.5)
        else:
            ax.set_ylim(0, 1.2)

        for _, t in thres.items():
            add_hline(ax, t, color="red", linestyle="--")


        if title is not None:
            fig.suptitle(title)

        plt.tight_layout()

        if save_path is not None:
            Path(save_path).mkdir(exist_ok=True, parents=True)
            plt.savefig((Path(save_path) / f'{(filename or "gi_methods")}_{cat}').with_suffix(".pdf"),
                        format="pdf", bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    return


def plot_gi_scores_comparison_by_category(true_gis, pred_gis, classif, title=None, save_path=None, filename=None,
                                          only_shared_guides=False, print_log=None, label_true="true", label_pred="pred"):
    if print_log is None:
        print_log = print


    thresholds = defaultdict(dict)
    shared_guides = sorted(list(set(true_gis.keys()).intersection(set(pred_gis.keys()))))

    plt.rcParams.update(MATPLOTLIB_RCPARAMS)

    fig, axes = plt.subplots(2, 3, figsize=(
        FIGSIZE["si_comparison_gi_metrics_by_category"]["ax_width"],
        FIGSIZE["si_comparison_gi_metrics_by_category"]["ax_height"],
    ))

    axes = axes.flatten()

    for i, (ax, cat, metric) in enumerate(zip(axes, *zip(*GI_DEF_METRIC))):

        true_values = [true_gis[guide][metric] for guide in true_gis.keys()]
        pred_values = [pred_gis[guide][metric] for guide in pred_gis.keys()]

        if only_shared_guides:
            # filter out the classified doubles that are not in shared_guides
            shared_doubles = [tuple(sorted(tup)) for tup in classif[cat] if tuple(sorted(tup)) in shared_guides]
            true_doubles = shared_doubles
            pred_doubles = shared_doubles

        else:
            # consider all classified doubles for both original and adata scores
            true_doubles = [tuple(sorted(tup)) for tup in classif[cat] if tuple(sorted(tup)) in set(true_gis.keys())]
            pred_doubles = [tuple(sorted(tup)) for tup in classif[cat] if tuple(sorted(tup)) in set(pred_gis.keys())]

        cat_true_values = [true_gis[guide][metric] for guide in true_doubles]
        cat_pred_values = [pred_gis[guide][metric] for guide in pred_doubles]

        if not cat_pred_values:
            warnings.warn(f"No doubles found in adata that match category {cat}. "
                          f"Consider changing the data config to expand the set of covered doubles.")

            ax.remove()
            if cat in GI_METRICS_MIN:
                thresholds[cat]["min"] = None
            if cat in GI_METRICS_MAX:
                thresholds[cat]["max"] = None

            continue

        # Plot each subset separately with manual x-position offsets for wider dodge
        dodge_offset = 0.20
        swarm_kwargs = dict(size=1.8, linewidth=0.0, edgecolor=None, legend=False, warn_thresh=1.0)

        # True: cat (left) and all (right)
        df_true_cat = pd.DataFrame({"x": 0 - dodge_offset, "value": cat_true_values})
        df_true_all = pd.DataFrame({"x": 0 + dodge_offset, "value": true_values})

        # Pred: cat (left) and all (right)
        df_pred_cat = pd.DataFrame({"x": 1 - dodge_offset, "value": cat_pred_values})
        df_pred_all = pd.DataFrame({"x": 1 + dodge_offset, "value": pred_values})

        for df_sub, color in [
            (df_true_cat, COLORS_2[5]), (df_true_all, "grey"),
            (df_pred_cat, COLORS_2[5]), (df_pred_all, "grey"),
        ]:
            if not df_sub.empty:
                sns.swarmplot(ax=ax, data=df_sub, x="x", y="value", color=color, native_scale=True, **swarm_kwargs)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([label_true, label_pred])
        ax.set_xlabel("")
        ax.set_xlim(-0.7, 1.7)

        ax.set_title(GI_NAMES.get(cat, cat))
        ax.set_ylabel(GI_SCORE_TO_LABEL[metric])
        if metric == "mag":
            ax.set_ylim(0, 2.5)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        else:
            ax.set_ylim(0, 1.2)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

        # define thresholds in our scores based on quantiles of original scores
        min_true = onp.quantile(cat_true_values, 0.05)
        max_true = onp.quantile(cat_true_values, 0.95)

        min_pred = onp.quantile(cat_pred_values, 0.05)
        max_pred = onp.quantile(cat_pred_values, 0.95)

        # cutoffs in our scores with the same quantiles as the cutoffs in the original scores
        lower_quantile_cutoff = quantile_of_x(true_values, min_true)
        relative_min_adata = onp.quantile(pred_values, lower_quantile_cutoff)

        upper_quantile_cutoff = quantile_of_x(true_values, max_true)
        relative_max_adata = onp.quantile(pred_values, upper_quantile_cutoff)

        if cat in GI_METRICS_MIN:
            add_hline(ax, min_true, yoffset=-0.03)
            add_hline(ax, relative_min_adata, color=COLORS_2[6], yoffset=0.03)

            thresholds[cat]["min"] = onp.round(relative_min_adata, 2)

        if cat in GI_METRICS_MAX:
            add_hline(ax, max_true, yoffset=-0.03)
            add_hline(ax, relative_max_adata, color=COLORS_2[6], yoffset=0.03)

            thresholds[cat]["max"] = onp.round(relative_max_adata, 2)

        print_log(f"{cat} {metric}")
        print_log(f"min   {label_true}: {min_true:.3f} {label_pred}: {min_pred:.3f}  quantile: {lower_quantile_cutoff:.3f}")
        print_log(f"max   {label_true}: {max_true:.3f} {label_pred}: {max_pred:.3f}  quantile: {upper_quantile_cutoff:.3f}")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig((Path(save_path) / (filename or "interaction_category_thresholds")).with_suffix(".pdf"),
                    format="pdf", bbox_inches='tight')
        plt.close()

        # Save legend separately
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS_2[5], markersize=5, linewidth=0),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=5, linewidth=0),
            Line2D([0], [0], color=COLORS_2[6], linestyle='--', linewidth=1),
        ]
        legend_labels = ['Gene pairs originally annotated with GI', 'All gene pairs', 'Calibrated threshold']
        fig_legend = plt.figure(figsize=(FIGSIZE["legend"]["ax_width_col"] * 2,
                                         FIGSIZE["legend"]["ax_height"]))
        ax_legend = fig_legend.add_subplot(111)
        _ = ax_legend.legend(legend_elements, legend_labels, loc='center', ncols=1, frameon=False, handlelength=2.4, handletextpad=1.0)
        ax_legend.axis('off')
        fig_legend.savefig((Path(save_path) / ((filename or "interaction_category_thresholds") + "_legend")).with_suffix(".pdf"),
                           format="pdf", bbox_inches='tight')
        plt.close(fig_legend)
    else:
        plt.show()


    return thresholds


def adata_categorize_interactions(adata, *, min_umi_count, save_path=None, title=None, classify_gi_norman=False,
                                  only_shared_guides=False, ctrl_key="ctrl", guide_key="guide_ids"):


    # faster than converting sparse to dense matrix in preprocessing operations
    if type(adata.X) == onp.ndarray:
        x = adata.X
    else:
        x = adata.X.toarray()
    adata.X = None
    adata.X = x

    """
    Compute GI scores as described in Norman et al. 2019
    """
    # convert adata to our data format (expects normalized)
    ctrl = normalizer(adata[adata.obs[guide_key] == ctrl_key].X)
    data = []
    intv = []
    for g in adata.obs[guide_key].unique():
        data_g = adata[adata.obs[guide_key] == g].X
        data.append(normalizer(data_g))
        intv.append(tuple() if g == ctrl_key else tuple(sorted(g.split(","))))

    # compute GI scores
    gene_selection = compute_gi_gene_mask_from_raw(data, min_umi_count=min_umi_count)
    if not classify_gi_norman:
        return gene_selection, None

    scores = compute_gi_scores_from_raw(data, intv, ctrl, gene_selection)

    """
    Load original GI scores reported by Norman et al. 2019
    """
    if save_path is not None:
        save_path = Path(save_path)
        seed = save_path.stem
        save_path = save_path.parent / "gi"
        save_path.mkdir(exist_ok=True, parents=True)
    else:
        seed = ""

    txt_file = save_path / f"log_{seed}.txt" if save_path is not None else None
    if txt_file is not None:
        if txt_file.exists():
            os.remove(txt_file)
    print_log = lambda content: print_to_txt_file(content, txt_file)

    print_log("Categorizing gene interactions")
    print_log(adata.shape)

    norman_stats_path = ROOT_DIR / SUBDIR_ASSETS / "aax4438_tables9_raw.csv"
    orig = pd.read_csv(norman_stats_path, index_col=0, header=0)

    flipped = [tuple(sorted(g.split("_"))) != tuple(g.split("_")) for g in orig.index]
    orig.index = [tuple(sorted(g.split("_"))) for g in orig.index]
    temp = orig.loc[flipped, "ts_coef_first"].copy()
    orig.loc[flipped, "ts_coef_first"] = orig.loc[flipped, "ts_coef_second"]
    orig.loc[flipped, "ts_coef_second"] = temp

    orig['mag'] = onp.sqrt((orig['ts_coef_first'] ** 2 + orig['ts_coef_second'] ** 2))
    orig['dominance'] = onp.log10(onp.abs(orig['ts_coef_first']) / onp.abs(orig['ts_coef_second']))

    true_scores = {guide: {k: v for k, v in series.items()} for guide, series in orig.iterrows()}

    """
    Plotting
    """
    # comparison adata vs norman
    plot_gi_scores_comparison(
        true_scores,
        scores,
        title=title,
        save_path=save_path,
        filename=f"true_gi_scores_shared_doubles_{seed}",
        label_true="Norman et al.",
        label_pred="Ours",
    )

    # comparison adata vs norman by category
    classif = gears_classifications.copy()
    thresholds = plot_gi_scores_comparison_by_category(
        true_scores,
        scores,
        classif,
        title=title,
        save_path=save_path,
        filename=f"interaction_category_thresholds_{seed}",
        only_shared_guides=only_shared_guides,
        print_log=print_log,
        label_true="Norman et al.",
        label_pred="Ours",
    )

    """
    Annotate adata perturbations based on computed thresholds to compute summary statistics
    """
    # adjust thresholds to be consistent for classification
    thresholds = dict(**thresholds)
    print_log("thresholds")
    print_log(pprint.pformat(thresholds))

    if thresholds["synergy"]["min"] is not None and thresholds["additive"]["max"] is not None:
        thresholds["synergy"]["min"] = thresholds["additive"]["max"] = \
            max(thresholds["synergy"]["min"], thresholds["additive"]["max"])

    if thresholds["suppression"]["max"] is not None and thresholds["additive"]["min"] is not None:
        thresholds["suppression"]["max"] = thresholds["additive"]["min"] = \
            min(thresholds["suppression"]["max"], thresholds["additive"]["min"])

    print_log("thresholds after adjustment")
    print_log(pprint.pformat(thresholds))

    for gi_type, _ in GI_DEF_METRIC:
        adata.obs[f"gi_{gi_type}"] = False

    for double in scores.keys():
        # find original double string
        double_str_0 = ",".join(double)
        double_str_1 = ",".join(reversed(double))
        assert (double_str_0 in adata.obs[guide_key].unique()) != \
               (double_str_1 in adata.obs[guide_key].unique())
        double_str = double_str_0 if double_str_0 in adata.obs[guide_key].unique() else double_str_1

        gis = classify_interaction_types(scores[double], thresholds)

        for gi_type, is_type in gis.items():
            adata.obs.loc[adata.obs[guide_key] == double_str, f"gi_{gi_type}"] = is_type

    print_log("Proportions of double interactions by category")
    adata_doubles = adata[adata.obs[guide_key].str.contains(",")]
    agg = adata_doubles.obs.groupby(guide_key).agg({f"gi_{gi_type}": "first" for gi_type, _ in GI_DEF_METRIC})
    print_log(agg.mean())
    print_log(agg.sum())

    return gene_selection, thresholds
