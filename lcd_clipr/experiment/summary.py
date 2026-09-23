import warnings
import argparse
from pathlib import Path

from collections import defaultdict
import numpy as onp
import copy
import scanpy as sc

import matplotlib.pyplot as plt
from lcd_clipr.experiment.plot_config import MATPLOTLIB_RCPARAMS
plt.rcParams.update(MATPLOTLIB_RCPARAMS)


from lcd_clipr.definitions import PATH_NORMAN, PATH_REPLOGLE
from lcd_clipr.utils.parse import load_json, load_methods_config, get_fold, load_data, load_json_zip, compress_tar_gz, \
    print_timer, safe_int
from lcd_clipr.experiment.plot_config import METRICS_ORDERED, TRUE, REF
from lcd_clipr.experiment.table import table_summary
from lcd_clipr.experiment.figures import figure_summary, causal_visualization
from lcd_clipr.experiment.dump import dump_metrics_to_df
from lcd_clipr.metrics import (
    DE_RANK_BY,
    compute_differential_expression_sign,
    top_hvg_mask,
    ed,
    mmd_ave,
    w1_eltwise,
    rmse_m,
    rmse_s,
    bias_s,
    pearson_de,
    precision,
    recall,
    f1,
    roc_auc,
    average_precision,
)
from lcd_clipr.experiment.analyze_causal import compute_causal_mat_metrics
from lcd_clipr.core import normalizer

from lcd_clipr.utils.gi import run_all_gi_score_metrics, assemble_gi_dataset, mask_from_seeding, \
    plot_gi_scores_comparison, plot_gi_scores_comparison_by_methods
from lcd_clipr.realworld_data.wessels import load_adata_raw as load_wessels_adata_raw

from lcd_clipr.manager import ExperimentManager


# only compute GI metrics for datasets with sufficient singles and doubles overlap
# (in wessels, we have a lot of doubles but not sufficient singles to disentangle an interaction effect)
GI_DATASETS = [
    "norman",
]

CAUSAL_DATASETS = [
    "linear",
]

DATASETS_NO_NORMALIZATION = [
    "linear",
]


def load_adata_raw(dataset_id):
    """Load raw dataset once to speed up dataloading"""
    if dataset_id == "linear":
        adata_raw = None
    elif dataset_id == "replogle":
        adata_raw = sc.read_h5ad(PATH_REPLOGLE)
    elif dataset_id == "norman":
        adata_raw = sc.read_h5ad(PATH_NORMAN)
    elif dataset_id == "wessels":
        adata_raw = load_wessels_adata_raw()
    else:
        raise ValueError(f"Unknown dataset id {dataset_id}")
    return adata_raw

from lcd_clipr.utils.cache import init_cache
# function caching for fast prototyping / debugging (mirrors run.py)
memory = init_cache()

load_data_cached = memory.cache(load_data)
compute_differential_expression_sign_cached = memory.cache(compute_differential_expression_sign)
load_json_zip_cached = memory.cache(load_json_zip)


def compute_metrics(
        *,
        dataset_id,
        seeding,
        path,
        train_datasets,
        test_datasets,
        remaining_datasets,
        test_de_info,
        kwargs,
    ):

    results = dict()
    aux = dict()

    # load predictions
    method_preds = load_json_zip_cached(path)

    # legacy: handle unnormalized predictions
    pred_train = method_preds["norm_train"] if "norm_train" in method_preds else normalizer(method_preds["samples_train"])
    pred_test = method_preds["norm_test"] if "norm_test" in method_preds else normalizer(method_preds["samples_test"])
    pred_remaining = method_preds["norm_remaining"] if "norm_remaining" in method_preds else normalizer(method_preds["samples_remaining"])

    intv_train = method_preds["intv_train"]
    intv_test = method_preds["intv_test"]
    intv_remaining = method_preds["intv_remaining"]

    assert onp.allclose(train_datasets.intv, intv_train)
    assert test_datasets is None or onp.allclose(test_datasets.intv, intv_test)
    assert remaining_datasets is None or onp.allclose(remaining_datasets.intv, intv_remaining)

    """
    Test distribution metrics on normalized counts (noisy data)
    """
    if test_datasets is not None:
        # control and mean perturbation control
        ctrl = train_datasets.data_normalized[0]
        ref_perturb_means = onp.stack([data.mean(0) for data in train_datasets.data_normalized[1:]], axis=0)
        perturb_ctrl = ref_perturb_means.mean(axis=0, keepdims=True)

        # construct gene subsets
        metric_modes = [
            (None, ""),
        ]

        de_mask = test_datasets.differential_expression_mask
        if de_mask is not None:
            metric_modes.append((de_mask.astype(bool), "_20"))

        if "hv_genes" in seeding:
            m = test_datasets.intv.shape[0]
            metric_modes.extend([
                (top_hvg_mask(seeding["hv_genes"], seeding["var_names"], quantiles=(None, 0.05), tile=m), "_hvg_0-5"),
                (top_hvg_mask(seeding["hv_genes"], seeding["var_names"], quantiles=(0.05, 0.20), tile=m), "_hvg_5-20"),
                (top_hvg_mask(seeding["hv_genes"], seeding["var_names"], quantiles=(0.20, None), tile=m), "_hvg_20-100"),
            ])

        """
        Expression metrics
        """
        for metric_name, metric, metric_kwargs in  [
            ("rmse_m", rmse_m, {}),
            ("pearson_de", pearson_de, dict(ctrl=ctrl)),
            ("pearson_systema", pearson_de, dict(ctrl=perturb_ctrl)),
            ("ed", ed, {}),
            ("mmd", mmd_ave, {}),
            ("w1_eltwise", w1_eltwise, {}),
        ]:
            for mask, suffix in metric_modes:
                results[f"{metric_name}{suffix}"] = \
                    metric(pred_test, test_datasets.data_normalized, mask=mask, **metric_kwargs)

        """
        Classification metrics
        """
        test_signs, _, test_ranks, _ = test_de_info
        pred_signs, _, pred_ranks, _ = compute_differential_expression_sign(pred_test, ctrl, return_details=True)

        # convert to binary (DE vs not DE)
        test_de = onp.abs(test_signs)
        pred_de = onp.abs(pred_signs)

        # classification of whether or not a gene is DE
        for metric_name, metric, metric_kwargs in  [
            ("classify_de_prec", precision, {}),
            ("classify_de_recall", recall, {}),
            ("classify_de_f1", f1, {}),
        ]:
            for mask, suffix in metric_modes:
                results[f"{metric_name}{suffix}"] = \
                    metric(pred_de, test_de, mask=mask, **metric_kwargs)

        # classification of top-k most-significant DE
        for top_k in [20, 100]:
            for rank_by in DE_RANK_BY:
                test_top_k = (test_ranks["pval"] < top_k).astype(int)
                pred_top_k = (pred_ranks[rank_by] < top_k).astype(int)
                pred_score = -pred_ranks[rank_by]

                # score-based
                for metric_name, metric, metric_kwargs in [
                    ("classify_topk_auroc", roc_auc, {}),
                    ("classify_topk_ap", average_precision, {}),
                ]:
                    results[f"{metric_name}_{rank_by}_{top_k}"] = \
                        metric(pred_score, test_top_k, mask=None, **metric_kwargs)

                # top-k based
                for metric_name, metric, metric_kwargs in  [
                    ("classify_topk_precision", precision, {}),
                ]:
                    results[f"{metric_name}_{rank_by}_{top_k}"] = \
                        metric(pred_top_k, test_top_k, mask=None, **metric_kwargs)

    """
    Train distribution metrics on states
    """
    if train_datasets.data_true is not None and "inferred_states" in method_preds:

        # check that number of envs match
        state_results = dict()
        states_true = onp.stack([x for x in train_datasets.data_true if x is not None])
        states_pred = method_preds["inferred_states"]
        states_ctrl = states_true[0]

        if not states_true.shape[0] == states_pred.shape[0]:
            print(f"State data environments do not match: {states_true.shape} != {states_pred.shape}. "
                  f"This should not happen. Skipping state metrics.")

        else:
            """
            Expression metrics
            """
            for metric_name, metric, metric_kwargs in [
                ("rmse_m", rmse_m, {}),
                ("rmse_s", rmse_s, {}),
                ("bias_s", bias_s, {}),
                ("pearson_de", pearson_de, dict(ctrl=states_ctrl)),
                ("ed", ed, {}),
                ("mmd", mmd_ave, {}),
            ]:
                state_results[f"state_{metric_name}"] = \
                    metric(states_pred, states_true, mask=None, **metric_kwargs)

            results.update(state_results)

    """
    GI metrics
    """
    if dataset_id in GI_DATASETS and not kwargs.skip_gi:

        # compute GI metrics for all fully covered doubles in test data
        if not kwargs.train_validation:
            # important: test environments have to be first argument
            true_gi_dataset = assemble_gi_dataset(
                test_datasets,
                train_datasets,
                remaining_datasets,
            )
            pred_gi_dataset = assemble_gi_dataset(
                (pred_test, intv_test, None),
                (pred_train, intv_train, None),
                (pred_remaining, intv_remaining, None) if remaining_datasets is not None else None,
            )
            ctrl = train_datasets.data_normalized[0]

            if true_gi_dataset is not None:
                # the first case occurs when prediction is same for all intvs
                # and pred_gi_dataset.data_normalized has shape [1, ..., d]
                assert pred_gi_dataset.intv is None or onp.allclose(true_gi_dataset.intv, pred_gi_dataset.intv)
                _, gi_score_metrics_per_env, gi_cache = run_all_gi_score_metrics(
                    true_gi_dataset.intv,
                    pred_gi_dataset.data_normalized,
                    {},
                    true_gi_dataset.data_normalized,
                    ctrl,
                    gi_mask=mask_from_seeding(seeding["gi_genes"], seeding["var_names"]),
                    thresholds=seeding["gi_thresholds"],
                    differential_expression_mask=true_gi_dataset.differential_expression_mask,
                )
                results.update(**{metr.replace("gi/metrics/", "gi_").replace("gim/metrics/", "gim_").replace("/", "_"): v
                                  for metr, v in gi_score_metrics_per_env.items()})

                aux["gi_scores_true"] = gi_cache["gi_scores_true"]
                aux["gi_scores_pred"] = gi_cache["gi_scores_pred"]

    """
    Causal metrics
    """
    if dataset_id in CAUSAL_DATASETS and "causal" in method_preds:
        # selected_mat = "state"
        selected_mat = "tfm"
        train_genes = sorted(list(set(env_idx for env_idx in seeding["train_perturbs"] if env_idx is not None)))

        # assemble ground truth parameters
        a_true = seeding["a"]
        a_causal_true = seeding["a_causal"]
        b_true = seeding["b"]
        c_true = seeding["c"][train_genes]

        # compute metrics
        anneal_times = sorted(method_preds["causal"][selected_mat].keys())
        anneal_time = anneal_times[-1]

        a_pred = method_preds["causal"][selected_mat][anneal_time]["a"]
        a_causal_pred = method_preds["causal"][selected_mat][anneal_time]["a_causal"]
        b_pred = method_preds["causal"][selected_mat][anneal_time]["b"]
        c_pred = method_preds["causal"][selected_mat][anneal_time]["c"]
        f0_pred = method_preds["causal"][selected_mat][anneal_time]["f0"]
        finf_pred = method_preds["causal"][selected_mat][anneal_time]["finf"]
        finf_ctrl_pred = method_preds["causal"][selected_mat][anneal_time]["finf_ctrl"]
        causal_mat_kwargs = method_preds["causal"][selected_mat][anneal_time]["causal_mat_kwargs"]

        causal_metrics = compute_causal_mat_metrics(true=a_causal_true, pred=a_causal_pred)

        for metr, v in causal_metrics.items():
            results[f"causal_{metr}"] = v

        # remember matrices for plotting
        aux["causal"] = dict(
            a_true=a_true,
            a_causal_true=a_causal_true,
            b_true=b_true,
            c_true=c_true,
            a_pred=a_pred,
            a_causal_pred=a_causal_pred,
            b_pred=b_pred,
            c_pred=c_pred,
            f0_pred=f0_pred,
            finf_pred=finf_pred,
            finf_ctrl_pred=finf_ctrl_pred,
            causal_mat_kwargs=causal_mat_kwargs,
        )

    # additional info
    if "walltime" in method_preds:
        results["walltime"] = method_preds["walltime"]

    if "theta" in method_preds:
        aux["theta"] = method_preds["theta"]

    # sanity checks
    for metr in results.keys():
        if metr not in METRICS_ORDERED:
            warnings.warn(f"Metric `{metr}` not configured in plot_config.py")

    # if nan in prediction, print to stdout to facilitate debugging
    if onp.isnan(pred_test).any():
        print(f"NaN: {path.stem}", flush=True)

    # return what is needed for plotting
    aux["samples"] = pred_test

    return results, aux


def make_summary(results_path, summary_path, data_paths, result_paths, kwargs):
    """
    Args:
        results_path
        summary_path
        data_paths (list): [paths to data csv's]
        result_paths (dict): {method: [paths to result csv's]}
        kwargs: if true, follows train validation procedure from methods.py and considers a train env as test
    """

    # method: dict of metrics
    results = defaultdict(lambda: defaultdict(list))
    samples = defaultdict(dict)
    aux = defaultdict(dict)

    """
    Load raw dataset once to speed up dataloading
    """
    dataset_ids = set([load_json(p)["id"] for p in data_paths])
    assert len(dataset_ids) == 1
    dataset_id = dataset_ids.pop()
    adata_raw = load_adata_raw(dataset_id)

    print("Loaded raw dataset.", flush=True)

    """
    Compute metrics sequentially by data seed to minimize dataloading and processing
    """
    for j, data_path in enumerate(sorted(data_paths, key=lambda p: safe_int(p.stem))):
        # skip if no method has predictions for this data path
        data_path_id = safe_int(data_path.stem)
        if not any([get_fold(p) == data_path_id for paths in result_paths.values() for p in paths]):
            continue

        # load dataset
        with print_timer("load_data"):
            print(f"data seed {data_path_id}/{len(data_paths)}", flush=True)
            train_datasets, test_datasets, remaining_datasets, data_config, seeding = \
                load_data_cached(data_path, adata_raw=adata_raw)

        # precompute differential expression signs
        ctrl = train_datasets.data_normalized[0]
        if test_datasets is not None:
            test_de_info = compute_differential_expression_sign_cached(test_datasets.data_normalized, ctrl, return_details=True)
        else:
            test_de_info = None

        # store info for plotting
        samples[data_path_id][REF] = ctrl
        samples[data_path_id][TRUE] = getattr(test_datasets, "data_normalized", None)

        aux[data_path_id]["intv_mask_train"] = train_datasets.intv
        aux[data_path_id]["intv_mask"] = getattr(test_datasets, "intv", None)
        aux[data_path_id]["differential_expression_mask"] = getattr(test_datasets, "differential_expression_mask", None)
        aux[data_path_id]["var_names"] = seeding["var_names"]
        aux[data_path_id]["ref_libsize"] = getattr(test_datasets, "ref_libsize", None)
        if "gi_thresholds" in seeding:
            aux[data_path_id]["gi_thresholds"] = seeding["gi_thresholds"]

        # iterate over all methods that have results for this data_path
        with print_timer("for loop: compute_metrics"):
            for n, (method, method_paths) in enumerate(result_paths.items()):

                # find result path of method with same data id as the dataset
                method_path_with_id = list(filter(lambda p: get_fold(p) == data_path_id, method_paths))
                assert len(method_path_with_id) == 1 or len(method_path_with_id) == 0, \
                    f"Multiple results found for method {method} and data {data_path_id}: {method_path_with_id}"
                if len(method_path_with_id) == 0:
                    # no results for this method and data path
                    continue
                method_path = method_path_with_id[0]

                # compute metrics
                results_method, aux_method = compute_metrics(
                    dataset_id=dataset_id,
                    seeding=seeding,
                    path=method_path,
                    train_datasets=train_datasets,
                    test_datasets=test_datasets,
                    remaining_datasets=remaining_datasets,
                    test_de_info=test_de_info,
                    kwargs=kwargs,
                )

                # aggregate results
                for metric, score in results_method.items():
                    results[method][metric].append(score)

                # store info for plotting
                samples[data_path_id][method] = aux_method["samples"]
                if "theta" in aux_method:
                    if "theta" not in aux[data_path_id]:
                        aux[data_path_id]["theta"] = dict()
                    aux[data_path_id]["theta"][method] = aux_method["theta"]

                if "gi_scores_pred" in aux_method:
                    if "gi_scores" not in aux[data_path_id]:
                        aux[data_path_id]["gi_scores"] = dict()
                    if "true" not in aux[data_path_id]["gi_scores"]:
                        aux[data_path_id]["gi_scores"]["true"] = aux_method["gi_scores_true"]
                    aux[data_path_id]["gi_scores"][method] = aux_method["gi_scores_pred"]

                if "causal" in aux_method:
                    if "causal" not in aux[data_path_id]:
                        aux[data_path_id]["causal"] = dict()
                    aux[data_path_id]["causal"][method] = aux_method["causal"]

    aux = {**aux}
    print("Computed metrics.", flush=True)


    """
    GI scores plotting
    """
    if dataset_id in GI_DATASETS:
        for data_path_id, aux_data in aux.items():
            if "gi_scores" in aux_data:
                assert "true" in aux_data["gi_scores"]

                if "gi_thresholds" in aux_data and aux_data["gi_thresholds"] is not None:
                    plot_gi_scores_comparison_by_methods(
                        aux_data["gi_scores"],
                        aux_data["gi_thresholds"],
                        save_path=summary_path / "plots" / "gi_methods",
                        filename=f"{data_path_id}",
                    )

                for method, gi_scores in aux_data["gi_scores"].items():
                    if method == "true":
                        continue

                    plot_gi_scores_comparison(
                        aux_data["gi_scores"]["true"],
                        gi_scores,
                        title=f"{method}",
                        save_path=summary_path / "plots" / "gi_true",
                        filename=f"{data_path_id}_{method}",
                        label_true=f"true",
                        label_pred=f"{method}",
                        thresholds=aux_data["gi_thresholds"],
                    )



    # dump all metrics for plotting
    df_name =  f"df__{summary_path.parents[0].name}__{results_path.stem.split('_', 1)[-1]}"
    df = dump_metrics_to_df(summary_path / df_name, results)


    """
    Metrics plotting and tables
    """

    if kwargs.train_validation:
        # method-wise summary for hparam calibration on training data
        table_summary(kwargs,
                      summary_path / "train_validation",
                      copy.deepcopy(df),
                      only_metrics=[ # first will be sorting criterion
                          "mmd_20",
                          "mmd",
                          "ed_20",
                          "ed",
                          "rmse_m_20",
                          "rmse_m",
                      ],
                      median_mode=True,
                      method_summaries=True)

        figure_summary(
            summary_path / "train_validation_plots",
            copy.deepcopy(df),
            None,
            None,
            train_validation=True,
            only_metrics=[
                "mmd_20",
            ],
        )


    else:
        # make all plots
        figure_summary(
            summary_path / "plots",
            copy.deepcopy(df),
            copy.deepcopy(samples),
            copy.deepcopy(aux),
            only_metrics=[
                "mmd",
                "ed",
                "rmse_m",
                "pearson_systema",
                #
                # top 20 DE
                "mmd_20",
                "ed_20",
                "rmse_m_20",
                "pearson_systema_20",
                #
                # HVG gene metrics
                "mmd_hvg_0-5",
                "rmse_m_hvg_0-5",
                "pearson_systema_hvg_0-5",
                #
                "mmd_hvg_5-20",
                "rmse_m_hvg_5-20",
                "pearson_systema_hvg_5-20",
                #
                "mmd_hvg_20-100",
                "rmse_m_hvg_20-100",
                "pearson_systema_hvg_20-100",
                #
                # DE classification metrics
                "classify_de_prec",
                "classify_de_recall",
                "classify_de_f1",
                #
                "classify_de_prec_20",
                "classify_de_recall_20",
                "classify_de_f1_20",
                #
                "classify_de_prec_hvg_0-5",
                "classify_de_recall_hvg_0-5",
                "classify_de_f1_hvg_0-5",
                #
                # top-k DE classification metrics
                "classify_topk_auroc_delta_20",
                "classify_topk_ap_delta_20",
                "classify_topk_precision_delta_20",
                "classify_topk_auroc_delta_100",
                "classify_topk_ap_delta_100",
                "classify_topk_precision_delta_100",
                #
                "classify_topk_auroc_pval_20",
                "classify_topk_ap_pval_20",
                "classify_topk_precision_pval_20",
                "classify_topk_auroc_pval_100",
                "classify_topk_ap_pval_100",
                "classify_topk_precision_pval_100",
            ],
        )

        figure_summary(
            summary_path / "state_plots",
            copy.deepcopy(df),
            None,
            None,
            only_metrics=[
                "state_rmse_m",
                "state_rmse_s",
                "state_bias_s",
                "state_pearson_de",
                "state_ed",
                "state_mmd",
            ],
        )

        # benchmark results tables
        table_summary(kwargs,
                      summary_path / "benchmark_20",
                      copy.deepcopy(df),
                      only_metrics=[
                          "mmd_20",
                          "ed_20",
                          "rmse_m_20",
                          "pearson_systema_20",
                      ])

        table_summary(kwargs,
                      summary_path / "benchmark",
                      copy.deepcopy(df),
                      only_metrics=[
                          "mmd",
                          "ed",
                          "rmse_m",
                          "pearson_systema",
                      ])

        # state-space metrics
        table_summary(kwargs,
                      summary_path / "state",
                      copy.deepcopy(df),
                      only_metrics=[
                          "state_rmse_m",
                          "state_rmse_s",
                          "state_bias_s",
                          "state_pearson_de",
                          "state_ed",
                          "state_mmd",
                      ])

        # GI metrics
        if dataset_id in GI_DATASETS:

            table_summary(kwargs,
                          summary_path / "gi",
                          copy.deepcopy(df),
                          only_metrics=[
                              "gi_se_mag",
                              "gi_se_linear_dcor",
                              "gi_se_dcor_ratio",
                              "gi_se_dcor",
                              "gi_pearson_mag",
                              "gi_pearson_linear_dcor",
                              "gi_pearson_dcor_ratio",
                              "gi_pearson_dcor",
                          ])

            table_summary(kwargs,
                          summary_path / "gi_cls",
                          copy.deepcopy(df),
                          only_metrics=[
                              "gi_cls_f1_syn",
                              "gi_cls_f1_add",
                              "gi_cls_f1_sup",
                              "gi_cls_f1_neo",
                              "gi_cls_f1_epi",
                              "gi_cls_f1_red",
                          ])

            table_summary(kwargs,
                          summary_path / "gi_benchmark",
                          copy.deepcopy(df),
                          only_metrics=[
                              "gim_syn_mmd",
                              "gim_add_mmd",
                              "gim_sup_mmd",
                              "gim_neo_mmd",
                              "gim_epi_mmd",
                              "gim_red_mmd",
                          ])

            table_summary(kwargs,
                          summary_path / "gi_benchmark_20",
                          copy.deepcopy(df),
                          only_metrics=[
                              "gim_syn_mmd_20",
                              "gim_add_mmd_20",
                              "gim_sup_mmd_20",
                              "gim_neo_mmd_20",
                              "gim_epi_mmd_20",
                              "gim_red_mmd_20",
                          ])

        # causal metrics
        if dataset_id in CAUSAL_DATASETS:
            table_summary(kwargs,
                          summary_path / "causal",
                          copy.deepcopy(df),
                          only_metrics=[
                              "causal_auroc",
                              "causal_pearson",
                              "causal_f1",
                          ])

            causal_visualization(
                kwargs,
                summary_path / "causal",
                copy.deepcopy(aux),
            )

    print("Finished successfully.", flush=True)

    # zip up directory for easy download and archiving
    archive_name = summary_path.parent.stem + "_" + summary_path.stem
    archive_path = summary_path.parent / archive_name
    compress_tar_gz(summary_path, archive_path)

    print("Stored archive successfully.", flush=True)



if __name__ == "__main__":
    """
    Runs plot call
    """


    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods_config_path", type=Path, required=True)
    parser.add_argument("--path_data", type=Path, required=True)
    parser.add_argument("--path_summary", type=Path, required=True)
    parser.add_argument("--path_results", type=Path, required=True)
    parser.add_argument("--train_validation", action="store_true")
    parser.add_argument("--skip_gi", action="store_true")
    parser.add_argument("--only_methods", nargs="+", type=str)
    parser.add_argument("--descr")
    parser.add_argument("--max_plots_env", type=int, default=10)
    parser.add_argument("--n_folds", type=int)
    kkwargs = parser.parse_args()

    # adjust configs based on only_methods
    onp.random.seed(kkwargs.seed)
    onp.set_printoptions(precision=5, suppress=True)
    methods_config = load_methods_config(kkwargs.methods_config_path, abspath=True,
                                         warn_if_grid=not kkwargs.train_validation,
                                         warn_if_not_grid=kkwargs.train_validation)

    if kkwargs.only_methods is not None:
        for k in list(methods_config.keys()):
            if not any([m in k for m in kkwargs.only_methods]):
                del methods_config[k]

    # retrieve data and results by filtering directories
    data_found, _, _ = ExperimentManager.list_data_found_and_expected(kkwargs.path_data,
                                                                      n_folds=kkwargs.n_folds,
                                                                      val=kkwargs.train_validation)


    _, results_found = ExperimentManager.list_results_found_and_expected(kkwargs.path_data,
                                                                         kkwargs.path_results,
                                                                         methods_config,
                                                                         val=kkwargs.train_validation,
                                                                         n_folds=kkwargs.n_folds,
                                                                         results_selection=True,
                                                                         verbose=True)

    any_results = any([len(v) for v in results_found.values()])
    if not any_results:
        print(f"No results found. (requested train validation: {kkwargs.train_validation})")
        exit()
    else:
        print("Using results:")
        for meth, res_paths in results_found.items():
            print(meth, flush=True)
            print(len(res_paths), flush=True)
            for r in sorted(res_paths, key=lambda p: safe_int(p.stem.rsplit("_", 1)[1])):
                print(r, flush=True)
            print(flush=True)

        print("Using data:")
        for p in data_found:
            print(p, flush=True)
        print(flush=True)

    with print_timer("summary.py"):
        make_summary(kkwargs.path_results, kkwargs.path_summary, data_found, results_found, kkwargs)
    print("Done.")

