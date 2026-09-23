import argparse
import re
import cmcrameri
import numpy as onp

from pathlib import Path
from collections import defaultdict
from lcd_clipr.experiment.figures import causal_figure_summary
from lcd_clipr.experiment.dump import load_metrics_from_df
from lcd_clipr.utils.parse import load_json, load_json_zip
from lcd_clipr.experiment.plot_causal import plot_mat, offdiag
from lcd_clipr.utils.cache import init_cache
from lcd_clipr.definitions import PROJECT_DIR, SUBDIR_PLOTS, SUBDIR_RESULTS, SUBDIR_RESULTS_DUMP


# memory = init_cache(ignore=True)
memory = init_cache(ignore=False)


@memory.cache
def load_json_cached(path):
    return load_json(path)


@memory.cache
def load_json_zip_cached(path):
    return load_json_zip(path)


def main():
    onp.set_printoptions(precision=2, suppress=True)
    rng = onp.random.default_rng(0)

    parser = argparse.ArgumentParser()

    parser.add_argument("experiments", nargs="+", type=str)
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--method", type=str, default="ours-mlp")

    parser.add_argument("--plot_experiment", type=str, default=None)
    parser.add_argument("--plot_seeds", nargs="*", type=int, default=[])
    parser.add_argument("--file_type", type=str, default="pdf")
    kwargs = parser.parse_args()

    kwargs.save_path = PROJECT_DIR / SUBDIR_PLOTS / "fig_3"
    base_load_path =  PROJECT_DIR / SUBDIR_RESULTS

    cmap_a_causal = cmcrameri.cm.berlin_r
    cmap_c = cmcrameri.cm.lisbon_r
    figsize_key = "adjacency_matrix_v3"

    """
    Visualization of A and C matrices
    """
    for seed in kwargs.plot_seeds:
        data_id = kwargs.predictions.split('_')[0]
        load_path_data = base_load_path / kwargs.plot_experiment / f"data_{data_id}" / f"{seed}.json"
        load_path_preds = base_load_path / kwargs.plot_experiment / f"predictions_{kwargs.predictions}" / f"{kwargs.method}_{seed}.zip"
        save_path_causal = kwargs.save_path / kwargs.plot_experiment

        # load true
        seeding = load_json_cached(load_path_data)
        a_causal_true = seeding["a_causal"]
        targets = onp.sort([tar for tar in seeding["train_perturbs"] if tar is not None])
        c_true = seeding["c"][targets]

        # load preds
        method_preds = load_json_zip_cached(load_path_preds)
        causal_preds = method_preds["causal"]["tfm"]["29"]
        a_causal_pred = causal_preds["a_causal"]
        c_pred = causal_preds["c"]

        # plot subselections
        out_degree = onp.abs(a_causal_true).sum(1)
        top_by_out_degree = onp.argsort(-out_degree)
        top = 150
        perturbs_shown = 80

        # drop perturbations targeting genes that are masked
        mask = onp.sort(top_by_out_degree[:top] if top is not None else top_by_out_degree)
        target_masked = [tar for tar in targets if tar not in mask]
        c_mask = onp.array([j for j, tar in enumerate(targets) if tar not in target_masked])
        if perturbs_shown is not None and perturbs_shown < len(c_mask):
            c_mask = onp.sort(rng.choice(c_mask, size=perturbs_shown, replace=False))

        # mask
        a_pred_mask = a_causal_pred[mask][:, mask]
        a_true_mask = a_causal_true[mask][:, mask]
        c_pred_mask = c_pred[c_mask][:, mask]
        c_true_mask = c_true[c_mask][:, mask]

        # unify scales
        vquantile = 0.99
        vmin = onp.nanquantile(offdiag(a_pred_mask), 1 - vquantile)
        vmax = onp.nanquantile(offdiag(a_pred_mask), vquantile)
        vmin_c = onp.min(c_true_mask)
        vmax_c = onp.max(c_true_mask)

        suffix = f"_top_{top}" if top is not None else ""

        plot_mat(
            save_path_causal / f"a_causal_{kwargs.method}_{seed}_{suffix}",
            a_pred_mask,
            cmap=cmap_a_causal,
            file_type=kwargs.file_type,
            vmin=vmin,
            vmax=vmax,
            figsize_key=figsize_key,
            plot_colorbar_inset=True,
            colorbar_inset_kwargs={
                'ticks': [-2, 0, 2],
                'long_side': 0.5,
                'ticklocation': 'bottom',
            },
            dpi=1000,
        )
        plot_mat(
            save_path_causal / f"a_causal_{kwargs.method}_{seed}_{suffix}_true",
            a_true_mask,
            cmap=cmap_a_causal,
            file_type=kwargs.file_type,
            vmin=vmin,
            vmax=vmax,
            figsize_key=figsize_key,
            dpi=1000,
        )

        plot_mat(
            save_path_causal / f"c_{kwargs.method}_{seed}_{suffix}",
            c_pred_mask,
            cmap=cmap_c,
            file_type=kwargs.file_type,
            vmin=vmin_c,
            vmax=vmax_c,
            figsize_key=figsize_key,
            dpi=1000,
        )
        plot_mat(
            save_path_causal / f"c_{kwargs.method}_{seed}_{suffix}_true",
            c_true_mask,
            cmap=cmap_c,
            file_type=kwargs.file_type,
            vmin=vmin_c,
            vmax=vmax_c,
            figsize_key=figsize_key,
            plot_colorbar_inset=True,
            colorbar_inset_kwargs={
                'ticks': [-10, 0, 10],
                'long_side': 0.5,
                'ticklocation': 'bottom',
            },
            dpi=1000,
        )

        print("Saved visualization to", save_path_causal)

    """
    Results aggregation
    """
    base_dump_path = PROJECT_DIR / SUBDIR_RESULTS / SUBDIR_RESULTS_DUMP

    # dump csvs are flat files named "<prefix>_df__<experiment>__<predictions>.csv"
    kwargs.dump_paths = [
        base_dump_path / f"{Path(exp).parent}_df__{Path(exp).name}__{kwargs.predictions}.csv"
        for exp in kwargs.experiments
    ]

    df_dict = defaultdict(dict)

    for dump_path in kwargs.dump_paths:
        dump_path = Path(dump_path) if not isinstance(dump_path, Path) else dump_path

        # extract number of vars and perturbations from experiment name
        match = re.search(r"linear-(\d+)-p=(\d+)", dump_path.stem)
        d, perturbs = int(match.group(1)), int(match.group(2))
        df_dict[d][perturbs] = load_metrics_from_df(dump_path)

    causal_figure_summary(
        kwargs,
        kwargs.save_path / f"merged_{Path(kwargs.dump_paths[-1]).stem}",
        df_dict,
        plot_legend=True,
        only_metrics=[
            "causal_f1",
            "causal_auroc",
            "causal_pearson",
        ])

    print("Saved merged results to", kwargs.save_path / f"merged_{Path(kwargs.dump_paths[-1]).stem}")

    return

if __name__ == "__main__":
    main()