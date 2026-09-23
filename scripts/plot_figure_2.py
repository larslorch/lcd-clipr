import argparse

from lcd_clipr.experiment.figures import figure_summary, comparison_figure_summary, figure_summary_side_by_side
from lcd_clipr.experiment.plot_config import METHODS_COLORS, MATPLOTLIB_RCPARAMS
from lcd_clipr.experiment.plot_differential_expression import plot_differential_expressions

from lcd_clipr.definitions import PROJECT_DIR, SUBDIR_RESULTS, SUBDIR_PLOTS, SUBDIR_RESULTS_DUMP
from lcd_clipr.utils.parse import load_json_zip, load_data
from lcd_clipr.utils.cache import init_cache

# higher dpi for rasterized hatches (introduced to avoid laggy scrolling)
RASTERIZED_MATPLOTLIB_RCPARAMS = {
    **MATPLOTLIB_RCPARAMS,
    "savefig.dpi": 800,
}

# memory = init_cache(ignore=True)
memory = init_cache(ignore=False)


@memory.cache
def load_data_cached(path):
    return load_data(path)


@memory.cache
def load_json_zip_cached(path):
    return load_json_zip(path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--method", type=str, default="ours-mlp")
    parser.add_argument("--plot_seeds", nargs="*", type=int, default=[])
    kwargs = parser.parse_args()

    save_path = PROJECT_DIR / SUBDIR_PLOTS / "fig_2"

    """
    Prediction visualization
    """
    base_load_path = PROJECT_DIR / SUBDIR_RESULTS
    load_path_data = base_load_path / kwargs.experiment / kwargs.data
    load_path_preds = base_load_path / kwargs.experiment / kwargs.predictions
    save_path_visualization = save_path / load_path_preds.parent.stem / load_path_preds.stem

    for j, seed in enumerate(kwargs.plot_seeds):
        # load true
        train_datasets, test_datasets, remaining_datasets, data_config, data_seeding = \
            load_data_cached(load_path_data / f"{seed}")

        print("Loaded data from", load_path_data / f"{seed}")

        # load preds
        preds = load_json_zip_cached(load_path_preds / f"{kwargs.method}_{seed}.zip")

        print("Loaded predictions from", load_path_preds / f"{kwargs.method}_{seed}.zip")

        # plot
        plot_differential_expressions(
            save_path_visualization,
            preds,
            train_datasets,
            test_datasets,
            remaining_datasets,
            data_seeding,
            plot_legend=j == 0,
            label_pred="LCD (ours)",
            color_pred=METHODS_COLORS["ours-mlp"],
            cols=2 + 20,
            wide_format=False,
            rcparams=RASTERIZED_MATPLOTLIB_RCPARAMS,
        )


    """
    Results plots
    """
    prediction_id = kwargs.predictions.removeprefix('predictions_')
    name = f"df__{kwargs.experiment}__{prediction_id}"
    dump_path = PROJECT_DIR / SUBDIR_RESULTS / SUBDIR_RESULTS_DUMP / f"{name}.csv"

    # side-by-side metrics
    for metric, metric_ylabel in [
        ("auroc", "AUROC"),
        ("ap", "Average precision"),
    ]:
        figure_summary_side_by_side(
            save_path / f"{name}_side_by_side",
            dump_path,
            ytitle=metric_ylabel,
            side_by_side_metrics=[f"classify_topk_{metric}_delta_100", f"classify_topk_{metric}_pval_100"],
            box_hatches=["////////////", None],
            metric_labels=["by mean\nlog-fold change", "by significance of\ntest statistic"],
            plot_legend=True,
            rcparams=RASTERIZED_MATPLOTLIB_RCPARAMS,
        )

    # boxplots for all metrics and methods
    figure_summary(
        save_path / f"{name}_hvg",
        dump_path,
        None,
        None,
        figsize_key_boxplot="metrics_boxplot_hvg",
        plot_legend=True,
        only_metrics=[
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
    ])

    figure_summary(
        save_path / f"{name}_short",
        dump_path,
        None,
        None,
        figsize_key_boxplot="metrics_boxplot_short",
        plot_legend=True,
        only_metrics=[
        "mmd_20",
        "mmd",
        "rmse_m_20",
        "rmse_m",
        "pearson_systema_20",
        "pearson_systema",
    ])

    figure_summary(
        save_path / name,
        dump_path,
        None,
        None,
        plot_legend=True,
        only_metrics=[
        "mmd_20",
        "mmd_hvg_0-5",
        "mmd_hvg_5-20",
        "mmd_hvg_20-100",
    ])


    if "norman" in dump_path.stem:
        comparison_figure_summary(
            save_path / name / "gim_mmd_20",
            dump_path,
            [
                "gim_syn_mmd_20",
                "gim_add_mmd_20",
                "gim_sup_mmd_20",
                "gim_epi_mmd_20",
                "gim_neo_mmd_20",
                "gim_red_mmd_20",
            ],
            [
                "salt",
                "ours-mlp",
            ],
            plot_legend=True,
            title="",
            ylabel="MMD (top 20)",
        )

    print("Generated metrics plots and saved to", save_path / name)

    return


if __name__ == "__main__":
    main()
