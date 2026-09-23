import argparse

from lcd_clipr.definitions import PROJECT_DIR, SUBDIR_PLOTS, SUBDIR_RESULTS, SUBDIR_RESULTS_DUMP
from lcd_clipr.experiment.figures import ablation_figure_summary
from lcd_clipr.experiment.plot_config import MATPLOTLIB_RCPARAMS, \
    COLOR_OURS, COLOR_OURS_ABLATION_GROUP_1, COLOR_OURS_ABLATION_GROUP_2


# higher dpi for rasterized hatches (introduced to avoid laggy scrolling)
RASTERIZED_MATPLOTLIB_RCPARAMS = {
    **MATPLOTLIB_RCPARAMS,
    "savefig.dpi": 800,
}

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("dumps", nargs="+", type=str)
    kwargs = parser.parse_args()

    save_path = PROJECT_DIR / SUBDIR_PLOTS / "fig_2_supplementary"

    for dump in kwargs.dumps:

        df_or_df_path = PROJECT_DIR / SUBDIR_RESULTS / SUBDIR_RESULTS_DUMP / dump
        name = df_or_df_path.stem

        ablation_figure_summary(
            save_path / f"{name}_methods",
            df_or_df_path,
            methods=[
                ["ours-mlp"],
                ["ours-mlp__no-unit-scaling", "ours-mlp__no-temp"],
                ["ours-mlp__drift-mult-direct", "ours-mlp__drift-simple"],
            ],
            methods_config={
                "ours-mlp": (
                    "LCD (ours)",
                    COLOR_OURS,
                    None,
                ),
                "ours-mlp__no-unit-scaling": (
                    "No gene-specific scaling\nin $\\mu_g$ (Eq. 2)",
                    COLOR_OURS_ABLATION_GROUP_1,
                    None,
                ),
                "ours-mlp__no-temp": (
                    "No loss temperature in\ninference step 1 (Eq. 11)",
                    COLOR_OURS_ABLATION_GROUP_1,
                    "////////////",
                ),
                "ours-mlp__drift-mult-direct": (
                    "Multiplicative perturbation\nof hidden state (Fig. 1B)",
                    COLOR_OURS_ABLATION_GROUP_2,
                    None,
                ),
                "ours-mlp__drift-simple": (
                    "Naive parameterization\n$f = h$ (Eq. 14)",
                    COLOR_OURS_ABLATION_GROUP_2,
                    "////////////",
                ),
            },
            plot_legend=True,
            ylim_override={
                "rmse_m_20": (0.0, 0.8),
            },
            only_metrics=[
                "mmd_20",
                "rmse_m_20",
                "pearson_systema",
                #
                "rmse_m",
                "mmd",
                "pearson_systema_20",
                "mmd_hvg_0-5",
                "mmd_hvg_5-20",
                "mmd_hvg_20-100",
                f"classify_topk_auroc_delta_100",
                f"classify_topk_ap_pval_100"
            ],
            rcparams=RASTERIZED_MATPLOTLIB_RCPARAMS,
        )



    return

if __name__ == "__main__":
    main()