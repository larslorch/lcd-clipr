import argparse
from pathlib import Path
import pandas as pd
from joblib import load
from pprint import pprint
from collections import defaultdict

from lcd_clipr.definitions import PROJECT_DIR, SUBDIR_PLOTS, SUBDIR_CLIPR_DUMP
from lcd_clipr.experiment.post_analyze_causal import (
    post_process_causal_analysis,
    BOOTSTRAP_CI_SAMPLES,
    comprehensive_implication_test_cached,
)
from lcd_clipr.experiment.implication import (
    plot_implication_conditional_probability,
    plot_implication_association_measures,
)


def main():

    """
    Analysis and visualization of CLIPR results
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("clipr_dump_folders", nargs="+", type=Path)
    kwargs = parser.parse_args()

    save_path = PROJECT_DIR / SUBDIR_PLOTS / "fig_4"
    single = len(kwargs.clipr_dump_folders) == 1

    aggregated_results = defaultdict(list)

    for experiment in kwargs.clipr_dump_folders:

        # load CLIPR results
        folder = PROJECT_DIR / SUBDIR_PLOTS / SUBDIR_CLIPR_DUMP / experiment
        joblibs = list(folder.glob("*.joblib"))
        assert len(joblibs) == 1
        causal_analysis = load(joblibs[0])

        save_folder = save_path / experiment
        save_folder.mkdir(parents=True, exist_ok=True)

        print("Loaded CLIPR results from", joblibs[0])

        # analyze and visualize results
        results = post_process_causal_analysis(
            save_folder,
            causal_analysis,
            plot_implication=True,
            plot_cluster_enrichr=single,
            plot_other_mats=single,
        )
        print("Saved CLIPR analysis to", save_folder)

        # aggregate results from analysis when running multiple folders
        if "implication_heldout_predictions" in results:
            aggregated_results["implication_heldout_predictions"].append(results["implication_heldout_predictions"])

        if "gene_names" in results:
            if "gene_names" in aggregated_results:
                assert aggregated_results["gene_names"] == results["gene_names"]
            else:
                aggregated_results["gene_names"] = results["gene_names"]

    if "implication_heldout_predictions" in aggregated_results:
        # Build aggregate implication_heldout_predictions DataFrame
        rows = []
        seen_targets = defaultdict(set)
        for predictions_dict in aggregated_results["implication_heldout_predictions"]:
            for quantile, data in predictions_dict.items():
                assert not seen_targets[quantile].intersection(data["targets"]), "targets overlap between heldout folds"
                seen_targets[quantile].update(data["targets"])
                for pred, true, pred_in_test_targets in zip(data["pred"], data["true"], data['pred_in_test_targets']):
                    rows.append({
                        "pred": pred,
                        "true": true,
                        "pred_in_test_targets": pred_in_test_targets,
                        "quantile": quantile,
                    })

        df = pd.DataFrame(rows)
        all_results_precision = {}

        for quantile in sorted(df["quantile"].unique()):
            df_quantile = df[df["quantile"] == quantile]
            results_df_precision = comprehensive_implication_test_cached(
                df_quantile["pred"].values,
                df_quantile["true"].values,
                bootstrap_ci_samples=BOOTSTRAP_CI_SAMPLES,
            )
            all_results_precision[quantile] = results_df_precision["value"]

            print(f"Quantile:                   {quantile}")
            print(f"Pred edges:                 {df_quantile['pred'].sum()}")
            print(f"Pred edges in test targets: {df_quantile['pred_in_test_targets'].sum()}")
            print()

        # Aggregate results across quantiles
        aggregated_precision_df = pd.DataFrame(all_results_precision)
        aggregated_precision_df.to_csv(save_path / "implication_heldout_all_quantiles.csv")
        pprint(aggregated_precision_df)

        plot_implication_conditional_probability(
            aggregated_precision_df,
            save_path / "implication_heldout_conditional_prob",
        )
        plot_implication_association_measures(
            aggregated_precision_df,
            save_path / "implication_heldout_association",
        )

        selected_quantiles = [0.99950, 0.99970]
        plot_implication_conditional_probability(
            aggregated_precision_df.loc[:, selected_quantiles],
            save_path / "implication_heldout_conditional_prob",
            show_yaxis=False,
        )
        plot_implication_association_measures(
            aggregated_precision_df.loc[:, selected_quantiles],
            save_path / "implication_heldout_association",
            show_yaxis=False,
        )

    return


if __name__ == "__main__":
    main()