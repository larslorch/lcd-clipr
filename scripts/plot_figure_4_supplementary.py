import argparse
from pathlib import Path
from joblib import load
import numpy as onp

from lcd_clipr.definitions import PROJECT_DIR, SUBDIR_PLOTS, SUBDIR_CLIPR_DUMP
from lcd_clipr.experiment.analyze_causal import linear_causal_params
from lcd_clipr.experiment.plot_analysis import plot_causal_l_curve
from lcd_clipr.experiment.plot_causal import compute_ordering_and_linkage, select_top_genes_for_visualization, compute_tight_clusters_at_height


def main():

    onp.set_printoptions(precision=3, suppress=True, linewidth=200)

    parser = argparse.ArgumentParser()
    parser.add_argument("clipr_dump_folder", type=Path)
    kwargs = parser.parse_args()

    """
    Load matrices
    """
    load_path = PROJECT_DIR / SUBDIR_PLOTS / SUBDIR_CLIPR_DUMP / kwargs.clipr_dump_folder
    save_path_folder = PROJECT_DIR / SUBDIR_PLOTS / "fig_4_supplementary"

    joblibs = list(load_path.glob("*.joblib"))
    assert len(joblibs) == 1
    causal_analysis = load(joblibs[0])

    save_path = save_path_folder / load_path.stem

    pred_causal = causal_analysis["pred_causal"]
    anneal_times = sorted(pred_causal.keys())
    anneal_time = anneal_times[-1]
    pred_causal_last = pred_causal[anneal_time]

    # extract matrices
    a = pred_causal_last['a']
    a_causal = pred_causal_last['a_causal']
    f0 = pred_causal_last['f0']
    finf = pred_causal_last['finf']
    finf_ctrl = pred_causal_last['finf_ctrl']
    causal_mat_kwargs = pred_causal_last["causal_mat_kwargs"]

    env_mask = pred_causal_last['env_mask']
    targets = env_mask.any(0)

    check = linear_causal_params(
        f0=f0,
        finf=finf,
        finf_ctrl=finf_ctrl,
        **causal_mat_kwargs,
    )
    assert onp.allclose(check["a"], a, atol=1e-4)

    # ordering and clustering settings from post_causal_analysis.py
    ordering, linkage_matrix, _ = compute_ordering_and_linkage(a_causal)
    cluster_labels = compute_tight_clusters_at_height(
        linkage_matrix,
        height=0.175,
        min_cluster_size=5,
    )
    top_mask, _, top_ordering = select_top_genes_for_visualization(
        a_causal,
        targets,
        ordering=ordering,
        labels=cluster_labels,
        top_shown=(cluster_labels != -1).sum(),
    )

    """
    Plot analysis
    """

    tikhonov_range = onp.logspace(-5, onp.log10(3), 50) # [1e-5, 3]
    tikhonov_labels = {
        1e-4: "$10^{-4}$",
        1e-3: "$10^{-3}$",
        1e-2: "$10^{-2}$",
        1e-1: "$10^{-1}$",
        1e0: "$1$",
    }

    plot_causal_l_curve(
        save_path / "causal_l_curve",
        f0=f0,
        finf=finf,
        finf_ctrl=finf_ctrl,
        tikhonov_range=tikhonov_range,
        tikhonov_labels=tikhonov_labels,
        mask=top_mask,
        ordering=top_ordering,
    )

    return


if __name__ == "__main__":
    main()