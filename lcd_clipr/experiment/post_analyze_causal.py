import numpy as onp
import pandas as pd
from pprint import pprint

from lcd_clipr.experiment.plot_config import *
from lcd_clipr.experiment.plot_annotation import ENRICHR_ANNOTATION

import matplotlib.pyplot as plt

from lcd_clipr.experiment.plot_causal import (
    plot_mat,
    plot_singular_value_spectrum,
    plot_causal_graph,
    compute_ordering_and_linkage,
    compute_tight_clusters_at_height,
    select_top_genes_for_visualization,
)
from lcd_clipr.experiment.plot_gsea import (
    generate_cluster_enrichr_table,
    plot_cluster_enrichr_from_table,
    is_gene_tf,
)
from lcd_clipr.core import normalizer
from lcd_clipr.experiment.implication import (
    comprehensive_implication_test,
    plot_implication_conditional_probability,
    plot_implication_association_measures,
    save_causal_links_csv,
)
from lcd_clipr.experiment.analyze_causal import tikhonov_pinv
from lcd_clipr.noise_models.likelihood import link_state
from lcd_clipr.utils.cache import init_cache

memory = init_cache()


BOOTSTRAP_CI_SAMPLES = 1000


@memory.cache
def comprehensive_implication_test_cached(pred_causal_effects, true_causal_effects, bootstrap_ci_samples):
    return comprehensive_implication_test(
        pred_causal_effects,
        true_causal_effects,
        bootstrap_ci_samples=bootstrap_ci_samples,
    )


def post_process_causal_analysis(
    folder,
    causal_analysis,
    *,
    plot_cluster_enrichr=False,
    plot_implication=False,
    plot_other_mats=False,
    cluster_cut_height=0.175,
    fdr_cutoff=0.01,
    top_n_per_cluster=5,
    min_size=5,
    rcparams=MATPLOTLIB_RCPARAMS,
    figsize_key_implication="implication_analysis",
    figsize_key_enrichr="causal_analysis_dotplot",
    figsize_key_causal="causal_matrix_fig4",
    figsize_key_causal_components="si_causal_small",
    figsize_key_causal_gsea_big="si_causal_big",
    figsize_key_chord_graph="causal_analysis_chord",
):
    plt.rcParams.update(rcparams)

    pred_causal = causal_analysis["pred_causal"]
    gene_names = causal_analysis["var_names"]
    gene_to_ensembl = causal_analysis["gene_to_ensembl"]
    ensembl_names = [gene_to_ensembl[gene] for gene in gene_names]
    is_tf = is_gene_tf(gene_names)
    assert onp.allclose(is_tf, is_gene_tf(ensembl_names, use_ensembl=True))

    results = dict(gene_names=gene_names)

    def to_rate(z):
        return causal_analysis["rate_unit_scaling"] * link_state(
            z,
            temp=causal_analysis["rate_link_temp"],
            shift=causal_analysis["rate_link_shift"],
        )

    anneal_times = sorted(pred_causal.keys())
    anneal_time = anneal_times[-1]
    pred_causal_last = pred_causal[anneal_time]

    a_causal = pred_causal_last['a_causal']
    f0 = pred_causal_last['f0']
    finf = pred_causal_last['finf']
    finf_ctrl = pred_causal_last['finf_ctrl']
    c_mat = pred_causal_last['c']
    n_envs, d = c_mat.shape

    env_mask = pred_causal_last['env_mask']
    tikhonov = pred_causal_last["causal_mat_kwargs"]['tikhonov']
    targets = env_mask.any(0)

    de_mat = causal_analysis["de_mat"]
    de_pred = normalizer(to_rate(finf)) - normalizer(to_rate(finf_ctrl))

    de_signs_single_gene = causal_analysis["de_signs_single_gene"]
    de_targets_single_gene = causal_analysis["de_targets_single_gene"]
    n_envs_single_gene = de_signs_single_gene.shape[0]

    test_de_signs_single_gene = causal_analysis["test_de_signs_single_gene"]
    test_de_targets_single_gene = causal_analysis["test_de_targets_single_gene"]

    assert de_mat.shape == f0.shape == finf.shape == env_mask.shape == c_mat.shape
    assert a_causal.shape == (d, d)

    """
    Causal matrices
    """
    cmap_causal = cmcrameri.cm.berlin_r
    cmap_de = cmcrameri.cm.vanimo

    # Compute ordering and linkage once for the a_causal matrix
    ordering, linkage_matrix, _ = compute_ordering_and_linkage(a_causal)

    cluster_labels = compute_tight_clusters_at_height(
        linkage_matrix,
        height=cluster_cut_height,
        min_cluster_size=min_size,
    )
    assert cluster_labels is not None

    # Cluster coloring for plots
    cluster_cmap = plt.cm.tab20
    cluster_coloring = {}
    for cluster in onp.sort(onp.unique(cluster_labels)):
        if cluster != -1 and cluster not in cluster_coloring:
            cluster_coloring[cluster] = cluster_cmap(len(cluster_coloring) % 20)

    """
    Implications from differential expression:
    Direct causal effect from A to B  ==> Differential expression of B after perturbing A
    """

    if plot_implication:

        a_causal_nan_diag = onp.array(a_causal).copy()
        a_causal_nan_diag[onp.diag_indices(d)] = onp.nan

        # select the masking threshold on a validation split to avoid selection bias
        quantiles = [0.9995, 0.9997, 0.9999, 0.99995, 0.99997]

        all_results_precision = {}
        implication_heldout_predictions = {}

        for causal_effect_quantile_threshold in quantiles:
            descr = f"{causal_effect_quantile_threshold}".replace(".", "_")

            causal_graph = onp.zeros((d, d)) * onp.nan
            de_graph = onp.zeros((d, d)) * onp.nan

            if test_de_targets_single_gene is not None:
                effect_threshold_mask = ~test_de_targets_single_gene.any(0)
            else:
                effect_threshold_mask = onp.ones(d, dtype=bool)

            effect_threshold = onp.nanquantile(
                onp.abs(a_causal_nan_diag[effect_threshold_mask]).flatten(),
                causal_effect_quantile_threshold,
            )

            pred_causal_effects = []
            true_causal_effects = []

            for de_sign, de_target in zip(de_signs_single_gene, de_targets_single_gene):
                gene_target = onp.where(de_target)[0][0]
                effect_pred = onp.abs(a_causal[gene_target]) > effect_threshold
                effect_true = ~onp.isclose(de_sign, 0)

                pred_causal_effects.append(onp.delete(effect_pred, gene_target))
                true_causal_effects.append(onp.delete(effect_true, gene_target))

                # Record predicted edges
                assert onp.isnan(de_graph[gene_target]).all()
                assert onp.isnan(causal_graph[gene_target]).all()
                de_graph[gene_target] = effect_true
                causal_graph[gene_target] = effect_pred * a_causal[gene_target]
                causal_graph[gene_target, gene_target] = 0

            pred_causal_effects = onp.concatenate(pred_causal_effects)
            true_causal_effects = onp.concatenate(true_causal_effects)

            assert len(pred_causal_effects) == n_envs_single_gene * (d - 1)

            # Test for implication: causal effect ==> DE
            results_df_precision = comprehensive_implication_test_cached(
                pred_causal_effects,
                true_causal_effects,
                bootstrap_ci_samples=BOOTSTRAP_CI_SAMPLES,
            )
            all_results_precision[causal_effect_quantile_threshold] = results_df_precision["value"]

            # Visualize causal graph
            active_nodes = causal_graph.any(axis=0) | causal_graph.any(axis=1)
            if active_nodes.sum() < 100:
                active_causal_graph = causal_graph[active_nodes][:, active_nodes]
                active_de_graph = de_graph[active_nodes][:, active_nodes]
                active_gene_names = [gene_names[i] for i, active in enumerate(active_nodes) if active]
                active_cluster_labels = cluster_labels[active_nodes]

                # Sort by cluster label first, then alphabetically within each cluster
                order_tuples = sorted([
                    (active_cluster_labels[i], active_gene_names[i], i) for i in range(len(active_gene_names))
                ])
                cluster_then_alpha_order = onp.array([t[2] for t in order_tuples])

                active_causal_graph = active_causal_graph[cluster_then_alpha_order][:, cluster_then_alpha_order]
                active_de_graph = active_de_graph[cluster_then_alpha_order][:, cluster_then_alpha_order]
                active_gene_names = [active_gene_names[i] for i in cluster_then_alpha_order]
                active_cluster_labels = active_cluster_labels[cluster_then_alpha_order]

                assert not onp.isnan(active_causal_graph).any()
                assert not onp.isnan(active_de_graph).any()

                plot_causal_graph(
                    folder / f"causal_graph_implication__{descr}",
                    active_causal_graph,
                    active_gene_names,
                    cluster_labels=active_cluster_labels,
                    cluster_colors=cluster_coloring,
                    edge_colors=active_de_graph,
                    rcparams=rcparams,
                    figsize_key=figsize_key_chord_graph,
                )
                save_causal_links_csv(
                    active_causal_graph,
                    active_gene_names,
                    folder / f"causal_links_implication__{descr}.csv",
                )

            # Evaluate on held-out targets
            # effect threshold based on all non-heldout target effects
            if test_de_signs_single_gene is not None and test_de_targets_single_gene is not None:
                effect_threshold_val = onp.nanquantile(
                    onp.abs(a_causal_nan_diag[~test_de_targets_single_gene.any(0)]).flatten(),
                    causal_effect_quantile_threshold,
                )

                pred_test_causal_effects = []
                true_test_causal_effects = []
                pred_test_causal_effects_in_test_targets = []
                test_targets = []

                for j, (de_sign, de_target) in enumerate(zip(test_de_signs_single_gene, test_de_targets_single_gene)):
                    assert onp.isclose(de_target.sum(), 1)
                    gene_target = onp.where(de_target)[0][0]

                    effect_pred = onp.abs(a_causal[gene_target]) > effect_threshold_val
                    effect_true = ~onp.isclose(de_sign, 0)

                    # 1 iff predicted effect and in test targets
                    effect_pred_in_test_targets = effect_pred & test_de_targets_single_gene.any(0)

                    pred_test_causal_effects.append(onp.delete(effect_pred, gene_target))
                    true_test_causal_effects.append(onp.delete(effect_true, gene_target))
                    pred_test_causal_effects_in_test_targets.append(onp.delete(effect_pred_in_test_targets, gene_target))
                    test_targets.append(gene_target)

                    # Record predicted edges
                    assert onp.isnan(de_graph[gene_target]).all()
                    assert onp.isnan(causal_graph[gene_target]).all()

                    de_graph[gene_target] = effect_true
                    causal_graph[gene_target] = effect_pred * a_causal[gene_target]
                    causal_graph[gene_target, gene_target] = 0

                pred_test_causal_effects = onp.concatenate(pred_test_causal_effects)
                true_test_causal_effects = onp.concatenate(true_test_causal_effects)
                pred_test_causal_effects_in_test_targets = onp.concatenate(pred_test_causal_effects_in_test_targets)

                implication_heldout_predictions[causal_effect_quantile_threshold] = dict(
                    pred=pred_test_causal_effects,
                    true=true_test_causal_effects,
                    pred_in_test_targets=pred_test_causal_effects_in_test_targets,
                    targets=test_targets,
                )

        # Aggregate results across quantiles
        aggregated_precision_df = pd.DataFrame(all_results_precision)
        aggregated_precision_df.to_csv(folder / "implication_all_quantiles.csv")
        pprint(aggregated_precision_df)

        if implication_heldout_predictions:
            results["implication_heldout_predictions"] = implication_heldout_predictions

        # Plot implication analysis results
        plot_implication_conditional_probability(
            aggregated_precision_df,
            folder / "implication_conditional_prob",
            plot_legend=True,
            rcparams=rcparams,
            figsize_key=figsize_key_implication,
        )
        plot_implication_association_measures(
            aggregated_precision_df,
            folder / "implication_association",
            plot_legend=True,
            rcparams=rcparams,
            figsize_key=figsize_key_implication,
        )

    """
    GSEA of causal matrix
    """
    label_lookup = {term: annot[0] for term, annot in ENRICHR_ANNOTATION.items()}
    gene_sets = {
        "c5.go.v2025.1.Hs.symbols.gmt": ("Gene_ontology", "Greys"),
        "c2.cp.v2025.1.Hs.symbols.gmt": ("Canonical_pathways", "Greys"),
    }

    if plot_cluster_enrichr:
        gseadescr = f"{cluster_cut_height:.3f}_top{top_n_per_cluster}_p{fdr_cutoff:.5f}_min{min_size}".replace('.', '_')

        plot_mat(
            folder / f"a_causal_{gseadescr}_big",
            a_causal,
            ordering=ordering,
            targets=cluster_labels,
            cluster_colors=cluster_coloring,
            figsize_key=figsize_key_causal_gsea_big,
            cmap=cmap_causal,
            plot_colorbar_inset=True,
            colorbar_inset_kwargs={
                'ticks': [-3, 0, 3],
                'long_side': 0.7,
                'ticklocation': 'bottom',
            },
            rcparams=rcparams,
        )

        # subselections for small figure visualizations
        top_mask, top_mask_c, top_ordering = select_top_genes_for_visualization(
            a_causal,
            targets,
            ordering=ordering,
            labels=cluster_labels,
            top_shown=(cluster_labels != -1).sum(),
        )

        vquantile_top = 0.95
        plot_mat(
            folder / f"top_a_causal_{gseadescr}",
            a_causal[top_mask][:, top_mask],
            ordering=top_ordering,
            targets=cluster_labels[top_mask],
            cluster_colors=cluster_coloring,
            figsize_key=figsize_key_causal,
            cmap=cmap_causal,
            plot_colorbar_inset=True,
            colorbar_inset_kwargs={'ticks': [-3, 0, 3]},
            vquantile=vquantile_top,
            rcparams=rcparams,
        )
        plot_mat(
            folder / f"top_differential_expression",
            de_mat[top_mask_c][:, top_mask],
            env_mask=env_mask[top_mask_c][:, top_mask],
            ordering=top_ordering,
            cmap=cmap_de,
            figsize_key=figsize_key_causal,
            plot_colorbar_inset=True,
            colorbar_inset_kwargs={'ticks': [-0.2, 0.2]},
            vquantile=vquantile_top,
            rcparams=rcparams,
        )
        plot_mat(
            folder / f"top_differential_expression_pred",
            de_pred[top_mask_c][:, top_mask],
            env_mask=env_mask[top_mask_c][:, top_mask],
            ordering=top_ordering,
            cmap=cmap_de,
            figsize_key=figsize_key_causal,
            plot_colorbar_inset=True,
            colorbar_inset_kwargs={'ticks': [-0.1, 0.1]},
            vquantile=vquantile_top,
            rcparams=rcparams,
        )

        # Enrichr analysis
        for gene_sets_file, (gene_set_name, gene_cmap) in gene_sets.items():
            file_path = folder / f"cluster_gsea_{gseadescr}_{gene_set_name}"
            generate_cluster_enrichr_table(
                file_path=file_path,
                cluster_labels=cluster_labels,
                gene_names=gene_names,
                gene_sets_file=gene_sets_file,
                top_n_per_cluster=top_n_per_cluster,
                fdr_cutoff=fdr_cutoff,
            )

            # hand-tuned reordering to make plots comparable
            # permutation[i] = j means row i is mapped to row j in new plot (counting from bottom)
            permutation = None
            if gene_set_name == "Gene_ontology" and "replogle-1000__1__tikh=0_1" in file_path.parent.stem:
                permutation = (3, 0, 2, 1)

            plot_cluster_enrichr_from_table(
                file_path=file_path,
                row_permutation=permutation,
                cmap=gene_cmap,
                label_lookup=label_lookup,
                plot_legend=True,
                rcparams=rcparams,
                figsize_key=figsize_key_enrichr,
            )


    """
    Other plots
    """
    if plot_other_mats:

        vquantile_mats = 0.97
        long_side_mats = 0.7

        # c
        plot_mat(
            folder / f"c",
            c_mat,  # [k, d]
            transpose=False,
            env_mask=env_mask,
            ordering=ordering,
            cmap=cmcrameri.cm.vanimo,
            figsize_key=figsize_key_causal_components,
            plot_colorbar_inset=True,
            colorbar_inset_kwargs={
                'ticks': [-5, 0, 5],
                'long_side': long_side_mats,
                'ticklocation': 'bottom',
            },
            vquantile=vquantile_mats,
            rcparams=rcparams,
        )

        # f0 and finf
        plot_mat(
            folder / f"f0",
            f0,  # [k, d]
            transpose=False,
            env_mask=env_mask,
            ordering=ordering,
            cmap=cmcrameri.cm.vanimo,
            figsize_key=figsize_key_causal_components,
            plot_colorbar_inset=True,
            colorbar_inset_kwargs={
                'ticks': [-20, 0, 20],
                'long_side': long_side_mats,
                'ticklocation': 'bottom',
            },
            vquantile=vquantile_mats,
            rcparams=rcparams,
        )
        plot_mat(
            folder / f"finf",
            finf,  # [k, d]
            transpose=False,
            env_mask=env_mask,
            ordering=ordering,
            cmap=cmcrameri.cm.vanimo,
            figsize_key=figsize_key_causal_components,
            plot_colorbar_inset=True,
            colorbar_inset_kwargs={
                'ticks': [-1.5, 0, 1.5],
                'long_side': long_side_mats,
                'ticklocation': 'bottom',
            },
            vquantile=vquantile_mats,
            rcparams=rcparams,
        )
        plot_mat(
            folder / f"finf_pinv",
            # tikhonov_pinv expects [d, k] input, and returns [k, d]
            # finf has orientation [k, d] (perturbations on first axis), so we transpose before pinv
            tikhonov_pinv(finf.T, tikhonov=tikhonov),  # [k, d]
            transpose=False,
            env_mask=env_mask,
            ordering=ordering,
            cmap=cmcrameri.cm.vanimo,
            figsize_key=figsize_key_causal_components,
            plot_colorbar_inset=True,
            colorbar_inset_kwargs={
                'ticks': [-0.1, 0, 0.1],
                'long_side': long_side_mats,
                'ticklocation': 'bottom',
            },
            vquantile=vquantile_mats,
            rcparams=rcparams,
        )

        # differential expression matrix
        plot_mat(
            folder / f"differential_expression",
            de_mat,  # [k, d]
            transpose=False,
            env_mask=env_mask,
            ordering=ordering,
            cmap=cmap_de,
            figsize_key=figsize_key_causal_components,
            plot_colorbar_inset=True,
            colorbar_inset_kwargs={
                'ticks': [-0.2, 0, 0.2],
                'long_side': long_side_mats,
                'ticklocation': 'bottom',
            },
            vquantile=vquantile_mats,
            rcparams=rcparams,
        )
        plot_mat(
            folder / f"differential_expression_pred",
            de_pred,  # [k, d]
            transpose=False,
            env_mask=env_mask,
            ordering=ordering,
            cmap=cmap_de,
            figsize_key=figsize_key_causal_components,
            plot_colorbar_inset=True,
            colorbar_inset_kwargs={
                'ticks': [-0.1, 0, 0.1],
                'long_side': long_side_mats,
                'ticklocation': 'bottom',
            },
            vquantile=vquantile_mats,
            rcparams=rcparams,
        )
        plot_singular_value_spectrum(
            folder / f"finf_singular_values",
            finf,
            tikhonov=onp.array([0.1, 0.01, 0.001]),
            rcparams=rcparams,
        )

    return results
