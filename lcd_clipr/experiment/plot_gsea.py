from pathlib import Path
from collections import defaultdict
import math
import re
from collections import Counter

import textwrap
import numpy as onp
import matplotlib.pyplot as plt
import gseapy as gp
import pandas as pd
from scipy import stats

from lcd_clipr.experiment.plot_config import *
from lcd_clipr.experiment.plot_utils import setup_spread_labels
from lcd_clipr.definitions import ROOT_DIR, SUBDIR_ASSETS


def is_gene_tf(
    var_names,
    use_ensembl=False,
    gene_path=ROOT_DIR / SUBDIR_ASSETS / "tf_names.txt",
    ensembl_path=ROOT_DIR / SUBDIR_ASSETS / "tf_ensembls.txt",
):
    """
    Assets taken from:
    https://humantfs.ccbr.utoronto.ca/download.php
    """
    df = pd.read_csv(ensembl_path if use_ensembl else gene_path, header=None)
    tfs = set(df[0])
    is_tf = onp.array([name in tfs for name in var_names])
    return is_tf


def sort_by_common_prefixes(genes, prefix_lengths=(2, 3, 4)):
    """
    Sort strings so that groups with common prefixes appear first.
    """
    # count prefix frequencies for each length
    prefix_counters = {
        l: Counter(g[:l] for g in genes if len(g) >= l)
        for l in prefix_lengths
    }

    # score each gene by summed prefix frequencies
    def score(g):
        return sum(
            prefix_counters[l].get(g[:l], 0)
            for l in prefix_lengths
            if len(g) >= l
        )

    return sorted(genes, key=lambda g: (-score(g), g))


def _load_gene_sets_from_gmt(gmt_path):
    """
    Load gene sets from GMT file.

    Returns
    -------
    dict
        Dictionary mapping term names to lists of genes
    """
    gene_sets = {}
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                term_name = parts[0]
                gene_sets[term_name] = {
                    "description": parts[1],
                    "genes": parts[2:],
                }
    return gene_sets


def generate_cluster_enrichr_table(
    *,
    file_path,
    cluster_labels,
    gene_names,
    gene_sets_file="c5.go.v2025.1.Hs.symbols.gmt",
    fdr_cutoff=0.01,
    top_n_per_cluster=10,
):
    """
    Perform GSEA on clusters and save enrichment results as a CSV table.

    Genesets downloaded from:
    https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp
    on 17 November 2025

    Parameters
    ----------
    file_path : Path
        Path to save the CSV table (without extension)
    cluster_labels : ndarray
        Cluster assignment for each gene (-1 for excluded genes)
    gene_names : list
        List of gene names
    gene_sets_file : str
        Gene set database (default: GO_Biological_Process_2021)
    fdr_cutoff : float
        Adjusted p-value (FDR) threshold of Benjamini-Hochberg correction
    top_n_per_cluster : int
        Top terms per cluster to consider
    """
    if cluster_labels is None:
        print("Warning: cluster_labels is None. Skipping GSEA.")
        return

    unique_clusters = onp.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters >= 0]  # Exclude cluster -1

    # Load gene sets from GMT file
    gene_sets_path = ROOT_DIR / SUBDIR_ASSETS / gene_sets_file
    all_gene_sets = _load_gene_sets_from_gmt(gene_sets_path)

    # Run Enrichr on each cluster
    table_rows = []
    for cluster_id in unique_clusters:
        cluster_genes = [gene_names[i] for i, c in enumerate(cluster_labels) if c == cluster_id]
        try:
            gene_sets_path = ROOT_DIR / SUBDIR_ASSETS / gene_sets_file
            enr = gp.enrichr(gene_list=cluster_genes, gene_sets=str(gene_sets_path), background=gene_names)

        except Exception as e:
            print(f"  Cluster {cluster_id}: Failed - {e}")
            continue

        if enr.results is None:
            print(f"  Cluster {cluster_id}: Enrichr results `None`")
            continue

        if isinstance(enr.results, list) and not enr.results:
            print(f"  Cluster {cluster_id}: Enrichr results empty")
            continue

        # Save all terms irrespective of significance for FDR correction later
        result = enr.results
        for _, row in result.iterrows():
            term = row['Term']
            table_rows.append({
                'cluster': int(cluster_id),
                'term': term,
                'odds_ratio': row['Odds Ratio'],
                'p_value': row.get('P-value', onp.nan),
                'combined_score': row.get('Combined Score', onp.nan),
                'overlap_size': len(row.get('Genes', '').split(';')) if row.get('Genes', '') else 0,
                'overlap_genes': row.get('Genes', ''),
                'cluster_size': len(cluster_genes),
                'cluster_genes': ';'.join(cluster_genes),
                'set_size': len(all_gene_sets[term]["genes"]),
                'set_description': all_gene_sets[term]["description"],
            })

    if len(table_rows) == 0:
        print(f"Skipping GSEA table: No significant GO terms found for:\n\t{Path(file_path).stem}")
        return

    # FDR adjustment across all clusters
    df = pd.DataFrame(table_rows)
    df["adjusted_p_value"] = stats.false_discovery_control(df["p_value"], method="bh")
    df = df[df['adjusted_p_value'] < fdr_cutoff]

    # Rank top N
    for cluster_id in df['cluster'].unique():
        mask = df['cluster'] == cluster_id
        top_terms = df.loc[mask].nlargest(top_n_per_cluster, 'odds_ratio')['term'].tolist()
        df.loc[mask, "in_top_n"] = df.loc[mask, "term"].isin(top_terms)

    # Save to CSV in enrichr subfolder
    enrichr_dir = file_path.parent / "enrichr"
    enrichr_dir.mkdir(exist_ok=True, parents=True)
    csv_path = enrichr_dir / file_path.with_suffix(".csv").name
    df.to_csv(csv_path, index=False)


def plot_cluster_enrichr_from_table(
    *,
    file_path,
    row_permutation=None,
    drop_empty_clusters=True,
    dot_size_scale=12.0,
    plot_colobar=True,
    plot_legend=False,
    label_lookup=None,
    filter_by_lookup=True,
    print_set_terms=False,
    xlabel_max_width=16,
    xlabel_max_len=None,
    label_fontsize=6,
    cmap='Reds',
    cmap_vmax=1000,
    use_spread_labels=True,
    spread_label_x_offset=1.0,
    spread_label_kwargs=None,
    rcparams=MATPLOTLIB_RCPARAMS,
    figsize_key="causal_analysis_dotplot",
):
    """
    Load enrichment table from CSV and create visualization.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file (and where to save the figure)
    row_permutation : tuple[int, ...], optional
        Permutation of cluster row indices: original row `i` is placed at position `row_permutation[i]`.
        Columns are automatically re-derived (first-seen across rows in the new order) to keep dots
        roughly diagonal. Must be a valid permutation of `range(n_clusters)`.
    drop_empty_clusters : bool
        If True, exclude clusters without any enriched pathways from the plot
    dot_size_scale : float
        Scaling factor for dot sizes (default: 1.0)
    plot_colobar: bool
        If True, create and save a separate colorbar figure (default: False)
    plot_legend : bool
        If True, create and save a separate legend figure (default: False)
    label_lookup : dict, optional
        Dictionary mapping GO terms to custom labels. If a GO term is in the dict,
        the value will be used as the label (unless the value is None, in which case
        the original term is used). If a term is not in the dict, the original term is used.
    filter_by_lookup : bool
        If True and label_lookup is provided, only plot GO terms that are present
        in the label_lookup dict. If False or label_lookup is None, all terms are plotted.
        Default: False.
    print_set_terms : bool
        If True, print the set descriptions for filtered out terms. Default: False.
    xlabel_max_width : int
        Maximum width for x-axis labels (in characters) when wrapping
    xlabel_max_len : int
        Maximum length to truncate x-axis labels
    label_fontsize : float
        Font size for axis labels
    cmap : str
        Matplotlib colormap name for odds ratio coloring (default: 'Reds')
    use_spread_labels : bool
        If True, use kinked connectors to spread out x-axis labels to avoid overlaps.
        Default: False.
    spread_label_x_offset: float
        Horizontal offset for spread labels when use_spread_labels=True
    spread_label_kwargs : dict, optional
        Additional keyword arguments to pass to setup_spread_labels when use_spread_labels=True
    rcparams
    figsize_key
    """
    plt.rcParams.update(rcparams)

    # Load the CSV table from enrichr subfolder
    csv_path = file_path.parent / "enrichr" / file_path.with_suffix(".csv").name
    if not csv_path.exists():
        print(f"Warning: CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if len(df) == 0:
        print(f"Skipping GSEA plot: No enrichment data in table:\n\t{Path(file_path).stem}")
        return

    # Display union of top N terms across all clusters
    unique_clusters = onp.sort(df['cluster'].unique())
    all_pathways = defaultdict(list)
    for cluster_id in unique_clusters:
        cluster_df = df[df['cluster'] == cluster_id]
        top_terms = cluster_df[cluster_df['in_top_n'] == True]
        for _, row in top_terms.iterrows():
            all_pathways[row['term']].append((
                row['cluster'],
                row['odds_ratio'],
                row['adjusted_p_value']
            ))

    assert all_pathways

    # Filter out empty clusters if requested
    enriched_terms_ordered = list(all_pathways.keys())

    # Filter terms by lookup dict if requested
    if filter_by_lookup and label_lookup is not None:
        enriched_terms_ordered = [term for term in enriched_terms_ordered if term in label_lookup]
        if len(enriched_terms_ordered) == 0:
            print(f"Skipping GSEA plot: No GO terms from label_lookup found in enrichment results:\n\t{Path(file_path).stem}")
            return

    if drop_empty_clusters:
        # Determine which clusters have at least one term from enriched_terms_ordered
        clusters_with_pathways = set()
        for cluster_id in unique_clusters:
            cluster_df = df[df['cluster'] == cluster_id]
            for term in enriched_terms_ordered:
                if term in cluster_df['term'].values:
                    clusters_with_pathways.add(cluster_id)
                    break

        unique_clusters = onp.array([
            cluster_id for cluster_id in unique_clusters if cluster_id in clusters_with_pathways
        ])
        n_clusters = len(unique_clusters)

        if n_clusters == 0:
            print(f"Skipping GSEA plot: No clusters with enriched pathways:\n\t{Path(file_path).stem}")
            return
    else:
        n_clusters = len(unique_clusters)

    # Apply manual row permutation and re-derive columns to keep diagonal alignment
    if row_permutation is not None:
        assert sorted(row_permutation) == list(range(n_clusters)), \
            f"Invalid row permutation {row_permutation} for {n_clusters} clusters"
        unique_clusters = unique_clusters[onp.argsort(row_permutation)]
        term_set = set(enriched_terms_ordered)
        reordered = [t for cid in unique_clusters
                     for t in df[(df['cluster'] == cid) & df['in_top_n']]['term']
                     if t in term_set]
        enriched_terms_ordered = list(dict.fromkeys(reordered + enriched_terms_ordered))

    # Print summary of GO terms that will be plotted
    print(f"\nPlotting {len(enriched_terms_ordered)} GO term(s):")
    for i, term in enumerate(enriched_terms_ordered, 1):
        print(f"  {term}")

    # compute vmin/vmax across odds ratios
    all_odds_ratios = []
    for cluster_id in unique_clusters:
        cluster_df = df[df['cluster'] == cluster_id]
        for term in enriched_terms_ordered:
            term_rows = cluster_df[cluster_df['term'] == term]
            if len(term_rows) > 0:
                all_odds_ratios.append(term_rows.iloc[0]['odds_ratio'])

    vmin = 0
    vmax = cmap_vmax if cmap_vmax is not None else onp.percentile(all_odds_ratios, 75)

    # plot
    width = len(enriched_terms_ordered) * FIGSIZE[figsize_key]["ax_width_per_dot"]
    height = len(unique_clusters) * FIGSIZE[figsize_key]["ax_height_per_dot"]
    fig, ax = plt.subplots(figsize=(width, height))

    label_fontsize = FIGSIZE[figsize_key].get("label_fontsize", label_fontsize)
    spread_label_kwargs = FIGSIZE[figsize_key].get("spread_label_kwargs", spread_label_kwargs)

    x_pos = onp.arange(len(enriched_terms_ordered))

    # plot dots for each cluster-pathway combination
    scatter = None
    for i, cluster_id in enumerate(unique_clusters):
        cluster_df = df[df['cluster'] == cluster_id]

        for term in enriched_terms_ordered:
            # Check if this term is significant for this cluster (even if not in top N)
            term_rows = cluster_df[cluster_df['term'] == term]
            if len(term_rows) > 0:
                row = term_rows.iloc[0]
                odds_ratio = row['odds_ratio']
                adj_pval = row['adjusted_p_value']

                scatter = ax.scatter(
                    x_pos[enriched_terms_ordered.index(term)], i,
                    s=-onp.log10(adj_pval) * dot_size_scale,
                    c=[odds_ratio],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=0.5,
                )

    # Format GO terms for x-axis
    # Use label_lookup if provided, otherwise use the original GO term
    # If the lookup value is None, use the original GO term as label
    if label_lookup:
        gene_set_labels = [
            label_lookup.get(go_term, go_term) or go_term
            for go_term in enriched_terms_ordered
        ]
    else:
        gene_set_labels = enriched_terms_ordered

    if xlabel_max_width is not None:
        def wrap_label(label):
            # If label contains "{...}" patterns, extract contents and join with newlines
            matches = re.findall(r'\{([^}]*)\}', label)
            if matches:
                return '\n'.join(matches)
            return textwrap.fill(label, width=xlabel_max_width)
        gene_set_labels = [wrap_label(label) for label in gene_set_labels]

    if xlabel_max_len is not None:
        gene_set_labels = [label[:xlabel_max_len] for label in gene_set_labels]

    if use_spread_labels:
        kwargs = spread_label_kwargs or {}
        setup_spread_labels(
            ax,
            x_pos,
            gene_set_labels,
            label_x_offset=spread_label_x_offset,
            font_size=label_fontsize,
            **kwargs,
        )
    else:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(gene_set_labels, rotation=90, ha='right', fontsize=label_fontsize)

    ax.set_yticks(range(n_clusters))

    # Y-axis labels with gene names from the table
    cluster_labels = []
    for cid in unique_clusters:
        cluster_genes_unsrt = df[df['cluster'] == cid]['cluster_genes'].iloc[0].split(";")
        cluster_genes = sort_by_common_prefixes(cluster_genes_unsrt)
        # cluster_label = f"Cluster {int(cid) + 1} ({len(cluster_genes)} genes):\n"
        # cluster_label = r"$\bf{" + f"Cluster~{int(cid) + 1}" + r"}$" + "\n"
        # cluster_label = "\n"
        cluster_label = ""
        if len(cluster_genes) <= 4:
            cluster_label += ", ".join(cluster_genes)
        elif 5 <= len(cluster_genes) <= 9:
            mid = math.ceil(len(cluster_genes) / 2)
            cluster_label += ", ".join(cluster_genes[:mid])
            cluster_label += "\n"
            cluster_label += ", ".join(cluster_genes[mid:])
        else:
            cluster_label += ", ".join(cluster_genes[:4])
            cluster_label += "\n"
            cluster_label += ', '.join(cluster_genes[4:7])
            cluster_label += f" (+{len(cluster_genes) - 7} more)"
        cluster_labels.append(cluster_label)
    ax.set_yticklabels(cluster_labels, fontsize=label_fontsize)

    ax.set_ylim(-0.5, n_clusters - 0.5)
    ax.set_xlim(-0.5, len(enriched_terms_ordered) - 0.5)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    file_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(file_path.with_suffix(".pdf"), format="pdf", bbox_inches='tight', dpi=300)

    # Create separate colorbar figure with fixed width and matching height
    if plot_colobar and scatter is not None:
        cbar_thickness = FIGSIZE[figsize_key]["ax_width_colobar"]
        horizontal = FIGSIZE[figsize_key].get("colorbar_horizontal", 0)
        if horizontal:
            fig_cbar, ax_cbar = plt.subplots(figsize=(horizontal, cbar_thickness))
        else:
            fig_cbar, ax_cbar = plt.subplots(figsize=(cbar_thickness, height))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        orientation = 'horizontal' if horizontal else 'vertical'
        cbar = plt.colorbar(sm, cax=ax_cbar, label='Odds Ratio', orientation=orientation)
        cbar.ax.tick_params(labelsize=label_fontsize)
        cbar.outline.set_visible(False)
        if horizontal:
            ax_cbar.xaxis.set_label_position('top')

        cbar_path = file_path.with_name("colorbar_" + file_path.stem).with_suffix(".pdf")
        fig_cbar.savefig(cbar_path, format="pdf", bbox_inches='tight', dpi=300)
        plt.close(fig_cbar)

    plt.close()

    if print_set_terms:
        print("{")
        for term in enriched_terms_ordered:
            print(f'    "{term}",')
        print("}")

    # Create separate legend figure (optional)
    if plot_legend:
        legend_elements = [
            plt.scatter(
                [], [],
                s=dot_size_scale * -onp.log10(p),
                c='gray',
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5,
                label=f'$10^{{{int(onp.log10(p))}}}$'
            )
            for p in [1e-2, 1e-4, 1e-6]
        ]

        legend_horizontal = FIGSIZE[figsize_key].get("legend_horizontal", 0)
        legend_width = FIGSIZE[figsize_key]["ax_width_legend"]
        legend_height = FIGSIZE[figsize_key]["ax_height_legend"]
        fig_legend, ax_legend = plt.subplots(
            figsize=(
                (legend_height, legend_width) if legend_horizontal else (legend_width, legend_height)
            )
        )
        ax_legend.axis('off')
        # legend_title = '$p$-value\n(corrected)'
        legend_title = 'P-value\n(corrected)'
        legend = ax_legend.legend(
            handles=legend_elements,
            title=legend_title.replace("\n", " ") if legend_horizontal else legend_title,
            loc='center',
            frameon=False,
            labelspacing=0.5,
            handletextpad=0.5,
            columnspacing=0.5,
            fontsize=rcparams.get('xtick.labelsize', label_fontsize),
            ncol=len(legend_elements) if legend_horizontal else 1,
        )
        legend.get_title().set_ha("center")

        # Save legend as separate file
        legend_path = file_path.with_name("legend_" + file_path.stem).with_suffix(".pdf")
        fig_legend.savefig(legend_path, format="pdf", bbox_inches='tight', dpi=300)
        plt.close()

