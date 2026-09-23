
import numpy as onp
import pandas as pd
import wandb
from scipy.sparse.csgraph import dijkstra
import cmcrameri

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pycirclize import Circos
from pycirclize.parser import Matrix

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster


from lcd_clipr.experiment.plot_config import MATPLOTLIB_RCPARAMS, FIGSIZE, COLORS_2, COLOR_RED
from lcd_clipr.experiment.plot_utils import save_colorbar
from lcd_clipr.experiment.analyze_causal import maxeig, CAUSAL_EPS, thres_sgn


LARGE_EFFECT_THRES = 0.25


def select_top_genes_for_visualization(mat, targets, labels=None, top_shown=150, seed=0, ordering=None):
    rng = onp.random.default_rng(seed)
    out_degree = onp.abs(mat).sum(1)
    top_by_out_degree = onp.argsort(-out_degree)

    # Prioritize genes with labels != -1 if labels provided
    if labels is not None:
        labeled_indices = onp.where(labels != -1)[0]
        unlabeled_top = [idx for idx in top_by_out_degree if labels[idx] == -1]
        n_remaining = max(0, top_shown - len(labeled_indices))
        top_a_mask = onp.sort(onp.concatenate([labeled_indices, unlabeled_top[:n_remaining]]))
    else:
        top_a_mask = onp.sort(top_by_out_degree[:top_shown])

    target_masked = [tar for tar in targets if tar not in top_a_mask]
    top_c_mask = onp.array([j for j, tar in enumerate(targets) if tar not in target_masked])
    if targets.all() and len(targets) == len(mat):
        top_c_mask = top_a_mask
    elif top_shown < len(top_c_mask):
        top_c_mask = onp.sort(rng.choice(top_c_mask, size=top_shown, replace=False))

    # If ordering provided, filter and remap to new indices
    if ordering is not None:
        orig_to_new = {orig: new for new, orig in enumerate(top_a_mask)}
        filtered_ordering = onp.array([orig_to_new[idx] for idx in ordering if idx in top_a_mask])
        return top_a_mask.astype(int), top_c_mask.astype(int), filtered_ordering.astype(int)

    else:
        return top_a_mask.astype(int), top_c_mask.astype(int)


def reorder_vec(vec, *, ordering):
    if isinstance(vec, list):
        assert len(vec) == len(ordering)
        return [vec[i] for i in ordering]
    else:
        assert vec.ndim == 1
        assert ordering.shape == vec.shape
        return vec[ordering]


def reorder_mat(mat, *, ordering, env_mask=None):
    assert mat.ndim == 2
    if env_mask is not None:
        # reorders rows based on which variable was targeted by the perturbation in env_mask
        # equivalent to assembling square mat with nan rows and then later dropping nan rows again
        assert mat.shape == env_mask.shape
        targets = env_mask.any(0)
        assert targets.shape == ordering.shape
        assert onp.allclose(onp.sort(ordering), onp.arange(ordering.shape[0]))
        pos_in_square = onp.where(targets)[0]
        ordered_pos_in_square = onp.where(pos_in_square[..., None] == ordering[None])[1]
        row_ordering = onp.argsort(ordered_pos_in_square)
    else:
        row_ordering = ordering
    return mat[row_ordering][:, ordering]


def offdiag(mat):
    rows, cols = onp.indices(mat.shape)
    off_diag_mask = rows != cols
    return mat[off_diag_mask].copy()


def threshold_mask(mat, below_quantile=None, below_threshold=None, axis=None, diag=False):
    drop = onp.zeros_like(mat, dtype=bool)
    if diag:
        rows, cols = onp.indices(mat.shape)
        drop |= rows == cols
    if below_quantile is not None:
        v = onp.abs(mat)
        onp.fill_diagonal(v, onp.nan)
        thres = onp.nanquantile(v, below_quantile, axis=axis, keepdims=True)
        drop |= onp.abs(mat) < thres
    if below_threshold is not None:
        drop |= onp.abs(mat) < below_threshold
    # return mask of entries we keep
    return ~drop


def threshold_mat(mat, drop=False, **kwargs):
    keep = threshold_mask(mat, **kwargs)
    if drop:
        return mat[keep]
    else:
        return onp.where(keep, mat, 0.0)


def compute_distance_matrix(mat, eps=1e-10, inf_away_from_max=5):
    def close_to_dist_inverse(z):
        z = 1 / (z + eps)
        z[onp.diag_indices_from(z)] = 0
        return z

    absmat = onp.abs(mat)
    closeness = absmat + absmat.T
    weight = close_to_dist_inverse(closeness)
    dists = dijkstra(csgraph=weight, directed=True, return_predecessors=False)

    if not onp.allclose(dists, dists.T):
        # distance not symmetric, e.g. when weight matrix is not symmetric
        dists = onp.minimum(dists, dists.T)

    isinf = onp.isinf(dists)
    dists = onp.where(isinf, inf_away_from_max * onp.max(dists[~isinf]), dists)

    # Convert to condensed form
    assert dists.ndim == 2
    assert onp.allclose(dists, dists.T) and onp.allclose(onp.diag(dists), 0), "dists must be symmetric distance matrix"
    dists = squareform(dists, checks=False)
    assert dists.ndim == 1
    return dists


def plot_singular_value_spectrum(
    file_path,
    mat,
    tikhonov=None,
    eps=1e-10,
    color=COLORS_2[0],
    xlim=None,
    ylim=None,
    rcparams=MATPLOTLIB_RCPARAMS,
):
    """
    Plot the singular value spectrum of a square matrix `mat` and print its effective rank.

    The effective rank (Roy & Vetterli, 2007) is exp(H) where H is the Shannon entropy of the
    singular values normalized to a probability distribution.
    """
    plt.rcParams.update(rcparams)

    mat = onp.asarray(mat)
    assert mat.ndim == 2 and mat.shape[0] == mat.shape[1]

    sigma = onp.linalg.svd(mat, compute_uv=False)
    sigma = onp.sort(sigma)[::-1]

    # effective rank = exp(entropy of normalized singular values)
    p = sigma / (sigma.sum() + eps)
    p = p[p > eps]
    entropy = -onp.sum(p * onp.log(p))
    effective_rank = float(onp.exp(entropy))

    print(f"{file_path.name}\neffective rank = {effective_rank:.3f}")

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(FIGSIZE["eigenvalue_spectrum_analysis"]["ax_width"],
                 FIGSIZE["eigenvalue_spectrum_analysis"]["ax_height"]),
    )

    idx = onp.arange(1, len(sigma) + 1)
    ax.plot(idx, sigma, marker='o', markersize=1.0, linewidth=0.5, color=color)
    ax.axvline(effective_rank, color="gray", linestyle="--", linewidth=0.5)

    if tikhonov is not None:
        for tik in tikhonov:
            ax.axhline(onp.sqrt(tik), color=COLOR_RED, linestyle="--", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_xlabel("Index $i$")
    ax.set_ylabel("Singular value $\\sigma_i$")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()

    file_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(file_path.parent / (f"{file_path.name}" + f".pdf"),
                format="pdf", bbox_inches='tight', dpi=fig.dpi)
    plt.close()

    return effective_rank


def plot_causal_graph(
    file_path,
    mat,
    var_names,
    node_cmap="tab20",
    thres=None,
    edge_colors=None,
    cluster_labels=None,
    cluster_colors=None,
    start=-20,
    # color_correct="#22c55e",  # green
    # color_wrong="#ef4444",  # red
    color_correct=COLORS_2[3], # green
    color_wrong=COLORS_2[6],  # red
    link_alpha=0.65,
    rcparams=MATPLOTLIB_RCPARAMS,
    figsize_key="causal_analysis_chord",
):
    """
    Plot a causal graph as a chord diagram using pycirclize.

    Args:
        file_path: Path to save the plot
        mat: Square matrix of edge weights (shape [n, n])
        var_names: List of variable names
        node_cmap: Colormap for nodes (default: "tab20")
        thres: Threshold for edge inclusion
        edge_colors: Optional boolean matrix for edge coloring (True -> green, False -> red)
        cluster_labels: Optional array of cluster labels for each variable (same length as var_names)
        cluster_colors: Optional dict mapping cluster labels to colors. If a label is not in the dict, white is used.
        start: Starting angle in degrees for the first sector (default: 0). Use to rotate the diagram.
        color_correct: Color for "correct" edges (True in edge_colors)
        color_wrong: Color for "false" edges (False in edge_colors)
        link_alpha: Alpha transparency for links
        rcparams
        figsize_key
    """
    plt.rcParams.update(rcparams)

    # Build from-to table and edge color map
    fromto_data = []
    edge_color_map = {}
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i == j:
                continue
            abs_weight = abs(float(mat[i, j]))
            if thres is None or abs_weight > thres:
                source, target = var_names[i], var_names[j]
                fromto_data.append([source, target, abs_weight])
                if edge_colors is not None:
                    edge_color_map[(source, target)] = color_correct if edge_colors[i, j] else color_wrong

    if not fromto_data:
        print(f"No edges above threshold {thres} for {file_path.stem}, skipping.")
        return

    matrix = Matrix.parse_fromto_table(pd.DataFrame(fromto_data, columns=["from", "to", "value"]))

    # Custom edge colors handler
    if edge_colors is not None:
        def link_kws_handler(from_label: str, to_label: str) -> dict:
            color = edge_color_map.get((from_label, to_label), "#888888")
            return dict(fc=color, alpha=link_alpha)
    else:
        link_kws_handler = None

    # Determine node colors based on cluster_labels and cluster_colors
    effective_cmap = node_cmap
    if cluster_labels is not None and cluster_colors is not None:
        var_to_cluster = {name: cluster_labels[i] for i, name in enumerate(var_names)}
        sector_names = set([d[0] for d in fromto_data] + [d[1] for d in fromto_data])
        effective_cmap = {
            name: cluster_colors.get(var_to_cluster.get(name), 'white')
            for name in sector_names
        }

    # Create and plot chord diagram
    if start > 0:
        effective_start, effective_end = start - 360, start
    else:
        effective_start, effective_end = start, start + 360

    circos = Circos.chord_diagram(
        matrix,
        space=1.5,
        cmap=effective_cmap,
        r_lim=(95, 100), # controls width of color on outer radius (reduce first value to make wider)
        start=effective_start,
        end=effective_end,
        label_kws=dict(size=rcparams.get("axes.labelsize", 6), orientation="vertical"),
        link_kws=dict(ec="black", lw=0, direction=1),
        link_kws_handler=link_kws_handler,
    )
    fig = circos.plotfig(figsize=(
        FIGSIZE[figsize_key]["ax_width"],
        FIGSIZE[figsize_key]["ax_height"],
    ))

    # Remove sector edge lines
    for patch in fig.axes[0].patches:
        patch.set_linewidth(0)

    # Save legend for edge colors as separate file if provided
    if edge_colors is not None:
        legend_elements = [
            Patch(facecolor=color_correct, alpha=link_alpha, label="DE"),
            Patch(facecolor=color_wrong, alpha=link_alpha, label="no DE"),
        ]
        legend_fig, legend_ax = plt.subplots(figsize=(1, 0.5))
        legend_ax.axis("off")
        legend_ax.legend(
            handles=legend_elements,
            loc="center",
            frameon=False,
            ncol=1,
            handletextpad=0.3,
            columnspacing=0.8,
            labelspacing=0.3,
        )
        legend_fig.savefig(file_path.parent / f"legend-{file_path.stem}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(legend_fig)

    # Save
    file_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(file_path.parent / f"{file_path.stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def compute_ordering_and_linkage(
        mat,
        *,
        gene_names=None,
):
    """
    Compute hierarchical clustering ordering and linkage matrix from a matrix.

    Args:
        mat: Input matrix for clustering
        gene_names: Optional gene names for labeling

    Returns:
        ordering: Array of indices representing the optimal ordering from hierarchical clustering
        linkage_matrix: Linkage matrix from scipy.cluster.hierarchy.linkage
        distance_matrix: Condensed distance matrix (1D array)
    """
    assert not onp.isnan(mat).any(), "Double check that distance functions can handle nans here"

    dists = compute_distance_matrix(mat)
    assert dists.ndim == 1, "dists should be condensed, otherwise linkage used the wrong metric"

    linkage_matrix = linkage(
        dists,
        method="average",
        optimal_ordering=True,
    )

    dendro_result = dendrogram(
        linkage_matrix,
        no_plot=True,
        labels=gene_names or None,
    )
    ordering = onp.array(dendro_result['leaves'])

    return ordering, linkage_matrix, dists


def compute_tight_clusters_at_height(
        linkage_matrix,
        *,
        height,
        min_cluster_size,
        distance_matrix=None,
        linkage_method="average",
        n_clusters=None,
):
    """
    Find up all clusters of minimum size `min_cluster_size` by cutting the dendrogram at a given `height`.

    Args:
        linkage_matrix: Linkage matrix from scipy.cluster.hierarchy.linkage
        height: Height at which to cut the dendrogram
        min_cluster_size: Minimum size of each cluster
        distance_matrix: Optional condensed distance matrix (1D array) for sorting clusters by tightness
        linkage_method: Linkage criterion for computing tightness
        n_clusters: Optional maximum number of clusters to return (only the n tightest clusters)

    Returns:
        cluster_labels: Array of cluster assignments (-1 for unclustered samples) in original gene order
    """
    if linkage_matrix is None:
        return None

    n_samples = linkage_matrix.shape[0] + 1

    # cut the dendrogram at the specified height
    initial_labels = fcluster(linkage_matrix, t=height, criterion='distance') - 1  # 0-indexed

    # count cluster sizes and filter by min_cluster_size
    unique_clusters, counts = onp.unique(initial_labels, return_counts=True)
    valid_mask = counts >= min_cluster_size
    valid_clusters = unique_clusters[valid_mask]

    if len(valid_clusters) == 0:
        print(f"compute_tight_clusters_at_height: Warning - no clusters of size >={min_cluster_size} found at height {height:.4f}, returning None.")
        return None

    # Order clusters by tightness
    if distance_matrix is not None:
        dist_square = squareform(distance_matrix)
        cluster_tightness = []

        for cluster_idx in valid_clusters:
            cluster_members = onp.where(initial_labels == cluster_idx)[0]

            if len(cluster_members) == 1:
                cluster_tightness.append(0.0)
                continue

            cluster_dist_submatrix = dist_square[onp.ix_(cluster_members, cluster_members)]
            triu_indices = onp.triu_indices(len(cluster_members), k=1)
            intra_cluster_dists = cluster_dist_submatrix[triu_indices]

            if linkage_method == "single":
                tightness = onp.min(intra_cluster_dists)
            elif linkage_method == "complete":
                tightness = onp.max(intra_cluster_dists)
            elif linkage_method == "average":
                tightness = onp.mean(intra_cluster_dists)
            else:
                raise ValueError(f"Unsupported linkage_method: {linkage_method}")
            cluster_tightness.append(tightness)

        # sort valid_clusters by tightness in ascending order
        tightness_order = onp.argsort(cluster_tightness)
        valid_clusters = valid_clusters[tightness_order]

    # optionally keep only the n tightest clusters
    if n_clusters is not None and len(valid_clusters) > n_clusters:
        valid_clusters = valid_clusters[:n_clusters]

    # create cluster labels array
    cluster_labels = onp.full(n_samples, -1, dtype=int)
    for new_idx, old_idx in enumerate(valid_clusters):
        cluster_labels[initial_labels == old_idx] = new_idx

    return cluster_labels


def _split_contiguous_unassigned(cluster_labels):
    """
    Replace contiguous sections of -1 with unique cluster IDs.

    Args:
        cluster_labels: Array of cluster assignments with -1 for unassigned

    Returns:
        modified_labels: Array where each contiguous -1 section has a unique ID
    """
    modified_labels = cluster_labels.copy()
    unassigned_mask = cluster_labels == -1

    if not unassigned_mask.any():
        return modified_labels

    # Find start and end of contiguous -1 sections
    diff = onp.diff(onp.concatenate([[0], unassigned_mask.astype(int), [0]]))
    starts = onp.where(diff == 1)[0]
    ends = onp.where(diff == -1)[0]

    # Assign unique IDs to each contiguous section
    max_id = cluster_labels[cluster_labels >= 0].max() if (cluster_labels >= 0).any() else -1
    next_id = max_id + 1

    for start, end in zip(starts, ends):
        modified_labels[start:end] = next_id
        next_id += 1

    return modified_labels


def _block_average_mat(mat, cluster_labels, unassigned_as_cluster=True, ignore_diagonal=True):
    """
    Block-average a square matrix based on cluster labels by filling each block with its mean.

    For each pair of clusters (including within-cluster blocks), compute the average
    of all matrix entries between samples in those clusters, then fill all entries
    in that block with the average value.

    Args:
        mat: Input square matrix (n x n)
        cluster_labels: Cluster assignment for each sample (-1 for unclustered)
        unassigned_as_cluster: If True, treat contiguous sections of -1 as separate clusters (default: True)
        ignore_diagonal: If True, ignore diagonal entries when computing block means and preserve them (default: True)

    Returns:
        block_mat: Matrix with blocks filled by their averages (same size as input)
    """
    assert mat.shape[0] == mat.shape[1]
    assert not onp.isnan(mat).any()
    cluster_labels = onp.array(cluster_labels)

    # Split contiguous -1 sections into separate clusters if requested
    modified_labels = _split_contiguous_unassigned(cluster_labels) if unassigned_as_cluster else cluster_labels
    unique_clusters = onp.unique(modified_labels)

    if not unassigned_as_cluster:
        unique_clusters = unique_clusters[unique_clusters >= 0]  # Exclude -1

    def _mean_fun(x):
        # return onp.mean(x)  # simple mean
        return onp.sign(onp.mean(x)) * onp.mean(onp.abs(x))  # magnitude-preserving mean

    # Copy matrix to preserve original
    block_mat = onp.array(mat).copy()
    diagonal = onp.diag(mat).copy()

    # Fill each block with its mean
    for cluster_i in unique_clusters:
        for cluster_j in unique_clusters:
            # Get indices for each cluster
            idx_i = onp.where(modified_labels == cluster_i)[0]
            idx_j = onp.where(modified_labels == cluster_j)[0]

            # Extract block
            block = block_mat[onp.ix_(idx_i, idx_j)]

            # For diagonal blocks with ignore_diagonal, split into upper and lower triangular
            if ignore_diagonal and cluster_i == cluster_j and idx_i.size > 1:
                b = idx_i.size
                filled_block = block.copy()

                # Upper triangular (excluding diagonal)
                upper = onp.triu_indices(b, k=1)
                filled_block[*upper] = _mean_fun(block[*upper])

                # Lower triangular (excluding diagonal)
                lower = onp.tril_indices(b, k=-1)
                filled_block[*lower] = _mean_fun(block[*lower])

                block_mat[onp.ix_(idx_i, idx_j)] = filled_block
            else:
                block_mat[onp.ix_(idx_i, idx_j)] = _mean_fun(block)

    if ignore_diagonal:
        onp.fill_diagonal(block_mat, diagonal)

    return block_mat


def plot_mat(
        file_path,
        mat,
        *,
        cmap=cmcrameri.cm.berlin_r,
        targets=None,
        env_mask=None,
        ordering=None,
        linkage_matrix=None,
        plot_dendro=False,
        plot_targets=True,
        gene_names=None,
        title=None,
        vmin=None,
        vmax=None,
        show_spines=False,
        show_gene_ticks=False,
        plot_colorbar=False,
        plot_colorbar_inset=False,
        colorbar_inset_kwargs=None,
        transpose=False,
        block_average_labels=None,
        unassigned_as_cluster=True,
        ignore_diagonal=True,
        figsize_key="adjacency_matrix_v2",
        file_type="pdf",
        vquantile=0.99,
        dpi=None,
        cluster_colors=None,
        rcparams=MATPLOTLIB_RCPARAMS,
):
    """
    Setup figure
    """
    plt.rcParams.update(rcparams)

    if targets is None and block_average_labels is None:
        plot_targets = False

    mat_ratio = max(0.2, mat.shape[0] / mat.shape[1])
    scale_height = 1 if transpose else mat_ratio
    scale_width = mat_ratio if transpose else 1

    fig_width = FIGSIZE[figsize_key]["ax_width"] * scale_width
    fig_height_dendro = FIGSIZE[figsize_key]["ax_height_dendrogram"]
    fig_height_main = FIGSIZE[figsize_key]["ax_height"] * scale_height
    fig_height_targets = FIGSIZE[figsize_key]["ax_height_targets"]

    height_ratios = []
    if plot_dendro:
        height_ratios.append(fig_height_dendro)
    height_ratios.append(fig_height_main)
    if plot_targets:
        height_ratios.append(fig_height_targets)

    fig, axes = plt.subplots(len(height_ratios), 1, figsize=(fig_width, sum(height_ratios)),
                             gridspec_kw={"height_ratios": height_ratios})
    axes = onp.atleast_1d(axes)

    # Assign axes
    ax_dendro = axes[0] if plot_dendro else None
    ax = axes[1] if plot_dendro else axes[0]
    ax_targets = axes[2] if (plot_dendro and plot_targets) else (axes[1] if plot_targets else None)

    """
    Apply ordering
    """
    if ordering is not None:
        mat = reorder_mat(mat, ordering=ordering, env_mask=env_mask)
        if gene_names is not None:
            gene_names = reorder_vec(gene_names, ordering=ordering)
        if targets is not None:
            targets = reorder_vec(targets, ordering=ordering)
        if block_average_labels is not None:
            block_average_labels = reorder_vec(block_average_labels, ordering=ordering)

    """
    Dendrogram
    """

    # Plot dendrogram if requested
    # Ordering of dend
    if plot_dendro:
        assert linkage_matrix is not None
        with plt.rc_context({
            'lines.linewidth': 0.2,
        }):
            dendro = dendrogram(
                linkage_matrix,
                ax=ax_dendro,
                labels=gene_names or None,
                leaf_rotation=90,
                color_threshold=-1,  # never color links
                above_threshold_color="black",
                truncate_mode="none",
            )

        ax_dendro.set_xticks([])
        ax_dendro.set_yticks([])
        ax_dendro.spines[['left', 'right', 'top', 'bottom']].set_visible(show_spines)

        if ordering is not None:
            assert onp.array_equal(ordering, onp.array(dendro['leaves'])), "Mismatch computed and plotted dendro ordering"
            if gene_names is not None:
                assert [gene_names[i] for i in ordering] == dendro["ivl"]


    """
    Matrix
    """
    # Block-average matrix if block_average_labels provided
    if block_average_labels is not None:
        mat = _block_average_mat(
            mat, block_average_labels,
            unassigned_as_cluster=unassigned_as_cluster,
            ignore_diagonal=ignore_diagonal
        )
        if targets is None:
            targets = block_average_labels

    if vmin is None:
        vmin = onp.nanquantile(offdiag(mat), 1 - vquantile)

    if vmax is None:
        vmax = onp.nanquantile(offdiag(mat), vquantile)

    vmin, vmax = min(-vmax, vmin), max(-vmin, vmax)

    ax.matshow(mat.T if transpose else mat, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.grid(False)
    ax.set_aspect("auto")
    ax.xaxis.tick_bottom()

    if show_gene_ticks:
        ax.tick_params(axis='both', which='major', labelsize=5)

        if gene_names is not None:
            ax.set_xticks(range(len(gene_names)))
            ax.set_xticklabels(gene_names, rotation=90, ha='center')

            ax.set_yticks(range(len(gene_names)))
            ax.set_yticklabels(gene_names)

    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.spines[['left', 'right', 'top', 'bottom']].set_visible(show_spines)

    if plot_colorbar:
        save_colorbar(
            (file_path.parent / f"colorbar-{file_path.stem}"),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            long_side=fig_width,
            label="Causal effect $\widehat{A}_{j,i}$", orientation='horizontal',
            rcparams=rcparams,
        )

    if plot_colorbar_inset:
        inset_kwargs = {
            'long_side': 0.35,
            'short_side': 0.07,
            'label': None,
            'orientation': 'horizontal',
            'show_outline': False,
            'ticklocation': 'top',
            'tickpad': 1,
        }
        if colorbar_inset_kwargs is not None:
            inset_kwargs.update(colorbar_inset_kwargs)

        save_colorbar(
            (file_path.parent / f"colorbar-inset-{file_path.stem}"),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            rcparams=rcparams,
            **inset_kwargs,
        )

    """
    Targets
    """
    if plot_targets:
        unique_targets = onp.unique(targets)
        is_binary = set(unique_targets).issubset({0, 1}) and cluster_colors is None
        if is_binary:
            ax_targets.matshow(onp.array([targets]), cmap='binary', vmin=0, vmax=1)
        else:
            tab20 = plt.cm.tab20
            unique_sorted = sorted(unique_targets)

            # Create custom colormap: use cluster_colors dict if provided, else tab20
            if cluster_colors is not None:
                colors = [cluster_colors.get(val, 'white') for val in unique_sorted]
            else:
                colors = ['white' if val == -1 else tab20(val % 20) for val in unique_sorted]
            target_cmap = mcolors.ListedColormap(colors)
            norm = mcolors.BoundaryNorm(onp.arange(len(unique_sorted) + 1) - 0.5, target_cmap.N)

            # Map target values to indices
            targets_mapped = onp.searchsorted(unique_sorted, targets)
            ax_targets.matshow(onp.array([targets_mapped]), cmap=target_cmap, norm=norm)

        ax_targets.grid(False)
        ax_targets.set_aspect("auto")
        ax_targets.set_xticks([])
        ax_targets.set_yticks([])
        ax_targets.spines[['left', 'right', 'top', 'bottom']].set_visible(show_spines)
        if show_gene_ticks:
            ax_targets.tick_params(axis='y', which='major', labelsize=5)
        else:
            ax_targets.set_yticks([])

    """
    Figure
    """

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    if file_path is None:
        plt.show()
    else:
        file_path.parent.mkdir(exist_ok=True, parents=True)
        name = file_path.name

        # save_format = "pdf" # pdf savefig of annotation matrix is buggy for some reason
        # save_format = "png" # png savefig works, but when imported into keynote, the size changes
        # thus we save as svg below
        # fig.savefig(file_path.parent / f"{name}.png", format="png", bbox_inches='tight', dpi=dpi or fig.dpi)
        # fig.savefig(file_path.parent / f"{name}.svg", format="svg", bbox_inches='tight', dpi=dpi or fig.dpi)
        fig.savefig(file_path.parent / f"{name}.{file_type}", format=file_type, bbox_inches='tight', dpi=dpi or fig.dpi)

    plt.close()


def _make_mat_title(mat, *, eps=None):
    title = ""
    title += f"eig: {maxeig(mat):.2f} "
    title += f"(diag: {onp.diag(mat).mean():.2f})   "
    if eps is not None:
        classes, freqs = onp.unique(thres_sgn(mat, eps=eps), return_counts=True)
        freqs = freqs / freqs.sum()
        title += f"cls: {' '.join([f'{int(c)}={f:.2f}' for c, f in zip(classes, freqs)])}  "
    return title


def plot_two_matrices(pred_mat, true_mat, size=2.5, save_path=None, cmap=cmcrameri.cm.berlin_r,
                      threshold_eps=None, annotation_eps=None, colorbar_label=None, vscale_quantile=0.995,
                      wandb_path=None, to_wandb=False, title=None, ignore_diagonal_for_cmap=False,
                      rcparams=MATPLOTLIB_RCPARAMS):

    plt.rcParams.update(rcparams)

    assert pred_mat is not None
    wandb_images = {}
    fig, axes = plt.subplots(1, 2, figsize=(2 * size, 1.2 * size))

    if threshold_eps is not None:
        pred_mat = threshold_mat(pred_mat, below_threshold=threshold_eps)
        if true_mat is not None:
            true_mat = threshold_mat(true_mat, below_threshold=threshold_eps)
        annotation_eps = threshold_eps

    if annotation_eps is None:
        annotation_eps = CAUSAL_EPS

    if ignore_diagonal_for_cmap:
        vscale = onp.nanquantile(onp.abs(offdiag(pred_mat)), vscale_quantile)
    else:
        vscale = onp.nanquantile(onp.abs(pred_mat), vscale_quantile)

    axes[0].imshow(pred_mat, cmap=cmap, vmin=-vscale, vmax=vscale)

    if pred_mat.shape[0] == pred_mat.shape[1] and not onp.isnan(pred_mat).any():
        axes[0].set_title(_make_mat_title(pred_mat, eps=annotation_eps))

    if true_mat is not None:
        if true_mat.shape[0] == true_mat.shape[1] and ignore_diagonal_for_cmap:
            vscale_true = onp.nanquantile(onp.abs(offdiag(true_mat)), vscale_quantile)
        else:
            vscale_true = onp.nanquantile(onp.abs(true_mat), vscale_quantile)

        axes[1].imshow(true_mat, cmap=cmap, vmin=-vscale_true, vmax=vscale_true)

        if true_mat.shape[0] == true_mat.shape[1] and not onp.isnan(true_mat).any():
            axes[1].set_title(_make_mat_title(true_mat, eps=annotation_eps))
    else:
        axes[1].axis('off')


    for ax in axes.ravel():
        for key in ax.spines.keys():
            ax.spines[key].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])


    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    if save_path and colorbar_label is not None:
        save_colorbar((save_path.parent / f"colorbar-{save_path.stem}"),
                      vmin=-vscale, vmax=vscale, cmap=cmap,
                      label_right=True,
                      long_side=FIGSIZE["colorbar_two"]["long_side"],
                      short_side=FIGSIZE["colorbar_two"]["short_side"],
                      rcparams=rcparams,
                      label=colorbar_label, orientation='horizontal')

    if to_wandb:
        assert wandb_path is not None
        wandb_images[wandb_path] = wandb.Image(plt)
        plt.close()
    elif save_path is not None:
        plt.savefig(save_path.with_suffix(".png"), format="png", bbox_inches='tight', dpi=500)
        plt.close()
    else:
        plt.show()

    return wandb_images
