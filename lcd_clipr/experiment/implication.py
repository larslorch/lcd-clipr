import numpy as onp
from scipy.stats import fisher_exact, bootstrap
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from pathlib import Path
import matplotlib.pyplot as plt
from lcd_clipr.experiment.plot_config import FIGSIZE, MATPLOTLIB_RCPARAMS, COLORS
from lcd_clipr.experiment.plot_utils import save_legend


def _quantiles_to_percentile_labels(aggregated_df):
    """
    Convert quantile column names to top percentile labels and prepare x-axis label.

    Args:
        aggregated_df: DataFrame with quantiles as column names

    Returns:
        tuple: (quantiles as floats, percentile labels as strings, x-axis label)
    """
    quantiles = [float(col) for col in aggregated_df.columns]
    percentile_labels = []
    for q in quantiles:
        # Convert quantile to top percentile (e.g., 0.99 -> 1%)
        percentile = (1 - q) * 100
        label = f'{percentile:.7f}'.rstrip('0').rstrip('.')
        percentile_labels.append(label)

    xlabel = 'Top percentile (total edges)'
    return quantiles, percentile_labels, xlabel


def comprehensive_implication_test(p, q, fisher_significance=0.01, significance=0.05, seed=0, bootstrap_ci_samples=1000, file_path=None):
    """
    Comprehensive statistical test for "P implies Q"

    Args:
        p: boolean array indicating whether condition P holds
        q: boolean array indicating whether condition Q holds
        fisher_significance: significance level for Fisher's exact test (default: 0.01)
        significance: significance level for statistical tests (default: 0.05)
        seed: random seed for bootstrap (default: 0)
        bootstrap_ci_samples: number of bootstrap samples for confidence intervals (default: 1000)
        file_path: Optional path to save results as CSV (default: None)

    Returns:
        pd.DataFrame: Single-row DataFrame containing all test results
    """
    n = len(p)
    assert p.ndim == q.ndim == 1
    assert p.shape == q.shape
    assert p.dtype == q.dtype == bool
    assert not onp.isnan(p).any()
    assert not onp.isnan(q).any()

    contingency = pd.crosstab(p.astype(int), q.astype(int))
    contingency.index = ["P=0", "P=1"]
    contingency.columns = ["Q=0", "Q=1"]
    results = {}

    prob = contingency / n
    assert onp.isclose(prob.sum().sum(), 1.0)

    results["n00"] = contingency.loc["P=0", "Q=0"]
    results["n01"] = contingency.loc["P=0", "Q=1"]
    results["n10"] = contingency.loc["P=1", "Q=0"]  # violations
    results["n11"] = contingency.loc["P=1", "Q=1"]
    results["n1."] = contingency.loc["P=1"].sum()
    results["n.1"] = contingency.loc[:, "Q=1"].sum()
    results["n"] = n

    results["p00"] = prob.loc["P=0", "Q=0"]
    results["p01"] = prob.loc["P=0", "Q=1"]
    results["p10"] = prob.loc["P=1", "Q=0"]  # violations
    results["p11"] = prob.loc["P=1", "Q=1"]
    results["p1."] = prob.loc["P=1"].sum()
    results["p.1"] = prob.loc[:, "Q=1"].sum()

    """
    Overall implication accuracy
    """
    implication_holds = (~p) | q
    results['overall_accuracy'] = onp.mean(implication_holds)
    results['overall_violation'] = onp.mean(~implication_holds)

    """
    Fisher's exact test (one-sided)
    Testing if P=1 makes Q=1 more likely.
    A significant p-value (e.g., < 0.05) with oddsratio > 1 suggests P=1 genuinely increases the likelihood of Q=1,
    supporting the implication.
    The odds ratio is: n00 * n11 / (n10 * n01)
    """
    results['fisher_oddsratio'], results['fisher_pval'] = fisher_exact(contingency, alternative='greater')
    results['fisher_significant'] = results['fisher_pval'] < fisher_significance

    """
    Confidence intervals for P(Q=1|P=1)
    """
    results['P(Q=1)'] = prob.loc[:, "Q=1"].sum()
    results['P(Q=1|P=1)'] = contingency.loc["P=1", "Q=1"] / contingency.loc["P=1"].sum()
    ci_low, ci_high = proportion_confint(
        contingency.loc["P=1", "Q=1"],
        contingency.loc["P=1"].sum(),
        method='wilson',
        alpha=significance,
    )
    results['P(Q=1|P=1) CI low'] = ci_low
    results['P(Q=1|P=1) CI high'] = ci_high

    """
    Lift: P(Q=1|P=1) / P(Q=1)
    How likely is Q=1 given P=1 compared to its baseline Q=1 (indendent of P).
    If they are independent, lift = 1.
    """
    results['lift'] = (
        (prob.loc["P=1", "Q=1"] / prob.loc["P=1"].sum()) /
        prob.loc[:, "Q=1"].sum()
    )

    def lift_statistic(x, y, axis=0):
        with onp.errstate(divide='ignore', invalid='ignore'):
            lft = (x & y).sum(axis) / x.sum(axis) / y.mean(axis)
            undefined = onp.isclose(x.sum(axis), 0) | onp.isclose(y.mean(axis), 0)
            return onp.where(undefined, 1.0, lft)


    assert onp.isclose(results["lift"], lift_statistic(p, q))

    lift_ci_low, lift_ci_high = bootstrap(
        (p, q),
        lift_statistic,
        vectorized=True,
        paired=True,
        confidence_level=1 - significance,
        n_resamples=bootstrap_ci_samples,
        method='percentile',
        rng=onp.random.default_rng(seed),
    ).confidence_interval

    results['lift CI low'] = lift_ci_low
    results['lift CI high'] = lift_ci_high

    """
    Conviction: P(Q=0) / P(Q=0|P=1)
    How likely is a violation Q=0 given P=1 compared to its baseline Q=0 (indendent of P).
    """
    results['conviction'] = (
        prob.loc[:, "Q=0"].sum() /
        (prob.loc["P=1", "Q=0"] / prob.loc["P=1"].sum())
    )

    def conviction_statistic(x, y, axis=0):
        with onp.errstate(divide='ignore', invalid='ignore'):
            cnvtn = (~y).mean(axis) / ((x & ~y).sum(axis) / x.sum(axis))
            undefined = onp.isclose(x.sum(axis), 0) | onp.isclose((x & ~y).sum(axis), 0)
            return onp.where(undefined, 1.0, cnvtn)

    assert onp.isclose(results["conviction"], conviction_statistic(p, q))

    conviction_ci_low, conviction_ci_high = bootstrap(
        (p, q),
        conviction_statistic,
        vectorized=True,
        paired=True,
        confidence_level=1 - significance,
        n_resamples=bootstrap_ci_samples,
        method='percentile',
        rng=onp.random.default_rng(seed),
    ).confidence_interval

    results['conviction CI low'] = conviction_ci_low
    results['conviction CI high'] = conviction_ci_high


    # Convert to DataFrame with metrics as rows
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['value'])
    results_df.index.name = 'metric'

    # Save to file if path is provided
    if file_path is not None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(file_path)

    return results_df


def plot_implication_conditional_probability(aggregated_df, file_path, text_y_offset=-0.013,
                                             show_yaxis=True, plot_legend=False,
                                             flip_annotations=True, flip_or_offset=0.17, flip_star_offset=0.06,
                                             rcparams=MATPLOTLIB_RCPARAMS, figsize_key="implication_analysis"):
    """
    Plot conditional probability P(Q=1|P=1) with confidence intervals across quantiles.

    Args:
        aggregated_df: DataFrame with quantiles as columns and metrics as index.
                      Expected to contain: 'P(Q=1|P=1)', 'P(Q=1|P=1) CI low', 'P(Q=1|P=1) CI high',
                      'fisher_significant', 'fisher_oddsratio'
        file_path: Path to save the figure (without extension)
        text_y_offset: Vertical shift for OR text and stars (default: -0.015)
        show_yaxis: If True, show y-axis label, ticks, and spine (default: True)
        plot_legend: If True, save legend as a separate PDF file (default: False)
        flip_annotations: If True, place OR annotations below error bars instead of above (default: False)
        flip_or_offset: Offset for OR text when flip_annotations=True (default: 0.03)
        flip_star_offset: Offset for stars when flip_annotations=True (default: 0.06)
        rcparams
        figsize_key
    """
    plt.rcParams.update(rcparams)

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract quantiles and create percentile labels
    quantiles, percentile_labels, xlabel = _quantiles_to_percentile_labels(aggregated_df)

    # Extract metrics
    pq1_p1 = aggregated_df.loc['P(Q=1|P=1)'].values.astype(float)
    ci_low = aggregated_df.loc['P(Q=1|P=1) CI low'].values.astype(float)
    ci_high = aggregated_df.loc['P(Q=1|P=1) CI high'].values.astype(float)
    fisher_sig = aggregated_df.loc['fisher_significant'].values
    odds_ratio = aggregated_df.loc['fisher_oddsratio'].values.astype(float)
    n_edges = aggregated_df.loc['n1.'].values.astype(int)

    # Create combined labels with percentile and number of edges
    combined_labels = [f'{perc}\n({n})' for perc, n in zip(percentile_labels, n_edges)]

    # Create figure
    figsize = FIGSIZE[figsize_key]
    n = len(pq1_p1)
    fig, ax = plt.subplots(1, 1, figsize=(
        (n - 0.1) * figsize["ax_width_per_quantile"],
        figsize["ax_height"],
    ))

    # P(Q=1|P=1) with confidence intervals
    x = onp.arange(len(quantiles))
    line = ax.errorbar(
        x, pq1_p1,
        yerr=[pq1_p1 - ci_low, ci_high - pq1_p1],
        fmt="d:", capsize=2, elinewidth=0.5, markersize=2.5, capthick=0.5,
        color='black', ecolor='black', label='P(Q=1|P=1)'
    )
    line[-1][0].set_clip_on(False)      # Error bars
    line[1][0].set_clip_on(False)       # Caps
    line[1][1].set_clip_on(False)

    # Value taken from p(Q=1) of replogle-1000__1__tikh=0_01 (all perturbations)
    _ = ax.axhline(y=0.13781281281, color='gray', linestyle='--', alpha=0.5, linewidth=0.5, label='Baseline')

    # Add significance stars and odds ratio annotations
    for i, (sig, or_val) in enumerate(zip(fisher_sig, odds_ratio)):
        if flip_annotations:
            base_y = ci_low[i]
            star_y = base_y - flip_star_offset - text_y_offset
            or_y = base_y - flip_or_offset - text_y_offset
        else:
            base_y = ci_high[i]
            star_y = base_y + 0.06 + text_y_offset
            or_y = base_y + 0.10 + text_y_offset

        text_fontsize = 7 if "thesis" in figsize_key else 5
        star_fontsize = 9 if "thesis" in figsize_key else 7
        ax.text(i, or_y, f'OR\n{or_val:.2f}', ha='center', va='center', fontsize=text_fontsize, style='italic', color='#555')
        if sig:
            ax.text(i, star_y, '**', ha='center', va='top', fontsize=star_fontsize, color='black')

    # ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    # ax.set_xlabel(xlabel)

    # ax.set_xlabel(xlabel)
    if show_yaxis:
        ax.set_ylabel('Probability')
        ax.set_xlim(left=-0.5)
    else:
        # Hide y-axis label, tick labels, and ticks (but keep gridlines)
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
        ax.set_xlim(left=-0.5, right=ax.get_xlim()[1] + 0.25)

    ax.set_xticks(x)
    ax.set_xticklabels(combined_labels)
    ax.set_ylim(0.0, 0.85)

    # Save legend separately if requested
    if plot_legend:
        handles = [line]
        # labels = ['P(Q=1|P=1)']
        labels = ['Conditional\nprobability']
        save_legend(file_path.parent / f'{file_path.stem}_legend', handles, labels, ncols=3, rcparams=rcparams)

    # plt.tight_layout()
    plt.savefig(f'{file_path}.pdf', bbox_inches='tight')
    plt.close()


def save_causal_links_csv(causal_graph, gene_names, file_path):
    """
    Save gene-to-gene causal links from adjacency matrix to CSV file.

    Args:
        causal_graph: Adjacency matrix (numpy array or similar) where
                      causal_graph[i, j] indicates a link from gene i to gene j with weight
        gene_names: List of gene names corresponding to the rows/columns of causal_graph
        file_path: Path to save the CSV file (Path or str)

    Returns:
        pd.DataFrame: DataFrame containing the causal links with columns ['source', 'target', 'weight']
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all edges in the causal graph (non-zero entries)
    source_indices, target_indices = onp.where(causal_graph)

    # Extract edge weights
    weights = causal_graph[source_indices, target_indices]

    # Create DataFrame with source, target gene names, and weights
    links_df = pd.DataFrame({
        'source': [gene_names[i] for i in source_indices],
        'target': [gene_names[j] for j in target_indices],
        'weight': weights
    })

    # Save to CSV
    links_df.to_csv(file_path, index=False)

    return links_df


def plot_implication_association_measures(
        aggregated_df, file_path, color_lift=COLORS[6], color_conviction=COLORS[5], plot_legend=False,
        label_lift=None, label_conviction=None, alpha_error=0.7, show_yaxis=True, rcparams=MATPLOTLIB_RCPARAMS,
        figsize_key="implication_analysis",
):
    """
    Plot association measures (lift and conviction) across quantiles.

    Args:
        aggregated_df: DataFrame with quantiles as columns and metrics as index.
                      Expected to contain: 'lift', 'conviction'
        color_lift: color of lift line
        color_conviction: color of conviction line
        file_path: Path to save the figure (without extension)
        plot_legend: If True, save legend as a separate PDF file (default: False)
        label_lift: Optional custom label for lift line (default: 'Lift')
        label_conviction: Optional custom label for conviction line (default: 'Conviction')
        alpha_error: Alpha value for error bars and caps (default: 0.7)
        show_yaxis: If True, show y-axis label, ticks, and spine (default: True)
        rcparams
        figsize_key
    """
    plt.rcParams.update(rcparams)

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract quantiles and create percentile labels
    quantiles, percentile_labels, xlabel = _quantiles_to_percentile_labels(aggregated_df)

    # Extract metrics
    lift = aggregated_df.loc['lift'].values.astype(float)
    lift_ci_low = aggregated_df.loc['lift CI low'].values.astype(float)
    lift_ci_high = aggregated_df.loc['lift CI high'].values.astype(float)

    conviction = aggregated_df.loc['conviction'].values.astype(float)
    conviction_ci_low = aggregated_df.loc['conviction CI low'].values.astype(float)
    conviction_ci_high = aggregated_df.loc['conviction CI high'].values.astype(float)

    n_edges = aggregated_df.loc['n1.'].values.astype(int)

    # Create combined labels with percentile and number of edges
    combined_labels = [f'{perc}\n({n})' for perc, n in zip(percentile_labels, n_edges)]

    # Create figure
    figsize = FIGSIZE[figsize_key]
    n = len(lift)
    fig, ax = plt.subplots(1, 1, figsize=(
        (n - 0.1) * figsize["ax_width_per_quantile"],
        figsize["ax_height"],
    ))

    # Use default labels if not provided
    if label_lift is None:
        label_lift = 'Lift'
    if label_conviction is None:
        label_conviction = 'Conviction'

    # Lift and Conviction with error bars
    x = onp.arange(len(quantiles))
    dodge = 0.0  # Small horizontal offset to prevent overlap

    line_lift = ax.errorbar(
        x - dodge, lift,
        yerr=[lift - lift_ci_low, lift_ci_high - lift],
        fmt="d:", capsize=2, elinewidth=0.5, markersize=2.5, capthick=0.5,
        color=color_lift, ecolor=color_lift, label=label_lift,
    )
    line_lift[-1][0].set_linestyle("-")

    line_lift[-1][0].set_clip_on(False)      # Allow error bars to extend outside axis
    line_lift[1][0].set_clip_on(False)
    line_lift[1][1].set_clip_on(False)

    line_conviction = ax.errorbar(
        x + dodge, conviction,
        yerr=[conviction - conviction_ci_low, conviction_ci_high - conviction],
        fmt="d:", capsize=2, elinewidth=0.5, markersize=2.5, capthick=0.5,
        color=color_conviction, ecolor=color_conviction, label=label_conviction,
    )
    line_conviction[-1][0].set_alpha(alpha_error)  # Error bars
    line_conviction[-1][0].set_clip_on(False)      # Allow error bars to extend outside axis
    line_conviction[1][0].set_clip_on(False)
    line_conviction[1][1].set_clip_on(False)

    _ = ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=0.5, label='Baseline')

    # ax.set_xlabel(xlabel)
    if show_yaxis:
        ax.set_ylabel('Association Strength')
        ax.set_xlim(left=-0.5)
    else:
        # Hide y-axis label and tick labels
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
        ax.set_xlim(left=-0.5, right=ax.get_xlim()[1] + 0.25)

    ax.set_xticks(x)
    ax.set_xticklabels(combined_labels)
    ax.set_ylim(0.0, 5.4)

    # Save legend separately if requested
    if plot_legend:
        # handles = [line_lift, line_conviction, line_baseline]
        # labels = ['Lift', 'Conviction', 'Baseline']
        handles = [line_lift, line_conviction]
        labels = [label_lift, label_conviction]
        save_legend(file_path.parent / f'{file_path.stem}_legend', handles, labels, ncols=3, rcparams=rcparams)

    # plt.tight_layout()
    plt.savefig(f'{file_path}.pdf', bbox_inches='tight')
    plt.close()

