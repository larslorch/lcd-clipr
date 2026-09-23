import copy
import numpy as onp
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from lcd_clipr.experiment.plot_config import MATPLOTLIB_RCPARAMS, GI_DEF_METRIC, BOXPLOT_TRANSPARENCY, FIGSIZE, GENE_LABEL_RENAMES
from lcd_clipr.experiment.plot_utils import save_legend
from lcd_clipr.utils.gi import (
    itup,
    assemble_gi_dataset,
    compute_gi_scores_from_raw,
    mask_from_seeding,
    classify_interaction_types,
)
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import to_rgb
from seaborn.utils import desaturate


def plot_differential_expressions(
    save_path,
    preds,
    train_datasets,
    test_datasets,
    remaining_datasets,
    data_seeding,
    *,
    label_pred,
    color_pred,
    cols=None,
    plot_legend=False,
    wide_format=False,
    large_fonts=False,
    rasterize_fliers_and_hatches=True,
    rcparams=MATPLOTLIB_RCPARAMS,
):
    # determine true GI type for each test double
    true_gi_dataset = assemble_gi_dataset(
        test_datasets,
        train_datasets,
        remaining_datasets,
    )
    true_gis = compute_gi_scores_from_raw(
        true_gi_dataset.data_normalized,
        true_gi_dataset.intv,
        train_datasets.data_normalized[0],
        mask_from_seeding(data_seeding["gi_genes"], data_seeding["var_names"]),
    )
    for double in true_gis.keys():
        true_gis[double]["type"] = classify_interaction_types(
            true_gis[double],
            data_seeding["gi_thresholds"],
        )
    assert all(onp.isclose(intv.sum(), 2) and itup(intv) in true_gis for intv in test_datasets.intv)

    # plot
    for j, test_intv in enumerate(test_datasets.intv):
        for gi_type, gi_metric in GI_DEF_METRIC:
            tar_a, tar_b = itup(test_intv)
            gi = true_gis[(tar_a, tar_b)]
            if not gi["type"][gi_type]:
                continue

            gi_score = gi[gi_metric]
            file_name = f"{gi_type}__score={gi_score:.4f}".replace(".", "_")

            differential_expression_plot(
                save_path / file_name,
                pred=preds["norm_test"][j],
                true=test_datasets.data_normalized[j],
                ctrl=train_datasets.data_normalized[0],
                de_mask=test_datasets.differential_expression_mask[j].astype(bool),
                intv_mask=test_datasets.intv[j].astype(bool),
                var_names=[GENE_LABEL_RENAMES.get(g, g) for g in data_seeding["var_names"]], # with tighter renaming
                label_pred=label_pred,
                color_pred=color_pred,
                cols=cols,
                plot_legend=plot_legend,
                wide_format=wide_format,
                large_fonts=large_fonts,
                rasterize_fliers_and_hatches=rasterize_fliers_and_hatches,
                rcparams=rcparams,
            )


def _compute_whisker_bounds(data, column_order, extend=0.7):
    """Compute whisker bounds based on IQR for boxplot y-axis limits."""
    pivoted = data.pivot(columns=['variable', 'method'], values='value')
    pivoted = pivoted.reindex(columns=column_order, level=0)

    iqrs = pivoted.quantile(0.75) - pivoted.quantile(0.25)
    whisker_max = pivoted.quantile(0.75) + extend * iqrs
    whisker_min = pivoted.quantile(0.25) - extend * iqrs

    return whisker_min, whisker_max


def differential_expression_plot(
    file_path,
    *,
    pred,
    true,
    ctrl,
    de_mask,
    intv_mask,
    var_names,
    label_pred,
    color_pred,
    color_true="white",
    cols=None,
    vertical_space_around_whiskers=0.5, # DE units
    flierprops_marker=".",
    rotation=45,
    plot_legend=False,
    hide_xlabels=False,
    vertical_xlabels=False,
    wide_format=False,
    large_fonts=False,
    rasterize_fliers_and_hatches=False,
    rcparams=MATPLOTLIB_RCPARAMS,
):
    assert pred.ndim == 2
    assert ctrl.ndim == 2
    assert true.ndim == 2

    assert de_mask.dtype == bool
    assert intv_mask.dtype == bool

    mask = intv_mask | de_mask

    assert pred.shape[-1] == ctrl.shape[-1] == true.shape[-1]
    if cols is None:
        cols = mask.sum()

    # variable subselection: first intv, then DEGs
    idx = onp.where(de_mask)[0]
    intv = onp.where(intv_mask)[0]
    idx = onp.concatenate([intv, idx[~onp.isin(idx, intv)]])
    idx = idx[:cols]

    # compute differential expressions and make dataframe of subsets
    ctrl_mean = onp.mean(ctrl, axis=0)
    df_pred = pd.DataFrame(pred - ctrl_mean, columns=var_names).iloc[:, idx]
    df_true = pd.DataFrame(true - ctrl_mean, columns=var_names).iloc[:, idx]

    df_pred["method"] = "pred"
    df_true["method"] = "true"

    # concatenate the dataframes and melt them into long form
    melted_df = pd.melt(
        pd.concat([df_pred, df_true]),
        id_vars='method',
        var_name='variable',
        value_name='value',
    )

    # filter data for the intervened columns and other columns separately
    first_cols_order = sorted(var_names[i] for i in idx if intv_mask[i])
    remaining_cols_order = sorted(var_names[i] for i in idx if not intv_mask[i] and de_mask[i])
    assert not set(first_cols_order).intersection(set(remaining_cols_order))

    first_cols = melted_df[melted_df['variable'].isin(first_cols_order)]
    remaining_cols = melted_df[melted_df['variable'].isin(remaining_cols_order)]

    # boxplot
    rcparams = {**copy.deepcopy(rcparams), 'hatch.linewidth': 0.3}  # thin hatch lines for small plots
    plt.rcParams.update(rcparams)

    if large_fonts:
        figsize = (
            FIGSIZE["fig_2_differential_expression"]["ax_width_large"],
            FIGSIZE["fig_2_differential_expression"]["ax_height_large"],
        )

    elif wide_format:
        figsize = (
            FIGSIZE["fig_2_differential_expression"]["ax_width_wide"],
            FIGSIZE["fig_2_differential_expression"]["ax_height"],
        )

    else:
        figsize = (
            FIGSIZE["fig_2_differential_expression"]["ax_width"],
            FIGSIZE["fig_2_differential_expression"]["ax_height"],
        )


    fig, (ax, ax2) = plt.subplots(
        1, 2,
        figsize=figsize,
        gridspec_kw={
            'width_ratios': [len(intv), len(idx) - len(intv)],
            'wspace': 0.0,
        },
    )

    shared_kwargs = dict(
        x='variable',
        y='value',
        hue='method',
        hue_order=['pred', 'true'],
        palette=[color_pred, color_true],
        saturation=0.8,
    )

    shared_kwargs_boxplot = dict(
        flierprops={
            "marker": flierprops_marker,
            "markersize": 2.0,
            "color": "black",
            "markeredgewidth": 0,
        },
        medianprops={"color": "black", "linewidth": 1},
        boxprops={"edgecolor": "black", "linewidth": 0.5},
        whiskerprops={"color": "black", "linewidth": 0.5},
        capprops={"color": "black", "linewidth": 0.5},
        # meanprops={"linestyle": '--', "color": 'black'},
        # showmeans=True,
        # meanline=True,
        width=0.8,
    )

    # first columns
    sns_plot = sns.boxplot(
        data=first_cols,
        ax=ax,
        order=first_cols_order,
        **shared_kwargs,
        **shared_kwargs_boxplot,
    )

    # remaining columns
    sns_plot_2 = sns.boxplot(
        data=remaining_cols,
        ax=ax2,
        order=remaining_cols_order,
        **shared_kwargs,
        **shared_kwargs_boxplot,
    )

    # rasterize fliers via a per-axes zorder layer so PDFs scroll fast in browsers
    raster_zorder = 0.5
    if rasterize_fliers_and_hatches:
        for axis in (ax, ax2):
            for line in axis.lines:
                if line.get_marker() == flierprops_marker:
                    line.set_zorder(raster_zorder - 0.1)
            axis.set_rasterization_zorder(raster_zorder)

    # control line and legends
    ax.yaxis.grid(False)

    # save legend info before removing
    handles, labels = ax.get_legend_handles_labels()

    # apply hatch to every artist representing the 'true' hue (boxplot patches and legend handles)
    # seaborn's `saturation` arg desaturates the palette before drawing,
    # so we match patches against the desaturated color rather than the raw `color_true`.
    # Patch order from seaborn is not reliable across versions / can be interleaved per x-category, hence color-matching
    true_rgb = to_rgb(desaturate(color_true, shared_kwargs["saturation"]))

    def _is_true_artist(artist):
        r, g, b, _ = artist.get_facecolor()
        return all(abs(x - y) < 1e-3 for x, y in zip((r, g, b), true_rgb))

    def _hatch_true(artist):
        artist.set_hatch('////////////')
        artist.set_edgecolor('black')

    for snsp in [sns_plot, sns_plot_2]:
        for patch in snsp.patches:
            if _is_true_artist(patch):
                _hatch_true(patch)
            if rasterize_fliers_and_hatches:
                patch.set_zorder(raster_zorder - 0.1)
            r, g, b, _ = patch.get_facecolor()
            patch.set_facecolor((r, g, b, BOXPLOT_TRANSPARENCY))

    for h in handles:
        if _is_true_artist(h):
            _hatch_true(h)

    ax.get_legend().remove()
    ax2.get_legend().remove()

    ax.axhline(y=0, color="black", linestyle='-', linewidth=0.5, zorder=100)
    ax2.axhline(y=0, color="black", linestyle='-', label='control', linewidth=0.5, zorder=100)

    # move y-axis of ax2 to the right side
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")


    # set y limits based on boxplot whiskers and ignore outliers
    first_whisker_min, first_whisker_max = _compute_whisker_bounds(first_cols, first_cols_order)
    remaining_whisker_min, remaining_whisker_max = _compute_whisker_bounds(remaining_cols, remaining_cols_order)

    ax.set_ylim(
        first_whisker_min.min() - vertical_space_around_whiskers,
        first_whisker_max.max() + vertical_space_around_whiskers,
    )
    ax2.set_ylim(
        remaining_whisker_min.min() - vertical_space_around_whiskers,
        remaining_whisker_max.max() + vertical_space_around_whiskers,
    )

    # formatting
    ax.set_xlabel('')
    ax2.set_xlabel('')

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    # spines
    ax.spines.top.set_visible(False)
    ax2.spines.top.set_visible(False)

    ax.spines.bottom.set_visible(True)
    ax2.spines.bottom.set_visible(True)

    ax.spines.left.set_visible(True)
    ax2.spines.left.set_visible(False)

    ax.spines.right.set_visible(False)
    ax2.spines.right.set_visible(True)

    ylabel = "Log fold change"
    ylabel += "\n(perturbed genes)"

    ax.set_ylabel(ylabel, fontsize=rcparams['axes.labelsize'])

    ylabel2 = "Log fold change"
    ylabel2 += "\n(marker genes)"

    ax2.set_ylabel(ylabel2, fontsize=rcparams['axes.labelsize'])

    if hide_xlabels:
        ax.set_xticks([])
        ax2.set_xticks([])
    else:
        xtick_rotation = 90 if vertical_xlabels else rotation
        xtick_ha = 'center' if vertical_xlabels else 'right'

        # make intervened column labels bold
        x_labels = ax.get_xticklabels()
        for j in range(len(intv)):
            x_labels[j].set_fontweight('bold')
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=xtick_rotation, ha=xtick_ha, fontsize=rcparams['xtick.labelsize'])

        ax2.set_xticks(range(len(ax2.get_xticklabels())))
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=xtick_rotation, ha=xtick_ha, fontsize=rcparams['xtick.labelsize'])

    # align y=0 on ax with y=0 on ax2
    _, y1 = ax.transData.transform((0, 0))
    _, y2 = ax2.transData.transform((0, 0))
    inv = ax.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y2 - y1))
    miny, maxy = ax.get_ylim()

    # ax.set_ylim(miny + dy, maxy + dy)
    # increase ylims such that previous ylims are still visible
    scaling = max(abs(miny / (miny + dy)), abs(maxy / (maxy + dy))) + 0.1
    new_miny = scaling * (miny + dy)
    new_maxy = scaling * (maxy + dy)
    if new_miny <= miny < 0 < maxy <= new_maxy:
        ax.set_ylim(new_miny, new_maxy)
    else:
        # alignment shift would push y=0 outside [miny, maxy], so just align zeros directly
        ax.set_ylim(miny + dy, maxy + dy)

    # highlight shading to visually separate intervened targets
    xlim = ax.get_xlim()
    ax.axvspan(-1, len(intv) - 0.5, alpha=0.2, facecolor='grey', edgecolor=None, linewidth=0.0, zorder=-100)
    ax.set_xlim(xlim)

    # save legend separately
    if plot_legend:
        labels_map = {"pred": label_pred, "true": "True"}
        labels_renamed = [labels_map.get(label, label) for label in labels]
        save_legend((file_path.parent / f"legend-{file_path.stem}"), handles, labels_renamed, ncols=1, rcparams=rcparams)

    # save
    if hide_xlabels:
        suffix = "_".join(str(i) for i in onp.where(intv_mask)[0])
    else:
        suffix = "_".join(first_cols_order)
    suffix = suffix.translate(str.maketrans("", "", "${}"))
    file_path_with_gene_names = file_path.parent / (file_path.stem + "_" + suffix)
    file_path_with_gene_names.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(file_path_with_gene_names.with_suffix(".pdf"), format="pdf", bbox_inches='tight')
    plt.close()

    return
