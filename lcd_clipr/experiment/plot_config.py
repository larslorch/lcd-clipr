# import matplotlib
# matplotlib.use('Agg')  # consistent with cluster

import cmcrameri

PTS_TO_INCH = 1 / 72.27

MATPLOTLIB_RCPARAMS = {
    # 'font.size': 6,
    # 'font.family': 'serif',
    # 'font.serif': ['Times'],
    # 'axes.titlesize': 7,
    # 'axes.labelsize': 6,
    # 'xtick.labelsize': 6,
    # 'ytick.labelsize': 6,
    # 'legend.fontsize': 5.5,
    # 'text.usetex': True,

    # # min font size 5
    # 'font.size': 5,
    # 'font.family': 'sans-serif',
    # 'font.sans-serif': 'Arial',
    # 'axes.titlesize': 6,
    # 'axes.labelsize': 5,
    # 'xtick.labelsize': 5,
    # 'ytick.labelsize': 5,
    # 'legend.fontsize': 5,

    # min font size 6
    'font.size': 6,
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    'axes.titlesize': 6,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,

    'axes.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    'xtick.major.size': 2.0,
    'ytick.major.size': 2.0,
    'xtick.minor.size': 1.0,
    'ytick.minor.size': 1.0,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,

    'boxplot.boxprops.color': 'black',
    'boxplot.boxprops.linestyle': '-',
    'boxplot.boxprops.linewidth': 0.5,
    'boxplot.capprops.color': 'black',
    'boxplot.capprops.linewidth': 0.5,
    'boxplot.whiskerprops.color': 'black',
    'boxplot.whiskerprops.linewidth': 0.5,
    'boxplot.flierprops.marker': 'o',
    'boxplot.flierprops.color': 'black',
    'boxplot.flierprops.markerfacecolor': 'black',
    'boxplot.flierprops.markersize': 2,
    # 'boxplot.medianprops.color': 'red',  # Change the color for ease of identification
    'boxplot.medianprops.color': 'black',  # Change the color for ease of identification
    'boxplot.medianprops.linewidth': 1.0,
    'boxplot.meanprops.color': 'black',
    'boxplot.meanprops.linewidth': 0.5,

    'legend.frameon': False,
    'legend.loc': 'best',

    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,

    'axes.unicode_minus': False,  # use hyphen instead of Unicode minus

    'pdf.fonttype': 42,  # ensuring fonts are embedded in PDFs
    'ps.fonttype': 42,

    'figure.dpi': 300,
    'figure.facecolor': 'none',
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'axes.grid': False,
}


FIGSIZE = {
    "grid": {
        "size_per_var": 1.0,
        "grid_envs": 3,
        "grid_per_env": 1,
        "grid_cols": 12,
    },
    "metrics_boxplot": {
        "ax_width": 78 * PTS_TO_INCH,
        "ax_height": 165 * PTS_TO_INCH,
    },
    "metrics_boxplot_short": {
        "ax_width": 78 * PTS_TO_INCH,
        "ax_height": 78 * PTS_TO_INCH,
    },
    "metrics_boxplot_hvg": {
        "ax_width": 90 * PTS_TO_INCH,
        "ax_height": 105 * PTS_TO_INCH,
    },
    "metrics_boxplot_side_by_side": {
        "ax_width": 210 * PTS_TO_INCH,
        "ax_height": 145 * PTS_TO_INCH,
    },
    "metrics_boxplot_ablation": {
        "ax_width": 122 * PTS_TO_INCH,
        "ax_height": 145 * PTS_TO_INCH,
    },
    "metrics_boxplot_comparison": {
        "ax_width": 280 * PTS_TO_INCH,
        "ax_height": 65 * PTS_TO_INCH,
    },
    "fig_2_differential_expression": {
        "ax_width": 280 * PTS_TO_INCH,
        "ax_width_wide": 380 * PTS_TO_INCH,
        "ax_height": 65 * PTS_TO_INCH,
        "ax_width_large": 300 * PTS_TO_INCH,
        "ax_height_large": 90 * PTS_TO_INCH,
},
    "metric_pairwise_comparison": {
        "ax_width": 53 * PTS_TO_INCH,
        "ax_height": 53 * PTS_TO_INCH,
    },
    "metric_causal_scaling": {
        "ax_width": 110 * PTS_TO_INCH,
        "ax_height": 110 * PTS_TO_INCH,
    },
    "metric_causal_scaling_thesis": {
        "ax_width": 130 * PTS_TO_INCH,
        "ax_height": 130 * PTS_TO_INCH,
    },
    "metric_causal_barplot": {
        "ax_width": 110 * PTS_TO_INCH,
        "ax_height": 55 * PTS_TO_INCH,
    },
    "metric_causal_barplot_thesis": {
        "ax_width": 130 * PTS_TO_INCH,
        "ax_height": 65 * PTS_TO_INCH,
    },
    "metric_causal_effect_correlation": {
        "ax_width": 125 * PTS_TO_INCH,
        "ax_height": 130 * PTS_TO_INCH,
    },
    "legend": {
        "ax_width_col": 20 * PTS_TO_INCH,
        "ax_height": 40 * PTS_TO_INCH,
    },
    "colorbar": {
        "short_side": 10 * PTS_TO_INCH,
        "long_side": 100 * PTS_TO_INCH,
    },
    "colorbar_two": {
        "short_side": 5 * PTS_TO_INCH,
        "long_side": 50 * PTS_TO_INCH,
    },
    "adjacency_matrix": {
        "ax_width": 168 * PTS_TO_INCH,
        "ax_height": 168 * PTS_TO_INCH,
    },
    "adjacency_matrix_v2": {
        "ax_width": 168 * PTS_TO_INCH,
        "ax_height": 168 * PTS_TO_INCH,
        "ax_height_dendrogram": 30 * PTS_TO_INCH,
        "ax_height_targets": 5 * PTS_TO_INCH,
    },
    "adjacency_matrix_v3": {
        "ax_width": 125 * PTS_TO_INCH,
        "ax_height": 125 * PTS_TO_INCH,
        "ax_height_dendrogram": 30 * PTS_TO_INCH,
        "ax_height_targets": 5 * PTS_TO_INCH,
    },
    "adjacency_matrix_vthesis": {
        "ax_width": 145 * PTS_TO_INCH,
        "ax_height": 145 * PTS_TO_INCH,
        "ax_height_dendrogram": 30 * PTS_TO_INCH,
        "ax_height_targets": 5 * PTS_TO_INCH,
    },
    "adjacency_matrix_cbar_height": 25 * PTS_TO_INCH,
    "pathway_correlations_matrix": {
        "ax_width": 160 * PTS_TO_INCH,
        "ax_height": 120 * PTS_TO_INCH,
    },
    "small_roc": {
        "ax_width": 105 * PTS_TO_INCH,
        "ax_height": 95 * PTS_TO_INCH,
    },
    "2d_embedding": {
        "ax_width": 250 * PTS_TO_INCH,
        "ax_height": 250 * PTS_TO_INCH,
    },
    "2d_embedding_visualize_double": {
        "ax_width": 90 * PTS_TO_INCH,
        "ax_height": 90 * PTS_TO_INCH,
    },
    "triple_properties_embedding": {
        "ax_width": 120 * PTS_TO_INCH,
        "ax_height": 105 * PTS_TO_INCH,
    },
    "umap_embedding_ref_vectors": {
        "ax_width": 230 * PTS_TO_INCH,
        "ax_height": 230 * PTS_TO_INCH,
    },
    "stats_analysis": {
        "ax_width": 250 * PTS_TO_INCH,
        "ax_height": 170 * PTS_TO_INCH,
    },
    "mat_spectrum_analysis": {
        "ax_width": 120 * PTS_TO_INCH,
        "ax_height": 90 * PTS_TO_INCH,
    },
    "eigenvalue_spectrum_analysis": {
        "ax_width": 230 * PTS_TO_INCH,
        "ax_height": 180 * PTS_TO_INCH,
    },
    "split_agreement_analysis": {
        "ax_width": 120 * PTS_TO_INCH,
        "ax_height": 120 * PTS_TO_INCH,
    },
    "figure_1_density": {
        "ax_width": 150 * PTS_TO_INCH,
        "ax_height": 150 * PTS_TO_INCH,
    },
    "figure_1_contour": {
        "ax_width": 120 * PTS_TO_INCH,
        "ax_height": 120 * PTS_TO_INCH,
    },
    "figure_1_clipr": {
        "ax_width": 120 * PTS_TO_INCH,
        "ax_height": 120 * PTS_TO_INCH,
    },
    "causal_analysis_dotplot": {
        "ax_width_per_dot": 13 * PTS_TO_INCH,
        "ax_height_per_dot": 22 * PTS_TO_INCH,
        "ax_width_legend": 20 * PTS_TO_INCH,
        "ax_height_legend": 20 * PTS_TO_INCH,
        "ax_width_colobar": 5 * PTS_TO_INCH,
    },
    "causal_analysis_chord": {
        "ax_width": 210 * PTS_TO_INCH,
        "ax_height": 210 * PTS_TO_INCH,
    },
    "implication_analysis": {
        "ax_width_per_quantile": 22 * PTS_TO_INCH,
        "ax_height": 95 * PTS_TO_INCH,
    },
    "de_graph_analysis" : {
        "ax_width_per_path_len": 22 * PTS_TO_INCH,
        "ax_height": 95 * PTS_TO_INCH,
    },
    "causal_matrix_fig4": {
        "ax_width": 100 * PTS_TO_INCH,
        "ax_height": 100 * PTS_TO_INCH,
        "ax_height_dendrogram": 10 * PTS_TO_INCH,
        "ax_height_targets": 5 * PTS_TO_INCH,
    },
    "si_comparison_gi_metrics": {
        "ax_width": 400 * PTS_TO_INCH,
        "ax_height": 280 * PTS_TO_INCH,
    },
    "si_comparison_gi_metrics_by_category": {
        "ax_width": 400 * PTS_TO_INCH,
        "ax_height": 230 * PTS_TO_INCH,
    },
    "si_causal_big": {
        "ax_width": 512 * PTS_TO_INCH,
        "ax_height": 512 * PTS_TO_INCH,
        "ax_height_dendrogram": 10 * PTS_TO_INCH,
        "ax_height_targets": 10 * PTS_TO_INCH,
    },
    "si_causal": {
        "ax_width": 316 * PTS_TO_INCH,
        "ax_height": 316 * PTS_TO_INCH,
        "ax_height_dendrogram": 10 * PTS_TO_INCH,
        "ax_height_targets": 10 * PTS_TO_INCH,
    },
    "si_causal_small": {
        "ax_width": 195 * PTS_TO_INCH,
        "ax_height": 195 * PTS_TO_INCH,
        "ax_height_dendrogram": 10 * PTS_TO_INCH,
        "ax_height_targets": 10 * PTS_TO_INCH,
    },
    # thesis
    "implication_analysis_thesis": {
        "ax_width_per_quantile": 27 * PTS_TO_INCH,
        "ax_height": 120 * PTS_TO_INCH,
    },
    "causal_matrix_fig4_thesis": {
        "ax_width": 145 * PTS_TO_INCH,
        "ax_height": 145 * PTS_TO_INCH,
        "ax_height_dendrogram": 10 * PTS_TO_INCH,
        "ax_height_targets": 10 * PTS_TO_INCH,
    },
    "si_causal_big_thesis": {
        "ax_width": 300 * PTS_TO_INCH,
        "ax_height": 300 * PTS_TO_INCH,
        "ax_height_dendrogram": 10 * PTS_TO_INCH,
        "ax_height_targets": 10 * PTS_TO_INCH,
    },
    "si_causal_small_thesis": {
        # CLIPR components plots
        "ax_width": 130 * PTS_TO_INCH,  # 5 per page
        "ax_height": 130 * PTS_TO_INCH,
        "ax_height_dendrogram": 10 * PTS_TO_INCH,
        "ax_height_targets": 10 * PTS_TO_INCH,
    },
    "causal_analysis_dotplot_thesis": {
        "ax_width_per_dot": 14 * PTS_TO_INCH,
        "ax_height_per_dot": 25 * PTS_TO_INCH,
        "ax_width_legend": 20 * PTS_TO_INCH,
        "ax_height_legend": 20 * PTS_TO_INCH,
        "ax_width_colobar": 5 * PTS_TO_INCH,
        "label_fontsize": 7,
        "spread_label_kwargs" : {"space_per_line": 0.7, "extra_space_per_label": 0.42, "label_y": -0.16},
        "colorbar_horizontal": 1.1, # if provided, sets width of horizontal colorbar
        "legend_horizontal": True,
    },
    "causal_analysis_chord_thesis": {
        "ax_width": 335 * PTS_TO_INCH,
        "ax_height": 335 * PTS_TO_INCH,
    },
    # additional analyses
    "causal_l_curve": {
        "ax_width": 350 * PTS_TO_INCH,
        "ax_height": 160 * PTS_TO_INCH,
    },
    "causal_l_curve_adjacency_matrix": {
        "ax_width": 140 * PTS_TO_INCH,
        "ax_height": 140 * PTS_TO_INCH,
        "ax_height_dendrogram": 10 * PTS_TO_INCH,
        "ax_height_targets": 10 * PTS_TO_INCH,
    },
    "causal_gcv_curve": {
        "ax_width": 160 * PTS_TO_INCH,
        "ax_height": 160 * PTS_TO_INCH,
    },
    "noise_statistics_analysis": {
        "ax_width": 150 * PTS_TO_INCH,
        "ax_height": 140 * PTS_TO_INCH,
    },
}


COLORS_0 = [
    "#DF000D", # red
    "#23DD20", # green
    "#1ACBF7", # light blue
    "#FB0D89", # pink
    "#7600DA", # purple
    "#FC6608", # orange
    "#001AFD", # dark blue
]

COLORS_1 = [
    "#e41a1c", # 0: red
    "#377eb8", # 1: blue
    "#4daf4a", # 2: green
    "#984ea3", # 3: purple
    "#ff7f00", # 4: orange
    "#a65628", # 5: brown
    "#999999", # 6: grey
    "#ffff33", # 7: yellow
]

# https://www.nature.com/articles/nmeth.1618
COLORS_2 = [
    "#000000", # 0: black
    "#e69f00", # 1: orange
    "#56b4e9", # 2: sky blue
    "#009e73", # 3: blueish green
    "#f0e442", # 4: yellow
    "#0072b2", # 5: blue
    "#d55e00", # 6: vermillion (red)
    "#cc79a7", # 7: reddish purple
]

# COLORS = COLORS_0
# COLORS = COLORS_1
COLORS = COLORS_2

COLOR_RED = COLORS_2[6]
COLOR_BLUE = COLORS_2[5]

COLOR_OURS = COLORS[6]
COLOR_OURS_ABLATION_GROUP_1 = "dimgray"
COLOR_OURS_ABLATION_GROUP_2 = COLORS_2[5]

MARKERS = ["d"] * 10

BOXPLOT_TRANSPARENCY_SWARM = 0.4
BOXPLOT_TRANSPARENCY = 0.75

METHODS_CONFIG = [
    ("ctrl", COLORS[0], "Control"),
    ("perturbed", COLORS[1], "All Perturb."),
    ("salt", COLORS[2], "SALT"),
    ("peper", COLORS[3], "PEPER"),
    ("cpa",  COLORS[7], "CPA"),
    ("gears",  COLORS[5], "GEARS"),
    ("ours-mlp", COLORS[6], "LCD (ours)"),
    # ("ours-mlp", COLORS[6], "LCD"),
]

METHODS_DEFAULT_CONFIG = (COLORS[0], "default")

TRUE = "__true__"
REF = "__ref__"

METHODS_CONFIG_DICT = {k: tup for k, *tup in METHODS_CONFIG}

GI_SCORE_TO_LABEL = {
    "mag": r"$\sqrt{c_a^2 + c_b^2}$",
    "ts_linear_dcor": r"$\mathrm{dcorr}(\delta_{a,b}, c_a \delta_a + c_b \delta_b)$",
    "dcor": r"$\mathrm{dcorr}([\delta_a, \delta_b], \delta_{a,b})$",
    "dcor_singles": r"$\mathrm{dcorr}(\delta_a, \delta_b)$",
    "dcor_ratio": r"$\frac{\min\{\mathrm{dcorr}(\delta_a, \delta_{a,b}), \mathrm{dcorr}(\delta_b, \delta_{a,b})\}}{\max\{\mathrm{dcorr}(\delta_a, \delta_{a,b}), \mathrm{dcorr}(\delta_b, \delta_{a,b})\}}$",
    "dominance": r"$\log_{10}(|c_a|/|c_b|)$",
}

GI_DEF_METRIC = [
    ("synergy", "mag"),
    ("additive", "mag"),
    ("suppression", "mag"),
    ("neomorphism", "ts_linear_dcor"),
    ("epistasis", "dcor_ratio"),
    ("redundancy", "dcor"),
]

GI_NAMES = {
    "synergy": "Synergy",
    "additive": "Additive",
    # "suppression": "Suppression",
    "suppression": "Suppress.",
    # "neomorphism": "Neomorphism",
    "neomorphism": "Neomorph.",
    "epistasis": "Epistasis",
    "redundancy": "Redundancy",
}

GI_METRICS_MIN = ["synergy", "redundancy", "additive"]  # the larger the more significant
GI_METRICS_MAX = ["suppression", "neomorphism", "epistasis", "additive"]  # the smaller the more significant


GI_CMAPS = {
    "synergy": (cmcrameri.cm.bamako_r, (0.7, 2.0)),
    "additive": (cmcrameri.cm.grayC_r, (None, None)),
    "suppression": (cmcrameri.cm.lajolla_r, (-1.4, -0.8)),
    "neomorphism": (cmcrameri.cm.devon_r, (-0.97, -0.8)),
    "epistasis": (cmcrameri.cm.acton_r, (None, None)),
    "redundancy": (cmcrameri.cm.navia_r, (0.75, 1.0)),
}

# For tighter plotting in some cases
GENE_LABEL_RENAMES = {
    "RP11-301G19.1": "ERVH-3",
}

METHODS_EXCLUDED_FROM_NORMALIZATION = [
    "ctrl",
    "nonctrl",
    "perturbed",
]

METHODS_COLORS = {k: tup[-2] for k, *tup in METHODS_CONFIG}
METHODS_NAMES = {k: tup[-1] for k, *tup in METHODS_CONFIG}


METRIC_MASKS = [
    ("", ""),
    ("_20", " (top 20)"),
    ("_hvg_0-5", " (top 5% HVG)"),
    ("_hvg_5-20", " (top 5-20% HVG)"),
    ("_hvg_20-100", " (bottom 80% 20-100%)"),
]

# Expression-based metrics: name, label, higher_better, higher_prec
EXPRESSION_METRICS = [
    ("ed",              "Energy Distance",          False, False),
    ("mmd",             "MMD",                      False, True),
    ("w1_eltwise",      "$W_1$, elementwise",       False, False),
    ("rmse_m",          "RMSE",                     False, False),
    ("pearson_de",      "Pearson $r$",              True,  False),
    ("pearson_systema", "Pearson $\\Delta$",        True,  False),
]

METRICS_CONFIG = []
METRICS_PAIRWISE_COMPARISONS = []  # should only add metrics here that are never nan
METRICS_HIGHER_BETTER = []
METRICS_HIGHER_PREC = []

for _suffix, _label_suffix in METRIC_MASKS:
    for _name, _label, _higher_better, _higher_prec in EXPRESSION_METRICS:
        _key = f"{_name}{_suffix}"
        METRICS_CONFIG.append((_key, f"{_label}{_label_suffix}"))
        METRICS_PAIRWISE_COMPARISONS.append(_key)
        if _higher_better:
            METRICS_HIGHER_BETTER.append(_key)
        if _higher_prec:
            METRICS_HIGHER_PREC.append(_key)

# state-space distribution metrics
STATE_METRICS = [
    ("state_rmse_m",     "RMSE (state means)",          False, False),
    ("state_rmse_s",     "RMSE (state std. devs.)",     False, False),
    ("state_bias_s",     "Bias (state std. devs.)",     False, False),
    ("state_pearson_de", "Pearson $r$ (state)",         True,  False),
    ("state_ed",         "Energy Distance (state)",     False, False),
    ("state_mmd",        "MMD (state)",                 False, True),
]

for _name, _label, _higher_better, _higher_prec in STATE_METRICS:
    METRICS_CONFIG.append((_name, _label))
    if _higher_better:
        METRICS_HIGHER_BETTER.append(_name)
    if _higher_prec:
        METRICS_HIGHER_PREC.append(_name)

# DE classification metrics
CLASSIFY_DE_METRICS = [
    ("classify_de_prec", "DE Precision"),
    ("classify_de_recall", "DE Recall"),
    ("classify_de_f1", "DE F1 score"),
]

for _suffix, _label_suffix in METRIC_MASKS:
    for _name, _label in CLASSIFY_DE_METRICS:
        _key = f"{_name}{_suffix}"
        METRICS_CONFIG.append((_key, f"{_label}{_label_suffix}"))
        METRICS_HIGHER_BETTER.append(_key)

# top-k DE classification metrics
for _top_k in [20, 100]:
    for _rank_by, _rank_label in [
        ("delta", "log fold change"),
        ("pval", "P-value"),
    ]:
        for _name, _label in [
            ("classify_topk_auroc", "AUROC @{top_k} (by {rank_label})"),
            ("classify_topk_ap", "Average precision @{top_k} (by {rank_label})"),
            ("classify_topk_precision", "Precision @{top_k} (by {rank_label})"),
        ]:
            _key = f"{_name}_{_rank_by}_{_top_k}"
            METRICS_CONFIG.append((_key, _label.format(rank_label=_rank_label, top_k=_top_k)))
            METRICS_HIGHER_BETTER.append(_key)

# GI metrics
METRICS_CONFIG += [
    ("gi_se_mag", f"Squared error {GI_SCORE_TO_LABEL['mag']}"),
    ("gi_se_linear_dcor", f"Squared error {GI_SCORE_TO_LABEL['ts_linear_dcor']}"),
    ("gi_se_dcor_ratio", f"Squared error {GI_SCORE_TO_LABEL['dcor_ratio']}"),
    ("gi_se_dcor", f"Squared error {GI_SCORE_TO_LABEL['dcor']}"),
    ("gi_pearson_mag", f"Pearson $r$ {GI_SCORE_TO_LABEL['mag']}"),
    ("gi_pearson_linear_dcor", f"Pearson $r$ {GI_SCORE_TO_LABEL['ts_linear_dcor']}"),
    ("gi_pearson_dcor_ratio", f"Pearson $r$ {GI_SCORE_TO_LABEL['dcor_ratio']}"),
    ("gi_pearson_dcor", f"Pearson $r$ {GI_SCORE_TO_LABEL['dcor']}"),
    #
    ("gim_syn_mmd_20", f"{GI_NAMES['synergy']}"),
    ("gim_sup_mmd_20", f"{GI_NAMES['suppression']}"),
    ("gim_neo_mmd_20", f"{GI_NAMES['neomorphism']}"),
    ("gim_add_mmd_20", f"{GI_NAMES['additive']}"),
    ("gim_epi_mmd_20", f"{GI_NAMES['epistasis']}"),
    ("gim_red_mmd_20", f"{GI_NAMES['redundancy']}"),
]

METRICS_HIGHER_BETTER += [
    "gi_pearson_mag",
    "gi_pearson_linear_dcor",
    "gi_pearson_dcor_ratio",
    "gi_pearson_dcor",
]

# causal metrics + walltime
METRICS_CONFIG += [
    ("causal_f1", "F1 (causal effects)"),
    ("causal_auroc", "AUROC (causal effects)"),
    ("causal_prec", "Precision"),
    ("causal_pearson", "Pearson $r$"),
    ("causal_gene_out_f1", "F1 (outgoing effects)"),
    ("causal_gene_in_f1", "F1 (ingoing effects)"),
    ("causal_gene_out_prec", "Precision (outgoing effects)"),
    ("causal_gene_in_prec", "Precision (ingoing effects)"),
    #
    ("walltime", "Walltime (min)"),
]


for c in ["acc", "f1", "prec", "rec"]:
    for m in ["syn", "add", "sup", "neo", "epi", "red"]:
        mm = f"gi_cls_{c}_{m}"
        METRICS_CONFIG.append(mm)
        METRICS_HIGHER_BETTER.append(mm)

for c in [
    "ed",
    "ed_20",
    "mmd",
    "mmd_20",
    "rmse",
    "rmse_20",
    "pearson_de",
    "pearson_de_20",
]:
    for m in ["syn", "add", "sup", "neo", "epi", "red"]:
        mm = f"gim_{m}_{c}"
        if not any([mm == mmm for mmm, *_ in METRICS_CONFIG]):
            METRICS_CONFIG.append(mm)
        if c not in [
            "ed",
            "ed_20",
            "mmd",
            "mmd_20",
            "rmse",
            "rmse_20",
        ]:
            METRICS_HIGHER_BETTER.append(mm)

        if c in [
            "mmd",
            "mmd_20",
        ]:
            METRICS_HIGHER_PREC.append(mm)


METRICS_ORDERED = [m if isinstance(m, str) else m[0] for m in METRICS_CONFIG]

METRICS_YLIMS = {
    "norman": {
        "mmd_hvg_":     (    0.00,   0.055),
        "rmse_m_hvg_":  (    0.00,   0.25),
        "rmse":         (    0.00,   1.65),
        "pearson":      (   -0.20,   1.00),
        "topk_auroc":   (    0.00,   1.00),
        "topk_ap":      (    0.00,   1.00),
    },
    "wessels": {
        "mmd_hvg_":     (    0.00,   0.036),
        "rmse_m_hvg_":  (    0.00,   0.50),
        "rmse":         (    0.00,   1.65),
        "pearson":      (   -0.20,   1.00),
        "topk_auroc":   (    0.00,   1.00),
        "topk_ap":      (    0.00,   1.00),
    },
}
METRICS_LOG_YLIMS = {}