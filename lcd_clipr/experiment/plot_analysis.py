import copy

import cmcrameri
import numpy as onp

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from lcd_clipr.experiment.plot_config import MATPLOTLIB_RCPARAMS, FIGSIZE, COLOR_RED

from lcd_clipr.experiment.analyze_causal import linear_causal_params
from lcd_clipr.experiment.plot_causal import plot_mat


def plot_causal_l_curve(
        save_path,
        *,
        f0,
        finf,
        finf_ctrl,
        tikhonov_range,
        tikhonov_labels=None,
        rcparams=MATPLOTLIB_RCPARAMS,
        mask=None,
        ordering=None,
        figsize_key_analysis="causal_l_curve",
        figsize_key_matrix="causal_l_curve_adjacency_matrix",
):

    """
    Compute analysis
    """
    tikhonov_labels = tikhonov_labels or {}
    curve_labels = {v: lab for v, lab in tikhonov_labels.items() if v > 0}
    values = set(onp.array(tikhonov_range)[onp.array(tikhonov_range) > 0].tolist()) | curve_labels.keys()
    tikhonov_range = onp.array(sorted(values))

    solution_norms = []
    residual_norms = []
    for tikhonov in tikhonov_range:
        result = linear_causal_params(
            f0=f0,
            finf=finf,
            finf_ctrl=finf_ctrl,
            tikhonov=tikhonov,
        )
        # f0, finf are [n_env, d] by convention, so we transpose here
        residual = result["a"] @ result["finf"].T + result["f0"].T
        solution_norms.append(onp.linalg.norm(result["a"], ord="fro"))
        residual_norms.append(onp.linalg.norm(residual, ord="fro"))

    """
    Plot
    """
    rcparams_local = copy.deepcopy(rcparams)
    rcparams_local["text.latex.preamble"] = r"\usepackage{amsmath}"
    plt.rcParams.update(rcparams_local)

    fig, (ax, ax_curv) = plt.subplots(1, 2, figsize=(
        FIGSIZE[figsize_key_analysis]["ax_width"],
        FIGSIZE[figsize_key_analysis]["ax_height"],
    ))

    log_residual = onp.log(residual_norms)
    log_solution = onp.log(solution_norms)

    labeled_idx = {
        int(onp.argmin(onp.abs(tikhonov_range - tikhonov))): label
        for tikhonov, label in curve_labels.items()
    }
    is_labeled = onp.array([i in labeled_idx for i in range(len(tikhonov_range))])

    ax.plot(log_residual, log_solution, color="black", zorder=1)
    ax.scatter(
        log_residual[is_labeled], log_solution[is_labeled],
        color=COLOR_RED, marker="o", zorder=3,
    )

    for idx, label in labeled_idx.items():
        ax.annotate(
            label,
            xy=(log_residual[idx], log_solution[idx]),
            xytext=(3, 3),
            textcoords="offset points",
        )

    # ax.set_xlabel(r"$\log \lVert \widehat{\mathbf{A}}_\lambda \, \mathbf{L} + \mathbf{V} \rVert_\mathrm{F}$", usetex=True)
    # ax.set_ylabel(r"$\log \lVert \widehat{\mathbf{A}}_\lambda \rVert_\mathrm{F}$", usetex=True)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

    """
    Hansen L-curve curvature (parameterized by log lambda):
        kappa = (rho' eta'' - rho'' eta') / (rho'^2 + eta'^2)^(3/2)
    where
        eta = log ||solution||^2
        rho = log ||residual||^2
    """
    log_lambda = onp.log(tikhonov_range)
    rho = log_residual
    eta = log_solution
    rho_d = onp.gradient(rho, log_lambda)
    eta_d = onp.gradient(eta, log_lambda)
    rho_dd = onp.gradient(rho_d, log_lambda)
    eta_dd = onp.gradient(eta_d, log_lambda)
    curvature = (rho_d * eta_dd - rho_dd * eta_d) / ((rho_d ** 2 + eta_d ** 2) ** 1.5)

    # drop boundary points from finite-difference derivatives to avoid edge artefacts
    keep = onp.zeros_like(curvature, dtype=bool)
    keep[2:-2] = True

    ax_curv.axhline(0.0, color="black", linestyle="--", linewidth=0.5, zorder=0)
    ax_curv.plot(log_lambda[keep], curvature[keep], color="black", zorder=1)
    ax_curv.scatter(
        log_lambda[is_labeled & keep], curvature[is_labeled & keep],
        color=COLOR_RED, marker="o", zorder=3,
    )

    for idx, label in labeled_idx.items():
        if not keep[idx]:
            continue
        above = curvature[idx] >= 0.5
        ax_curv.annotate(
            label,
            xy=(log_lambda[idx], curvature[idx]),
            xytext=(-1, 2) if above else (0, -4),
            textcoords="offset points",
            ha="right",
            va="bottom" if above else "top",
        )

    # ax_curv.set_xlabel(r"$\log \lambda$", usetex=True)
    # ax_curv.set_ylabel(r"Curvature $\kappa(\lambda)$", usetex=True)

    ax_curv.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax_curv.yaxis.set_major_locator(MaxNLocator(nbins=4))

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path is not None:
        plt.savefig(save_path.with_suffix(".pdf"), format="pdf", bbox_inches='tight')
        plt.close()
    else:
        plt.show()


    """
    Plot matrices for labeled values
    """
    for tikhonov, label in tikhonov_labels.items():
        tikhonov_label = str(tikhonov).replace('.', '_')
        a_causal = linear_causal_params(
            f0=f0,
            finf=finf,
            finf_ctrl=finf_ctrl,
            tikhonov=tikhonov,
        )["a_causal"]

        if mask is not None:
            a_causal = a_causal[mask][:, mask]

        plot_mat(
            save_path.parent / f"{save_path.name}_a_lambda_{tikhonov_label}",
            a_causal,
            ordering=ordering,
            cmap=cmcrameri.cm.berlin_r,
            rcparams=rcparams,
            figsize_key=figsize_key_matrix,
            vquantile=0.97,
            plot_colorbar_inset=True,
            colorbar_inset_kwargs={
                'long_side': 0.5,
                'ticklocation': 'bottom',
                'integer_ticks': True,
            },
            dpi=500,
        )


