import math
import wandb
import numpy as onp
import pandas as pd
import seaborn as sns
from jax import numpy as jnp
from scipy.stats import gaussian_kde
import matplotlib.cm as mplcm

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker

from lcd_clipr.utils.scatter_matrix import scatter_matrix, get_2d_percentile_contours


def plot(samples,
         target_samples,
         intv_mask=None,
         title_prefix=None,
         filename_prefix="",
         subfolder_name=None,
         var_names=None,
         theta=None,
         true_param=None,
         ax_size_matshow=None,
         ax_size=1.0,
         t_current=None,
         max_points=3000,
         n_bins_default=20,
         size_per_var=0.5,
         diff_width=6.0,
         diff_height=1.8,
         grid_type="both",
         scat_kwds=None,
         scat_kwds_target=None,
         hist_kwds=None,
         hist_kwds_target=None,
         method_name=None,
         differential_envs=None,
         plot_marginals_violin=False,
         grid_envs=None,
         grid_per_env=None,
         grid_cols=15,
         fig_suptitle=None,
         contours=(0.80,),
         contours_alphas=(1.0,),
         scotts_bw_scaling=1.0,
         minmin=-1e5,
         maxmax=1e5,
         matrix_range_padding=0.5,
         matrix_percentile_cutoff_limits=0.05,
         cbar_pad=0.001,
         proj=None,
         proj_kde=True,
         cmain="#1976D2",
         ctarget="grey",
         cref="grey",
         ref_data=None,
         ref_mask=None,
         ref_alpha=0.9,
         plot_params=False,
         plot_pairwise_grid=False,
         omit_target_from_pairwise_grid=False,
         plot_intv_marginals=False,
         plot_differential_marginals=False,
         prioritize_marker_genes=True,
         share_scatter_axis_limits=False,
         config=None,
         to_wandb=False,
         to_file=False,
         ):

    _proj_scat_kwds = dict(
        marker=".",
        alpha=0.30,
        s=50,
    )

    _proj_scat_kwds_target = dict(
        marker=".",
        alpha=0.30,
        s=50,
    )

    _scat_kwds = dict(
        marker=".",
        alpha=0.25,
        s=10,
    )
    if scat_kwds is not None:
        _scat_kwds.update(scat_kwds)

    _scat_kwds_target = dict(
        marker=".",
        alpha=0.25,
        s=10,
    )
    if scat_kwds_target is not None:
        _scat_kwds_target.update(scat_kwds_target)

    _hist_kwds = dict(
        linewidth=0.5,
        alpha=0.7,
        histtype="stepfilled",
    )
    if hist_kwds is not None:
        _hist_kwds.update(hist_kwds)

    _hist_kwds_target = dict(
        linewidth=0.5,
        alpha=0.8,
        histtype="stepfilled",
    )
    if hist_kwds_target is not None:
        _hist_kwds_target.update(hist_kwds_target)

    _density1d_kwds = dict(linestyle="dashed")
    _density1d_kwds_target = dict(linestyle="solid")

    _density2d_kwds = dict(linestyles="dashed")
    _density2d_kwds_target = dict(linestyles="solid")

    # preprocess data
    if isinstance(samples, (onp.ndarray, jnp.ndarray)):
        if samples.ndim == 2:
            samples = samples[None]
    elif not isinstance(samples, (list, tuple)):
        samples = [samples]

    samples = [s[:max_points] for s in samples]
    assert all([s.ndim == 2 for s in samples])

    if isinstance(target_samples, (onp.ndarray, jnp.ndarray)):
        if target_samples.ndim == 2:
            target_samples = target_samples[None]
    elif not isinstance(target_samples, (list, tuple)):
        target_samples = [target_samples]

    target_samples = [ts[:max_points] for ts in target_samples]
    assert all([ts.ndim == 2 for ts in target_samples])

    if ref_data is not None:
        assert ref_data.ndim == 2
        ref_data = ref_data[:max_points]
    
    n_datasets = len(samples)
    d = samples[0].shape[-1]

    if intv_mask is None:
        intv_mask = onp.zeros((n_datasets, d))
    
    if ax_size_matshow is None:
        ax_size_matshow = max(3.0, d ** 0.4)

    wandb_images = {}
    fig_suptitle = "" if fig_suptitle is None else f"{fig_suptitle} "
    var_names = var_names or [r"$x_{" + f"{j}" + r"}$" for j in range(d)]

    if subfolder_name is None:
        wandb_folder = f"plots/"
    else:
        wandb_folder = f"{subfolder_name}/"

    if to_file:
        assert subfolder_name is not None
        subfolder_name.mkdir(exist_ok=True, parents=True)

    # parameter plot
    if plot_params:

        # assemble available params
        params = dict()
        if theta is not None and plot_params:
            for k, v in theta.items():
                if k == "__config__":
                    continue
                if type(v) == dict:
                    if k in config.components.keys():
                        for kk, vv in v.items():
                            params[f"{k}-{kk}"] = vv, True
                else:
                    params[f"{k}"] = v, True

        # plot each individually
        for param_name, (arr_raw, is_model) in params.items():

            is_noise_scale_param = "log_sigma_scale" in param_name
            if is_noise_scale_param:
                arr = onp.exp(arr_raw).T
            else:
                arr = arr_raw

            arr_mat = None
            cmap_arr = mplcm.get_cmap('seismic_r') # blue positive, red negative

            # plot drift parameter summary
            # if applicable, transform matrix into a matrix with adjacency matrix convention to compare with ground truth
            # essentially applies to parameter groups targeted in fix_speed_scaling part of update step
            # which imply the causal dependencies in the system (e.g. for MLPs)
            if param_name == "weights":
                arr_mat = arr.copy()
            if param_name == "linear-weights":
                arr_mat = arr.copy()

            # plot diffusion matrix parameter summary
            if is_noise_scale_param:
                cmap_arr = mplcm.get_cmap('YlGnBu') # white/yellow zero, then viridis-like for larger values
                if arr.ndim == 1:
                    arr_mat = onp.diag(arr).copy()
                    arr_mat[*onp.where(~onp.eye(d).astype(bool))] = onp.nan
                else:
                    arr_mat = onp.array(arr).copy()

            if arr_mat is not None:

                fig, axes = plt.subplots(1, 1 if true_param is None else 2,
                                         figsize=((1 if true_param is None else 2) * ax_size_matshow, ax_size_matshow))
                if true_param is None:
                    axes = onp.array([axes])

                # transpose to match adjacency matrix convention
                arr_mat = arr_mat.T
                if true_param is not None:
                    plot_true_param = true_param.T

                # set color bar range
                if is_noise_scale_param:
                    nondiag = arr_mat
                else:
                    nondiag = arr_mat[~onp.eye(d).astype(bool)]
                vmin, vmax = nondiag.min() - cbar_pad, nondiag.max() + cbar_pad

                vmin = min(-vmax, vmin) if nondiag.min() < 0 else 0
                vmax = max(-vmin, vmax)

                matim_arr = axes[0].matshow(arr_mat, vmin=vmin, vmax=vmax, cmap=cmap_arr)

                divider = make_axes_locatable(axes[0])
                cax = divider.append_axes("right", size="5%", pad=0.15)
                plt.colorbar(matim_arr, cax=cax)

                # set color bar based on nondiagonal elements, otherwise they will be unvisible in very large systems
                if true_param is not None:

                    nondiag = plot_true_param[~onp.eye(d).astype(bool)]
                    vmin, vmax = nondiag.min() - cbar_pad, nondiag.max() + cbar_pad

                    vmin = min(-vmax, vmin)
                    vmax = max(-vmin, vmax)
                    matim_true = axes[1].matshow(plot_true_param, vmin=vmin, vmax=vmax, cmap=mplcm.get_cmap('seismic_r'))

                    divider = make_axes_locatable(axes[1])
                    cax = divider.append_axes("right", size="5%", pad=0.15)
                    plt.colorbar(matim_true, cax=cax)

                    axes[1].set_title("true")

                for ax in axes.ravel():
                    ax.tick_params(axis='both', which='both', length=0)
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                    # ax.axis('off')
                    ax.grid(False)

                suptitle_ = fig_suptitle
                if t_current is not None:
                    suptitle_ += f"{param_name} true $t={t_current}$"

                fig.suptitle(suptitle_)
                plt.tight_layout()

                wandb_name = f"param/{param_name}"
                if to_wandb:
                    wandb_images[wandb_folder + wandb_name] = wandb.Image(plt)
                    plt.close()
                elif to_file:
                    filename = wandb_name.replace("/", "-")
                    file_path = subfolder_name / (filename_prefix + filename)
                    plt.savefig(file_path.with_suffix(".pdf"), format="pdf", bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()


    if plot_pairwise_grid:

        perm = onp.random.default_rng(0).permutation(d)

        for ii, s in enumerate(samples):
            if grid_envs is not None and ii >= grid_envs:
                break

            if onp.isnan(s).all(0).any():
                # at least on dimension only has nan, skip scatter plot
                print(f"Skipping {title_prefix} {subfolder_name} {ii} due to nan values.", flush=True)
                continue

            for kk in range(grid_per_env or 1):

                # variable subselection
                lo, hi = kk * grid_cols, (kk + 1) * grid_cols
                hi = min(hi, d)
                if lo >= d:
                    break

                idx = perm[lo:hi]

                # if intervened, also plot intervened variables
                intv = onp.where(intv_mask[ii])[0]
                if intv.size:
                    idx = onp.concatenate([idx[~onp.isin(idx, intv)], intv])

                # visualize in scatter matrix
                data = pd.DataFrame(s, columns=var_names).iloc[:, idx]
                data_target = pd.DataFrame(onp.array(target_samples[ii]), columns=var_names).iloc[:, idx]
                ref = None if ref_data is None else ref_data[:, idx]
                mask = intv_mask[ii, idx]

                if grid_type == "hist-kde":
                    grids = [("hist", "kde")]
                elif grid_type == "kde-kde":
                    grids = [("kde", "kde")]
                elif grid_type == "hist-scatter":
                    grids = [("hist", "scatter")]
                elif grid_type == "kde-scatter":
                    grids = [("kde", "scatter")]
                else:
                    raise KeyError(f"Unknown grid type {grid_type}")

                for diagonal, offdiagonal in grids:
                    figax = scatter_matrix(
                        data,
                        data_target if not omit_target_from_pairwise_grid else None,
                        size_per_var=size_per_var,
                        highlight_mask=mask,
                        binwidth=None,
                        n_bins_default=n_bins_default,
                        percentile_cutoff_limits=matrix_percentile_cutoff_limits,
                        range_padding=matrix_range_padding,
                        contours=contours,
                        contours_alphas=contours_alphas,
                        cmain=cmain,
                        ctarget=ctarget,
                        cref=cref,
                        ref_alpha=ref_alpha,
                        ref_data=ref,
                        diagonal=diagonal,
                        scotts_bw_scaling=scotts_bw_scaling,
                        offdiagonal=offdiagonal,
                        scat_kwds=_scat_kwds,
                        scat_kwds_target=_scat_kwds_target,
                        hist_kwds=_hist_kwds,
                        hist_kwds_target=_hist_kwds_target,
                        density1d_kwds=_density1d_kwds,
                        density1d_kwds_target=_density1d_kwds_target,
                        density2d_kwds=_density2d_kwds,
                        density2d_kwds_target=_density2d_kwds_target,
                        minmin=minmin,
                        maxmax=maxmax,
                        prioritize_significant=prioritize_marker_genes,
                        share_axis_limits=share_scatter_axis_limits,
                    )
                    if figax is None:
                        plt.close()
                        break

                    fig, axes = figax

                    # format title
                    intv_vals = data_target.values.mean(-2)
                    intv_vals_std = data_target.values.std(-2)
                    title = title_prefix or ""
                    if onp.where(mask)[0].size:
                        title += " interv"
                        for j in onp.where(mask)[0]:
                            if onp.allclose(intv_vals_std[j], 0.0):
                                title += f" {data.columns[j]}$=${intv_vals[j].item():.1f}"
                            else:
                                title += f" {data.columns[j]}"
                    else:
                        title += " observ"

                    if t_current is not None:
                        title += f"  ($t={t_current}$)"
                    fig.suptitle(fig_suptitle + title)
                    plt.tight_layout()
                    plt.gcf().subplots_adjust(wspace=0, hspace=0)

                    wandb_name = f"{offdiagonal}-{kk}/intv={ii}"
                    if to_wandb:
                        wandb_images[wandb_folder + wandb_name] = wandb.Image(plt)
                        plt.close()
                    elif to_file:
                        filename = wandb_name.replace("/", "-")
                        file_path = subfolder_name / (filename_prefix + filename)
                        plt.savefig(file_path.with_suffix(".pdf"), format="pdf", bbox_inches='tight')
                        plt.close()
                    else:
                        plt.show()


    # ============  interventional marginals  ============
    max_n_intvs = int(intv_mask.sum(1).max().item())

    # 1: intv target marginals
    if plot_intv_marginals and max_n_intvs > 0:
        envs = intv_mask.sum(1).nonzero()[0]

        # limit max number of envs plotted to avoid matplotlib error in large scale experiments
        envs = envs[:20]

        n_envs = len(envs)
        n_rows_marginals = max_n_intvs
        n_cols_marginals = n_envs
        fig, axes = plt.subplots(n_rows_marginals, n_cols_marginals, figsize=(ax_size * n_cols_marginals,
                                                                              ax_size * n_rows_marginals * 1.3))
        try:
            if axes.ndim == 1:
                if axes.shape == (n_rows_marginals,):
                    axes = axes[..., None]
                elif axes.shape == (n_cols_marginals,):
                    axes = axes[None]
        except AttributeError:
            axes = onp.array([[axes]])

        # for each env, get intervention indices
        assert intv_mask.ndim == 2
        intv_indices = []
        for env in envs:
            intv_indices.append(intv_mask[env].nonzero()[0].tolist())

        n_deleted_axes = 0
        for j, (env, env_intv_indices) in enumerate(zip(envs, intv_indices)):
            # delete unnecessary axes for envs with less than `max_n_intvs` intervened nodes
            if not env_intv_indices:
                for i in range(n_rows_marginals):
                    fig.delaxes(axes[i, j])
                    n_deleted_axes += 1
                continue

            for i in range(len(env_intv_indices), n_rows_marginals):
                fig.delaxes(axes[i, j])
                n_deleted_axes += 1

            for i, idx_j in enumerate(env_intv_indices):
                ax = axes[i, j]

                marginal_j = samples[env][:, idx_j].copy()
                marginal_target_j = target_samples[env][:, idx_j].copy()
                if onp.sum(~(onp.isnan(marginal_j) | (marginal_j < minmin) | (marginal_j > maxmax))) <= 10:
                    fig.delaxes(axes[i, j])
                    n_deleted_axes += 1
                    continue

                marginal_j = marginal_j[~onp.isnan(marginal_j)]
                marginal_target_j = marginal_target_j[~onp.isnan(marginal_target_j)]

                degenerate = onp.isclose(onp.std(marginal_j), 0.0, atol=1e-3)
                degenerate_target = onp.isclose(onp.std(marginal_target_j), 0.0, atol=1e-3)

                binmin = min(marginal_j.min(), marginal_target_j.min())
                binmax = max(marginal_j.max(), marginal_target_j.max())
                if ref_data is not None:
                    assert ref_data.shape[-1] == d
                    ref_data_j = onp.array(ref_data[:, idx_j])
                    binmin = min(binmin, ref_data_j.min())
                    binmax = max(binmax, ref_data_j.max())
                if degenerate and degenerate_target:
                    binmin, binmax = binmax - 1.0, binmax + 1.0

                binwidth_i = (binmax - binmin) / int(n_bins_default * 1.5)
                bins = onp.arange(binmin, binmax + binwidth_i, binwidth_i)

                # Handle count case
                # in this case, we want a histogram that captures all integer bins individually to avoid artifacts
                if onp.all(onp.isclose(onp.mod(marginal_j, 1), 0) | onp.isnan(marginal_j)):
                    bins = range(int(marginal_j.min()) - 1, int(max(marginal_j.max(), 10)) + 2)

                ax.hist(marginal_target_j, color=ctarget, bins=bins, align="mid", density=True, **_hist_kwds_target)
                ax.hist(marginal_j, color=cmain, bins=bins, align="mid", density=True, **_hist_kwds)
                ax.set_xlabel(var_names[idx_j])
                # ax.set_title(var_names[idx_j])

                # plot reference data
                if ref_data is not None:
                    assert ref_data.shape[-1] == d
                    ref_data_j = onp.array(ref_data[:, idx_j])
                    ax.hist(ref_data_j, histtype="step", color=cref, bins=bins, density=True,
                            alpha=ref_alpha, zorder=-1, linewidth=_hist_kwds.get("linewidth"))

                # format title
                if i == 0:
                    title = "intv on " + ", ".join([var_names[jj] for jj in env_intv_indices])
                    ax.set_title(title)

        # remove gridlines
        for ax in onp.asarray(axes).ravel():
            ax.grid(visible=False, axis='both')

        suptitle = title_prefix or ""
        fig.suptitle(fig_suptitle + suptitle + " intv-marginals")

        plt.tight_layout()

        filename = f"intv-marginals"
        if n_deleted_axes < n_rows_marginals * n_cols_marginals:
            if to_wandb:
                wandb_images[wandb_folder + filename] = wandb.Image(plt)
                plt.close()
            elif to_file:
                file_path = subfolder_name / (filename_prefix + filename)
                plt.savefig(file_path.with_suffix(".pdf"), format="pdf", bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        else:
            plt.close()

    # 2: differential expression plot
    if plot_differential_marginals and ref_mask is not None and ref_data is not None:

        assert len(samples) == len(intv_mask) == len(target_samples) == len(ref_mask)

        def remove_outliers(df, dist=1.5):
            q1 = df.quantile(0.25)
            q3 = df.quantile(0.75)
            iqr = q3 - q1
            if q3 == q1:
                iqr = 1.0
            return df[(df >= (q1 - dist * iqr)) & (df <= (q3 + dist * iqr))]


        for ii, s in enumerate(samples):
            if differential_envs is not None and ii >= differential_envs:
                break

            if onp.isnan(s).all(0).any():
                # at least on dimension only has nan, skip plot
                print(f"Skipping DE plot {title_prefix} {subfolder_name} {ii} due to nan values.", flush=True)
                continue

            # variable subselection are the differentially expressed genes
            idx = onp.where(ref_mask[ii])[0]

            # plot intervened variables and move them to the left of the plot
            intv = onp.where(intv_mask[ii])[0]
            if intv.size:
                idx = onp.concatenate([intv, idx[~onp.isin(idx, intv)]])

            # skip if no intervention
            if len(intv) == 0 or idx.sum() == 0:
                continue

            data = onp.array(s, dtype=onp.float32)
            data_target = onp.array(target_samples[ii], dtype=onp.float32)
            data_ctrl = onp.array(ref_data, dtype=onp.float32)

            # compute differential expressions and make dataframe
            ctrl_vec = onp.mean(data_ctrl, axis=0, keepdims=True)

            data -= ctrl_vec
            data_target -= ctrl_vec
            data_ctrl -= ctrl_vec

            data = pd.DataFrame(data, columns=var_names).iloc[:, idx]
            data_target = pd.DataFrame(data_target, columns=var_names).iloc[:, idx]
            data_ctrl = pd.DataFrame(data_ctrl, columns=var_names).iloc[:, idx]

            # remove outliers of both dataframes if using violinplot
            # data = data.apply(remove_outliers, axis=0)
            # data_target = data_target.apply(remove_outliers, axis=0)

            # add identifier column to distinguish dataframes
            data['dataset'] = method_name or 'method'
            data_target['dataset'] = 'true'
            data_ctrl['dataset'] = 'ctrl'

            all_datasets = [data, data_target]
            order = [method_name or 'method', 'true']
            colors = [cmain, ctarget]

            if not plot_marginals_violin:
                colors.append("white")
                order.append("ctrl")
                all_datasets.append(data_ctrl)

            # concatenate the dataframes and melt them
            combined_df = pd.concat(all_datasets)
            melted_df = pd.melt(combined_df, id_vars=['dataset'], var_name='variable', value_name='value')

            # filter data for the intervened columns and other columns separately
            first_cols_vars = melted_df['variable'].unique()[:len(intv)]
            first_cols = melted_df[melted_df['variable'].isin(first_cols_vars)]
            remaining_cols = melted_df[~melted_df['variable'].isin(first_cols_vars)]

            first_cols_order = sorted(first_cols.variable.unique())
            remaining_cols_order = sorted(remaining_cols.variable.unique())

            # violinplot
            fig, (ax, ax2) = plt.subplots(1, 2,
                                          figsize=(diff_width, diff_height),
                                          gridspec_kw={'width_ratios': [len(intv), len(idx) - len(intv)]})

            shared_kwargs = dict(
                x='variable',
                y='value',
                hue='dataset',
                hue_order=order,
                palette=colors,
                saturation=0.8,
            )

            if not plot_marginals_violin:
                shared_kwargs_boxplot = dict(
                    flierprops={"marker": "x", "markersize": 1.5, "linewidth": 0.5},
                    medianprops={"color": "black", "linewidth": 1},
                    showmeans=True,
                    meanline=True,
                    meanprops={"linestyle": '--', "color": 'black'},
                    width=0.8
                )

                # first columns
                sns.boxplot(
                    data=first_cols,
                    ax=ax,
                    order=first_cols_order,
                    **shared_kwargs,
                    **shared_kwargs_boxplot,
                )

                # remaining columns
                # ax2 = ax.twinx()
                sns.boxplot(
                    data=remaining_cols,
                    ax=ax2,
                    order=remaining_cols_order,
                    **shared_kwargs,
                    **shared_kwargs_boxplot,
                )

            else:
                shared_kwargs_violinplot = dict(
                    split=True,
                    density_norm="area", # pdf
                    # density_norm="width", # all have same width
                    bw_adjust=0.75, # bandwidth scaling factor relative to scott's estimate
                    width=1.2, # violin width relative to column width
                    inner=None,
                    # inner="quart", # plots the quartiles inside the violinplot density
                    cut=0
                )

                # first columns
                sns.violinplot(
                    data=first_cols,
                    ax=ax,
                    order=first_cols_order,
                    **shared_kwargs,
                    **shared_kwargs_violinplot,
                )

                # remaining columns
                # ax2 = ax.twinx()
                sns.violinplot(
                    data=remaining_cols,
                    ax=ax2,
                    order=remaining_cols_order,
                    **shared_kwargs,
                    **shared_kwargs_violinplot,
                )

            # control line and legends
            ax.yaxis.grid(False)
            ax.get_legend().remove()
            ax2.get_legend().remove()

            # handles, labels = ax2.get_legend_handles_labels()
            # ax2.legend(handles=handles, labels=labels)

            if plot_marginals_violin:
                ax.axhline(y=0, color="black", linestyle='--')
                ax2.axhline(y=0, color="black", linestyle='--', label='control')

            # move y-axis of ax2 to the right side
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")

            # set y limits based on boxplot whiskers and ignore outliers
            if not plot_marginals_violin:

                first_cols_x = first_cols.pivot(columns=['variable', 'dataset'], values='value')
                first_cols_x = first_cols_x.reindex(columns=first_cols_order, level=0)

                remaining_cols_x = remaining_cols.pivot(columns=['variable', 'dataset'], values='value')
                remaining_cols_x = remaining_cols_x.reindex(columns=remaining_cols_order, level=0)

                first_iqrs = first_cols_x.quantile(0.75) - first_cols_x.quantile(0.25)
                remaining_iqrs = remaining_cols_x.quantile(0.75) - remaining_cols_x.quantile(0.25)

                extend = 1.0
                first_whisker_max = first_cols_x.quantile(0.75) + extend * first_iqrs
                first_whisker_min = first_cols_x.quantile(0.25) - extend * first_iqrs
                remaining_whisker_max = remaining_cols_x.quantile(0.75) + extend * remaining_iqrs
                remaining_whisker_min = remaining_cols_x.quantile(0.25) - extend * remaining_iqrs

                # set column y limits based on boxplot whiskers and ignore outliers
                extra_space = 0.5
                ax.set_ylim(first_whisker_min.min() - extra_space, first_whisker_max.max() + extra_space)
                ax2.set_ylim(remaining_whisker_min.min() - extra_space, remaining_whisker_max.max() + extra_space)

            # formatting
            ax.set_xlabel('')
            ax2.set_xlabel('')

            # spines
            ax.spines.top.set_visible(False)
            ax2.spines.top.set_visible(False)

            ax.spines.bottom.set_visible(True)
            ax2.spines.bottom.set_visible(True)

            ax.spines.left.set_visible(True)
            ax2.spines.left.set_visible(False)

            ax.spines.right.set_visible(False)
            ax2.spines.right.set_visible(True)

            title = fig_suptitle
            if intv.size:
                title += " interv"
                for j in intv:
                    title += f" {var_names[j]}"
            else:
                title += " observ"

            if t_current is not None:
                title += f"  ($t={t_current}$)"
            fig.suptitle(title)

            ylabel = "Differential gene expression"
            ylabel += "\nPerturbed Genes"

            ax.set_ylabel(ylabel)

            ylabel2 = "Differential gene expression"
            ylabel2 += "\n Marker Genes"

            ax2.set_ylabel(ylabel2)


            # make intervened column labels bold
            x_labels = ax.get_xticklabels()
            for j in range(len(intv)):
                x_labels[j].set_fontweight('bold')
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=90)

            ax2.set_xticks(range(len(ax2.get_xticklabels())))
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

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
            assert new_miny <= miny < 0 < maxy <= new_maxy
            ax.set_ylim(new_miny, new_maxy)

            # highlight to visually separate intervened targets
            xlim = ax.get_xlim()
            # ax.axvline(x=len(intv) - 0.5, color="black", linestyle='-', linewidth=0.8)  # line
            ax.axvspan(-1, len(intv) - 0.5, alpha=0.2, facecolor='grey', edgecolor=None, linewidth=0.0)  # shading
            ax.set_xlim(xlim)

            plt.tight_layout()

            wandb_name = f"differential_expression"
            if plot_marginals_violin:
                wandb_name += "_violin"
            wandb_name += f"/intv={ii}"

            if to_wandb:
                wandb_images[wandb_folder + wandb_name] = wandb.Image(plt)
                plt.close()
            elif to_file:
                filename = wandb_name.replace("/", "-")
                file_path = subfolder_name / (filename_prefix + filename)
                plt.savefig(file_path.with_suffix(".pdf"), format="pdf", bbox_inches='tight')
                plt.close()
            else:
                plt.show()


    # ============  low-dim projection  ============

    if proj is not None:
        n_rows_proj = ((n_datasets - 1) // grid_cols) + 1
        n_cols_proj = min(n_datasets, grid_cols)
        fig, axes = plt.subplots(n_rows_proj, n_cols_proj, figsize=(ax_size * n_cols_proj, ax_size * n_rows_proj))
        if n_datasets == 1:
            axes = [[axes]]
        elif n_rows_proj == 1:
            axes = [axes]
        for k, s in enumerate(samples):
            ii, jj = (k // n_cols_proj),  (k % n_cols_proj)

            # filter nan and inf
            target_data = target_samples[k]
            s_data = s

            target_mask = ~onp.isnan(target_data) & (target_data < maxmax) & (target_data > minmin)
            s_mask = ~onp.isnan(s_data) & (s_data < maxmax) & (s_data > minmin)

            target_data = target_data[onp.where(target_mask.all(-1))[0]]
            s_data = s_data[onp.where(s_mask.all(-1))[0]]

            if not target_data.size or not s_data.size:
                fig.delaxes(axes[ii][jj])
                continue

            target_proj = proj(target_data)
            s_proj = proj(s_data)

            # scatter
            if not proj_kde:
                axes[ii][jj].scatter(target_proj[..., 0], target_proj[..., 1], c=ctarget, **_proj_scat_kwds_target)
                axes[ii][jj].scatter(s_proj[..., 0], s_proj[..., 1], c=cmain, **_proj_scat_kwds)

            # kde
            else:
                try:
                    target_proj = target_proj.T
                    s_proj = s_proj.T

                    contours_alphas_ = onp.sort(onp.array(contours_alphas))

                    bw_scotts_target = float(target_proj.shape[-1]) ** float(-1. / (2 + 4))
                    kde_target = gaussian_kde(target_proj, bw_method=scotts_bw_scaling * bw_scotts_target)
                    kde_target_fun = lambda x: kde_target(x.reshape(-1, 2).T).reshape(*x.shape[:-1])

                    xlim_target = (target_proj[0].min(), target_proj[0].max())
                    ylim_target = (target_proj[1].min(), target_proj[1].max())

                    z_target, z_target_levels = get_2d_percentile_contours(kde_target_fun, levels=contours,
                                                                           xlim=xlim_target, ylim=ylim_target)

                    for z_target_level, z_alpha in zip(z_target_levels, contours_alphas_):
                        axes[ii][jj].contour(z_target, levels=onp.array([z_target_level]),
                                             extent=[*xlim_target, *ylim_target],
                                             colors=ctarget, alpha=z_alpha,
                                             **_density2d_kwds_target)

                    bw_scotts = float(s_proj.shape[-1]) ** float(-1. / (2 + 4))
                    kde = gaussian_kde(s_proj, bw_method=scotts_bw_scaling * bw_scotts)
                    kde_fun = lambda x: kde(x.reshape(-1, 2).T).reshape(*x.shape[:-1])

                    xlim_s = (s_proj[0].min(), s_proj[0].max())
                    ylim_s = (s_proj[1].min(), s_proj[1].max())

                    z, z_levels = get_2d_percentile_contours(kde_fun, levels=contours,
                                                             xlim=xlim_s, ylim=ylim_s)

                    for z_level, z_alpha in zip(z_levels, contours_alphas_):
                        axes[ii][jj].contour(z, levels=onp.array([z_level]),
                                             extent=[*xlim_s, *ylim_s],
                                             colors=cmain, alpha=z_alpha,
                                             **_density2d_kwds)

                except ValueError:
                    # sometimes happens that error like this occurs if system is not stable yet:
                    # "A value ... in x_new is below the interpolation range's minimum value."
                    pass

            # format title
            intv_vals = target_samples[k].mean(-2)
            intv_vals_std = target_samples[k].std(-2)
            title = title_prefix or ""
            if onp.where(intv_mask[k])[0].size:
                title += " interv"
                for j in onp.where(intv_mask[k])[0]:
                    if onp.allclose(intv_vals_std[j], 0.0):
                        title += f" {var_names[j]}$=${intv_vals[j].item():.1f}"
                    else:
                        # title += f" {var_names[j]}$\\approx${intv_vals[j].item():.1f} ($\\pm${intv_vals_std[j].item():.1f})"
                        # title += f" {var_names[j]}$\\approx${intv_vals[j].item():.1f}"
                        title += f" {var_names[j]}"
            else:
                title += " observ"

            axes[ii][jj].set_title(title)

        # remove gridlines
        for ax in onp.asarray(axes).ravel():
            ax.grid(visible=False, axis='both')

        # unify axes
        x0, x1, y0, y1 = math.inf, -math.inf, math.inf, -math.inf
        for kk in range(n_cols_proj * n_rows_proj):
            ii, jj = (kk // n_cols_proj), (kk % n_cols_proj)
            if kk <= d:
                x0 = min(x0, axes[ii][jj].get_xlim()[0])
                x1 = max(x1, axes[ii][jj].get_xlim()[1])
                y0 = min(y0, axes[ii][jj].get_ylim()[0])
                y1 = max(y1, axes[ii][jj].get_ylim()[1])
            else:
                fig.delaxes(axes[ii][jj])

        for kk in range(n_cols_proj * n_rows_proj):
            ii, jj = (kk // n_cols_proj), (kk % n_cols_proj)
            if kk <= d:
                axes[ii][jj].set_xlim((x0, x1))
                axes[ii][jj].set_ylim((y0, y1))

        plt.suptitle(fig_suptitle)
        plt.gcf().subplots_adjust(wspace=0, hspace=0)
        plt.draw()
        plt.tight_layout()

        filename = "proj"
        if to_wandb:
            wandb_images[wandb_folder + filename] = wandb.Image(plt)
            plt.close()
        elif to_file:
            file_path = subfolder_name / (filename_prefix + filename)
            plt.savefig(file_path.with_suffix(".pdf"), format="pdf", bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    plt.close('all')


    return wandb_images if wandb_images else None


def clip(x):
    return onp.maximum(0, x)


def bounds(mean, std, factor=1.0):
    return clip(mean - factor * std), mean + factor * std


def plot_sim_stats(
        step,
        sde_log,
        train_data,
        anneal_scales,
        subfolder_name="simulation",
        env=1,
        to_wandb=False,
):
    wandb_images = {}
    if subfolder_name is None:
        wandb_folder = f"plots/"
    else:
        wandb_folder = f"{subfolder_name}/"

    if env > len(train_data) - 1:
        env = len(train_data) - 1

    train_means = train_data[env].mean(-2)[None]
    train_stds = train_data[env].std(-2)[None]
    noise_scales = anneal_scales.tolist()


    fig, axes = plt.subplots(2, 4, figsize=(10, 5))

    for i, log_space in enumerate([False, True]):
        # mean and std mismatch
        mean_fit = jnp.abs(sde_log["samples_mean"][env] - train_means)
        mean_fit_means = mean_fit.mean(-1)
        mean_fit_stds = mean_fit.std(-1)

        std_fit = jnp.abs(sde_log["samples_std"][env] - train_stds)
        std_fit_means = std_fit.mean(-1)
        std_fit_stds = std_fit.std(-1)


        axes[i][0].plot(noise_scales, mean_fit_means, color="green")
        axes[i][0].fill_between(noise_scales, *bounds(mean_fit_means, mean_fit_stds), alpha=0.3, facecolor="green")

        axes[i][1].plot(noise_scales, std_fit_means, color="blue")
        axes[i][1].fill_between(noise_scales, *bounds(std_fit_means, std_fit_stds), alpha=0.3, facecolor="blue")

        axes[i][2].plot(noise_scales, sde_log["update_mean"][env], color="black")
        axes[i][2].fill_between(noise_scales, *bounds(sde_log["update_mean"][env], sde_log["update_std"][env]), alpha=0.3, facecolor="black")

        axes[i][3].plot(noise_scales, sde_log["drift_mean"][env], color="brown")
        axes[i][3].fill_between(noise_scales, *bounds(sde_log["drift_mean"][env], sde_log["drift_std"][env]), alpha=0.3, facecolor="brown")

        if i == 0:
            axes[i][0].set_title("|mean pred - mean true|")
            axes[i][1].set_title("|std pred - std true|")
            axes[i][2].set_title("update delta")
            axes[i][3].set_title("drift delta")

        if log_space:
            axes[i][0].set_ylim((1e-3, 5.0))
            axes[i][1].set_ylim((1e-3, 10.0))
            axes[i][2].set_ylim((1e-4, 10.0))
            axes[i][3].set_ylim((1e-3, 100.0))
        else:
            axes[i][0].set_ylim((0.0, 1.0))
            axes[i][1].set_ylim((0.0, 1.0))
            axes[i][2].set_ylim((0.0, 3.0))
            axes[i][3].set_ylim((0.0, 3.0))

        for j in [0, 1, 2, 3]:
            axes[i][j].set_xscale("log")
            if i == 1:
                axes[i][j].set_xlabel("noise scale")
            axes[i][j].set_xlim((anneal_scales.min(), anneal_scales.max()))
            if log_space:
                axes[i][j].set_yscale("log")

            # ticks
            axes[i][j].grid(False)
            axes[i][j].set_xticks(noise_scales[::5] + [noise_scales[-1]], minor=False)
            axes[i][j].set_xticks(noise_scales, minor=True)
            axes[i][j].xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}'))
            axes[i][j].xaxis.set_minor_formatter(ticker.NullFormatter())
            plt.setp(axes[i][j].get_xticklabels(), rotation=45, fontsize=10)

        fig.suptitle(f"SDE simulation statistics (step: {step:10d})")

    plt.tight_layout()
    if to_wandb:
        wandb_images[wandb_folder + f"simulation_stats_env={env}"] = wandb.Image(plt)
        plt.close()
    else:
        plt.show()

    return wandb_images if wandb_images else None

