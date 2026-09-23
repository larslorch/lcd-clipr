import argparse
import functools

import jax
import jax.numpy as jnp
import numpy as onp
from jax import random

import matplotlib.pyplot as plt
from matplotlib import cm

from lcd_clipr.sde import SDE
from lcd_clipr.noise_models.likelihood import zipoisson_generative_model
from lcd_clipr.utils.sde import to_diag

from lcd_clipr.definitions import PROJECT_DIR, SUBDIR_PLOTS
from lcd_clipr.experiment.plot_config import *

plt.rcParams.update(MATPLOTLIB_RCPARAMS)


class Langevin(SDE):
    def __init__(self, *, pdf, wiener_procs, sde_kwargs=None):
        super().__init__(wiener_procs, **sde_kwargs)
        self.logpdf = jax.jit(lambda x: jnp.log(pdf(x)))
        self.grad_logpdf = jnp.vectorize(jax.jacrev(self.logpdf), signature='(n)->(n)')

    def init_params(self, key, d, scale=0.001):
        pass

    def init_intv_theta(self, key, n_envs, d, scale=0.001, train_datasets=None):
        pass

    def drift(self, param, x, diffusion_time, noise_scale, mask_train, mask_query):
        return self.grad_logpdf(x)

    def diffusion(self, param, x, diffusion_time, noise_scale, mask_train, mask_query):
        return to_diag(jnp.ones_like(x)) * jnp.sqrt(2.0)


def sample_pdf_as_langevin(
    key,
    pdf,
    *,
    n,
    steps=30000,
    dt=5e-5,
    x_init=None,
    probability_flow=False,
):
    n_samples_sampler = n * steps
    model = Langevin(
        pdf=pdf,
        wiener_procs=2,
        sde_kwargs=dict(
            anneal=False,
            dt=dt,
            thinning=1,
            n_samples_burnin=0,
            rollouts_shape=(n,),
        )
    )
    key, subk = random.split(key)
    _, _traj = model._simulate_dynamical_system(
        None,
        subk,
        jnp.zeros((1, 2)),
        jnp.zeros((1, 2)),
        x_init=x_init,
        n_samples=n_samples_sampler,
        return_traj=True,
        probability_flow=probability_flow,
    )
    traj = onp.array(_traj["x_traj"].squeeze(1))  # drop thinning axis which is 1 here
    samples = traj[-1]
    traj = onp.swapaxes(traj, 0, 1)

    # [n, 2], [n, steps, 2]
    return samples, traj, model


def pdf_raw(
    state,
    a=1.5,
    b=0.5,
    scale=0.5,
    # scale=1.0,
    rot_deg=35,
    shift_x=-0.3,
    shift_y=3.5,
	# shift_x=1.5,
	# shift_y=1.5,
):

    x, y = state[..., 0], state[..., 1]

    x += shift_x
    y += shift_y

    # rotate x, y values
    rot = rot_deg * jnp.pi / 180
    x, y = x * jnp.cos(rot) - y * jnp.sin(rot), x * jnp.sin(rot) + y * jnp.cos(rot)

    # potential
    f = (x / a) ** 2 + (a * (y - b * ((x / a) ** 2 + a ** 2))) ** 2

    # unnormalized pdf
    return jnp.exp(- scale * f)


def draw_gradient_arrow(ax, x, y, gradient, length, color='red', linewidth=1.5, head_width=0.2, head_length=0.2, zorder=10):
    """Draw an arrow with the tip ending at (x + gradient[0]*length, y + gradient[1]*length)."""
    # Calculate the end point
    dx = gradient[0] * length
    dy = gradient[1] * length

    # Arrow starts at (x, y) and ends at (x+dx, y+dy)
    ax.arrow(
        x, y,  # starting point
        dx,  # dx
        dy,  # dy
        head_width=head_width,
        head_length=head_length,
        fc=color,
        ec=color,
        linewidth=linewidth,
        zorder=zorder,
        length_includes_head=True,  # Makes the total length include the head
    )


def make_contour_plot(
    pdf,
    xmin, xmax, ymin, ymax,
    x_ticks, y_ticks,
    grid_density,
    contour_levels,
    cmap,
    cmapf,
    figsize,
):
    # Compute PDF on grid
    xs = onp.linspace(xmin, xmax, grid_density)
    ys = onp.linspace(ymin, ymax, grid_density)
    xs, ys = onp.meshgrid(xs, ys)
    states_grid = onp.stack([xs, ys], axis=-1)
    zs = pdf(states_grid)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # # Draw contour lines
    if cmap is not None:
        if isinstance(cmap, str):
            # If cmap is a string, treat it as a color
            ax.contour(xs, ys, zs, levels=contour_levels, colors=cmap, linewidths=0.5, alpha=0.5, zorder=-199)
        else:
            # Otherwise treat it as a colormap
            ax.contour(xs, ys, zs, levels=contour_levels, cmap=cmap, linewidths=0.5, alpha=0.5, zorder=-199)

    # Draw filled contours
    if cmapf is not None:
        if isinstance(cmapf, str):
            ax.contourf(xs, ys, zs, levels=contour_levels, colors=cmapf, alpha=0.4, zorder=-200)
        else:
            ax.contourf(xs, ys, zs, levels=contour_levels, cmap=cmapf, alpha=0.4, zorder=-200)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    return fig, ax


def make_3d_fig_ax(figsize, rect=(0.1, 0.1, 0.8, 0.8)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect, projection="3d", computed_zorder=False)
    ax.view_init(elev=25, azim=-25, roll=-1.5)

    # ax.set_box_aspect([1.0, 1.5, 0.8], zoom=1.0)

    # Default (x, y, z): ax.set_box_aspect([4.0, 4.0, 3.0], zoom=1.0)
    ax.set_box_aspect([4.0, 4.0, 2.2], zoom=1.0)

    plt.gca().patch.set_facecolor('white')
    ax.xaxis.set_pane_color((0, 0, 0, 0))
    ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))

    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_zticks([])

    ax.tick_params(axis='x', pad=-3,direction='out', zorder=100)
    ax.tick_params(axis='y', pad=-5, direction='out', zorder=100)
    ax.tick_params(axis='z', pad=0, direction='out', zorder=100)

    # Set grid with specific z-order and styling
    ax.grid(True, zorder=0)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 0.3)  # Light gray with transparency
        axis._axinfo['grid']['linewidth'] = 0.5

    # Change z-axis position
    ax.zaxis.set_rotate_label(False)
    ax.zaxis._axinfo['juggled'] = (1, 2, 0)


    return fig, ax


def main():

    jnp.set_printoptions(precision=5, suppress=True)
    kwargs = argparse.Namespace()
    kwargs.seed = 0
    kwargs.seed_traj = 1000

    root_path = PROJECT_DIR / SUBDIR_PLOTS / "fig_1"
    root_path.mkdir(parents=True, exist_ok=True)

    """
    Axes
    x: left (going towards camera)
    y: bottom (going right)
    """

    unit_scaling = (2, 10)
    dropout_probs = onp.array([0.2, 0.05])

    x_ticks_density = [0, 3, 6]
    y_ticks_density = [-4, -2, 0, 2]
    xmin_density, xmax_density = -1.5, 7.5
    ymin_density, ymax_density = -5, 4
    alpha_density = 0.8

    grid_density = 200

    vector_grid_density = 10
    max_arrow_length = 30.0
    trajetory_plots = 10
    # trajetory_plot_selected = None
    trajetory_plot_selected = 7 # select the 'final' one
    trajectory_thinning = 20
    trajectory_z_scale = 3e-6
    trajectory_z_end = 1.4
    trajectory_alpha = 0.8
    trajectory_linewidth = 0.5

    clipr_contour_levels = 10
    clipr_arrow_length = 1.0
    clipr_max_arrow_length = 5
    clipr_arrow_length_scale = 65
    clipr_vector_grid_density = 10
    clipr_x_ticks_density = [-1, 0, 1, 2]
    clipr_y_ticks_density = [-3, -2, -1, 0]
    clipr_xmin_density, clipr_xmax_density = -1.6, 2.4
    clipr_ymin_density, clipr_ymax_density = -3.2, 0.8

    x_ticks_counts = [0, 5, 10]
    y_ticks_counts = [0, 5, 10]
    xmin_counts, xmax_counts = 0, 13
    ymin_counts, ymax_counts = 0, 13
    alpha_counts = 0.8

    samples_counts = 3000
    # samples_counts = 100

    # cmap = cm.coolwarm
    # cmap = cmcrameri.cm.grayC_r
    cmap = cmcrameri.cm.devon_r

    cmap_clipr = cm.Greys # inside grey
    # cmap_clipr = cm.Greys_r # inside white

    color_stochastic_process = COLORS[5]
    color_f0 = COLORS[6]
    color_finf = COLORS[5]

    """
    Density
    """
    save_path = root_path
    save_path.mkdir(parents=True, exist_ok=True)

    # compute pdf
    xs = onp.linspace(xmin_density, xmax_density, grid_density)
    ys = onp.linspace(ymin_density, ymax_density, grid_density)
    xs, ys = onp.meshgrid(xs, ys)
    states_grid = onp.stack([xs, ys], axis=-1)

    pdf = functools.partial(pdf_raw)
    zs = pdf(states_grid)

    # plotting - use explicit axis positioning for consistent size
    fig, ax = make_3d_fig_ax((
        FIGSIZE["figure_1_density"]["ax_width"],
        FIGSIZE["figure_1_density"]["ax_height"]
    ))

    ax.plot_surface(xs, ys, zs, cmap=cmap, linewidth=0, antialiased=False, alpha=alpha_density)
    ax.set_zlim(0.0, 1.0)

    ax.set_xlim((xmin_density, xmax_density))
    ax.set_ylim((ymin_density, ymax_density))
    ax.set_xticks(x_ticks_density)
    ax.set_yticks(y_ticks_density)

    fig_path = save_path / "fig_1_density"
    plt.savefig(fig_path.with_suffix(".pdf"), format="pdf", dpi=500)
    plt.close()

    # sample from pdf
    key, subk = random.split(random.PRNGKey(kwargs.seed))
    samples, traj, model = sample_pdf_as_langevin(subk, pdf, n=samples_counts)

    """
    CLIPR illustration
    """
    f0 = model.grad_logpdf(jnp.array([0.0, 0.0]))
    _, finf_traj, _ = sample_pdf_as_langevin(
        key, pdf, n=1,
        steps=int(1e4),
        dt=1e-3,
        x_init=jnp.array([0.0, 0.0]),
        probability_flow=True,
    )
    finf_traj = onp.array(finf_traj).squeeze(0)
    finf = finf_traj[-1]

    fig, ax = make_contour_plot(
        pdf,
        clipr_xmin_density, clipr_xmax_density, clipr_ymin_density, clipr_ymax_density,
        clipr_x_ticks_density, clipr_y_ticks_density,
        grid_density,
        clipr_contour_levels,
        None,
        cmap_clipr,
        (FIGSIZE["figure_1_clipr"]["ax_width"], FIGSIZE["figure_1_clipr"]["ax_height"])
    )

    # Plot vector field as 2D quiver
    x_vec_clipr, y_vec_clipr = onp.meshgrid(
        onp.linspace(clipr_xmin_density, clipr_xmax_density, clipr_vector_grid_density),
        onp.linspace(clipr_ymin_density, clipr_ymax_density, clipr_vector_grid_density),
    )
    points_vec_clipr = onp.stack([x_vec_clipr.flatten(), y_vec_clipr.flatten()], axis=-1)

    drift_vectors_clipr = onp.array(model.grad_logpdf(points_vec_clipr))
    u_clipr = drift_vectors_clipr[:, 0].reshape(x_vec_clipr.shape)
    v_clipr = drift_vectors_clipr[:, 1].reshape(x_vec_clipr.shape)
    arrow_magnitudes_clipr = onp.sqrt(u_clipr ** 2 + v_clipr ** 2)
    scale_factors_clipr = onp.minimum(1.0, clipr_max_arrow_length / (arrow_magnitudes_clipr + 1e-10))
    u_clipr = u_clipr * scale_factors_clipr
    v_clipr = v_clipr * scale_factors_clipr

    ax.quiver(x_vec_clipr, y_vec_clipr, u_clipr, v_clipr, pivot='middle', scale=clipr_arrow_length_scale,
              headaxislength=4, headlength=4, headwidth=4,
              linewidth=0.5, color='black', alpha=0.3, width=0.008, zorder=0)

    ax.scatter(0, 0, c="black", s=20, marker='+', zorder=1)

    # f0 vector
    draw_gradient_arrow(ax, 0, 0, f0, clipr_arrow_length, color=color_f0)

    # finf vector
    draw_gradient_arrow(ax, 0, 0, finf, clipr_arrow_length, color=color_finf)
    ax.plot(finf_traj[:, 0], finf_traj[:, 1], color='black', alpha=0.4, linestyle="dashed", linewidth=1.0)

    fig.tight_layout()

    fig_path = save_path / f"fig_1_clipr"
    plt.savefig(fig_path.with_suffix(".pdf"), format="pdf", bbox_inches='tight', dpi=500)
    plt.close()


    """
    Vector field
    """
    # Create grid for vector field
    x_vec, y_vec = onp.meshgrid(
        onp.linspace(xmin_density, xmax_density, vector_grid_density),
        onp.linspace(ymin_density, ymax_density, vector_grid_density),
    )
    points_vec = onp.stack([x_vec.flatten(), y_vec.flatten()], axis=-1)

    # Compute drift (gradient of log pdf) at each point
    drift_vectors = onp.array(model.grad_logpdf(points_vec))
    u = drift_vectors[:, 0].reshape(x_vec.shape)
    v = drift_vectors[:, 1].reshape(x_vec.shape)

    # Clip arrow lengths to max_arrow_length
    arrow_magnitudes = onp.sqrt(u**2 + v**2)
    scale_factors = onp.minimum(1.0, max_arrow_length / (arrow_magnitudes + 1e-10))
    u = u * scale_factors
    v = v * scale_factors

    # Plot trajectory with time along z-axis
    # sample from pdf
    key, subk = random.split(random.PRNGKey(kwargs.seed_traj))
    _, traj, _ = sample_pdf_as_langevin(subk, pdf, n=trajetory_plots, steps=int(1e6), x_init=jnp.array([0.0, 0.0]))

    for j in range(trajetory_plots):
        if trajetory_plot_selected is not None and trajetory_plot_selected != j:
            continue

        # Create 3D vector field plot
        fig, ax = make_3d_fig_ax((
            FIGSIZE["figure_1_density"]["ax_width"],
            FIGSIZE["figure_1_density"]["ax_height"]
        ))

        # Plot vector field as 3D quiver plot at z=0 (only arrows, no density)
        ax.quiver(x_vec, y_vec, onp.zeros_like(x_vec), u, v, onp.zeros_like(u),
                  length=0.05, color='black', arrow_length_ratio=0.11,
                  linewidth=0.5, pivot='middle')

        traj_to_plot = traj[j][::trajectory_thinning]
        z_values = onp.arange(len(traj_to_plot)) * trajectory_thinning * trajectory_z_scale
        z_end_idx = onp.searchsorted(z_values, trajectory_z_end)
        ax.plot(traj_to_plot[:z_end_idx, 0], traj_to_plot[:z_end_idx, 1], z_values[:z_end_idx],
                color=color_stochastic_process, linewidth=trajectory_linewidth, alpha=trajectory_alpha, zorder=100)

        ax.set_zlim(0.0, 1.3) # zlim top limit control the "height" of the arrow heads
        ax.set_xlim((xmin_density, xmax_density))
        ax.set_ylim((ymin_density, ymax_density))
        ax.set_xticks(x_ticks_density)
        ax.set_yticks(y_ticks_density)

        if trajetory_plot_selected is not None:
            fig_path = save_path / f"fig_1_vector_field"
        else:
            fig_path = save_path / f"fig_1_vector_field_{j}"
        plt.savefig(fig_path.with_suffix(".pdf"), format="pdf", dpi=500)
        plt.close()

    """
    Noisy data
    """

    counts_per_sample = 1
    # counts_per_sample = 2

    # sample noisy data from state samples
    noise_model = functools.partial(
        zipoisson_generative_model,
        link_temp=1.0,
        link_shift=0.0,
        warp_temp=0.0,
        warp_scaling=1.0,
        unit_scaling=jnp.array(unit_scaling, dtype=float),
    )

    key, subk = random.split(key)
    dropout_odds = onp.log(dropout_probs / (1 - dropout_probs))

    # data = noise_model(subk, samples, dict(logit_pi=dropout_odds))
    data = noise_model(subk, jnp.tile(samples, (counts_per_sample, 1)), dict(logit_pi=dropout_odds))

    fig, ax = make_3d_fig_ax((
        FIGSIZE["figure_1_density"]["ax_width"],
        FIGSIZE["figure_1_density"]["ax_height"]
    ))

    x = onp.array(data[..., 0])
    y = onp.array(data[..., 1])

    hist, xedges, yedges = onp.histogram2d(
        x, y,
        bins=(
            onp.arange(xmax_counts + 1),
            onp.arange(ymax_counts + 1),
        ),
    )
    # histogram2d returns hist[i,j] where i=x-index, j=y-index
    # meshgrid returns arrays where rows vary with y, columns vary with x
    # Need to transpose hist so it aligns with meshgrid when flattened
    xpos, ypos = onp.meshgrid(xedges[:-1], yedges[:-1])

    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = onp.zeros_like(xpos)

    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    dz = hist.T.flatten()

    # Filter out bars where z=0 to avoid grey appearance from shading
    nonzero_mask = dz > 0
    xpos = xpos[nonzero_mask]
    ypos = ypos[nonzero_mask]
    zpos = zpos[nonzero_mask]
    dz = dz[nonzero_mask]

    max_height = onp.max(dz)
    min_height = onp.min(dz)

    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]

    ax.bar3d(
        xpos, ypos, zpos, dx, dy, dz,
        color=rgba,
        zsort='max',
        alpha=alpha_counts,
    )

    ax.set_xlim(xmin_counts, xmax_counts)
    ax.set_ylim(ymin_counts, ymax_counts)
    ax.set_zlim(0, None)
    ax.set_xticks(x_ticks_counts)
    ax.set_yticks(y_ticks_counts)

    fig_path = save_path / "fig_1_counts"
    plt.savefig(fig_path.with_suffix(".pdf"), format="pdf", dpi=500)
    plt.close()


    return


if __name__ == "__main__":
    main()