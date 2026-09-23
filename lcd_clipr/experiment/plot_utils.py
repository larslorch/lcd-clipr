import numpy as np
import matplotlib.pyplot as plt
from lcd_clipr.experiment.plot_config import *
import matplotlib
from matplotlib.ticker import MaxNLocator


DATA_UNITS_PER_LINE = 0.5


def draw_kinked_connector(
        ax,
        tick_x,
        label_x,
        label_text,
        va='center',
        axis_tick_length=0.018,
        label_tick_length=0.025,
        label_y=-0.13,
        label_gap=0.035,
        line_color='black',
        line_width=0.5,
        font_size=5,
        line_spacing=1.0,
        cap_style='round',
):
    """
    Draw a kinked connector line from an axis tick to a spread-out label.

    Parameters:
    -----------
    ax : matplotlib Axes
        The axes to draw on
    tick_x : float
        X position of the tick on the axis (in data coordinates)
    label_x : float
        X position of the label (in data coordinates)
    label_text : str
        The label text (can be multi-line)
    axis_tick_length : float
        Length of vertical tick at axis (in axes fraction)
    label_tick_length : float
        Length of vertical tick at label (in axes fraction)
    label_y : float
        Y position of label top (in axes fraction, negative = below axis)
    label_gap : float
        Vertical gap between label tick and label text (in axes fraction)
    line_color : str
        Color of connector lines
    line_width : float
        Width of connector lines
    font_size : float
        Font size for label text
    line_spacing : float
        Line spacing for multi-line labels
    cap_style: str
        Line cap style for connector lines
    """
    transform = ax.get_xaxis_transform()

    # Draw vertical tick at axis (from axis down)
    ax.plot(
        [tick_x, tick_x],
        [0, -axis_tick_length],
        color=line_color, lw=line_width,
        transform=transform,
        clip_on=False,
        solid_capstyle=cap_style,
    )

    # Draw diagonal part (from bottom of axis tick to top of label tick)
    ax.plot(
        [tick_x, label_x],
        [-axis_tick_length, label_y + label_tick_length],
        color=line_color, lw=line_width,
        transform=transform,
        clip_on=False,
        solid_capstyle=cap_style,
    )

    # Draw vertical tick at label end
    ax.plot(
        [label_x, label_x],
        [label_y + label_tick_length, label_y],
        color=line_color, lw=line_width,
        transform=transform,
        clip_on=False,
        solid_capstyle=cap_style,
    )

    # Add the rotated multi-line label with gap
    ax.text(
        label_x, label_y - label_gap,
        label_text,
        transform=transform,
        rotation=90,
        rotation_mode="anchor",
        va=va,
        ha='right',
        fontsize=font_size,
        linespacing=line_spacing
    )


def calculate_label_positions(
        tick_positions,
        labels,
        va='center',
        space_per_line=DATA_UNITS_PER_LINE,
        extra_space_per_label=DATA_UNITS_PER_LINE * 0.75,
        offset=None,
):
    """
    Calculate spread-out label positions based on the number of lines in each label.

    Labels with more lines need more horizontal space (since they're rotated 90°).
    This function spreads labels apart so they don't overlap, while ensuring that
    the label spacing is at least as large as the tick spacing (only increasing, never decreasing).

    Parameters:
    -----------
    tick_positions : array-like
        X positions of the ticks (in data coordinates)
    labels : list of str
        The label strings (can be multi-line with \n)
    space_per_line: float
    extra_space_per_label : float
    offset: float
        Offset (in data coordinates) to add to all label positions

    Returns:
    --------
    label_positions : numpy array
        X positions for each label (in data coordinates)
    """
    tick_positions = np.array(tick_positions)
    n = len(labels)
    line_counts = np.array([label.count('\n') + 1 for label in labels])

    # Start from leftmost position (always origin) and place labels with required gaps
    label_positions = np.zeros(n)
    for i in range(1, n):
        gap = line_counts[i - 1] * space_per_line + extra_space_per_label
        label_positions[i] = label_positions[i - 1] + gap

    # Adjust raw spacing so that it's correct given the vertical alignment
    for i in range(0, n):
        if va == 'top':
            pass
        elif va == 'center':
            label_positions[i] += 0.5 * line_counts[i] * space_per_line
        elif va == 'bottom':
            label_positions[i] += line_counts[i] * space_per_line
        else:
            raise ValueError(f"Invalid va: {va}")

    # Center the labels such that right end of ticks align
    label_positions -= (label_positions[-1] - tick_positions[-1])
    if offset is not None:
        label_positions += offset

    assert len(label_positions) == len(tick_positions)

    return label_positions


def setup_spread_labels(ax, tick_positions, labels, label_x_offset, va='center',
                        space_per_line=DATA_UNITS_PER_LINE,
                        extra_space_per_label=DATA_UNITS_PER_LINE * 0.75,
                        **connector_kwargs):
    """
    Set up spread-out labels with kinked connectors on a plot.

    Parameters:
    -----------
    ax : matplotlib Axes
        The axes to draw on
    tick_positions : array-like
        X positions of the ticks (in data coordinates)
    labels : list of str
        The label strings (can be multi-line with \n)
    va: str
        Vertical alignment for labels
    label_x_offset: float
        Offset (in data coordinates) to add to all label positions
    space_per_line: float
    extra_space_per_label : float
    **connector_kwargs : dict
        Additional keyword arguments passed to draw_kinked_connector
    """
    # Hide default tick labels and ticks
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0)
    labels = [l.strip() for l in labels]

    # Calculate automatic label positions
    label_positions = calculate_label_positions(tick_positions, labels, va=va, offset=label_x_offset,
                                                space_per_line=space_per_line,
                                                extra_space_per_label=extra_space_per_label)

    # Draw connectors and labels
    for tick_x, label_x, label in zip(tick_positions, label_positions, labels):
        draw_kinked_connector(ax, tick_x, label_x, label, va=va, **connector_kwargs)

    return label_positions


def save_legend(save_path, handles, labels, hide_patch=False, ncols=1, add_width=0.0, edgecolor='none',
                rcparams=MATPLOTLIB_RCPARAMS, **kwargs):
    save_path.parent.mkdir(exist_ok=True, parents=True)

    matplotlib.rcdefaults()
    plt.rcParams.update(rcparams)

    # create new figure for the legend
    fig_legend = plt.figure(figsize=(FIGSIZE["legend"]["ax_width_col"] * ncols + add_width,
                                     FIGSIZE["legend"]["ax_height"]))  # Adjust the size to fit the legend appropriately
    ax_legend = fig_legend.add_subplot(111)

    # create and save the legend
    legend = ax_legend.legend(handles, labels, loc='center', ncols=ncols, **kwargs)
    for patch in legend.get_patches():
        patch.set_edgecolor(edgecolor)
        patch.set_alpha(1.0)

    if hide_patch:
        for item in legend.legendHandles:
            item.set_visible(False)

    ax_legend.axis('off')
    fig_legend.savefig(save_path.with_suffix(".pdf"), format="pdf", bbox_inches='tight')
    plt.close(fig_legend)
    return


def save_colorbar(save_path, vmin, vmax, cmap, orientation='horizontal', label_right=False, tickpad=5, rcparams=MATPLOTLIB_RCPARAMS,
                  long_side=None, short_side=None, label=None, show_outline=True, ticks=None, integer_ticks=False, **kwargs):
    save_path.parent.mkdir(exist_ok=True, parents=True)

    plt.rcParams.update(rcparams)

    # Set default figure size based on orientation
    if long_side is None:
        long_side = FIGSIZE["colorbar"]["long_side"]
    if short_side is None:
        short_side = FIGSIZE["colorbar"]["short_side"]
    if orientation == 'vertical':
        figsize = (short_side, long_side)
    else:
        figsize = (long_side, short_side)

    # Create new figure for the colorbar
    fig_cbar = plt.figure(figsize=figsize)
    ax_cbar = fig_cbar.add_subplot(111)

    # Create colorbar using ScalarMappable
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    if integer_ticks and ticks is None:
        ticks = MaxNLocator(nbins=3, integer=True)

    cbar = plt.colorbar(sm, cax=ax_cbar, orientation=orientation, ticks=ticks, **kwargs)
    cbar.ax.tick_params(pad=tickpad)

    if label is not None:
        if label_right:
            cbar.ax.text(1.15, 0.2, label, transform=cbar.ax.transAxes, va='center', ha='left', rotation=0)
        else:
            cbar.set_label(label)

    if not show_outline:
        cbar.outline.set_visible(False)

    fig_cbar.savefig(save_path.with_suffix(".pdf"), format="pdf", bbox_inches='tight')
    plt.close(fig_cbar)
    return

