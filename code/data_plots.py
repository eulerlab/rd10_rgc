import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, ticker
from matplotlib.colors import ColorConverter
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

# Size of figures
_FIG_WIDTHS = dict(
    col=3.37,  # Single column in bioRxiv template
    full=7,  # Full width in bioRxiv template
    slide_col=5.67,  # Half powerpoint slide
    slide_full=11.5,  # Full powerpoint slide
    poster=14.46,  # Full poster width
)


def set_rc_params(kind, notebook_dpi=None):
    sns.set_context('paper')
    sns.set_style('ticks')

    path = os.path.dirname(os.path.abspath(__file__))
    plt.style.use(os.path.join(path, 'mplstyles', f'{kind}.mplstyle'))

    if notebook_dpi is not None:
        plt.rcParams['figure.dpi'] = notebook_dpi


def set_figsize(fig, width, height_ratio=None):
    if isinstance(width, str):
        if width not in _FIG_WIDTHS:
            raise NotImplementedError(f"Unknown width `{width}`")
        else:
            width = _FIG_WIDTHS[width]

    fig.set_figwidth(width)

    if height_ratio is not None:
        fig.set_figheight(width * height_ratio)


def show_saved_figure(fig):
    fig.savefig('.temp.jpg', dpi=600)
    plt.figure(figsize=(10, 10), facecolor=(0.5, 0.5, 0.5, 0.5))

    im = plt.imread('.temp.jpg')

    if np.any(im[0, :] < 255) or np.any(im[-1, :] < 255) or np.any(im[:, 0] < 255) or np.any(im[:, -1] < 255):
        print('Warning: Figure is probably clipped!')

    plt.imshow(im, aspect='equal')
    plt.axis('off')
    plt.show()

    from os import remove as removefile
    removefile('.temp.jpg')


def tight_layout(h_pad=1, w_pad=1, rect=(0, 0, 1, 1), pad=None):
    """Like `tight_layout` with different default"""
    plt.tight_layout(h_pad=h_pad, w_pad=w_pad, pad=pad or 2. / plt.rcParams['font.size'], rect=rect)


def iterate_axes(axs):
    """Make axes iterable, independent of type.
    axs (list of matplotlib axes or matplotlib axis) : Axes to apply function to.
    """

    if isinstance(axs, list):
        return axs
    elif isinstance(axs, np.ndarray):
        return axs.flatten()
    else:
        return [axs]


def int_format_ticks(axs, which='both'):
    """Use integer ticks for integers."""
    from matplotlib.ticker import FuncFormatter

    def int_formatter(x):
        if x.is_integer():
            return str(int(x))
        else:
            return f"{x:g}"

    formatter = FuncFormatter(int_formatter)
    if which in ['x', 'both']:
        for ax in iterate_axes(axs):
            ax.xaxis.set_major_formatter(formatter)
    if which in ['y', 'both']:
        for ax in iterate_axes(axs):
            ax.yaxis.set_major_formatter(formatter)


def scale_ticks(axs, scale, x=True, y=False):
    ticks = ticker.FuncFormatter(lambda xi, pos: '{0:g}'.format(xi * scale))
    for ax in iterate_axes(axs):
        if x:
            ax.xaxis.set_major_formatter(ticks)
        if y:
            ax.yaxis.set_major_formatter(ticks)


def move_xaxis_outward(axs, scale=3):
    """Move xaxis outward.
    axs (array or list of matplotlib axes) : Axes to apply function to.
    scale (float) : How far xaxis will be moved.
    """
    for ax in iterate_axes(axs):
        ax.spines['bottom'].set_position(('outward', scale))


def move_yaxis_outward(axs, scale=3):
    """Move xaxis outward.
    axs (array or list of matplotlib axes) : Axes to apply function to.
    scale (float) : How far xaxis will be moved.
    """
    for ax in iterate_axes(axs):
        ax.spines['left'].set_position(('outward', scale))


def adjust_log_tick_padding(axs, pad=2.1):
    """ Change tick padding for all log scaled axes.
    Parameters:
    axs (array or list of matplotlib axes) : Axes to apply function to.
    pad (float) : Size of padding.
    """

    for ax in iterate_axes(axs):
        if ax.xaxis.get_scale() == 'log':
            ax.tick_params(axis='x', which='major', pad=pad)
            ax.tick_params(axis='x', which='minor', pad=pad)

        if ax.yaxis.get_scale() == 'log':
            ax.tick_params(axis='y', which='major', pad=pad)
            ax.tick_params(axis='y', which='minor', pad=pad)


def set_labs(axs, xlabs=None, ylabs=None, titles=None, panel_nums=None, panel_num_space=0, panel_num_va='bottom',
             panel_num_pad=0, panel_num_y=None):
    """Set labels and titles for all given axes.
    Parameters:

    axs : array or list of matplotlib axes.
        Axes to apply function to.

    xlabs, ylabs, titles : str, list of str, or None
        Labels/Titles.
        If single str, will be same for all axes.
        Otherwise, should have same length as axes.

    """

    for i, ax in enumerate(iterate_axes(axs)):
        if xlabs is not None:
            if isinstance(xlabs, str):
                xlab = xlabs
            else:
                xlab = xlabs[i]
            ax.set_xlabel(xlab)

        if ylabs is not None:
            if isinstance(ylabs, str):
                ylab = ylabs
            else:
                ylab = ylabs[i]
            ax.set_ylabel(ylab)

        if titles is not None:
            if isinstance(titles, str):
                title = titles
            else:
                title = titles[i]
            ax.set_title(title)

        if panel_nums is not None:
            if panel_nums == 'auto':
                panel_num = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]
            elif isinstance(panel_nums, str):
                panel_num = panel_nums
            else:
                panel_num = panel_nums[i]
            ax.set_title(panel_num + panel_num_space * ' ', loc='left', fontweight='bold', ha='right', va=panel_num_va,
                         pad=panel_num_pad, y=panel_num_y)


def left2right_ax(ax):
    """Create a twin axis, but remove all duplicate spines.
    Parameters:
    ax (Matplotlib axis) : Original axis to create twin from.
    Returns:
    ax (Matplotlib axis) : Twin axis with no duplicate spines.
    """

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax = ax.twinx()
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return ax


def move_box(axs, dx=0, dy=0):
    """Change offset of box"""
    for ax in iterate_axes(axs):
        box = np.array(ax.get_position().bounds)
        if dx != 0:
            box[0] += dx
        if dy != 0:
            box[1] += dy
        ax.set_position(box)


def change_box(axs, dx=0, dy=0):
    """Change offset of box"""
    for ax in iterate_axes(axs):
        box = np.array(ax.get_position().bounds)
        if dx != 0:
            box[2] += dx
        if dy != 0:
            box[3] += dy
        ax.set_position(box)


def align_x_box(ax, ref_ax):
    """Change offset of box"""
    box = np.array(ax.get_position().bounds)
    ref_box = np.array(ref_ax.get_position().bounds)
    box[0] = ref_box[0]
    box[2] = ref_box[2]
    ax.set_position(box)


def plot_scale_bar(ax, x0, y0, size, unit, y0text=None, text=None):
    if y0text is None:
        y0text = y0
    ax.plot([x0, x0 + size], [y0, y0], c='k', solid_capstyle='butt')
    ax.text(x0 + size / 2, y0text, f'{size} {unit}' if text is None else text, va='top', ha='center')


def idx2color(idx, colorpalette='tab10', isscatter=False):
    if idx is None:
        c = (0, 0, 0)
    else:
        c = ColorConverter.to_rgb(sns.color_palette(colorpalette).as_hex()[idx])
    if isscatter:
        c = np.atleast_2d(np.asarray(c))
    return c


def text2mathtext(txt):
    txt = txt.replace('^', '}^\mathrm{')
    txt = txt.replace('_', '}_\mathrm{')
    txt = txt.replace(' ', '} \mathrm{')
    txt = txt.replace('-', '\mathrm{-}')
    return r"$\mathrm{" + txt + "}$"


def get_legend_handle(**kwargs):
    """Make method legend on given axis."""
    return Line2D([0], [0], **kwargs)


def get_legend_handles(markers, colors, lss, **kwargs):
    """Get handle for list of markers, colors and linestyles"""
    legend_handles = []
    for marker, color, ls in zip(markers, colors, lss):
        legend_handles.append(get_legend_handle(marker=marker, color=color, ls=ls, **kwargs))
    return legend_handles


def row_title(ax, title, pad=70, size='large', ha='left', va='center', **kwargs):
    """Create axis row title using annotation"""
    ax.annotate(title, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=size, ha=ha, va=va, **kwargs)


def grid(ax, axis='both', major=True, minor=False, **kwargs):
    """make grid on axis"""
    for axi in iterate_axes(ax):
        if major:
            axi.grid(True, axis=axis, which='major', alpha=.3, c='k',
                     lw=plt.rcParams['ytick.major.width'], zorder=-10000, **kwargs)
        if minor:
            axi.grid(True, axis=axis, which='minor', alpha=.3, c='gray',
                     lw=plt.rcParams['ytick.minor.width'], zorder=-20000, **kwargs)


def make_share_xlims(axs, symmetric=False, xlim=None):
    """Use xlim lower and upper bounds for all axes."""
    if xlim is None:
        xlb = np.min([ax.get_xlim()[0] for ax in iterate_axes(axs)])
        xub = np.max([ax.get_xlim()[1] for ax in iterate_axes(axs)])

        if not symmetric:
            xlim = (xlb, xub)
        else:
            xlim = (-np.max(np.abs([xlb, xub])), np.max(np.abs([xlb, xub])))
    for ax in iterate_axes(axs): ax.set_xlim(xlim)


def make_share_ylims(axs, symmetric=False, ylim=None):
    """Use ylim lower and upper bounds for all axes."""
    if ylim is None:
        ylb = np.min([ax.get_ylim()[0] for ax in iterate_axes(axs)])
        yub = np.max([ax.get_ylim()[1] for ax in iterate_axes(axs)])

        if not symmetric:
            ylim = (ylb, yub)
        else:
            ylim = (-np.max(np.abs([ylb, yub])), np.max(np.abs([ylb, yub])))
    for ax in iterate_axes(axs):
        ax.set_ylim(ylim)


def plot_srf_gauss_fit(ax, srf=None, vabsmax=None, srf_params=None, n_std=2, color='k', ms=3, plot_cb=False, **kwargs):
    if srf_params is not None:
        ax.plot(srf_params['x_mean'], srf_params['y_mean'], zorder=100, marker='x', ms=ms, c=color, **kwargs)
        ax.add_patch(Ellipse(
            xy=(srf_params['x_mean'], srf_params['y_mean']),
            width=n_std * 2 * srf_params['x_stddev'],
            height=n_std * 2 * srf_params['y_stddev'],
            angle=np.rad2deg(srf_params['theta']), color=color, fill=False, **kwargs))

    if srf is not None:
        if vabsmax is None:
            vmin = np.min(srf)
            vmax = np.max(srf)
            cmap = 'gray'
        else:
            vmin = -vabsmax
            vmax = vabsmax
            cmap = 'bwr'
        im = ax.imshow(srf, vmin=vmin, vmax=vmax, cmap=cmap, zorder=0, origin='lower')
        if plot_cb:
            plt.colorbar(im, ax=ax)
