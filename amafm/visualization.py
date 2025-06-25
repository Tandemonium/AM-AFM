from collections.abc import Callable
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import preprocessing
from .preprocessing import Measurement


colors = ['tab:blue', 'g', 'r', 'tab:orange', 'c', 'y', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']


def get_color(i: int) -> str:
    return colors[i % len(colors)]


def plot_all(measurements: list[Measurement], single_color: bool = True, alpha: float = 0.1):
    """
    Plot all measurements in a 2x2 grid for approach and retract for phase and amplitude curves.

    Parameters
    ----------
    measurements : list[Measurement]
        List of `Measurement` objects to plot.
    single_color : bool, optional
        If `True` plots all curves in a graph in the same color, otherwise plots them in different colors, by default True
    alpha : float, optional
        Sets the opacity of the plotted curves. 
        When plotting many measurements, a lower `alpha` shows the density of curves better, by default 0.1
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    labels = ['phase approach', 'phase retract', 'amplitude approach', 'amplitude retract']
    for i, (signal_type, z_type) in enumerate(zip(Measurement.signal_types(), 
                                                  Measurement.z_types() * 2)):
        ax = axs[i // 2, i % 2]
        for m in measurements:
            y, z = m[signal_type], m[z_type]
            if single_color:
                ax.plot(z, y, color='tab:blue', alpha=alpha)
            else:
                ax.plot(z, y)
        ax.set_title(labels[i])
    plt.show()


def compare_directions(measurement: Measurement):
    """
    Plot and compare the approach and retract directions of phase and amplitude curve from a `Measurement` object.

    Parameters
    ----------
    measurement : Measurement
        The `Measurement` object to plot.
    """
    fig, axs = plt.subplots(2, 1, figsize=(8, 4))
    axs[0].plot(measurement.z_out, measurement.phase_out, label='retract', alpha=0.5)
    axs[0].plot(measurement.z_in, measurement.phase_in, color='r', label='approach', alpha=0.5)
    axs[0].set_xlabel('z (nm)')
    axs[0].set_ylabel('phase (deg)')
    axs[0].legend()
    axs[1].plot(measurement.z_out, measurement.amp_out, label='retract', alpha=0.5)
    axs[1].plot(measurement.z_in, measurement.amp_in, color='r', label='approach', alpha=0.5)
    axs[1].set_xlabel('z (nm)')
    axs[1].set_ylabel('amplitude (V)')
    axs[1].legend()
    plt.tight_layout()
    plt.show()


def compare_raw_and_processed(measurements: list[Measurement], measurements_raw: list[Measurement],
                              signal_type: Literal['amp', 'phase'] = 'phase', direction: Literal['in', 'out'] = 'out',
                              columns: int = 4, rows: int = 7, col_width: int = 5, row_height: int = 2):
    """
    Create a grid of plots, comparing a raw curve with its preprocessed version.

    Parameters
    ----------
    measurements : list[Measurement]
        List of `Measurement` objects containing preprocessed curves.
    measurements_raw : list[Measurement]
        List of `Measurement` objects containing raw curves.
    signal_type : Literal[&#39;amp&#39;, &#39;phase&#39;], optional
        The signal type to plot, by default 'phase'
    direction : Literal[&#39;in&#39;, &#39;out&#39;], optional
        Direction of the probe movement for which to plot curves,
        `in` for 'approach', `out` for 'retract', by default 'out'
    columns : int, optional
        Number of columns of plots to create, by default 4
    rows : int, optional
        Number of rows of plots to create, by default 7
    col_width : int, optional
        Width of each plot, by default 5
    row_height : int, optional
        Height of each plot, by default 2
    """
    z_name, signal_name = f'z_{direction}', f'{signal_type}_{direction}'
    fig, axs = plt.subplots(rows, columns, figsize=(columns * col_width, rows * row_height))
    for i in range(rows * columns):
        ax = axs[i % rows, i // rows]
        zraw = measurements_raw[i][z_name]
        yraw = measurements_raw[i][signal_name]
        z = measurements[i][z_name]
        y = measurements[i][signal_name]
        ax.plot(zraw, yraw, alpha=0.5, label='raw')
        ax.plot(z, y, 'r', label='processed')
        if i == 0:
            ax.legend()
    plt.show()


def compare_smoothing(measurement_raw: Measurement, smoothing_methods: list[dict[str, Any]],
                      signal_type: Literal['amp', 'phase'] = 'phase', 
                      direction: Literal['in', 'out'] = 'out'):
    """Plot compare a curve with different smoothing methods applied."""
    y = measurement_raw[f'{signal_type}_{direction}']
    z = measurement_raw[f'z_{direction}']
    n_plots = len(smoothing_methods)
    
    fig, axs = plt.subplots(n_plots, 1, figsize=(8, n_plots + 2))
    for i, method in enumerate(smoothing_methods):
        smooth_func = method['smooth_func']
        smooth_kwargs = method['smooth_kwargs']
        d = smooth_func(y, **smooth_kwargs)
        axs[i].plot(z, d, color=get_color(i), label=f'{smooth_func.__name__} {smooth_kwargs}')
        axs[i].plot(z, y, color='k', alpha=0.25)
        axs[i].set_xticks([])
        axs[i].legend()
    fig.tight_layout()
    plt.show()


def plot_compare_yalign(measurements: list[Measurement], calib_params: dict[str, float], 
                        signal_type: Literal['amp', 'phase'] = 'phase', direction: Literal['in', 'out'] = 'out', 
                        alpha: float = 0.5):
    """Plot comparison of y-aligned curve with original curve."""
    st, zt = f'{signal_type}_{direction}', f'z_{direction}'
    signal = [m[st] for m in measurements]
    far_param = calib_params[f'{signal_type}_far']
    aligned_med = [preprocessing.y_align(s, far_param, 'median') for s in signal]
    aligned_avg = [preprocessing.y_align(s, far_param, 'mean') for s in signal]

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    for i in range(len(measurements)):
        for j, aligned in enumerate([aligned_med[i], aligned_avg[i]]):
            label = 'original' if i == 0 else None
            axs[j].plot(measurements[i][zt], measurements[i][st], 
                        color='tab:blue', alpha=alpha, label=label)
    for i in range(len(measurements)):
        for j, aligned in enumerate([aligned_med[i], aligned_avg[i]]):
            label = ('y-aligned (median)' if j == 0 else 'y-aligned (mean)') if i == 0 else None
            axs[j].plot(measurements[i][zt], aligned, color='r', alpha=alpha, label=label)
    axs[0].legend()
    axs[1].legend()
    fig.tight_layout()
    plt.show()


def plot_compare_feature_xalign(measurements: list[Measurement], 
                                signal_type: Literal['amp', 'phase'] = 'phase', 
                                direction: Literal['in', 'out'] = 'out'):
    """Plot compare a curve with different x-alignment methods applied."""
    feature_types = ['extrema', 'increase', 'decrease', 'maximum', 'minimum']
    ns = [2, None, None, 1, 1]
    labels = ['unaligned'] + feature_types
    zs_signals = [(m[f'z_{direction}'], m[f'{signal_type}_{direction}']) for m in measurements]
    aligned = [[preprocessing.feature_x_align(z, s, f, n) for f, n in zip(feature_types, ns)] for z, s in zs_signals]
    for curves, (z, s) in zip(aligned, zs_signals):
        zconcat = np.concatenate(curves)
        zmin, zmax = zconcat.min(), zconcat.max()
        smin, smax = s.min(), s.max()

        fig, axs = plt.subplots(len(feature_types) + 1, 1, figsize=(8, len(feature_types) + 2))
        for i, zshift in enumerate([z] + curves):
            axs[i].plot(zshift, s, label=labels[i], color=get_color(i))
            axs[i].vlines(0, smin, smax, color='k', linestyle='--')
            axs[i].set_xlim(zmin, zmax)
            axs[i].legend(loc='lower right')
        fig.tight_layout()
        plt.show()


def plot_compare_dtw_xalign(measurements: list[Measurement], lead_curve_idx: int,
                            signal_type: Literal['amp', 'phase'] = 'phase', 
                            direction: Literal['in', 'out'] = 'out', alpha: float = 0.5):
    """Plot comparison of original curve with curve with dtw-algorithm x-alignment applied."""
    m_lead = measurements[lead_curve_idx]
    st, zt = f'{signal_type}_{direction}', f'z_{direction}'
    step_patterns = ['rj', 'sym']
    fig, axs = plt.subplots(3, 1, figsize=(8, 5))
    for i, m in enumerate(measurements):
        label1 = 'original' if i == 0 else None
        axs[0].plot(measurements[i][zt], measurements[i][st], 
                    color='tab:blue', alpha=alpha, label=label1)
        for j, (sp, c) in enumerate(zip(step_patterns, ['r', 'g'])):
            z, s = preprocessing.dtw_x_align(m[st], m_lead[zt], m_lead[st], step_pattern=sp)
            label2 = f'x-aligned (dtw {sp})' if i == 0 else None
            axs[j + 1].plot(z, s, color=c, alpha=alpha, label=label2)
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.tight_layout()
    plt.legend(lines, labels, loc='lower right')
    plt.show()


def plot_smallest_largest_distance(distances: np.ndarray, measurements: list[Measurement], 
                                   ideal_measurement: Measurement, n: int = 3, 
                                   signal_type: Literal['amp', 'phase'] = 'phase', 
                                   direction: Literal['in', 'out'] = 'out'):
    """Plot the cumulative frquency of curve distances to a given ideal curve plus  
       the `n` curves with smallest and largest distance to the ideal curve."""
    st = f'{signal_type}_{direction}'
    ideal_signal = ideal_measurement[st]

    plt.figure(figsize=(8, 2))
    plt.hist(distances[1:], bins=len(distances), cumulative=True)
    plt.xlabel('distance')
    plt.ylabel('number of curves')
    plt.title('Distances to ideal curve')
    plt.show()
    
    fig, axs = plt.subplots(n, 1, figsize=(8, n + 2))
    for i in range(n):
        ax = axs[i]
        ax.plot(ideal_signal, label=f'(ideal)', color='k', alpha=0.4)
        ax.plot(measurements[i][st], label=f'({i}) d={distances[i]:.2g}', color='tab:blue')
        ax.legend()
    plt.suptitle(f'Top {n} smallest distances')
    fig.tight_layout()
    plt.show()
    
    fig, axs = plt.subplots(n, 1, figsize=(8, n + 2))
    for i in reversed(range(n)):
        idx = len(measurements) - i - 1
        ax = axs[i]
        ax.plot(ideal_signal, label=f'(ideal)', color='k', alpha=0.4)
        ax.plot(measurements[-i][st], label=f'({idx}) d={distances[i]:.2g}', color='tab:blue')
        ax.legend()
    plt.suptitle(f'Top {n} largest distances')
    fig.tight_layout()
    plt.show()


def plot_averaging(avrg_measurement: Measurement, measurements: list[Measurement],
                   signal_type: Literal['amp', 'phase'] = 'phase', 
                   direction: Literal['in', 'out'] = 'out', cutoff: int = None, figsize: tuple[int, int] = (8, 3)):
    """
    Plot selected curves from measurements and together with the respective average curve.

    Parameters
    ----------
    avrg_measurement : Measurement
        `Measurment` object containing averaged curves.
    measurements : list[Measurement]
        A list of the individual measurements used for the average measurement.
    signal_type : Literal[&#39;amp&#39;, &#39;phase&#39;], optional
        The signal type to plot, by default 'phase'
    direction : Literal[&#39;in&#39;, &#39;out&#39;], optional
        Direction of the probe movement for which to plot curves,
        `in` for 'approach', `out` for 'retract', by default 'out'
    cutoff : int, optional
        Cut off the curves beyond this threshold, by default None
    figsize : tuple[int, int], optional
        Size of the plot, by default (8, 3)
    """
    r = slice(cutoff)
    plt.figure(figsize=figsize)
    for m in measurements:
        plt.plot(m[f'z_{direction}'][r], m[f'{signal_type}_{direction}'][r], color='tab:blue', alpha=0.1)
    plt.plot(avrg_measurement[f'z_{direction}'][r], avrg_measurement[f'{signal_type}_{direction}'][r], color='r')
    plt.show()


def plot_with_uncertainty(main_curve: Measurement|tuple[np.ndarray, np.ndarray], 
                          zscore_data: list[Measurement]|list[tuple[np.ndarray, np.ndarray]],
                          zscores: list[float], get_x_y_func: Callable[..., tuple[np.ndarray, np.ndarray]],
                          cutoff: int,
                          figsize: tuple[int, int]):
    """Plot a curve with uncertainty bands based on given z-scores."""
    r = slice(cutoff)
    zscores = np.array(zscores)
    zrange = np.unique(np.abs(zscores))[::-1]
    grouped_idcs = []
    for zs in zrange:
        g = []
        if -zs in zscores:
            g.append(np.argwhere(zscores == -zs).flatten()[0])
        if zs in zscores:
            g.append(np.argwhere(zscores == zs).flatten()[0])
        grouped_idcs.append(g)
    alphas = np.linspace(0, 1, len(grouped_idcs) + 2)[1:-1]
    plt.figure(figsize=figsize)
    for i, idcs in enumerate(grouped_idcs):
        for ii, idx in enumerate(idcs):
            label = f'$\\pm{abs(zscores[idx])}\\sigma$' if ii == 0 else None
            x, y = get_x_y_func(zscore_data[idx])
            plt.plot(x[r], y[r], color='tab:orange', alpha=alphas[i], label=label)
    x, y = get_x_y_func(main_curve)
    plt.plot(x[r], y[r], color='b', label='mean')
    plt.legend()
    plt.show()


def plot_average_with_uncertainty(avrg_measurement: Measurement, zscore_measurements: list[Measurement],
                                  zscores: list[float], signal_type: Literal['amp', 'phase'] = 'phase', 
                                  direction: Literal['in', 'out'] = 'out', cutoff: int = None,
                                  figsize: tuple[int, int] = (8, 3)):
    """
    Plot average curve with uncertainty bands based on given z-scores.

    Parameters
    ----------
    avrg_measurement : Measurement
        `Measurement` object containing averaged curves.
    zscore_measurements : list[Measurement]
        `Measurement` objects for each z-score containing curves corresponding to the z-score offset.
    zscores : list[float]
        List of z-scores, required for the plot legend.
    signal_type : Literal[&#39;amp&#39;, &#39;phase&#39;], optional
        The signal type to plot, by default 'phase'
    direction : Literal[&#39;in&#39;, &#39;out&#39;], optional
        Direction of the probe movement for which to plot curves,
        `in` for 'approach', `out` for 'retract', by default 'out'
    cutoff : int, optional
        Cut off the curves beyond this threshold, by default None
    figsize : tuple[int, int], optional
        Size of the plot, by default (8, 3)
    """
    get_x_y_func = lambda m: (m[f'z_{direction}'], m[f'{signal_type}_{direction}'])
    plot_with_uncertainty(avrg_measurement, zscore_measurements, zscores, get_x_y_func, cutoff, figsize)


def plot_force_with_uncertainty(force_result: tuple[np.ndarray, np.ndarray], 
                                zscore_forces: list[tuple[np.ndarray, np.ndarray]],
                                zscores: list[float], cutoff: int = None,
                                figsize: tuple[int, int] = (8, 4)):
    """
    Plot force curve with uncertainty bands based on given z-scores.

    Parameters
    ----------
    force_result : tuple[np.ndarray, np.ndarray]
        A tuple containing the separation and force received from the `force.force_hoelscher` function.
    zscore_forces : list[tuple[np.ndarray, np.ndarray]]
        A list of tuples, each containing separation and force corresponding to a z-score.
    zscores : list[float]
        List of z-scores, required for the plot legend.
    cutoff : int, optional
        Cut off the curves beyond this threshold, by default None
    figsize : tuple[int, int], optional
        Size of the plot, by default (8, 4)
    """
    get_x_y_func = lambda x: (x[0], x[1])
    plot_with_uncertainty(force_result, zscore_forces, zscores, get_x_y_func, cutoff, figsize)


def plot_conf_mat(conf_mat: np.ndarray, labels: list[Any], model_name: str):
    sns.set_theme(font_scale=1.4)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax = sns.heatmap(conf_mat, annot=True, xticklabels=labels, yticklabels=labels, 
                     vmin=0.0, vmax=1.0, fmt=".2f", ax=ax, square=True, annot_kws={"size": 14})
    ax.tick_params(left=False, bottom=False)
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.title(model_name)
    plt.show()
    # conf_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
    # conf_disp.plot(cmap=plt.cm.PiYG, values_format=".2f")  # vanimo, coolwarm
    print(conf_mat)
