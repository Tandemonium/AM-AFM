import warnings
from typing import Literal

import numpy as np
from tqdm import tqdm

from .preprocessing import Measurement


def bin_curves(measurements: list[Measurement], direction: Literal['in', 'out'], 
               n_bins: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zs = [m['z_' + direction] for m in measurements]
    amps = [m['amp_' + direction] for m in measurements]
    phases = [m['phase_' + direction] for m in measurements]
    
    min_z = min([z[0] for z in zs])
    max_z = max([z[-1] for z in zs])
    bins_left = np.linspace(min_z, max_z, n_bins + 1)

    z_bins = []
    amp_bins = []
    phase_bins = []
    for z, a, p in tqdm(zip(zs, amps, phases), total=len(zs), desc=f'Average curves for {direction=}'):
        hist = np.histogram(z, bins=bins_left)
        idcs = np.cumulative_sum(hist[0])[:-1]
        z_bins.append(np.split(z, idcs))
        amp_bins.append(np.split(a, idcs))
        phase_bins.append(np.split(p, idcs))
    z_bins = np.array([np.array(curve_bins, dtype=object) for curve_bins in z_bins]).T
    amp_bins = np.array([np.array(curve_bins, dtype=object) for curve_bins in amp_bins]).T
    phase_bins = np.array([np.array(curve_bins, dtype=object) for curve_bins in phase_bins]).T
    return z_bins, amp_bins, phase_bins


def average_binned_curves(z_bins: np.ndarray, amp_bins: np.ndarray, phase_bins: np.ndarray, 
                          zscore_cutoff: float|None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'.*Mean of empty slice')
        if zscore_cutoff is not None:
            amp_bins = [np.concatenate(bins) for bins in amp_bins]
            phase_bins = [np.concatenate(bins) for bins in phase_bins]
            a_zscore = [(a_bin - a_bin.mean()) / a_bin.std() for a_bin in amp_bins]
            p_zscore = [(p_bin - p_bin.mean()) / p_bin.std() for p_bin in phase_bins]
            amp_bins = [a_bin[np.abs(az) < zscore_cutoff] for a_bin, az in zip(amp_bins, a_zscore)]
            phase_bins = [p_bin[np.abs(pz) < zscore_cutoff] for p_bin, pz in zip(phase_bins, p_zscore)]
            amp_mean = np.array([np.nanmean(a_bin) for a_bin in amp_bins])
            phase_mean = np.array([np.nanmean(p_bin) for p_bin in phase_bins])
        else:
            amp_mean = np.array([np.nanmean(np.concatenate(bins)) for bins in amp_bins])
            phase_mean = np.array([np.nanmean(np.concatenate(bins)) for bins in phase_bins])
        z_mean = np.array([np.nanmean(np.concatenate(bins)) for bins in z_bins])
        return z_mean, amp_mean, phase_mean


def get_average_uncertainty(binned_amp_curves: np.ndarray, binned_phase_curves: np.ndarray,
                            zscores: list[float]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Return a list of curves through values of the input bins corresponding to the given z-scores.
    """
    #z_bins, amp_bins, phase_bins = bin_curves(measurements_0024_001, direction, n_bins)
    amp_bins = [np.concatenate(bins) for bins in binned_amp_curves]
    phase_bins = [np.concatenate(bins) for bins in binned_phase_curves]
    amps_at_zscore, phases_at_zscore = [], []
    for zscore in zscores:
        a_at_zs = np.array([(zscore * bin.std()) + bin.mean() for bin in amp_bins])
        p_at_zs = np.array([(zscore * bin.std()) + bin.mean() for bin in phase_bins])
        amps_at_zscore.append(a_at_zs)
        phases_at_zscore.append(p_at_zs)
    return amps_at_zscore, phases_at_zscore


def average_curves(measurements: list[Measurement], n_bins: int = 1000, zscore_cutoff: float|None = None,
                   zscores: list[float]|None = None) -> Measurement|tuple[Measurement, list[Measurement]]:
    """
    Creates a new Measurement object containing the curve averages from the given measurements.
    Requires the measurements to be at least x- and y-aligned.

    Parameters
    ----------
    measurements : list[preprocessing.Measurement]
    n_bins : int, optional
        nuber of bins in which to separate the total range of the curves, by default 1000
    zscore_cutoff: float|None, optional
        In each bin discard all values with a Z-score beyond the given threshold, by default None
    zscores: list[float]|None, optional
        List of z-scores for which to return the corresponding values of the curves, by default None

    Returns
    -------
    Measurement
        If `zscore`is `None`: A Measurment object containing averaged curves.
        Else: A tuple of the average Measurement and a list containing a Measurement for each given z-score.
    """
    avrg_data = {k: [] for k in Measurement.__match_args__}
    zscore_curves_at_dir = []
    for direction in ['in', 'out']:
        z_bins, amp_bins, phase_bins = bin_curves(measurements, direction, n_bins)
        z_mean, amp_mean, phase_mean = average_binned_curves(z_bins, amp_bins, phase_bins, zscore_cutoff=zscore_cutoff)
        if zscores:
            amps_at_zscore, phases_at_zscore = get_average_uncertainty(amp_bins, phase_bins, zscores)
            zscore_curves_at_dir.append([z_mean, amps_at_zscore, phases_at_zscore])
        avrg_data['amp_' + direction] = amp_mean
        avrg_data['phase_' + direction] = phase_mean
        avrg_data['z_' + direction] = z_mean
    avrg_measurement = Measurement(**avrg_data)
    if zscores:
        zs_measurements = []
        zsc_in, zsc_out = zscore_curves_at_dir[0], zscore_curves_at_dir[1]
        for i, _ in enumerate(zscores):
            m = Measurement(zsc_in[0], zsc_out[0], zsc_in[2][i], zsc_out[2][i], zsc_in[1][i], zsc_out[1][i])
            zs_measurements.append(m)
        return avrg_measurement, zs_measurements
    else:
        return avrg_measurement
