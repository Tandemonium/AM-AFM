import dtw
import numpy as np

from typing import Any, Callable, Literal

from tqdm import tqdm

from . import denoise
from .data_loading import Measurement


def find_extrema_indices(signal_array: np.ndarray, n: int = 2) -> list[int]:
    idcs = []
    for i in range(1, len(signal_array) - 1):
        if ((signal_array[i] < signal_array[i - 1] and signal_array[i] < signal_array[i + 1]) or 
            (signal_array[i] > signal_array[i - 1] and signal_array[i] > signal_array[i + 1])):
            idcs.append(i)
        if len(idcs) >= (n + 1):
            break
    return idcs


def feature_x_align(z: np.ndarray, signal: np.ndarray, 
                    feature: Literal['increase', 'decrease', 'extrema', 'maximum', 'minimum'],
                    n: int = 1) -> np.ndarray:
    """
    Parameters
    ----------
    z : np.ndarray
        x-axis array
    signal : np.ndarray
        y-axis array
    feature : ['increase', 'decrease', 'extrema', 'maximum', 'minimum']
        at what feature to align:\n
        'increase' for maximum derivative,\n
        'decrease' for minimum derivative,\n
        'extrema' for the mid-point between local neighboring minimum and maximum,\n
        'maximum' for the `n`-th local maximum,\n
        'minimum' for the `n`-th local minimum  
    n : int, optional
        Used for `feature='extrema'`: Index of extremas to use. Must be equal or larger than 1.

    Returns
    -------
    np.ndarray
        the shifted x-axis array
    """
    if feature in ['increase', 'decrease']:
        deriv = np.diff(signal)
        if feature == 'increase':
            z_idx = np.argmax(deriv)
        elif feature == 'decrease':
            z_idx = np.argmin(deriv)
        shift = z[z_idx]
    elif feature in ['maximum', 'minimum']:
        if feature == 'maximum':
            masks = [signal[1:-1] > signal[:-2], signal[1:-1] > signal[2:]]
        else:
            masks = [signal[1:-1] < signal[:-2], signal[1:-1] < signal[2:]]
        idcs = np.argwhere(np.all(masks, axis=0)).flatten()
        i = min(n - 1, len(idcs) - 1)
        idx = idcs[i] + 1
        shift = z[idx]
    else:
        idcs = find_extrema_indices(signal, n)
        n = min(n, len(idcs) - 1)
        if n < 1:
            return z
        min_idx, max_idx = idcs[n - 1], idcs[n]
        shift = z[int(min_idx + (max_idx - min_idx) / 2)]
    xshift = z - shift
    return xshift


def dtw_x_align(signal: np.ndarray, lead_z: np.ndarray, lead_signal: np.ndarray, 
                step_pattern: Literal['sym', 'rj'] = 'rj') -> np.ndarray:
    match step_pattern:
        case 'sym':
            step_pattern = 'symmetric2'
        case 'rj':
            step_pattern = dtw.rabinerJuangStepPattern(1, 'c')
    alignment = dtw.dtw(signal, lead_signal, step_pattern=step_pattern)
    return lead_z[alignment.index2], signal[alignment.index1]


def y_align(array: np.ndarray, far_param: float, method: Literal['mean', 'median']) -> np.ndarray:
    tail = round(0.80 * len(array))
    yshift = np.__getattribute__(method)(array[tail:])
    if yshift == far_param:
        return array
    else:
        return array - (yshift - far_param)


def scaling(measurements: list[Measurement], signal_type: str) -> list[Measurement]:
    sig_vals = np.concatenate([m[signal_type] for m in measurements])
    data_min, data_max = sig_vals.min(), sig_vals.max()
    for m in tqdm(measurements, desc='Min-max-scaling'):
        m[signal_type] = (m[signal_type] - data_min) / (data_max - data_min)
    return measurements


class MeasurementScaler:
    def __init__(self, measurements: list[Measurement]):
        self.ranges: dict[str, tuple[float, float]] = {}
        for signal_type in Measurement.signal_types():
            sig_vals = np.concatenate([m[signal_type] for m in measurements])
            self.ranges[signal_type] = (sig_vals.min(), sig_vals.max())
    
    def scale(self, measurement: Measurement, signal_type: str) -> np.ndarray:
        vmin, vmax = self.ranges[signal_type]
        return (measurement[signal_type] - vmin) / (vmax - vmin)
    
    def inverse_scale(self, measurement: Measurement, signal_type: str) -> np.ndarray:
        vmin, vmax = self.ranges[signal_type]
        return measurement[signal_type] * (vmax - vmin) + vmin


def preprocess(measurements: list[Measurement], calib_params: dict[str, float],
               scale: bool = False, inverse_scale: bool = True, scale_per_measurement: bool = False, smooth: bool = True, 
               smooth_func: Callable[..., np.ndarray|tuple[np.ndarray]] = denoise.gauss, 
               smooth_kwargs: dict[str, Any] = {'s': 4}, reduce_length: int = 512,
               yalign: Literal['mean', 'median']|None = 'median', 
               xalign: Literal['increase', 'decrease', 'extrema', 'maximum', 'minimum', 'sym', 'rj']|None = 'maximum', 
               xalign_guide_type: Literal['amp', 'phase'] = 'phase', 
               xalign_n: int = 1, xalign_guide_idx: int|None = None) -> tuple[list[Measurement], list[int]]:
    """
    Preprocess am-afm measurements from .ibw files.
    * smooth measurements to reduce noise
    * scale measurements using min-max-scaling
    * align measurements on x- and y-axis
    * store the data in `Measurement`-objects containing distance-, amplitude- and phase-data  
      for approach and retraction of an experiment.
    
    Parameters
    ----------
    measurements: list[Measurement]
        List of `Measurement`-objects created from `.ibw`-files to preprocess.
    scale : bool, optional
        Set to `True` to min-max-scale amplitude- and phase-data, by default False
    inverse_scale : bool, optional
        Set to `True` to reverse scaling of amplitude- and phase-data at the end, by default True
    scale_per_measurement : bool, optional
        Set to `True` to scale each measurement individually otherwise scale over all measurements, by default False.
    smooth : bool, optional
        Set to `True` to smooth amplitude- and phase-data to reduce noise, by default True
    smooth_func : Callable[..., np.ndarray | tuple[np.ndarray]], optional
        The function which applys smoothing on each curve. See `amafm.denoise`-module for available functions  
        by default denoise.gauss
    smooth_kwargs : _type_, optional
        Keyword arguments to pass to the smoothing-function, by default {'s': 4}
    reduce_length : int, optional
        Reduce curves to a given length by interpolating using cubic bsplines. Set >=1 to apply, by default 512
    yalign : Literal[&#39;mean&#39;, &#39;median&#39;] | None, optional
        Method for aligning the curves on the y-axis.  
        Either aligning them to the `mean`or `median` of all curves of the same measurement-type.  
        by default 'median'
    xalign : Literal[&#39;increase&#39;, &#39;decrease&#39;, &#39;extrema&#39;, &#39;maximum&#39;, 
                     &#39;minimum&#39;, &#39;sym&#39;, &#39;rj&#39;] | None, optional
        Method for aligning the curves on the x-axis.   
        * `increase` for aligning to the maximum derivative,
        * `decrease` for aligning to the minimum derivative,
        * `extrema` for aligning to the mid-point between local neighboring minimum and maximum,
        * `maximum` for aligning to the `n`-th local maximum,
        * `minimum` for aligning to the `n`-th local minimum,
        * `sym` for using the symmetric step pattern of the DTW-algorithm,
        * `rj` for using the Rabiner-Juang step pattern of the DTW-algorithm.
        by default 'maximum'
    xalign_guide_type : Literal[&#39;amp&#39;, &#39;phase&#39;], optional
        Not used for DTW x-alignment. The curve-type to base the x-alignment on.  
        Either `amp` for the amplitude or `phase`for the phase, by default 'phase'
    xalign_n : int, optional
        Only used for `xalign`-types `extrema`, `maximum` and `minimum`.  
        Chooses the `n`-th (0-based) identified feature along the x-axis to align at, by default 1
    xalign_guide_idx : int | None, optional
        Only used for DTW x-alignment. The index of loaded measurements on which the x-alignment is based, by default None

    Returns
    -------
    tuple[list[Measurement], list[int]]
        Return a list of preprocessed `Measurement`-objects and a list of indices of the measurements-list of failed measurements.
    """
    measurements = [m.copy() for m in measurements]
    signal_types, z_types = Measurement.signal_types(), Measurement.z_types()
    n_steps = sum([(reduce_length > 1), smooth, scale, (scale and inverse_scale), bool(yalign), bool(xalign)])
    step = 0
    skipped_idcs = []
      
    # reduce curve length
    if reduce_length > 1:
        step += 1
        skipped_measurements = []
        for i, m in tqdm(enumerate(measurements), total=len(measurements), 
                         desc=f'Step {step}/{n_steps}: Equalize lengths by interpolation'):
            try:
                z_in, m.amp_in = denoise.reduce_curve(m.z_in, m.amp_in, reduce_length)
                z_out, m.amp_out = denoise.reduce_curve(m.z_out, m.amp_out, reduce_length)
                _, m.phase_in = denoise.reduce_curve(m.z_in, m.phase_in, reduce_length)
                _, m.phase_out = denoise.reduce_curve(m.z_out, m.phase_out, reduce_length)
                m.z_in, m.z_out = z_in, z_out
            except AssertionError:
                skipped_measurements.append(m)
                skipped_idcs.append(i)
                print(f"   Curves in the file '{m.file_path}' are shorter than the desired `{reduce_length=}`. "
                      "Skipping file.")
                continue
        for m in skipped_measurements:
            measurements.remove(m)

    # denoise signals
    if smooth:
        step += 1
        for m in tqdm(measurements, desc=f'Step {step}/{n_steps}: Denoising'):
            for signal_type in signal_types:
                m[signal_type] = smooth_func(m[signal_type], **smooth_kwargs)
    
    # y-alignment
    if yalign:
        step += 1
        for m in tqdm(measurements, desc=f'Step {step}/{n_steps}: y-alignment'):
            for signal_type in signal_types:
                curve_metric = signal_type.split('_')[0]
                far_param = 0 if scale else calib_params[curve_metric + '_far']
                m[signal_type] = y_align(m[signal_type], far_param, yalign)

    # normalize to [0, 1]
    if scale:
        step += 1
        if scale_per_measurement:
            for m in tqdm(measurements, desc=f'Step {step}/{n_steps}: Min-max-scaling'):
                for signal_type in signal_types:
                    vmin, vmax = m[signal_type].min(), m[signal_type].max()
                    m[signal_type] = (m[signal_type] - vmin) / (vmax - vmin)
        else:
            mscaler = MeasurementScaler(measurements)
            for m in tqdm(measurements, desc=f'Step {step}/{n_steps}: Min-max-scaling'):
                for signal_type in signal_types:
                    m[signal_type] = mscaler.scale(m, signal_type)
    
    # process z-curves (x-alignment)
    if xalign:
        step += 1
        for m in tqdm(measurements, desc=f'Step {step}/{n_steps}: x-alignment'):
            if xalign in ['sym', 'rj']:
                guide = measurements[xalign_guide_idx]
                for signal_type in signal_types:
                    direction = signal_type.split('_')[1]
                    m[f'z_{direction}'], m[signal_type] = dtw_x_align(m[signal_type], guide[f'z_{direction}'], 
                                                                      guide[signal_type], xalign)
            else:
                for z_type in z_types:
                    direction = z_type.split('_')[1]
                    guide_curve = m[f'{xalign_guide_type}_{direction}']
                    m[z_type] = feature_x_align(m[z_type], guide_curve, xalign, xalign_n)
    
    # scale curves back again
    if scale and inverse_scale:
        step += 1
        if scale_per_measurement:
            pass
        else:
            for m in tqdm(measurements, desc=f'Step {step}/{n_steps}: Inverse min-max-scaling'):
                for signal_type in signal_types:
                    m[signal_type] = mscaler.inverse_scale(m, signal_type)
    
    return measurements, skipped_idcs
