import dtw
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from tqdm import tqdm

from . import calibration, data_loading, denoise


@dataclass
class Measurement:
    z_in: np.ndarray
    z_out: np.ndarray
    phase_in: np.ndarray
    phase_out: np.ndarray
    amp_in: np.ndarray
    amp_out: np.ndarray

    @classmethod
    def signal_types(cls) -> list[str]:
        z_types = cls.z_types()
        return [name for name in cls.__match_args__ if name not in z_types]
    
    @classmethod
    def z_types(cls) -> list[str]:
        return [name for name in cls.__match_args__ if name.startswith('z')]
    
    def __getitem__(self, item: str) -> np.ndarray:
        return getattr(self, item)
    
    def __setitem__(self, key: str, value: np.ndarray) -> None:
        setattr(self, key, value)
    
    def copy(self) -> 'Measurement':
        return Measurement(**self.__dict__)
    
    def deepcopy(self) -> 'Measurement':
        return Measurement(self.z_in.copy(), self.z_out.copy(), self.phase_in.copy(), self.phase_out.copy(),
                           self.amp_in.copy(), self.amp_out.copy())


def separate_signal(signal_array: np.ndarray, turning_point: int) -> tuple[np.ndarray, np.ndarray]:
    curve_in = signal_array[:turning_point]
    curve_in = np.flip(curve_in)
    curve_out = signal_array[turning_point:]
    return curve_in, curve_out


def separate_drive(drive: np.ndarray, turning_point: int) -> tuple[np.ndarray, np.ndarray]:
    z_in = drive[:turning_point]
    z_out = drive[turning_point:]
    z_out = np.flip(z_out)
    return z_in, z_out


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


def matz_Uhlig(labels: list[str], wave_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    drive = wave_data[:, labels.index('Drive')]  # m
    drive = (drive - np.min(drive)) * 10**9  # nm, relative 0-point
    amp = wave_data[:, labels.index('Amp')]  # observable
    phase = wave_data[:, labels.index('Phase')]  # observable
    turning_point = np.argmax(drive)
    return drive, amp, phase, turning_point


def retrieve_signals(file: str|Path) -> Measurement:
    constants, labels, wave_data, name = data_loading.load_ibw_force(file)
    if np.isnan(wave_data).any():
        raise ValueError('NaN values in wave data.')
    drive, amp, phase, turning_point = matz_Uhlig(labels, wave_data)

    # separate curves into approach and retract curves
    z_in, z_out = separate_drive(drive, turning_point)
    phase_in, phase_out = separate_signal(phase, turning_point)
    amp_in, amp_out = separate_signal(amp, turning_point)

    return Measurement(z_in, z_out, phase_in, phase_out, amp_in, amp_out)


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


def preprocess(data_dir: str, num_files: int = -1, start_at: int = 0, folders: list[str]|None = None, 
               files: list[str|Path]|None = None, far_probe_avrg_tol: int = 100, 
               scale: bool = True, smooth: bool = True, reduce_length: int = -1,
               smooth_func: Callable[..., np.ndarray|tuple[np.ndarray]] = denoise.savgol, 
               smooth_kwargs: dict[str, Any] = {'w': 50, 'p': 3}, 
               yalign: Literal['mean', 'median']|None = 'median', 
               xalign: Literal['increase', 'decrease', 'extrema', 'maximum', 'minimum', 'sym', 'rj']|None = 'maximum', 
               xalign_guide_type: Literal['amp', 'phase'] = 'amp', 
               xalign_n: int = 1, xalign_guide_idx: int|None = None) -> tuple[list[Measurement], dict[str, float]]:
    """
    Preprocess am-afm measurements from .ibw files.
    * load data from igor-binarywave files
    * smooth measurements to reduce noise
    * scale measurements using min-max-scaling
    * align measurements on x- and y-axis
    * store the data in `Measurement`-objects containing distance-, amplitude- and phase-data  
      for approach and retraction of an experiment.
    
    Parameters
    ----------
    data_dir : str
        Path of the directory containing the data in .ibw-format.
    num_files : int, optional
        Number of files to load. -1 to load all files, by default -1
    start_at : int, optional
        Number of files in alphabetical order to skip, by default 0
    folders : list[str] | None, optional
        List of names of subfolders in the data directory. Only load files from given folders, by default None
    files : list[str | Path] | None, optional
        Restrict to certain filenames from which to load, by default None
    far_probe_avrg_tol : int, optional
        by default 100
    scale : bool, optional
        Set to `True` to min-max-scale amplitude- and phase-data, by default True
    smooth : bool, optional
        Set to `True` to smooth amplitude- and phase-data to reduce noise, by default True
    reduce_length : int, optional
        Reduce curves to a given length by interpolating using cubic bsplines. Set >=1 to apply, by default -1
    smooth_func : Callable[..., np.ndarray | tuple[np.ndarray]], optional
        The function which applys smoothing on each curve. See `amafm.denoise`-module for available functions  
        by default denoise.savgol
    smooth_kwargs : _type_, optional
        Keyword arguments to pass to the smoothing-function, by default {'w': 50, 'p': 3}
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
        Either `amp` for the amplitude or `phase`for the phase, by default 'amp'
    xalign_n : int, optional
        Only used for `xalign`-types `extrema`, `maximum` and `minimum`.  
        Chooses the `n`-th (0-based) identified feature along the x-axis to align at, by default 1
    xalign_guide_idx : int | None, optional
        Only used for DTW x-alignment. The index of loaded measurements on which the x-alignment is based, by default None

    Returns
    -------
    tuple[list[Measurement], dict[str, float]]
        Return a list of preprocessed `Measurement`-objects and a dictionary containing calibration parameters.
    """

    # load data and cailbration parameters
    if files is None:
        files = data_loading.get_ibw_paths(data_dir, calib_only=False, n=num_files, folders=folders)
        data_files = files['data'][start_at:]
        folders = set([f.parts[1] for f in data_files])
        calib_files = [fc for fc in files['calib'] if any([fc.parts[1].startswith(fo) for fo in folders])]
    calib_params = calibration.get_calibration_parameters(files=calib_files, far_probe_avrg_tol=far_probe_avrg_tol)
    signal_types, z_types = Measurement.signal_types(), Measurement.z_types()

    # retrieve separate signals from files
    i_max = sum([reduce := reduce_length > 1, smooth, scale, bool(yalign), bool(xalign)]) + 1
    i = 1
    measurements = []
    for file in tqdm(data_files, desc=f'Step {i}/{i_max}: Loading files'):
        try:
            m = retrieve_signals(file)
            measurements.append(m)
        except ValueError:
            print(f"   Error in file '{file}'. Skipping file.")
            continue
      
    # reduce curve length
    if reduce:
        i += 1
        for m in tqdm(measurements, desc=f'Step {i}/{i_max}: Denoising'):
            m.z_in, m.amp_in = denoise.reduce_curve(m.z_in, m.amp_in, reduce_length)
            m.z_out, m.amp_out = denoise.reduce_curve(m.z_out, m.amp_out, reduce_length)
            _, m.phase_in = denoise.reduce_curve(m.z_in, m.phase_in, reduce_length)
            _, m.phase_out = denoise.reduce_curve(m.z_out, m.phase_out, reduce_length)

    # denoise signals
    if smooth:
        i += 1
        for m in tqdm(measurements, desc=f'Step {i}/{i_max}: Denoising'):
            for signal_type in signal_types:
                m[signal_type] = smooth_func(m[signal_type], **smooth_kwargs)
        
    # normalize to [0, 1]
    if scale:
        i += 1
        mscaler = MeasurementScaler(measurements)
        for m in tqdm(measurements, desc=f'Step {i}/{i_max}: Min-max-scaling'):
            for signal_type in signal_types:
                m[signal_type] = mscaler.scale(m, signal_type)

    # y-alignment
    if yalign:
        i += 1
        for m in tqdm(measurements, desc=f'Step {i}/{i_max}: y-alignment'):
            for signal_type in signal_types:
                curve_metric = signal_type.split('_')[0]
                far_param = 0 if scale else calib_params[curve_metric + '_far']
                m[signal_type] = y_align(m[signal_type], far_param, yalign)
    
    # process z-curves (x-alignment)
    if xalign:
        i += 1
        for m in tqdm(measurements, desc=f'Step {i}/{i_max}: x-alignment'):
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
    
    return measurements, calib_params
