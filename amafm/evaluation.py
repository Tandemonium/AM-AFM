from typing import Any

import numpy as np
import pandas as pd

from sklearn import metrics

from . import analysis, preprocessing
from .preprocessing import Measurement


signal_types = Measurement.signal_types()


def _signaltonoise(a: np.ndarray, axis=0, ddof=0) -> np.ndarray:
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 0 if sd == 0 else m/sd


def _snr_increase(y_true: np.ndarray, y_denoised: np.ndarray) -> np.ndarray:
    return _signaltonoise(y_denoised) / _signaltonoise(y_true)


def _smooth_all(measurements: list[Measurement], smooth_func: callable, 
               **smooth_kwargs) -> list[Measurement]:
    new_measurements = []
    for m in measurements:
        new_data = [m.z_in, m.z_out]
        for signal_type in signal_types:
            c = smooth_func(m[signal_type], **smooth_kwargs)
            new_data.append(c)
        new_measurements.append(Measurement(*new_data))
    return new_measurements


def _evaluate_smoothing_method(measurements: list[Measurement], all: bool, 
                               smooth_func: callable, **smooth_kwargs) -> pd.DataFrame:
    s_types = signal_types if all else ['amp_out']
    denoised = _smooth_all(measurements, smooth_func, **smooth_kwargs)
    df = pd.DataFrame(columns=['r2', 'snr', 'ccorr', 'rmse'])
    for m, d in zip(measurements, denoised):
        for signal_type in s_types:
            r2 = metrics.r2_score(m[signal_type], d[signal_type])                   # share of explained variance, maximize
            with np.errstate(divide="ignore", invalid="ignore"): 
                ccorr = np.corrcoef(m[signal_type], d[signal_type])[0, 1]           # similarity of signals, maximize 2.
            snr = _snr_increase(m[signal_type], d[signal_type])                     # preservation of signal, maximize 1.
            rmse = metrics.root_mean_squared_error(m[signal_type], d[signal_type])  # deviation from original signal, minimize
            df.loc[len(df)] = [ r2, snr,ccorr, rmse]
    return df


def evaluate_smoothing(measurements: list[Measurement], smoothing_methods: list[dict[str, Any]], 
                       all: bool = False) -> pd.DataFrame:
    results = pd.DataFrame()
    for smooth_method in smoothing_methods:
        df = _evaluate_smoothing_method(measurements, all, smooth_method['smooth_func'], 
                                        **smooth_method['smooth_kwargs'])
        results[f'{smooth_method['smooth_func'].__name__} {smooth_method['smooth_kwargs']}'] = df.mean()
    return results.T.sort_values(['r2', 'snr'], ascending=False)


def evaluate_curve_average(avrg_measurement: Measurement) -> float:
    # TODO: determine accuracy of curve average
    raise NotImplementedError


def _evaluate_preproc_config(data_dir: str, results: pd.DataFrame, config: dict[str, Any], 
                             fixed_kwargs: dict[str, Any]) -> None:
    measurements, calib_params = preprocessing.preprocess(data_dir, **config, **fixed_kwargs)
    avrg_measurement, aligned_measurements = analysis.average_curves(measurements, direction='out', 
                                                                     method='bin', bin_width=5)
    res = evaluate_curve_average(avrg_measurement)
    results.loc[len(results)] = [config, res]


def _get_best_config(results: pd.DataFrame, by: str, ascending: bool = False) -> dict[str, Any]:
    best_res = results.sort_values(by, ascending=ascending).iloc[0]
    return best_res.config


def evaluate_preprocessing(data_dir: str, fixed_kwargs: dict[str, Any], 
                           smoothing_configs: list[dict[str, Any]], 
                           sort_by: str = 'accuracy') -> pd.DataFrame:
    # basic config
    results = pd.DataFrame(columns=['config', 'accuracy'])
    best_config = {}
    _evaluate_preproc_config(data_dir, results, best_config, fixed_kwargs)

    # evalaute yalign
    for yalign in ['mean', 'median']:
        best_config['yalign'] = yalign
        _evaluate_preproc_config(data_dir, results, best_config, fixed_kwargs)
    best_conf = _get_best_config(results, sort_by)

    # evalaute xalign
    for xalign in ['extrema', 'sym', 'rj']:
        best_conf['xalign'] = xalign
        _evaluate_preproc_config(data_dir, results, best_conf, fixed_kwargs)
    best_conf = _get_best_config(results, sort_by)

    # evalaute scaling
    for scale in [False, True]:
        best_conf['scale'] = scale
        _evaluate_preproc_config(data_dir, results, best_conf, fixed_kwargs)
    best_conf = _get_best_config(results, sort_by)

    # evaluate smoothing
    for sc in smoothing_configs:
        best_conf['smooth_func'] = sc['smooth_func']
        best_conf['smooth_kwargs'] = sc['smooth_kwargs']
        _evaluate_preproc_config(data_dir, results, best_conf, fixed_kwargs)
    best_conf['smooth'] = False
    _evaluate_preproc_config(data_dir, results, best_conf, fixed_kwargs)
    best_conf = _get_best_config(results, sort_by)

    # evaluate xalign_guide
    for gt in ['amp', 'phase']:
        best_conf['xalign_guide_type'] = gt
        _evaluate_preproc_config(data_dir, results, best_conf, fixed_kwargs)
    best_conf = _get_best_config(results, sort_by)

    return results.sort_values(sort_by, ascending=False)


def evaluate_averaging(data_dir: str, preprocessing_kwargs: dict[str, Any], 
                       configurations: list[dict[str, Any]]) -> pd.DataFrame:
    measurements, calib_params = preprocessing.preprocess(data_dir, **preprocessing_kwargs)
    results = pd.DataFrame(columns=['config', 'accuracy'])
    for config in configurations:
        avrg_measurement, aligned_measurements = analysis.average_curves(measurements, **config)
        res = evaluate_curve_average(avrg_measurement)
        results.loc[len(results)] = [config, res]
    return results.sort_values('accuracy', ascending=False)
