from typing import Any, Callable, Literal

import dtw
import numpy as np
import pandas as pd

from scipy import spatial
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange

from . import analysis, data_loading, ml, preprocessing, selection
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
        new_measurements.append(Measurement(*new_data, m.file_path))
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


class ClassificationByDistance:
    def __init__(self, data_dir: str, folders: list[str], test_size: float = 0.25, 
                 signal_type: Literal['phase', 'amp'] = 'phase', direction: Literal['in', 'out'] = 'out'):
        self.test_size = test_size

        # load filepaths and data
        df, measurements = ml.load_dataset(data_dir, folders)

        # add z-arrays to dataframe
        df['z'] = [m[f'z_{direction}'] for m in measurements]

        # separate accept-status to balance splits
        self.df_accepted = df[df.accept]
        self.df_rejected = df[~df.accept]
    
    def evaluate_classification(self, name: str, distance_func: Callable[[pd.DataFrame, pd.Series], int], 
                                k: int = 40):
        # k-fold cross validation
        accuracies = []
        for i in trange(k, desc='Validation cycle'):
            f_train, f_test = ml.train_test_split(self.df_accepted, self.df_rejected, self.test_size, seed=i * 2)

            # find closest train-item for each test-item
            total = len(f_test)
            n_correct = 0
            for _, test_row in tqdm(f_test.iterrows(), desc='Compute distances', leave=False, total=len(f_test)):
                n_correct += distance_func(f_train, test_row)
            accuracies.append(n_correct / total)
        print(f"Mean accuracy ({name}): {sum(accuracies) / k:.2%}")
    
    # predict by minimum distance
    def dist_min_norm(self, df_train: pd.DataFrame, test_row: pd.Series) -> int:
        distances = df_train['curve'].apply(lambda array: np.linalg.norm(test_row.curve - array))
        pred = df_train.loc[distances.idxmin()].accept
        return 1 if pred == test_row.accept else 0

    # predict by lowest mean distance
    def dist_mean_norm(self, df_train: pd.DataFrame, test_row: pd.Series) -> int:
        distances = df_train['curve'].apply(lambda array: np.linalg.norm(test_row.curve - array))
        acc_dist = distances[df_train.accept].mean()
        rej_dist = distances[~df_train.accept].mean()
        pred = True if acc_dist < rej_dist else False
        return 1 if pred == test_row.accept else 0

    # predict by correlation coefficient
    def dist_correlation(self, df_train: pd.DataFrame, test_row: pd.Series) -> int:
        corrs = df_train['curve'].apply(lambda array: np.corrcoef(test_row.curve[:100], array[:100])[0, 1])
        pred = df_train.loc[corrs.idxmax()].accept
        return 1 if pred == test_row.accept else 0

    # predict using dtw
    def dist_dtw(self, df_train: pd.DataFrame, test_row: pd.Series) -> int:
        distances = df_train['curve'].apply(lambda array: dtw.dtw(test_row.curve[:100], array[:100]).distance)
        pred = df_train.loc[distances.idxmin()].accept
        return 1 if pred == test_row.accept else 0

    # predict by procrustes distance
    def dist_procrustes(self, df_train: pd.DataFrame, test_row: pd.Series) -> int:
        disparities = df_train[['curve', 'z']].apply(lambda array: 
                                                    spatial.procrustes(np.stack([test_row.curve, test_row.z])[:, :100], 
                                                                        np.stack(array)[:, :100])[2], axis=1)
        pred = df_train.loc[disparities.idxmin()].accept
        return 1 if pred == test_row.accept else 0
    
    def by_min_norm(self, k: int = 40):
        self.evaluate_classification('min norm', self.dist_min_norm, k=k)

    def by_mean_norm(self, k: int = 40):
        self.evaluate_classification('mean norm', self.dist_mean_norm, k=k)

    def by_correlation(self, k: int = 40):
        self.evaluate_classification('correlation', self.dist_correlation, k=k)

    def by_dtw(self, k: int = 40):
        self.evaluate_classification('dtw', self.dist_dtw, k=k)

    def by_procrustes(self, k: int = 40):
        self.evaluate_classification('procrustes', self.dist_procrustes, k=k)
