from typing import Any, Callable, Literal

import dtw
import numpy as np
import pandas as pd

from scipy import spatial
from sklearn import metrics
from tqdm import tqdm, trange

from . import ml, visualization
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


class ClassificationByDistance:
    def __init__(self, data_dir: str, folders: list[str], test_size: float = 0.25, 
                 signal_type: Literal['phase', 'amp'] = 'phase', direction: Literal['in', 'out'] = 'out',
                 cutoff: int|None = None, **preprocess_kwargs):
        self.test_size = test_size

        # load filepaths and data
        df, mct = ml.load_dataset(data_dir, folders)
        df, measurements = ml.preprocess_data(df, mct, curve_type=f'{signal_type}_{direction}', sample_cutoff=cutoff, 
                                              **preprocess_kwargs)

        # add z-arrays to dataframe
        df['z'] = [m[f'z_{direction}'] for m in measurements]

        # separate accept-status to balance splits
        self.df_accepted = df[df.accept]
        self.df_rejected = df[~df.accept]
    
    def evaluate_classification(self, name: str, distance_func: Callable[[pd.DataFrame, pd.Series], int], 
                                k: int = 40, in_numpy: bool = False) -> None:
        # k-fold cross validation
        scores = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
        conf_mats = []
        for i in trange(k, desc='Validation cycle'):
            f_train, f_test = ml.train_test_split(self.df_accepted, self.df_rejected, self.test_size, seed=i * 2)
            f_train = f_train.reset_index(drop=True)

            # find closest train-item for each test-item
            pred = []
            for _, test_row in tqdm(f_test.iterrows(), desc='Compute distances', leave=False, total=len(f_test)):
                if in_numpy:
                    train = np.stack(f_train['curve'].to_numpy())
                    pred.append(distance_func(f_train, train, test_row['curve']))
                else:
                    pred.append(distance_func(f_train, test_row['curve']))
            scores['acc'].append(metrics.balanced_accuracy_score(f_test.accept, pred))
            scores['prec'].append(metrics.precision_score(f_test.accept, pred))
            scores['rec'].append(metrics.recall_score(f_test.accept, pred))
            scores['f1'].append(metrics.f1_score(f_test.accept, pred))
            conf_mats.append(metrics.confusion_matrix(f_test.accept, pred, normalize='all'))
        scores = {k: np.mean(v) for k, v in scores.items()}
        conf_mat = np.mean(conf_mats, axis=0)
        print(f'{name}:')
        for k, v in scores.items():
            print(f'{k:4}  {v:.2f}')
        visualization.plot_conf_mat(conf_mat, ['reject', 'accept'], name)
    
    def get_pred(self, df: pd.DataFrame, values: np.ndarray, by: Literal['min', 'max'], method: Literal['abs', 'mean']) -> bool:
        if method == 'abs':
            attr = f'arg{by}'
            return df.iloc[getattr(values, attr)()].accept
        else:
            acc_dist = values[df.accept].mean()
            rej_dist = values[~df.accept].mean()
            if by == 'min':
                return True if acc_dist < rej_dist else False
            else:
                return True if acc_dist > rej_dist else False
    
    # predict by minimum distance
    def dist_min_norm(self, df_train: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> int:
        distances = np.linalg.norm(train - test, axis=1)
        return self.get_pred(df_train, distances, by='min', method='abs')
    
    def dist_min_squared(self, df_train: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> int:
        distances = ((train - test)**2).mean(axis=1)
        return self.get_pred(df_train, distances, by='min', method='abs')
    
    def dist_min_abs(self, df_train: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> int:
        distances = np.abs(train - test).mean(axis=1)
        return self.get_pred(df_train, distances, by='min', method='abs')
    
    def dist_min_max(self, df_train: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> int:
        max_dists = np.abs(train - test).max(axis=1)
        return self.get_pred(df_train, max_dists, by='min', method='mean')

    # predict by correlation coefficient
    def dist_correlation(self, df_train: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> int:
        an = train - train.mean(axis=1, keepdims=True)
        bn = test - test.mean()
        corrs = np.sum(an * bn, axis=1) / np.sqrt(np.sum(an**2, axis=1) * np.sum(bn**2))
        return self.get_pred(df_train, corrs, by='max', method='mean')

    # predict using dtw
    def dist_dtw(self, df_train: pd.DataFrame, test: pd.Series) -> int:
        distances = df_train['curve'].apply(lambda array: dtw.dtw(test, array).distance)
        return self.get_pred(df_train, distances, by='min', method='abs')

    # predict by procrustes distance
    def dist_procrustes(self, df_train: pd.DataFrame, test_row: pd.Series) -> int:
        disparities = df_train[['curve', 'z']].apply(lambda array: 
                                                    spatial.procrustes(np.stack([test_row.curve, test_row.z])[:, :100], 
                                                                        np.stack(array)[:, :100])[2], axis=1)
        pred = df_train.loc[disparities.idxmin()].accept
        return pred
    
    def by_min_norm(self, k: int = 20):
        self.evaluate_classification('min norm', self.dist_min_norm, k=k, in_numpy=True)

    def by_min_max(self, k: int = 20):
        self.evaluate_classification('min maxium error', self.dist_min_max, k=k, in_numpy=True)
    
    def by_min_squared(self, k: int = 20):
        self.evaluate_classification('min squared error', self.dist_min_squared, k=k, in_numpy=True)
    
    def by_min_abs(self, k: int = 20):
        self.evaluate_classification('min absolute error', self.dist_min_abs, k=k, in_numpy=True)

    def by_correlation(self, k: int = 20):
        self.evaluate_classification('correlation', self.dist_correlation, k=k, in_numpy=True)

    def by_dtw(self, k: int = 20):
        self.evaluate_classification('dtw', self.dist_dtw, k=k)

    def by_procrustes(self, k: int = 20):
        self.evaluate_classification('procrustes', self.dist_procrustes, k=k)
