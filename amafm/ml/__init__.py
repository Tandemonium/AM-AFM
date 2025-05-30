import joblib
import os

from pathlib import Path
from typing import Any, Literal
from IPython.display import display

import numpy as np
import pandas as pd
import torch

from sklearn import ensemble, metrics, svm
from sklearn import model_selection as ms
from sklearn.base import BaseEstimator

from tqdm import trange

from .. import data_loading, preprocessing, selection, visualization
from ..preprocessing import Measurement

from .models import *
from .trainer import NeuralNetworkClassifier


MODELS = {
    'SVM': (
        svm.SVC, 
        {
            'kernel': 'rbf',
        }
    ),
    'RF': (
        ensemble.RandomForestClassifier, 
        {
            'n_estimators': 20, 
            'n_jobs': -1, 
            'random_state': 22,
        }
    ),
    'GB': (
        ensemble.GradientBoostingClassifier, 
        {
            'n_estimators': 20, 
            'max_depth': 10, 
            'random_state': 22,
        }
    ),
    'CRF': (
        CRF, 
        {
            'algorithm': 'lbfgs',
        }
    ),
    'CNN': (
        NeuralNetworkClassifier, 
        {
            'name': 'CNN', 
            'model': CNN, 
            'save_dir': 'saves',
            'in_channels': 1, 
            'out_channels': 1, 
            'batch_size': 64,
            'val_size': 0.2, 
            'expand': 1, 
            'num_workers': 4, 
            'max_epochs': 500,
            'hidden_dims': [16, 32, 64, 128, 256, 512, 1000], 
            'input_len': 512,
        }
    ),
    'CNN_cutoff': (
        NeuralNetworkClassifier, 
        {
            'name': 'CNN', 
            'model': CNN, 
            'save_dir': 'saves',
            'in_channels': 1, 
            'out_channels': 1, 
            'batch_size': 64,
            'val_size': 0.2, 
            'expand': 1, 
            'num_workers': 4, 
            'max_epochs': 500,
            'hidden_dims': [16, 32, 64, 128, 256, 512, 1000], 
            'input_len': 100,
        }
    ),
    'CNN_BN_DO': (
        NeuralNetworkClassifier, 
        {
            'name': 'CNNDropout', 
            'model': CNN, 
            'save_dir': 'saves',
            'in_channels': 1, 
            'out_channels': 1, 
            'batch_size': 64,
            'val_size': 0.2, 
            'expand': 1, 
            'num_workers': 4, 
            'max_epochs': 500,
            'hidden_dims': [128, 256, 1000], 
            'input_len': 512,
            'norm': True, 
            'dropout': 0.5,
        }
    ),
    'CNN_BN_DO_cutoff': (
        NeuralNetworkClassifier, 
        {
            'name': 'CNNDropout', 
            'model': CNN, 
            'save_dir': 'saves',
            'in_channels': 1, 
            'out_channels': 1, 
            'batch_size': 64,
            'val_size': 0.2, 
            'expand': 1, 
            'num_workers': 4, 
            'max_epochs': 500,
            'hidden_dims': [128, 256, 1000], 
            'input_len': 100,
            'norm': True, 
            'dropout': 0.5,
        }
    ),
    'LSTM': (
        NeuralNetworkClassifier, 
        {
            'name': 'LSTM', 
            'model': LSTM, 
            'save_dir': 'saves',
            'in_channels': 512, 
            'out_channels': 1, 
            'batch_size': 16,
            'val_size': 0.2, 
            'num_workers': 4, 
            'max_epochs': 500,
            'lstm_size': 128, 
            'n_lstm_layers': 3, 
            'fc_size': 1000, 
            'dropout': 0.5,
        }
    ),
}


MCT_TYPE = list[tuple[list[Measurement], dict[str, float]]]
def load_dataset(data_dir: str, folders: list[str]) -> tuple[pd.DataFrame, MCT_TYPE]:
    """
    Load the `screened_files.csv` from the given folders and load the data. 

    Parameters
    ----------
    data_dir : str
        Directory containing the given `folders`.
    folders : list[str]
        A list of folders, each containing .ibw-files and a `screened_files.csv`.

    Returns
    -------
    tuple[pd.DataFrame, MCT_TYPE]
        A dataframe with the screening-results and a list of tuples, 
        each containing a list of `Measurement` objects and a dictionary with the respective calibration parameters.
    """
    data_df = []
    mct = []
    data_dir = Path(data_dir)
    for folder in folders:
        directory = data_dir / folder
        df = selection.load_screening_results(directory)
        measurements, calib_params = data_loading.load_data(files=df.filepath.tolist())
        data_df.append(df)
        mct.append((measurements, calib_params))
    return pd.concat(data_df, ignore_index=True), mct


def preprocess_data(data_df: pd.DataFrame, mct: MCT_TYPE, 
                    curve_type: Literal['phase_in', 'phase_out', 'amp_in', 'amp_out'] = 'phase_out', 
                    sample_cutoff: int|None = None, **preprocess_kwargs) -> tuple[pd.DataFrame, list[Measurement]]:
    """
    Preprocess the data. 
    The preprocess_kwargs are passed to the `preprocessing.preprocess`-method.
    
    Returns the dataframe with the screening-results and a list of measurements.
    """
    measurements = []
    for m, c in mct:
        m_list, _ = preprocessing.preprocess(m, c, **preprocess_kwargs)
        measurements.extend(m_list)

    # drop filepaths from which data could not be loaded
    m_filepaths = [m.file_path for m in measurements]
    data_df = data_df.drop(index=data_df.index[~data_df['filepath'].isin(m_filepaths)])
    data_df['curve'] = [m[curve_type][:sample_cutoff] for m in measurements]
    return data_df, measurements


def train_test_split(df_accepted: pd.DataFrame, df_rejected: pd.DataFrame, test_size: float = 0.2,
                     seed: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    # random split into train and test
    train, test = ms.train_test_split(df_accepted, test_size=test_size, random_state=seed)
    train2, test2 = ms.train_test_split(df_rejected, test_size=test_size, random_state=seed + 1)
    train = pd.concat([train, train2])
    test = pd.concat([test, test2])
    return train, test


def transform_data(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X, y = data['curve'], data['accept']
    return np.stack(X.values), y.values


def split_data(df_accepted: pd.DataFrame, df_rejected: pd.DataFrame, test_size: float = 0.2,
                seed: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train, test = train_test_split(df_accepted, df_rejected, test_size=test_size, seed=seed)
    X_train, y_train = transform_data(train)
    X_test, y_test = transform_data(test)
    return X_train, y_train, X_test, y_test


def evaluate_model(model, test_X: np.ndarray, test_targets: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    pred = model.predict(test_X)
    scores = {
        'acc': metrics.balanced_accuracy_score(test_targets, pred),
        'prec': metrics.precision_score(test_targets, pred),
        'rec': metrics.recall_score(test_targets, pred),
        'f1': metrics.f1_score(test_targets, pred),
    }
    conf_mat = metrics.confusion_matrix(test_targets, pred, normalize='all')
    return scores, conf_mat


def train_curve_selection_models(name: str, screening_data: pd.DataFrame, mct: MCT_TYPE, 
                                 model_constr: list[tuple[type[BaseEstimator], dict[str, Any]]], cross_val_k: int = 5, 
                                 model_dir: str = 'models', test_size: float = 0.2, 
                                 deciding_score: Literal['acc', 'prec', 'rec', 'fi'] = 'prec', seed: int = 12, 
                                 curve_type: Literal['phase_in', 'phase_out', 'amp_in', 'amp_out'] = 'phase_out', 
                                 cutoff: int|None = None, **preprocess_kwargs) -> tuple[list[BaseEstimator], pd.DataFrame, 
                                                                                        dict[str, np.ndarray]]:
    """
    Train one or more models for curve selection. 
    The models are first evaluated using k-fold cross validation.
    The best performing model is trained with the full dataset and saved to disk.

    Parameters
    ----------
    name : str
        Name of this model training, used when saving the model.
    data : pd.DataFrame
        Dataframe containing screening results. Available via the `load_dataset`-function.
    mct : list[tuple[list[Measurement], dict[str, float]]]
        A list of tuples, each containing a list of `Measurement` objects and a dictionary 
        with the respective calibration parameters. Available via the `load_dataset`-function.
    model_constr : list[tuple[type[BaseEstimator], dict[str, Any]]]
        A list of tuples, each containing a model class and a dictionary with the model's keyword parameters.
        Many predefined tuples are available in the `amafm.ml.MODELS` dictionary.
    cross_val_k : int, optional
        Number of cross validation folds, by default 5
    model_dir : str, optional
        Directory where the final model is saved to, by default 'models'
    test_size : float, optional
        Size of the test set used for cross validation, by default 0.2
    deciding_score : Literal['acc', 'prec', 'rec', 'fi'], optional
        Name of the score, which is used to determine the best performance, by default 'prec'
    seed : int, optional
        Seed for all randomizations, by default 12
    curve_type : Literal[&#39;phase_in&#39;, &#39;phase_out&#39;, &#39;amp_in&#39;, &#39;amp_out&#39;], optional
        Curve type contained in a `Measurement` object, on which the selection is based on, by default 'phase_out'
    cutoff : int | None, optional
        Only use the part of the curves until this index for model training. 
        This focuses the training on the beginning of the curve, which contains the most significant information, by default None

    Returns
    -------
    tuple[BaseEstimator, pd.DataFrame, dict[str, np.ndarray]]
        The best performing fully trained model, a `DataFrame` containing the evaluation scores 
        and a dictionary containg a confusion matrix, averaged over the validation steps, for each model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed - 1)

    # preprocess data
    if preprocess_kwargs.get('scale', False):
        preprocess_kwargs['scale_per_measurement'] = True
    screening_data, _ = preprocess_data(screening_data, mct, curve_type, cutoff, **preprocess_kwargs)

    # separate accept-status to balance splits
    df_accepted = screening_data[screening_data.accept]
    df_rejected = screening_data[~screening_data.accept]

    results = {}
    conf_mats = {}
    best_score = 0.0
    best_model = None
    for i in trange(cross_val_k, desc='Validation cycle'):
        models = [model(**kwargs) for model, kwargs in model_constr]
        X_train, y_train, X_test, y_test = split_data(df_accepted, df_rejected, test_size=test_size, seed=seed + (i * 2))
        for j, model in enumerate(models):
            model.fit(X_train, y_train)
            scores, conf_mat = evaluate_model(model, X_test, y_test)
            model_name = f'{name}_{j}'
            results[(model_name, i)] = scores
            print(f'{model_name, i}:{scores}')
            if scores[deciding_score] > best_score:
                best_score = scores[deciding_score]
                best_model = model
            if model_name not in conf_mats:
                conf_mats[model_name] = []
            conf_mats[model_name].append(conf_mat)
    results = pd.DataFrame(results).T.groupby(level=0).mean()

    # save the best performing model to disk
    X, y = transform_data(pd.concat([df_accepted, df_rejected]))
    best_model_number = int(results[deciding_score].idxmax().split('_')[-1])
    best_model_cls, best_model_kwargs = model_constr[best_model_number]
    best_model = best_model_cls(**best_model_kwargs)
    best_model.fit(X, y)
    scores, conf_mat = evaluate_model(best_model, X, y)
    results.loc['full_model'] = scores
    conf_mats['full_model'] = [conf_mat]
    conf_mats = {name: np.mean(cms, axis=0) for name, cms in conf_mats.items()}
    save_model(name, best_model, model_dir, curve_type, cutoff, **preprocess_kwargs)
    return best_model, results, conf_mats


def save_model(name: str, model, model_dir: str, curve_type: Literal['phase_in', 'phase_out', 'amp_in', 'amp_out'] = 'phase_out', 
               cutoff: int|None = None, **preprocess_kwargs):
    model.preprocess_kwargs = preprocess_kwargs
    model.cutoff = cutoff
    model.curve_type = curve_type
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    name = model.name if hasattr(model, 'name') else name
    joblib.dump(model, f'{model_dir}/{name}.joblib')


def load_model(model_dir: str, model_name: str):
    return joblib.load(f'{model_dir}/{model_name}.joblib')


def predict(experiments_dir: str|Path, measurements_raw: list[Measurement]|None = None, calib_params: dict[str, float]|None = None, 
            model_path: str='models/SVM.joblib', 
            far_probe_avrg_tol: int = 100) -> tuple[list[Measurement], dict[str, float]]:
    """
    Classify measurements using a pre-trained model and return accepted instances of 
    unpreprocessed measurements.

    Parameters
    ----------
    experiments_dir : str | Path
        Store results here. If `measurements` is not given, also load experiments from here, 
        by default `None`
    measurements : list[Measurement] | None, optional
        Raw unprocessed measurements to classify. If `None` load from `experiments_dir`, 
        by default `None`
    model_path : str, optional
        Path to a pre-trained model, by default 'models/SVM.joblib'
    far_probe_avrg_tol : int, optional
        Tolerance for averaging the far probe signal, only used if loading measurements from directory, 
        by default 100

    Returns
    -------
    tuple[list[Measurement], dict[str, float]]
        A list of raw (not preprocessed) measurements classified as acceptable 
        and the calibration parameters required for preprocessing.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file {model_path} does not exist.')
    experiments_dir = Path(experiments_dir)
    model = joblib.load(model_path)
    if measurements_raw is None:
        print(f'> Load data from files for experiment {Path(experiments_dir).name}...')
        measurements_raw, calib_params = data_loading.load_data(experiments_dir, far_probe_avrg_tol=far_probe_avrg_tol)
    print('> Preprocess measurements...')
    measurements, skipped_idcs = preprocessing.preprocess(measurements_raw, calib_params, **model.preprocess_kwargs)
    curves = [m[model.curve_type] for m in measurements]
    curves = np.stack(curves)[:, :model.cutoff]
    pred = model.predict(curves)
    
    screening_df = selection.load_screening_results(experiments_dir)
    accepted_measurements = []
    measurements_raw = [m for i, m in enumerate(measurements_raw) if i not in skipped_idcs]
    for m, p in zip(measurements_raw, pred):
        selection.write_to_df(screening_df, m.file_path, p)
        if p:
            accepted_measurements.append(m)
    selection.save_screening_results(screening_df, experiments_dir)
    print(f'> Found {len(accepted_measurements)} acceptable measurements out of {len(measurements_raw)}.')
    print(f'> Saved classification results to `{experiments_dir / selection.SAVE_NAME}`.')
    return accepted_measurements, calib_params


def print_results(results: pd.DataFrame, conf_mats: dict[str, np.ndarray]):
    """
    Print the scores and display the confusion matrices of the model training.

    Parameters
    ----------
    results : pd.DataFrame
        The `DataFrame` containing the evaluation scores, returned by the `train_curve_selection_models`-function.
    conf_mats : dict[str, np.ndarray]
        The dictionary containg a confusion matrix for each model, 
        returned by the `train_curve_selection_models`-function.
    """
    labels = ['reject', 'accept']
    for name, cm in conf_mats.items():
        visualization.plot_conf_mat(cm, labels, name)
    display(results)
