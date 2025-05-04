import joblib
import os

from typing import Any, Literal
from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import ensemble, metrics, svm
from sklearn import model_selection as ms
from sklearn.base import BaseEstimator

from tqdm import trange

from .. import data_loading, preprocessing, selection
from ..preprocessing import Measurement

from .models import CNN
from .trainer import NeuralNetworkClassifier


def load_dataset(data_dir: str, folders: list[str], 
                 curve_type: Literal['phase_in', 'phase_out', 'amp_in', 'amp_out'] = 'phase_out', 
                 **kwargs) -> tuple[pd.DataFrame, list[Measurement]]:
    """
    Load the `screened_files.csv` from the given folders and load and preprocess the data. 
    The kwargs are passed to the `preprocessing.preprocess`-method.
    
    Returns the dataframe with the screening-results and a list of measurements.
    """
    # load filepaths and data
    df_list = []
    m_list = []
    for folder in folders:
        df = selection.load_screening_results(f'{data_dir}/{folder}')
        measurements, calib_params = data_loading.load_data(data_dir, files=df.filepath.tolist())
        measurements = preprocessing.preprocess(measurements, calib_params, **kwargs)

        # drop filepaths which from which data could not be loaded
        m_filepaths = [m.file_path for m in measurements]
        df = df.drop(index=df.index[~df.filepath.isin(m_filepaths)])
        
        df_list.append(df)
        m_list.extend(measurements)
    df = pd.concat(df_list, ignore_index=True)
    df['curve'] = [m[curve_type] for m in m_list]
    return df, m_list


def train_test_split(df_accepted: pd.DataFrame, df_rejected: pd.DataFrame, test_size: float = 0.2,
                     seed: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    # random split into train and test
    train, test = ms.train_test_split(df_accepted, test_size=test_size, random_state=seed)
    train2, test2 = ms.train_test_split(df_rejected, test_size=test_size, random_state=seed + 1)
    train = pd.concat([train, train2])
    test = pd.concat([test, test2])
    return train, test


def split_data(df_accepted: pd.DataFrame, df_rejected: pd.DataFrame, test_size: float = 0.2,
                seed: int = 12) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    train, test = train_test_split(df_accepted, df_rejected, test_size=test_size, seed=seed)
    X_train, y_train = train['curve'], train['accept']
    X_test, y_test = test['curve'], test['accept']
    X_train, y_train, X_test, y_test = np.stack(X_train.values), y_train.values, np.stack(X_test.values), y_test.values
    return X_train, y_train, X_test, y_test


def evaluate_model(model, test_X: np.ndarray, test_targets: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    pred = model.predict(test_X)
    scores = {
        'acc': metrics.balanced_accuracy_score(test_targets, pred),
        'prec': metrics.precision_score(test_targets, pred),
        'rec': metrics.recall_score(test_targets, pred),
        'f1': metrics.f1_score(test_targets, pred),
    }
    conf_mat = metrics.confusion_matrix(test_targets, pred, normalize='true')
    return scores, conf_mat


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


def train_curve_selection_models(data_dir: str, folders: list[str], cross_val_k: int = 10,
                                 curve_type: Literal['phase_in', 'phase_out', 'amp_in', 'amp_out'] = 'phase_out', 
                                 save_dir: str = 'saves', model_dir: str = 'models', test_size: float = 0.2, epochs: int = 100, 
                                 batch_size: int = 64, val_size: float = 0.2, num_workers: int = 4, seed: int = 12, 
                                 **kwargs) -> tuple[list[BaseEstimator], pd.DataFrame, dict[str, list[np.ndarray]]]:
    # load filepaths and data
    df, measurements = load_dataset(data_dir, folders, curve_type, **kwargs)

    # separate accept-status to balance splits
    df_accepted = df[df.accept]
    df_rejected = df[~df.accept]

    results = {}
    conf_mats = {}
    for i in trange(cross_val_k, desc='Validation cycle'):
        models = [
            # svm.SVC(),
            # ensemble.RandomForestClassifier(n_jobs=-1, random_state=seed),
            NeuralNetworkClassifier(CNN, save_dir=save_dir, in_channels=1, out_channels=1, hidden_dims=[16, 32, 64, 128, 256, 512, 1000], 
                                    batch_size=batch_size, input_len=len(df.curve[0]), val_size=val_size, num_workers=num_workers, 
                                    max_epochs=epochs)
        ]

        X_train, y_train, X_test, y_test = split_data(df_accepted, df_rejected, test_size=test_size, seed=seed + (i * 2))
        for model in models:
            model.fit(X_train, y_train)
            scores, conf_mat = evaluate_model(model, X_test, y_test)
            model_name = model.__class__.__qualname__
            results[(model_name, i)] = scores

            if model_name not in conf_mats:
                conf_mats[model_name] = []
            conf_mats[model_name].append(conf_mat)
    results = pd.DataFrame(results).T.groupby(level=0).mean()

    for model in models:
        save_model(model, model_dir)
    return models, results, conf_mats


def save_model(model, model_dir: str):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(model, f'{model_dir}/{model.__class__.__qualname__}.joblib')


def load_model(model_dir: str, model_name: str):
    return joblib.load(f'{model_dir}/{model_name}.joblib')


def print_results(results: pd.DataFrame, conf_mats: dict[str, list[np.ndarray]]):
    labels = ['reject', 'accept']
    for name, cms in conf_mats.items():
        cm = np.mean(cms, axis=0)
        plot_conf_mat(cm, labels, name)
    display(results)
