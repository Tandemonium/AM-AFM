import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import ensemble, metrics, svm
from sklearn import model_selection as ms

from . import selection
from .preprocessing import Measurement


def split_training_data(measurements: list[Measurement], test_size: float = 0.2, 
                        seed: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    good_idcs = np.array(selection.load_selected_idcs())
    good_idcs = good_idcs[good_idcs < len(measurements)]
    bad_idcs = np.arange(len(measurements))[~good_idcs]
    good_train, good_test, bad_train, bad_test = ms.train_test_split(good_idcs, bad_idcs, 
                                                                     test_size=test_size, 
                                                                     random_state=seed)
    
    train_idcs = np.concatenate([good_train, bad_train])
    train_target = np.array([True] * len(good_train) + [False] * len(bad_train))
    shuffle_idcs = np.random.permutation(len(train_idcs))
    train_idcs, train_target = train_idcs[shuffle_idcs], train_target[shuffle_idcs]

    test_idcs = np.concatenate([good_test, bad_test])
    test_target = np.array([True] * len(good_test) + [False] * len(bad_test))
    shuffle_idcs = np.random.permutation(len(test_idcs))
    test_idcs, test_target = test_idcs[shuffle_idcs], test_target[shuffle_idcs]

    m_array = np.array(measurements)
    return m_array[train_idcs], train_target, m_array[test_idcs], test_target


def create_training_data(train_measurements: list[Measurement], test_measurements: list[Measurement],
                         curve_types: str|list[str]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(curve_types, str):
        curve_types = [curve_types]
    train_data = [np.concatenate([m[ct] for ct in curve_types]) for m in train_measurements]
    test_data = [np.concatenate([m[ct] for ct in curve_types]) for m in test_measurements]
    return np.array(train_data), np.array(test_data)


def evaluate_model(model, test_X: np.ndarray, test_targets: np.ndarray,
                   print_conf_mat: bool = True) -> tuple[dict[str, float], np.ndarray]:
    pred = model.predict(test_X)
    scores = {
        'acc': metrics.balanced_accuracy_score(test_targets, pred),
        'prec': metrics.precision_score(test_targets, pred),
        'rec': metrics.recall_score(test_targets, pred),
        'f1': metrics.f1_score(test_targets, pred),
    }
    conf_mat = metrics.confusion_matrix(test_targets, pred)
    if print_conf_mat:
        conf_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        conf_disp.plot(cmap=plt.cm.PiYG, values_format=".2f")  # vanimo, coolwarm
    return scores, conf_mat


def train_curve_selection_models(measurements: list[Measurement], curve_types: str|list[str] = 'amp_out',
                                 test_size: float = 0.2, seed: int = 12, 
                                 print_conf_mat: bool = True) -> tuple[svm.SVC, 
                                                                       ensemble.RandomForestClassifier, 
                                                                       pd.DataFrame]:
    train_m, train_targets, test_m, test_targets = split_training_data(measurements, test_size, seed)
    train_X, test_X = create_training_data(train_m, test_m, curve_types=curve_types)

    results = {}
    model_svm = svm.SVC()
    model_svm.fit(train_X, train_targets)
    scores, conf_mat = evaluate_model(model_svm, test_X, test_targets, print_conf_mat)
    results[model_svm.__class__.__qualname__] = scores

    model_rf = ensemble.RandomForestClassifier(n_jobs=-1, random_state=seed)
    model_rf.fit(train_X, train_targets)
    scores, conf_mat = evaluate_model(model_rf, test_X, test_targets, print_conf_mat)
    results[model_rf.__class__.__qualname__] = scores
    results = pd.DataFrame(results).T
    return model_svm, model_rf, results


def save_model(model, model_dir: str):
    joblib.dump(model, f'{model_dir}_{model.__class__.__qualname__}.joblib')


def load_model(model_dir: str, model_name: str):
    return joblib.load(f'{model_dir}_{model_name}.joblib')
