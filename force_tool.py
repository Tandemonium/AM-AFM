# Load experiments, select acceptable measurements, preprocess and compute force curve.
# Resulting average curves and force curve are saved to the experiments-directory.
# usage:
# python force_tool.py <experiments_directory> [-t <number_target_curves>] [-f <force_step_size>] 
#     [-m <model_path_for_ml_selection>] [-s <signal_type>] [-d <probe_direction>]

import argparse

from pathlib import Path
from typing import Any, Literal

import numpy as np

from amafm import data_loading, preprocessing, selection, ml, analysis, force


def force_from_experiment(experiments_dir: str, target_curves: int = 100, force_step_size: int = 10, 
                          model_path: str = 'models/SVM.joblib', direction: Literal['in', 'out'] = 'out', 
                          data_load_kwargs: dict[str, Any] = {}, preprocess_kwargs: dict[str, Any] = {}, 
                          average_kwargs: dict[str, Any] = {}) -> np.ndarray:
    print('>', Path(model_path).resolve().as_posix(), '<')
    experiments_path = Path(experiments_dir).resolve()
    experiments_dir = experiments_path.as_posix()

    # load from folder
    print(f'Loading files from `{experiments_dir}`...')
    measurements, calib_params = data_loading.load_data(experiments_path, **data_load_kwargs)

    # load accepted files information from CSV
    screening_df = selection.load_screening_results(experiments_path)
    print(f'> Found screening results for {len(screening_df)} of which {len(screening_df[screening_df['accept']])} are acceptable.')

    # preprocess
    print('\nPreprocessing measurements...')
    measurements, _ = preprocessing.preprocess(measurements, calib_params, **preprocess_kwargs)
    print(f'> Loaded {len(measurements)} measurements from {experiments_dir}.\n')

    # ask user to enter either y(es) or n(o) wether they want to classifgy more curves
    classify = input('Do you want to select more curves by classifying them? [y/n]: ').strip().lower()
    if classify == 'y':
        classify = input('Classify curves using machine learning (ml) or manual selection GUI (man)? [ml/man]: ').strip().lower()

    # classify more measurements by ml or manual GUI
    if classify == 'ml':
        selected_measurements, calib_params = ml.predict(experiments_path, measurements, model_path)
        print(f'> {len(selected_measurements)} measurements have been selected by the classification model.')
    if classify == 'manual':
        # execute select_experiments.py
        selection.gui_select_experiments(experiments_dir, target=target_curves)
    
    # load updated accepted files information from CSV
    if classify != 'n':
        print('\nReloading accepted measurements...')
        measurements, calib_params = selection.load_accepted_data(experiments_dir, **data_load_kwargs)
        measurements, _ = preprocessing.preprocess(measurements, calib_params, **preprocess_kwargs)
        print(f'> Loaded {len(measurements)} selected measurements.')

    # -> average curves
    print('\nAverage measurements...')
    avrg_measurement = analysis.average_curves(measurements, **average_kwargs)
    if average_kwargs.get('zscores', None):
        avrg_measurement, zscore_measurements = avrg_measurement

    # -> force calculation
    print('\nCalculating force curve...')
    separation, force_curve = force.force_hoelscher(avrg_measurement, calib_params, direction, force_step_size)

    # -> save average measurement and force curve to disk
    print('\nSaving results...')
    analysis.save(avrg_measurement, experiments_dir)
    force.save(separation, force_curve, experiments_dir)
    print(f'> Saved average measurement as `{analysis.SAVE_NAME}` and force curve as `{force.SAVE_NAME}` to `{experiments_dir}`.')


def parse_arguments():
    parser = argparse.ArgumentParser(description=("Load files of an experiment, select acceptable measurements, "
                                                  "preprocess, average and compute force curve. "
                                                  "Resulting average curves and force curve are saved to the experiments-directory."))
    parser.add_argument('experiments_dir', type=str,
                        help='Path to directory containing the experiments `.ibw`-files to load.')
    parser.add_argument('-t', '--target', type=int, default=100,
                        help='Target number of accepted curves (default: 100).')
    parser.add_argument('-f', '--force_step_size', type=int, default=10,
                        help='Step size for averaging in the force calculation (default: 10).')
    parser.add_argument('-m', '--model_path', type=str, default='models/SVM.joblib',
                        help='Path to the machine learning model for classification (default: `models/SVM.joblib`).')
    parser.add_argument('-d', '--direction', type=str, choices=['in', 'out'], default='out',
                        help='Direction of the cantilever, either `in` or `out` (default: `out`).')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    experiments_dir = args.experiments_dir
    target_curves = args.target
    force_step_size = args.force_step_size
    model_path = args.model_path
    direction = args.direction
    force_from_experiment(experiments_dir, target_curves, force_step_size, model_path, direction)
