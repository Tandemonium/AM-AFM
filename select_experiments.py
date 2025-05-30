# select good curves from preprocessed data
# usage:
# python select_experiments.py <data_dir> [-t <target>] [-n <n_files>] [-r <revise>]

import argparse
from amafm import selection


def parse_arguments():
    parser = argparse.ArgumentParser(description=("Open a GUI to screen curves and accept or reject measurements. "
                                                  "The classification of the measurements is saved to `screened_files.csv` "
                                                  "inside the given directory."))
    parser.add_argument('data_dir', type=str,
                        help="Path to directory containing experiment `.ibw`-files to load.")
    parser.add_argument('-t', '--target', type=int, default=100,
                        help="Target number of accepted curves (default: 100).")
    parser.add_argument('-n', '--n_files', type=int, default=-1,
                        help="Number of unseen files to load and preprocess (default: -1, meaning all files).")
    parser.add_argument('-r', '--revise', type=int, default=False,
                        help=("If a previously saved `screened_files.csv` exists and this is `True`, "
                              "these files are also included to revisit the labels, "
                              "if it is `False` it skips those files (default: False)."))
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    data_dir = args.data_dir
    target = args.target
    n_files = args.n_files
    revise = bool(args.revise)
    selection.gui_select_experiments(data_dir, target, n_files, revise)
