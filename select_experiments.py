# select good curves as target data
from amafm import selection


DATA_DIR = '../experiments'  # path to directory containing experiment data
num_files = -1               # number of files to load and preprocess
start_at = 0                 # to skip a given number of files


if __name__ == '__main__':
    selection.gui_select_experiments(DATA_DIR, start_at, num_files)
