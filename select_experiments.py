# select good curves as target data
from amafm import selection


DATA_DIR = 'experiments'
num_files = -1
start_at = 0  # 400


if __name__ == '__main__':
    selection.gui_select_experiments(DATA_DIR, start_at, num_files)
