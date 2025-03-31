# select good curves as target data
from amafm import selection


DATA_DIR: str = '../experiments'  # path to directory containing experiment data
num_files: int = -1               # number of files to load and preprocess
start_at: int = 0                 # to skip a given number of files
folders: list[str] = None         # add a list of names of specific experiment-subfolders to load just those


if __name__ == '__main__':
    selection.gui_select_experiments(DATA_DIR, start_at, num_files, folders)
