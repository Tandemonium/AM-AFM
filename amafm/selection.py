import glob
import pickle

from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.widgets import Button

from . import data_loading, preprocessing
from .data_loading import Measurement


SAVE_NAME = 'screened_files.csv'


def load_screening_results(dir: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(dir / SAVE_NAME, converters={'filepath': lambda fp: Path(dir, fp).resolve()})
    except FileNotFoundError:
        return pd.DataFrame(columns=['filepath', 'accept'])


def load_accepted_filepaths(dir: Path) -> list[Path]:
    df = load_screening_results(dir)
    return df[df['accept']].filepath.tolist()


def load_accepted_data(experiments_dir: str, n_files: int = -1, 
                       far_probe_avrg_tol: int = 100) -> tuple[list[Measurement], dict[str, float]]:
    """
    Load data from files marked as accepted via file-screening. 
    The directory needs to contain .ibw-files and a `screened_files.csv`.

    Parameters
    ----------
    experiments_dir : str
        Directory containing .ibw-files and a `screened_files.csv`.
    n_files : int, optional
        Number of files to load. -1 to load all 'accepted' files, by default -1
    far_probe_avrg_tol : int, optional
        The number of measuring steps at the beginning/end of the approach/retract curve 
        over which the measurements are averaged for the parameters of the probe at maximum distance, by default 100

    Returns
    -------
    tuple[list[Measurement], dict[str, float]]
        A list of Measurements, each containing the data of a file, and 
        a dictionary containing calibration parameters.
    """
    filepaths = load_accepted_filepaths(Path(experiments_dir))
    return data_loading.load_data(files=filepaths, n_files=n_files, far_probe_avrg_tol=far_probe_avrg_tol)


def write_to_df(df: pd.DataFrame, filepath: Path, accepted: bool) -> pd.DataFrame:
    if filepath in df.filepath.values:
        df.loc[df.filepath == filepath, 'accept'] = accepted
    else:
        df.loc[len(df)] = [filepath, accepted]


def save_screening_results(df: pd.DataFrame, dir: Path):
    data_loading.backup_existing(dir, SAVE_NAME)
    df = df.copy()
    df['filepath'] = df['filepath'].apply(lambda path: path.name)
    df.to_csv(dir / SAVE_NAME, index=False)


def sort_curves_by_distance(measurements: list[Measurement], ideal_curve_idx: int,
                            curve_type: Literal['z_in', 'z_out', 'phase_in', 'phase_out', 'amp_in', 'amp_out'],
                            ) -> tuple[list[Measurement], np.ndarray, np.ndarray]:
    length = min([len(m[curve_type]) for m in measurements])
    ideal_curve = measurements[ideal_curve_idx][curve_type][:length]
    d_norms = np.array([np.linalg.norm(ideal_curve - m[curve_type][:length]) for m in measurements])
    sort_idcs = np.argsort(d_norms)
    with open('sorted_file_indices.pkl', 'wb') as f:
        pickle.dump(sort_idcs, f)
    return [measurements[i] for i in sort_idcs], d_norms[sort_idcs], sort_idcs


class Index(object):
    COLORS = [['tab:blue', 'tab:red'], ['tab:green', 'tab:pink']]

    def __init__(self, data_dir: str, axs, target: int = 100, n_files: int = -1, revise: bool = False,
                 ):
        super().__init__()
        self.axs = axs
        self.dir = Path(data_dir).resolve()
        self.target = target
        self.revise = revise
        self.n_accepted = 0

        # load previously stored scrrening results:
        files = data_loading.get_ibw_paths(self.dir, n_files)
        self.df = load_screening_results(self.dir)
        if not self.df.empty:
            if not revise:
                self.n_accepted = len(self.df[self.df['accept']])
                files = list(set(files) - set(self.df.filepath))
                if self.n_accepted >= self.target and not revise:
                    print(f'> {self.n_accepted} accepted measurements already found in `{SAVE_NAME}`, closing the application.'
                        'Set the -t argument accordingly if you want to select more curves.')
                    return
        measurements, calib_params = data_loading.load_data(files=files)
        self.measurements, _ = preprocessing.preprocess(measurements, calib_params)
        self.n_screen = len(self.measurements)
        print(f'> Loaded {self.n_screen} measurement(s) to screen.')
        print(f'> Additionally, {len(self.df)} already screened results with {self.n_accepted} accepted measurements have been found on disk.')

    def update_view(self, setup: bool = False):
        if self.n_accepted >= self.target or len(self.df) >= self.n_screen:
            plt.close()
        else:
            measurement = self.measurements[0]
            self.filepath = measurement.file_path
            self.measurements.remove(measurement)
            n_screened, n_screen_remaining = self._scale_prog(len(self.df), self.n_screen)
            n_accepted, n_accept_remaining = self._scale_prog(self.n_accepted, self.target)
            
            # define texts
            text = (f'files screened: {'|' * n_screened}{'_' * (n_screen_remaining)} '
                    f'{len(self.df) / self.n_screen * 100:.0f}% '
                    f'({len(self.df)}/{self.n_screen})\n'
                    f'files accepted: {'|' * n_accepted}{'_' * (n_accept_remaining)} '
                    f'{self.n_accepted / self.target * 100:.0f}% '
                    f'({self.n_accepted}/{self.target})')
            if self.revise:
                status = self.df[self.df['filepath'] == self.filepath]
                if status.empty:
                    state = 'UNSCREENED'
                    color = 'k'
                else:
                    status = status.iloc[0]
                    state = 'ACCEPTED' if status['accept'] else 'REJECTED'
                    color = 'g' if status['accept'] else 'r'
                text_revise = f'current file status: {state}'
            
            # plot curves
            for i, direction in enumerate(['in', 'out']):
                for j, signal_type in enumerate(['amp', 'phase']):
                    if not setup:
                        self.axs[i][j].clear()
                    self.axs[i][j].plot(measurement[f'z_{direction}'], 
                                        measurement[f'{signal_type}_{direction}'], 
                                        color=self.COLORS[i][j], label=f'{signal_type}_{direction}')
                    self.axs[i][j].legend(loc='lower right')
            
            # set progress texts
            if setup:
                self.txt_ax1 = plt.axes([0.1, 0.05, 0.5, 0.075])
                self.txt_ax1.axis('off')
                self.txt_ax1.text(0.0, 0.5, text, fontsize=8, horizontalalignment='left', 
                                 verticalalignment='center')
                if self.revise:
                    self.txt_ax2 = plt.axes([0.5, 0.05, 0.5, 0.075])
                    self.txt_ax2.axis('off')
                    self.txt_ax2.text(0.0, 0.5, text_revise, fontsize=8, horizontalalignment='left', 
                                      verticalalignment='center')
                    self.txt_ax2.texts[0].set_color(color)
            else:
                self.txt_ax1.texts[0].set_text(text)
                if self.revise:
                    self.txt_ax2.texts[0].set_text(text_revise)
                    self.txt_ax2.texts[0].set_color(color)
            plt.draw()
    
    def _scale_prog(self, current: int, total: int):
        lmax = 50
        factor = lmax / total
        n_done = int(current * factor)
        n_remaining = lmax - n_done
        return n_done, n_remaining
    
    def store_result(self, accepted: bool):
        write_to_df(self.df, self.filepath, accepted)

    def next(self, event):
        """ accept """
        self.store_result(True)
        self.n_accepted += 1
        self.update_view()
        
    def prev(self, event):
        """ reject """
        self.store_result(False)
        self.update_view()
    
    def on_close(self, event):
        print(f'> Saved screening results to {self.dir / SAVE_NAME}')
        save_screening_results(self.df, self.dir)


def gui_select_experiments(data_dir: str, target: int = 100, n_files: int = -1, revise: bool = False):
    plt.rcParams['font.family'] = 'monospace'

    fig, axs = plt.subplots(2, 2, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.2)

    callback = Index(data_dir, axs, target, n_files, revise)
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Accept')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Reject')
    bprev.on_clicked(callback.prev)
    fig.canvas.mpl_connect('close_event', callback.on_close)
    callback.update_view(setup=True)
    plt.show()
