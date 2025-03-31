import pickle

from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Button

from . import preprocessing
from .preprocessing import Measurement


SELECTED_IDCS_NAME = 'selected_file_indices'


def load_selected_idcs() -> list[int]:
    with open(SELECTED_IDCS_NAME, 'rb') as f:
        good_idcs = pickle.load(f)
    return good_idcs


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


def load_preprocess(data_dir: str, start_at: int = 0, num_files: int = -1, 
                    files: list[str]|None = None, folders: list[str]|None = None) -> tuple[list[Measurement], dict[str, float]]:
    return preprocessing.preprocess(data_dir, start_at=start_at, num_files=num_files, files=files, folders=folders)


class Index(object):
    colors = [['tab:blue', 'tab:red'], ['tab:green', 'tab:pink']]

    def __init__(self, data_dir: str, axs, start_at: int = 0, num_files: int = -1, folders: list[str]|None = None):
        super().__init__()
        self.idx = -1
        self.axs = axs
        self.start_at = start_at
        self.chosen = []
        self.measurements: list[Measurement] = load_preprocess(data_dir, start_at, num_files, folders)[0]
        self.f_count = len(self.measurements)
        print(f'Loaded {self.f_count} measurement(s).')

    def update_view(self):
        self.idx += 1
        if self.idx >= self.f_count:
            plt.close()
        else:
            measurement = self.measurements[0]
            self.measurements.remove(measurement)
            n_done, n_remaining = self._scale_prog()
            text = (f'progress: {'|' * n_done}{'_' * (n_remaining)} {self.idx / self.f_count * 100:.0f}% '
                    f'({self.idx}/{self.f_count})\n'
                    f'chosen files: {len(self.chosen)}')
            for i, direction in enumerate(['in', 'out']):
                for j, signal_type in enumerate(['amp', 'phase']):
                    if self.idx > 0:
                        self.axs[i][j].clear()
                    self.axs[i][j].plot(measurement[f'z_{direction}'], 
                                        measurement[f'{signal_type}_{direction}'], 
                                        color=self.colors[i][j], label=f'{signal_type}_{direction}')
                    self.axs[i][j].legend(loc='lower right')
            if self.idx == 0:
                self.txt_ax = plt.axes([0.1, 0.05, 0.5, 0.075])
                self.txt_ax.axis('off')
                self.txt_ax.text(0.0, 0.5, text, fontsize=8, horizontalalignment='left', 
                                 verticalalignment='center')
            else:
                self.txt_ax.texts[0].set_text(text)
            plt.draw()
    
    def _scale_prog(self):
        lmax = 50
        factor = lmax / self.f_count
        n_done = int(self.idx * factor)
        n_remaining = lmax - n_done
        return n_done, n_remaining

    def next(self, event):
        """ take """
        self.chosen.append(self.idx)
        self.update_view()
        
    def prev(self, event):
        """ reject """
        self.update_view()
    
    def on_close(self, event):
        print('chosen file-indices:', self.chosen)
        with open(f'{SELECTED_IDCS_NAME}_{self.start_at}-{self.start_at + self.idx - 1}.pkl', 'wb') as f:
            pickle.dump(self.chosen, f)


def gui_select_experiments(data_dir: str, start_at: int = 0, num_files: int = -1, folders: list[str]|None = None):
    plt.rcParams['font.family'] = 'monospace'

    fig, axs = plt.subplots(2, 2, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.2)

    callback = Index(data_dir, axs, start_at, num_files, folders)
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Take')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Reject')
    bprev.on_clicked(callback.prev)
    fig.canvas.mpl_connect('close_event', callback.on_close)
    callback.update_view()
    plt.show()
