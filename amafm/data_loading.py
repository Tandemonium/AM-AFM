import glob
import re
import numpy as np

from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from amafm.igor import binarywave as ibw
from . import calibration


_surrogates = re.compile(r"[\uDC80-\uDCFF]")


@dataclass
class Measurement:
    z_in: np.ndarray
    z_out: np.ndarray
    phase_in: np.ndarray
    phase_out: np.ndarray
    amp_in: np.ndarray
    amp_out: np.ndarray
    file_path: Path|None = None

    @classmethod
    def signal_types(cls) -> list[str]:
        z_types = cls.z_types() + ['file_path']
        return [name for name in cls.__match_args__ if name not in z_types]
    
    @classmethod
    def z_types(cls) -> list[str]:
        return [name for name in cls.__match_args__ if name.startswith('z')]
    
    def __getitem__(self, item: str) -> np.ndarray:
        return getattr(self, item)
    
    def __setitem__(self, key: str, value: np.ndarray) -> None:
        setattr(self, key, value)
    
    def copy(self) -> 'Measurement':
        return Measurement(**self.__dict__)
    
    def deepcopy(self) -> 'Measurement':
        return Measurement(self.z_in.copy(), self.z_out.copy(), self.phase_in.copy(), self.phase_out.copy(),
                           self.amp_in.copy(), self.amp_out.copy(), self.file_path)
    
    def __eq__(self, value):
        if isinstance(value, Measurement):
            return self.file_path == value.file_path
        return False


def backup_existing(directory: str, filename: str):
    # rename file if already exists:
    dir = Path(directory)
    filepath = dir / filename
    if filepath.exists():
        files = glob.glob(f'{dir.as_posix()}/{filepath.stem}_*{filepath.suffix}')
        file_numbers = [f.split(filepath.stem)[-1].split('.')[0].split('_')[-1] for f in files]
        file_numbers = sorted([int(n) for n in file_numbers if n.isdigit()])
        new_number = file_numbers[-1] + 1 if file_numbers else 0
        new_fp = dir / f'{filepath.stem}_{new_number}{filepath.suffix}'
        filepath.rename(new_fp)


def get_ibw_paths(directory: Path, n_files: int = -1) -> list[Path]:
    count = 0
    files = []
    for f in directory.iterdir():
        if n_files >= 1 and count >= n_files:
            break
        if f.is_file() and f.suffix == '.ibw':
            files.append(f)
            count += 1
    return files


def detect_decoding_errors_line(line, _s=_surrogates.finditer):
    """Return decoding errors in a line of text
    Works with text lines decoded with the surrogateescape
    error handler.     Returns a list of (pos, byte) tuples
    Readout of additional data not saved in traditional ibw style, but as plain text
    """
    # DC80 - DCFF encode bad bytes 80-FF
    return [(m.start(), bytes([ord(m.group()) - 0xDC00]))
            for m in _s(line)]


def load_ibw_force(file: Path) -> tuple[dict[str, str], list[str], np.ndarray, str]:
    constants = []
    data = ibw.load(file)
    with open(file, encoding="utf8", errors="surrogateescape") as f:
        for line in f:
            if not detect_decoding_errors_line(line):
                constants.append(line)
    constants = [x for x in constants if ':' in x]
    constants = {x.split(':')[0]: (str(x.split(':')[1])).strip() for x in constants}

    ##################### DATA IGOR BINARY WAVE ###############
    # GET THE DATA packed as Igor binary wave
    # Data and its labels read
    wave_data = data['wave']['wData']
    labels = data['wave']['labels'][1]
    labels = [x.decode('utf-8') for x in labels][1:]
    name = str(data['wave']['wave_header']['bname'])

    return constants, labels, wave_data, name


def separate_signal(signal_array: np.ndarray, turning_point: int) -> tuple[np.ndarray, np.ndarray]:
    curve_in = signal_array[:turning_point]
    curve_in = np.flip(curve_in)
    curve_out = signal_array[turning_point:]
    return curve_in, curve_out


def separate_drive(drive: np.ndarray, turning_point: int) -> tuple[np.ndarray, np.ndarray]:
    z_in = drive[:turning_point]
    z_out = drive[turning_point:]
    z_out = np.flip(z_out)
    return z_in, z_out


def matz_Uhlig(labels: list[str], wave_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    drive = wave_data[:, labels.index('Drive')]  # m
    drive = (drive - np.min(drive))              # m, relative 0-point
    amp = wave_data[:, labels.index('Amp')]      # observable
    phase = wave_data[:, labels.index('Phase')]  # observable
    turning_point = np.argmax(drive)
    return drive, amp, phase, turning_point


def retrieve_signals(file: Path) -> Measurement:
    constants, labels, wave_data, name = load_ibw_force(file)
    if np.isnan(wave_data).any():
        raise ValueError('NaN values in wave data.')
    drive, amp, phase, turning_point = matz_Uhlig(labels, wave_data)

    # separate curves into approach and retract curves
    z_in, z_out = separate_drive(drive, turning_point)
    phase_in, phase_out = separate_signal(phase, turning_point)
    amp_in, amp_out = separate_signal(amp, turning_point)
    return Measurement(z_in, z_out, phase_in, phase_out, amp_in, amp_out, file)


def load_data(data_dir: str|Path|None = None, files: list[str|Path]|None = None, n_files: int = -1, 
              far_probe_avrg_tol: int = 100) -> tuple[list[Measurement], dict[str, float]]:
    """
    Load data from .ibw files from the given directory and return measurements and calibration parameters.

    Parameters
    ----------
    data_dir : str | Path | None, optional
        Directory containing the `.ibw` files of an experiment. 
        Not required, if instead a list of filepaths is given, by default None
    files : list[str | Path] | None, optional
        A list of filepaths of the .ibw files to load.
        Not required if a directory is given, by default None
    n_files : int, optional
        Number of files to load. `-1` to load all files , by default -1
    far_probe_avrg_tol : int, optional
        The number of measuring steps at the beginning/end of the approach/retract curve 
        over which the measurements are averaged for the parameters of the probe at maximum distance, by default 100

    Returns
    -------
    tuple[list[Measurement], dict[str, float]]
        A list of Measurements, each containing the data of a file, and 
        a dictionary containing calibration parameters.
    """
    assert data_dir or files, "Either data_dir or files must be provided."
    if files is None:
        data_dir: Path = Path(data_dir).resolve()
        files = get_ibw_paths(data_dir, n_files)
    else:
        files = [Path(f).resolve() for f in files]
        data_dir: Path = files[0].parent
    calib_params = calibration.get_calibration_parameters(data_dir.with_name(data_dir.name + '_calib'), 
                                                          far_probe_avrg_tol=far_probe_avrg_tol)

    # retrieve separate signals from files
    measurements: list[Measurement] = []
    for file in tqdm(files, desc=f'Loading data from .ibw-files'):
        try:
            m = retrieve_signals(file)
            measurements.append(m)
        except ValueError:
            print(f"   Error in file '{file.as_posix()}'. Skipping file.")
            continue
    return measurements, calib_params
