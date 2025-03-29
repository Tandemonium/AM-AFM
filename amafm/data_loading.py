import re
import numpy as np

from pathlib import Path
from typing import IO

from amafm.igor import binarywave as ibw


_surrogates = re.compile(r"[\uDC80-\uDCFF]")


def get_ibw_paths(data_dir: str, calib_only: bool = False, n: int = -1, 
                  folders: list[str]|None = None) -> dict[str, list[Path]]|list[Path]:
    paths: dict[str, list[Path]] = {'data': [], 'calib': []}
    p = Path(data_dir)
    d_counter = 0
    c_counter = 0
    for cp in p.iterdir():
        if cp.is_dir():
            is_calib = cp.name.endswith('_calib')
            experiment_name = cp.name.split('_calib')[0] if is_calib else cp.name
            select = not ((calib_only and not is_calib) or 
                          (folders is not None and experiment_name not in folders))
            if select:
                key = 'calib' if is_calib else 'data'
                for ccp in cp.iterdir():
                    if n >= 1 and ((is_calib and c_counter > n) or (not is_calib and d_counter > n)):
                        continue
                    paths[key].append(ccp)
                    if is_calib:
                        c_counter += 1
                    else:
                        d_counter += 1
                    if n >= 1 and d_counter >= n and c_counter >= n:
                        return paths['calib'] if calib_only else paths
    return paths['calib'] if calib_only else paths


def detect_decoding_errors_line(line, _s=_surrogates.finditer):
    """Return decoding errors in a line of text
    Works with text lines decoded with the surrogateescape
    error handler.     Returns a list of (pos, byte) tuples
    Readout of additional data not saved in traditional ibw style, but as plain text
    """
    # DC80 - DCFF encode bad bytes 80-FF
    return [(m.start(), bytes([ord(m.group()) - 0xDC00]))
            for m in _s(line)]


def load_ibw_force(file: str | IO) -> tuple[dict[str, str], list[str], np.ndarray, str]:
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
