import os

import numpy as np

from typing import Literal

from . import data_loading
from .preprocessing import Measurement


# --------------- Hölscher model --------------- #
def force_hoelscher(measurement: Measurement, calib_params: dict[str, float], 
                    direction: Literal['in', 'out'] = 'out', n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Create force curve from measurement according the Hölscher algorithm.

    Parameters
    ----------
    measurement : Measurement
        A Measurement-object for which to compute the force curve.
    calib_params : dict[str, float]
        Calibration parameters of the Measurement-object:
    direction : Literal[&#39;in&#39;, &#39;out&#39;], optional
        Direction of the cantilever: 'in' for approaching, 'out' for retracting, by default 'out'
    n : int, optional
        Steps to average to avoid 0s, by default 10

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of separation (z-distance) and force arrays.
    """
    separation = measurement[f'z_{direction}']   # m
    phase = measurement[f'phase_{direction}']    # deg
    amplitude = measurement[f'amp_{direction}']  # m

    lb = separation - amplitude    # m
    # ub = separation + amplitude  # m

    integrand = ((calib_params['kc'] * calib_params['amp_far'] * np.cos(phase * (np.pi / 180))) 
                 / (calib_params['Qfact'] * np.sqrt(2)))
    integral = integrand * 2 * amplitude
    hl = (lb[1:n+1] - lb[:n]).mean()    # multiple step sizes are averaged here to avoid 0s
    force = -np.gradient(integral, hl)  # -> 1e-12 - 1e-10
    return separation, force


# --------------- Payman/Garcia model --------------- #
def force_payman(measurement: Measurement, calib_params: dict[str, float], 
                 direction: Literal['in', 'out'] = 'out', n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    separation = measurement[f'z_{direction}']
    phase = measurement[f'phase_{direction}']
    amplitude = measurement[f'amp_{direction}']

    lb = separation - amplitude  # m
    cos_phase = np.cos(phase * (np.pi / 180))
    b = 2 * amplitude
    integrand1 = calib_params['amp_far'] / (2 * calib_params['Qfact'] * amplitude) * cos_phase
    integrand2 = calib_params['amp_far'] / (2 * calib_params['Qfact'] * np.sqrt(2)) * cos_phase
    integral1 = integrand1 * b
    integral2 = integrand2 * b

    hl = (lb[1:n+1] - lb[:n]).mean()  # multiple step sizes are averaged here to avoid 0s
    payterm2 = -np.gradient(integral2, hl)
    forceterm1 = 2 * calib_params['kc'] * integral1
    forceterm2 = 2 * calib_params['kc'] * payterm2
    force = forceterm1 + forceterm2  # N
    return separation, force


SAVE_NAME = 'force_curve.npz'
def save(separation: np.ndarray, force: np.ndarray, dir: str) -> None:
    """
    Save separation and force, received from the `force_hoelscher` function as `force_curve.npz` in the given directory.

    Parameters
    ----------
    separation : np.ndarray
        Separation (z-distance) array.
    force : np.ndarray
        Force array.
    dir : str
        Save location.
    """
    data_loading.backup_existing(dir, SAVE_NAME)
    np.savez(os.path.join(dir, SAVE_NAME), separation=separation, force=force)


def load(dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load separation and force saved in the file `force_curve.npz` in the given directory.

    Parameters
    ----------
    dir : str
        Directory containing the file `force_curve.npz`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Separation and force array.

    Raises
    ------
    FileNotFoundError
        If the file `force_curve.npz` does not exist in the given directory.
    """
    data = np.load(os.path.join(dir, SAVE_NAME))
    return data['separation'], data['force']
