from typing import Literal

import numpy as np

from .preprocessing import Measurement


# --------------- Hölscher model --------------- #
def force_hoelscher(measurement: Measurement, calib_params: dict[str, float], 
                    direction: Literal['in', 'out'] = 'out', n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    separation = measurement[f'z_{direction}']
    phase = measurement[f'phase_{direction}']
    amplitude = measurement[f'amp_{direction}']

    lb = separation - amplitude  # m
    #ub = separation + amplitude  # m

    integrand = ((calib_params['kc'] * calib_params['amp_far'] * np.cos(phase * (np.pi / 180))) 
                 / (calib_params['Qfact'] * np.sqrt(2)))
    integral = integrand * 2 * amplitude
    hl = (lb[1:n+1] - lb[:n]).mean()  # multiple step sizes are averaged here to avoid 0s
    force = -np.gradient(integral, hl)
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
