import numpy as np


def template(signal: np.ndarray, **kwargs) -> np.ndarray:
    return signal


def rolling_window(signal: np.ndarray, agg: str = 'mean', w: int = 1000, 
                   s: int = 10) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    d = pd.Series(signal)
    d = (d.rolling(w, step=s).agg(agg)
          .reindex(range(len(signal)))
          .interpolate('linear', limit_direction='both'))
    return d.values


def savgol(sig: np.ndarray, w: int = 1000, p: int = 3) -> np.ndarray:
    from scipy import signal
    if len(sig) < w:
        w = len(sig) - 1
    return signal.savgol_filter(sig, w, p)


def gauss(signal: np.ndarray, s: int = 100) -> np.ndarray:
    from scipy import ndimage
    return ndimage.gaussian_filter1d(signal, s)


def reduce_curve(signal_x: np.ndarray, signal_y: np.ndarray, target_length: int) -> tuple[np.ndarray, np.ndarray]:
    assert target_length <= len(signal_x), \
        f'Target length {target_length} is larger than signal length {len(signal_x)}'
    
    from scipy.interpolate import CubicSpline
    spline = CubicSpline(signal_x, signal_y)
    newx = np.linspace(signal_x.min(), signal_y.max(), target_length)
    return newx, spline(newx)


def lowess(signal: np.ndarray, frac: float = 0.05) -> np.ndarray:
    import statsmodels.api as sm
    y = sm.nonparametric.lowess(exog=np.arange(len(signal)), endog=signal, frac=frac)
    return y[:, 1]


def fft(signal: np.ndarray, threshold: int = 0.8) -> np.ndarray:
    """
    threshold: float, between 0 and 1
    """
    y = np.fft.fft(signal)
    psd = y * np.conj(y) / len(signal)  # Power spectral density
    psd = (psd - psd.min()) / (psd.max() - psd.min())
    indices = psd > threshold
    y = indices * y
    y_clean = np.fft.ifft(y).real
    return y_clean
