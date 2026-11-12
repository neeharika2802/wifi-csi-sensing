"""
dsp.py - the DSP toolbox for the CSI pipeline.

Each function takes an array of shape (N, K): N time samples (rows) and
K subcarriers (columns), and works along time, one subcarrier at a time.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from csi_io import packet_intervals_s


def hampel(x, window=7, n_sigmas=3.0):
    """Hampel filter: replace isolated outlier samples by the local median.

    For every sample we look at a sliding window of `window` samples centred
    on it, compute the window median and the MAD (median absolute deviation).
    If the sample is further than n_sigmas * 1.4826 * MAD from the median,
    it is declared an outlier and replaced by the median.

    x : (N, K) real array.  Returns (filtered copy, boolean mask of replaced samples).
    """
    if window % 2 == 0:
        raise ValueError("window must be odd")
    half = window // 2
    padded = np.pad(x, ((half, half), (0, 0)), mode="reflect")
    win = sliding_window_view(padded, window, axis=0)     # (N, K, window)
    med = np.median(win, axis=-1)                          # (N, K)
    mad = np.median(np.abs(win - med[..., None]), axis=-1)
    threshold = n_sigmas * 1.4826 * mad
    outlier = np.abs(x - med) > threshold
    return np.where(outlier, med, x), outlier


def resample_uniform(timestamp_us, x, fs=100.0):
    """Put irregularly timed packets onto an exactly uniform time grid.

    Uses the ESP32 packet timestamps and linear interpolation, so lost or
    late packets no longer distort the spectrum later (the STFT assumes
    equally spaced samples).

    x : (N, K) real array.  Returns (t_new in seconds, x_new of shape (M, K)).
    """
    t = np.concatenate([[0.0], np.cumsum(packet_intervals_s(timestamp_us))])
    keep = np.concatenate([[True], np.diff(t) > 0])       # drop duplicate timestamps
    t, x = t[keep], x[keep]
    t_new = np.arange(0.0, t[-1], 1.0 / fs)
    x_new = np.empty((t_new.size, x.shape[1]), dtype=x.dtype)
    for k in range(x.shape[1]):
        x_new[:, k] = np.interp(t_new, t, x[:, k])
    return t_new, x_new