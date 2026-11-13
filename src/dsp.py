"""
dsp.py - the DSP toolbox for the CSI pipeline.

Each function takes an array of shape (N, K): N time samples (rows) and
K subcarriers (columns), and works along time, one subcarrier at a time.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

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


def butter_sos(cutoff_hz, fs=100.0, order=4):
    """Design a Butterworth low-pass filter (second-order sections)."""
    return signal.butter(order, cutoff_hz, btype="low", fs=fs, output="sos")


def butter_lowpass(x, cutoff_hz, fs=100.0, order=4):
    """Zero-phase Butterworth low-pass filter along time.

    sosfiltfilt runs the filter forward and then backward, so the output has
    no time delay (zero phase). This is fine for offline analysis; the
    effective magnitude response is |H(f)|^2 (twice the attenuation in dB).

    x : (N, K) real array.  Returns an array of the same shape.
    """
    return signal.sosfiltfilt(butter_sos(cutoff_hz, fs, order), x, axis=0)



def remove_static(x):
    """Subtract each subcarrier's own mean over time.

    The static room (walls, furniture) gives every subcarrier a fixed
    amplitude. Subtracting that fixed value leaves only the time-varying
    part, which is what human movement causes.

    x : (N, K) real array.  Returns an array of the same shape.
    """
    return x - x.mean(axis=0, keepdims=True)


def stft_power(x, fs=100.0, nperseg=128, hop=16, nfft=256):
    """Short-time Fourier transform (STFT) power, averaged over subcarriers.

    The signal is cut into overlapping frames of `nperseg` samples that start
    `hop` samples apart. Each frame gets its own mean removed (this removes the
    static room pattern locally), is multiplied by a Hann window and Fourier
    transformed. Power |Z|^2 is then averaged over all subcarriers, which
    reduces noise.

    x : (N, K) real array.
    Returns f (F,) in Hz, t (T,) in seconds, P (F, T).
    """
    f, t, Z = signal.stft(x, fs=fs, window="hann", nperseg=nperseg,
                          noverlap=nperseg - hop, nfft=nfft,
                          detrend="constant", boundary=None, padded=False,
                          axis=0)
    P = np.mean(np.abs(Z) ** 2, axis=1)       # Z is (F, K, T) -> average over K
    return f, t, P


def sanitize_phase(H, subcarrier_idx):
    """Remove hardware phase errors from complex CSI (linear phase fit).

    The receiver's clock is not locked to the transmitter's, so every packet's
    phase has an unknown constant offset and an unknown straight-line slope
    across the subcarriers. Both change randomly from packet to packet and
    hide the real channel phase. For each packet we:
      1. unwrap the phase along the subcarriers,
      2. fit a straight line  phase = slope * k + offset  (least squares),
      3. subtract that line.
    What remains is the phase variation that comes from the channel itself.

    H            : (N, K) complex CSI of the K chosen subcarriers.
    subcarrier_idx : the K raw subcarrier numbers (the x-axis of the fit).
    Returns (residual (N, K) float32 in radians, slope (N,), offset (N,)).
    """
    k = np.asarray(subcarrier_idx, dtype=np.float64)
    ph = np.unwrap(np.angle(H).astype(np.float64), axis=1)
    kc = k - k.mean()
    slope = (ph * kc).sum(axis=1) / (kc ** 2).sum()
    offset = ph.mean(axis=1) - slope * k.mean()
    resid = ph - (slope[:, None] * k[None, :] + offset[:, None])
    return resid.astype(np.float32), slope, offset


def window_features(x, fs=100.0, win_s=4.0, split_hz=4.0):
    """Two simple features for every non-overlapping window of x.

    1. power: mean of x^2 after removing each subcarrier's mean
    2. share: fraction of the (non-DC) spectral power above split_hz

    x : (N, K) real array.  Returns (power (W,), share (W,)).
    """
    n = int(win_s * fs)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    taper = np.hanning(n)[:, None]
    power, share = [], []
    for s in range(0, len(x) - n + 1, n):
        w = remove_static(x[s:s + n])
        power.append(np.mean(w ** 2))
        P = np.mean(np.abs(np.fft.rfft(w * taper, axis=0)) ** 2, axis=1)
        share.append(P[freqs > split_hz].sum() / P[1:].sum())
    return np.array(power), np.array(share)



def hampel_chunked(x, window=15, n_sigmas=4.0, chunk=5000):
    """Same result as hampel(), but processes the rows in blocks.

    A long recording (60,000 packets x 52 subcarriers) would otherwise need
    several hundred MB of memory inside hampel(). Each block is given
    window//2 extra rows on both sides, so every sample still sees its full
    window and the result is identical.

    x : (N, K) real array.  Returns only the filtered array (no mask).
    """
    half = window // 2
    out = np.empty_like(x)
    for s in range(0, len(x), chunk):
        e = min(s + chunk, len(x))
        a, b = max(0, s - half), min(len(x), e + half)
        y, _ = hampel(x[a:b], window, n_sigmas)
        out[s:e] = y[s - a:s - a + (e - s)]
    return out