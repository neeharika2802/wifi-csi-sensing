"""
step17_features.py - Step 17a: feature blocks for walking vs walking + arm-waving.

Why: with the subcarrier-averaged spectrogram the two moving classes stay at about 60 %
(chance 50 %). This module computes other descriptions of the same 4-8 s windows:

  base   mean / std / max over time of every spectrogram row (0-15 Hz), amplitude and phase,
         from the subcarrier-averaged power (the Step 9 representation)
  pca    the same statistics for the spectrograms of the first 3 principal components across
         the 52 subcarriers of the window (coherent combination of the subcarriers instead of
         averaging their powers)
  eig    how the movement variance spreads over the principal components: the 5 largest
         eigenvalue shares and the total variance (an extra moving body part = more
         independent components)
  cad    cadence of the first principal component: autocorrelation at lags 0.1 s ... 2 s, spectral
         centroid, bandwidth, entropy, flatness, dominant frequency and the share of power in
         4 frequency bands

All blocks use the same cleaned signal as the rest of the project (Hampel 15/4 -> uniform
100 Hz grid -> 12 Hz low-pass; phase sanitized).
"""
import numpy as np
from scipy import signal

FS = 100.0
NPERSEG, HOP, NFFT = 128, 8, 256
FMAX = 15.0
BANDS = [(0.3, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 12.0)]


def _db_rows(power, freqs):
    return 10 * np.log10(power[freqs <= FMAX] + 1e-10)


def _stats(db):
    """mean / std / max over time of every row of a (rows, time) spectrogram."""
    return np.concatenate([db.mean(axis=1), db.std(axis=1), db.max(axis=1)])


def _spectrogram_power(x):
    """x (n, K) -> (freqs, power (rows, K, time))."""
    f, _, Z = signal.stft(x, fs=FS, window="hann", nperseg=NPERSEG, noverlap=NPERSEG - HOP,
                          nfft=NFFT, detrend="constant", boundary=None, padded=False, axis=0)
    return f, np.abs(Z) ** 2


def _cadence(pc):
    """Cadence features of one time series (the first principal component)."""
    n = len(pc)
    pc = pc - pc.mean()
    ac = np.correlate(pc, pc, mode="full")[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    lags = np.arange(10, min(200, n // 2) + 1, 10)
    f = np.fft.rfftfreq(1024, 1 / FS)
    P = np.abs(np.fft.rfft(pc * np.hanning(n), 1024)) ** 2
    use = (f >= 0.3) & (f <= 15)
    f, P = f[use], P[use]
    p = P / (P.sum() + 1e-12)
    centroid = (f * p).sum()
    bandwidth = np.sqrt(((f - centroid) ** 2 * p).sum())
    entropy = -(p * np.log(p + 1e-12)).sum() / np.log(len(p))
    flatness = np.exp(np.mean(np.log(P + 1e-12))) / (P.mean() + 1e-12)
    dominant = f[np.argmax(P)]
    shares = [p[(f >= lo) & (f < hi)].sum() for lo, hi in BANDS]
    return np.concatenate([ac[lags], [centroid, bandwidth, entropy, flatness, dominant], shares])


def window_blocks(A, P, win_s, start, stop, stride_s=1.0):
    """Feature blocks of every window of win_s seconds in samples start..stop of A and P (M, 52)."""
    n, step = int(win_s * FS), int(stride_s * FS)
    out = {k: [] for k in ("base", "pca", "eig", "cad")}
    for s in range(start, stop - n + 1, step):
        base, pca, eig, cad = [], [], [], []
        for x in (A[s:s + n], P[s:s + n]):
            f, power = _spectrogram_power(x)
            base.append(_stats(_db_rows(power.mean(axis=1), f)))
            x0 = x - x.mean(axis=0, keepdims=True)
            U, S, _ = np.linalg.svd(x0, full_matrices=False)
            pcs = U[:, :3] * S[:3]
            _, ppower = _spectrogram_power(pcs)
            pca.append(np.concatenate([_stats(_db_rows(ppower[:, k], f)) for k in range(3)]))
            share = S ** 2 / (S ** 2).sum()
            eig.append(np.concatenate([share[:5], [np.log((S ** 2).sum() / n + 1e-12)]]))
            cad.append(_cadence(pcs[:, 0]))
        for k, v in zip(("base", "pca", "eig", "cad"), (base, pca, eig, cad)):
            out[k].append(np.concatenate(v))
    return {k: np.array(v, dtype=np.float32) for k, v in out.items()}