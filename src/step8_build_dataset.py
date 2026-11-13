"""
step8_build_dataset.py - Step 8: turn the recordings into a training set.

For every recording we run the full pipeline
    Hampel -> uniform 100 Hz grid -> Butterworth low-pass
on BOTH the amplitude and the sanitized phase, cut the result into 4-second
windows (one new window every second) and compute an STFT spectrogram of each
window. Every window becomes an image with 2 channels (amplitude, phase).

Each window also gets a GROUP number (1-5). Group = recording number
(w3.csv and ww3.csv are group 3). The empty room has only one long recording
(b1.csv), so it is cut into 5 consecutive time blocks with a gap in between.
Later we train on some groups and test on a different group, so test windows
never come from a recording the model has seen.

The first run parses 11 big CSV files and takes a minute or two. Afterwards the
parsed data is cached in data/processed/cache, so re-runs (for example with a
different CUTOFF) are much faster.
"""
import time
from pathlib import Path

import numpy as np

from csi_io import load_csi, LLTF_VALID
from dsp import (hampel_chunked, resample_uniform, butter_lowpass,
                  sanitize_phase, stft_power)

ROOT = Path(__file__).resolve().parent.parent

# ---- settings you can play with -------------------------------------------
SCENARIO = "LOS"        # LOS or NLOS (check the folder name in data/raw)
SYSTEM = "PIFA"         # PIFA or BQ  (the two antenna systems)
FS = 100.0              # sampling rate after resampling (Hz)
CUTOFF = 12.0           # low-pass cutoff in Hz; None = no low-pass
ORDER = 4               # Butterworth order
WIN_S = 4.0             # window length (seconds)
STRIDE_S = 1.0          # a new window starts every STRIDE_S seconds
NPERSEG, HOP, NFFT = 128, 8, 256   # STFT settings (frames of 1.28 s, step 0.08 s)
FMAX_KEEP = 15.0        # keep spectrogram rows up to this frequency (Hz)
N_GROUPS = 5            # w1..w5, ww1..ww5 and 5 time blocks of b1
GUARD_S = WIN_S / 2     # seconds dropped on each side of a block boundary in b1
# ---------------------------------------------------------------------------

DATA = ROOT / "data" / "raw" / "wallhack1.8k" / SCENARIO / SYSTEM
OUT = ROOT / "data" / "processed"
CLASS_NAMES = ["Empty room", "Walking", "Walking + arm-waving"]


def load_cached(path):
    """Parse a CSV once; later runs reuse the saved complex CSI."""
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{SCENARIO}_{SYSTEM}_{path.stem}.npz"
    if f.exists():
        z = np.load(f)
        return z["H"], z["ts"], int(z["label"])
    d = load_csi(path)
    labels = np.unique(d["label"])
    assert len(labels) == 1, f"{path.name}: expected one class label, got {labels}"
    H = d["csi"][:, LLTF_VALID]
    np.savez(f, H=H, ts=d["timestamp_us"], label=labels[0])
    return H, d["timestamp_us"], int(labels[0])


def chain(x, timestamps):
    """Hampel -> uniform grid -> low-pass."""
    x = hampel_chunked(x, 15, 4.0)
    _, x = resample_uniform(timestamps, x, FS)
    if CUTOFF:
        x = butter_lowpass(x, CUTOFF, FS, ORDER)
    return x


def process_file(path, expected_label):
    """Returns the cleaned amplitude and sanitized phase, each of shape (M, 52)."""
    H, ts, label = load_cached(path)
    assert label == expected_label, f"{path.name}: label {label}, expected {expected_label}"
    phase, _, _ = sanitize_phase(H, LLTF_VALID)
    return chain(np.abs(H), ts), chain(phase, ts)


def windows(A, P, start, stop, keep):
    """Spectrogram images of every WIN_S-second window between samples start and stop."""
    n, step = int(WIN_S * FS), int(STRIDE_S * FS)
    out = []
    for s in range(start, stop - n + 1, step):
        chans = []
        for x in (A, P):
            _, _, power = stft_power(x[s:s + n], FS, NPERSEG, HOP, NFFT)
            chans.append(10 * np.log10(power[keep] + 1e-10))
        out.append(np.stack(chans))                    # (2, freq, time)
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    freqs = np.fft.rfftfreq(NFFT, 1 / FS)
    keep = freqs <= FMAX_KEEP
    X, y, g = [], [], []

    def add(specs, label, group):
        X.extend(specs)
        y.extend([label] * len(specs))
        g.extend([group] * len(specs))

    # walking (class 1) and walking + arm-waving (class 2): recording number = group
    for label, prefix in ((1, "w"), (2, "ww")):
        for gi in range(1, N_GROUPS + 1):
            name = f"{prefix}{gi}.csv"
            A, P = process_file(DATA / name, label)
            specs = windows(A, P, 0, len(A), keep)
            add(specs, label, gi)
            print(f"{name:<8} {len(specs):>4} windows   ({time.time() - t0:5.0f} s)")

    # empty room (class 0): one long recording cut into N_GROUPS time blocks
    A, P = process_file(DATA / "b1.csv", 0)
    edges = np.linspace(0, len(A), N_GROUPS + 1).astype(int)
    guard = int(GUARD_S * FS)
    for gi in range(1, N_GROUPS + 1):
        a = edges[gi - 1] + (guard if gi > 1 else 0)
        b = edges[gi] - (guard if gi < N_GROUPS else 0)
        specs = windows(A, P, a, b, keep)
        add(specs, 0, gi)
    print(f"b1.csv   {int(np.sum(np.array(y) == 0)):>4} windows   ({time.time() - t0:5.0f} s)")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    g = np.array(g, dtype=np.int64)
    assert np.isfinite(X).all(), "found NaN or infinite values in the spectrograms"

    print(f"\nDataset: X has shape {X.shape}  (windows, channels, frequency rows, time columns)")
    print(f"Frequency rows cover 0 to {freqs[keep][-1]:.1f} Hz in steps of {freqs[1]:.2f} Hz")

    print("\nWindows per class and group:")
    print(f"{'class':<22}" + "".join(f"{'group ' + str(i):>9}" for i in range(1, N_GROUPS + 1))
          + f"{'total':>9}")
    for c, cname in enumerate(CLASS_NAMES):
        counts = [int(np.sum((y == c) & (g == i))) for i in range(1, N_GROUPS + 1)]
        print(f"{cname:<22}" + "".join(f"{n:>9}" for n in counts) + f"{sum(counts):>9}")

    print("\nAverage spectrogram level per class (dB):")
    print(f"{'class':<22}{'amplitude':>11}{'phase':>9}")
    for c, cname in enumerate(CLASS_NAMES):
        m = X[y == c].mean(axis=(0, 2, 3))
        print(f"{cname:<22}{m[0]:>11.1f}{m[1]:>9.1f}")

    cut = f"{CUTOFF:g}" if CUTOFF else "none"
    out_file = OUT / f"{SCENARIO}_{SYSTEM}_cut{cut}.npz"
    np.savez_compressed(out_file, X=X, y=y, group=g, freqs=freqs[keep],
                        channels=np.array(["amplitude_dB", "phase_dB"]),
                        win_s=WIN_S, stride_s=STRIDE_S, cutoff_hz=CUTOFF or 0.0)
    print(f"\nSaved {out_file}  ({out_file.stat().st_size / 1e6:.1f} MB)")
    print(f"Total time: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()