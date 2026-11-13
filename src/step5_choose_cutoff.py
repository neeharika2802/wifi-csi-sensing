"""
step5_choose_cutoff.py - Step 5: choose the low-pass cutoff from the data.

Idea: at every frequency, compare the power of the MOVING recordings with the
power of the EMPTY room recording (which is pure noise + static room).
Where the ratio falls to about 1 (0 dB), there is no movement information
left, so a low-pass filter can safely cut there.

Also shows what different cutoffs do to the deepest signal fade (a real
movement event), so we do not filter away the thing we want to detect.
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # draw straight into PNG files, no pop-up window
import matplotlib.pyplot as plt
from scipy import signal

from csi_io import load_csi, amplitude
from dsp import hampel, resample_uniform, butter_lowpass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "wallhack1.8k" / "LOS" / "PIFA"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- settings you can play with -------------------------------------------
N_PACKETS = 12000  # about 2 minutes per class (more data = smoother spectrum)
FS = 100.0
SC = 26                       # subcarrier used for the dip zoom
TRY_CUTOFFS = [20, 12, 8]     # cutoffs (Hz) compared in the dip zoom
ORDER = 4
# ---------------------------------------------------------------------------

FILES = {
    "Empty room": "b1.csv",
    "Walking": "w1.csv",
    "Walking + arm-waving": "ww1.csv",
}

clean, tt = {}, {}
for name, fn in FILES.items():
    d = load_csi(DATA / fn, max_rows=N_PACKETS)
    A, _ = hampel(amplitude(d["csi"]), 15, 4.0)
    tt[name], clean[name] = resample_uniform(d["timestamp_us"], A, FS)


def mean_psd(x):
    """Welch PSD averaged over subcarriers, after removing each subcarrier's mean."""
    f, p = signal.welch(x - x.mean(axis=0), fs=FS, nperseg=512, axis=0)
    return f, p.mean(axis=1)


def smooth(y, width=5):
    """Moving average over `width` frequency bins (resolution is about 0.2 Hz)."""
    return np.convolve(y, np.ones(width) / width, mode="same")


f, p_empty = mean_psd(clean["Empty room"])
ratio_db = {}
for name in ("Walking", "Walking + arm-waving"):
    _, p = mean_psd(clean[name])
    ratio_db[name] = smooth(10 * np.log10(p / p_empty))


def first_below(r_db, thr_db, f_min=0.5):
    """First frequency (above f_min) where the movement-to-noise ratio is below thr_db."""
    idx = np.where((f > f_min) & (r_db < thr_db))[0]
    return f[idx[0]] if idx.size else float("nan")


print("Movement power relative to the empty-room noise floor.")
print("Frequency (Hz) at which that ratio first falls below:")
print(f"{'class':<22}{'6 dB (4x)':>11}{'3 dB (2x)':>11}{'1 dB (1.26x)':>14}")
for name, r in ratio_db.items():
    print(f"{name:<22}{first_below(r, 6):>11.1f}{first_below(r, 3):>11.1f}"
          f"{first_below(r, 1):>14.1f}")

# --------------------------------------------- how much of a real dip survives
name = "Walking"
A, t = clean[name], tt[name]
x = A[:, SC]
i0 = int(np.argmin(x))
base = np.median(x)
lo, hi = max(0, i0 - 150), min(len(x), i0 + 150)
depth_clean = base - x[lo:hi].min()
print(f"\nDeepest fade in '{name}' (subcarrier {SC}) at t = {t[i0]:.1f} s: "
      f"|H| drops from {base:.1f} to {x[lo:hi].min():.1f}")
print(f"{'cutoff':>8}{'dip depth kept':>16}")
filt = {}
for c in TRY_CUTOFFS:
    filt[c] = butter_lowpass(A, c, FS, ORDER)[:, SC]
    kept = (base - filt[c][lo:hi].min()) / depth_clean * 100
    print(f"{c:>6} Hz{kept:>15.0f}%")

# ------------------------------------------------------------------- figure 08
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
for name_, r in ratio_db.items():
    ax1.plot(f, r, label=name_, lw=1.5)
ax1.axhline(0, color="k", lw=0.8)
ax1.axhline(3, color="gray", ls=":", lw=0.9, label="3 dB (2x noise floor)")
ax1.set_xlim(0, 30)
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Movement power / empty-room power (dB)")
ax1.set_title("Where does movement rise above the noise floor?")
ax1.legend(fontsize=8)

tw = t[lo:hi]
ax2.plot(tw, x[lo:hi], color="lightgray", lw=1.5, label="before low-pass")
for c in TRY_CUTOFFS:
    ax2.plot(tw, filt[c][lo:hi], lw=1.2, label=f"cutoff {c} Hz")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("|H|")
ax2.set_title(f"Deepest fade, subcarrier {SC}: what each cutoff does to it")
ax2.legend(fontsize=8)
fig.savefig(FIG / "08_choose_cutoff.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved figure to {FIG / '08_choose_cutoff.png'}")