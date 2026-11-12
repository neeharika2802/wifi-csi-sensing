"""
step3_clean_amplitude.py - Step 3: Hampel filter + uniform resampling of the amplitude.

Prints how many samples were changed and saves a before/after figure.
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # draw straight into PNG files, no pop-up window
import matplotlib.pyplot as plt

from csi_io import load_csi, amplitude, packet_intervals_s
from dsp import hampel, resample_uniform

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "wallhack1.8k" / "LOS" / "PIFA"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- settings you can play with -------------------------------------------
N_PACKETS = 3000   # about 30 seconds
WINDOW = 15        # Hampel window length in packets (must be odd)
N_SIGMAS = 4.0     # bigger number = less aggressive filter
FS = 100.0         # target sampling rate after resampling (Hz)
SC = 26            # which subcarrier to plot
# ---------------------------------------------------------------------------

FILES = {
    "Empty room": "b1.csv",
    "Walking": "w1.csv",
    "Walking + arm-waving": "ww1.csv",
}

print(f"Hampel filter: window = {WINDOW} packets, threshold = {N_SIGMAS} x scaled MAD\n")
print(f"{'class':<22}{'samples replaced':>18}{'output samples':>16}{'longest gap (ms)':>18}")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
for ax, (name, fn) in zip(axes, FILES.items()):
    d = load_csi(DATA / fn, max_rows=N_PACKETS)
    A = amplitude(d["csi"])                                   # (N, 52)

    # 1) outlier removal on the raw packet sequence
    A_clean, mask = hampel(A, WINDOW, N_SIGMAS)
    # 2) put the cleaned packets on a uniform 100 Hz time grid
    t_new, A_uni = resample_uniform(d["timestamp_us"], A_clean, FS)

    gap_ms = packet_intervals_s(d["timestamp_us"]).max() * 1e3
    print(f"{name:<22}{mask.mean() * 100:>17.2f}%{A_uni.shape[0]:>16}{gap_ms:>18.1f}")

    t_raw = (d["timestamp_us"] - d["timestamp_us"][0]) * 1e-6
    ax.plot(t_raw, A[:, SC], color="lightgray", lw=1.0, label="raw")
    ax.plot(t_new, A_uni[:, SC], color="tab:blue", lw=0.8, label="Hampel + resampled")
    flagged = mask[:, SC]
    ax.plot(t_raw[flagged], A[flagged, SC], "r.", ms=6, label="replaced by Hampel")
    ax.set_xlim(0, 15)
    ax.set_ylabel("|H|")
    ax.set_title(name, loc="left", fontsize=10)
axes[0].legend(loc="lower right", fontsize=8, ncol=3)
axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"Amplitude of subcarrier {SC}: before and after cleaning "
             f"(Hampel window={WINDOW}, {N_SIGMAS} sigma)")
fig.savefig(FIG / "05_hampel_before_after.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved figure to {FIG / '05_hampel_before_after.png'}")