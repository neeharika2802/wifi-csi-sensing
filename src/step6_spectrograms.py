"""
step6_spectrograms.py - Step 6: lock in the cutoff, remove the static room
pattern and build time-frequency spectrograms (STFT).

Saves figure 09 and prints a simple per-window feature table, so we can see
how well the three classes separate before any machine learning.
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # draw straight into PNG files, no pop-up window
import matplotlib.pyplot as plt

from csi_io import load_csi, amplitude
from dsp import hampel, resample_uniform, butter_lowpass, remove_static, stft_power

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "wallhack1.8k" / "LOS" / "PIFA"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- settings you can play with -------------------------------------------
N_PACKETS = 12000   # about 2 minutes per class
FS = 100.0
CUTOFF = 12.0       # Butterworth low-pass cutoff (Hz), chosen in Step 5
ORDER = 4
NPERSEG = 128       # STFT frame length in samples (1.28 s -> 0.78 Hz resolution)
HOP = 16            # frames start every 16 samples (0.16 s)
NFFT = 256          # zero-padded FFT length (smoother picture, same resolution)
SHOW_SECONDS = 30   # how much time to show in the figure
FMAX = 25.0         # highest frequency shown (Hz)
WIN_S = 4.0         # window length for the feature table (seconds)
SPLIT_HZ = 4.0      # "slow" below this, "fast" above it
# ---------------------------------------------------------------------------

FILES = {
    "Empty room": "b1.csv",
    "Walking": "w1.csv",
    "Walking + arm-waving": "ww1.csv",
}

filtered = {}
for name, fn in FILES.items():
    d = load_csi(DATA / fn, max_rows=N_PACKETS)
    A, _ = hampel(amplitude(d["csi"]), 15, 4.0)          # 1) remove glitches
    _, A = resample_uniform(d["timestamp_us"], A, FS)    # 2) uniform 100 Hz grid
    filtered[name] = butter_lowpass(A, CUTOFF, FS, ORDER)  # 3) low-pass

# ------------------------------------------------------- Figure 9: spectrograms
specs = {name: stft_power(x, FS, NPERSEG, HOP, NFFT) for name, x in filtered.items()}
db = {n: 10 * np.log10(P + 1e-12) for n, (f, t, P) in specs.items()}
allv = np.concatenate([v.ravel() for v in db.values()])
vmin, vmax = np.percentile(allv, 1), np.percentile(allv, 99.7)

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, name in zip(axes, FILES):
    f, t, _ = specs[name]
    im = ax.pcolormesh(t, f, db[name], shading="auto", cmap="magma",
                       vmin=vmin, vmax=vmax)
    ax.set_ylim(0, FMAX)
    ax.set_xlim(0, SHOW_SECONDS)
    ax.axhline(CUTOFF, color="w", ls=":", lw=0.8)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(name, loc="left", fontsize=10)
axes[-1].set_xlabel("Time (s)")
fig.colorbar(im, ax=axes, label="Power (dB)")
fig.suptitle(f"STFT spectrograms of the cleaned amplitude "
             f"(frame {NPERSEG / FS:.2f} s, low-pass {CUTOFF:g} Hz, dotted line)")
fig.savefig(FIG / "09_spectrograms.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------- feature table: classes before any ML
n = int(WIN_S * FS)
print(f"Per {WIN_S:g}-second window (static pattern removed), median [10th - 90th percentile]:")
print(f"{'class':<22}{'windows':>8}   {'signal power':<22}{'share of power > ' + str(SPLIT_HZ) + ' Hz':<24}")
freqs = np.fft.rfftfreq(n, 1 / FS)
for name, x in filtered.items():
    pw, share = [], []
    for s in range(0, len(x) - n + 1, n):
        w = remove_static(x[s:s + n])
        pw.append(np.mean(w ** 2))
        X = np.fft.rfft(w * np.hanning(n)[:, None], axis=0)
        P = np.mean(np.abs(X) ** 2, axis=1)
        share.append(P[freqs > SPLIT_HZ].sum() / P[1:].sum())
    q = lambda v: np.percentile(v, [50, 10, 90])
    p50, p10, p90 = q(pw)
    s50, s10, s90 = q(share)
    print(f"{name:<22}{len(pw):>8}   {p50:6.2f} [{p10:5.2f} - {p90:5.2f}]   "
          f"{s50:5.2f} [{s10:4.2f} - {s90:4.2f}]")

print(f"\nSaved figure to {FIG / '09_spectrograms.png'}")