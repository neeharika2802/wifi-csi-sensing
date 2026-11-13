"""
step7_phase.py - Step 7: phase sanitization.

1. Shows raw phase vs sanitized phase for consecutive packets (figure 10).
2. Builds the same cleaned time series and spectrograms for PHASE as we did
   for amplitude (figure 11).
3. Prints how much packet-to-packet phase jitter was removed, and compares how
   well amplitude and phase separate walking from walking+waving.
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # draw straight into PNG files, no pop-up window
import matplotlib.pyplot as plt

from csi_io import load_csi, amplitude, LLTF_VALID
from dsp import (hampel, resample_uniform, butter_lowpass, sanitize_phase,
                 stft_power, window_features)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "wallhack1.8k" / "LOS" / "PIFA"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- settings you can play with -------------------------------------------
N_PACKETS = 12000
FS = 100.0
CUTOFF = 12.0          # low-pass cutoff (Hz) chosen in Step 5
ORDER = 4
NPERSEG, HOP, NFFT = 128, 16, 256   # STFT settings (same as Step 6)
SC = 26                # subcarrier shown in the time-domain figure
DEMO_CLASS = "Empty room"   # class used for the before/after demo (figure 10)
DEMO_PACKETS = 6
# ---------------------------------------------------------------------------

FILES = {
    "Empty room": "b1.csv",
    "Walking": "w1.csv",
    "Walking + arm-waving": "ww1.csv",
}

amp, pha, t_axis = {}, {}, {}
raw = {}      # (unwrapped raw phase, sanitized phase) of the first packets, for the figure
jitter = {}   # packet-to-packet phase change, raw vs sanitized


def clean(x, timestamps):
    """Hampel -> uniform 100 Hz grid -> Butterworth low-pass."""
    x, _ = hampel(x, 15, 4.0)
    t, x = resample_uniform(timestamps, x, FS)
    return t, butter_lowpass(x, CUTOFF, FS, ORDER)


for name, fn in FILES.items():
    d = load_csi(DATA / fn, max_rows=N_PACKETS)
    H = d["csi"][:, LLTF_VALID]                        # (N, 52) complex
    resid, slope, offset = sanitize_phase(H, LLTF_VALID)

    # how much does the phase jump from one packet to the next? (median, in rad)
    raw_step = np.angle(H[1:] * np.conj(H[:-1]))       # wrapped phase change
    jitter[name] = (np.median(np.abs(raw_step)),
                    np.median(np.abs(np.diff(resid, axis=0))))

    k = DEMO_PACKETS
    raw[name] = (np.unwrap(np.angle(H[:k]), axis=1), resid[:k])

    t_axis[name], amp[name] = clean(np.abs(H), d["timestamp_us"])
    _, pha[name] = clean(resid, d["timestamp_us"])

print("Median change of phase between consecutive packets (radians):")
print(f"{'class':<22}{'raw phase':>12}{'sanitized':>12}")
for name, (r, s) in jitter.items():
    print(f"{name:<22}{r:>12.3f}{s:>12.3f}")

# --------------------------------------------- Figure 10: before / after demo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
r, s = raw[DEMO_CLASS]
for i in range(DEMO_PACKETS):
    ax1.plot(LLTF_VALID, r[i], label=f"packet {i}")
    ax2.plot(LLTF_VALID, s[i])
ax1.set_title("Raw phase (unwrapped)")
ax2.set_title("Sanitized phase (line removed)")
for ax in (ax1, ax2):
    ax.set_xlabel("Subcarrier index")
    ax.set_ylabel("Phase (rad)")
ax1.legend(ncol=2, fontsize=8)
fig.suptitle(f"{DEMO_CLASS}: {DEMO_PACKETS} consecutive packets")
fig.savefig(FIG / "10_phase_sanitization_demo.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------- Figure 11: phase over time + phase spectrogram
specs = {n: stft_power(x, FS, NPERSEG, HOP, NFFT) for n, x in pha.items()}
db = {n: 10 * np.log10(P + 1e-12) for n, (f, t, P) in specs.items()}
allv = np.concatenate([v.ravel() for v in db.values()])
vmin, vmax = np.percentile(allv, 1), np.percentile(allv, 99.7)
seg = np.concatenate([pha[n][:1500, SC] for n in FILES])
pad = 0.05 * (seg.max() - seg.min())
ylo, yhi = seg.min() - pad, seg.max() + pad

fig, axes = plt.subplots(3, 2, figsize=(14, 9), layout="constrained",
                         gridspec_kw={"width_ratios": [1, 1.2]})
for row, name in enumerate(FILES):
    ax_t, ax_s = axes[row]
    ax_t.plot(t_axis[name], pha[name][:, SC], lw=0.9)
    ax_t.set_xlim(0, 15)
    ax_t.set_ylim(ylo, yhi)
    ax_t.set_ylabel("Phase (rad)")
    ax_t.set_title(f"{name}: sanitized phase, subcarrier {SC}", loc="left", fontsize=10)
    f, t, _ = specs[name]
    im = ax_s.pcolormesh(t, f, db[name], shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
    ax_s.set_ylim(0, 25)
    ax_s.set_xlim(0, 30)
    ax_s.axhline(CUTOFF, color="w", ls=":", lw=0.8)
    ax_s.set_ylabel("Frequency (Hz)")
    ax_s.set_title("Phase spectrogram", loc="left", fontsize=10)
axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")
fig.colorbar(im, ax=axes[:, 1], label="Power (dB)")
fig.savefig(FIG / "11_phase_time_and_spectrogram.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --------------------------------- feature table: amplitude vs phase, per class
feats = {}
for name in FILES:
    feats[name] = {"amp": window_features(amp[name]), "pha": window_features(pha[name])}

print("\nPer 4-second window, median [10th - 90th percentile]")
print(f"{'class':<22}{'amp power':<22}{'phase power (rad^2)':<24}")
for name in FILES:
    cells = []
    for key in ("amp", "pha"):
        p = feats[name][key][0]
        p50, p10, p90 = np.percentile(p, [50, 10, 90])
        cells.append(f"{p50:6.3f} [{p10:5.3f}-{p90:5.3f}]")
    print(f"{name:<22}{cells[0]:<22}{cells[1]:<24}")


def separability(a, b):
    """0.5 = the two groups cannot be told apart, 1.0 = perfectly separated."""
    auc = np.mean(a[:, None] > b[None, :]) + 0.5 * np.mean(a[:, None] == b[None, :])
    return max(auc, 1 - auc)


w, ww = "Walking", "Walking + arm-waving"
print("\nWalking vs walking+waving: how well does each feature tell them apart?")
print("(0.50 = coin flip, 1.00 = perfect; rough guide only, windows from one recording each)")
for label, key, idx in [("amplitude power", "amp", 0), ("amplitude share > 4 Hz", "amp", 1),
                        ("phase power", "pha", 0), ("phase share > 4 Hz", "pha", 1)]:
    print(f"  {label:<24}{separability(feats[w][key][idx], feats[ww][key][idx]):.2f}")

print(f"\nSaved figures 10 and 11 to {FIG}")