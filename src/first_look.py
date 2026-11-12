"""
first_look.py - first look at the raw CSI signals (Step 2 of the project).

Produces 4 figures in results/figures/ and prints basic statistics.
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # draw straight into PNG files, no pop-up window
import matplotlib.pyplot as plt

from csi_io import load_csi, amplitude, packet_intervals_s, LLTF_VALID

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "wallhack1.8k" / "LOS" / "PIFA"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

N_PACKETS = 3000  # about 30 seconds at 100 packets per second
FILES = {
    "Empty room": "b1.csv",
    "Walking": "w1.csv",
    "Walking + arm-waving": "ww1.csv",
}

# ---------------------------------------------------------------- load + stats
data = {}
print(f"{'class':<22}{'packets':>8}{'skipped':>8}{'median dt (ms)':>16}"
      f"{'Fs (Hz)':>9}{'late >1.5x':>12}{'max dt (ms)':>13}{'mean RSSI':>11}")
for name, fn in FILES.items():
    d = load_csi(DATA / fn, max_rows=N_PACKETS)
    d["t"] = (d["timestamp_us"] - d["timestamp_us"][0]) * 1e-6  # seconds
    d["dt"] = packet_intervals_s(d["timestamp_us"])
    d["A"] = amplitude(d["csi"])                                # (N, 52)
    data[name] = d
    med = np.median(d["dt"])
    late = np.mean(d["dt"] > 1.5 * med) * 100
    print(f"{name:<22}{len(d['csi']):>8}{d['skipped']:>8}{med * 1e3:>16.2f}"
          f"{1 / med:>9.1f}{late:>11.1f}%{d['dt'].max() * 1e3:>13.1f}"
          f"{d['rssi'].mean():>11.1f}")

print("\nAmplitude variation over time (averaged over the 52 subcarriers):")
for name, d in data.items():
    A = d["A"]
    cv = np.mean(A.std(axis=0) / A.mean(axis=0))
    print(f"  {name:<22} mean amplitude = {A.mean():6.2f}   "
          f"std/mean over time = {cv:.3f}")

names = list(data.keys())

# ------------------------------------------------ Figure 1: amplitude heatmaps
vmax = np.percentile(np.concatenate([d["A"].ravel() for d in data.values()]), 99)
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for ax, name in zip(axes, names):
    d = data[name]
    im = ax.imshow(d["A"].T, aspect="auto", origin="lower", cmap="viridis",
                   vmin=0, vmax=vmax, extent=[0, d["t"][-1], 0, d["A"].shape[1]])
    ax.set_ylabel("Subcarrier #")
    ax.set_title(name, loc="left", fontsize=10)
axes[-1].set_xlabel("Time (s)")
fig.colorbar(im, ax=axes, label="|H| (amplitude)")
fig.suptitle("Raw CSI amplitude: subcarrier vs time (LOS, PIFA)")
fig.savefig(FIG / "01_raw_amplitude_heatmaps.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ----------------------------------- Figure 2: one subcarrier over time
sc = 26  # a subcarrier near the middle of the 52 valid ones
all_vals = np.concatenate([d["A"][:, sc] for d in data.values()])
ymin, ymax = all_vals.min() * 0.9, all_vals.max() * 1.1
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, sharey=True)
for ax, name in zip(axes, names):
    d = data[name]
    ax.plot(d["t"], d["A"][:, sc], lw=0.8)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("|H|")
    ax.set_title(name, loc="left", fontsize=10)
axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"Amplitude of one subcarrier (valid index {sc}, "
             f"raw subcarrier {LLTF_VALID[sc]})")
fig.savefig(FIG / "02_single_subcarrier_timeseries.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)

# ------------------------------- Figure 3: packet interval (sampling) analysis
fig, ax = plt.subplots(figsize=(8, 4))
for name in names:
    ax.hist(np.clip(data[name]["dt"] * 1e3, 0, 50), bins=100, range=(0, 50),
            alpha=0.5, label=name)
ax.set_yscale("log")
ax.set_xlabel("Time between consecutive packets (ms), clipped at 50 ms")
ax.set_ylabel("Count (log scale)")
ax.set_title("Is the sampling uniform? (ideal: a single spike at 10 ms)")
ax.legend()
fig.savefig(FIG / "03_packet_interval_histogram.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)

# ------------------------------- Figure 4: raw phase of consecutive packets
H = data["Empty room"]["csi"][:6][:, LLTF_VALID]
fig, ax = plt.subplots(figsize=(8, 4))
for k in range(H.shape[0]):
    ax.plot(np.unwrap(np.angle(H[k])), label=f"packet {k}")
ax.set_xlabel("Subcarrier # (52 valid ones)")
ax.set_ylabel("Unwrapped phase (rad)")
ax.set_title("Raw phase, EMPTY room, 6 consecutive packets\n"
             "(static room: curves should match; differences = hardware offsets)")
ax.legend(ncol=3, fontsize=8)
fig.savefig(FIG / "04_raw_phase_consecutive_packets.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)

print(f"\nSaved 4 figures to {FIG}")