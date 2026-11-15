"""
step18_window_sweep.py - Step 18a: how long should the window be for walking vs waving?

Step 17 showed that longer windows help the Step 9 features (4 s -> 10 s gave about +6 points in
three of four settings). Here the window is pushed further: 4, 8, 12, 20 and 30 seconds. Only the
Step 9 features (mean / std / max over time of every spectrogram row, amplitude and phase) are
computed, for walking (w1-w5) and walking + arm-waving (ww1-ww5), LOS and NLOS.
A new window starts every second, so long windows overlap heavily (a 30 s window overlaps its
neighbour by 29 s); the recording-wise test in Step 18b keeps this honest, but there are only 5
test recordings per setting, so look at trends, not at single numbers.

Needs the Step 8 / 11a caches. numpy + scipy only. This takes longer than Step 17 (long windows).
    python src/step18_window_sweep.py            Output: data/processed/walkwave_sweep/*.npz
"""
import time
from pathlib import Path

import numpy as np

from csi_io import LLTF_VALID
from dsp import hampel_chunked, resample_uniform, butter_lowpass, sanitize_phase
from step17_features import FS, _db_rows, _spectrogram_power, _stats

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "processed" / "cache"
OUT = ROOT / "data" / "processed" / "walkwave_sweep"
WINDOWS_S = [4, 8, 12, 20, 30]


def chain(x, ts):
    x = hampel_chunked(x, 15, 4.0)
    _, x = resample_uniform(ts, x, FS)
    return butter_lowpass(x, 12.0, FS, 4)


def base_features(A, P, win_s, stride_s=1.0):
    n, step = int(win_s * FS), int(stride_s * FS)
    rows = []
    for s in range(0, len(A) - n + 1, step):
        parts = []
        for x in (A[s:s + n], P[s:s + n]):
            f, power = _spectrogram_power(x)
            parts.append(_stats(_db_rows(power.mean(axis=1), f)))
        rows.append(np.concatenate(parts))
    return np.array(rows, dtype=np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for scen in ("LOS", "NLOS"):
        if all((OUT / f"{scen}_W{w}.npz").exists() for w in WINDOWS_S):
            print(f"{scen}: all files already there, skipped")
            continue
        store = {w: {"base": [], "y": [], "group": []} for w in WINDOWS_S}
        for label, prefix in ((0, "w"), (1, "ww")):
            for gi in range(1, 6):
                f = CACHE / f"{scen}_PIFA_{prefix}{gi}.npz"
                if not f.exists():
                    raise SystemExit(f"Missing {f.name}. Run Step 8 (LOS) and Step 11a (NLOS) first.")
                z = np.load(f)
                A = chain(np.abs(z["H"]), z["ts"])
                P = chain(sanitize_phase(z["H"], LLTF_VALID)[0], z["ts"])
                counts = []
                for w in WINDOWS_S:
                    b = base_features(A, P, w)
                    store[w]["base"].append(b)
                    store[w]["y"].append(np.full(len(b), label))
                    store[w]["group"].append(np.full(len(b), gi))
                    counts.append(len(b))
                print(f"{scen}/{prefix}{gi:<3} windows for {'/'.join(map(str, WINDOWS_S))} s: {counts}"
                      f"   ({time.time() - t0:.0f} s)", flush=True)
        for w in WINDOWS_S:
            np.savez_compressed(OUT / f"{scen}_W{w}.npz",
                                **{k: np.concatenate(v) for k, v in store[w].items()})
        print(f"{scen}: saved {len(WINDOWS_S)} files")
    print(f"\nDone. Files are in {OUT}")


if __name__ == "__main__":
    main()