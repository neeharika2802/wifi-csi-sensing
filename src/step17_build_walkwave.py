"""
step17_build_walkwave.py - Step 17b: build the walking vs walking + arm-waving feature files.

Cuts every walking (w1-w5) and waving (ww1-ww5) recording of LOS and NLOS into windows of
4, 6, 8 and 10 seconds (a new window every second) and computes the feature blocks of
step17_features.py. The empty room is not needed here. Same cleaning as the rest of the project.

Needs the Step 8 / Step 11a caches. numpy + scipy only. Output: data/processed/walkwave/*.npz
    python src/step17_build_walkwave.py
"""
import time
from pathlib import Path

import numpy as np

from csi_io import LLTF_VALID
from dsp import hampel_chunked, resample_uniform, butter_lowpass, sanitize_phase
from step17_features import window_blocks

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "processed" / "cache"
OUT = ROOT / "data" / "processed" / "walkwave"
WINDOWS_S = [4, 6, 8, 10]


def chain(x, ts):
    x = hampel_chunked(x, 15, 4.0)
    _, x = resample_uniform(ts, x, 100.0)
    return butter_lowpass(x, 12.0, 100.0, 4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for scen in ("LOS", "NLOS"):
        if all((OUT / f"{scen}_W{w}.npz").exists() for w in WINDOWS_S):
            print(f"{scen}: all files already there, skipped")
            continue
        store = {w: {k: [] for k in ("base", "pca", "eig", "cad", "y", "group")} for w in WINDOWS_S}
        for label, prefix in ((0, "w"), (1, "ww")):
            for gi in range(1, 6):
                f = CACHE / f"{scen}_PIFA_{prefix}{gi}.npz"
                if not f.exists():
                    raise SystemExit(f"Missing {f.name}. Run Step 8 (LOS) and Step 11a (NLOS) first.")
                z = np.load(f)
                H, ts = z["H"], z["ts"]
                A = chain(np.abs(H), ts)
                P = chain(sanitize_phase(H, LLTF_VALID)[0], ts)
                counts = []
                for w in WINDOWS_S:
                    blk = window_blocks(A, P, w, 0, len(A))
                    n = len(blk["base"])
                    for k in ("base", "pca", "eig", "cad"):
                        store[w][k].append(blk[k])
                    store[w]["y"].append(np.full(n, label))
                    store[w]["group"].append(np.full(n, gi))
                    counts.append(n)
                print(f"{scen}/{prefix}{gi:<3} windows for 4/6/8/10 s: {counts}   ({time.time() - t0:.0f} s)")
        for w in WINDOWS_S:
            d = {k: np.concatenate(v) for k, v in store[w].items()}
            np.savez_compressed(OUT / f"{scen}_W{w}.npz", **d)
        print(f"{scen}: saved {len(WINDOWS_S)} files")
    print(f"\nDone. Files are in {OUT}")


if __name__ == "__main__":
    main()