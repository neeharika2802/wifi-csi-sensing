"""
step15_hampel_spikes.py - Step 15a: why does the Hampel filter matter across environments?

Part 1 (needs the Step 8 / 11a caches): how many samples does the Hampel filter replace in
every recording, how big are those spikes, and are they glitches that hit most subcarriers
of one packet at the same time?
Part 2 (needs the Step 12a ablation files, optional): how much does the spectrogram level of
every class change when the Hampel filter is switched off, in LOS and in NLOS? If the shift is
different in the two environments, the gap between the environments grows -> that is the
mechanism we are testing.

numpy + scipy only.        python src/step15_hampel_spikes.py
"""
from pathlib import Path

import numpy as np

from csi_io import LLTF_VALID
from dsp import hampel, sanitize_phase

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "processed" / "cache"
ABL = ROOT / "data" / "processed" / "ablation"
FILES = ["b1"] + [f"w{i}" for i in range(1, 6)] + [f"ww{i}" for i in range(1, 6)]
CLASS_OF = lambda name: "empty" if name == "b1" else ("walking" if name.startswith("w") and not name.startswith("ww") else "waving")


def hampel_flags(x, window=15, n_sigmas=4.0, chunk=5000):
    """Boolean array: which samples the Hampel filter (window 15, 4 sigma) replaces."""
    half = window // 2
    flags = np.zeros(x.shape, dtype=bool)
    out = np.empty_like(x)
    for s in range(0, len(x), chunk):
        e = min(s + chunk, len(x))
        a, b = max(0, s - half), min(len(x), e + half)
        y, m = hampel(x[a:b], window, n_sigmas)
        flags[s:e] = m[s - a:s - a + (e - s)]
        out[s:e] = y[s - a:s - a + (e - s)]
    return flags, out


def part1():
    print("PART 1 - what does the Hampel filter replace?")
    print("  amp %      share of amplitude samples replaced")
    print("  size %     typical size of a replaced amplitude spike, in % of that subcarrier's level")
    print("  pkts %     share of packets with at least one replaced amplitude sample")
    print("  wide %     share of replaced amplitude samples that sit in a packet where at least half")
    print("             of the 52 subcarriers were replaced (a whole-packet glitch)")
    print("  ph %       share of phase samples replaced (after phase sanitization)")
    print("  ph size    typical size of a replaced phase spike (rad)\n")
    head = (f"{'file':<10}{'amp %':>7}{'size %':>8}{'pkts %':>8}{'wide %':>8}{'ph %':>7}{'ph size':>9}")
    rows = {}
    for scen in ("LOS", "NLOS"):
        print(head)
        for name in FILES:
            f = CACHE / f"{scen}_PIFA_{name}.npz"
            if not f.exists():
                print(f"{scen + '/' + name:<10} (no cache file)")
                continue
            H = np.load(f)["H"]
            A = np.abs(H)
            P, _, _ = sanitize_phase(H, LLTF_VALID)
            fa, ya = hampel_flags(A)
            fp, yp = hampel_flags(P)
            level = np.median(A, axis=0, keepdims=True)
            size = np.median((np.abs(A - ya) / level)[fa]) * 100 if fa.any() else 0.0
            wide_rows = fa.sum(axis=1) >= A.shape[1] // 2
            wide = 100 * fa[wide_rows].sum() / max(fa.sum(), 1)
            r = (100 * fa.mean(), size, 100 * fa.any(axis=1).mean(), wide,
                 100 * fp.mean(), float(np.median(np.abs(P - yp)[fp])) if fp.any() else 0.0)
            rows[(scen, name)] = r
            print(f"{scen + '/' + name:<10}{r[0]:>7.2f}{r[1]:>8.1f}{r[2]:>8.1f}{r[3]:>8.1f}"
                  f"{r[4]:>7.2f}{r[5]:>9.2f}")
        print()
    if rows:
        print("Averages (mean over the files of each group):")
        print(f"{'':<14}{'amp %':>7}{'size %':>8}{'pkts %':>8}{'wide %':>8}{'ph %':>7}{'ph size':>9}")
        for scen in ("LOS", "NLOS"):
            for cls in ("empty", "walking", "waving"):
                sel = [v for (s, n), v in rows.items() if s == scen and CLASS_OF(n) == cls]
                if sel:
                    m = np.mean(sel, axis=0)
                    print(f"{scen + ' ' + cls:<14}{m[0]:>7.2f}{m[1]:>8.1f}{m[2]:>8.1f}{m[3]:>8.1f}"
                          f"{m[4]:>7.2f}{m[5]:>9.2f}")


def part2():
    print("\nPART 2 - spectrogram level (dB) with and without the Hampel filter")
    files = {k: ABL / f"{k}.npz" for k in
             ("full_LOS", "no_hampel_LOS", "full_NLOS", "no_hampel_NLOS")}
    if not all(p.exists() for p in files.values()):
        print("  (ablation files not found - run Step 12a first, skipping)")
        return
    Z = {k: np.load(p) for k, p in files.items()}
    freqs = Z["full_LOS"]["freqs"]
    bands = [(0, 2), (2, 6), (6, 15)]
    classes = ["Empty room", "Walking", "Walking + waving"]
    print("  'gap' = NLOS level minus LOS level. If the gap grows without Hampel, the filter was")
    print("  protecting the model from an environment-dependent level shift.\n")
    for ch, cname in enumerate(["amplitude", "phase"]):
        print(f"  {cname} channel")
        print(f"  {'class':<18}{'band Hz':>9}{'LOS full':>10}{'LOS noH':>9}{'NLOS full':>10}"
              f"{'NLOS noH':>9}{'gap full':>10}{'gap noH':>9}")
        for c, cls in enumerate(classes):
            for lo, hi in bands:
                rows = (freqs >= lo) & (freqs < hi + 1e-9)
                v = {}
                for k, z in Z.items():
                    X, y = z["X"], z["y"]
                    v[k] = X[y == c][:, ch][:, rows].mean()
                g_full = v["full_NLOS"] - v["full_LOS"]
                g_no = v["no_hampel_NLOS"] - v["no_hampel_LOS"]
                print(f"  {cls:<18}{lo:>4}-{hi:<4}{v['full_LOS']:>10.1f}{v['no_hampel_LOS']:>9.1f}"
                      f"{v['full_NLOS']:>10.1f}{v['no_hampel_NLOS']:>9.1f}{g_full:>10.1f}{g_no:>9.1f}")
        print()


if __name__ == "__main__":
    part1()
    part2()