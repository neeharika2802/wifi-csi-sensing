"""
step12_ablation.py - Step 12b (run on Google Colab): which DSP steps matter?

Usage on Colab, after unzipping the 14 files into /content/ablation:
    !python step12_ablation.py /content/ablation
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore", ConvergenceWarning)

VARIANTS = ["full", "no_hampel", "no_resample", "no_phase_sanitize",
            "no_lowpass", "no_frame_mean", "raw_all"]
FEATURE_SETS = {"amplitude only": [0], "phase only": [1], "amplitude + phase": [0, 1]}


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def features(X, channels):
    parts = []
    for c in channels:
        x = X[:, c]
        parts += [x.mean(axis=2), x.std(axis=2), x.max(axis=2)]
    return np.concatenate(parts, axis=1)


def per_group_acc(pred, y, g):
    return [100 * np.mean(pred[g == k] == y[g == k]) for k in np.unique(g)]


def evaluate(XA, yA, gA, XB, yB, gB, channels):
    FA, FB = features(XA, channels), features(XB, channels)
    inside, cross = [], []
    pred_cross_all, y_cross_all = [], []
    for F, y, g in ((FA, yA, gA), (FB, yB, gB)):
        pred = np.empty_like(y)
        for k in np.unique(g):
            pred[g == k] = make_model().fit(F[g != k], y[g != k]).predict(F[g == k])
        inside += per_group_acc(pred, y, g)
    for Ftr, ytr, Fte, yte, gte in ((FA, yA, FB, yB, gB), (FB, yB, FA, yA, gA)):
        pred = make_model().fit(Ftr, ytr).predict(Fte)
        cross += per_group_acc(pred, yte, gte)
        pred_cross_all.append(pred)
        y_cross_all.append(yte)
    p, y = np.concatenate(pred_cross_all), np.concatenate(y_cross_all)
    moving = y > 0
    empty_vs_moving = 100 * np.mean((y == 0) == (p == 0))
    walk_vs_wave = 100 * np.mean(p[moving] == y[moving])
    return inside, cross, empty_vs_moving, walk_vs_wave


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python step12_ablation.py <folder with the .npz files>")
    folder = Path(sys.argv[1])
    rows = {}
    for v in VARIANTS:
        fa, fb = folder / f"{v}_LOS.npz", folder / f"{v}_NLOS.npz"
        if not (fa.exists() and fb.exists()):
            print(f"(skipping {v}: files not found)")
            continue
        za, zb = np.load(fa), np.load(fb)
        for fname, chans in FEATURE_SETS.items():
            rows[(v, fname)] = evaluate(za["X"], za["y"], za["group"],
                                        zb["X"], zb["y"], zb["group"], chans)
        print(f"done {v}", flush=True)

    for fname in FEATURE_SETS:
        print(f"\n=== {fname} ===")
        print(f"{'variant':<20}{'in-domain %':>16}{'cross-domain %':>18}"
              f"{'cross: empty/moving':>21}{'cross: walk/wave':>18}{'cross vs full':>15}")
        base = np.mean(rows[("full", fname)][1]) if ("full", fname) in rows else None
        for v in VARIANTS:
            if (v, fname) not in rows:
                continue
            inside, cross, em, ww = rows[(v, fname)]
            diff = "" if base is None else f"{np.mean(cross) - base:+.1f}"
            print(f"{v:<20}{np.mean(inside):>9.1f} +-{np.std(inside):>3.1f}"
                  f"{np.mean(cross):>11.1f} +-{np.std(cross):>3.1f}"
                  f"{em:>21.1f}{ww:>18.1f}{diff:>15}")


if __name__ == "__main__":
    main()