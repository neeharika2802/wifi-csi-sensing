"""
step15_hampel_diagnosis.py - Step 15b (run on Google Colab): what goes wrong without Hampel?

Usage on Colab, after unzipping the Step 12a files into /content/ablation:
    !python step15_hampel_diagnosis.py /content/ablation

For the variants `full` and `no_hampel` (logistic regression, cross-domain LOS <-> NLOS):
  1. accuracy with four normalizations. If the damage of switching Hampel off is just an
     environment-dependent level shift, normalizing every environment (no labels needed) should
     repair most of it.
  2. confusion matrices (no normalization): do empty-room windows get called "moving", or the
     other way round?
Accuracy = mean +- SD over the 10 test recordings (5 per direction).
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
FEATURE_SETS = {"amplitude only": [0], "amplitude + phase": [0, 1]}
NORMS = {"none": (False, False), "domain standardized": (False, True),
         "window level removed": (True, False), "both": (True, True)}
CLASSES = ["Empty", "Walking", "Waving"]


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def normalize(X, window, domain):
    X = X.copy()
    if window:
        X = X - X.mean(axis=(2, 3), keepdims=True)
    if domain:
        m = X.mean(axis=(0, 3), keepdims=True)
        s = X.std(axis=(0, 3), keepdims=True) + 1e-6
        X = (X - m) / s
    return X


def features(X, channels):
    parts = []
    for c in channels:
        x = X[:, c]
        parts += [x.mean(axis=2), x.std(axis=2), x.max(axis=2)]
    return np.concatenate(parts, axis=1)


def cross(variant, folder, channels, win, dom):
    za, zb = np.load(folder / f"{variant}_LOS.npz"), np.load(folder / f"{variant}_NLOS.npz")
    FA = features(normalize(za["X"], win, dom), channels)
    FB = features(normalize(zb["X"], win, dom), channels)
    accs, ys, ps = [], [], []
    for Ftr, ytr, Fte, yte, gte in ((FA, za["y"], FB, zb["y"], zb["group"]),
                                    (FB, zb["y"], FA, za["y"], za["group"])):
        pred = make_model().fit(Ftr, ytr).predict(Fte)
        accs += [100 * np.mean(pred[gte == k] == yte[gte == k]) for k in np.unique(gte)]
        ys.append(yte)
        ps.append(pred)
    return np.array(accs), ys, ps


def confusion(y, p):
    cm = np.zeros((3, 3), dtype=int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    return cm


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python step15_hampel_diagnosis.py <ablation folder>")
    folder = Path(sys.argv[1])
    print("1. Cross-domain accuracy (%), mean +- SD over 10 test recordings")
    for fname, chans in FEATURE_SETS.items():
        print(f"\n  {fname}")
        print(f"  {'normalization':<24}{'full':>14}{'no_hampel':>14}{'loss':>8}")
        for nname, (win, dom) in NORMS.items():
            a, _, _ = cross("full", folder, chans, win, dom)
            b, _, _ = cross("no_hampel", folder, chans, win, dom)
            print(f"  {nname:<24}{a.mean():>8.1f} +-{a.std():>3.1f}{b.mean():>8.1f} +-{b.std():>3.1f}"
                  f"{b.mean() - a.mean():>+8.1f}")

    print("\n2. Confusion matrices, no normalization (rows = true class, columns = predicted)")
    for fname, chans in FEATURE_SETS.items():
        for variant in ("full", "no_hampel"):
            _, ys, ps = cross(variant, folder, chans, False, False)
            for d, label in enumerate(("LOS -> NLOS", "NLOS -> LOS")):
                cm = confusion(ys[d], ps[d])
                print(f"\n  {fname}, {variant}, {label}")
                print("  " + " " * 9 + "".join(f"{c:>9}" for c in CLASSES) + f"{'recall':>9}")
                for i, c in enumerate(CLASSES):
                    print(f"  {c:<9}" + "".join(f"{cm[i, j]:>9}" for j in range(3))
                          + f"{100 * cm[i, i] / cm[i].sum():>8.1f}%")


if __name__ == "__main__":
    main()