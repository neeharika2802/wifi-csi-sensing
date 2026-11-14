"""
step17_walkwave_test.py - Step 17c (run on Google Colab): can we separate walking from
walking + arm-waving better?

Usage on Colab, after unzipping the 8 files into /content/walkwave:
    !python step17_walkwave_test.py /content/walkwave

Binary task on moving windows only (0 = walking, 1 = walking + arm-waving; chance about 50 %).
Accuracy in %, mean +- SD over the 5 test recordings (in-domain: leave one recording out;
cross-domain: train on all recordings of the other environment).

Feature blocks (see step17_features.py): base = Step 9 representation, pca, cad = cadence
features. Planned comparison (chosen before the full run, from a small pilot): base vs cad vs
base+cad at 4 s and 8 s. Everything else is exploratory; picking the best of many settings on
the same test recordings is optimistic, so the last section repeats the best setting with
shuffled training labels.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore", ConvergenceWarning)
WINDOWS_S = [4, 6, 8, 10]
COMBOS = {"base": ["base"], "pca": ["pca"], "cad": ["cad"],
          "base+cad": ["base", "cad"], "base+pca+cad": ["base", "pca", "cad"]}
MODELS = {"LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
          "RF": lambda: RandomForestClassifier(100, n_jobs=-1, random_state=0, min_samples_leaf=2)}


def load(folder, scen, w, blocks):
    z = np.load(Path(folder) / f"{scen}_W{w}.npz")
    return np.concatenate([z[b] for b in blocks], axis=1), z["y"], z["group"]


def per_group(pred, y, g):
    return np.array([100 * np.mean(pred[g == k] == y[g == k]) for k in np.unique(g)])


def logo(F, y, g, mk, rng=None):
    pred = np.empty_like(y)
    for k in np.unique(g):
        ytr = y[g != k]
        if rng is not None:
            ytr = rng.permutation(ytr)
        pred[g == k] = mk().fit(F[g != k], ytr).predict(F[g == k])
    return per_group(pred, y, g)


def fmt(a):
    return f"{a.mean():5.1f} +-{a.std():4.1f}"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python step17_walkwave_test.py <folder with the .npz files>")
    folder = sys.argv[1]
    sections = {"LOS in-domain": ("LOS", None), "NLOS in-domain": ("NLOS", None),
                "LOS -> NLOS": ("LOS", "NLOS"), "NLOS -> LOS": ("NLOS", "LOS")}
    results = {}
    for title, (a, b) in sections.items():
        print(f"\n=== {title} (walking vs waving, chance about 50 %) ===")
        print(f"{'features':<14}{'model':<6}" + "".join(f"{str(w) + ' s':>13}" for w in WINDOWS_S))
        for cname, blocks in COMBOS.items():
            for mname, mk in MODELS.items():
                cells = []
                for w in WINDOWS_S:
                    F, y, g = load(folder, a, w, blocks)
                    if b is None:
                        acc = logo(F, y, g, mk)
                    else:
                        F2, y2, g2 = load(folder, b, w, blocks)
                        acc = per_group(mk().fit(F, y).predict(F2), y2, g2)
                    results[(title, cname, mname, w)] = acc
                    cells.append(f"{acc.mean():>8.1f} +-{acc.std():>2.0f}")
                print(f"{cname:<14}{mname:<6}" + "".join(f"{c:>13}" for c in cells))

    print("\n=== Planned comparison, 4 s vs 8 s: base vs cad vs base+cad (paired by recording) ===")
    for title in sections:
        for mname in MODELS:
            for w in (4, 8):
                ref = results[(title, "base", mname, w)]
                row = []
                for cname in ("cad", "base+cad"):
                    d = results[(title, cname, mname, w)] - ref
                    row.append(f"{cname}: {d.mean():+5.1f} (worse in {int((d < 0).sum())}/5)")
                print(f"  {title:<16}{mname:<4}{w:>3} s   vs base   " + "   ".join(row))

    best = max(((k, v.mean()) for k, v in results.items() if k[0] == "LOS in-domain"), key=lambda kv: kv[1])[0]
    _, cname, mname, w = best
    F, y, g = load(folder, "LOS", w, COMBOS[cname])
    rng = np.random.default_rng(0)
    ctrl = np.array([logo(F, y, g, MODELS[mname], rng).mean() for _ in range(10)])
    print(f"\n=== Control: best LOS in-domain setting ({cname}, {mname}, {w} s: "
          f"{results[best].mean():.1f} %) with shuffled training labels, 10 shuffles ===")
    print(f"  mean {ctrl.mean():.1f} %, SD {ctrl.std():.1f}, range {ctrl.min():.1f} - {ctrl.max():.1f} "
          f"(expected about 50 %)")


if __name__ == "__main__":
    main()