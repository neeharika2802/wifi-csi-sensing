"""
step11_cross_domain.py - Step 11b (run on Google Colab): train in one environment, test in another.

Usage on Colab (upload this file and both .npz files):
    !python step11_cross_domain.py /content/LOS_PIFA_cut12.npz /content/NLOS_PIFA_cut12.npz
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore", ConvergenceWarning)

CLASS_NAMES = ["Empty room", "Walking", "Walking + arm-waving"]
FEATURE_SETS = {"amplitude only": [0], "phase only": [1], "amplitude + phase": [0, 1]}
NORMS = {                                   # (remove window level, standardize per domain)
    "none": (False, False),
    "window level removed": (True, False),
    "domain standardized": (False, True),
    "both": (True, True),
}


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def features(X, channels):
    """Mean, std and max over time for every frequency row of the chosen channels."""
    parts = []
    for c in channels:
        x = X[:, c]
        parts += [x.mean(axis=2), x.std(axis=2), x.max(axis=2)]
    return np.concatenate(parts, axis=1)


def normalize(X, window, domain):
    X = X.copy()
    if window:
        X = X - X.mean(axis=(2, 3), keepdims=True)
    if domain:
        m = X.mean(axis=(0, 3), keepdims=True)
        s = X.std(axis=(0, 3), keepdims=True) + 1e-6
        X = (X - m) / s
    return X


def metrics(y, pred):
    moving = y > 0
    return (100 * np.mean(pred == y),
            100 * np.mean((y == 0) == (pred == 0)),
            100 * np.mean(pred[moving] == y[moving]))


def in_domain(F, y, g):
    accs = []
    for k in np.unique(g):
        m = make_model().fit(F[g != k], y[g != k])
        accs.append(np.mean(m.predict(F[g == k]) == y[g == k]))
    return 100 * np.mean(accs)


def main():
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python step11_cross_domain.py <domain A .npz> <domain B .npz>")
    names, data = [], []
    for p in sys.argv[1:3]:
        z = np.load(p)
        names.append(Path(p).stem.split("_cut")[0])
        data.append((z["X"], z["y"], z["group"]))
        y = z["y"]
        print(f"{names[-1]}: {len(y)} windows, class sizes {np.bincount(y).tolist()}, "
              f"image shape {z['X'].shape[1:]}")
    (XA, yA, gA), (XB, yB, gB) = data
    assert XA.shape[1:] == XB.shape[1:], "the two datasets must have the same image shape"
    A, B = names
    t0 = time.time()

    print("\nAccuracy (%). Cross-domain = train on every window of one domain, test on the other.")
    print(f"{'features':<19}{'normalization':<22}{A + '->' + B:>16}{B + '->' + A:>16}"
          f"{'in-domain ' + A:>18}{'in-domain ' + B:>18}")
    best = {}
    for fname, chans in FEATURE_SETS.items():
        for nname, (win, dom) in NORMS.items():
            FA = features(normalize(XA, win, dom), chans)
            FB = features(normalize(XB, win, dom), chans)
            p_ab = make_model().fit(FA, yA).predict(FB)
            p_ba = make_model().fit(FB, yB).predict(FA)
            m_ab, m_ba = metrics(yB, p_ab), metrics(yA, p_ba)
            print(f"{fname:<19}{nname:<22}{m_ab[0]:>16.1f}{m_ba[0]:>16.1f}"
                  f"{in_domain(FA, yA, gA):>18.1f}{in_domain(FB, yB, gB):>18.1f}")
            for key, m, yt, pr in ((f"{A}->{B}", m_ab, yB, p_ab), (f"{B}->{A}", m_ba, yA, p_ba)):
                if key not in best or m[0] > best[key][0][0]:
                    best[key] = (m, fname, nname, yt, pr)
    print(f"({time.time() - t0:.0f} s)")

    for key, (m, fname, nname, yt, pr) in best.items():
        cm = np.zeros((3, 3), dtype=int)
        for t, p in zip(yt, pr):
            cm[t, p] += 1
        print(f"\nBest cross-domain setting for {key}: {fname}, {nname}  -> {m[0]:.1f} % "
              f"(empty vs moving {m[1]:.1f} %, walking vs waving {m[2]:.1f} %)")
        print(f"{'':<22}" + "".join(f"{n[:14]:>16}" for n in CLASS_NAMES) + f"{'recall':>9}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"{name:<22}" + "".join(f"{cm[i, j]:>16}" for j in range(3))
                  + f"{100 * cm[i, i] / cm[i].sum():>8.1f}%")


if __name__ == "__main__":
    main()