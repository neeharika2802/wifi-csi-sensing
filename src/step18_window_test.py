"""
step18_window_test.py - Step 18b (run on Google Colab): accuracy of walking vs waving against
window length.

Usage on Colab, after unzipping the 10 files into /content/sweep:
    !python step18_window_test.py /content/sweep

Logistic regression on the Step 9 features (amplitude + phase), binary task (chance about 50 %),
mean +- SD over the 5 test recordings. In-domain = leave one recording out; cross-domain =
train on all recordings of the other environment. "vs 4 s" = paired change per recording,
"worse" = in how many of the 5 recordings.

The last block estimates the 3-class accuracy of a two-stage system on LOS from numbers we
already measured: stage 1 (empty vs moving, 4 s windows) was 99.7 % right, and 574 of 1,842
LOS windows are empty room. Overall = share_empty * 0.997 + share_moving * 0.997 * stage-2
accuracy. It is an estimate, not a measured result.
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
WINDOWS_S = [4, 8, 12, 20, 30]


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def load(folder, scen, w):
    z = np.load(Path(folder) / f"{scen}_W{w}.npz")
    return z["base"], z["y"], z["group"]


def per_group(pred, y, g):
    return np.array([100 * np.mean(pred[g == k] == y[g == k]) for k in np.unique(g)])


def logo(F, y, g):
    pred = np.empty_like(y)
    for k in np.unique(g):
        pred[g == k] = make_model().fit(F[g != k], y[g != k]).predict(F[g == k])
    return per_group(pred, y, g)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python step18_window_test.py <folder with the .npz files>")
    folder = sys.argv[1]
    settings = {"LOS in-domain": ("LOS", None), "NLOS in-domain": ("NLOS", None),
                "LOS -> NLOS": ("LOS", "NLOS"), "NLOS -> LOS": ("NLOS", "LOS")}
    res = {}
    for title, (a, b) in settings.items():
        for w in WINDOWS_S:
            F, y, g = load(folder, a, w)
            if b is None:
                res[(title, w)] = logo(F, y, g)
            else:
                F2, y2, g2 = load(folder, b, w)
                res[(title, w)] = per_group(make_model().fit(F, y).predict(F2), y2, g2)
        print(f"done {title}", flush=True)

    print("\nWalking vs waving, accuracy % (mean +- SD over 5 recordings; chance about 50 %)")
    print(f"{'window':<8}" + "".join(f"{t:>24}" for t in settings))
    for w in WINDOWS_S:
        print(f"{str(w) + ' s':<8}" + "".join(f"{res[(t, w)].mean():>14.1f} +-{res[(t, w)].std():>4.1f}  "
                                                for t in settings))
    print("\nPaired change versus the 4 s window")
    for w in WINDOWS_S[1:]:
        cells = []
        for t in settings:
            d = res[(t, w)] - res[(t, 4)]
            cells.append(f"{d.mean():+6.1f} (worse {int((d < 0).sum())}/5)")
        print(f"{str(w) + ' s':<8}" + "".join(f"{c:>24}" for c in cells))

    print("\nEstimated LOS 3-class accuracy of a two-stage system (stage 1 at 4 s, stage 2 at the window below)")
    empty, total = 574, 1842
    for w in WINDOWS_S:
        a2 = res[("LOS in-domain", w)].mean() / 100
        est = (empty / total) * 0.997 + ((total - empty) / total) * 0.997 * a2
        print(f"  {w:>2} s: stage 2 = {100 * a2:.1f} %  ->  about {100 * est:.1f} % overall")


if __name__ == "__main__":
    main()