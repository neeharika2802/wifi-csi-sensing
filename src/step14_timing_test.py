"""
step14_timing_test.py - Step 14b (run on Google Colab): is the LOS/NLOS gap a timing artifact?

Usage on Colab, after unzipping the 4 files into /content/timing:
    !python step14_timing_test.py /content/timing

Model: logistic regression, no extra normalization. Accuracy in %, mean +- SD over test
recordings (groups 1-5). "paired diff" = the change versus the reference, computed group by
group; "worse" = in how many groups the accuracy went down.

  Test 1  Timing only: LOS_clean vs LOS_nlostiming (same room, NLOS packet timing).
          Train on LOS (leave-one-recording-out), test on the held-out recording in both versions.
  Test 2  Does matching the timing close the cross-domain gap?
          Train on LOS_clean or on LOS_nlostiming, test on NLOS_linear (and the reverse).
  Test 3  NLOS interpolation: NLOS_linear vs NLOS_nearest (nearest packet, no straight lines).
  Test 4  Dose-response: accuracy of the LOS_clean-trained model on NLOS_linear windows,
          sorted within every class by how many packets were lost in the window.
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
FEATURE_SETS = {"amplitude only": [0], "phase only": [1], "amplitude + phase": [0, 1]}
GROUPS = [1, 2, 3, 4, 5]


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def features(X, channels):
    parts = []
    for c in channels:
        x = X[:, c]
        parts += [x.mean(axis=2), x.std(axis=2), x.max(axis=2)]
    return np.concatenate(parts, axis=1)


class Data:
    def __init__(self, folder, name, channels):
        z = np.load(Path(folder) / f"{name}.npz")
        self.F, self.y, self.g, self.lost = features(z["X"], channels), z["y"], z["group"], z["lost"]


def per_group(pred, d):
    return np.array([100 * np.mean(pred[d.g == k] == d.y[d.g == k]) for k in GROUPS])


def logo(train, test):
    """Leave-one-recording-out: train on the groups != k of `train`, predict group k of `test`."""
    pred = np.zeros_like(test.y)
    for k in GROUPS:
        m = make_model().fit(train.F[train.g != k], train.y[train.g != k])
        pred[test.g == k] = m.predict(test.F[test.g == k])
    return pred


def cross(train, test):
    return make_model().fit(train.F, train.y).predict(test.F)


def fmt(a):
    return f"{a.mean():5.1f} +-{a.std():4.1f}"


def paired(a, ref):
    d = a - ref
    return f"{d.mean():+5.1f}   worse in {int((d < 0).sum())}/{len(d)} groups"


def row(label, acc, ref=None):
    extra = "" if ref is None else "   paired diff " + paired(acc, ref)
    print(f"  {label:<44}{fmt(acc)}{extra}")


def run(fname, channels, folder):
    print(f"\n{'=' * 8} {fname} {'=' * 8}")
    D = {n: Data(folder, n, channels) for n in
         ("LOS_clean", "LOS_nlostiming", "NLOS_linear", "NLOS_nearest")}
    lc, lt, nl, nn = D["LOS_clean"], D["LOS_nlostiming"], D["NLOS_linear"], D["NLOS_nearest"]

    print("Test 1 - timing only, same room (train on LOS, leave one recording out)")
    a_cc = per_group(logo(lc, lc), lc)
    row("train clean -> test clean (reference)", a_cc)
    row("train clean -> test with NLOS timing", per_group(logo(lc, lt), lt), a_cc)
    row("train NLOS timing -> test clean", per_group(logo(lt, lc), lc), a_cc)
    row("train NLOS timing -> test NLOS timing", per_group(logo(lt, lt), lt), a_cc)

    print("Test 2 - does matching the timing close the cross-domain gap?")
    a1 = per_group(cross(lc, nl), nl)
    row("LOS_clean -> NLOS (reference)", a1)
    row("LOS with NLOS timing -> NLOS", per_group(cross(lt, nl), nl), a1)
    a2 = per_group(cross(nl, lc), lc)
    row("NLOS -> LOS_clean (reference)", a2)
    row("NLOS -> LOS with NLOS timing", per_group(cross(nl, lt), lt), a2)

    print("Test 3 - NLOS interpolation method")
    a3 = per_group(logo(nl, nl), nl)
    row("NLOS linear -> NLOS linear (in-domain)", a3)
    row("NLOS nearest -> NLOS nearest (in-domain)", per_group(logo(nn, nn), nn), a3)
    row("LOS_clean -> NLOS nearest", per_group(cross(lc, nn), nn), a1)
    row("NLOS nearest -> LOS_clean", per_group(cross(nn, lc), lc), a2)

    print("Test 4 - LOS_clean-trained model on NLOS_linear windows, by packets lost per window")
    pred = cross(lc, nl)
    rank = np.zeros(len(nl.y), dtype=int)               # 0 = fewest lost, 2 = most, within class
    for c in np.unique(nl.y):
        idx = np.where(nl.y == c)[0]
        order = np.argsort(nl.lost[idx], kind="stable")
        rank[idx[order]] = np.minimum(np.arange(len(idx)) * 3 // len(idx), 2)
    for r, name in enumerate(["fewest lost (1/3)", "middle (1/3)", "most lost (1/3)"]):
        m = rank == r
        print(f"  {name:<22}lost per window {nl.lost[m].mean():6.1f}   "
              f"accuracy {100 * np.mean(pred[m] == nl.y[m]):5.1f} %   "
              f"empty-vs-moving {100 * np.mean((nl.y[m] == 0) == (pred[m] == 0)):5.1f} %   "
              f"({m.sum()} windows)")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python step14_timing_test.py <folder with the 4 .npz files>")
    for fname, chans in FEATURE_SETS.items():
        run(fname, chans, sys.argv[1])
    print("\nReading guide: differences smaller than about one SD (and 'worse in 2-3 of 5 groups') "
          "are noise.")


if __name__ == "__main__":
    main()