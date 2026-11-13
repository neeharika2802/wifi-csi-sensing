"""
step9_baseline.py - Step 9: a classical machine-learning baseline.

Features: for every frequency row of the spectrogram we take the mean, the
standard deviation and the maximum over time (so 3 x 39 = 117 numbers per
channel). Two simple classifiers are trained on these numbers.

Evaluation: leave-one-group-out. Each group (1-5) is held out once as the test
set while the model trains on the other four, so test windows always come from
recordings the model has never seen. We report the mean and spread over the
five test groups.
"""
import time
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # draw straight into PNG files, no pop-up window
import matplotlib.pyplot as plt

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as e:   # a Windows-blocked DLL shows up as an ImportError
    raise SystemExit(f"scikit-learn could not be loaded here:\n{e}\n"
                     "Send me this message; we will run this step in Google Colab instead.")

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "processed" / "LOS_PIFA_cut12.npz"   # made by step 8
FIG = ROOT / "results" / "figures"
if len(sys.argv) > 1:   # on Google Colab: python step9_baseline.py /content/LOS_PIFA_cut12.npz
    DATASET = Path(sys.argv[1])
    FIG = DATASET.parent
CLASS_NAMES = ["Empty room", "Walking", "Walking + arm-waving"]

FEATURE_SETS = {                       # which channels of the image to use
    "amplitude only": [0],
    "phase only": [1],
    "amplitude + phase": [0, 1],
}
MODELS = {
    "Logistic regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
    "Random forest": lambda: RandomForestClassifier(
        n_estimators=300, random_state=0, n_jobs=1),
}


def features(X, channels):
    """X: (N, 2, F, T) -> (N, 3*F*len(channels)): mean, std, max over time per row."""
    parts = []
    for c in channels:
        x = X[:, c]                                    # (N, F, T)
        parts += [x.mean(axis=2), x.std(axis=2), x.max(axis=2)]
    return np.concatenate(parts, axis=1)


def leave_one_group_out(model_fn, F, y, g, shuffle_labels=False, seed=0):
    """Train on 4 groups, test on the 5th, for every group. Returns (accuracies, predictions)."""
    rng = np.random.default_rng(seed)
    pred = np.empty_like(y)
    accs = []
    for k in np.unique(g):
        train, test = g != k, g == k
        y_train = rng.permutation(y[train]) if shuffle_labels else y[train]
        model = model_fn().fit(F[train], y_train)
        pred[test] = model.predict(F[test])
        accs.append(np.mean(pred[test] == y[test]))
    return np.array(accs), pred


def confusion(y, pred, n=3):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm


def main():
    if not DATASET.exists():
        raise SystemExit(f"{DATASET} not found. Run step8_build_dataset.py first.")
    z = np.load(DATASET)
    X, y, g = z["X"], z["y"], z["group"]
    groups = np.unique(g)
    t0 = time.time()
    print(f"Loaded {len(y)} windows. Chance level is about {100 / 3:.0f} %"
          f" (always guessing the biggest class gives {100 * np.bincount(y).max() / len(y):.0f} %).\n")

    results = {}
    header = f"{'model':<21}{'features':<19}" + "".join(f"{'grp ' + str(k):>8}" for k in groups)
    print("Test accuracy (%) when each group is held out:")
    print(header + f"{'mean':>8}{'std':>7}")
    for mname, mfn in MODELS.items():
        for fname, chans in FEATURE_SETS.items():
            F = features(X, chans)
            accs, pred = leave_one_group_out(mfn, F, y, g)
            results[(mname, fname)] = (accs, pred)
            print(f"{mname:<21}{fname:<19}" + "".join(f"{100 * a:>8.1f}" for a in accs)
                  + f"{100 * accs.mean():>8.1f}{100 * accs.std():>7.1f}")
    print(f"({time.time() - t0:.0f} s)")

    best = max(results, key=lambda k: results[k][0].mean())
    accs, pred = results[best]
    cm = confusion(y, pred)
    print(f"\nBest: {best[0]} on {best[1]}  (mean {100 * accs.mean():.1f} %)")
    print("\nConfusion matrix, all five test groups together (rows = true class, columns = predicted):")
    print(f"{'':<22}" + "".join(f"{n[:14]:>16}" for n in CLASS_NAMES) + f"{'recall':>9}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<22}" + "".join(f"{cm[i, j]:>16}" for j in range(3))
              + f"{100 * cm[i, i] / cm[i].sum():>8.1f}%")

    side_ok = np.mean((y == 0) == (pred == 0))
    moving = y > 0
    ww_ok = np.mean(pred[moving] == y[moving])
    print(f"\nEmpty room vs moving: {100 * side_ok:.1f} % of windows land on the correct side.")
    print(f"Walking vs walking+waving (windows that truly are one of the two): "
          f"{100 * ww_ok:.1f} % get the exact right label.")

    F_best = features(X, FEATURE_SETS[best[1]])
    ctrl, _ = leave_one_group_out(MODELS[best[0]], F_best, y, g, shuffle_labels=True)
    p_cls = np.bincount(y) / len(y)
    expected = 100 * np.sum(p_cls ** 2)       # accuracy of random guessing with these class sizes
    print(f"\nControl: same model, but training labels shuffled -> {100 * ctrl.mean():.1f} % "
          f"(random guessing is expected to give about {expected:.0f} %; "
          f"if the control is much higher, something leaks).")

    FIG.mkdir(parents=True, exist_ok=True)
    pct = 100 * cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{pct[i, j]:.0f} %\n({cm[i, j]})", ha="center", va="center",
                    color="white" if pct[i, j] > 55 else "black")
    ax.set_xticks(range(3), [n.replace(" + ", "\n+ ") for n in CLASS_NAMES])
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{best[0]}, {best[1]}\nleave-one-group-out, mean accuracy {100 * accs.mean():.1f} %")
    fig.colorbar(im, ax=ax, label="% of true class")
    fig.savefig(FIG / "12_baseline_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to {FIG / '12_baseline_confusion.png'}")


if __name__ == "__main__":
    main()