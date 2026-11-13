"""
step10_cnn.py - Step 10: a small CNN on the spectrogram images (run on Google Colab).

Same testing protocol as the baseline: leave-one-group-out. Each group (1-5) is
held out once as the test set; the CNN trains on the other four groups. The
number of training epochs is FIXED in advance (no peeking at the test set to
pick the best epoch), so the test accuracy is honest.

Usage on Colab:
    !python step10_cnn.py /content/LOS_PIFA_cut12.npz --quick   (about 1-2 minutes, checks that everything runs)
    !python step10_cnn.py /content/LOS_PIFA_cut12.npz           (the full experiment)
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
QUICK = "--quick" in sys.argv
DATASET = Path(args[0]) if args else ROOT / "data" / "processed" / "LOS_PIFA_cut12.npz"
FIG = DATASET.parent if args else ROOT / "results" / "figures"
CLASS_NAMES = ["Empty room", "Walking", "Walking + arm-waving"]

# ---- settings you can play with -------------------------------------------
CHANNEL_SETS = {"amplitude only": [0], "phase only": [1], "amplitude + phase": [0, 1]}
EPOCHS = 30          # fixed number of passes over the training data
BATCH = 64
LR = 1e-3
MAX_SHIFT = 4        # training augmentation: shift the image in time by up to +-4 columns
SEED = 0
RUN_CONTROL = True   # also train once with shuffled labels (should give ~chance accuracy)
# ---------------------------------------------------------------------------
if QUICK:
    EPOCHS, RUN_CONTROL = 3, False
    CHANNEL_SETS = {"amplitude + phase": [0, 1]}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_model(in_ch):
    """3 convolution blocks -> global average -> 3 outputs. Works for any image size."""
    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
    return nn.Sequential(
        block(in_ch, 16), nn.MaxPool2d(2),
        block(16, 32), nn.MaxPool2d(2),
        block(32, 64), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 3),
    )


def sanity_check(X):
    """Does the network accept our image size and give 3 outputs?"""
    model = make_model(X.shape[1]).to(DEVICE)
    out = model(torch.zeros(8, X.shape[1], X.shape[2], X.shape[3], device=DEVICE))
    assert tuple(out.shape) == (8, 3), f"unexpected output shape {tuple(out.shape)}"
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Sanity check passed: input (8, {X.shape[1]}, {X.shape[2]}, {X.shape[3]}) -> output (8, 3), "
          f"{n_par:,} parameters, running on {DEVICE}.")


def train_fold(Xtr, ytr, Xte, in_ch, seed):
    """Train on Xtr, return the predicted classes for Xte."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean = Xtr.mean(axis=(0, 2, 3), keepdims=True)       # statistics from the TRAINING data only
    std = Xtr.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    Xtr_t = torch.tensor((Xtr - mean) / std, dtype=torch.float32, device=DEVICE)
    Xte_t = torch.tensor((Xte - mean) / std, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)

    model = make_model(in_ch).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    n = len(Xtr_t)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            shift = int(rng.integers(-MAX_SHIFT, MAX_SHIFT + 1))
            xb = torch.roll(xb, shifts=shift, dims=3)    # shift along the time axis
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(Xte_t).argmax(dim=1).cpu().numpy()


def run_cv(X, y, g, chans, shuffle_labels=False):
    """Leave-one-group-out. Returns (accuracy per group, predictions for every window)."""
    rng = np.random.default_rng(SEED)
    pred = np.empty_like(y)
    accs = []
    for k in np.unique(g):
        train, test = g != k, g == k
        y_train = rng.permutation(y[train]) if shuffle_labels else y[train]
        t0 = time.time()
        pred[test] = train_fold(X[train][:, chans], y_train, X[test][:, chans],
                                len(chans), seed=SEED + int(k))
        accs.append(np.mean(pred[test] == y[test]))
        print(f"    group {k}: {100 * accs[-1]:.1f} %   ({time.time() - t0:.0f} s)")
    return np.array(accs), pred


def confusion(y, pred, n=3):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm


def main():
    if not DATASET.exists():
        raise SystemExit(f"{DATASET} not found.")
    z = np.load(DATASET)
    X, y, g = z["X"], z["y"], z["group"]
    groups = np.unique(g)
    print(f"Loaded {len(y)} windows of shape {X.shape[1:]}.  "
          f"{'QUICK TEST: ' if QUICK else ''}{EPOCHS} epochs per model.")
    sanity_check(X)

    results = {}
    for fname, chans in CHANNEL_SETS.items():
        print(f"\nCNN on {fname}:")
        results[fname] = run_cv(X, y, g, chans)

    print("\nTest accuracy (%) when each group is held out:")
    print(f"{'features':<19}" + "".join(f"{'grp ' + str(k):>8}" for k in groups) + f"{'mean':>8}{'std':>7}")
    for fname, (accs, _) in results.items():
        print(f"{fname:<19}" + "".join(f"{100 * a:>8.1f}" for a in accs)
              + f"{100 * accs.mean():>8.1f}{100 * accs.std():>7.1f}")
    print("(for comparison, the baseline's best was 72.5 %, logistic regression on phase only)")

    best = max(results, key=lambda k: results[k][0].mean())
    accs, pred = results[best]
    cm = confusion(y, pred)
    print(f"\nBest: CNN on {best}  (mean {100 * accs.mean():.1f} %)")
    print("\nConfusion matrix, all test groups together (rows = true class, columns = predicted):")
    print(f"{'':<22}" + "".join(f"{n[:14]:>16}" for n in CLASS_NAMES) + f"{'recall':>9}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<22}" + "".join(f"{cm[i, j]:>16}" for j in range(3))
              + f"{100 * cm[i, i] / cm[i].sum():>8.1f}%")
    moving = y > 0
    print(f"\nEmpty room vs moving: {100 * np.mean((y == 0) == (pred == 0)):.1f} % on the correct side.")
    print(f"Walking vs walking+waving: {100 * np.mean(pred[moving] == y[moving]):.1f} % exact label "
          f"(baseline: 59.9 %).")

    if RUN_CONTROL:
        print("\nControl run (training labels shuffled), amplitude + phase:")
        ctrl, _ = run_cv(X, y, g, CHANNEL_SETS["amplitude + phase"], shuffle_labels=True)
        expected = 100 * np.sum((np.bincount(y) / len(y)) ** 2)
        print(f"Control accuracy {100 * ctrl.mean():.1f} %  (random guessing is about {expected:.0f} %)")

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
    ax.set_title(f"CNN, {best}\nleave-one-group-out, mean accuracy {100 * accs.mean():.1f} %")
    fig.colorbar(im, ax=ax, label="% of true class")
    out = FIG / "13_cnn_confusion.png"
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    main()