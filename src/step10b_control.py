"""
step10b_control.py - repeat the shuffled-label control run of the CNN many times.

A single shuffled-label run is noisy: the network learns a random label for each
cluster of similar windows, and a lucky draw can score well above 33 %. Repeating it with
different shuffles shows what the control really gives on average.

Run on Colab, in the same folder as step10_cnn.py:
    !python step10b_control.py /content/LOS_PIFA_cut12.npz
"""
import contextlib
import io

import numpy as np

import step10_cnn as c

N_SHUFFLES = 10
CONFIGS = ["amplitude + phase", "amplitude only"]

z = np.load(c.DATASET)
X, y, g = z["X"], z["y"], z["group"]
expected = 100 * np.sum((np.bincount(y) / len(y)) ** 2)
print(f"{len(y)} windows. Random guessing with these class sizes: {expected:.1f} %\n")

for name in CONFIGS:
    means = []
    for s in range(N_SHUFFLES):
        c.SEED = 100 + s                       # a different shuffle and different network start each time
        with contextlib.redirect_stdout(io.StringIO()):   # hide the per-group lines
            accs, _ = c.run_cv(X, y, g, c.CHANNEL_SETS[name], shuffle_labels=True)
        means.append(100 * accs.mean())
        print(f"  {name:<18} shuffle {s + 1:>2}: {means[-1]:.1f} %")
    print(f"{name}: mean {np.mean(means):.1f} %, SD {np.std(means):.1f}, "
          f"range {min(means):.1f} - {max(means):.1f} %\n")
print("Real (unshuffled) CNN results for comparison: amplitude only 72.2 %, amplitude + phase 67.6 %")