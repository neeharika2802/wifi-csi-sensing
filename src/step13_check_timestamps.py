"""
step13_check_timestamps.py - Step 13: how regular are the packet arrival times?

Reads the timestamps from the cache made in Step 8 (data/processed/cache) for
every recording in LOS and NLOS (PIFA). numpy only. Prints one line per file:

  rate      average packets per second (nominal: 100)
  median    median gap between packets in ms (nominal: 10)
  jitter    SD of the arrival time around a local regular clock, in ms
  lost %    packets missing, estimated from gaps that are multiples of 10 ms
  >15 ms    number of gaps longer than 15 ms
  max gap   longest gap in ms
  span 400  how many seconds 400 consecutive packets really cover (ideal 4.000)

    python src/step13_check_timestamps.py
"""
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent.parent / "data" / "processed" / "cache"
FILES = ["b1"] + [f"w{i}" for i in range(1, 6)] + [f"ww{i}" for i in range(1, 6)]


def moving_median(x, width=101):
    pad = width // 2
    xp = np.pad(x, pad, mode="edge")
    w = np.lib.stride_tricks.sliding_window_view(xp, width)
    return np.median(w, axis=1)


print(f"{'file':<14}{'packets':>8}{'rate Hz':>9}{'median ms':>10}{'jitter ms':>10}"
      f"{'lost %':>8}{'>15 ms':>8}{'max gap':>9}{'span 400 (s)':>14}")
rates = []
for scen in ("LOS", "NLOS"):
    for name in FILES:
        f = CACHE / f"{scen}_PIFA_{name}.npz"
        if not f.exists():
            print(f"{scen + '/' + name:<14} (no cache file - run Step 8 / Step 11a first)")
            continue
        ts = np.load(f)["ts"].astype(np.int64)

        step = np.diff(ts)
        step[step < 0] += 2 ** 32

        t = np.concatenate([[0.0], np.cumsum(step)]) / 1e6
        dt = np.diff(t) * 1e3
        n = len(ts)
        rate = (n - 1) / t[-1]
        k = np.arange(n)
        p = np.polyfit(k, t, 1)
        res = (t - np.polyval(p, k)) * 1e3
        jitter = np.std(res - moving_median(res))
        lost = np.round(dt / 10.0 - 1).clip(min=0).sum()
        span = t[400:] - t[:-400]
        rates.append(rate)
        print(f"{scen + '/' + name:<14}{n:>8}{rate:>9.2f}{np.median(dt):>10.2f}{jitter:>10.2f}"
              f"{100 * lost / (n + lost):>8.2f}{(dt > 15).sum():>8}{dt.max():>9.1f}"
              f"{np.median(span):>8.3f} ({span.min():.2f}-{span.max():.2f})")
if rates:
    print(f"\nRate over all files: {min(rates):.2f} - {max(rates):.2f} Hz (assumed 100 Hz when "
          f"timestamps are ignored)")