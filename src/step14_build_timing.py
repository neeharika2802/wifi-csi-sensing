"""
step14_build_timing.py - Step 14a: isolate the effect of NLOS packet timing / interpolation.

Question: LOS and NLOS differ in the room AND in how cleanly the packets were logged
(NLOS: more lost packets, more jitter, so the linear interpolation in the resampling
step fills many more gaps). Does that timing difference change the features?

The LOS preprocessing is never changed. We build 4 datasets (all with the same DSP chain
as Step 8, except where stated):

  LOS_clean       LOS recordings, normal timing                  (= the Step 8 dataset)
  NLOS_linear     NLOS recordings, normal linear interpolation   (= the Step 11 dataset)
  LOS_nlostiming  LOS CONTENT sampled at the arrival times of the matching NLOS recording
                  (w3 gets the timing of NLOS w3, ...): same room as LOS_clean, but with the
                  packet loss and jitter of NLOS. Differences to LOS_clean are due to timing only.
  NLOS_nearest    NLOS recordings, but the resampling picks the nearest packet instead of
                  interpolating linearly (gaps become flat steps instead of straight lines)

Every window also gets `lost`: the estimated number of missing packets inside the window.

Needs the Step 8 / Step 11a caches (data/processed/cache). numpy + scipy only.
Output: data/processed/timing/*.npz        Run:  python src/step14_build_timing.py
"""
import contextlib
import io
import time

import numpy as np

import step8_build_dataset as b
from csi_io import load_csi, LLTF_VALID
from dsp import hampel_chunked, resample_uniform, butter_lowpass, sanitize_phase

CACHE_DIR = b.OUT / "cache"
OUT_DIR = b.OUT / "timing"
JOBS = [("LOS_clean", "LOS", "clean"), ("NLOS_linear", "NLOS", "clean"),
        ("LOS_nlostiming", "LOS", "transplant"), ("NLOS_nearest", "NLOS", "nearest")]

MODE = "clean"
LOST = []          # one number per window, in the order the windows are created
CUM = None         # cumulative number of lost packets along the 100 Hz grid of the current file


def times_s(ts):
    """Arrival times in seconds from 0, with the 32-bit microsecond counter wrap handled."""
    step = np.diff(np.asarray(ts).astype(np.int64))
    step[step < 0] += 2 ** 32
    return np.concatenate([[0.0], np.cumsum(step)]) * 1e-6


def load_cached(path, scen=None):
    scen = scen or b.SCENARIO
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"{scen}_{b.SYSTEM}_{path.stem}.npz"
    if f.exists():
        z = np.load(f)
        return z["H"], z["ts"], int(z["label"])
    if scen != b.SCENARIO:
        raise SystemExit(f"Missing {f.name}. Run Step 8 (LOS) and Step 11a (NLOS) first.")
    d = load_csi(path)
    labels = np.unique(d["label"])
    assert len(labels) == 1, f"{path.name}: expected one class label, got {labels}"
    H = d["csi"][:, LLTF_VALID]
    np.savez(f, H=H, ts=d["timestamp_us"], label=labels[0])
    return H, d["timestamp_us"], int(labels[0])


def resample_nearest(ts, x, fs=100.0):
    """Like resample_uniform, but every grid point takes the value of the nearest packet."""
    t = times_s(ts)
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, x = t[keep], x[keep]
    tg = np.arange(0.0, t[-1], 1.0 / fs)
    j = np.clip(np.searchsorted(t, tg), 1, len(t) - 1)
    j = np.where(tg - t[j - 1] < t[j] - tg, j - 1, j)
    return tg, x[j]


def chain(x, timestamps):
    x = hampel_chunked(x, 15, 4.0)
    _, x = (resample_nearest if MODE == "nearest" else resample_uniform)(timestamps, x, b.FS)
    if b.CUTOFF:
        x = butter_lowpass(x, b.CUTOFF, b.FS, b.ORDER)
    return x


def interp_rows(t_old, x, t_new):
    return np.stack([np.interp(t_new, t_old, x[:, k]) for k in range(x.shape[1])],
                    axis=1).astype(np.float32)


def process_file(path, expected_label):
    global CUM
    H, ts, label = load_cached(path)
    assert label == expected_label, f"{path.name}: label {label}, expected {expected_label}"
    A = np.abs(H)
    P, _, _ = sanitize_phase(H, LLTF_VALID)
    if MODE == "transplant":                       # LOS content, NLOS arrival times
        t_los = times_s(ts)
        t_new = times_s(load_cached(path, scen="NLOS")[1])
        t_new = t_new[t_new <= t_los[-1]]
        A, P = interp_rows(t_los, A, t_new), interp_rows(t_los, P, t_new)
        ts = np.round(t_new * 1e6).astype(np.int64)
    Ao, Po = chain(A, ts), chain(P, ts)
    # estimated lost packets, placed on the 100 Hz grid (gap length / 10 ms - 1)
    t = times_s(ts)
    lost = np.round(np.diff(t) * 100 - 1).clip(min=0)
    idx = np.minimum((t[1:] * b.FS).astype(int), len(Ao) - 1)
    arr = np.zeros(len(Ao) + 1)
    np.add.at(arr, idx + 1, lost)
    CUM = np.cumsum(arr)
    return Ao, Po


def windows(A, P, start, stop, keep):
    n, step = int(b.WIN_S * b.FS), int(b.STRIDE_S * b.FS)
    out = []
    for s in range(start, stop - n + 1, step):
        chans = []
        for x in (A, P):
            _, _, power = b.stft_power(x[s:s + n], b.FS, b.NPERSEG, b.HOP, b.NFFT)
            chans.append(10 * np.log10(power[keep] + 1e-10))
        out.append(np.stack(chans))
        LOST.append(int(CUM[s + n] - CUM[s]))
    return out


b.load_cached, b.chain, b.process_file, b.windows = load_cached, chain, process_file, windows


def main():
    global MODE
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for name, scen, mode in JOBS:
        target = OUT_DIR / f"{name}.npz"
        if target.exists():
            print(f"{target.name:<22} already there, skipped")
            continue
        MODE = mode
        LOST.clear()
        b.SCENARIO = scen
        b.DATA = b.ROOT / "data" / "raw" / "wallhack1.8k" / scen / b.SYSTEM
        b.OUT = OUT_DIR / "_tmp"
        for old in (b.OUT.glob("*.npz") if b.OUT.exists() else []):
            old.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            b.main()
        made = list(b.OUT.glob("*.npz"))
        assert len(made) == 1, made
        z = dict(np.load(made[0]))
        assert len(LOST) == len(z["y"]), (len(LOST), len(z["y"]))
        z["lost"] = np.array(LOST, dtype=np.int32)
        np.savez_compressed(target, **z)
        made[0].unlink()
        print(f"{target.name:<22} {len(z['y'])} windows, mean lost packets per window "
              f"{z['lost'].mean():.1f}   ({time.time() - t0:.0f} s)")
    print(f"\nDone. Files are in {OUT_DIR}")


if __name__ == "__main__":
    main()