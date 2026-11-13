"""
step12_build_ablation.py - Step 12a: build one dataset per "DSP step switched off".

Every variant runs the SAME pipeline as step8_build_dataset.py, except that
one DSP step is turned off. Comparing the accuracy of the variants later shows
how much each DSP step really contributes.

    full               everything on (must reproduce the dataset from Step 8)
    no_hampel          no outlier removal
    no_resample        packets are used as they arrive (no uniform 100 Hz grid)
    no_phase_sanitize  raw wrapped phase instead of the line-removed phase
    no_lowpass         no Butterworth low-pass
    no_frame_mean      no mean removal inside every STFT frame
    raw_all            all five steps off at once (the "no DSP" reference)

For LOS and NLOS (PIFA antenna) -> 14 files in data/processed/ablation/.
Runs on your laptop (numpy + scipy only). Files that already exist are skipped,
so you can stop and restart.

    python src/step12_build_ablation.py                  # everything
    python src/step12_build_ablation.py --only full,no_hampel
"""
import contextlib
import io
import sys
import time

import numpy as np
from scipy import signal

import step8_build_dataset as b
from csi_io import load_csi, LLTF_VALID
from dsp import hampel_chunked, resample_uniform, butter_lowpass, sanitize_phase

CACHE_DIR = b.OUT / "cache"                  # parsed CSVs from Step 8 are reused
ABL_DIR = b.OUT / "ablation"
SCENARIOS = ["LOS", "NLOS"]

# which DSP steps are ON in each variant
ALL_ON = dict(hampel=True, resample=True, sanitize=True, lowpass=True, frame_mean=True)
VARIANTS = {
    "full": {},
    "no_hampel": dict(hampel=False),
    "no_resample": dict(resample=False),
    "no_phase_sanitize": dict(sanitize=False),
    "no_lowpass": dict(lowpass=False),
    "no_frame_mean": dict(frame_mean=False),
    "raw_all": dict(hampel=False, resample=False, sanitize=False, lowpass=False,
                    frame_mean=False),
}
ON = dict(ALL_ON)          # the switches currently in use (set per variant)


def load_cached(path):
    """Same as Step 8: parse a CSV once and reuse it (shares the Step 8 cache)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"{b.SCENARIO}_{b.SYSTEM}_{path.stem}.npz"
    if f.exists():
        z = np.load(f)
        return z["H"], z["ts"], int(z["label"])
    d = load_csi(path)
    labels = np.unique(d["label"])
    assert len(labels) == 1, f"{path.name}: expected one class label, got {labels}"
    H = d["csi"][:, LLTF_VALID]
    np.savez(f, H=H, ts=d["timestamp_us"], label=labels[0])
    return H, d["timestamp_us"], int(labels[0])


def chain(x, timestamps):
    """Hampel -> uniform grid -> low-pass, each one only if switched on."""
    if ON["hampel"]:
        x = hampel_chunked(x, 15, 4.0)
    if ON["resample"]:
        _, x = resample_uniform(timestamps, x, b.FS)
    if ON["lowpass"] and b.CUTOFF:
        x = butter_lowpass(x, b.CUTOFF, b.FS, b.ORDER)
    return x


def process_file(path, expected_label):
    H, ts, label = load_cached(path)
    assert label == expected_label, f"{path.name}: label {label}, expected {expected_label}"
    if ON["sanitize"]:
        phase, _, _ = sanitize_phase(H, LLTF_VALID)
    else:
        phase = np.angle(H).astype(np.float32)          # raw, wrapped, with hardware errors
    return chain(np.abs(H), ts), chain(phase, ts)


def stft_power(x):
    """Same STFT as dsp.stft_power, but the per-frame mean removal can be switched off."""
    _, _, Z = signal.stft(x, fs=b.FS, window="hann", nperseg=b.NPERSEG,
                          noverlap=b.NPERSEG - b.HOP, nfft=b.NFFT,
                          detrend="constant" if ON["frame_mean"] else False,
                          boundary=None, padded=False, axis=0)
    return np.mean(np.abs(Z) ** 2, axis=1)


def windows(A, P, start, stop, keep):
    n, step = int(b.WIN_S * b.FS), int(b.STRIDE_S * b.FS)
    out = []
    for s in range(start, stop - n + 1, step):
        chans = [10 * np.log10(stft_power(x[s:s + n])[keep] + 1e-10) for x in (A, P)]
        out.append(np.stack(chans))
    return out


# plug our versions into the Step 8 builder (it reads these names when it runs)
b.load_cached, b.chain, b.process_file, b.windows = load_cached, chain, process_file, windows


def main():
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
        bad = [v for v in only if v not in VARIANTS]
        if bad:
            raise SystemExit(f"Unknown variant(s) {bad}. Choose from {list(VARIANTS)}")
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for name, change in VARIANTS.items():
        if only and name not in only:
            continue
        ON.update(ALL_ON)
        ON.update(change)
        for scen in SCENARIOS:
            target = ABL_DIR / f"{name}_{scen}.npz"
            if target.exists():
                print(f"{target.name:<34} already there, skipped")
                continue
            b.SCENARIO = scen
            b.DATA = b.ROOT / "data" / "raw" / "wallhack1.8k" / scen / b.SYSTEM
            if not b.DATA.exists():
                raise SystemExit(f"Folder not found: {b.DATA}")
            b.OUT = ABL_DIR / "_tmp"
            for old in b.OUT.glob("*.npz") if b.OUT.exists() else []:
                old.unlink()
            with contextlib.redirect_stdout(io.StringIO()):      # keep the screen quiet
                b.main()
            made = list(b.OUT.glob("*.npz"))
            assert len(made) == 1, made
            made[0].replace(target)
            z = np.load(target)
            print(f"{target.name:<34} {len(z['y'])} windows   ({time.time() - t0:5.0f} s)")
    print(f"\nDone. Files are in {ABL_DIR}")


if __name__ == "__main__":
    main()