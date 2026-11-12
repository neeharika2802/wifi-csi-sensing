"""
csi_io.py - read Wallhack1.8k raw CSI CSV files into NumPy arrays.

Only uses the Python standard library + NumPy (no pandas needed).
"""
import csv
from pathlib import Path

import numpy as np

# Some CSV rows are very long, so lift the default csv field size limit.
csv.field_size_limit(2**31 - 1)

FS_NOMINAL = 100.0   # packets per second (stated in the dataset documentation)
N_RAW = 256          # numbers per packet = 128 subcarriers x (imag, real)

# Valid subcarrier indices (from the dataset's Zenodo page)
LLTF_VALID = list(range(6, 32)) + list(range(33, 59))      # 52 subcarriers
HTLTF_VALID = list(range(66, 94)) + list(range(95, 123))   # 56 subcarriers


def parse_csi_string(s):
    """Turn the text '[0,0,6,-1,...]' into a complex vector of length 128.

    The numbers are stored as [I, R, I, R, ...]: imaginary part first,
    then the real part, for each subcarrier.
    Returns None if the row is malformed or has the wrong length.
    """
    body = s.strip().strip("[]").strip().strip(",")
    try:
        nums = np.array([int(x) for x in body.split(",")], dtype=np.float32)
    except ValueError:
        return None
    if nums.size != N_RAW:
        return None
    imag = nums[0::2]
    real = nums[1::2]
    return (real + 1j * imag).astype(np.complex64)


def load_csi(path, max_rows=None):
    """Load one Wallhack1.8k CSV file.

    Returns a dict with:
      csi          (N, 128) complex64   complex CSI per packet
      timestamp_us (N,)     int64       ESP32 local timestamp in microseconds
      rssi         (N,)     int16
      label        (N,)     int8        0 = empty, 1 = walking, 2 = walking + arm-waving
      skipped      int                  number of malformed rows that were dropped
    """
    csi, ts, rssi, label = [], [], [], []
    skipped = 0
    with open(Path(path), newline="") as f:
        for row in csv.DictReader(f):
            h = parse_csi_string(row["data"])
            if h is None:
                skipped += 1
                continue
            csi.append(h)
            ts.append(int(row["local_timestamp"]))
            rssi.append(int(row["rssi"]))
            label.append(int(row["class"]))
            if max_rows is not None and len(csi) >= max_rows:
                break
    return {
        "csi": np.array(csi, dtype=np.complex64),
        "timestamp_us": np.array(ts, dtype=np.int64),
        "rssi": np.array(rssi, dtype=np.int16),
        "label": np.array(label, dtype=np.int8),
        "skipped": skipped,
    }


def amplitude(csi, idx=LLTF_VALID):
    """Amplitude |H| of the chosen subcarriers. Shape (N, len(idx))."""
    return np.abs(csi[:, idx])


def packet_intervals_s(timestamp_us):
    """Time between consecutive packets in seconds.

    The ESP32 timestamp is a 32-bit microsecond counter, so it wraps around
    after about 71.6 minutes; a negative difference means a wrap happened.
    """
    dt = np.diff(timestamp_us).astype(np.float64)
    dt[dt < 0] += 2**32
    return dt * 1e-6