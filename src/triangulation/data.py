"""
GMR file loader for 10-second mid-price candle data.

File format (discovered empirically):
  Header: 9 bytes
    - bytes 0-3: version (big-endian int32, = 1)
    - bytes 4-7: row count (big-endian int32)
    - byte  8:   reserved (= 1)
  Row: 56 bytes
    - bytes  0-7:  timestamp (big-endian int64, milliseconds since Unix epoch)
    - bytes  8-15: open  (big-endian float64)
    - bytes 16-23: high  (big-endian float64)
    - bytes 24-31: low   (big-endian float64)
    - bytes 32-39: close (big-endian float64)
    - bytes 40-47: bid   (bid price at bar close — confirmed via (bid+ask)/2 ≈ close)
    - bytes 48-55: ask   (ask price at bar close)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd


_HEADER_SIZE = 9
_ROW_SIZE = 56
_ROW_FMT = ">q6d"  # big-endian: int64 timestamp + 6x float64


def load_gmr(path: str | Path) -> pd.DataFrame:
    """Load a single .gmr file. Returns DataFrame with columns:
    timestamp (UTC, DatetimeTZDtype), open, high, low, close at 10s resolution.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        raw = fh.read()

    n_rows = struct.unpack_from(">I", raw, offset=4)[0]

    # Use numpy structured dtype for efficient bulk parse (big-endian)
    row_dtype = np.dtype([
        ("ts",    ">i8"),
        ("open",  ">f8"),
        ("high",  ">f8"),
        ("low",   ">f8"),
        ("close", ">f8"),
        ("bid",   ">f8"),
        ("ask",   ">f8"),
    ])
    arr = np.frombuffer(raw, dtype=row_dtype, count=n_rows, offset=_HEADER_SIZE)

    timestamps = pd.to_datetime(arr["ts"], unit="ms", utc=True)

    # Cast to native byte order — pandas operations require little-endian arrays
    df = pd.DataFrame(
        {
            "open":  arr["open"].astype("<f8"),
            "high":  arr["high"].astype("<f8"),
            "low":   arr["low"].astype("<f8"),
            "close": arr["close"].astype("<f8"),
            "bid":   arr["bid"].astype("<f8"),
            "ask":   arr["ask"].astype("<f8"),
        },
        index=timestamps,
    )
    df.index.name = "timestamp"
    return df


def load_pair(data_dir: str | Path, symbol: str) -> pd.DataFrame:
    """Load all monthly .gmr files for a symbol and return native 10s bars.

    Args:
        data_dir: Directory containing tick10s-mid-{SYMBOL}-YYYY-MM.gmr files.
        symbol:   E.g. "EURUSD", "AUDUSD", "EURAUD".

    Returns:
        DataFrame indexed by UTC timestamp with open/high/low/close columns at
        10-second resolution. Duplicate timestamps are dropped (keep first).

    Memory note: ~4.5M rows per pair across the full history (~150 MB per pair).
    All three pairs loaded simultaneously require ~800 MB working set including
    the aligned signal frame and multi-scale features.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(f"tick10s-mid-{symbol}-*.gmr"))
    if not files:
        raise FileNotFoundError(f"No .gmr files found for {symbol} in {data_dir}")

    frames = [load_gmr(f) for f in files]
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="first")]
    out.sort_index(inplace=True)
    return out
