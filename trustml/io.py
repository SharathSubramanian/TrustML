"""Dataset loading. The original file is never modified; we record its hash."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from trustml.schemas import DatasetHandle

SUPPORTED = {".csv", ".xlsx", ".xls"}
MAX_ROWS = 200_000  # documented MVP limit


class DatasetError(ValueError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(path: str | Path, sheet: str | int | None = 0) -> tuple[pd.DataFrame, DatasetHandle]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise DatasetError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED)}")

    if ext == ".csv":
        df = pd.read_csv(path)
        sheet_name = None
    else:
        df = pd.read_excel(path, sheet_name=sheet)
        sheet_name = str(sheet)

    if df.empty:
        raise DatasetError("Dataset has no rows.")
    if len(df) > MAX_ROWS:
        raise DatasetError(f"Dataset has {len(df):,} rows; MVP limit is {MAX_ROWS:,}.")
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise DatasetError(f"Duplicate column names: {dupes}")

    df.columns = [str(c) for c in df.columns]
    handle = DatasetHandle(path=str(path), sha256=_sha256(path), n_rows=len(df), n_cols=df.shape[1], sheet=sheet_name)
    return df, handle


# ---------------------------------------------------------------- prediction files
# CSV is kept for humans; an .npz sidecar stores the exact float64 values. CSV parsing is not
# guaranteed to round-trip floats (pandas 3 deprecates float_precision), and tied scores can then
# split, changing ROC-AUC in the 6th decimal. All code reads predictions through load_predictions().
def save_predictions(df: pd.DataFrame, csv_path: Path) -> None:
    import numpy as np
    df.to_csv(csv_path, index=False)
    np.savez(Path(csv_path).with_suffix(".npz"), **{c: df[c].to_numpy() for c in df.columns})


def load_predictions(csv_path: Path) -> pd.DataFrame:
    import numpy as np
    npz = Path(csv_path).with_suffix(".npz")
    if npz.exists():
        with np.load(npz, allow_pickle=False) as z:
            return pd.DataFrame({k: z[k] for k in z.files})
    return pd.read_csv(csv_path)
