import os
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import h5py  # optional, only needed for raw HDF5 datasets
except ImportError:
    print(f"Import of h5py failed")
    h5py = None


def _as_clean_numeric_pair(q_xy, q_z) -> Tuple[np.ndarray, np.ndarray]:
    """
    Take two 1D sequences, coerce both to numeric, drop rows where either is NaN,
    and return aligned numpy arrays of equal length.
    """
    df = pd.DataFrame({
        "q_xy": pd.to_numeric(pd.Series(q_xy, copy=False), errors="coerce"),
        "q_z":  pd.to_numeric(pd.Series(q_z,  copy=False), errors="coerce"),
    }).dropna()
    if df.empty:
        raise ValueError("No valid numeric q_xy/q_z values after dropping NaNs.")
    arr = df.to_numpy(copy=False)
    return arr[:, 0], arr[:, 1]


def _detect_and_convert_units_to_angstrom_inverse(q_xy: np.ndarray, q_z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Heuristic: if any q value > 6.0, assume nm^-1 and convert to Å^-1 by multiplying by 0.1.
    Otherwise, assume already in Å^-1.
    Returns (q_xy_Å^-1, q_z_Å^-1, unit_string_used).
    """
    unit = "Å⁻¹"
    if (np.nanmax(q_xy) > 6.0) or (np.nanmax(q_z) > 6.0):
        # nm^-1 → Å^-1
        q_xy = q_xy * 0.1
        q_z  = q_z  * 0.1
        unit = "nm⁻¹→Å⁻¹ (×0.1)"
    return q_xy, q_z, unit


def _load_from_path(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load q_xy and q_z from a file path that may be .npy/.npz, .csv, or .h5/.hdf5.
    Expects either a 2-column table/array or named columns/datasets 'q_xy' and 'q_z'.
    """
    ext = os.path.splitext(path)[1].lower()

    # NumPy: .npy or .npz
    if ext in (".npy", ".npz"):
        data = np.load(path, allow_pickle=False)
        if isinstance(data, np.lib.npyio.NpzFile):
            # Prefer explicit names if present
            keys = set(data.files)
            if {"q_xy", "q_z"} <= keys:
                q_xy, q_z = data["q_xy"], data["q_z"]
            else:
                # Fallback: find a single 2D array
                if len(keys) == 1:
                    arr = data[data.files[0]]
                else:
                    # try a conventional name
                    for k in ("arr", "data", "peaks", "peak_list"):
                        if k in keys:
                            arr = data[k]
                            break
                    else:
                        raise ValueError(f"NPZ file {path} must contain 'q_xy' and 'q_z' or a single 2D array.")
                if arr.ndim != 2 or arr.shape[1] < 2:
                    raise ValueError("NPZ 2D array must have at least two columns for q_xy and q_z.")
                q_xy, q_z = arr[:, 0], arr[:, 1]
        else:
            # .npy – expect either 2-column array or dict-like with named fields
            if isinstance(data, np.ndarray):
                if data.ndim == 2 and data.shape[1] >= 2:
                    q_xy, q_z = data[:, 0], data[:, 1]
                else:
                    raise ValueError(".npy must be a 2D array with ≥2 columns (q_xy, q_z).")
            else:
                raise ValueError("Unsupported .npy content; expected a 2D numpy array.")
        return _as_clean_numeric_pair(q_xy, q_z)

    # CSV
    if ext == ".csv":
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        if "q_xy" in cols and "q_z" in cols:
            q_xy_raw = df[cols["q_xy"]]
            q_z_raw  = df[cols["q_z"]]
        else:
            # fallback: use first two columns
            if df.shape[1] < 2:
                raise ValueError("CSV must have columns 'q_xy' and 'q_z' or at least two columns.")
            q_xy_raw = df.iloc[:, 0]
            q_z_raw  = df.iloc[:, 1]
        return _as_clean_numeric_pair(q_xy_raw, q_z_raw)

    # HDF5: try pandas first, then raw h5py
    if ext in (".h5", ".hdf5"):
        # Try pandas HDFStore layout (tables/frames saved by pandas)
        try:
            df = pd.read_hdf(path)
            if isinstance(df, pd.DataFrame):
                cols = {c.lower(): c for c in df.columns}
                if "q_xy" in cols and "q_z" in cols:
                    return _as_clean_numeric_pair(df[cols["q_xy"]], df[cols["q_z"]])
                elif df.shape[1] >= 2:
                    return _as_clean_numeric_pair(df.iloc[:, 0], df.iloc[:, 1])
        except Exception:
            pass  # fall through to raw h5py

        if h5py is None:
            raise ImportError("h5py is required to read raw HDF5 files that are not pandas stores.")

        # Explore datasets to find q_xy and q_z or a 2-column dataset
        with h5py.File(path, "r") as f:
            # Look for named datasets first
            def _find_dataset(name_options):
                for name in name_options:
                    if name in f:
                        obj = f[name]
                        if isinstance(obj, h5py.Dataset):
                            return np.array(obj)
                return None

            q_xy = _find_dataset(["q_xy", "Q_XY", "qxy", "/q_xy"])
            q_z  = _find_dataset(["q_z",  "Q_Z",  "qz",  "/q_z"])

            if q_xy is not None and q_z is not None:
                return _as_clean_numeric_pair(q_xy, q_z)

            # Otherwise, search for a single 2D dataset anywhere in the file
            print(f"Could not find columns with q_xy and q_z names, nor looking for 2D dataset anywhere")
            candidate = None
            def _is_2d_two_col(ds):
                return isinstance(ds, h5py.Dataset) and ds.ndim == 2 and ds.shape[1] >= 2

            for _, obj in f.items():
                if _is_2d_two_col(obj):
                    candidate = np.array(obj)
                    break
                # Recursively check groups
                if isinstance(obj, h5py.Group):
                    for _, obj2 in obj.items():
                        if _is_2d_two_col(obj2):
                            candidate = np.array(obj2)
                            break
                if candidate is not None:
                    break

            if candidate is None:
                raise ValueError("HDF5 must contain datasets 'q_xy' and 'q_z' or a 2D dataset with ≥2 columns.")
            return _as_clean_numeric_pair(candidate[:, 0], candidate[:, 1])

    raise ValueError(f"Unsupported file extension: {ext}. Use .npy/.npz, .csv, or .h5/.hdf5.")

def load_peaks_qxy_qz(
    sample_data: Union[str, np.ndarray, pd.DataFrame],
    feature_config,
    q_scale: Optional[float] = None,
    *,
    max_fill: float = 0.97,   # leave a few percent margin from the border
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load (q_xy, q_z) and scale them for model input.

    If q_scale is None, compute one shared scale from the shorter q axis so it
    fills the model domain. Points from the longer axis may then exceed the
    model bounds and are removed by the caller's existing domain filter.

    Returns float32 arrays and the q_scale used.
    """
    # --- 1) Load raw q_xy / q_z as numeric arrays ---
    if isinstance(sample_data, np.ndarray):
        if sample_data.ndim != 2 or sample_data.shape[1] < 2:
            raise ValueError("NumPy array must be 2D with ≥2 columns (q_xy, q_z).")
        q_xy_raw, q_z_raw = _as_clean_numeric_pair(sample_data[:, 0], sample_data[:, 1])

    elif isinstance(sample_data, pd.DataFrame):
        cols = {c.lower(): c for c in sample_data.columns}
        if "q_xy" in cols and "q_z" in cols:
            q_xy_raw, q_z_raw = _as_clean_numeric_pair(sample_data[cols["q_xy"]], sample_data[cols["q_z"]])
        else:
            if sample_data.shape[1] < 2:
                raise ValueError("DataFrame must have columns 'q_xy' and 'q_z' or at least two columns.")
            q_xy_raw, q_z_raw = _as_clean_numeric_pair(sample_data.iloc[:, 0], sample_data.iloc[:, 1])

    elif isinstance(sample_data, (str, os.PathLike)):
        q_xy_raw, q_z_raw = _load_from_path(os.fspath(sample_data))

    else:
        raise TypeError("sample_data must be a file path, a 2D numpy array, or a pandas DataFrame.")

    # --- 2) Unit normalization to Å^-1 ---
    q_xy_ang, q_z_ang, _ = _detect_and_convert_units_to_angstrom_inverse(q_xy_raw, q_z_raw)

    # --- 3) Determine scaling (with margin) ---
    if not (0.0 < max_fill < 1.0):
        raise ValueError("max_fill must be in (0, 1). For a few-percent margin use e.g. 0.95–0.99.")

    # Get config limits
    max_q_xy = float(feature_config.max_q_xy)
    max_q_z = float(feature_config.max_q_z)
    
    if q_scale is None:
        # Scale from the shorter axis. This preserves a square reciprocal-space
        # field of view and leaves longer-axis overflow for domain filtering.
        max_data_q_xy = float(np.nanmax(q_xy_ang))
        max_data_q_z = float(np.nanmax(q_z_ang))
        
        if max_data_q_xy <= 0.0 or max_data_q_z <= 0.0 or \
           not np.isfinite(max_data_q_xy) or not np.isfinite(max_data_q_z):
            q_scale_used = 1.0
        else:
            q_scale_xy = max_data_q_xy / max_q_xy
            q_scale_z = max_data_q_z / max_q_z
            q_scale_used = min(q_scale_xy, q_scale_z)
    else:
        q_scale_used = float(q_scale)

    # --- 4) Apply the shared scale. Out-of-domain values are deliberately
    # preserved here so the workflow can discard them instead of clipping them
    # into the image border.
    q_xy_scaled = (q_xy_ang / q_scale_used).astype(np.float32, copy=False)
    q_z_scaled  = (q_z_ang  / q_scale_used).astype(np.float32, copy=False)

    return q_xy_scaled, q_z_scaled, q_scale_used
