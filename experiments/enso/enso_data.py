"""Monthly ERSSTv5 loader and BSO dataset builder for the ENSO experiment.

The repository ships the monthly ERSSTv5 field ``sst.mnmean.nc``. This module
turns it into (a) area-weighted anomalies with the standard Niño-3.4 readout and
chronological splits, and (b) a ``grid2d`` dataset dict consumable by
``edmd_bso_method.run_bso_pipeline`` (the BNO), plus a fixed weighted-EOF
dictionary so the BSO operator dimension is exactly ``M`` leading EOFs.

The land/ice grid points are given zero area weight, so they never enter the
Galerkin inner product or the reported skill.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


NINO_BOUNDS = {
    "nino34": (-5.0, 5.0, 190.0, 240.0),
    "nino3": (-5.0, 5.0, 210.0, 270.0),
    "nino4": (-5.0, 5.0, 160.0, 210.0),
}
BASE_YEARS = (1981, 2010)
DEV_END_DATE = "1998-01-01"
VAL_START_DATE = "1978-01-01"
PRIMARY_END_DATE = "2018-01-01"


def _monthly_anomalies(raw, months, base_mask):
    raw = np.asarray(raw, dtype=np.float64)
    climatology = np.empty((12,) + raw.shape[1:], dtype=np.float64)
    for month in range(1, 13):
        values = raw[base_mask & (months == month)]
        count = np.sum(np.isfinite(values), axis=0)
        climatology[month - 1] = np.divide(
            np.nansum(values, axis=0), count,
            out=np.full(values.shape[1:], np.nan), where=count > 0)
    return raw - climatology[months - 1], climatology


def _region_mask(lat, lon, bounds):
    lat_lo, lat_hi, lon_lo, lon_hi = bounds
    return ((lat[:, None] >= lat_lo) & (lat[:, None] <= lat_hi)
            & (lon[None, :] >= lon_lo) & (lon[None, :] <= lon_hi))


def area_mean(field_flat, weights_flat, mask_flat):
    w = np.asarray(weights_flat) * np.asarray(mask_flat)
    total = w.sum()
    if total <= 0:
        raise ValueError("Niño region has no valid ocean cells")
    return np.asarray(field_flat) @ (w / total)


@dataclass
class ENSOData:
    times: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    shape: tuple[int, int]
    anomalies: np.ndarray        # (T, H*W) area-normalized anomalies (land=0)
    weights: np.ndarray          # (H*W,) area weights, land=0, sum=1
    fno_grid: np.ndarray         # (H, W, 2) normalized coords
    basis_coords: np.ndarray     # (H*W, 2)
    nino_masks: dict             # name -> (H*W,) bool
    scale: float
    dev_end: int
    val_start: int
    primary_end: int


def load_ersstv5(path: str | Path) -> ENSOData:
    ds = xr.open_dataset(path)
    sst = ds["sst"].sortby("lat").sortby("lon")
    sst = sst.sel(time=slice("1870-01-01", None), lat=slice(-30.0, 20.0),
                  lon=slice(28.0, 290.0)).transpose("time", "lat", "lon")
    times = sst.time.values.astype("datetime64[ns]")
    lat = sst.lat.values.astype(float)
    lon = sst.lon.values.astype(float)
    raw = sst.values.astype(np.float64)
    H, W = raw.shape[1], raw.shape[2]
    months = (times.astype("datetime64[M]").astype(int) % 12) + 1
    years = times.astype("datetime64[Y]").astype(int) + 1970
    base = (years >= BASE_YEARS[0]) & (years <= BASE_YEARS[1])

    def _first(date):
        hits = np.where(times >= np.datetime64(date))[0]
        if not len(hits):
            raise ValueError(f"ERSSTv5 record does not reach {date}")
        return int(hits[0])

    dev_end = _first(DEV_END_DATE)
    val_start = _first(VAL_START_DATE)
    primary_end = _first(PRIMARY_END_DATE)
    anomalies, _ = _monthly_anomalies(raw, months, base)
    ocean_mask = np.all(np.isfinite(raw[:dev_end]), axis=0)          # (H, W)
    anomalies = np.where(ocean_mask[None], anomalies, 0.0)
    weights = np.cos(np.deg2rad(lat))[:, None] * ocean_mask
    weights = weights / weights.sum()
    flat_anom = anomalies.reshape(len(anomalies), -1)
    scale = float(np.sqrt(np.mean(
        flat_anom[:dev_end] ** 2 @ weights.ravel())))
    flat_anom = flat_anom / max(scale, 1e-8)

    gx, gy = np.meshgrid((lon - lon.min()) / (np.ptp(lon) + 1e-12),
                         (lat - lat.min()) / (np.ptp(lat) + 1e-12))
    fno_grid = np.stack([gy, gx], axis=-1)                          # (H, W, 2)
    basis_coords = fno_grid.reshape(-1, 2)
    nino_masks = {name: (_region_mask(lat, lon, b).ravel()
                         & (weights.ravel() > 0))
                  for name, b in NINO_BOUNDS.items()}
    if not (0 < val_start < dev_end < primary_end < len(times)):
        raise AssertionError("ENSO chronological split is malformed")
    return ENSOData(times, lat, lon, (H, W), flat_anom.astype(np.float64),
                    weights.ravel().astype(np.float64), fno_grid, basis_coords,
                    nino_masks, scale, dev_end, val_start, primary_end)


def nino_index(data: ENSOData, fields_flat: np.ndarray, name="nino34"):
    """Niño index (physical units) from area-normalized anomaly fields."""
    return area_mean(np.asarray(fields_flat) * data.scale,
                     data.weights, data.nino_masks[name])


@dataclass
class EOFBasis:
    patterns: np.ndarray         # (n_modes, H*W) area-orthonormal EOF patterns
    weights: np.ndarray

    def fixed_columns(self) -> np.ndarray:
        """Dictionary columns (p, n_modes-1) for run_bso_pipeline fixed_basis.

        The leading constant is added by ``build_basis``; supply the remaining
        n_modes-1 EOF patterns as fixed dictionary functions.
        """
        return self.patterns[1:].T.copy() if self.patterns.shape[0] > 1 \
            else self.patterns.T.copy()


def fit_eofs(data: ENSOData, fit_end: int, n_modes: int) -> EOFBasis:
    """Weighted EOFs of the anomalies over ``[0, fit_end)`` (development record)."""
    w = data.weights
    mask = w > 0
    flat = data.anomalies[:fit_end]
    sqrt_w = np.sqrt(w[mask])
    mean = flat[:, mask].mean(axis=0)
    _, _, vt = np.linalg.svd((flat[:, mask] - mean) * sqrt_w[None],
                             full_matrices=False)
    n_modes = min(int(n_modes), vt.shape[0])
    patterns = np.zeros((n_modes, w.shape[0]))
    patterns[:, mask] = vt[:n_modes] / sqrt_w[None]
    return EOFBasis(patterns, w)


def build_grid_dataset(data: ENSOData, start: int, end: int,
                       horizon: int = 1) -> dict:
    """A ``grid2d`` BSO dataset of consecutive-month anomaly pairs in [start,end)."""
    H, W = data.shape
    stop = min(int(end), len(data.anomalies) - 2)
    ends = np.arange(int(start), stop)
    fields = data.anomalies.reshape(len(data.anomalies), H, W)
    return dict(
        system="enso", kind="grid2d", T=float(horizon),
        u0=fields[ends].copy(), uT=fields[ends + 1].copy(),
        u2T=fields[ends + 2].copy(),
        fno_grid=data.fno_grid.copy(),
        basis_coords=data.basis_coords.copy(),
        weights=data.weights.copy(),
        meta=dict(shape=(H, W), scale=data.scale))
