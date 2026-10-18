"""Build the official SST forecasting dataset from public sources.

Sources
-------
* ERA5 / WeatherBench2 (anonymous GCS): daily sea-surface temperature and
  wind stress derived from 10 m U/V wind.
* NOAA GODAS (public NCSS): monthly SSH, T(55/105 m), 5 m salinity and
  currents, mixed-layer depth, and total downward surface heat flux.  A
  monthly mean is exposed to the daily model only from the first day of the
  *following* month, then forward-filled.  This causal one-month availability
  rule prevents future-within-month leakage.

The daily SST target is built from the same anonymous WeatherBench2 archive.
Intermediate source files are cached below ``data/sst_raw`` so interrupted
preparation can resume without repeating completed transfers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from urllib.parse import urlencode

_CPU_WORKERS = max(1, min(8, (os.cpu_count() or 4) // 2))
for _thread_env in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

import dask
import numcodecs
import numpy as np
import pandas as pd
import requests
import xarray as xr
from dask.diagnostics import ProgressBar

try:
    numcodecs.blosc.set_nthreads(1)
except AttributeError:
    pass


ERA5_REMOTE = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
)
from project_paths import PROJECT_ROOT

DEFAULT_SST = PROJECT_ROOT / "data" / "era5_64x32_sst_daily_1993_2022.zarr"
DEFAULT_OUT = PROJECT_ROOT / "data" / "sst_multivar_64x32_daily_1993_2022.zarr"
DEFAULT_RAW = PROJECT_ROOT / "data" / "sst_raw"
START = "1993-01-01"
END = "2022-12-31"
GODAS_NCSS = "https://psl.noaa.gov/thredds/ncss/grid/Datasets/godas"
GODAS_PREPROCESSING_VERSION = "godas-ocean-state-causal-month-lag-v3"
FINAL_PREPROCESSING_VERSION = "sst-data-v6-1993-2022"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst", type=Path, default=DEFAULT_SST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild derived Zarr stores. Raw NetCDF downloads are retained.")
    return parser.parse_args()


def _compressor():
    return numcodecs.Blosc(
        cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)


def _zarr_encoding(ds, time_chunk=128):
    compressor = _compressor()
    encoding = {}
    for name, da in ds.data_vars.items():
        chunks = tuple(
            min(int(time_chunk), int(ds.sizes[dim])) if dim == "time"
            else int(ds.sizes[dim])
            for dim in da.dims)
        encoding[name] = {"chunks": chunks, "compressor": compressor}
    return encoding


def _safe_rmtree(path, parent):
    path = Path(path).resolve()
    parent = Path(parent).resolve()
    if path == parent or parent not in path.parents:
        raise ValueError(f"refusing to remove unsafe path: {path}")
    if path.exists():
        shutil.rmtree(path)


def _write_zarr(ds, path, *, force=False):
    path = Path(path)
    if path.exists() and not force:
        print(f"[data] reuse {path}")
        return xr.open_zarr(path, consolidated=True)
    partial = path.with_name(path.name + ".partial")
    _safe_rmtree(partial, path.parent)
    partial.parent.mkdir(parents=True, exist_ok=True)
    ds = ds.chunk({
        dim: min(128, int(size)) if dim == "time" else int(size)
        for dim, size in ds.sizes.items()
    })
    print(f"[data] writing {partial}")
    with dask.config.set(scheduler="threads", num_workers=_CPU_WORKERS):
        with ProgressBar():
            ds.to_zarr(
                partial, mode="w", consolidated=True,
                encoding=_zarr_encoding(ds),
                zarr_version=2)
    if path.exists():
        _safe_rmtree(path, path.parent)
    partial.replace(path)
    return xr.open_zarr(path, consolidated=True)


def _download(url, path, attempts=4):
    path = Path(path)
    url_digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    signature = path.with_suffix(path.suffix + ".url.sha256")
    if path.exists() and path.stat().st_size > 1024:
        cached_digest = (
            signature.read_text(encoding="ascii").strip()
            if signature.exists() else "")
        if cached_digest == url_digest:
            return path
        print(f"[download] query changed; refreshing {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    signature_partial = signature.with_suffix(signature.suffix + ".partial")
    for attempt in range(1, int(attempts) + 1):
        try:
            print(f"[download] {path.name} (attempt {attempt})")
            with requests.get(
                    url, stream=True, timeout=(30, 1800),
                    headers={"User-Agent": "neural-bso-data-prep/1.0"}) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" in content_type:
                    raise RuntimeError(
                        f"server returned HTML instead of NetCDF for {url}")
                with partial.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
            if partial.stat().st_size <= 1024:
                raise RuntimeError(f"download is unexpectedly small: {partial}")
            partial.replace(path)
            signature_partial.write_text(url_digest + "\n", encoding="ascii")
            signature_partial.replace(signature)
            return path
        except Exception:
            if partial.exists():
                partial.unlink()
            if signature_partial.exists():
                signature_partial.unlink()
            if attempt == int(attempts):
                raise
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def _target_base(sst_path):
    ds = xr.open_zarr(sst_path, consolidated=True)
    sst = ds["sst"]
    rename = {}
    if "latitude" in sst.dims:
        rename["latitude"] = "lat"
    if "longitude" in sst.dims:
        rename["longitude"] = "lon"
    sst = sst.rename(rename).transpose("time", "lat", "lon")
    lsm = ds["land_sea_mask"].rename(rename).transpose("lat", "lon")
    times = pd.DatetimeIndex(pd.to_datetime(sst.time.values)).normalize()
    expected = pd.date_range(START, END, freq="D")
    if not times.equals(expected):
        raise ValueError(
            f"SST record must contain every day over {START}..{END}; "
            f"received {times[0]}..{times[-1]} with {len(times)} days")
    return xr.Dataset(
        {"sst": sst.assign_coords(time=times), "land_sea_mask": lsm})


def prepare_era5_sst(sst_path, force=False):
    """Build the complete daily SST target and static land mask."""
    sst_path = Path(sst_path)
    if sst_path.exists() and not force:
        return _target_base(sst_path)
    print(f"[era5] preparing daily SST from {ERA5_REMOTE}")
    remote = xr.open_zarr(
        ERA5_REMOTE, storage_options={"token": "anon"}, consolidated=True)
    selection = slice(f"{START}T00:00:00", f"{END}T18:00:00")
    sst = (
        remote["sea_surface_temperature"]
        .sel(time=selection)
        .resample(time="1D")
        .mean()
        .astype("float32"))
    lsm = remote["land_sea_mask"]
    if "time" in lsm.dims:
        lsm = lsm.isel(time=0, drop=True)
    prepared = xr.Dataset({
        "sst": sst,
        "land_sea_mask": lsm.astype("float32"),
    })
    prepared["sst"].attrs.update(
        long_name="Sea surface temperature",
        short_name="sst",
        source_variable="sea_surface_temperature",
        units="K")
    prepared["land_sea_mask"].attrs.update(
        long_name="Land-sea mask",
        short_name="lsm",
        note="WeatherBench2 ERA5 land fraction; ocean is land_sea_mask < 0.5.",
        units="1")
    prepared.attrs.update(
        source=ERA5_REMOTE,
        temporal="daily",
        target_period=[START, END],
        resolution="64x32")
    _write_zarr(prepared, sst_path, force=force)
    return _target_base(sst_path)


def _align_era5_grid(ds, target):
    """Put WeatherBench2 fields on the target's explicit (time, lat, lon) axes.

    The source Zarr stores its variables as (time, longitude, latitude), while
    the rest of this preparation pipeline uses the conventional latitude-first
    ordering.  Coordinate equality is checked before assigning the target
    coordinate objects, so this cannot silently re-label a different grid.
    """
    rename = {}
    if "latitude" in ds.dims:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename["longitude"] = "lon"
    aligned = ds.rename(rename).transpose("time", "lat", "lon")
    if not (
            np.allclose(aligned.lat.values, target.lat.values)
            and np.allclose(aligned.lon.values, target.lon.values)):
        raise ValueError("ERA5 wind-stress grid does not match the SST target grid")
    return aligned.assign_coords(lat=target.lat.values, lon=target.lon.values)


def prepare_era5_wind_stress(target, raw_dir, force=False):
    out = raw_dir / "era5_daily_wind_stress_1993_2022.zarr"
    if out.exists() and not force:
        return _align_era5_grid(
            xr.open_zarr(out, consolidated=True), target)
    print(f"[era5] opening anonymous WeatherBench2 store {ERA5_REMOTE}")
    remote = xr.open_zarr(
        ERA5_REMOTE, storage_options={"token": "anon"}, consolidated=True)
    selection = slice(f"{START}T00:00:00", f"{END}T18:00:00")
    u10 = remote["10m_u_component_of_wind"].sel(time=selection)
    v10 = remote["10m_v_component_of_wind"].sel(time=selection)
    speed = np.hypot(u10, v10)
    rho_air, drag = 1.225, 1.3e-3
    tau_x = (rho_air * drag * speed * u10).resample(time="1D").mean()
    tau_y = (rho_air * drag * speed * v10).resample(time="1D").mean()
    ds = xr.Dataset({
        "wind_stress_x": tau_x.astype("float32"),
        "wind_stress_y": tau_y.astype("float32"),
    })
    ds["wind_stress_x"].attrs.update(
        long_name="Eastward wind stress derived from ERA5 10 m wind",
        units="N m**-2", formula="rho_air * C_D * |U10| * U10",
        rho_air_kg_m3=rho_air, drag_coefficient=drag)
    ds["wind_stress_y"].attrs.update(
        long_name="Northward wind stress derived from ERA5 10 m wind",
        units="N m**-2", formula="rho_air * C_D * |U10| * V10",
        rho_air_kg_m3=rho_air, drag_coefficient=drag)
    ds.attrs.update(
        source=ERA5_REMOTE, cadence="daily mean of 6-hourly stress",
        access="anonymous")
    return _align_era5_grid(
        _write_zarr(ds, out, force=force), target)


def _open_loaded(path):
    with xr.open_dataset(path) as ds:
        return ds.load()


def _normalise_ocean_piece(ds):
    for vertical_dim in ("depth", "level"):
        if vertical_dim in ds.dims and ds.sizes[vertical_dim] == 1:
            ds = ds.isel({vertical_dim: 0}, drop=True)
    times = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    ds = ds.assign_coords(time=times)
    lon = np.mod(ds.lon.values.astype(np.float64), 360.0)
    ds = ds.assign_coords(lon=lon).sortby("lon").sortby("lat")
    _, unique = np.unique(ds.time.values, return_index=True)
    return ds.isel(time=np.sort(unique))


def _interp_periodic(da, target):
    da = da.sortby("lon").sortby("lat")
    left = da.isel(lon=-1, drop=True).expand_dims(
        lon=[float(da.lon.values[-1]) - 360.0])
    right = da.isel(lon=0, drop=True).expand_dims(
        lon=[float(da.lon.values[0]) + 360.0])
    extended = xr.concat([left, da, right], dim="lon")
    return extended.interp(
        lat=target.lat.values, lon=target.lon.values, method="linear"
    ).transpose("time", "lat", "lon")


def _godas_url(variable, year, start, end, depth=None):
    query = [
        ("var", variable),
        ("north", "65"),
        ("west", "-179"),
        ("east", "179"),
        ("south", "-74"),
        ("horizStride", "6"),
        ("time_start", f"{start}T00:00:00Z"),
        ("time_end", f"{end}T00:00:00Z"),
        ("accept", "netcdf4"),
    ]
    if depth is not None:
        query.insert(-1, ("vertCoord", str(float(depth))))
    return (
        f"{GODAS_NCSS}/{variable}.{int(year)}.nc?"
        f"{urlencode(query)}")


def prepare_godas(target, raw_dir, force=False):
    out = raw_dir / "godas_causal_daily_state_1993_2022.zarr"
    if out.exists() and not force:
        cached = xr.open_zarr(out, consolidated=True)
        if cached.attrs.get("preprocessing_version") == GODAS_PREPROCESSING_VERSION:
            return cached
        print(f"[godas] preprocessing changed; rebuilding {out}")
        force = True
    specs = (
        ("ssh", "sshg", None),
        ("temp_55m", "pottmp", 55.0),
        ("temp_105m", "pottmp", 105.0),
        ("current_5m_u", "ucur", 5.0),
        ("current_5m_v", "vcur", 5.0),
        ("mld", "dbss_obml", None),
        ("salinity_5m", "salt", 5.0),
        ("q_net", "thflx", None),
    )
    by_variable = {}
    for output_name, variable, depth in specs:
        pieces = []
        first_source_year = pd.Timestamp(START).year - 1
        final_source_year = pd.Timestamp(END).year
        for year in range(first_source_year, final_source_year + 1):
            start = (
                f"{year}-12-01"
                if year == first_source_year else f"{year}-01-01")
            end = f"{year}-12-01"
            depth_tag = "" if depth is None else f"_{int(depth)}m"
            path = raw_dir / "godas" / (
                f"{output_name}_{year}{depth_tag}.nc")
            url = _godas_url(variable, year, start, end, depth=depth)
            pieces.append(_normalise_ocean_piece(
                _open_loaded(_download(url, path)))[variable])
        monthly = xr.concat(pieces, dim="time").sortby("time")
        by_variable[output_name] = _interp_periodic(monthly, target)

    monthly = xr.Dataset(by_variable)

    def _converted_attrs(attrs, scale=1.0, offset=0.0):
        converted = dict(attrs)
        for key in ("actual_range", "valid_range"):
            if key in converted:
                values = np.asarray(converted[key], dtype=np.float64)
                converted[key] = (scale * values + offset).astype(np.float32)
        return converted

    # Normalize from the source metadata because some mirrors use
    # centimetre-based SSH/current units) and fail closed on an unexpected
    # convention rather than silently mixing scales across channels.
    ssh_units = str(monthly["ssh"].attrs.get("units", "")).lower().replace(" ", "")
    if ssh_units.startswith("cm"):
        original = _converted_attrs(monthly["ssh"].attrs, scale=0.01)
        monthly["ssh"] = 0.01 * monthly["ssh"]
        monthly["ssh"].attrs.update(original)
        monthly["ssh"].attrs.update(units="m", source_units=original.get("units", "cm"))
    elif ssh_units not in ("m", "meter", "meters"):
        raise ValueError(f"unexpected GODAS sshg units: {ssh_units!r}")
    for name in ("current_5m_u", "current_5m_v"):
        units = str(monthly[name].attrs.get("units", "")).lower().replace(" ", "")
        if units.startswith("cm"):
            original = _converted_attrs(monthly[name].attrs, scale=0.01)
            monthly[name] = 0.01 * monthly[name]
            monthly[name].attrs.update(original)
            monthly[name].attrs.update(
                units="m s**-1", source_units=original.get("units", "cm/s"))
        elif not (units.startswith("m/") or units.startswith("ms")):
            raise ValueError(f"unexpected GODAS current units for {name}: {units!r}")
    for name in ("temp_55m", "temp_105m"):
        units = str(monthly[name].attrs.get("units", "")).lower().replace(" ", "")
        if units in ("k", "kelvin"):
            original = _converted_attrs(
                monthly[name].attrs, offset=-273.15)
            monthly[name] = monthly[name] - 273.15
            monthly[name].attrs.update(original)
            monthly[name].attrs.update(
                units="degC", source_units=original.get("units", "K"))
        elif units not in ("degc", "c", "celsius", "degree_celsius"):
            raise ValueError(
                f"unexpected GODAS temperature units for {name}: {units!r}")
    salinity_units = str(
        monthly["salinity_5m"].attrs.get("units", "")).lower().replace(" ", "")
    if salinity_units in ("kg/kg", "kgkg-1", "1"):
        original = _converted_attrs(
            monthly["salinity_5m"].attrs, scale=1000.0)
        monthly["salinity_5m"] = 1000.0 * monthly["salinity_5m"]
        monthly["salinity_5m"].attrs.update(original)
        monthly["salinity_5m"].attrs.update(
            units="g kg**-1",
            source_units=original.get("units", "kg/kg"))
    elif not (
            salinity_units.startswith("g/") or
            salinity_units.startswith("gkg") or
            salinity_units in ("psu", "1e-3")):
        raise ValueError(
            f"unexpected GODAS salinity units: {salinity_units!r}")
    # GODAS stamps a monthly mean on the first day of that month.  Make it
    # available only on the next month boundary, then hold the last completed
    # monthly analysis constant until a newer one becomes available.
    source_months = (
        pd.DatetimeIndex(pd.to_datetime(monthly.time.values))
        .to_period("M")
        .to_timestamp(how="start"))
    available = (source_months + pd.offsets.MonthBegin(1)).normalize()
    monthly = monthly.assign_coords(time=available)
    target_times = pd.DatetimeIndex(pd.to_datetime(target.time.values)).normalize()
    source_index = available.get_indexer(target_times, method="ffill")
    if np.any(source_index < 0):
        first_missing = target_times[int(np.flatnonzero(source_index < 0)[0])]
        raise ValueError(
            f"GODAS has no completed month available for {first_missing.date()}")
    # Chunk before expanding the 372 monthly fields to 10,957 daily fields.
    # Fancy indexing then stays lazy and the full ~0.7 GiB ocean state is
    # streamed to Zarr instead of being materialized twice in host memory.
    monthly = monthly.chunk({
        "time": min(12, int(monthly.sizes["time"])),
        "lat": int(monthly.sizes["lat"]),
        "lon": int(monthly.sizes["lon"]),
    })
    daily = (
        monthly
        .isel(time=xr.DataArray(source_index, dims=("time",)))
        .assign_coords(time=target_times))
    for name, da in daily.data_vars.items():
        if not np.all(da.notnull().any(dim=("lat", "lon")).values):
            raise ValueError(
                f"causally aligned GODAS channel '{name}' has an entirely "
                "missing day")
    plausible_ranges = {
        "ssh": (-5.0, 5.0),
        "temp_55m": (-5.0, 45.0),
        "temp_105m": (-5.0, 45.0),
        "current_5m_u": (-5.0, 5.0),
        "current_5m_v": (-5.0, 5.0),
        "mld": (0.0, 10000.0),
        "salinity_5m": (0.0, 50.0),
        "q_net": (-3000.0, 3000.0),
    }
    for name, (lower, upper) in plausible_ranges.items():
        finite_min = float(daily[name].min(skipna=True))
        finite_max = float(daily[name].max(skipna=True))
        if finite_min < lower or finite_max > upper:
            raise ValueError(
                f"GODAS channel '{name}' range [{finite_min}, {finite_max}] "
                f"falls outside [{lower}, {upper}]")
    daily["ssh"].attrs.update(
        long_name="GODAS sea surface height from last completed month")
    daily["temp_55m"].attrs.update(
        long_name=(
            "GODAS potential temperature at 55 m from last completed month"),
        units="degC")
    daily["temp_105m"].attrs.update(
        long_name=(
            "GODAS potential temperature at 105 m from last completed month"),
        units="degC")
    daily["current_5m_u"].attrs.update(
        long_name=(
            "GODAS eastward current at 5 m from last completed month"))
    daily["current_5m_v"].attrs.update(
        long_name=(
            "GODAS northward current at 5 m from last completed month"))
    daily["mld"].attrs.update(
        long_name="GODAS ocean mixed-layer depth from last completed month",
        units="m")
    daily["salinity_5m"].attrs.update(
        long_name="GODAS salinity at 5 m from last completed month",
        units="g kg**-1")
    daily["q_net"].attrs.update(
        long_name=(
            "GODAS total downward surface heat flux from last completed month"),
        units="W m**-2",
        positive="downward")
    daily.attrs.update(
        source="NOAA PSL NCEP GODAS public NCSS",
        source_cadence="monthly mean",
        availability_rule=(
            "month M is first exposed on the first day of month M+1; "
            "daily values are causal forward fills"),
        horizontal_preselection_stride=6,
        requested_depths_m=dict(
            temp_55m=55, temp_105m=105,
            current_5m_u=5, current_5m_v=5, salinity_5m=5),
        preprocessing_version=GODAS_PREPROCESSING_VERSION,
    )
    return _write_zarr(daily, out, force=force)


def build_final(target, era5, godas, out, force=False):
    channel_names = (
        "sst",
        "wind_stress_x",
        "wind_stress_y",
        "ssh",
        "temp_55m",
        "temp_105m",
        "current_5m_u",
        "current_5m_v",
        "mld",
        "salinity_5m",
        "q_net",
    )
    manifest = dict(
        preprocessing_version=FINAL_PREPROCESSING_VERSION,
        target="ERA5 daily sea-surface temperature",
        era5_wind=dict(
            source=ERA5_REMOTE,
            rho_air_kg_m3=1.225,
            drag_coefficient=1.3e-3,
            reduction="daily mean of 6-hourly stress"),
        godas=dict(
            preprocessing_version=GODAS_PREPROCESSING_VERSION,
            variables=[
                "sshg", "pottmp@55m", "pottmp@105m",
                "ucur@5m", "vcur@5m", "dbss_obml",
                "salt@5m", "thflx"],
            source_bbox=dict(
                west=-179, east=179, south=-74, north=65),
            unit_conversions=dict(
                pottmp="K to degC",
                salt="kg/kg to g/kg",
                sshg="validated as m",
                currents="validated as m/s"),
            horizontal_stride=6,
            regrid="periodic linear interpolation to target",
            availability=(
                "monthly mean first exposed at next month boundary, "
                "then causal daily forward-fill")),
        target_period=[START, END],
        target_resolution="64x32",
        channels=list(channel_names),
    )
    manifest_text = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(
        manifest_text.encode("utf-8")).hexdigest()[:20]
    if out.exists() and not force:
        cached = xr.open_zarr(out, consolidated=True)
        if cached.attrs.get("data_fingerprint") == fingerprint:
            print(f"[data] final store already exists: {out}")
            return cached
        print(f"[data] preprocessing manifest changed; rebuilding {out}")
        force = True

    times = target.time.values
    components = []
    for ds in (era5, godas):
        if not np.array_equal(ds.time.values.astype("datetime64[D]"),
                              times.astype("datetime64[D]")):
            raise ValueError("a prepared component is not exactly time-aligned")
        components.append(ds.assign_coords(time=times))
    final = xr.merge([target, *components], join="exact")
    for name in channel_names:
        final[name] = final[name].astype("float32").transpose(
            "time", "lat", "lon")
    final["land_sea_mask"] = final["land_sea_mask"].astype(
        "float32").transpose("lat", "lon")
    final = final.rename({"lat": "latitude", "lon": "longitude"})

    final.attrs.update(
        title="Daily SST forecasting dataset with causal ocean-atmosphere state",
        prediction_target="sst",
        auxiliary_role=(
            "causally available internal state for autonomous forecasts; "
            "future observations must never be supplied during rollout"),
        data_fingerprint=fingerprint,
        preprocessing_version=FINAL_PREPROCESSING_VERSION,
        manifest=manifest_text,
    )
    prepared = _write_zarr(final, out, force=force)
    print(f"[data] complete: {out}")
    print(f"[data] fingerprint: {fingerprint}")
    print(f"[data] variables: {list(prepared.data_vars)}")
    print(f"[data] sizes: {dict(prepared.sizes)}")
    return prepared


def main():
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    target = prepare_era5_sst(args.sst, force=args.force)
    era5 = prepare_era5_wind_stress(target, args.raw_dir, force=args.force)
    godas = prepare_godas(target, args.raw_dir, force=args.force)
    build_final(target, era5, godas, args.out, force=args.force)


if __name__ == "__main__":
    main()
