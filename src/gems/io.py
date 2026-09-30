"""Raster I/O, the submission format gate, and the fix for the upload error.

The operator reported this rejection from the DrivenData submission form:

    "Predicted values must be in range [0, 1]"

This module exists so that failure mode cannot recur, and it is deliberately
paranoid because the error message names only one of several ways a GeoTIFF can
fail. Everything the competition's data page states about the required format is
checked here, in one function, with a machine-readable result:

* exactly one band
* ``float32`` dtype
* EPSG:32611
* the same width, height and geotransform as the sample submission
* finite values inside ``[0, 1]`` -- the reported failure
* NaN permitted only where the sample submission itself is NaN (outside the
  data footprint); NaN *inside* the footprint is a rejection even though it is
  not out of range
* values written as ``nodata`` must not appear inside the footprint

The last two are the ones a naive range check misses: a file can have every
finite value inside ``[0, 1]`` and still be rejected, or be silently scored as
though the holes were real predictions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import rasterio

EXPECTED_CRS = "EPSG:32611"
EXPECTED_DTYPE = "float32"


# ---------------------------------------------------------------------------
def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sample_footprint(sample_path: str | Path) -> np.ndarray:
    """Boolean array that is True where the sample submission has finite values."""
    with rasterio.open(sample_path) as ds:
        return np.isfinite(ds.read(1))


# ---------------------------------------------------------------------------
def validate_submission(path: str | Path, sample_path: str | Path) -> dict:
    """Run every format requirement. Returns a report; never raises on bad input.

    The report's ``checks`` dict maps a check name to ``{"ok": bool, "detail": str}``
    so a caller can print a table, and ``errors`` is a flat list of the failures
    with the exact numbers that caused them.
    """
    path, sample_path = Path(path), Path(sample_path)
    report: dict = {"path": str(path), "sample": str(sample_path),
                    "checks": {}, "errors": [], "ok": False}

    def check(name: str, ok: bool, detail: str) -> bool:
        report["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            report["errors"].append(f"{name}: {detail}" if detail else name)
        return bool(ok)

    if not path.exists():
        check("file_exists", False, f"not found: {path}")
        return report
    check("file_exists", True, f"{path.stat().st_size:,} bytes")

    try:
        with rasterio.open(path) as ds, rasterio.open(sample_path) as sd:
            arr = ds.read(1, masked=False)
            meta = {
                "count": ds.count, "dtype": ds.dtypes[0], "crs": str(ds.crs),
                "shape": (ds.height, ds.width), "transform": ds.transform,
                "nodata": ds.nodata,
            }
            smeta = {"shape": (sd.height, sd.width), "transform": sd.transform}
            sample_finite = np.isfinite(sd.read(1))
    except Exception as exc:                                   # pragma: no cover
        check("readable_geotiff", False, f"{type(exc).__name__}: {exc}")
        return report
    check("readable_geotiff", True, "opened by rasterio")

    check("single_band", meta["count"] == 1, f"file={meta['count']} required=1")
    check("dtype_float32", meta["dtype"] == EXPECTED_DTYPE,
          f"file={meta['dtype']} required={EXPECTED_DTYPE}")
    check("crs_epsg_32611", meta["crs"] == EXPECTED_CRS,
          f"file={meta['crs']} required={EXPECTED_CRS}")
    check("shape_matches_sample", meta["shape"] == smeta["shape"],
          f"file={meta['shape']} required={smeta['shape']}")
    check("transform_matches_sample", meta["transform"] == smeta["transform"],
          f"file={meta['transform']} required={smeta['transform']}")

    finite = np.isfinite(arr)
    vals = arr[finite]
    if vals.size == 0:
        check("values_in_0_1", False, "no finite values at all")
        return report

    lo, hi = float(vals.min()), float(vals.max())
    n_low = int((vals < 0.0).sum())
    n_high = int((vals > 1.0).sum())
    check("values_in_0_1", (n_low == 0 and n_high == 0),
          f"file=[{lo:.6g}, {hi:.6g}] required=[0, 1]; "
          f"below_0={n_low:,} above_1={n_high:,}")
    report["min"], report["max"] = lo, hi
    report["finite_px"] = int(finite.sum())
    report["nan_px"] = int((~finite).sum())

    if finite.shape == sample_finite.shape:
        n_nan_in = int((~finite & sample_finite).sum())
        check("nan_only_outside_footprint", n_nan_in == 0,
              f"{n_nan_in:,} NaN pixels lie inside the data footprint")
        report["nan_inside_footprint"] = n_nan_in
        # A nodata sentinel inside the footprint would be read by some tooling as
        # a missing value rather than a prediction.
        if meta["nodata"] is not None and np.isfinite(meta["nodata"]):
            n_sent = int((finite & (arr == np.float32(meta["nodata"]))).sum())
            check("nodata_sentinel_absent_inside_footprint", n_sent == 0,
                  f"{n_sent:,} pixels equal the declared nodata value "
                  f"({meta['nodata']}) inside the footprint")
            report["nodata_sentinel_px"] = n_sent
    else:
        check("nan_only_outside_footprint", False,
              "shape differs from the sample; cannot compare footprints")

    # A prediction that emits nothing scores zero, which is legal but almost
    # certainly a bug rather than a choice.
    emitted = int((vals > 0).sum())
    report["emitted_px"] = emitted
    check("non_empty_prediction", emitted > 0, f"{emitted:,} pixels above zero")

    report["ok"] = not report["errors"]
    return report


# ---------------------------------------------------------------------------
def write_submission(path: str | Path, field: np.ndarray,
                     sample_path: str | Path, compress: str = "deflate") -> dict:
    """Write ``field`` as a competition-legal single-band float32 GeoTIFF.

    The array is made safe rather than trusted:

    * non-finite values are replaced by the value the sample submission uses at
      that cell (NaN outside the footprint, 0.0 inside it if the field is
      missing there), so a hole in the detector output cannot become a hole in
      the submission;
    * values are clipped into ``[0, 1]``, which is what the upload form checks;
    * the profile is copied from the sample so the CRS, transform and size cannot
      drift.

    Returns the validation report for the file that was just written, so a
    caller can refuse to publish a failing artifact.
    """
    field = np.asarray(field, dtype=np.float32)
    with rasterio.open(sample_path) as sd:
        ref = sd.read(1)
        profile = sd.profile.copy()
        sample_finite = np.isfinite(ref)
    if field.shape != ref.shape:
        raise ValueError(f"field shape {field.shape} != sample shape {ref.shape}")

    n_bad = int((~np.isfinite(field)).sum())
    fixed = np.where(np.isfinite(field), field, np.where(sample_finite, 0.0, np.nan))
    n_clip = int(((fixed > 1.0) | (fixed < 0.0)).sum())
    fixed = np.clip(fixed, 0.0, 1.0).astype(np.float32)

    profile.update(driver="GTiff", count=1, dtype="float32", nodata=float("nan"),
                   compress=compress)
    profile.pop("tiled", None)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as ds:
        ds.write(fixed, 1)

    rep = validate_submission(out, sample_path)
    rep["repairs"] = {"nonfinite_replaced": n_bad, "clipped_into_range": n_clip}
    rep["sha256"] = sha256_file(out)
    rep["bytes"] = out.stat().st_size
    return rep
