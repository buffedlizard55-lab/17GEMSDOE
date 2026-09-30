"""Loading the competition rasters, with the band identity asserted, not assumed.

Band names and order were read directly from the band *descriptions* stored in
the delivered GeoTIFF, not from documentation and not from the file name. Two
consequences worth recording:

* Band 3, ``tmi_hg``, is the total-magnetic-intensity horizontal gradient. The
  delivered stack therefore already contains a magnetic edge product, and any
  "new" edge hypothesis must be shown to differ from it (see
  ``evidence/band_notes.json``).
* Band 6 is described in the file itself as *"Tilt angle or total curvature --
  magnetic field derivative for edge detection"*. The description is
  ambiguous between two different quantities, so this repository refers to it as
  ``tc`` and never claims it is definitely the tilt angle. Detectors that want a
  tilt angle compute one from ``tmi``/``rtp`` rather than trusting the label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

# Verified from rasterio band descriptions of the official training_features.tif,
# in file order, 1-based.
BAND_NAMES: tuple[str, ...] = (
    "mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope", "tc",
    "geod_shearrate", "geod_dilaterate", "tmi_vg", "deq_n100a15",
    "iso_grav_anom_vg", "det_elev", "iso_grav_anom", "tmi", "depth_to_base_surf",
    "ieq_n100a15", "cond_surf", "iso_grav_anom_hg", "det_elev_slope",
)
N_BANDS = len(BAND_NAMES)
BAND_INDEX = {name: i + 1 for i, name in enumerate(BAND_NAMES)}

NODATA = -3.4028234663852886e38


def band_descriptions(path: str | Path) -> list[str]:
    with rasterio.open(path) as ds:
        return list(ds.descriptions)


def verify_band_layout(path: str | Path) -> bool:
    """True if the file's band descriptions start with the expected names."""
    descs = band_descriptions(path)
    if len(descs) != N_BANDS:
        return False
    for i, name in enumerate(BAND_NAMES):
        if not (descs[i] or "").startswith(name):
            return False
    return True


def load_band(path: str | Path, name: str) -> np.ndarray:
    """Read one band as float32 with the nodata sentinel turned into NaN."""
    with rasterio.open(path) as ds:
        a = ds.read(BAND_INDEX[name]).astype(np.float32)
    return mask_nodata(a)


def load_bands(path: str | Path, names: list[str]) -> dict[str, np.ndarray]:
    return {n: load_band(path, n) for n in names}


def mask_nodata(a: np.ndarray, nodata: float = NODATA) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).copy()
    a[~np.isfinite(a)] = np.nan
    a[np.abs(a - np.float32(nodata)) < 1e6] = np.nan
    return a


def valid_mask(path: str | Path) -> np.ndarray:
    """Cells where every band carries real data."""
    with rasterio.open(path) as ds:
        n = ds.count
        valid = np.ones(ds.shape, bool)
        for b in range(1, n + 1):
            a = ds.read(b).astype(np.float32)
            valid &= np.isfinite(a) & (np.abs(a - np.float32(NODATA)) >= 1e6)
    return valid


def load_labels(path: str | Path) -> np.ndarray:
    """Boolean fault raster. The delivered file uses -1 for outside-footprint."""
    with rasterio.open(path) as ds:
        a = ds.read(1)
    return a == 1


def load_sample(path: str | Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1)
