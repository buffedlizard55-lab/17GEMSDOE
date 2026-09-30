#!/usr/bin/env python3
"""Verify the competition rasters against the official SHA-256 pins, then describe them.

The three official files are large and were transported into the sibling
repositories as chunks. Before any of them is used, this script reassembles (if
needed), hashes, and compares against the pins recorded in the sibling projects'
own data-verification records:

    training_features.tif   4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5
    labels.tif              7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093
    sample_submission.tif   2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc

The first of those hashes is also recorded inside the feature-metadata file that
the sibling project published alongside a 105-channel derived stack
(`features_sha256`), which makes it independently corroborated rather than
merely copied.

Refusing to proceed on a hash mismatch is the point of the script. A raster that
is one byte different is a different input, and every number produced downstream
would be unattributable.

Writes ``evidence/data_manifest.json``.
"""

from __future__ import annotations

import argparse
import json

import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]

OFFICIAL = {
    "training_features.tif": {
        "sha256": "4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5",
        "bytes": 418_912_844,
        "role": "19-band predictor stack",
    },
    "labels.tif": {
        "sha256": "7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093",
        "bytes": 425_830,
        "role": "USGS/INGENIOUS known-fault catalogue; 1 = fault, -1 = outside footprint",
    },
    "sample_submission.tif": {
        "sha256": "2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc",
        "bytes": 1_599_597,
        "role": "the required grid, CRS, transform and nodata convention",
    },
}

BAND_DESCRIPTIONS_PREFIX = (
    "mag_anom", "rtp", "tmi_hg", "geod_2ndinv", "iso_grav_anom_slope", "tc",
    "geod_shearrate", "geod_dilaterate", "tmi_vg", "deq_n100a15",
    "iso_grav_anom_vg", "det_elev", "iso_grav_anom", "tmi", "depth_to_base_surf",
    "ieq_n100a15", "cond_surf", "iso_grav_anom_hg", "det_elev_slope",
)

# Bands whose *description in the file itself* claims a property the band does
# not have, or is ambiguous. Recorded here so the claim is auditable.
DISCREPANCIES = [
    "Band 6 is described as 'Tilt angle or total curvature' -- ambiguous between "
    "two different quantities, so this project refers to it as 'tc' and computes "
    "its own tilt angle from tmi rather than trusting the label.",
    "The problem description advertises a 'top-of-crustal magnetic source depth "
    "estimate'. No delivered band matches it: band 15 is depth_to_base_surf, the "
    "thickness of sedimentary cover, which is a different quantity.",
    "There is no radiometric (gamma-ray) band anywhere in the stack, although the "
    "GeoDAWN release publishes airborne radiometric grids for the same survey.",
    "Bands 10 and 16 (deq_n100a15, ieq_n100a15) are smoothed with a 100 km radius, "
    "so they cannot localise structure at the 1-5 km scale of a single fault.",
]


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def catalogue_components(labels: np.ndarray, min_px: int = 20) -> dict:
    """Connectivity facts about the catalogue.

    Computed here rather than quoted, because several structural ideas in this
    project turn out to depend on how fragmented the catalogue is and that number
    was not written down anywhere until it was measured. An earlier revision of
    the site said "3,199 components" by hand; it now reads this.
    """
    comp, n = ndimage.label(labels, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return {"components_8conn": 0, "components_ge_20px": 0,
                "components_ge_30px": 0, "largest_component_px": 0,
                "median_component_px": 0.0}
    sizes = np.bincount(comp.ravel())[1:]
    return {
        "components_8conn": int(n),
        "components_ge_20px": int((sizes >= min_px).sum()),
        "components_ge_30px": int((sizes >= 30).sum()),
        "largest_component_px": int(sizes.max()),
        "median_component_px": float(np.median(sizes)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--out", default=str(REPO / "evidence" / "data_manifest.json"))
    args = ap.parse_args()
    data = Path(args.data_dir)

    manifest: dict = {
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {},
        "verification": {},
        "band_descriptions": [],
        "band_discrepancies": DISCREPANCIES,
    }

    ok = True
    for name, spec in OFFICIAL.items():
        p = data / name
        # the sibling transport names the label file existing_faults.tif
        alt = data / "existing_faults.tif"
        if not p.exists() and name == "labels.tif" and alt.exists():
            p = alt
        if not p.exists():
            print(f"MISSING {name}: expected at {p}")
            manifest["verification"][name] = {"present": False}
            ok = False
            continue
        got = sha256_file(p)
        size = p.stat().st_size
        match = (got == spec["sha256"])
        ok &= match
        manifest["verification"][name] = {
            "present": True, "path": str(p), "bytes": size, "sha256": got,
            "expected_sha256": spec["sha256"], "match": bool(match),
            "bytes_match": size == spec["bytes"],
        }
        print(f"{'OK  ' if match else 'FAIL'} {name:24s} {size:>12,} bytes  sha256={got[:24]}...")

    if not ok:
        print("\nREFUSING to describe the rasters: at least one official hash did "
              "not match. Every downstream number would be unattributable.")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=1))
        return 1

    print("\nAll three files verified against the official pins.")

    feats = data / "training_features.tif"
    with rasterio.open(feats) as ds:
        manifest["grid"] = {
            "shape": [ds.height, ds.width], "crs": str(ds.crs),
            "transform": list(ds.transform)[:6], "count": ds.count,
            "dtype": ds.dtypes[0], "nodata": ds.nodata,
            "cell_size_m": ds.transform.a,
        }
        manifest["band_descriptions"] = [d or "" for d in ds.descriptions]
    print(f"Grid: {manifest['grid']['shape']} {manifest['grid']['crs']} "
          f"{manifest['grid']['cell_size_m']} m, {manifest['grid']['count']} bands")

    labels = data / "labels.tif"
    if not labels.exists():
        labels = data / "existing_faults.tif"
    with rasterio.open(labels) as ds:
        lab = ds.read(1)
    values = {int(v): int((lab == v).sum()) for v in np.unique(lab)}
    manifest["label_values"] = values
    manifest["label_positive_px"] = int((lab == 1).sum())
    print(f"Labels: {values}")

    # The catalogue's own connectivity, measured rather than assumed. Recorded
    # because several structural ideas in this project turned out to depend on
    # how fragmented the catalogue is, and that number was not written down
    # anywhere until this script computed it.
    c = manifest["catalogue"] = catalogue_components(lab == 1)
    print(f"Catalogue: {c['components_8conn']:,} components, median "
          f"{c['median_component_px']:.0f} px, largest {c['largest_component_px']} px")

    with rasterio.open(data / "sample_submission.tif") as ds:
        sample = ds.read(1)
    manifest["sample_submission"] = {
        "finite_px": int(np.isfinite(sample).sum()),
        "nan_px": int((~np.isfinite(sample)).sum()),
        "nodata": None if ds.nodata is None or np.isnan(ds.nodata) else float(ds.nodata),
        "min": float(np.nanmin(sample)), "max": float(np.nanmax(sample)),
    }

    # cells where all 19 bands carry data -- the real working footprint
    with rasterio.open(feats) as ds:
        valid = np.ones(ds.shape, bool)
        for b in range(1, ds.count + 1):
            a = ds.read(b).astype(np.float32)
            valid &= np.isfinite(a) & (np.abs(a - np.float32(ds.nodata)) >= 1e6)
    manifest["valid_all_bands_px"] = int(valid.sum())
    print(f"Valid footprint (all 19 bands): {manifest['valid_all_bands_px']:,} cells; "
          f"sample submission finite: {manifest['sample_submission']['finite_px']:,}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
