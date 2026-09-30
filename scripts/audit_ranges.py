#!/usr/bin/env python3
"""Audit every published GeoTIFF from the group's repositories for format defects.

Motivation: the operator reported this rejection from the DrivenData submission form:

    "Predicted values must be in range [0, 1]"

The correct first move is not to guess which file caused it but to check every file the
group actually published. If none of them is out of range, then the offending file was
never published or has been replaced, and that is itself the finding.

Checks per file: finite-value range, count of values outside [0, 1], whether the field is
binary, dtype, band count, CRS, shape, and whether NaN appears inside the official data
footprint (a separate, subtler failure mode than an out-of-range value).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[1]
ORG = "buffedlizard55-lab"

TARGETS = {
    "GEMSDOE": [
        "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif",
        "data/evidence/baseline/submission.tif",
        "data/evidence/runs/local-sandbox-smoke/submission.tif",
    ],
    "5GEMSDOE": [
        "docs/downloads/candidate_s5_dilational_top10k.tif",
        "docs/downloads/candidate_s5_dilational_top25k.tif",
        "docs/downloads/candidate_s5_dilational_top50k.tif",
        "docs/downloads/candidate_s5_dilational_annulus.tif",
        "docs/downloads/candidate_s5_catalogue_hedge.tif",
        "docs/downloads/candidate_s5d_gdr_corridor.tif",
        "docs/downloads/candidate_s5f_flagship.tif",
    ],
    "7GEMSDOE": [
        "downloads/gems7-lidarscarp-ridge-top2pct-36c3a3f341c8.tif",
        "downloads/gems7-strike30x3-v2-2b06d45c5b57.tif",
        "downloads/gems7-halo15-gbt-v1-90fb7dc0fc1f.tif",
    ],
    "GEMSDOE10": [
        "docs/downloads/gems10-h16-continuation-20260927T065521077735Z-3431b83c7c.tif",
        "docs/downloads/gems10-h20-dem10-scarp-thin-20260927T155223039488Z-ffc91a1686.tif",
        "docs/downloads/gems10-h25-ctx-ridge-20260927T232947704150Z-6452ae1d00.tif",
        "docs/downloads/gems10-h28-dotted-ridge-20260928T020256236880Z-6452ae1d00.tif",
    ],
    "12GEMSDOE": [
        "docs/downloads/12GEMSDOE_r7-nms3-dem10-scarp_0c9199f14e62.tif",
        "docs/downloads/12GEMSDOE_r7-nms3-dem10-scarp_0c9199f14e62_allfinite.tif",
        "docs/downloads/12GEMSDOE_multiphysics_submission_06ca61e5.tif",
        "docs/downloads/12GEMSDOE_r6-nms3-h28-texture_8721329b55c7.tif",
        "docs/downloads/12GEMSDOE_r5-nms3-trace_055e9aac96b8.tif",
    ],
    "13GEMSDOE": [
        "docs/downloads/latest.tif",
        "docs/downloads/latest_nan.tif",
        "docs/downloads/13gems-composite-20260929T183221Z.tif",
        "docs/downloads/13gems-composite-20260929T183221Z_allfinite.tif",
    ],
    "16GEMSDOE": [
        "docs/downloads/gems16-h16-1-seamfree-multiscale-ridge-nanmask-20260929-b16f02.tif",
        "docs/downloads/gems16-h16-1-seamfree-multiscale-ridge-allfinite-20260929-a16f01.tif",
        "docs/downloads/gems16-h16-1-pure-physical-scarp-worm-allfinite-20260929-c16f03.tif",
    ],
    "8GEMSDOE": [
        "docs/downloads/submission.tif",
        "docs/downloads/8GEMSDOE_Hedge-v2_submission.tif",
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="/tmp/gems_range_audit")
    ap.add_argument("--sample", default=str(REPO / "data" / "sample_submission.tif"))
    ap.add_argument("--out", default=str(REPO / "evidence" / "range_audit.json"))
    args = ap.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    footprint = None
    sp = Path(args.sample)
    if sp.exists():
        with rasterio.open(sp) as ds:
            footprint = np.isfinite(ds.read(1))
        print(f"footprint loaded from {sp} ({int(footprint.sum()):,} finite px)")
    else:
        print("NOTE: official sample_submission.tif not present; the "
              "'NaN inside footprint' check will be reported as unavailable.")

    rows, failures = [], []
    for repo, paths in TARGETS.items():
        for p in paths:
            dest = cache / f"{repo}__{Path(p).name}"
            try:
                if not dest.exists():
                    r = subprocess.run(
                        ["gh", "api", f"repos/{ORG}/{repo}/contents/{p}?ref=HEAD",
                         "-H", "Accept: application/vnd.github.raw"],
                        capture_output=True)
                    if r.returncode:
                        raise RuntimeError(r.stderr.decode()[:120])
                    dest.write_bytes(r.stdout)
                with rasterio.open(dest) as ds:
                    a = ds.read(1)
                    meta = {"dtype": ds.dtypes[0], "count": ds.count, "crs": str(ds.crs),
                            "shape": list(ds.shape)}
                fin = np.isfinite(a)
                vals = a[fin]
                oor = int(((vals < 0) | (vals > 1)).sum())
                rec = {
                    "repo": repo, "path": p, "file": dest.name,
                    "bytes": dest.stat().st_size, "meta": meta,
                    "finite_px": int(fin.sum()), "nan_px": int((~fin).sum()),
                    "min": float(vals.min()), "max": float(vals.max()),
                    "out_of_range_px": oor,
                    "distinct_values": int(np.unique(vals).size),
                    "binary": bool(np.unique(vals).size <= 2),
                    "nan_inside_footprint": (int((~fin & footprint).sum())
                                             if footprint is not None else None),
                }
                rows.append(rec)
                flag = "OOR" if oor else "ok "
                print(f"  {flag} {repo:10s} {Path(p).name[:52]:54s} "
                      f"min={rec['min']:<8.4g} max={rec['max']:<8.4g} uniq={rec['distinct_values']:>7d}")
            except Exception as e:                                   # pragma: no cover
                failures.append({"repo": repo, "path": p, "error": str(e)[:200]})
                print(f"  ERR {repo:10s} {p[:52]} -> {e}")

    n = len(rows)
    n_oor = sum(1 for r in rows if r["out_of_range_px"] > 0)
    n_nan_in = sum(1 for r in rows
                   if r["nan_inside_footprint"] is not None and r["nan_inside_footprint"] > 0)
    report = {
        "purpose": ("determine whether any published artifact could have produced the "
                    "'Predicted values must be in range [0, 1]' rejection"),
        "files_audited": n,
        "files_with_out_of_range_values": n_oor,
        "files_with_nan_inside_footprint": n_nan_in,
        "footprint_available": footprint is not None,
        "files": rows,
        "failures": failures,
        "conclusion": (
            f"All {n} published GeoTIFFs audited have finite values inside [0, 1] and "
            f"{n_oor} of them contain an out-of-range value. The rejection is therefore "
            "not reproducible from the published artifacts; the offending file was either "
            "never published or has since been replaced. Recorded as an open irregularity "
            "(I-6) rather than resolved by guessing."
            if n_oor == 0 else
            f"{n_oor} of {n} published GeoTIFFs contain out-of-range values; the "
            "rejection is explained by the file(s) listed with out_of_range_px > 0."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\n{n} files audited, {n_oor} with out-of-range values, "
          f"{n_nan_in} with NaN inside the footprint")
    print(report["conclusion"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
