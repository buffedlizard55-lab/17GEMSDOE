#!/usr/bin/env python3
"""Standalone submission-format validator.

Run this on the file *before* uploading it.  It is the gate that would have
caught the operator's rejected upload:

    "Predicted values must be in range [0, 1]"

Every check is derived from the official "Submission format" section of the
problem description (verified 2026-09-30):

    - same projected CRS as the training data (UTM zone 11N, EPSG 32611)
    - same resolution (100 m)
    - same bounds; data outside the bounds is null or nan
    - a single layer, datatype float32, values between 0 and 1

Exit code 0 means the file passed every check and is safe to upload.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gems.io import validate_submission  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tif", help="the GeoTIFF you intend to upload")
    ap.add_argument("--sample", default=str(REPO / "data" / "sample_submission.tif"),
                    help="official sample submission defining the required grid")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    sample = Path(args.sample)
    if not sample.exists():
        print(f"FATAL: official sample submission not found at {sample}")
        print("Run: bash scripts/fetch_competition_data.sh")
        return 2

    report = validate_submission(Path(args.tif), sample)

    print(f"file    : {report['file']}")
    if "file_sha256" in report:
        print(f"sha256  : {report['file_sha256']}")
        print(f"bytes   : {report['bytes']}")
    print()
    widths = max((len(k) for k in report["checks"]), default=10)
    for name, c in report["checks"].items():
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {name:<{widths}}  {c['detail']}")
    print()
    if "min" in report:
        print(f"  finite value range : [{report['min']}, {report['max']}]")
        print(f"  finite pixels      : {report['finite_px']}")
        print(f"  NaN pixels         : {report['nan_px']} "
              f"(inside the official footprint: {report.get('nan_inside_footprint')})")
        print(f"  positive pixels    : {report.get('positive_px')}")
    for w in report["warnings"]:
        print(f"  WARN  {w}")
    for e in report["errors"]:
        print(f"  ERROR {e}")

    print()
    if report["ok"]:
        print("RESULT: PASS -- this file satisfies the official submission format "
              "and is safe to upload.")
    else:
        print("RESULT: FAIL -- do NOT upload this file.")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=1))
        print(f"report written to {args.json_out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
