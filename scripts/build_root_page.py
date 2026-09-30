#!/usr/bin/env python3
"""Write the repository-root ``index.html``.

GitHub Pages for this repository is configured to serve the repository *root*,
and the token available to this session cannot change that setting (the Pages
update endpoint returns 403 "Resource not accessible by integration"). The site
itself lives in ``docs/``, so without a root entry point a visitor would land on
a directory listing instead of a download button.

This module therefore writes a small root page that carries the same one-click
download, reading every value from ``docs/downloads/submission_meta.json`` so it
cannot disagree with the file it points at. It is invoked by
``scripts/build_site.py`` rather than checked in by hand, so a rebuilt submission
renames the download here too.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
META = REPO / "docs" / "downloads" / "submission_meta.json"


def esc(s) -> str:
    return html.escape(str(s))


def build() -> str:
    meta = json.loads(META.read_text()) if META.exists() else {}
    tif = meta.get("downloads", {}).get("tif", "")
    name = meta.get("latest", "")
    comment = meta.get("comment", "")
    rng = meta.get("value_range") or [None, None]

    if not tif:
        body = (
            '<p class="warn">No submission has been built yet. Run '
            "<code>python scripts/build_submission.py --source reference</code>.</p>"
        )
    else:
        body = (
            '<a class="btn" href="docs/downloads/' + esc(tif) + '" download>'
            "Download the submission GeoTIFF</a>\n"
            '<p class="tiny">sha256 <code>' + esc(meta.get("sha256", "")) + "</code>"
            " &middot; " + f"{meta.get('bytes', 0):,}" + " bytes &middot; value range "
            "<code>[" + esc(rng[0]) + ", " + esc(rng[1]) + "]</code> &middot; "
            + str(len(meta.get("format_checks_passed", []))) + " format checks passed, "
            + str(len(meta.get("format_checks_failed", []))) + " failed</p>\n"
            '<div class="fields"><label>Submission name</label><pre>' + esc(name)
            + "</pre>\n<label>Short comment</label><pre>" + esc(comment)
            + "</pre></div>"
        )

    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>17GEMSDOE &middot; download the submission</title>\n"
        '<link rel="stylesheet" href="docs/assets/style.css">\n'
        '</head><body><main class="wrap">\n'
        '<section class="hero">\n'
        "<h1>Download the submission GeoTIFF</h1>\n"
        '<p class="lede">One click, then paste the name and the comment into the\n'
        'competition form. This is the same file the <a href="docs/index.html">full\n'
        "site</a> serves; both are generated from the same evidence.</p>\n"
        + body
        + "\n</section>\n"
        '<p class="tiny">Full documentation: <a href="docs/index.html">executive '
        'summary</a> &middot; <a href="docs/how-to-submit.html">how to submit</a> '
        '&middot; <a href="docs/findings.html">findings</a> &middot; '
        '<a href="docs/hypotheses.html">hypotheses</a> &middot; '
        '<a href="docs/sources.html">sources</a> &middot; '
        '<a href="docs/limitations.html">limitations</a>.</p>\n'
        "</main></body></html>\n"
    )


def main() -> int:
    (REPO / ".nojekyll").write_text("")
    (REPO / "index.html").write_text(build())
    print("wrote index.html and .nojekyll at the repository root")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
