#!/usr/bin/env python3
"""Answer the two forensic questions from bytes rather than from score arithmetic.

    "why do we keep scoring 0.1563, are we copying the same work over and over?"
    "why do 5GEMSDOE and GEMSDOE1 have the same score?"

Two submissions can show the same rounded score for very different reasons:
two different models can genuinely agree, or one file can have been uploaded
twice. Score arithmetic cannot tell those apart, so this script compares the
actual bytes.

What it does, per sibling repository reachable through the GitHub API:

1. Lists the git tree and records the **blob SHA** of every file. Two files with
   the same blob SHA are byte-identical, and unlike a file size or a name this
   cannot be a coincidence.
2. Compares repositories by path: how many paths are shared, and how many of
   those shared paths hold an identical blob. A high count means one repository
   is a copy of another.
3. Downloads every submission GeoTIFF it can find and records its SHA-256, its
   raster metadata, and the *support set* (which pixels are above zero).
4. Computes the Jaccard overlap of the support sets, because two submissions can
   be different files with identical scores only if their supports are very
   nearly equal.

Everything is written to ``evidence/forensic_audit.json`` and every number on
docs/findings.html comes from that file.

Requires the ``gh`` CLI with repo read access. Network access is used only
against api.github.com.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ORG = "buffedlizard55-lab"

# Repositories to audit. The set is the project family named in the brief plus
# the ones that hold graded artifacts; unrelated repositories are not touched.
REPOS = ["GEMSDOE", "5GEMSDOE", "GEMSDOE2", "GEMSDOE4", "8GEMSDOE", "6GEMSDOE",
         "GEMSDOE3", "7GEMSDOE", "9GEMSDOE", "10GEMSDOE", "13GEMSDOE",
         "14GEMSDOE", "15GEMSDOE", "16GEMSDOE", "11GEMSDOE", "12GEMSDOE"]

# Paths that are competition inputs rather than project work; if these are
# shared between repositories it says nothing about copying.
SHARED_DATA_PATHS = {"data/bridge/example_submission.tif",
                     "data/bridge/existing_faults.tif",
                     "data/labels.tif",
                     "data/sample_submission.tif"}


def gh(*args: str) -> bytes:
    r = subprocess.run(["gh", *args], capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr.decode()[:300])
    return r.stdout


def gh_json(path: str):
    return json.loads(gh("api", path).decode() or "null")


def repo_tree(repo: str, ref: str = "HEAD") -> dict[str, str]:
    """Map path -> blob SHA for every file in the repository."""
    out: dict[str, str] = {}
    try:
        data = gh_json(f"repos/{ORG}/{repo}/git/trees/{ref}?recursive=1")
    except Exception as exc:                                    # pragma: no cover
        print(f"  !! {repo}: {exc}")
        return out
    for item in data.get("tree", []):
        if item.get("type") == "blob":
            out[item["path"]] = item["sha"]
    return out


def fetch_blob(repo: str, sha: str, dest: Path) -> Path | None:
    if dest.exists() and dest.stat().st_size:
        return dest
    try:
        raw = gh("api", f"repos/{ORG}/{repo}/git/blobs/{sha}",
                 "-H", "Accept: application/vnd.github.raw")
        dest.write_bytes(raw)
        return dest
    except Exception as exc:                                    # pragma: no cover
        print(f"  !! blob {repo}/{sha[:8]}: {exc}")
        return None


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="/tmp/gems_forensic")
    ap.add_argument("--out", default="evidence/forensic_audit.json")
    args = ap.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    print(f"== 1. git trees for {len(REPOS)} repositories ==")
    trees: dict[str, dict[str, str]] = {}
    for repo in REPOS:
        t = repo_tree(repo)
        trees[repo] = t
        print(f"  {repo:14s} {len(t):5d} files")

    print("\n== 2. cross-repository blob comparison ==")
    pairs = []
    for i, a in enumerate(REPOS):
        for b in REPOS[i + 1:]:
            ta, tb = trees.get(a, {}), trees.get(b, {})
            if not ta or not tb:
                continue
            shared = set(ta) & set(tb)
            same = {p for p in shared if ta[p] == tb[p]}
            work_shared = shared - SHARED_DATA_PATHS
            work_same = same - SHARED_DATA_PATHS
            if not shared:
                continue
            pairs.append({
                "a": a, "b": b,
                "shared_paths": len(shared), "identical_blobs": len(same),
                "percent_identical": round(100.0 * len(same) / len(shared), 1),
                "work_shared_paths": len(work_shared),
                "work_identical_blobs": len(work_same),
                "percent_work_identical": (round(100.0 * len(work_same) / len(work_shared), 1)
                                           if work_shared else 0.0),
            })
    pairs.sort(key=lambda p: -p["percent_work_identical"])
    for p in pairs[:12]:
        print(f"  {p['a']:12s} vs {p['b']:12s} "
              f"{p['work_identical_blobs']:5d}/{p['work_shared_paths']:<5d} "
              f"work files identical ({p['percent_work_identical']}%)")

    # Group blob SHAs that appear at more than one path/repo: these are literal
    # duplicates of the same file.
    where: dict[str, list[str]] = {}
    for repo, t in trees.items():
        for path, sha in t.items():
            if path in SHARED_DATA_PATHS:
                continue
            where.setdefault(sha, []).append(f"{repo}:{path}")
    dupe_groups = {sha: locs for sha, locs in where.items() if len(locs) > 1}

    print(f"\n  {len(dupe_groups)} blob SHAs appear at more than one location")

    print("\n== 3. submission rasters: bytes and support ==")
    subs = []
    seen_sha: dict[str, int] = {}
    for repo in REPOS:
        t = trees.get(repo, {})
        for path, sha in sorted(t.items()):
            if not path.lower().endswith(".tif"):
                continue
            if path in SHARED_DATA_PATHS:
                continue
            dest = cache / f"{repo}__{Path(path).name}"
            got = fetch_blob(repo, sha, dest)
            if got is None:
                continue
            rec = {"repo": repo, "path": path, "blob_sha": sha,
                   "bytes": got.stat().st_size,
                   "sha256": sha256_file(got)}
            try:
                import numpy as np
                import rasterio
                with rasterio.open(got) as ds:
                    a = ds.read(1)
                fin = np.isfinite(a)
                rec["shape"] = list(a.shape)
                rec["crs"] = str(ds.crs)
                rec["dtype"] = ds.dtypes[0]
                rec["min"] = float(a[fin].min()) if fin.any() else None
                rec["max"] = float(a[fin].max()) if fin.any() else None
                rec["distinct_values"] = int(np.unique(a[fin]).size) if fin.any() else 0
                rec["finite_px"] = int(fin.sum())
                # support = strictly positive cells
                sup = fin & (a > 0)
                rec["support_px"] = int(sup.sum())
                if rec["support_px"]:
                    np.save(cache / f"{repo}__{Path(path).stem}.support.npy",
                            np.packbits(sup.ravel()))
            except Exception as exc:                            # pragma: no cover
                rec["error"] = f"{type(exc).__name__}: {exc}"
            subs.append(rec)
            seen_sha.setdefault(rec["sha256"], 0)
            seen_sha[rec["sha256"]] += 1

    byte_dupes = {s: c for s, c in seen_sha.items() if c > 1}
    print(f"  {len(subs)} rasters read; {len(byte_dupes)} SHA-256 values occur "
          f"more than once")
    for s, c in sorted(byte_dupes.items(), key=lambda kv: -kv[1]):
        locs = [f"{r['repo']}:{r['path']}" for r in subs if r["sha256"] == s]
        print(f"    {s[:16]}...  x{c}")
        for loc in locs:
            print(f"        {loc}")

    print("\n== 4. support-set Jaccard overlap (distinct predictions only) ==")
    # Files that are byte-identical are already reported as such; computing the
    # overlap of a file with itself adds nothing. One representative per distinct
    # SHA-256 is therefore enough, and it makes the comparison tractable.
    reps: dict[str, dict] = {}
    for r in subs:
        if r.get("support_px") and r["sha256"] not in reps:
            reps[r["sha256"]] = r
    mats: dict[str, "np.ndarray"] = {}
    for sha, r in reps.items():
        f = cache / f"{r['repo']}__{Path(r['path']).stem}.support.npy"
        if f.exists():
            mats[sha] = np.unpackbits(np.load(f))
    matrix = []
    keys = sorted(mats)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if mats[a].shape != mats[b].shape:
                continue
            inter = int((mats[a] & mats[b]).sum())
            union = int((mats[a] | mats[b]).sum())
            if union:
                ra, rb = reps[a], reps[b]
                matrix.append({
                    "a": f"{ra['repo']}:{Path(ra['path']).name}",
                    "b": f"{rb['repo']}:{Path(rb['path']).name}",
                    "jaccard": round(inter / union, 4),
                    "intersection_px": inter, "union_px": union,
                    "a_support_px": ra["support_px"], "b_support_px": rb["support_px"]})
    matrix.sort(key=lambda m: -m["jaccard"])
    for m in matrix[:10]:
        print(f"  {m['jaccard']:.4f}  {m['a'][:42]:44s} {m['b'][:42]}")

    report = {
        "org": ORG,
        "repositories_audited": [r for r in REPOS if trees.get(r)],
        "file_counts": {r: len(t) for r, t in trees.items()},
        "duplicate_blob_groups": {sha: locs for sha, locs in
                                  sorted(dupe_groups.items(), key=lambda kv: -len(kv[1]))[:40]},
        "n_duplicate_blob_groups": len(dupe_groups),
        "repo_pairs": pairs,
        "submissions": subs,
        "byte_identical_submission_sha256": {
            s: [f"{r['repo']}:{r['path']}" for r in subs if r["sha256"] == s]
            for s in byte_dupes},
        "support_jaccard": matrix[:40],
        "conclusions": _conclusions(
            {s: [f"{r['repo']}:{r['path']}" for r in subs if r["sha256"] == s]
             for s in byte_dupes},
            subs, matrix, pairs),
        "method": "git blob SHAs for byte-identity of files; SHA-256 of the "
                  "downloaded raster bytes; Jaccard of the sets of strictly "
                  "positive pixels for near-identity of predictions",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")
    for c in report["conclusions"]:
        print(f"  * {c}")
    return 0


def _conclusions(byte_dupes, subs, matrix, pairs) -> list[str]:
    out = []
    for sha, locs in byte_dupes.items():
        repos = sorted({l.split(":")[0] for l in locs})
        if len(repos) > 1:
            out.append(
                f"The same raster bytes (sha256 {sha[:16]}...) are published in "
                f"{len(repos)} different repositories ({', '.join(repos)}). Two "
                "accounts submitting that file would necessarily receive the "
                "same score, because the scored object is identical.")
    close = [m for m in matrix if m["jaccard"] >= 0.9]
    for m in close[:3]:
        out.append(f"Support sets of {m['a']} and {m['b']} overlap with Jaccard "
                   f"{m['jaccard']:.4f}: near-identical predictions.")
    worst = [p for p in pairs if p["percent_work_identical"] >= 50]
    for p in worst[:4]:
        out.append(f"{p['b']} is substantially a copy of {p['a']}: "
                   f"{p['work_identical_blobs']} of {p['work_shared_paths']} shared "
                   f"work files are byte-identical "
                   f"({p['percent_work_identical']} %).")
    if not out:
        out.append("No evidence of duplicated work or duplicated predictions was "
                   "found among the audited repositories.")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
