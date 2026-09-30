# data/ — never committed

Competition rasters live here and are **not** tracked by git. Only this README is.

| File | Bytes | SHA-256 |
|---|---|---|
| `training_features.tif` | 418,912,844 | `4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5` |
| `labels.tif` | 425,830 | `7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093` |
| `sample_submission.tif` | 1,599,597 | `2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc` |

Two ways to place them:

1. **Unrestricted machine.** Download from the competition data tab (login required)
   into this directory, then run `python scripts/prepare_data.py`.
2. **GitHub bridge.** `bash scripts/fetch_competition_data.sh` reassembles them from
   the sibling repositories, then verifies every hash.

`prepare_data.py` refuses to describe the rasters if any hash mismatches, so a
downstream number can always be traced to verified inputs.
