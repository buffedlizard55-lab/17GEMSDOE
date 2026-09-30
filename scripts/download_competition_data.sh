#!/usr/bin/env bash
# Alias kept because the standing brief names this entry point verbatim:
#   "run `bash scripts/download_competition_data.sh` on any unrestricted
#    machine into `data/`, then `python scripts/prepare_data.py`"
# All logic lives in fetch_competition_data.sh; this wrapper exists so the
# command in the brief is literally correct.
set -euo pipefail
exec bash "$(dirname "$0")/fetch_competition_data.sh" "$@"
