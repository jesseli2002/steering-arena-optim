#!/usr/bin/env python
# Simple utility to print the best score, to pipe into `xclip`

import json
import os
from pathlib import Path

# Which run to read from
# READ_FROM = "2026-07-28T17-19-18Z"
READ_FROM = "2026-07-28_tok32_pro"
# READ_FROM = "2026-07-27_tok24_anti"

REPO_ROOT = Path(__file__).resolve().parent
ARTEFACT_ROOT = REPO_ROOT / "data"

run_dir = ARTEFACT_ROOT / "optimization" / READ_FROM
assert run_dir.is_dir(), f"RESUME_FROM is not a directory: {run_dir}"

BEST = run_dir / "best.json"
BEST = run_dir / "latest.json"
best_json = json.loads(BEST.read_text())
print(best_json["prompt"], end="")
