#!/usr/bin/env python
# Simple utility to print the best score; convenient to pipe into xclip

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Print a prompt from an optimization run.")
    ap.add_argument("read_from", help="Run directory name under data/optimization/")
    ap.add_argument(
        "--latest",
        action="store_true",
        help="Print the latest candidate instead of the best-scoring one",
    )
    return ap.parse_args(argv)


args = parse_args()

REPO_ROOT = Path(__file__).resolve().parent
ARTEFACT_ROOT = REPO_ROOT / "data"

run_dir = ARTEFACT_ROOT / "optimization" / args.read_from
assert run_dir.is_dir(), f"read_from is not a directory: {run_dir}"

RECORD = run_dir / ("latest.json" if args.latest else "best.json")
record = json.loads(RECORD.read_text())
print(record["prompt"], end="")
