#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Competition Math subsets in MMLU-like layout (CSV).

Outputs under <math-root>:
  - evaluation/                   # 4% of full train (by type, one CSV per type)
  - small_team_selection/         # 10% of evaluation (nested subset)
  - medium_team_selection/        # 40% of evaluation (nested subset)
  - <split-name>/                 # (optional via --emit-base-split) full-by-type CSVs

Filenames: <slug_of_type>_<split>.csv  (e.g., algebra_test.csv)
Columns:   question,answer,level,type

Notes:
- Sampling is deterministic per type given --seed.
- Subsets are nested: small ⊂ medium ⊂ evaluation within each type.
"""

import argparse
import csv
import hashlib
import math
import random
import re
from pathlib import Path
from typing import List
from datasets import load_dataset, Dataset


# Default fractions
EVAL_FRACTION = 0.04   # 4% of full train
SEL_SMALL     = 0.10   # 10% of evaluation
SEL_MED       = 0.40   # 40% of evaluation


def slugify(s: str) -> str:
    """Filesystem-friendly slug similar to MMLU subject filenames."""
    s = s.strip().lower()
    s = s.replace("&", "and").replace("/", " ").replace("\\", " ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "untitled"


def per_group_rng(seed: int, name: str) -> random.Random:
    """Stable RNG per group/type using a hash of the type name."""
    h = hashlib.sha256(name.encode("utf-8")).hexdigest()
    per = int(h[:12], 16)  # 48-bit chunk
    return random.Random(seed ^ per)


def ensure_cols(ds: Dataset) -> None:
    """Ensure dataset contains the expected columns."""
    needed = ["problem", "solution", "level", "type"]
    missing = [c for c in needed if c not in ds.column_names]
    if missing:
        raise KeyError(f"Dataset missing columns: {missing}. Have: {ds.column_names}")


def write_rows(path: Path, rows: List[List[str]]) -> None:
    """Write CSV rows to path (UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--math-root",
        type=Path,
        required=True,
        help="Root output dir (will create evaluation/, small_team_selection/, medium_team_selection/).",
    )
    ap.add_argument(
        "--split-name",
        type=str,
        default="test",
        help="Filename suffix tag for CSVs (purely cosmetic): e.g. test/val/train.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Global seed for deterministic sampling.")
    ap.add_argument("--eval-frac", type=float, default=EVAL_FRACTION, help="Fraction for evaluation.")
    ap.add_argument("--sel-small-frac", type=float, default=SEL_SMALL, help="Fraction of evaluation for small subset.")
    ap.add_argument("--sel-med-frac", type=float, default=SEL_MED, help="Fraction of evaluation for medium subset.")
    ap.add_argument(
        "--emit-base-split",
        action="store_true",
        help="Also write full-by-type CSVs under <math-root>/<split-name>/",
    )
    args = ap.parse_args()

    # 1) Load the only split available (train)
    print("Loading 'qwedsacf/competition_math' (split=train) ...")
    ds = load_dataset("qwedsacf/competition_math", split="train")
    ensure_cols(ds)
    types = sorted(set(ds["type"]))
    print(f"Loaded {len(ds)} rows; {len(types)} types: {types}")

    # 2) Prepare output directories
    out_eval  = args.math_root / "evaluation"
    out_small = args.math_root / "small_team_selection"
    out_med   = args.math_root / "medium_team_selection"
    out_split = args.math_root / args.split_name  # optional full split export

    for d in [out_eval, out_small, out_med] + ([out_split] if args.emit_base_split else []):
        d.mkdir(parents=True, exist_ok=True)

    # 3) Per-type processing
    for t in types:
        ds_t = ds.filter(lambda ex: ex["type"] == t)
        n = len(ds_t)
        if n == 0:
            continue

        rng = per_group_rng(args.seed, t)

        # Determine evaluation size (rounded up; min 1)
        k_eval = max(1, math.ceil(n * args.eval_frac))

        # Sample k_eval examples from the type
        all_idx = list(range(n))
        rng.shuffle(all_idx)
        eval_idx = all_idx[:k_eval]

        # Within evaluation, create nested small/medium by prefix of a shuffled order
        order_eval = list(range(len(eval_idx)))
        rng.shuffle(order_eval)
        eval_idx_ordered = [eval_idx[i] for i in order_eval]

        k_small = max(1, math.ceil(len(eval_idx_ordered) * args.sel_small_frac))
        k_med   = max(1, math.ceil(len(eval_idx_ordered) * args.sel_med_frac))

        small_idx = eval_idx_ordered[:k_small]
        med_idx   = eval_idx_ordered[:k_med]

        def build_rows(idxs: List[int]) -> List[List[str]]:
            rows = [["question", "answer", "level", "type"]]
            for i in idxs:
                ex = ds_t[i]
                rows.append([ex["problem"], ex["solution"], ex["level"], ex["type"]])
            return rows

        fname = f"{slugify(t)}_{args.split_name}.csv"

        # Write three subset CSVs
        write_rows(out_eval / fname,  build_rows(eval_idx_ordered))
        write_rows(out_small / fname, build_rows(small_idx))
        write_rows(out_med / fname,   build_rows(med_idx))

        # Optionally write the full-by-type split as well
        if args.emit_base_split:
            write_rows(out_split / fname, build_rows(list(range(n))))

        print(f"[{t}] full={n}, eval={len(eval_idx_ordered)}, "
              f"small={len(small_idx)}, med={len(med_idx)} -> {fname}")

    print("\n Done.")
    print(f"Saved CSVs under: {args.math_root}")


if __name__ == "__main__":
    main()
