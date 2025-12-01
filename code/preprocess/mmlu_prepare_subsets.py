#!/usr/bin/env python3
"""
Prepare MMLU subsets for DyLAN experiments.

It creates three sibling directories under <MMLU root>:
  - evaluation/         # 4% of the chosen split (val/test)
  - small_team_selection/   # 10% of that 4%
  - medium_team_selection/   # 40% of that 4%

Why:
  * Paper (GR/MMLU): down-sample the test set by 1/5 for evaluation.
  * Team selection can be done on a tiny subset (1% or 10%), then evaluated on the 1/5 set.

Usage:
  python code/preprocess/mmlu_prepare_subsets.py --mmlu-root data/MMLU --source-split val --seed 0

Notes:
  * Keeps the original CSV filenames (e.g., abstract_algebra_val.csv) so your
    exp_mmlu.sh can point dir=.../one_percent_team_selection (or .../one_fifth_evaluation)
    without any other changes.
  * Sampling is deterministic per file given --seed.
"""

import argparse
import csv
import hashlib
import math
import random
from pathlib import Path
from typing import List

EVAL_FRACTION = 0.04   # 4% of the chosen split (val/test)
SEL_SMALL = 0.10       # 10% of the 4%
SEL_MED = 0.40         # 40% of the 4%


def read_rows(csv_path: Path) -> List[List[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows


def write_rows(csv_path: Path, rows: List[List[str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def per_file_rng(seed: int, name: str) -> random.Random:
    h = hashlib.sha256(name.encode("utf-8")).hexdigest()
    per = int(h[:12], 16)  # stable-ish 48-bit chunk
    return random.Random(seed ^ per)


def sample_rows(all_rows: List[List[str]], k: int, rng: random.Random) -> List[List[str]]:
    header, body = all_rows[0], all_rows[1:]
    if k >= len(body):
        # If tiny subject: keep everything (but still return header+body).
        return [header] + body
    idx = list(range(len(body)))
    rng.shuffle(idx)
    chosen = [body[i] for i in idx[:k]]
    return [header] + chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlu-root", type=Path, required=True,
                    help="Path to MMLU root (the folder that contains val/, test/, train/, etc.)")
    ap.add_argument("--source-split", choices=["val", "test"], default="test",
                    help="Which split to start from (default: test).")
    ap.add_argument("--seed", type=int, default=0, help="Global seed for deterministic sampling.")
    ap.add_argument("--eval-frac", type=float, default=EVAL_FRACTION,
                    help="Fraction for evaluation (default 0.10).")
    ap.add_argument("--sel-small-frac", type=float, default=SEL_SMALL,
                    help="Fraction (of the eval set) for small_team_selection (default 0.20).")
    ap.add_argument("--sel-med-frac", type=float, default=SEL_MED,
                    help="Fraction (of the eval set) for medium_team_selection (default 0.40).")
    args = ap.parse_args()

    src_dir = args.mmlu_root / args.source_split
    if not src_dir.exists():
        raise SystemExit(f"Source split not found: {src_dir}")

    out_eval = args.mmlu_root / "evaluation"
    out_sel1 = args.mmlu_root / "small_team_selection"
    out_sel10 = args.mmlu_root / "medium_team_selection"

    created = {p: 0 for p in (out_eval, out_sel1, out_sel10)}

    csv_files = sorted(src_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {src_dir}")

    for csv_path in csv_files:
        rows = read_rows(csv_path)
        if not rows:
            continue

        header, body = rows[0], rows[1:]
        n = len(body)
        rng = per_file_rng(args.seed, csv_path.name)

        # size for evaluation (1/5)
        k_eval = max(1, int(math.ceil(n * args.eval_frac)))
        sampled_eval = sample_rows(rows, k_eval, rng)

        # sizes for selection sets (percentages of the eval set)
        m = len(sampled_eval) - 1  # body size of eval
        k_small = max(1, int(math.ceil(m * args.sel_small_frac)))
        k_med = max(1, int(math.ceil(m * args.sel_med_frac)))

        # For selection subsets, sample *from the eval body* deterministically but
        # using the same rng so subsets are nested prefixes.
        eval_header, eval_body = sampled_eval[0], sampled_eval[1:]
        idx_eval = list(range(len(eval_body)))
        rng.shuffle(idx_eval)

        small_body = [eval_body[i] for i in idx_eval[:k_small]]
        med_body = [eval_body[i] for i in idx_eval[:k_med]]

        # Write three outputs, preserving the original filename.
        rel_name = csv_path.name  # e.g., abstract_algebra_val.csv or ..._test.csv
        write_rows(out_eval / rel_name, [header] + eval_body)
        created[out_eval] += 1

        write_rows(out_sel1 / rel_name, [eval_header] + small_body)
        created[out_sel1] += 1

        write_rows(out_sel10 / rel_name, [eval_header] + med_body)
        created[out_sel10] += 1

    print("Done.")
    for folder, count in created.items():
        print(f"  Wrote {count:4d} files → {folder}")


if __name__ == "__main__":
    main()
