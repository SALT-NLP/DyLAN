#!/usr/bin/env python3
"""
Prepare MMLU subsets for DyLAN experiments.

This creates three sibling directories under <MMLU root>:

  - evaluation/             # sampled per-CSV from test/
  - small_team_selection/   # sampled per-CSV from val/ (smaller)
  - larger_team_selection/  # sampled per-CSV from val/ (larger)

Design:
  * evaluation is built from test/ using --eval-frac.
  * small_team_selection is a deterministic subset of larger_team_selection,
    both sampled from val/ using --sel-small-frac and --sel-large-frac.
  * Sampling is per subject CSV (header preserved), deterministic per file + seed.

Usage (from repo root, adjust paths as needed):
  python code/preprocess/mmlu_prepare_subsets_v2.py \
      --mmlu-root data/MMLU \
      --seed 0 \
      --eval-frac 0.20 \
      --sel-small-frac 0.01 \
      --sel-large-frac 0.10
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
from pathlib import Path
from typing import List, Tuple

# Defaults: match the paper-style setup (1/5 test for evaluation, 1% and 10% of val for team selection)
DEF_EVAL_FRAC = 0.20
DEF_SEL_SMALL = 0.01
DEF_SEL_LARGE = 0.10


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
    """Deterministic, per-file RNG so results are stable across runs for the same seed."""
    h = hashlib.sha256(name.encode("utf-8")).hexdigest()
    per = int(h[:12], 16)  # 48-bit chunk
    return random.Random(seed ^ per)


def sample_k_from_body(header: List[str], body: List[List[str]], k: int, rng: random.Random) -> List[List[str]]:
    if k >= len(body):
        return [header] + body  # tiny files: keep all
    idx = list(range(len(body)))
    rng.shuffle(idx)
    chosen = [body[i] for i in idx[:k]]
    return [header] + chosen


def build_eval_from_split(src_dir: Path, out_dir: Path, frac: float, seed: int) -> Tuple[int, int]:
    """Sample per-CSV from src_dir into out_dir with fraction frac."""
    count_files = 0
    count_rows = 0
    for csv_path in sorted(src_dir.glob("*.csv")):
        rows = read_rows(csv_path)
        if not rows:
            continue
        header, body = rows[0], rows[1:]
        n = len(body)

        rng = per_file_rng(seed, csv_path.name)
        k = max(1, int(math.ceil(n * frac)))
        sampled = sample_k_from_body(header, body, k, rng)

        write_rows(out_dir / csv_path.name, sampled)
        count_files += 1
        count_rows += len(sampled) - 1
    return count_files, count_rows


def build_nested_small_large_from_split(src_dir: Path, out_small: Path, out_large: Path,
                                        frac_small: float, frac_large: float, seed: int) -> Tuple[int, int, int]:
    """
    Build two nested subsets from the same src_dir (val/):
      * First sample 'large' with frac_large
      * Then sample 'small' as a deterministic subset of the large sample with frac_small_of_large
        where frac_small_of_large is computed w.r.t. the original file size (not of the large size),
        but the selection is drawn from the large set to guarantee nesting.
    """
    if frac_small > frac_large:
        raise SystemExit("--sel-small-frac cannot be greater than --sel-large-frac (we enforce nesting).")

    files = sorted(src_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {src_dir}")

    files_written = 0
    rows_large_total = 0
    rows_small_total = 0

    for csv_path in files:
        rows = read_rows(csv_path)
        if not rows:
            continue
        header, body = rows[0], rows[1:]
        n = len(body)

        rng = per_file_rng(seed, csv_path.name)

        # Large first
        k_large = max(1, int(math.ceil(n * frac_large)))
        large_rows = sample_k_from_body(header, body, k_large, rng)
        large_header, large_body = large_rows[0], large_rows[1:]

        # Now pick small as a subset of large deterministically (nested)
        k_small = max(1, int(math.ceil(n * frac_small)))
        # If requested "small" is >= large, just reuse large
        if k_small >= len(large_body):
            small_body = large_body
        else:
            idx = list(range(len(large_body)))
            rng.shuffle(idx)
            small_body = [large_body[i] for i in idx[:k_small]]

        write_rows(out_large / csv_path.name, [large_header] + large_body)
        write_rows(out_small / csv_path.name, [large_header] + small_body)

        files_written += 1
        rows_large_total += len(large_body)
        rows_small_total += len(small_body)

    return files_written, rows_small_total, rows_large_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlu-root", type=Path, required=True,
                    help="Path to MMLU root (contains val/, test/, train/, etc.)")
    ap.add_argument("--seed", type=int, default=0, help="Global seed for deterministic sampling.")
    ap.add_argument("--eval-frac", type=float, default=DEF_EVAL_FRAC,
                    help="Per-CSV fraction for evaluation/ from test/ (default: 0.20).")
    ap.add_argument("--sel-small-frac", type=float, default=DEF_SEL_SMALL,
                    help="Per-CSV fraction for small_team_selection/ from val/ (default: 0.01).")
    ap.add_argument("--sel-large-frac", type=float, default=DEF_SEL_LARGE,
                    help="Per-CSV fraction for larger_team_selection/ from val/ (default: 0.10).")
    args = ap.parse_args()

    root = args.mmlu_root
    val_dir = root / "val"
    test_dir = root / "test"

    if not val_dir.exists():
        raise SystemExit(f"Missing split: {val_dir}")
    if not test_dir.exists():
        raise SystemExit(f"Missing split: {test_dir}")

    out_eval = root / "evaluation"                # from test/
    out_small = root / "small_team_selection"     # from val/
    out_large = root / "larger_team_selection"    # from val/

    print(f"[1/2] Building evaluation/ from {test_dir} with frac={args.eval_frac} …")
    n_eval_files, n_eval_rows = build_eval_from_split(test_dir, out_eval, args.eval_frac, args.seed)
    print(f"      Wrote {n_eval_files} files, {n_eval_rows} rows total → {out_eval}")

    print(f"[2/2] Building nested small/large team-selection from {val_dir} "
          f"(small={args.sel_small_frac}, large={args.sel_large_frac}) …")
    n_sel_files, n_small_rows, n_large_rows = build_nested_small_large_from_split(
        val_dir, out_small, out_large, args.sel_small_frac, args.sel_large_frac, args.seed
    )
    print(f"      Wrote {n_sel_files} files → {out_small} (rows: {n_small_rows}) "
          f"and {out_large} (rows: {n_large_rows})")

    print("Done.")


if __name__ == "__main__":
    main()
