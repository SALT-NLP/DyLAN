#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert per-type CSV files into per-question JSON shards.

Input:
  --src  : directory containing *.csv (e.g., data/math/small_team_selection)
           Each CSV must have header: question,answer,level,type
  --dst  : output parent directory (e.g., data/math_json/small_team_selection)
  --zpad : zero-padding width for filenames (default: 4 -> 0001.json)

Output layout example:
  <dst>/
    algebra_test/
      0001.json
      0002.json
      ...
    geometry_test/
      0001.json
      ...
"""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source directory containing *.csv files.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Destination parent directory for per-question JSON shards.",
    )
    parser.add_argument(
        "--zpad",
        type=int,
        default=4,
        help="Zero-padding width for JSON filenames (default: 4 => 0001.json).",
    )
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(args.src.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found under: {args.src}")

    for csv_path in csv_files:
        out_dir = args.dst / csv_path.stem  # e.g. algebra_test.csv -> <dst>/algebra_test/
        out_dir.mkdir(parents=True, exist_ok=True)

        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header != ["question", "answer", "level", "type"]:
                raise SystemExit(
                    f"{csv_path}: invalid header {header} "
                    "(expected: ['question','answer','level','type'])"
                )
            rows = list(reader)

        for i, row in enumerate(rows, 1):
            if len(row) != 4:
                raise SystemExit(f"{csv_path}: row {i+1} does not have 4 columns: {row}")
            q, a, level, typ = row
            obj = {"question": q, "answer": a, "level": level, "type": typ}
            fname = f"{str(i).zfill(args.zpad)}.json"
            with (out_dir / fname).open("w", encoding="utf-8") as g:
                json.dump(obj, g, ensure_ascii=False)
        print(f"Wrote {len(rows):5d} JSON files -> {out_dir}")


if __name__ == "__main__":
    main()
