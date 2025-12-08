#!/usr/bin/env python3
import csv
import sys
import json
from collections import defaultdict

"""
Usage:
  python select_math_roles.py importance_math_1to7.csv 4 math_eval_roles.json

功能：
  - 从 importance_math_1to7.csv 中读取每个 subject 的 agent importance
  - 对同一 subject 的多行做平均
  - 为每个 subject 选出 top-k 角色
  - 输出到一个 JSON: {subject: [role1, role2, ...], ...}
"""

def subject_from_filename(filename: str) -> str:
    # filename 形如: algebra_test_0001_0012
    parts = filename.split('_')
    if len(parts) > 2:
        return '_'.join(parts[:-2])
    return filename

def main(csv_path: str, k: int, out_json: str):
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"No rows in CSV: {csv_path}")
        sys.exit(1)

    # 找到所有 *_imp 列
    role_cols = [c for c in rows[0].keys() if c.endswith("_imp")]
    role_names = [c[:-4] for c in role_cols]

    # 按 subject 聚合
    agg = defaultdict(lambda: {"cnt": 0, "sums": {col: 0.0 for col in role_cols}})

    for row in rows:
        fname = row["filename"]
        subj = subject_from_filename(fname)
        agg[subj]["cnt"] += 1
        for col in role_cols:
            agg[subj]["sums"][col] += float(row[col])

    # 计算平均并选 top-k
    subject_roles = {}
    for subj, info in agg.items():
        cnt = info["cnt"]
        scores = []
        for col, s in info["sums"].items():
            avg = s / cnt
            role = col[:-4]  # 去掉 "_imp"
            scores.append((role, avg))
        scores.sort(key=lambda x: x[1], reverse=True)
        top = [r for r, _ in scores[:k]]
        subject_roles[subj] = top

    with open(out_json, "w") as f:
        json.dump(subject_roles, f, indent=2)

    print(f"Saved subject→roles mapping to {out_json}")
    for subj, roles in subject_roles.items():
        print(subj, ":", roles)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: select_math_roles.py <importance_csv> <num_roles> <out_json>")
        sys.exit(1)
    csv_path = sys.argv[1]
    k = int(sys.argv[2])
    out_json = sys.argv[3]
    main(csv_path, k, out_json)
