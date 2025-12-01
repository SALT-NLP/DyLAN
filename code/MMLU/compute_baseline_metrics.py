#!/usr/bin/env python3
"""
Compute bootstrap confidence intervals for baseline MMLU results.

This script post-processes the 6-line .txt files produced by baseline_mmlu.py
and generates comprehensive metrics with 95% bootstrap CIs, matching the
format used in exp_mmlu_evaluation.sh and exp_mmlu_single_llm.sh.

Usage:
    python compute_baseline_metrics.py <output_dir> <model> [n_boot]

Arguments:
    output_dir: Directory containing *_baseline.txt files
    model: Model name for metadata (e.g., "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
    n_boot: Number of bootstrap replicates (default: 1000)

Outputs:
    - baseline_by_test.json: Per-test detailed results
    - metrics_summary_baseline.json: Aggregated metrics with 95% CIs
    - Console output with formatted tables
"""

import os
import sys
import re
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------
# Subject → subcategory map (from exp_mmlu_evaluation.sh)
# ---------------------------
SUBCATEGORY = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

CATEGORIES = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

# ---------------------------
# Helpers (from exp_mmlu_evaluation.sh)
# ---------------------------
def subject_key_from_name(name: str) -> str:
    """
    Convert file/test name like 'college_mathematics_test_73' → 'college_mathematics'
    """
    base = Path(name).stem
    # strip any trailing _eval
    base = re.sub(r'_eval$', '', base)
    # remove suffix _test... or _val...
    base = re.sub(r'_(test|val)(_\d+)?$', '', base)
    # remove _baseline suffix if present
    base = re.sub(r'_baseline$', '', base)
    return base

def subcats_for_subject(subject: str) -> List[str]:
    return SUBCATEGORY.get(subject, ["other"])

def meta_for_subject(subject: str) -> str:
    subcats = subcats_for_subject(subject)
    # choose first subcat if multiple
    sub = subcats[0] if subcats else "other"
    for meta, subs in CATEGORIES.items():
        if sub in subs:
            return meta
    return "other (business, health, misc.)"

# ---------------------------
# Parse baseline .txt files
# ---------------------------
def parse_baseline_txt(filepath: str) -> Dict:
    """
    Parse 6-line baseline .txt file format:
    Line 1: [True, False, ...] 0.7234
    Line 2: 100 1.0
    Line 3: []
    Line 4: []
    Line 5: 12500
    Line 6: 3200
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError(f"Invalid baseline file {filepath}: expected 6 lines, got {len(lines)}")

    # Line 1: [True, False, ...] accuracy
    line1 = lines[0].strip()
    if ' ' in line1:
        accs_str, acc_str = line1.rsplit(' ', 1)
        try:
            accs = eval(accs_str)  # [True, False, ...]
        except:
            raise ValueError(f"Cannot parse correctness list from {filepath} line 1")
    else:
        raise ValueError(f"Invalid format in {filepath} line 1")

    # Line 2: total_questions avg_responses
    total_questions = int(lines[1].strip().split()[0])

    # Line 5-6: tokens
    prompt_tokens = int(lines[4].strip())
    completion_tokens = int(lines[5].strip())

    # Count correct
    correct = sum(accs)

    return {
        'questions': total_questions,
        'correct': correct,
        'responses': total_questions,  # Always 1:1 for baseline
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens
    }

# ---------------------------
# Bootstrap CI (from exp_mmlu_evaluation.sh:460-524)
# ---------------------------
def bootstrap_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0):
    """
    Bootstrap across tests (block bootstrap).
    df must have columns: questions, correct, responses, prompt_tokens, completion_tokens
    Returns dict of metric -> (point, [lo, hi]).
    """
    if df.empty:
        return {}

    rng = np.random.default_rng(seed)
    n = len(df)

    # point estimates (no bootstrap)
    q_sum = df['questions'].sum()
    acc_point = float(df['correct'].sum()) / q_sum if q_sum > 0 else float('nan')
    api_point = df['responses'].sum()
    tin_point = df['prompt_tokens'].sum(skipna=True)
    tout_point = df['completion_tokens'].sum(skipna=True)

    acc_samps, api_samps, tin_samps, tout_samps = [], [], [], []
    have_tin = df['prompt_tokens'].notna().any()
    have_tout = df['completion_tokens'].notna().any()

    for _ in range(n_boot):
        idx = rng.integers(low=0, high=n, size=n)
        s = df.iloc[idx]
        q = s['questions'].sum()
        acc = float(s['correct'].sum()) / q if q > 0 else float('nan')
        api = s['responses'].sum()
        tin = s['prompt_tokens'].sum(skipna=True) if have_tin else np.nan
        tout = s['completion_tokens'].sum(skipna=True) if have_tout else np.nan

        acc_samps.append(acc)
        api_samps.append(api)
        tin_samps.append(tin)
        tout_samps.append(tout)

    def pct_ci(arr):
        arr = np.array(arr, dtype=float)
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        return [float(lo), float(hi)]

    out = {
        'accuracy': {'point': float(acc_point), 'ci95': pct_ci(acc_samps)},
        'api_calls': {'point': int(api_point), 'ci95': [int(np.nanpercentile(api_samps, 2.5)), int(np.nanpercentile(api_samps, 97.5))]},
    }

    if have_tin:
        out['tokens_in'] = {'point': float(tin_point), 'ci95': pct_ci(tin_samps)}
    else:
        out['tokens_in'] = {'point': float('nan'), 'ci95': [float('nan'), float('nan')]}
    if have_tout:
        out['tokens_out'] = {'point': float(tout_point), 'ci95': pct_ci(tout_samps)}
    else:
        out['tokens_out'] = {'point': float('nan'), 'ci95': [float('nan'), float('nan')]}

    return out

def print_block(title: str, res: dict):
    """Print formatted metrics block (from exp_mmlu_evaluation.sh:526-544)"""
    print(f"\n{title}")
    print("-" * len(title))
    if not res:
        print("No data.")
        return
    acc = res['accuracy']
    api = res['api_calls']
    print(f"Accuracy             : {acc['point']:.4f}  [95% CI {acc['ci95'][0]:.4f}, {acc['ci95'][1]:.4f}]")
    print(f"API calls            : {api['point']}  [95% CI {api['ci95'][0]}, {api['ci95'][1]}]")

    ti = res.get('tokens_in', None)
    to = res.get('tokens_out', None)
    if ti and not (math.isnan(ti['point']) or math.isnan(ti['ci95'][0])):
        print(f"Tokens in            : {ti['point']:.0f}  [95% CI {ti['ci95'][0]:.0f}, {ti['ci95'][1]:.0f}]")
    else:
        print("Tokens in            : N/A")
    if to and not (math.isnan(to['point']) or math.isnan(to['ci95'][0])):
        print(f"Tokens out           : {to['point']:.0f}  [95% CI {to['ci95'][0]:.0f}, {to['ci95'][1]:.0f}]")
    else:
        print("Tokens out           : N/A")

# ---------------------------
# Main
# ---------------------------
def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        print("\nUsage: compute_baseline_metrics.py <output_dir> <model> [n_boot]")
        sys.exit(1)

    output_dir = sys.argv[1]
    model = sys.argv[2]
    n_boot = int(sys.argv[3]) if len(sys.argv) == 4 else 1000

    if not os.path.isdir(output_dir):
        print(f"ERROR: Output directory not found: {output_dir}")
        sys.exit(1)

    # Find all *_baseline.txt files
    txt_files = [f for f in os.listdir(output_dir) if f.endswith('_baseline.txt')]
    if not txt_files:
        print(f"ERROR: No *_baseline.txt files found in {output_dir}")
        sys.exit(1)

    txt_files.sort()
    print(f"Found {len(txt_files)} baseline result files")
    print(f"Model: {model}")
    print(f"Bootstrap replicates: {n_boot}")

    # Parse all baseline files
    results = []
    for txt_file in txt_files:
        filepath = os.path.join(output_dir, txt_file)
        test_name = Path(txt_file).stem  # e.g., "abstract_algebra_test_baseline"

        try:
            metrics = parse_baseline_txt(filepath)
            subject = subject_key_from_name(test_name)
            meta = meta_for_subject(subject)

            results.append({
                'test_name': test_name,
                'subject': subject,
                'meta': meta,
                **metrics
            })
        except Exception as e:
            print(f"WARNING: Failed to parse {txt_file}: {e}")
            continue

    if not results:
        print("ERROR: No valid results parsed")
        sys.exit(1)

    df = pd.DataFrame(results)

    print("\n" + "="*60)
    print("BASELINE EVALUATION RESULTS WITH BOOTSTRAP CIs")
    print("="*60)

    # Save per-test JSON
    per_test_json = os.path.join(output_dir, "baseline_by_test.json")
    with open(per_test_json, "w") as f:
        json.dump({r['test_name']: r for r in results}, f, indent=2)

    # Compute overall metrics
    overall = bootstrap_ci(df, n_boot=n_boot, seed=0)
    print_block("OVERALL (Single-LLM baseline)", overall)

    # Build summary structure
    summary = {
        'notes': {
            'mode': 'single_baseline',
            'n_boot': n_boot,
            'model': model
        },
        'overall': overall,
        'by_meta': {}
    }

    # Compute per-meta-category metrics
    for meta in CATEGORIES.keys():
        sub = df[df['meta'] == meta]
        if not sub.empty:
            res = bootstrap_ci(sub, n_boot=n_boot, seed=hash(meta) % (2**32))
            print_block(f"{meta} — Single-LLM baseline", res)
            summary['by_meta'][meta] = res
        else:
            summary['by_meta'][meta] = {}

    # Save summary JSON
    summary_json = os.path.join(output_dir, "metrics_summary_baseline.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print(f"Per-test results saved to: {per_test_json}")
    print(f"Summary metrics saved to: {summary_json}")
    print("="*60)

if __name__ == "__main__":
    main()
