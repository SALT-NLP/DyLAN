#!/usr/bin/env bash
set -euo pipefail

# DyLAN MATH Evaluation Script (analogous to MMLU evaluation)
# - Uses importance_math_1to7.csv (7-role math importance)
# - For each math subject, selects top-K roles
# - Runs llmlp_listwise_math.py on data/math_json/evaluation/<subject>
# - Reports accuracy, API calls, tokens with 95% bootstrap CIs

MODEL="${MODEL:-openai/gpt-oss-20b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
NUM_ROLES="${NUM_ROLES:-4}"   # roles selected per subject in evaluation
N_BOOT="${N_BOOT:-1000}"      # bootstrap replicates for CI

# 默认输入：在 code/MMLU 目录下的 importance_math_1to7.csv
IMPORTANCE_CSV="${IMPORTANCE_CSV:-importance_math_1to7.csv}"

# math evaluation 数据集根目录（注意是 evaluation 这一层）
MATH_EVAL_ROOT="${MATH_EVAL_ROOT:-$SCRIPT_DIR/../../data/math_json/evaluation}"

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-math_evaluation_results}"

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run reduced-role evaluation on MATH dataset and report accuracy, API calls,
tokens-in, tokens-out with 95% bootstrap CIs (overall and by math subcategory).

OPTIONS:
    -m, --model MODEL              LLM model to use (default: openai/gpt-oss-20b)
    -i, --importance-csv FILE      Path to math importance CSV (default: importance_math_1to7.csv)
    -d, --dataset DIR              Path to math evaluation root (default: ../../data/math_json/evaluation)
    -o, --output DIR               Output directory (default: math_evaluation_results)
    -n, --num-roles NUM            Number of roles to select per subject (default: 4)
    -p, --max-parallel NUM         Maximum parallel jobs (default: 4)
    --n-boot NUM                   Bootstrap replicates for CI (default: 1000)
    -h, --help                     Show help
EOF
}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model) MODEL="$2"; shift 2;;
        -i|--importance-csv) IMPORTANCE_CSV="$2"; shift 2;;
        -d|--dataset) MATH_EVAL_ROOT="$2"; shift 2;;
        -o|--output) OUTPUT_DIR="$2"; shift 2;;
        -n|--num-roles) NUM_ROLES="$2"; shift 2;;
        -p|--max-parallel) MAX_PARALLEL="$2"; shift 2;;
        --n-boot) N_BOOT="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done

if [[ ! -f "$IMPORTANCE_CSV" ]]; then
    log "ERROR: Importance CSV not found: $IMPORTANCE_CSV"
    exit 1
fi
if [[ ! -d "$MATH_EVAL_ROOT" ]]; then
    log "ERROR: Math evaluation root not found: $MATH_EVAL_ROOT"
    exit 1
fi
if [[ ! "$NUM_ROLES" =~ ^[1-7]$ ]]; then
    log "ERROR: Number of roles must be in [1..7], got: $NUM_ROLES"
    exit 1
fi

log "Starting DyLAN MATH Evaluation"
log "Model: $MODEL"
log "Importance CSV: $IMPORTANCE_CSV"
log "Math eval root: $MATH_EVAL_ROOT"
log "Output: $OUTPUT_DIR"
log "Roles per subject: $NUM_ROLES"
log "Max parallel jobs: $MAX_PARALLEL"
log "Bootstrap reps (95% CI): $N_BOOT"

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------
# 生成 Python driver: evaluate_math_roles.py
# ---------------------------------------------------------------
cat > "$OUTPUT_DIR/evaluate_math_roles.py" << 'EOF'
#!/usr/bin/env python3
import os, sys, re, json, ast, math, subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import concurrent.futures
from functools import partial

# ---------------------------
# math subjects & meta names
# ---------------------------
MATH_SUBJECTS = [
    "algebra_test",
    "counting_and_probability_test",
    "geometry_test",
    "intermediate_algebra_test",
    "number_theory_test",
    "prealgebra_test",
    "precalculus_test",
]

# 可选：更细粒度 meta 分类
SUBJECT_META = {
    "algebra_test": "algebra",
    "counting_and_probability_test": "counting_probability",
    "geometry_test": "geometry",
    "intermediate_algebra_test": "intermediate_algebra",
    "number_theory_test": "number_theory",
    "prealgebra_test": "prealgebra",
    "precalculus_test": "precalculus",
}

# ---------------------------
# Helpers
# ---------------------------
def subject_from_filename(fname: str) -> str:
    """
    根据 importance CSV 里的 filename 推断 math subject。
    规则：在 MATH_SUBJECTS 中找最长前缀匹配。
    例如: 'algebra_test_0001_0012' -> 'algebra_test'
    """
    for sub in sorted(MATH_SUBJECTS, key=len, reverse=True):
        if fname.startswith(sub):
            return sub
    return fname  # fallback: 原样返回

def meta_for_subject(subject: str) -> str:
    return SUBJECT_META.get(subject, "math")

# ---------------------------
# Role selection (per subject)
# ---------------------------
def select_top_roles_by_subject(importance_csv: str, num_roles: int = 4):
    df = pd.read_csv(importance_csv)

    # 角色列：以 _imp 结尾
    role_cols = [c for c in df.columns if c.endswith("_imp")]
    if not role_cols:
        raise ValueError(f"No *_imp columns found in {importance_csv}")
    role_names = [c.replace("_imp", "") for c in role_cols]

    # 如果 CSV 里没有 subject 列，就根据 filename 推断
    if "subject" in df.columns:
        df["subject"] = df["subject"].astype(str)
    else:
        df["subject"] = df["filename"].astype(str).apply(subject_from_filename)

    selected = {}   # subject -> [roles]
    avg_imp = {}    # subject -> dict(role -> avg_imp)

    for subject in sorted(df["subject"].unique()):
        sdf = df[df["subject"] == subject]
        # 对这个 subject 内所有行求每个角色的平均 importance
        mean_scores = sdf[role_cols].mean(axis=0).to_dict()
        # 转成 (role_name, score) 列表
        scores = [(role_names[i], float(mean_scores[role_cols[i]])) for i in range(len(role_cols))]
        scores.sort(key=lambda x: x[1], reverse=True)
        top = [r for r, _ in scores[:num_roles]]
        selected[subject] = top
        avg_imp[subject] = {r: s for r, s in scores}
    return selected, role_names, avg_imp

# ---------------------------
# 调用 llmlp_listwise_math.py 对一个 subject 做评估
# ---------------------------
def run_math_eval_for_subject(subject: str,
                              selected_roles: Dict[str, List[str]],
                              model: str,
                              math_eval_root: str,
                              out_dir: str) -> str:
    """
    对 evaluation/<subject> 目录里的所有 json 题目，用 selected_roles[subject]
    调用 llmlp_listwise_math.py 做多 agent 推理。
    """
    if subject not in selected_roles:
        print(f"Warning: subject {subject} has no selected roles; skip")
        return None

    roles = selected_roles[subject]
    roles_str = str(roles)

    subdir_path = os.path.join(math_eval_root, subject)
    if not os.path.isdir(subdir_path):
        print(f"Warning: eval subdir not found for subject {subject}: {subdir_path}")
        return None

    # 找到该 subject 下所有 NNNN.json
    json_files = [f for f in os.listdir(subdir_path) if f.endswith(".json")]
    if not json_files:
        print(f"Warning: no json files in {subdir_path}; skip")
        return None

    basenames = sorted(os.path.splitext(f)[0] for f in json_files)
    minf, maxf = basenames[0], basenames[-1]

    # exp_name_base 用一个统一前缀
    exp_name_base = "math_eval"
    # 目录名和 llmlp_listwise_math.py 的逻辑保持一致：
    # DIR_NAME = exp_name_base + '_' + '_'.join(roles)
    roles_joined = "_".join(roles)
    case_dir = os.path.join(out_dir, f"{exp_name_base}_{roles_joined}")
    os.makedirs(case_dir, exist_ok=True)

    # 预期输出文件名：
    # 在 llmlp_listwise_math.py 中，EXP_NAME = f"{SUBDIR_BASE}_{MIN}_{MAX}"
    subdir_base = Path(subdir_path).name
    exp_name = f"{subdir_base}_{minf}_{maxf}"
    result_txt = os.path.join(case_dir, f"{exp_name}_{len(roles)}3.txt")

    if os.path.exists(result_txt) and os.path.getsize(result_txt) > 0:
        print(f"Skipping {subject} (already evaluated): {result_txt}")
        return result_txt

    print(f"Evaluating subject={subject} with roles={roles}, files {minf}..{maxf}")

    # 找到 llmlp_listwise_math.py
    # 本脚本位于 OUTPUT_DIR，父目录是 code/MMLU
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mmlu_dir   = os.path.dirname(script_dir)
    llmlp = os.path.join(mmlu_dir, "llmlp_listwise_math.py")
    if not os.path.exists(llmlp):
        raise FileNotFoundError(f"llmlp_listwise_math.py not found at {llmlp}")

    cmd = [
        "python",
        llmlp,
        subdir_path,
        minf,
        maxf,
        model,
        exp_name_base,
        roles_str,
    ]
    # cwd=out_dir 保证输出目录在 math_evaluation_results 下面
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=out_dir)
    if res.returncode != 0:
        print(f"Error running math eval for {subject}:\n{res.stderr}")
        if res.stdout:
            print("STDOUT:", res.stdout)
        return None

    if not os.path.exists(result_txt):
        print(f"Warning: expected result file not found: {result_txt}")
        if res.stdout:
            print("STDOUT:", res.stdout)
        return None

    return result_txt

# ---------------------------
# 解析 result_txt（格式与 MMLU 一致）
# ---------------------------
def parse_result_file(result_file: str):
    with open(result_file, "r") as f:
        lines = f.readlines()
    if len(lines) < 6:
        raise ValueError(f"Unexpected result format in {result_file}")
    accs_parts = lines[0].strip().rsplit(" ", 1)
    accs = ast.literal_eval(accs_parts[0])
    resp_parts = lines[1].strip().split(" ", 1)
    total_resp = int(resp_parts[0])
    prompt_tokens = int(lines[4].strip())
    completion_tokens = int(lines[5].strip())
    q = len(accs)
    c = sum(1 for a in accs if a)
    return {
        "questions": q,
        "correct": c,
        "responses": total_resp,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }

def collect_post_metrics(result_files: List[str]) -> pd.DataFrame:
    rows = []
    for rf in result_files:
        if not rf or not os.path.exists(rf):
            continue
        # rf 形如: .../math_eval_<roles>/<subject_min_max>_43.txt
        name = Path(rf).stem          # subject_min_max_43
        parts = name.split("_")
        # subject 里本身有下划线，所以倒数三个是：min, max, "43"
        if len(parts) >= 4:
            subject = "_".join(parts[:-3])
        else:
            subject = name
        meta = meta_for_subject(subject)
        m = parse_result_file(rf)
        rows.append({
            "subject": subject,
            "meta": meta,
            **m,
        })
    return pd.DataFrame(rows)

# ---------------------------
# Pre-selection metrics (7 roles)
# ---------------------------
def collect_pre_metrics(importance_csv: str) -> pd.DataFrame:
    df = pd.read_csv(importance_csv)
    if "subject" in df.columns:
        df["subject"] = df["subject"].astype(str)
    else:
        df["subject"] = df["filename"].astype(str).apply(subject_from_filename)

    rows = []
    has_tok_in = "prompt_tokens" in df.columns
    has_tok_out = "completion_tokens" in df.columns

    for _, r in df.iterrows():
        subj = str(r["subject"])
        meta = meta_for_subject(subj)
        q_cnt = int(r["q_cnt"])
        acc = float(r["acc"])
        correct_float = acc * q_cnt
        responses = int(r["resp"])
        row = {
            "subject": subj,
            "meta": meta,
            "questions": q_cnt,
            "correct": correct_float,   # float 计数，用于更精确的加总
            "responses": responses,
        }
        if has_tok_in:
            row["prompt_tokens"] = int(r["prompt_tokens"])
        if has_tok_out:
            row["completion_tokens"] = int(r["completion_tokens"])
        rows.append(row)

    d = pd.DataFrame(rows)
    if "prompt_tokens" not in d.columns:
        d["prompt_tokens"] = np.nan
    if "completion_tokens" not in d.columns:
        d["completion_tokens"] = np.nan
    return d[["subject","meta","questions","correct","responses","prompt_tokens","completion_tokens"]]

# ---------------------------
# Bootstrap & summaries
# ---------------------------
def bootstrap_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0):
    if df.empty:
        return {}

    rng = np.random.default_rng(seed)
    n = len(df)

    q_sum = df["questions"].sum()
    acc_point = float(df["correct"].sum()) / q_sum if q_sum > 0 else float("nan")
    api_point = df["responses"].sum()
    tin_point = df["prompt_tokens"].sum(skipna=True)
    tout_point = df["completion_tokens"].sum(skipna=True)

    acc_samps, api_samps, tin_samps, tout_samps = [], [], [], []
    have_tin = df["prompt_tokens"].notna().any()
    have_tout = df["completion_tokens"].notna().any()

    for _ in range(n_boot):
        idx = rng.integers(low=0, high=n, size=n)
        s = df.iloc[idx]
        q = s["questions"].sum()
        acc = float(s["correct"].sum()) / q if q > 0 else float("nan")
        api = s["responses"].sum()
        tin = s["prompt_tokens"].sum(skipna=True) if have_tin else np.nan
        tout = s["completion_tokens"].sum(skipna=True) if have_tout else np.nan

        acc_samps.append(acc)
        api_samps.append(api)
        tin_samps.append(tin)
        tout_samps.append(tout)

    def pct_ci(arr):
        arr = np.array(arr, dtype=float)
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        return [float(lo), float(hi)]

    out = {
        "accuracy": {
            "point": float(acc_point),
            "ci95": pct_ci(acc_samps),
        },
        "api_calls": {
            "point": int(api_point),
            "ci95": [
                int(np.nanpercentile(api_samps, 2.5)),
                int(np.nanpercentile(api_samps, 97.5)),
            ],
        },
    }
    if have_tin:
        out["tokens_in"] = {"point": float(tin_point), "ci95": pct_ci(tin_samps)}
    else:
        out["tokens_in"] = {"point": float("nan"), "ci95": [float("nan"), float("nan")]}
    if have_tout:
        out["tokens_out"] = {"point": float(tout_point), "ci95": pct_ci(tout_samps)}
    else:
        out["tokens_out"] = {"point": float("nan"), "ci95": [float("nan"), float("nan")]}
    return out

def print_block(title: str, res: dict):
    print(f"\n{title}")
    print("-" * len(title))
    acc = res["accuracy"]
    api = res["api_calls"]
    print(f"Accuracy   : {acc['point']:.4f}  [95% CI {acc['ci95'][0]:.4f}, {acc['ci95'][1]:.4f}]")
    print(f"API calls  : {api['point']}  [95% CI {api['ci95'][0]}, {api['ci95'][1]}]")

    tin = res.get("tokens_in", None)
    tout = res.get("tokens_out", None)
    if tin and not (math.isnan(tin["point"]) or math.isnan(tin["ci95"][0])):
        print(f"Tokens in  : {tin['point']:.0f}  [95% CI {tin['ci95'][0]:.0f}, {tin['ci95'][1]:.0f}]")
    else:
        print("Tokens in  : N/A")
    if tout and not (math.isnan(tout["point"]) or math.isnan(tout["ci95"][0])):
        print(f"Tokens out : {tout['point']:.0f}  [95% CI {tout['ci95'][0]:.0f}, {tout['ci95'][1]:.0f}]")
    else:
        print("Tokens out : N/A")

# ---------------------------
# main
# ---------------------------
def main():
    if len(sys.argv) != 8:
        print("Usage: evaluate_math_roles.py <importance_csv> <math_eval_root> <model> <num_roles> <output_dir> <max_parallel> <n_boot>")
        sys.exit(1)

    importance_csv = sys.argv[1]
    math_eval_root = sys.argv[2]
    model          = sys.argv[3]
    num_roles      = int(sys.argv[4])
    out_dir        = sys.argv[5]
    max_parallel   = int(sys.argv[6])
    n_boot         = int(sys.argv[7])

    print(f"Loading importance data from: {importance_csv}")
    selected_roles, all_roles, avg_imp = select_top_roles_by_subject(importance_csv, num_roles)
    print("Subjects and selected roles:")
    for subj, roles in selected_roles.items():
        print(f"  {subj}: {roles}")

    # 需要评估的 subjects = MATH_SUBJECTS 与 selected_roles 的交集
    subjects = [s for s in MATH_SUBJECTS if s in selected_roles]
    print(f"\nEvaluating {len(subjects)} math subjects: {subjects}")

    result_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
        fn = partial(run_math_eval_for_subject,
                     selected_roles=selected_roles,
                     model=model,
                     math_eval_root=math_eval_root,
                     out_dir=out_dir)
        fut = {ex.submit(fn, s): s for s in subjects}
        for f in concurrent.futures.as_completed(fut):
            rf = f.result()
            if rf:
                result_files.append(rf)

    print("\n" + "="*60)
    print("MATH EVALUATION RESULTS")
    print("="*60)

    post_df = collect_post_metrics(result_files)
    pre_df  = collect_pre_metrics(importance_csv)

    # overall
    post_overall = bootstrap_ci(post_df, n_boot=n_boot, seed=0)
    pre_overall  = bootstrap_ci(pre_df,  n_boot=n_boot, seed=1)

    print_block("OVERALL (Post-selection / reduced roles)", post_overall)
    print_block("OVERALL (Pre-selection / 7 roles)",       pre_overall)

    # per meta-category (algebra / geometry / ...)
    metas = sorted(pre_df["meta"].unique())
    summary = {
        "overall": {"post": post_overall, "pre": pre_overall},
        "by_meta": {},
    }

    for meta in metas:
        post_cat = post_df[post_df["meta"] == meta]
        pre_cat  = pre_df[pre_df["meta"] == meta]
        post_res = bootstrap_ci(post_cat, n_boot=n_boot, seed=(hash("post_"+meta) % (2**32)))
        pre_res  = bootstrap_ci(pre_cat,  n_boot=n_boot, seed=(hash("pre_"+meta)  % (2**32)))
        print_block(f"{meta} — Post-selection", post_res)
        print_block(f"{meta} — Pre-selection",  pre_res)
        summary["by_meta"][meta] = {"post": post_res, "pre": pre_res}

    out_json = os.path.join(out_dir, f"math_metrics_summary_{num_roles}roles.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed JSON saved to: {out_json}")

if __name__ == "__main__":
    main()
EOF

chmod +x "$OUTPUT_DIR/evaluate_math_roles.py"

# ---------------------------------------------------------------
# 运行 math evaluation + metrics
# ---------------------------------------------------------------
log "Running math evaluation with $NUM_ROLES roles per subject..."
python "$OUTPUT_DIR/evaluate_math_roles.py" \
    "$IMPORTANCE_CSV" \
    "$MATH_EVAL_ROOT" \
    "$MODEL" \
    "$NUM_ROLES" \
    "$OUTPUT_DIR" \
    "$MAX_PARALLEL" \
    "$N_BOOT"

log "Math evaluation completed!"
log "Results saved in: $OUTPUT_DIR"
