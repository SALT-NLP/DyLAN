#!/usr/bin/env bash
set -euo pipefail

# DyLAN MMLU Evaluation + CI Script
# - Runs reduced-role evaluation (post-selection)
# - Computes pre-selection (7-role) metrics from importance_1to7.csv
# - Reports accuracy, API calls, tokens-in, tokens-out with 95% bootstrap CIs
# - Breaks out metrics overall and by meta-category groups

MODEL="${MODEL:-openai/gpt-oss-20b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLES="['Economist','Doctor','Lawyer','Mathematician','Psychologist','Programmer','Historian']"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
NUM_ROLES="${NUM_ROLES:-4}"   # roles selected per question in evaluation
N_BOOT="${N_BOOT:-1000}"      # bootstrap replicates for CI

# MMR diversity parameters (NEW)
LAMBDA="${LAMBDA:-1.0}"       # MMR trade-off: 1.0=pure importance, 0.0=pure diversity
EMBEDDINGS="${EMBEDDINGS:-$SCRIPT_DIR/embeddings_agent_subject.pkl}"  # embeddings file
USE_MMR="${USE_MMR:-auto}"    # auto, true, or false

# Default paths
IMPORTANCE_CSV="${IMPORTANCE_CSV:-importance_1to7.csv}"
EVAL_DATASET="${EVAL_DATASET:-$SCRIPT_DIR/../../data/MMLU/evaluation}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation_results}"

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run reduced-role evaluation and report accuracy, API calls, tokens-in, tokens-out
with 95% bootstrap CIs (overall and by meta-categories), and pre- vs post-selection.

OPTIONS:
    -m, --model MODEL              LLM model to use (default: openai/gpt-oss-20b)
    -i, --importance-csv FILE      Path to importance CSV file (default: importance_1to7.csv)
    -d, --dataset DIR              Path to evaluation dataset directory (default: ../../data/MMLU/evaluation)
    -o, --output DIR               Output directory (default: evaluation_results)
    -n, --num-roles NUM            Number of roles to select per question for evaluation (default: 4)
    -p, --max-parallel NUM         Maximum parallel jobs (default: 4)
    --n-boot NUM                   Bootstrap replicates for CI (default: 1000)

    MMR DIVERSITY OPTIONS (NEW):
    --lambda LAMBDA                MMR trade-off parameter (default: 1.0)
                                   1.0 = pure importance (greedy, current behavior)
                                   0.7-0.8 = slight diversity preference (recommended)
                                   0.5 = equal weight to importance and diversity
                                   0.0 = pure diversity
    --embeddings FILE              Path to embeddings file (default: embeddings_agent_subject.pkl)
    --use-mmr [auto|true|false]    Enable MMR selection (default: auto)
                                   auto = use MMR if lambda < 1.0 and embeddings exist
                                   true = require MMR (fail if embeddings missing)
                                   false = always use greedy selection

    -h, --help                     Show help

EXAMPLES:
    # Current behavior (greedy selection by importance)
    $0

    # Balanced importance and diversity
    $0 --lambda 0.7

    # Equal weight to importance and diversity
    $0 --lambda 0.5 --num-roles 4

    # Pure diversity (experimental)
    $0 --lambda 0.0

    # Use custom embeddings file
    $0 --lambda 0.7 --embeddings my_embeddings.pkl

    # Other options
    $0 --model "gpt-4" --dataset "/path/to/eval"
    $0 --num-roles 3 --lambda 0.8
    N_BOOT=2000 $0 --lambda 0.7              # change # bootstrap reps
EOF
}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model) MODEL="$2"; shift 2;;
        -i|--importance-csv) IMPORTANCE_CSV="$2"; shift 2;;
        -d|--dataset) EVAL_DATASET="$2"; shift 2;;
        -o|--output) OUTPUT_DIR="$2"; shift 2;;
        -n|--num-roles) NUM_ROLES="$2"; shift 2;;
        -p|--max-parallel) MAX_PARALLEL="$2"; shift 2;;
        --n-boot) N_BOOT="$2"; shift 2;;
        --lambda) LAMBDA="$2"; shift 2;;
        --embeddings) EMBEDDINGS="$2"; shift 2;;
        --use-mmr) USE_MMR="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done

if [[ ! -f "$IMPORTANCE_CSV" ]]; then
    log "ERROR: Importance CSV not found: $IMPORTANCE_CSV"
    exit 1
fi
if [[ ! -d "$EVAL_DATASET" ]]; then
    log "ERROR: Evaluation dataset not found: $EVAL_DATASET"
    exit 1
fi
if [[ ! "$NUM_ROLES" =~ ^[1-7]$ ]]; then
    log "ERROR: Number of roles must be in [1..7], got: $NUM_ROLES"
    exit 1
fi

# Determine MMR usage
ACTUAL_MMR="false"
if [[ "$USE_MMR" == "true" ]]; then
    if [[ ! -f "$EMBEDDINGS" ]]; then
        log "ERROR: MMR requested but embeddings file not found: $EMBEDDINGS"
        exit 1
    fi
    ACTUAL_MMR="true"
elif [[ "$USE_MMR" == "auto" ]]; then
    # Use MMR if lambda < 1.0 and embeddings exist
    if (( $(echo "$LAMBDA < 1.0" | bc -l) )) && [[ -f "$EMBEDDINGS" ]]; then
        ACTUAL_MMR="true"
    fi
fi

log "Starting DyLAN MMLU Evaluation"
log "Model: $MODEL"
log "Importance CSV: $IMPORTANCE_CSV"
log "Dataset: $EVAL_DATASET"
log "Output: $OUTPUT_DIR"
log "Roles per question: $NUM_ROLES"
log "Max parallel jobs: $MAX_PARALLEL"
log "Bootstrap reps (95% CI): $N_BOOT"
log "---"
log "MMR Selection: $ACTUAL_MMR"
if [[ "$ACTUAL_MMR" == "true" ]]; then
    log "Lambda (importance weight): $LAMBDA"
    log "Embeddings file: $EMBEDDINGS"
fi

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------
# Python driver: selection + evaluation + metrics + bootstrap CIs
# ---------------------------------------------------------------------
cat > "$OUTPUT_DIR/evaluate_roles.py" << 'EOF'
#!/usr/bin/env python3
import os, sys, re, json, ast, math, subprocess
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import concurrent.futures
from functools import partial

# ---------------------------
# Subject → subcategory map
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
# Helpers
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
# Role selection (with optional MMR)
# ---------------------------
def select_top_roles(importance_csv: str, num_roles: int = 4, use_mmr: bool = False,
                     lambda_param: float = 1.0, embeddings_pkl: str = None) -> Tuple[Dict[str, List[str]], List[str], pd.DataFrame]:
    """
    Select top roles for each test, optionally using MMR for diversity.

    Args:
        importance_csv: Path to importance CSV
        num_roles: Number of roles to select
        use_mmr: Whether to use MMR selection
        lambda_param: MMR trade-off (1.0=importance, 0.0=diversity)
        embeddings_pkl: Path to embeddings file (required if use_mmr=True)
    """
    if use_mmr and embeddings_pkl:
        # Use MMR selection from mmr_selection module
        import sys
        # This script is in the output directory, look in parent directory for mmr_selection.py
        script_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(script_dir)
        mmr_module_path = os.path.join(parent_dir, 'mmr_selection.py')
        if not os.path.exists(mmr_module_path):
            print(f"Warning: mmr_selection.py not found at {mmr_module_path}, falling back to greedy selection")
            use_mmr = False
        else:
            # Import select_agents_for_all_tests from mmr_selection
            sys.path.insert(0, parent_dir)
            try:
                from mmr_selection import select_agents_for_all_tests
                print(f"Using MMR selection (lambda={lambda_param})")
                return select_agents_for_all_tests(importance_csv, embeddings_pkl, num_roles, lambda_param)
            except Exception as e:
                print(f"Warning: MMR selection failed ({e}), falling back to greedy")
                use_mmr = False

    # Greedy selection (original behavior)
    df = pd.read_csv(importance_csv)
    role_cols = [c for c in df.columns if c.endswith('_imp')]
    role_names = [c.replace('_imp', '') for c in role_cols]
    selected = {}
    for _, row in df.iterrows():
        fname = row['filename']
        scores = [(role_names[i], row[role_cols[i]]) for i in range(len(role_cols))]
        scores.sort(key=lambda x: x[1], reverse=True)
        top = [r for r, _ in scores[:num_roles]]
        selected[fname] = top
    return selected, role_names, df

# ---------------------------
# Evaluation
# ---------------------------
def run_evaluation(test_file: str, selected_roles: Dict[str, List[str]], model: str, all_roles: List[str], out_dir: str) -> str:
    filename = Path(test_file).stem

    # try to match importance key
    lookup = filename
    if filename not in selected_roles:
        if filename.endswith('_val'):
            lookup = filename.replace('_val', '_test')
        elif filename.endswith('_test'):
            lookup = filename.replace('_test', '_val')
        elif '_test' not in filename and '_val' not in filename:
            lookup = filename + '_test'

    if lookup not in selected_roles:
        print(f"Warning: No importance for {filename} (tried {lookup}); skipping")
        return None

    roles = selected_roles[lookup]
    test_roles_str = str(roles)

    exp_name = f"eval_{filename}"
    roles_str_clean = test_roles_str.replace(' ', '').replace('[','').replace(']','').replace(',', '_').replace("'", '')
    case_dir = os.path.join(out_dir, f"{exp_name}_{roles_str_clean}")
    os.makedirs(case_dir, exist_ok=True)

    result_file = os.path.join(case_dir, f"{filename}_eval.txt")
    if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
        print(f"Skipping {filename} (already processed)")
        return result_file

    print(f"Evaluating {filename} with roles: {roles}")
    # run llmlp_listwise_mmlu.py in OUTPUT_DIR so it writes there
    script_dir = os.path.dirname(os.path.abspath(__file__))    # .../code/MMLU/evaluation_results
    mmlu_dir   = os.path.dirname(script_dir)                   # .../code/MMLU
    llmlp = os.path.join(mmlu_dir, 'llmlp_listwise_mmlu.py')
    if not os.path.exists(llmlp):
        raise FileNotFoundError(
            f"llmlp_listwise_mmlu.py not found at {llmlp}. "
            f"Expected alongside this script's parent directory."
        )

    expected_txt = os.path.join(case_dir, f"{exp_name}_{len(roles)}3.txt")
    expected_json = os.path.join(case_dir, f"{exp_name}_{len(roles)}3.json")

    cmd = ['python', llmlp, test_file, exp_name, model, exp_name, test_roles_str]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=out_dir)
    if res.returncode != 0:
        print(f"Error running evaluation for {filename}:\n{res.stderr}")
        if res.stdout: print("STDOUT:", res.stdout)
        return None

    # move outputs to stable names
    if os.path.exists(expected_txt):
        import shutil
        if expected_txt != result_file:
            shutil.move(expected_txt, result_file)
        json_out = result_file.replace('.txt', '.json')
        if os.path.exists(expected_json) and expected_json != json_out:
            shutil.move(expected_json, json_out)
        return result_file
    else:
        print(f"Warning: expected file not found: {expected_txt}")
        if res.stdout: print("STDOUT:", res.stdout)
        return None

# ---------------------------
# Parse evaluation outputs
# ---------------------------
def parse_result_file(result_file: str):
    """
    Expected format (6+ lines):
      0: [True, False, ...] 0.5
      1: <total_responses> <avg_responses>
      2: [[...]]                (importance matrix)
      3: [...]                  (avg importances)
      4: <prompt_tokens>
      5: <completion_tokens>
    """
    with open(result_file, 'r') as f:
        lines = f.readlines()
    accs_parts = lines[0].strip().rsplit(' ', 1)
    accs = ast.literal_eval(accs_parts[0])
    resp_parts = lines[1].strip().split(' ', 1)
    total_resp = int(resp_parts[0])
    prompt_tokens = int(lines[4].strip())
    completion_tokens = int(lines[5].strip())
    q = len(accs)
    c = sum(1 for a in accs if a)
    return {
        'questions': q,
        'correct': c,
        'responses': total_resp,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens
    }

def collect_post_metrics(result_files: List[str]) -> pd.DataFrame:
    rows = []
    for rf in result_files:
        if not rf or not os.path.exists(rf): continue
        test_name = Path(rf).stem.replace('_eval', '')
        m = parse_result_file(rf)
        subj = subject_key_from_name(test_name)
        meta = meta_for_subject(subj)
        rows.append({
            'test_name': test_name,
            'subject': subj,
            'meta': meta,
            **m
        })
    return pd.DataFrame(rows)

# ---------------------------
# Pre-selection metrics (7 roles)
# ---------------------------
def collect_pre_metrics(importance_csv: str) -> pd.DataFrame:
    df = pd.read_csv(importance_csv)
    rows = []
    # expected cols: filename, acc, resp, q_cnt (tokens may or may not exist)
    has_tok_in = 'prompt_tokens' in df.columns
    has_tok_out = 'completion_tokens' in df.columns
    for _, r in df.iterrows():
        fname = r['filename']
        subj = subject_key_from_name(fname)
        meta = meta_for_subject(subj)
        q_cnt = int(r['q_cnt'])
        # 'acc' is per-question average accuracy; keep fractional * q_cnt for aggregation
        correct_float = float(r['acc']) * q_cnt
        responses = int(r['resp'])
        row = {
            'test_name': Path(fname).stem,   # keep original stem
            'subject': subj,
            'meta': meta,
            'questions': q_cnt,
            'correct_float': correct_float,  # keep float; accuracy sums stay exact
            'responses': responses
        }
        if has_tok_in:  row['prompt_tokens'] = int(r['prompt_tokens'])
        if has_tok_out: row['completion_tokens'] = int(r['completion_tokens'])
        rows.append(row)
    d = pd.DataFrame(rows)
    # standardize for downstream
    if 'correct' not in d.columns:
        # we keep float counts (no rounding) for accuracy aggregation
        d['correct'] = d['correct_float']
    if 'prompt_tokens' not in d.columns:
        d['prompt_tokens'] = np.nan
    if 'completion_tokens' not in d.columns:
        d['completion_tokens'] = np.nan
    return d[['test_name','subject','meta','questions','correct','responses','prompt_tokens','completion_tokens']]

# ---------------------------
# Bootstrap & summaries
# ---------------------------
def bootstrap_ci(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0, scale_tokens_by: float = None):
    """
    Bootstrap across tests (blocks bootstrap).
    If scale_tokens_by is provided (e.g., R), tokens are multiplied by this constant
    in each replicate (used to estimate pre tokens from post tokens).
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

    if scale_tokens_by is not None:
        tin_point = float(scale_tokens_by) * tin_point
        tout_point = float(scale_tokens_by) * tout_point

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
        if scale_tokens_by is not None:
            tin = (float(scale_tokens_by) * tin) if not np.isnan(tin) else np.nan
            tout = (float(scale_tokens_by) * tout) if not np.isnan(tout) else np.nan

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

def print_block(title: str, res: dict, mark_est_tokens: bool = False):
    print(f"\n{title}")
    print("-" * len(title))
    acc = res['accuracy']; api = res['api_calls']
    print(f"Accuracy             : {acc['point']:.4f}  [95% CI {acc['ci95'][0]:.4f}, {acc['ci95'][1]:.4f}]")
    print(f"API calls            : {api['point']}  [95% CI {api['ci95'][0]}, {api['ci95'][1]}]")

    ti = res.get('tokens_in', None)
    to = res.get('tokens_out', None)
    if ti and not (math.isnan(ti['point']) or math.isnan(ti['ci95'][0])):
        est_tag = " (est.)" if mark_est_tokens else ""
        print(f"Tokens in            : {ti['point']:.0f}{est_tag}  [95% CI {ti['ci95'][0]:.0f}, {ti['ci95'][1]:.0f}]")
    else:
        print("Tokens in            : N/A")
    if to and not (math.isnan(to['point']) or math.isnan(to['ci95'][0])):
        est_tag = " (est.)" if mark_est_tokens else ""
        print(f"Tokens out           : {to['point']:.0f}{est_tag}  [95% CI {to['ci95'][0]:.0f}, {to['ci95'][1]:.0f}]")
    else:
        print("Tokens out           : N/A")

def main():
    if len(sys.argv) not in (7, 8, 10, 11):
        print("Usage: evaluate_roles.py <importance_csv> <dataset_dir> <model> <num_roles> <output_dir> <max_parallel> [<n_boot>] [<use_mmr> <lambda> <embeddings_pkl>]")
        sys.exit(1)
    importance_csv = sys.argv[1]
    dataset_dir   = sys.argv[2]
    model         = sys.argv[3]
    num_roles     = int(sys.argv[4])
    output_dir    = sys.argv[5]
    max_parallel  = int(sys.argv[6])
    n_boot        = int(sys.argv[7]) if len(sys.argv) >= 8 else 1000

    # MMR parameters (if provided)
    use_mmr = False
    lambda_param = 1.0
    embeddings_pkl = None
    if len(sys.argv) >= 11:
        use_mmr = sys.argv[8].lower() == 'true'
        lambda_param = float(sys.argv[9])
        embeddings_pkl = sys.argv[10]

    print(f"Loading importance data from: {importance_csv}")
    selected_roles, all_roles, df_imp = select_top_roles(
        importance_csv, num_roles, use_mmr, lambda_param, embeddings_pkl
    )
    print(f"Found importance data for {len(selected_roles)} tests")
    print(f"Selected {num_roles} roles per test from: {all_roles}")

    # gather candidate csv files
    tests = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    print(f"Found {len(tests)} test files")
    # filter to only those we have importance for (with suffix reconciliation)
    to_proc = []
    for tf in tests:
        fn = Path(tf).stem
        lookup = fn
        if fn not in selected_roles:
            if fn.endswith('_val'):
                lookup = fn.replace('_val', '_test')
            elif fn.endswith('_test'):
                lookup = fn.replace('_test', '_val')
            elif '_test' not in fn and '_val' not in fn:
                lookup = fn + '_test'
        if lookup in selected_roles:
            to_proc.append(tf)
    print(f"Processing {len(to_proc)} test files with importance data")

    # Run reduced-role evals in parallel
    result_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
        fn = partial(run_evaluation, selected_roles=selected_roles, model=model, all_roles=all_roles, out_dir=output_dir)
        fut = {ex.submit(fn, tf): tf for tf in to_proc}
        for f in concurrent.futures.as_completed(fut):
            rf = f.result()
            if rf: result_files.append(rf)

    # Collect metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    post_df = collect_post_metrics(result_files)
    pre_df  = collect_pre_metrics(importance_csv)

    # Check if we have any results
    if post_df.empty:
        print("\n⚠️  ERROR: No evaluation results collected!")
        print("   All evaluations failed or were skipped.")
        print("   Check the error messages above for details.")
        sys.exit(1)

    # Align sets by subject where needed
    # (We compute global response ratio using all available rows.)
    total_resp_pre  = pre_df['responses'].sum()
    total_resp_post = post_df['responses'].sum()
    R = (float(total_resp_pre) / float(total_resp_post)) if total_resp_post > 0 else float('nan')

    # Overall summaries
    post_overall = bootstrap_ci(post_df, n_boot=n_boot, seed=0, scale_tokens_by=None)
    # If pre has tokens recorded, use them; if not, estimate via R from post tokens
    pre_has_tok = pre_df['prompt_tokens'].notna().any() and pre_df['completion_tokens'].notna().any()
    pre_overall = bootstrap_ci(pre_df, n_boot=n_boot, seed=1,
                               scale_tokens_by=None if pre_has_tok else R)

    print_block("OVERALL (Post-selection / reduced roles)", post_overall, mark_est_tokens=False)
    print_block("OVERALL (Pre-selection / 7 roles)", pre_overall, mark_est_tokens=(not pre_has_tok))

    # Per meta-category
    categories = list(CATEGORIES.keys())
    summary = {
        'notes': {},
        'overall': {'post': post_overall, 'pre': pre_overall},
        'by_meta': {}
    }

    for meta in categories:
        post_cat = post_df[post_df['meta'] == meta]
        pre_cat  = pre_df[pre_df['meta'] == meta]
        # For token estimation at category level we keep using global R for stability.
        post_cat_res = bootstrap_ci(post_cat, n_boot=n_boot, seed=hash(meta) % (2**32))
        pre_cat_has_tok = pre_cat['prompt_tokens'].notna().any() and pre_cat['completion_tokens'].notna().any()
        pre_cat_res = bootstrap_ci(pre_cat, n_boot=n_boot, seed=(hash(meta)+1) % (2**32),
                                   scale_tokens_by=None if pre_cat_has_tok else R)

        print_block(f"{meta} — Post-selection", post_cat_res, mark_est_tokens=False)
        print_block(f"{meta} — Pre-selection", pre_cat_res, mark_est_tokens=(not pre_cat_has_tok))

        summary['by_meta'][meta] = {
            'post': post_cat_res,
            'pre': pre_cat_res
        }

    # Print notes clearly
    print("\nNOTES")
    print("-----")
    print("* API calls are computed as the total number of model responses (sum over all questions).")
    if not pre_has_tok:
        print(f"* Pre-selection tokens were NOT logged. We therefore estimate pre tokens as:")
        print(f"    tokens_pre ≈ R × tokens_post, where R = total_responses_pre / total_responses_post = {R:.4f}")
        print(f"  Estimated values are marked with '(est.)' above.")
        summary['notes']['pre_tokens_estimated'] = True
        summary['notes']['response_ratio_R'] = R
    else:
        summary['notes']['pre_tokens_estimated'] = False

    # Save JSON
    out_json = os.path.join(output_dir, f"metrics_summary_{num_roles}roles.json")
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed JSON saved to: {out_json}")

if __name__ == "__main__":
    main()
EOF

chmod +x "$OUTPUT_DIR/evaluate_roles.py"

# ---------------------------------------------------------------------
# Run evaluation + metrics (with bootstrap)
# ---------------------------------------------------------------------
log "Running evaluation with $NUM_ROLES roles per question..."

# Build python command with optional MMR parameters
if [[ "$ACTUAL_MMR" == "true" ]]; then
    python "$OUTPUT_DIR/evaluate_roles.py" \
        "$IMPORTANCE_CSV" \
        "$EVAL_DATASET" \
        "$MODEL" \
        "$NUM_ROLES" \
        "$OUTPUT_DIR" \
        "$MAX_PARALLEL" \
        "$N_BOOT" \
        "$ACTUAL_MMR" \
        "$LAMBDA" \
        "$EMBEDDINGS"
else
    python "$OUTPUT_DIR/evaluate_roles.py" \
        "$IMPORTANCE_CSV" \
        "$EVAL_DATASET" \
        "$MODEL" \
        "$NUM_ROLES" \
        "$OUTPUT_DIR" \
        "$MAX_PARALLEL" \
        "$N_BOOT"
fi

log "Evaluation completed!"
log "Results saved in: $OUTPUT_DIR"
