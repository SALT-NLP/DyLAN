#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------
# DyLAN Simple Evaluation Script
# Function: Select roles based on importance scores and run experiments on dataset
# ------------------------------------------------------------

# Default parameters
IMPORTANCE_CSV="${1:-importance_1to7.csv}"
DATASET_DIR="${2:-../../data/MMLU/test}"
NUM_ROLES="${3:-4}"
MODEL="${4:-meta-llama/Llama-3.3-70B-Instruct-Turbo-Free}"

# Check parameters
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <importance_csv_file> <dataset_directory> [num_roles] [model]"
    echo ""
    echo "Parameters:"
    echo "  importance_csv_file: CSV file containing role importance scores"
    echo "  dataset_directory: MMLU test dataset directory"
    echo "  num_roles: Number of roles to use per test (default: 4)"
    echo "  model: LLM model to use (default: meta-llama/Llama-3.3-70B-Instruct-Turbo-Free)"
    echo ""
    echo "Examples:"
    echo "  $0 importance_1to7.csv ../../data/MMLU/test 4"
    echo "  $0 importance_1to7.csv ../../data/MMLU/test 3 gpt-4"
    exit 1
fi

# Check if files exist
if [[ ! -f "$IMPORTANCE_CSV" ]]; then
    echo "Error: Importance CSV file not found: $IMPORTANCE_CSV"
    exit 1
fi

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "Error: Dataset directory not found: $DATASET_DIR"
    exit 1
fi

echo "=== DyLAN Simple Evaluation Script ==="
echo "Importance file: $IMPORTANCE_CSV"
echo "Dataset directory: $DATASET_DIR"
echo "Number of roles per test: $NUM_ROLES"
echo "Model: $MODEL"
echo ""

# Create Python script to select roles
cat > select_roles.py << 'EOF'
import pandas as pd
import sys
import json

def select_roles(importance_csv, num_roles):
    """Select top N most important roles from importance CSV file"""
    df = pd.read_csv(importance_csv)
    
    # Get all role columns
    role_cols = [col for col in df.columns if col.endswith('_imp')]
    role_names = [col.replace('_imp', '') for col in role_cols]
    
    selected_roles = {}
    
    for _, row in df.iterrows():
        filename = row['filename']
        
        # Get importance scores for each role
        scores = [(role_names[i], row[role_cols[i]]) for i in range(len(role_cols))]
        
        # Sort by importance score and select top N
        scores.sort(key=lambda x: x[1], reverse=True)
        top_roles = [role for role, _ in scores[:num_roles]]
        
        selected_roles[filename] = top_roles
    
    return selected_roles

if __name__ == "__main__":
    importance_csv = sys.argv[1]
    num_roles = int(sys.argv[2])
    
    selected_roles = select_roles(importance_csv, num_roles)
    
    # Output as JSON format
    print(json.dumps(selected_roles, indent=2))
EOF

echo "1. Selecting roles from importance scores..."
python select_roles.py "$IMPORTANCE_CSV" "$NUM_ROLES" > selected_roles.json

echo "Found importance data for $(python -c "import json; data=json.load(open('selected_roles.json')); print(len(data))") tests"
echo ""

# Create evaluation script
cat > run_evaluation.py << 'EOF'
import json
import os
import sys
import subprocess
from pathlib import Path

def run_single_test(test_file, selected_roles, model):
    """Run experiment for a single test file"""
    filename = Path(test_file).stem
    
    if filename not in selected_roles:
        print(f"Skipping {filename}: no importance data")
        return None
    
    # Get selected roles for this test
    test_roles = selected_roles[filename]
    print(f"Testing {filename}, using roles: {test_roles}")
    
    # Create output directory
    exp_name = f"simple_eval_{filename}"
    roles_str = '_'.join(test_roles)
    dir_name = f"{exp_name}_{len(test_roles)}roles_{roles_str}"
    os.makedirs(dir_name, exist_ok=True)
    
    # Check if already run
    result_file = f"{dir_name}/{filename}_simple.txt"
    if os.path.exists(result_file):
        print(f"Skipping {filename}: already run")
        return result_file
    
    # Run experiment
    try:
        # Call original evaluation script
        roles_arg = str(test_roles)
        cmd = [
            "python", "llmlp_listwise_mmlu.py",
            test_file, filename, model, exp_name, roles_arg
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Move result file to our directory
            original_result = f"{exp_name}_{len(test_roles)}3.txt"
            if os.path.exists(original_result):
                os.rename(original_result, result_file)
                print(f"Result saved to: {result_file}")
                return result_file
        else:
            print(f"Experiment failed: {filename}")
            print(f"Error: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Error running {filename}: {e}")
        return None

def main():
    if len(sys.argv) != 4:
        print("Usage: run_evaluation.py <selected_roles.json> <dataset_dir> <model>")
        sys.exit(1)
    
    roles_file = sys.argv[1]
    dataset_dir = sys.argv[2]
    model = sys.argv[3]
    
    # Load selected roles
    with open(roles_file, 'r') as f:
        selected_roles = json.load(f)
    
    # Find test files
    test_files = []
    for file in os.listdir(dataset_dir):
        if file.endswith('.csv'):
            test_files.append(os.path.join(dataset_dir, file))
    
    print(f"Found {len(test_files)} test files")
    print()
    
    # Run experiments
    result_files = []
    for test_file in test_files:
        filename = Path(test_file).stem
        if filename in selected_roles:
            result_file = run_single_test(test_file, selected_roles, model)
            result_files.append(result_file)
    
    print(f"\nCompleted {len([f for f in result_files if f])} tests")

if __name__ == "__main__":
    main()
EOF

echo "2. Running experiments..."
python run_evaluation.py selected_roles.json "$DATASET_DIR" "$MODEL"

echo ""

# Create results analysis script
cat > analyze_results.py << 'EOF'
import json
import os
import ast
from pathlib import Path

def calculate_metrics():
    """Calculate evaluation metrics"""
    total_questions = 0
    total_correct = 0
    total_responses = 0
    
    results = {}
    
    # Find all result files
    result_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('_simple.txt'):
                result_files.append(os.path.join(root, file))
    
    for result_file in result_files:
        if not os.path.exists(result_file):
            continue
            
        filename = Path(result_file).stem.replace('_simple', '')
        
        try:
            with open(result_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) >= 2:
                # Parse results
                accs = ast.literal_eval(lines[0].strip())
                resp_cnts = ast.literal_eval(lines[1].strip())
                
                test_questions = len(accs)
                test_correct = sum(accs)
                test_responses = sum(resp_cnts)
                
                total_questions += test_questions
                total_correct += test_correct
                total_responses += test_responses
                
                results[filename] = {
                    'accuracy': test_correct / test_questions if test_questions > 0 else 0,
                    'questions': test_questions,
                    'correct': test_correct,
                    'responses': test_responses
                }
                
        except Exception as e:
            print(f"Error parsing {result_file}: {e}")
            continue
    
    # Calculate overall metrics
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    avg_responses = total_responses / total_questions if total_questions > 0 else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_questions': total_questions,
        'total_correct': total_correct,
        'avg_responses': avg_responses,
        'by_test': results
    }

def main():
    print("3. Analyzing results...")
    metrics = calculate_metrics()
    
    # Display results
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall accuracy: {metrics['overall_accuracy']:.4f} ({metrics['total_correct']}/{metrics['total_questions']})")
    print(f"Average responses: {metrics['avg_responses']:.2f}")
    print()
    
    print("Per-test results:")
    print(f"{'Test Name':<25} {'Accuracy':<10} {'Questions':<8} {'Correct':<8}")
    print("-" * 55)
    
    for test_name, test_metrics in metrics['by_test'].items():
        print(f"{test_name:<25} {test_metrics['accuracy']:<10.4f} {test_metrics['questions']:<8} {test_metrics['correct']:<8}")
    
    print()
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
EOF

echo "3. Analyzing results..."
python analyze_results.py

# Clean up temporary files
rm -f select_roles.py run_evaluation.py analyze_results.py selected_roles.json

echo ""
echo "=== Evaluation Completed ==="
