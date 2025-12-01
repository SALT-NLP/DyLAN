#!/bin/bash

#MODEL=gpt-3.5-turbo
# MODEL=gpt-4
MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# original ROLES array
# ROLES=("Assistant" "Mathematician" "Mathematician" "Assistant")
# ROLES=("Economist" "Psychologist" "Historian")
# ROLES=("Programmer" "Doctor" "Historian")
ROLES=("Economist" "Doctor" "Lawyer")

# Resolve paths relative to this script so you can run from anywhere (incl. repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Your mini slice of MMLU:
dir="$SCRIPT_DIR/../../data/MMLU/val"
exp_name=mmlu_downsampled

# TODO: mkdir -p "$exp_name"

# number of roles
N=${#ROLES[@]}

# loop over .csv files
for file in "$dir"/*.csv
do
    # extract filename without extension
    filename=$(basename -- "$file")
    extension="${filename##*.}"
    filename="${filename%.*}"

    # generate all subsets of ROLES
    for ((i=1; i<(1<<N); ++i)) # i starts from 1 to exclude empty subset
    do
        subset=()
        for ((j=0; j<N; ++j))
        do
            ((i & (1 << j))) && subset+=(${ROLES[j]})
        done

        # convert subset array to string format required by python script
        subset_string="['${subset[0]}'"
        for ((k=1; k<${#subset[@]}; ++k))
        do
            subset_string+=", '${subset[k]}'"
        done
        subset_string+="]"

        # run python script in background with current subset
        python llmlp_listwise_mmlu.py "$file" "$filename" "$MODEL" "$exp_name" "$subset_string" &
        # echo "$file" "$filename" "$subset_string" &
    done
done
    
wait
echo "All done"
