#!/bin/bash

MODEL=35_0301
ENGINE=chatgpt0301
# MODEL=4_0613
# ENGINE=gpt4

# specify your directory
dir="<path-to-your-directory>"

# loop over sub-directories
for subdir in "$dir"/*
do
    if [ -d "$subdir" ]; then  # Only proceed if $subdir is a directory
        echo "Processing $subdir"
        
        # Get list of numerically sorted json files in the subdir
        json_files=($(find $subdir -name "*.json" -type f -exec basename {} .json \; | sort -n))

        # Calculate the total number of json files
        total_files=${#json_files[@]}

        # Calculate how many times the loop will run
        loops=$((($total_files + 99) / 100))

        # Loop over each set of 100 (or less) files
        for ((i=0; i<$loops; i++))
        do
            # Calculate start and end indices
            start=$((i*100))
            end=$(((i+1)*100-1))

            # If end index is greater than the total number of files, adjust it
            if ((end >= total_files)); then
                end=$((total_files-1))
            fi

            # Get the min and max filenames in the current range
            min_file=${json_files[$start]}
            max_file=${json_files[$end]}

            base_name=$(basename "$subdir")
            result_file_name=llmlp_math_cot_${MODEL}/${base_name}_${min_file}_${max_file}_4_3.json
            # continue if result file already exists
            if [ -f "$result_file_name" ]; then
                continue
            fi

            echo "$subdir $min_file $max_file"
            # Run python script in background
            python llmlp_gen_math_listwise_cot.py "$subdir" "$min_file" "$max_file" "$MODEL" "$ENGINE" &
            # echo "$subdir $min_file $max_file" &
        done

        # Wait for all background jobs to finish
        echo "Finished processing $subdir"
    fi
done

wait
echo "All done"

python eval_math.py llmlp_math_cot_$MODEL None
