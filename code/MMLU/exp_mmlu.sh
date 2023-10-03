#!/bin/bash

MODEL=gpt-3.5-turbo
# MODEL=gpt-4

# specify your directory
dir="<path-to-your-directory>"
exp_name=mmlu_downsampled

ROLES="['Economist', 'Doctor', 'Lawyer', 'Mathematician', 'Psychologist', 'Programmer', 'Historian']"
for file in "$dir"/*.csv
do
    # extract filename without extension
    filename=$(basename -- "$file")
    extension="${filename##*.}"
    filename="${filename%.*}"

    RES_NAME=mmlu_downsampled_Economist_Doctor_Lawyer_Mathematician_Psychologist_Programmer_Historian/${filename}_73.txt
    LOG_NAME=mmlu_downsampled_Economist_Doctor_Lawyer_Mathematician_Psychologist_Programmer_Historian/${filename}_73.log

    # check if RES_NAME has 4 lines
    if [ -f "$RES_NAME" ]; then
        linecount=$(wc -l < "$RES_NAME")
        if [ "$linecount" -eq 4 ]; then
            continue
        fi
    fi

    # run python script in background
    python llmlp_listwise_mmlu.py "$file" "$filename" "$MODEL" "$exp_name" "$ROLES" 2>&1 > "$LOG_NAME" &
    # echo "$file" "$filename" "$ROLES" &
done
# done

wait
echo "All done"

bash anal_imp.sh

