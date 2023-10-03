#!/bin/bash

TOTAL_AGENTS=7

DIR_NAME=mmlu_downsampled_Economist_Doctor_Lawyer_Mathematician_Psychologist_Programmer_Historian
TARGET_CSV=importance_1to7.csv

# Call Python script with three lists as arguments
python proc_lists.py $TOTAL_AGENTS $DIR_NAME $TARGET_CSV "[0]" "[1]" "[2]" "[3]" "[4]" "[5]" "[6]"
python build_csv.py $TOTAL_AGENTS $DIR_NAME $TARGET_CSV "[0]" "[1]" "[2]" "[3]" "[4]" "[5]" "[6]" Economist Doctor Lawyer Mathematician Psychologist Programmer Historian
python calc_ave_acc.py $TARGET_CSV
