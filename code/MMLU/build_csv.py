import csv
import sys
import ast
import os

TOTAL_AGENTS = int(sys.argv[1])
RES_DIR = sys.argv[2]
TARGET_CSV = sys.argv[3]

def process_lists(*lists):
    # Initialize a dictionary to store the data
    data_dict = {}
    # Loop through all files in the directory
    for filename in os.listdir(RES_DIR):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            with open(os.path.join(RES_DIR, filename), 'r') as f:
                # Read the lines
                lines = f.readlines()

                # Check if the file has at least four lines
                if len(lines) >= 4:
                    acc_score = float(lines[0].strip().rsplit(' ', 1)[-1])
                    resp, q_cnt = int(lines[1].strip().split(' ', 1)[0]), len(lines[0].strip().rsplit(' ', 1)[0].split(','))
                    # Parse the fourth line to a list
                    fourth_line = ast.literal_eval(lines[3].strip())
                    
                    # Check if the fourth line is a list
                    if isinstance(fourth_line, list):
                        # reshape to (TOTAL_AGENTS, -1)
                        fourth_line = [fourth_line[i:i+TOTAL_AGENTS] for i in range(0, len(fourth_line), TOTAL_AGENTS)] #[1:]
                        sums = [sum(pos) for pos in zip(*fourth_line)]
                        scores = [(sum([sums[idx] for idx in list_[0]])/len(list_[0]), list_[1]) for list_ in lists]

                        # Add scores to data_dict with filename as key
                        if filename[:-7].split('_')[-1] == 'test' or filename[:-7].split('_')[-2] == 'merged':
                            data_dict[filename[:-7]] = {score[1]+"_imp": score[0] for idx, score in enumerate(scores)}
                            data_dict[filename[:-7]]['acc'] = acc_score
                            data_dict[filename[:-7]]['resp'] = resp
                            data_dict[filename[:-7]]['q_cnt'] = q_cnt

    # Write to csv
    with open(TARGET_CSV, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['filename'] + [score[1]+"_imp" for score in scores] + ['acc', 'resp', 'q_cnt'])
        writer.writeheader()
        for filename, scores in data_dict.items():
            row_dict = {'filename': filename}
            row_dict.update(scores)
            writer.writerow(row_dict)

if __name__ == "__main__":
    lists = []
    idx_list = sys.argv[4:]
    idx_list, name_list = idx_list[:len(idx_list)//2], idx_list[len(idx_list)//2:]
    for idx, arg in enumerate(idx_list):
        try:
            # Convert string to list using ast.literal_eval
            list_ = ast.literal_eval(arg)
            if isinstance(list_, list):
                lists.append((list_, name_list[idx]))
            else:
                raise ValueError
        except (ValueError, SyntaxError):
            print(f"Invalid list argument: {arg}")
            sys.exit(1)
    process_lists(*lists)
