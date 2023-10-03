import sys
import ast
import os

from numpy import argmax


TOTAL_AGENTS = int(sys.argv[1])
RES_DIR = sys.argv[2]
TARGET_CSV = sys.argv[3]

def process_lists(*lists):
    # for idx, list_ in enumerate(lists, 1):
    #     print(f"List {idx}: {list_}")

    imp_tests = [[] for _ in lists]

    # Loop through all files in the directory
    for filename in os.listdir(RES_DIR):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            with open(os.path.join(RES_DIR, filename), 'r') as f:
                # Read the lines
                lines = f.readlines()

                # Check if the file has at least four lines
                if len(lines) >= 4:
                    # Parse the fourth line to a list
                    fourth_line = ast.literal_eval(lines[3].strip())
                    
                    # Check if the fourth line is a list
                    if isinstance(fourth_line, list):
                        # reshape to (TOTAL_AGENTS, -1)
                        fourth_line = [fourth_line[i:i+TOTAL_AGENTS] for i in range(0, len(fourth_line), TOTAL_AGENTS)] #[1:]
                        sums = [sum(pos) for pos in zip(*fourth_line)]
                        scores = [sum([sums[idx] for idx in list_])/len(list_) for list_ in lists]
                        # imp_tests[argmax(scores)].append((filename[:-4], max(scores)))
                        # use average score
                        # for tid in range(len(imp_tests)):
                            # imp_tests[tid].append((filename[:-4],  (scores[tid]-sum(scores)/len(scores)) / (sum(scores)/len(scores))))
                        norms = [(scores[tid]-sum(scores)/len(scores)) / (sum(scores)/len(scores)) for tid in range(len(imp_tests))]
                        norms = list(enumerate(norms))
                        norms.sort(key=lambda x: x[1], reverse=True)
                        for tid, norm in norms[:2]:
                            imp_tests[tid].append((filename[:-4], norm))
                        

    for idx, list_ in enumerate(lists, 1):
        print(f"List {idx}: {list_}")
        imp_tests[idx-1].sort(key=lambda x: x[1], reverse=True)
        imp_tests[idx-1] = list(filter(lambda x: x[0].split('_')[-2] == 'test' or x[0].split('_')[-2] == 'merged', imp_tests[idx-1]))

        print("best tests:", imp_tests[idx-1])
        print()
        print("best 10 tests:")
        for test in imp_tests[idx-1][:10]:
            print(test[0], test[1])
        print()

if __name__ == "__main__":
    lists = []
    for arg in sys.argv[4:]:
        try:
            # Convert string to list using ast.literal_eval
            list_ = ast.literal_eval(arg)
            if isinstance(list_, list):
                lists.append(list_)
            else:
                raise ValueError
        except (ValueError, SyntaxError):
            print(f"Invalid list argument: {arg}")
            sys.exit(1)
    process_lists(*lists)
