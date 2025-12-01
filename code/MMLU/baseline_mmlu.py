import json
import os
import random
import sys
from utils import get_mmlu_qa_pairs, parse_single_choice, generate_answer
from dotenv import load_dotenv

load_dotenv()  # Load .env file

QUERY_CSV = sys.argv[1]
EXP_NAME = sys.argv[2]
MODEL = sys.argv[3]
DIR_NAME = sys.argv[4]

TYPE = "single_choice"


def set_rd_seed(seed):
    random.seed(seed)


def main():
    set_rd_seed(0)
    os.makedirs(DIR_NAME, exist_ok=True)

    # Print configuration status
    if os.getenv("RATIONALE", "0") == "1":
        print("=" * 60)
        print("WARNING: RATIONALE flag is set but ignored for baseline")
        print("Baseline uses single LLM call with no scoring/rationales")
        print("=" * 60)

    qa_pairs = get_mmlu_qa_pairs(QUERY_CSV)

    # Initialize output file
    json_file = DIR_NAME + '/' + EXP_NAME + '_baseline.json'
    with open(json_file, 'w') as f:
        f.write("")

    accs = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    completions_list = []

    for que, ans in qa_pairs:
        # Simple single-call prompt
        context = [
            {
                "role": "system",
                "content": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans."
            },
            {
                "role": "user",
                "content": que + "\n\nThink step by step and explain your reasoning. Put your answer in the form (X) at the end of your response. (X) represents choice (A), (B), (C), or (D)."
            }
        ]

        # Single LLM call
        reply, p_tokens, c_tokens = generate_answer(context, MODEL)
        total_prompt_tokens += p_tokens
        total_completion_tokens += c_tokens

        # Log the response
        print("LLM Response:")
        print(reply)
        print()

        # Parse answer
        predicted = parse_single_choice(reply)
        is_correct = (ans == predicted)
        accs.append(is_correct)

        # Log parsed answer and correctness
        print(f"Parsed answer: {predicted}")
        print(f"Correct answer: {ans}")
        print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        print("-" * 80)
        print()

        # Store completion for JSON output
        completion = {
            "question": que,
            "correct_answer": ans,
            "predicted_answer": predicted,
            "response": reply
        }
        completions_list.append(completion)

        # Append to JSON file (one line per question)
        with open(json_file, 'a') as f:
            f.write(json.dumps(completion) + '\n')

    # Calculate metrics
    accuracy = sum(accs) / len(qa_pairs) if len(qa_pairs) > 0 else 0
    total_questions = len(qa_pairs)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Baseline Evaluation Complete")
    print(f"Accuracy: {accuracy:.4f} ({sum(accs)}/{total_questions})")
    print(f"Total Prompt Tokens: {total_prompt_tokens}")
    print(f"Total Completion Tokens: {total_completion_tokens}")
    print("=" * 60)

    # Write .txt output (6 lines for compatibility with existing analysis tools)
    txt_file = DIR_NAME + '/' + EXP_NAME + '_baseline.txt'
    with open(txt_file, 'w') as f:
        # Line 1: List of correctness + overall accuracy
        f.write(str(accs) + ' ' + str(accuracy) + '\n')
        # Line 2: Total responses + average responses per question (always 1 for baseline)
        f.write(str(total_questions) + " 1.0\n")
        # Line 3: Empty importance scores (baseline doesn't have multi-agent scores)
        f.write("[]\n")
        # Line 4: Empty average importance scores
        f.write("[]\n")
        # Line 5: Total prompt tokens
        f.write(str(total_prompt_tokens) + '\n')
        # Line 6: Total completion tokens
        f.write(str(total_completion_tokens) + '\n')

    print(f"\nResults written to:")
    print(f"  - {txt_file}")
    print(f"  - {json_file}")


if __name__ == "__main__":
    main()
