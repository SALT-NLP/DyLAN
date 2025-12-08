import ast
import json
import os
import random
import sys
from LLMLP import LLMLP
from utils import *

from dotenv import load_dotenv

try:
    import sys
    import os
    code_path = os.path.dirname(__file__) 
    parent_code_path = os.path.dirname(code_path) 
    if parent_code_path not in sys.path:
        sys.path.insert(0, parent_code_path)
    
    from memory_bank import MemoryBank, MemoryManager
    MEMORY_BANK_AVAILABLE = True
    print("Memory Bank package loaded successfully!")
except ImportError as e:
    MEMORY_BANK_AVAILABLE = False
    print(f"Warning: memory_bank package not available. Memory features disabled. Error: {e}")

load_dotenv()  # so .env is loaded when this script is run

QUERY_CSV = sys.argv[1]
EXP_NAME = sys.argv[2]
MODEL = sys.argv[3]

ACTIVATION = "listwise"
TYPE = "single_choice"
# ROLES = ["Assistant", "Mathematician", "Mathematician", "Assistant"]
DIR_NAME = sys.argv[4]
ROLES = ast.literal_eval(sys.argv[5])
DIR_NAME = DIR_NAME + '_' + '_'.join(ROLES)

# 【Memory Bank配置】通过环境变量控制是否启用
USE_MEMORY_BANK = os.getenv("USE_MEMORY_BANK", "0") == "1"
MEMORY_IMPORTANCE_THRESHOLD = float(os.getenv("MEMORY_IMPORTANCE_THRESHOLD", "0.3"))


def set_rd_seed(seed):
    random.seed(seed)

def extract_experience_from_agent(question, correct_answer, agent_reasoning, agent_role, model="gpt-3.5-turbo"):
    """
    Ask the agent to extract experience and knowledge from the question and their reasoning process.
    """
    prompt = f"""You are a {agent_role}. You just solved this question:

Question: {question}
Correct Answer: {correct_answer}
Your Reasoning: {agent_reasoning}

Based on this problem-solving experience, what key insights, methods, or knowledge did you learn that could help you solve similar problems in the future? 

Please provide a concise summary (1-2 sentences) of the most important takeaway or experience from this problem. Focus on the general principle or method rather than specific details.

Example format: "When dealing with [type of problem], [key method/insight] is crucial for [achieving what]."
"""

    try:
        from utils import generate_answer
        context = [{"role": "user", "content": prompt}]
        experience, _, _ = generate_answer(context, model)
        return experience.strip()
    except Exception as e:
        print(f"Warning: Failed to extract experience via LLM: {e}")
        # Fallback to a simple summary
        return f"Learned problem-solving approach for {agent_role} domain questions."

def main():
    set_rd_seed(0)
    assert len(ROLES) > 0
    os.makedirs(DIR_NAME, exist_ok=True)

    #Print configuration status
    if os.getenv("RATIONALE", "0") == "1":
        print("=" * 10)
        print("RATIONALE MODE ENABLED: Agents will provide rationales for their scores.")
        print("=" * 10) 

    llmlp = LLMLP(MODEL, len(ROLES), ROLES, 3, ACTIVATION, TYPE, MODEL)
    qa_pairs = get_mmlu_qa_pairs(QUERY_CSV)

    # Memory Bank initialization - modular design
    memory_bank = None
    memory_manager = None
    if USE_MEMORY_BANK and MEMORY_BANK_AVAILABLE:
        memory_bank = MemoryBank()
        memory_manager = MemoryManager(memory_bank, model=MODEL)
        # Try to load existing memories if available
        memory_file = os.path.join(DIR_NAME, "memory_bank.json")
        if os.path.exists(memory_file):
            try:
                memory_bank.load(memory_file)
                print(f"Loaded {len(memory_bank.all_entries())} memories from {memory_file}")
            except Exception as e:
                print(f"Warning: Failed to load memory bank: {e}")
        print("=" * 10)
        print("MEMORY BANK ENABLED")
        print(f"Importance threshold: {MEMORY_IMPORTANCE_THRESHOLD}")
        print("=" * 10)
    else:
        print("Memory bank disabled (set USE_MEMORY_BANK=1 to enable)")

    with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+'3.json', 'w') as f:
        f.write("")

    accs, resp_cnts, importances = [], 0, []
    completion_list = []
    total_prompt_tokens, total_completion_tokens = 0, 0

    try:
        for que, ans in qa_pairs:
            llmlp.zero_grad()
            # Forward phase: pass memory_bank to agents if enabled
            res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(que, memory_bank=memory_bank)
            imp_score = llmlp.backward(res, que)
            
            # Backward phase: extract and store memories if enabled
            if memory_manager is not None:
                max_imp = max(imp_score) if imp_score else 0.0
                if max_imp >= MEMORY_IMPORTANCE_THRESHOLD:
                    # Find the agent with highest importance score
                    top_idx = imp_score.index(max_imp)
                    top_role = ROLES[top_idx % len(ROLES)]
                    
                    # Extract reasoning process from completions
                    try:
                        # completions structure: [[agent0_round0, agent0_round1, ...], [agent1_round0, ...], ...]
                        agent_idx = top_idx % len(ROLES)
                        round_idx = top_idx // len(ROLES)
                        if round_idx < len(completions[agent_idx]) and completions[agent_idx][round_idx]:
                            top_reasoning = completions[agent_idx][round_idx][:500]  # Limit length for LLM processing
                        else:
                            top_reasoning = "Reasoning not available"
                    except:
                        top_reasoning = "Reasoning not available"
                    
                    # Let the agent extract their own experience and insights
                    experience = extract_experience_from_agent(
                        question=que,
                        correct_answer=ans, 
                        agent_reasoning=top_reasoning,
                        agent_role=top_role,
                        model=MODEL
                    )
                    
                    # Use MemoryManager to decide ADD/UPDATE/DELETE based on agent's experience
                    try:
                        event = memory_manager.process_fact(experience, owner=top_role)
                        if event.operation.value != "NOOP":
                            print(f"[Memory] {event.operation.value} ({top_role}): {event.reason[:100]}")
                    except Exception as e:
                        print(f"Warning: Memory processing failed: {e}")
                        import traceback
                        traceback.print_exc()

            completion_list.append(completions)
            accs.append(ans == res)
            resp_cnts += resp_cnt
            importances.append(imp_score)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+'3.json', 'a') as f:
                f.write(json.dumps(completions) + '\n')
            
            # Periodic Memory Bank saving - every 100 questions
            if memory_bank is not None and len(accs) % 100 == 0 and len(accs) > 0:
                memory_file = os.path.join(DIR_NAME, "memory_bank.json")
                try:
                    memory_bank.save(memory_file)
                    print(f"[Memory] Saved {len(memory_bank.all_entries())} memories to {memory_file}")
                except Exception as e:
                    print(f"Warning: Failed to save memory bank: {e}")

    except Exception as e:
        print(f"Critical error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure memories and results are saved even if exceptions occur
        print(f"Processing completed. Processed {len(accs)} questions.")

    print(accs)
    print(resp_cnts)
    print(importances)

    with open(DIR_NAME+'/'+EXP_NAME+'_'+str(len(ROLES))+'3.txt', 'w') as f:
        f.write(str(accs) + ' ' + str(sum(accs)/len(qa_pairs)) + '\n')
        f.write(str(resp_cnts) + " " + str(resp_cnts/len(qa_pairs)) + '\n')
        f.write(json.dumps(importances) + '\n')
        f.write(json.dumps([sum(pos)/len(qa_pairs) for pos in zip(*importances)]) + '\n')
        f.write(str(total_prompt_tokens) + '\n')
        f.write(str(total_completion_tokens) + '\n')
    
    # Final Memory Bank save
    if memory_bank is not None:
        memory_file = os.path.join(DIR_NAME, "memory_bank.json")
        try:
            print(f"[Memory] Attempting final save to {memory_file}")
            memory_bank.save(memory_file)
            print(f"[Memory] Final save successful: {len(memory_bank.all_entries())} memories saved to {memory_file}")
        except Exception as e:
            print(f"Error: Failed to save memory bank: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[Memory] No memory bank to save")

if __name__ == "__main__":
    main()
