#!/usr/bin/env python3
"""
Generate embeddings for agent responses at agent×subject level.

This script:
1. Parses all .log files from MMLU experiments
2. Extracts agent responses (marked by round/agent index)
3. Generates embeddings using instructor-xl model (locally)
4. Averages embeddings at the (agent_role, subject) level
5. Saves results to pickle and CSV files
"""

import re
import argparse
import pickle
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer

# Agent role mapping (index -> name)
AGENT_ROLES = {
    0: 'Economist',
    1: 'Doctor',
    2: 'Lawyer',
    3: 'Mathematician',
    4: 'Psychologist',
    5: 'Programmer',
    6: 'Historian'
}


def parse_log_file(log_path: Path) -> List[Tuple[str, str, int, int, str]]:
    """
    Parse a single log file to extract agent responses.

    Returns:
        List of tuples: (subject, agent_role, round_num, question_num, response_text)
    """
    subject = log_path.stem.replace('_test_73', '')

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split by agent markers (lines starting with "{round} {agent_index}")
    # Pattern: line starts with digit(s), space, digit(s)
    agent_pattern = re.compile(r'^(\d+)\s+(\d+)$', re.MULTILINE)

    responses = []
    matches = list(agent_pattern.finditer(content))

    for i, match in enumerate(matches):
        round_num = int(match.group(1))
        agent_idx = int(match.group(2))

        # Validate agent index
        if agent_idx not in AGENT_ROLES:
            continue

        agent_role = AGENT_ROLES[agent_idx]

        # Extract response text (from after this marker to before next marker)
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        response_text = content[start_pos:end_pos].strip()

        # Skip empty responses
        if not response_text or len(response_text) < 10:
            continue

        # Clean up the response (remove "question context:" sections if present)
        if 'question context:' in response_text:
            # Take only the actual response after the question context
            parts = response_text.split('question context:')
            if len(parts) > 1:
                # Find where the actual response starts (after the question)
                response_parts = parts[-1].split('\n')
                # Skip the question lines and get to the answer
                actual_response = []
                found_response = False
                for line in response_parts:
                    if line.strip() and not line.startswith('[{') and not found_response:
                        found_response = True
                    if found_response:
                        actual_response.append(line)
                response_text = '\n'.join(actual_response).strip()

        # Estimate question number based on round progression
        question_num = i // len(AGENT_ROLES)

        responses.append((subject, agent_role, round_num, question_num, response_text))

    return responses


def generate_embeddings_batch(texts: List[str], model: INSTRUCTOR, instruction: str, batch_size: int = 32) -> List[np.ndarray]:
    """
    Generate embeddings for a batch of texts using instructor-xl model.

    Args:
        texts: List of text strings to embed
        model: INSTRUCTOR model instance
        instruction: Instruction for the embedding task
        batch_size: Number of texts to process at once (for memory management)

    Returns:
        List of embedding vectors (numpy arrays)
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        try:
            # Format input as [[instruction, text], [instruction, text], ...]
            # This is the format required by InstructorEmbedding
            instructor_inputs = [[instruction, text] for text in batch]

            # Generate embeddings
            batch_embeddings = model.encode(instructor_inputs)

            # Convert to list of numpy arrays
            if isinstance(batch_embeddings, np.ndarray):
                all_embeddings.extend([batch_embeddings[i] for i in range(len(batch_embeddings))])
            else:
                all_embeddings.extend(batch_embeddings)

        except Exception as e:
            print(f"\n⚠️  Error embedding batch {i // batch_size + 1}: {e}")
            # Add zero vectors for failed embeddings
            embedding_dim = 768  # instructor-xl output dimension
            all_embeddings.extend([np.zeros(embedding_dim) for _ in batch])

    return all_embeddings


def main():
    parser = argparse.ArgumentParser(description='Generate agent embeddings from MMLU log files')
    parser.add_argument(
        '--log-dir',
        type=str,
        default='mmlu_downsampled_Economist_Doctor_Lawyer_Mathematician_Psychologist_Programmer_Historian',
        help='Directory containing .log files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='embeddings_agent_subject',
        help='Output filename prefix (without extension)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding (lower values use less memory)'
    )
    parser.add_argument(
        '--instruction',
        type=str,
        default='Represent the reasoning approach for question answering',
        help='Instruction for the embedding task'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='hkunlp/instructor-xl',
        help='HuggingFace model path for instructor model'
    )

    args = parser.parse_args()

    # Load the instructor model
    print(f"📥 Loading model: {args.model}")
    print("   (This may take a few minutes on first run...)")
    model = INSTRUCTOR(args.model)
    print("✅ Model loaded!")

    # Find all log files
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    log_files = list(log_dir.glob('*_test_73.log'))
    print(f"📂 Found {len(log_files)} log files in {log_dir}")

    # Step 1: Parse all log files
    print("\n📖 Parsing log files...")
    all_responses = []
    for log_file in tqdm(log_files, desc="Parsing logs"):
        responses = parse_log_file(log_file)
        all_responses.extend(responses)

    print(f"✅ Extracted {len(all_responses)} agent responses")

    # Step 2: Group by (agent_role, subject)
    print("\n🗂️  Grouping responses by (agent, subject)...")
    grouped = defaultdict(list)
    for subject, agent_role, round_num, question_num, response_text in all_responses:
        key = (agent_role, subject)
        grouped[key].append(response_text)

    print(f"✅ Created {len(grouped)} (agent × subject) groups")

    # Step 3: Generate embeddings for each group
    print(f"\n🔮 Generating embeddings with instruction: '{args.instruction}'")
    embeddings_dict = {}

    for (agent_role, subject), responses in tqdm(grouped.items(), desc="Embedding groups"):
        if not responses:
            continue

        # Generate embeddings for all responses in this group
        embeddings = generate_embeddings_batch(
            responses,
            model,
            args.instruction,
            args.batch_size
        )

        # Average embeddings across all questions/rounds
        mean_embedding = np.mean(embeddings, axis=0)
        embeddings_dict[(agent_role, subject)] = mean_embedding

    print(f"✅ Generated {len(embeddings_dict)} mean embeddings")

    # Step 4: Save results
    print(f"\n💾 Saving results...")

    # Save as pickle
    pickle_path = f"{args.output}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    print(f"✅ Saved pickle: {pickle_path}")

    # Save as CSV (with JSON-encoded embeddings for readability)
    csv_path = f"{args.output}.csv"
    with open(csv_path, 'w') as f:
        f.write("agent_role,subject,embedding_json\n")
        for (agent_role, subject), embedding in sorted(embeddings_dict.items()):
            embedding_json = json.dumps(embedding.tolist())
            f.write(f"{agent_role},{subject},\"{embedding_json}\"\n")
    print(f"✅ Saved CSV: {csv_path}")

    # Print summary statistics
    print("\n📊 Summary:")
    print(f"   Agents: {len(set(k[0] for k in embeddings_dict.keys()))}")
    print(f"   Subjects: {len(set(k[1] for k in embeddings_dict.keys()))}")
    print(f"   Total embeddings: {len(embeddings_dict)}")
    print(f"   Embedding dimension: {list(embeddings_dict.values())[0].shape[0]}")

    # Show sample keys
    print("\n🔍 Sample keys:")
    for key in list(embeddings_dict.keys())[:5]:
        print(f"   {key}")

    print("\n✨ Done!")


if __name__ == '__main__':
    main()
