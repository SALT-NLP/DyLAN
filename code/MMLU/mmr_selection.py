#!/usr/bin/env python3
"""
Maximal Marginal Relevance (MMR) Agent Selection

This module implements diversity-aware agent selection that balances:
1. Importance scores (from backward pass / agent importance)
2. Embedding-based diversity (from reasoning approach embeddings)

The MMR algorithm iteratively selects agents that maximize:
    MMR(agent) = λ × importance(agent) + (1-λ) × diversity(agent)

Where:
- λ=1.0: Pure importance-based selection (greedy, current behavior)
- λ=0.7-0.8: Slight diversity preference (recommended)
- λ=0.5: Equal weight to importance and diversity
- λ=0.0: Pure diversity-based selection
"""

import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings(embeddings_path: str) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Load pre-computed embeddings from pickle file.

    Args:
        embeddings_path: Path to embeddings_agent_subject.pkl

    Returns:
        Dictionary mapping (agent_role, subject) -> embedding vector
    """
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Embeddings file not found: {embeddings_path}\n"
            "Please run generate_embeddings.py first to create embeddings."
        )
    except Exception as e:
        raise RuntimeError(f"Error loading embeddings: {e}")


def extract_subject_from_filename(filename: str) -> str:
    """
    Extract subject name from test filename.

    Args:
        filename: Test filename (e.g., "college_biology_test" or "college_biology_test.csv")

    Returns:
        Subject name (e.g., "college_biology")
    """
    # Remove .csv extension if present
    if filename.endswith('.csv'):
        filename = filename[:-4]

    # Remove _test suffix
    if filename.endswith('_test'):
        filename = filename[:-5]

    return filename


def compute_diversity_score(
    candidate_agent: str,
    candidate_embedding: np.ndarray,
    selected_agents: List[str],
    embeddings_dict: Dict[Tuple[str, str], np.ndarray],
    subject: str
) -> float:
    """
    Compute diversity score for a candidate agent.

    Diversity is measured as: 1 - max_similarity_to_selected
    Higher diversity score = more different from already selected agents

    Args:
        candidate_agent: Role name of candidate agent
        candidate_embedding: Embedding vector for candidate
        selected_agents: List of already-selected agent role names
        embeddings_dict: Full embeddings dictionary
        subject: Subject name for embedding lookup

    Returns:
        Diversity score (0.0 to 1.0, higher = more diverse)
    """
    if not selected_agents:
        # First agent: maximum diversity by definition
        return 1.0

    # Compute similarity to all selected agents
    similarities = []
    for selected_agent in selected_agents:
        selected_embedding = embeddings_dict[(selected_agent, subject)]

        # Cosine similarity (returns value in [-1, 1], but typically [0, 1] for embeddings)
        similarity = cosine_similarity(
            candidate_embedding.reshape(1, -1),
            selected_embedding.reshape(1, -1)
        )[0, 0]

        similarities.append(similarity)

    # Diversity = 1 - max_similarity (maximal marginal relevance principle)
    max_similarity = max(similarities)
    diversity = 1.0 - max_similarity

    return diversity


def select_agents_mmr(
    importance_scores: Dict[str, float],
    embeddings_dict: Dict[Tuple[str, str], np.ndarray],
    subject: str,
    num_roles: int,
    lambda_param: float = 0.5
) -> List[str]:
    """
    Select agents using Maximal Marginal Relevance (MMR).

    Args:
        importance_scores: Dictionary {agent_role: importance_score}
        embeddings_dict: Dictionary {(agent_role, subject): embedding_vector}
        subject: Subject name for embedding lookup
        num_roles: Number of agents to select
        lambda_param: Trade-off parameter (0.0=pure diversity, 1.0=pure importance)

    Returns:
        List of selected agent role names
    """
    # Handle edge cases
    if num_roles <= 0:
        return []

    all_agents = list(importance_scores.keys())
    if num_roles >= len(all_agents):
        # Select all agents if k >= n
        return all_agents

    # Check which agents have embeddings for this subject
    agents_with_embeddings = [
        agent for agent in all_agents
        if (agent, subject) in embeddings_dict
    ]
    agents_without_embeddings = [
        agent for agent in all_agents
        if (agent, subject) not in embeddings_dict
    ]

    # If no embeddings at all, fall back to pure greedy
    if not agents_with_embeddings:
        print(f"⚠️  Warning: No embeddings found for subject '{subject}', using greedy selection")
        sorted_agents = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        return [agent for agent, _ in sorted_agents[:num_roles]]

    # If some (but not all) embeddings missing, use hybrid approach
    if agents_without_embeddings:
        print(f"ℹ️  Note: {len(agents_without_embeddings)}/{len(all_agents)} agents missing embeddings for '{subject}'")
        print(f"   Missing: {', '.join(agents_without_embeddings)}")
        print(f"   Using MMR for {len(agents_with_embeddings)} agents with embeddings, greedy for others")

        # Split importance scores
        importance_with_embeddings = {
            agent: importance_scores[agent]
            for agent in agents_with_embeddings
        }
        importance_without_embeddings = {
            agent: importance_scores[agent]
            for agent in agents_without_embeddings
        }

        # Determine how many to select from each group
        # Strategy: Prefer agents with embeddings, but include top agents without if needed
        num_with_embeddings = min(num_roles, len(agents_with_embeddings))
        num_without_embeddings = num_roles - num_with_embeddings

        # Select from agents WITH embeddings using MMR
        selected_with = select_agents_mmr(
            importance_with_embeddings,
            embeddings_dict,
            subject,
            num_with_embeddings,
            lambda_param
        ) if num_with_embeddings > 0 else []

        # Select from agents WITHOUT embeddings using greedy
        sorted_without = sorted(
            importance_without_embeddings.items(),
            key=lambda x: x[1],
            reverse=True
        )
        selected_without = [agent for agent, _ in sorted_without[:num_without_embeddings]]

        return selected_with + selected_without

    # Normalize importance scores to [0, 1] range for fair comparison with diversity
    importance_values = list(importance_scores.values())
    min_imp = min(importance_values)
    max_imp = max(importance_values)

    if max_imp > min_imp:
        normalized_importance = {
            agent: (score - min_imp) / (max_imp - min_imp)
            for agent, score in importance_scores.items()
        }
    else:
        # All scores equal: use uniform importance
        normalized_importance = {agent: 1.0 for agent in importance_scores.keys()}

    # MMR selection algorithm
    selected_agents = []
    remaining_agents = set(all_agents)

    for iteration in range(num_roles):
        best_agent = None
        best_mmr_score = -float('inf')

        # Evaluate all remaining agents
        for candidate_agent in remaining_agents:
            # Importance term
            importance = normalized_importance[candidate_agent]

            # Diversity term
            candidate_embedding = embeddings_dict[(candidate_agent, subject)]
            diversity = compute_diversity_score(
                candidate_agent,
                candidate_embedding,
                selected_agents,
                embeddings_dict,
                subject
            )

            # MMR score: λ × importance + (1-λ) × diversity
            mmr_score = lambda_param * importance + (1 - lambda_param) * diversity

            # Track best candidate
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_agent = candidate_agent

        # Select best agent
        selected_agents.append(best_agent)
        remaining_agents.remove(best_agent)

    return selected_agents


def select_agents_for_all_tests(
    importance_csv: str,
    embeddings_pkl: str,
    num_roles: int = 4,
    lambda_param: float = 0.5
) -> Tuple[Dict[str, List[str]], List[str], pd.DataFrame]:
    """
    Select agents for all tests using MMR.

    This function matches the signature of the original select_top_roles()
    for backward compatibility with exp_mmlu_evaluation.sh.

    Args:
        importance_csv: Path to importance CSV file (e.g., importance_1to7.csv)
        embeddings_pkl: Path to embeddings pickle file
        num_roles: Number of agents to select per test
        lambda_param: MMR trade-off parameter (0.0=diversity, 1.0=importance)

    Returns:
        Tuple of:
        - selected_roles: Dict mapping filename -> list of selected agent names
        - all_role_names: List of all available role names
        - importance_df: Original importance DataFrame
    """
    # Load importance scores
    df = pd.read_csv(importance_csv)

    # Extract role columns
    role_cols = [c for c in df.columns if c.endswith('_imp')]
    role_names = [c.replace('_imp', '') for c in role_cols]

    # Load embeddings
    embeddings_dict = load_embeddings(embeddings_pkl)

    # Select agents for each test
    selected_roles = {}

    for _, row in df.iterrows():
        filename = row['filename']
        subject = extract_subject_from_filename(filename)

        # Build importance dictionary for this test
        importance_scores = {
            role_names[i]: row[role_cols[i]]
            for i in range(len(role_cols))
        }

        # Apply MMR selection
        try:
            selected = select_agents_mmr(
                importance_scores,
                embeddings_dict,
                subject,
                num_roles,
                lambda_param
            )
            selected_roles[filename] = selected

        except Exception as e:
            print(f"⚠️  Error selecting agents for {filename}: {e}")
            print(f"   Falling back to greedy selection")

            # Fallback to greedy
            sorted_agents = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            selected_roles[filename] = [agent for agent, _ in sorted_agents[:num_roles]]

    return selected_roles, role_names, df


def main():
    parser = argparse.ArgumentParser(
        description='Select agents using MMR (balancing importance and diversity)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'importance_csv',
        type=str,
        help='Path to importance CSV file (e.g., importance_1to7.csv)'
    )
    parser.add_argument(
        'embeddings_pkl',
        type=str,
        help='Path to embeddings pickle file (e.g., embeddings_agent_subject.pkl)'
    )
    parser.add_argument(
        '--num-roles',
        type=int,
        default=4,
        help='Number of agents to select per test'
    )
    parser.add_argument(
        '--lambda',
        type=float,
        default=0.5,
        dest='lambda_param',
        help='MMR trade-off: 1.0=pure importance, 0.0=pure diversity'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for selected roles (optional)'
    )

    args = parser.parse_args()

    # Validate lambda parameter
    if not (0.0 <= args.lambda_param <= 1.0):
        parser.error("--lambda must be between 0.0 and 1.0")

    print(f"🔮 MMR Agent Selection")
    print(f"   Lambda (importance weight): {args.lambda_param}")
    print(f"   Number of roles: {args.num_roles}")
    print(f"   Importance CSV: {args.importance_csv}")
    print(f"   Embeddings: {args.embeddings_pkl}")
    print()

    # Run selection
    selected_roles, all_roles, df = select_agents_for_all_tests(
        args.importance_csv,
        args.embeddings_pkl,
        args.num_roles,
        args.lambda_param
    )

    print(f"✅ Selected agents for {len(selected_roles)} tests")
    print()

    # Print sample selections
    print("📋 Sample selections (first 5 tests):")
    for i, (filename, roles) in enumerate(list(selected_roles.items())[:5]):
        print(f"   {filename}: {', '.join(roles)}")

    # Optionally save to JSON
    if args.output:
        import json
        output_data = {
            'selected_roles': selected_roles,
            'all_roles': all_roles,
            'parameters': {
                'num_roles': args.num_roles,
                'lambda': args.lambda_param
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Saved selections to: {args.output}")


if __name__ == '__main__':
    main()
