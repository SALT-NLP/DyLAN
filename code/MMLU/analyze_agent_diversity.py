#!/usr/bin/env python3
"""
Analyze diversity of agent reasoning approaches using embedding similarity.

This script:
1. Loads embeddings from embeddings_agent_subject.pkl
2. Computes pairwise cosine similarity between agents for each subject
3. Calculates diversity metrics (min, max, median, mean)
4. Outputs detailed analysis and summary statistics
"""

import argparse
import pickle
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine as cosine_distance


def compute_pairwise_similarities(embeddings_dict: Dict[Tuple[str, str], np.ndarray]) -> Dict[str, pd.DataFrame]:
    """
    Compute pairwise cosine similarity between agents for each subject.

    Args:
        embeddings_dict: Dictionary mapping (agent_role, subject) -> embedding vector

    Returns:
        Dictionary mapping subject -> DataFrame of pairwise similarities
    """
    # Group embeddings by subject
    subjects = {}
    for (agent_role, subject), embedding in embeddings_dict.items():
        if subject not in subjects:
            subjects[subject] = {}
        subjects[subject][agent_role] = embedding

    # Compute pairwise similarities for each subject
    similarity_matrices = {}

    for subject, agent_embeddings in subjects.items():
        agents = sorted(agent_embeddings.keys())
        n_agents = len(agents)

        # Create embedding matrix
        embedding_matrix = np.array([agent_embeddings[agent] for agent in agents])

        # Compute cosine similarity matrix
        sim_matrix = cosine_similarity(embedding_matrix)

        # Create DataFrame with agent names as index/columns
        df = pd.DataFrame(sim_matrix, index=agents, columns=agents)
        similarity_matrices[subject] = df

    return similarity_matrices


def compute_diversity_metrics(similarity_matrices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute diversity metrics for each subject.

    Args:
        similarity_matrices: Dictionary mapping subject -> similarity DataFrame

    Returns:
        DataFrame with diversity metrics per subject
    """
    metrics = []

    for subject, sim_df in similarity_matrices.items():
        # Extract upper triangle (excluding diagonal) for unique pairs
        n = len(sim_df)
        upper_triangle_mask = np.triu(np.ones((n, n)), k=1).astype(bool)
        similarities = sim_df.values[upper_triangle_mask]

        # Compute metrics
        metrics.append({
            'subject': subject,
            'min_similarity': similarities.min(),
            'max_similarity': similarities.max(),
            'mean_similarity': similarities.mean(),
            'median_similarity': np.median(similarities),
            'std_similarity': similarities.std(),
            'n_agent_pairs': len(similarities),
            'n_agents': len(sim_df)
        })

    return pd.DataFrame(metrics).sort_values('subject')


def find_extreme_pairs(similarity_matrices: Dict[str, pd.DataFrame]) -> Dict:
    """
    Find most similar and most diverse agent pairs for each subject.

    Args:
        similarity_matrices: Dictionary mapping subject -> similarity DataFrame

    Returns:
        Dictionary with extreme pairs per subject
    """
    extreme_pairs = {}

    for subject, sim_df in similarity_matrices.items():
        n = len(sim_df)

        # Find min and max (excluding diagonal)
        max_sim = -1
        min_sim = 2
        max_pair = None
        min_pair = None

        for i in range(n):
            for j in range(i + 1, n):
                sim = sim_df.iloc[i, j]

                if sim > max_sim:
                    max_sim = sim
                    max_pair = (sim_df.index[i], sim_df.columns[j])

                if sim < min_sim:
                    min_sim = sim
                    min_pair = (sim_df.index[i], sim_df.columns[j])

        extreme_pairs[subject] = {
            'most_similar': {
                'agents': max_pair,
                'similarity': float(max_sim)
            },
            'most_diverse': {
                'agents': min_pair,
                'similarity': float(min_sim)
            }
        }

    return extreme_pairs


def compute_global_statistics(metrics_df: pd.DataFrame) -> Dict:
    """
    Compute global statistics across all subjects.

    Args:
        metrics_df: DataFrame with per-subject metrics

    Returns:
        Dictionary of global statistics
    """
    return {
        'overall_min_similarity': float(metrics_df['min_similarity'].min()),
        'overall_max_similarity': float(metrics_df['max_similarity'].max()),
        'median_of_subject_medians': float(metrics_df['median_similarity'].median()),
        'mean_of_subject_means': float(metrics_df['mean_similarity'].mean()),
        'median_of_subject_mins': float(metrics_df['min_similarity'].median()),
        'median_of_subject_maxs': float(metrics_df['max_similarity'].median()),
        'n_subjects': len(metrics_df)
    }


def print_summary(metrics_df: pd.DataFrame, global_stats: Dict, extreme_pairs: Dict):
    """
    Print a human-readable summary of the analysis.
    """
    print("\n" + "="*80)
    print("AGENT DIVERSITY ANALYSIS - SUMMARY")
    print("="*80)

    print("\n📊 GLOBAL STATISTICS (across all subjects)")
    print("-" * 80)
    print(f"Number of subjects analyzed: {global_stats['n_subjects']}")
    print(f"\nSimilarity ranges:")
    print(f"  Overall minimum similarity: {global_stats['overall_min_similarity']:.4f}")
    print(f"  Overall maximum similarity: {global_stats['overall_max_similarity']:.4f}")
    print(f"\nCentral tendencies:")
    print(f"  Median of subject medians: {global_stats['median_of_subject_medians']:.4f}")
    print(f"  Mean of subject means: {global_stats['mean_of_subject_means']:.4f}")
    print(f"\nTypical ranges per subject:")
    print(f"  Median of subject minimums: {global_stats['median_of_subject_mins']:.4f}")
    print(f"  Median of subject maximums: {global_stats['median_of_subject_maxs']:.4f}")

    print("\n\n📈 PER-SUBJECT STATISTICS")
    print("-" * 80)
    print("Top 10 subjects with HIGHEST diversity (lowest median similarity):")
    top_diverse = metrics_df.nsmallest(10, 'median_similarity')[['subject', 'median_similarity', 'min_similarity', 'max_similarity']]
    print(top_diverse.to_string(index=False))

    print("\n\nTop 10 subjects with LOWEST diversity (highest median similarity):")
    top_similar = metrics_df.nlargest(10, 'median_similarity')[['subject', 'median_similarity', 'min_similarity', 'max_similarity']]
    print(top_similar.to_string(index=False))

    print("\n\n🔍 EXTREME AGENT PAIRS")
    print("-" * 80)

    # Find global extremes
    all_diverse = [(subj, data['most_diverse']['similarity']) for subj, data in extreme_pairs.items()]
    all_similar = [(subj, data['most_similar']['similarity']) for subj, data in extreme_pairs.items()]

    all_diverse.sort(key=lambda x: x[1])
    all_similar.sort(key=lambda x: x[1], reverse=True)

    print("Top 5 most DIVERSE agent pairs (lowest similarity):")
    for i, (subject, sim) in enumerate(all_diverse[:5], 1):
        pair = extreme_pairs[subject]['most_diverse']['agents']
        print(f"  {i}. {subject}: {pair[0]} ↔ {pair[1]} (similarity: {sim:.4f})")

    print("\nTop 5 most SIMILAR agent pairs (highest similarity):")
    for i, (subject, sim) in enumerate(all_similar[:5], 1):
        pair = extreme_pairs[subject]['most_similar']['agents']
        print(f"  {i}. {subject}: {pair[0]} ↔ {pair[1]} (similarity: {sim:.4f})")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze agent diversity using embeddings')
    parser.add_argument(
        '--embeddings',
        type=str,
        default='embeddings_agent_subject.pkl',
        help='Path to embeddings pickle file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='diversity_analysis',
        help='Directory to save output files'
    )
    parser.add_argument(
        '--save-matrices',
        action='store_true',
        help='Save individual similarity matrices as CSV files'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load embeddings
    print(f"📂 Loading embeddings from {args.embeddings}...")
    with open(args.embeddings, 'rb') as f:
        embeddings_dict = pickle.load(f)

    print(f"✅ Loaded {len(embeddings_dict)} embeddings")

    # Compute pairwise similarities
    print("\n🔮 Computing pairwise cosine similarities...")
    similarity_matrices = compute_pairwise_similarities(embeddings_dict)
    print(f"✅ Computed similarity matrices for {len(similarity_matrices)} subjects")

    # Compute diversity metrics
    print("\n📊 Computing diversity metrics...")
    metrics_df = compute_diversity_metrics(similarity_matrices)

    # Find extreme pairs
    print("\n🔍 Finding extreme agent pairs...")
    extreme_pairs = find_extreme_pairs(similarity_matrices)

    # Compute global statistics
    print("\n📈 Computing global statistics...")
    global_stats = compute_global_statistics(metrics_df)

    # Print summary
    print_summary(metrics_df, global_stats, extreme_pairs)

    # Save outputs
    print("\n💾 Saving results...")

    # Save diversity metrics
    metrics_file = output_dir / 'diversity_metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✅ Saved diversity metrics: {metrics_file}")

    # Save global statistics
    stats_file = output_dir / 'global_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(global_stats, f, indent=2)
    print(f"✅ Saved global statistics: {stats_file}")

    # Save extreme pairs
    extremes_file = output_dir / 'extreme_pairs.json'
    with open(extremes_file, 'w') as f:
        json.dump(extreme_pairs, f, indent=2)
    print(f"✅ Saved extreme pairs: {extremes_file}")

    # Optionally save individual similarity matrices
    if args.save_matrices:
        matrices_dir = output_dir / 'similarity_matrices'
        matrices_dir.mkdir(exist_ok=True)

        for subject, sim_df in similarity_matrices.items():
            matrix_file = matrices_dir / f'{subject}_similarity.csv'
            sim_df.to_csv(matrix_file)

        print(f"✅ Saved {len(similarity_matrices)} similarity matrices: {matrices_dir}/")

    print("\n✨ Analysis complete!")
    print(f"\nResults saved in: {output_dir}/")


if __name__ == '__main__':
    main()
