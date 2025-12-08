#!/usr/bin/env python3
"""
Compare greedy vs MMR agent selection.

Shows how many selections differ between pure importance (greedy)
and diversity-aware (MMR) approaches.
"""

import argparse
import pickle
from typing import Dict, List, Tuple
import pandas as pd
from mmr_selection import select_agents_for_all_tests


def compare_selections(
    importance_csv: str,
    embeddings_pkl: str,
    num_roles: int = 4,
    lambda_mmr: float = 0.7
) -> Dict:
    """
    Compare greedy (λ=1.0) vs MMR (λ=lambda_mmr) selections.

    Args:
        importance_csv: Path to importance CSV
        embeddings_pkl: Path to embeddings file
        num_roles: Number of roles to select
        lambda_mmr: Lambda for MMR comparison

    Returns:
        Dictionary with comparison statistics
    """
    print(f"🔍 Comparing Greedy (λ=1.0) vs MMR (λ={lambda_mmr})")
    print(f"   Selecting {num_roles} roles per test\n")

    # Greedy selection (λ=1.0)
    print("📊 Running greedy selection (λ=1.0)...")
    greedy_selected, all_roles, df = select_agents_for_all_tests(
        importance_csv,
        embeddings_pkl,
        num_roles,
        lambda_param=1.0
    )

    # MMR selection
    print(f"\n🔮 Running MMR selection (λ={lambda_mmr})...")
    mmr_selected, _, _ = select_agents_for_all_tests(
        importance_csv,
        embeddings_pkl,
        num_roles,
        lambda_param=lambda_mmr
    )

    # Compare selections
    print("\n📈 Computing differences...\n")

    total_tests = len(greedy_selected)
    total_selections = total_tests * num_roles

    # Track differences
    tests_with_changes = 0
    total_role_changes = 0
    changes_per_test = []
    detailed_changes = []

    for test_name in greedy_selected.keys():
        greedy_roles = set(greedy_selected[test_name])
        mmr_roles = set(mmr_selected[test_name])

        # Count differences
        removed = greedy_roles - mmr_roles
        added = mmr_roles - greedy_roles
        num_changes = len(removed)  # Same as len(added)

        if num_changes > 0:
            tests_with_changes += 1
            total_role_changes += num_changes
            changes_per_test.append(num_changes)

            detailed_changes.append({
                'test': test_name,
                'greedy': greedy_selected[test_name],
                'mmr': mmr_selected[test_name],
                'removed': list(removed),
                'added': list(added),
                'num_changes': num_changes
            })

    # Compute statistics
    prop_tests_changed = tests_with_changes / total_tests
    prop_roles_changed = total_role_changes / total_selections
    avg_changes_per_test = total_role_changes / total_tests
    avg_changes_when_changed = (
        total_role_changes / tests_with_changes if tests_with_changes > 0 else 0
    )

    return {
        'total_tests': total_tests,
        'total_selections': total_selections,
        'tests_with_changes': tests_with_changes,
        'total_role_changes': total_role_changes,
        'prop_tests_changed': prop_tests_changed,
        'prop_roles_changed': prop_roles_changed,
        'avg_changes_per_test': avg_changes_per_test,
        'avg_changes_when_changed': avg_changes_when_changed,
        'changes_per_test': changes_per_test,
        'detailed_changes': detailed_changes,
        'num_roles': num_roles,
        'lambda_mmr': lambda_mmr
    }


def print_summary(results: Dict):
    """Print summary statistics."""
    print("=" * 80)
    print("GREEDY vs MMR SELECTION COMPARISON")
    print("=" * 80)

    print(f"\nParameters:")
    print(f"  Roles per test: {results['num_roles']}")
    print(f"  MMR Lambda: {results['lambda_mmr']}")
    print(f"  Total tests: {results['total_tests']}")

    print(f"\n📊 Overall Statistics:")
    print(f"  Tests with changes: {results['tests_with_changes']} / {results['total_tests']} "
          f"({results['prop_tests_changed']:.1%})")
    print(f"  Total role changes: {results['total_role_changes']} / {results['total_selections']} "
          f"({results['prop_roles_changed']:.1%})")
    print(f"  Avg changes per test (all): {results['avg_changes_per_test']:.2f} roles")
    print(f"  Avg changes per test (when changed): {results['avg_changes_when_changed']:.2f} roles")

    # Distribution of changes
    if results['changes_per_test']:
        print(f"\n📈 Distribution of Changes:")
        from collections import Counter
        dist = Counter(results['changes_per_test'])
        for n_changes in sorted(dist.keys()):
            count = dist[n_changes]
            print(f"  {n_changes} role(s) changed: {count} tests ({count/results['tests_with_changes']:.1%})")

    # Sample changes
    print(f"\n🔍 Sample Changes (first 10):")
    for i, change in enumerate(results['detailed_changes'][:10], 1):
        print(f"\n  {i}. {change['test']}")
        print(f"     Greedy: {', '.join(change['greedy'])}")
        print(f"     MMR:    {', '.join(change['mmr'])}")
        print(f"     Removed: {', '.join(change['removed'])}")
        print(f"     Added:   {', '.join(change['added'])}")

    if len(results['detailed_changes']) > 10:
        print(f"\n  ... and {len(results['detailed_changes']) - 10} more tests with changes")

    print("\n" + "=" * 80)


def save_detailed_report(results: Dict, output_file: str):
    """Save detailed comparison to CSV."""
    import csv

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'test', 'num_changes', 'greedy_roles', 'mmr_roles', 'removed', 'added'
        ])
        writer.writeheader()

        for change in results['detailed_changes']:
            writer.writerow({
                'test': change['test'],
                'num_changes': change['num_changes'],
                'greedy_roles': ', '.join(change['greedy']),
                'mmr_roles': ', '.join(change['mmr']),
                'removed': ', '.join(change['removed']),
                'added': ', '.join(change['added'])
            })

    print(f"\n💾 Detailed report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare greedy vs MMR agent selection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'importance_csv',
        type=str,
        help='Path to importance CSV file'
    )
    parser.add_argument(
        'embeddings_pkl',
        type=str,
        help='Path to embeddings pickle file'
    )
    parser.add_argument(
        '--num-roles',
        type=int,
        default=4,
        help='Number of roles to select per test'
    )
    parser.add_argument(
        '--lambda',
        type=float,
        default=0.7,
        dest='lambda_mmr',
        help='Lambda parameter for MMR (greedy is always 1.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file for detailed comparison (optional)'
    )

    args = parser.parse_args()

    # Run comparison
    results = compare_selections(
        args.importance_csv,
        args.embeddings_pkl,
        args.num_roles,
        args.lambda_mmr
    )

    # Print summary
    print_summary(results)

    # Save detailed report if requested
    if args.output:
        save_detailed_report(results, args.output)


if __name__ == '__main__':
    main()
