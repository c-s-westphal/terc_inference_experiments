#!/usr/bin/env python3
"""
Single Condition 1 validation experiment.

Validates that Condition 1 (no perfectly redundant different-sized subsets)
holds for a given environment by computing redundancy for all subset pairs.

Usage:
    python run_single_validation.py --env cartpole --seed 0
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from condition1_validation.mi_estimators import (
    compute_entropy_discrete,
    compute_redundancy,
    compute_mi,
    compute_differential_entropy_knn,
)
from condition1_validation.subset_enumeration import (
    enumerate_subset_pairs,
    get_env_config,
)
from condition1_validation.collect_trajectories import (
    collect_trajectories,
    load_trajectories,
    save_trajectories,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Condition 1 validation experiment')
    parser.add_argument('--env', type=str, required=True,
                        help='Environment name')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of trajectory samples to collect')
    parser.add_argument('--k_neighbors', type=int, default=5,
                        help='Number of neighbors for KSG/mixed MI estimation')
    parser.add_argument('--data_dir', type=str, default='trajectories',
                        help='Directory for trajectory data')
    parser.add_argument('--model_dir', type=str, default='models_trained',
                        help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='condition1_validation/results/individual',
                        help='Directory for output JSON')
    parser.add_argument('--use_cached_trajectories', action='store_true',
                        help='Use cached trajectories if available')
    parser.add_argument('--save_trajectories', action='store_true',
                        help='Save collected trajectories')
    return parser.parse_args()


def compute_action_entropy(actions: np.ndarray, action_type: str, n_actions: int = None) -> float:
    """
    Compute entropy of actions H(A).

    Args:
        actions: Action array
        action_type: 'discrete' or 'continuous'
        n_actions: Number of discrete actions (for reference)

    Returns:
        Entropy in bits
    """
    if action_type == 'discrete':
        return compute_entropy_discrete(actions)
    else:
        # For continuous actions, use differential entropy
        # Note: differential entropy can be negative
        h_diff = compute_differential_entropy_knn(actions, k=5)
        # Convert from nats to bits
        return h_diff * np.log2(np.e)


def main():
    args = parse_args()

    # Set seed for reproducibility
    np.random.seed(args.seed)

    # Get environment configuration
    config = get_env_config(args.env)
    if config is None:
        print(f"ERROR: Unknown environment: {args.env}")
        return 1

    print(f"\n{'='*70}")
    print(f"Condition 1 Validation: {args.env}")
    print(f"{'='*70}")
    print(f"TERC variables: {config['terc_vars']}")
    print(f"State type: {config['state_type']}")
    print(f"Action type: {config['action_type']}")
    print(f"Seed: {args.seed}")
    print(f"{'='*70}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or collect trajectory data
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    try:
        if args.use_cached_trajectories:
            print("Attempting to load cached trajectories...")
            states, actions = load_trajectories(args.env, args.seed, data_dir)
            print(f"Loaded {len(states)} cached samples")
        else:
            raise FileNotFoundError("Not using cache")
    except FileNotFoundError:
        print(f"Collecting {args.n_samples} trajectory samples...")
        states, actions = collect_trajectories(
            args.env,
            args.n_samples,
            args.seed,
            model_dir if model_dir.exists() else None
        )
        print(f"Collected {len(states)} samples")

        if args.save_trajectories:
            save_trajectories(states, actions, args.env, args.seed, data_dir)

    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")

    # Compute action entropy (threshold for Condition 1)
    H_A = compute_action_entropy(actions, config['action_type'], config.get('n_actions'))
    print(f"\nAction entropy H(A): {H_A:.4f} bits")

    # Enumerate all subset pairs
    pairs = enumerate_subset_pairs(config['n_vars'], config['terc_vars'])
    print(f"Number of subset pairs to check: {len(pairs)}")

    # Compute redundancy for each pair
    results = {
        'environment': args.env,
        'seed': args.seed,
        'n_vars': config['n_vars'],
        'terc_vars': config['terc_vars'],
        'state_type': config['state_type'],
        'action_type': config['action_type'],
        'H_A': float(H_A),
        'n_samples': len(states),
        'k_neighbors': args.k_neighbors,
        'timestamp': datetime.now().isoformat(),
        'pairs': []
    }

    print(f"\n{'='*70}")
    print("Computing redundancy for each subset pair...")
    print(f"{'='*70}")

    max_neg_R = -float('inf')
    any_violation = False

    for i, (s1_indices, s2_indices, label) in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] {label}")

        try:
            neg_R = compute_redundancy(
                states, actions,
                s1_indices, s2_indices,
                config['state_type'], config['action_type'],
                k=args.k_neighbors
            )

            # Check if this is a violation
            is_violation = neg_R >= H_A
            assumption_satisfied = not is_violation

            if neg_R > max_neg_R:
                max_neg_R = neg_R

            if is_violation:
                any_violation = True

            pair_result = {
                's1_indices': list(s1_indices),
                's2_indices': list(s2_indices),
                'label': label,
                'neg_R': float(neg_R),
                'H_A': float(H_A),
                'margin': float(H_A - neg_R),
                'assumption_satisfied': assumption_satisfied,
                'error': None
            }

            status = "SATISFIED" if assumption_satisfied else "VIOLATED"
            status_symbol = "+" if assumption_satisfied else "X"
            print(f"    -R = {neg_R:.4f}, H(A) = {H_A:.4f}, margin = {H_A - neg_R:.4f} [{status_symbol} {status}]")

        except Exception as e:
            print(f"    ERROR: {e}")
            pair_result = {
                's1_indices': list(s1_indices),
                's2_indices': list(s2_indices),
                'label': label,
                'neg_R': None,
                'H_A': float(H_A),
                'margin': None,
                'assumption_satisfied': None,
                'error': str(e)
            }

        results['pairs'].append(pair_result)

    # Overall result
    valid_pairs = [p for p in results['pairs'] if p['neg_R'] is not None]
    condition1_satisfied = all(p['assumption_satisfied'] for p in valid_pairs) if valid_pairs else None

    results['condition1_satisfied'] = condition1_satisfied
    results['max_neg_R'] = float(max_neg_R) if max_neg_R > -float('inf') else None
    results['n_valid_pairs'] = len(valid_pairs)
    results['n_violations'] = sum(1 for p in valid_pairs if not p['assumption_satisfied'])

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Environment: {args.env}")
    print(f"Seed: {args.seed}")
    print(f"H(A) = {H_A:.4f} bits")
    print(f"Max(-R) = {max_neg_R:.4f} bits")
    print(f"Margin = {H_A - max_neg_R:.4f} bits")
    print(f"Pairs checked: {len(valid_pairs)}/{len(pairs)}")
    print(f"Violations: {results['n_violations']}")

    if condition1_satisfied:
        print(f"\n*** CONDITION 1 SATISFIED for {args.env} (seed {args.seed}) ***")
    elif condition1_satisfied is False:
        violations = [p['label'] for p in valid_pairs if not p['assumption_satisfied']]
        print(f"\n!!! CONDITION 1 VIOLATED for {args.env} (seed {args.seed}) !!!")
        print(f"Violating pairs: {violations[:5]}{'...' if len(violations) > 5 else ''}")
    else:
        print(f"\n??? CONDITION 1 UNDETERMINED (errors in computation) ???")

    # Save results
    output_file = output_dir / f"{args.env}_seed{args.seed}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return 0 if condition1_satisfied else 1


if __name__ == '__main__':
    exit(main())
