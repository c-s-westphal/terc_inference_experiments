#!/usr/bin/env python3
"""
Subset enumeration for Condition 1 validation.

Enumerates all disjoint subset pairs (S1, S2) where |S1| ≠ |S2|
for computing redundancy between different-sized subsets.
"""

from itertools import combinations
from typing import List, Tuple, Dict


def enumerate_subset_pairs(
    n_vars: int,
    var_names: List[str]
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...], str]]:
    """
    Enumerate all disjoint subset pairs (S1, S2) where |S1| ≠ |S2|.

    Args:
        n_vars: Number of variables
        var_names: Names of variables for labeling

    Returns:
        List of (S1_indices, S2_indices, label_string) tuples
    """
    pairs = []
    indices = list(range(n_vars))

    # For each possible size of S1 (at least 1)
    for size1 in range(1, n_vars):
        # For each possible size of S2 (at least 1, from remaining)
        for size2 in range(1, n_vars - size1 + 1):
            if size1 == size2:
                continue  # Skip equal-sized pairs (Condition 1 is about |P| ≠ |P'|)

            # Generate all combinations for S1
            for s1_indices in combinations(indices, size1):
                remaining = [i for i in indices if i not in s1_indices]

                # Generate all combinations for S2 from remaining
                for s2_indices in combinations(remaining, size2):
                    s1_names = [var_names[i] for i in s1_indices]
                    s2_names = [var_names[i] for i in s2_indices]
                    label = f"{{{','.join(s1_names)}}} vs {{{','.join(s2_names)}}}"

                    pairs.append((tuple(s1_indices), tuple(s2_indices), label))

    return pairs


def count_subset_pairs(n_vars: int) -> int:
    """
    Count total number of disjoint subset pairs with different sizes.

    Args:
        n_vars: Number of variables

    Returns:
        Total count of pairs
    """
    count = 0
    for size1 in range(1, n_vars):
        for size2 in range(1, n_vars - size1 + 1):
            if size1 == size2:
                continue
            # Number of ways to choose size1 from n_vars
            from math import comb
            n_s1 = comb(n_vars, size1)
            # Number of ways to choose size2 from remaining (n_vars - size1)
            n_s2 = comb(n_vars - size1, size2)
            count += n_s1 * n_s2
    return count


def get_env_config(env_name: str) -> Dict:
    """
    Return configuration for each environment.

    Args:
        env_name: Environment identifier

    Returns:
        Configuration dictionary with terc_vars, n_vars, state_type, action_type, n_actions
    """
    configs = {
        'cartpole': {
            'terc_vars': ['x', 'x_dot', 'theta', 'theta_dot'],
            'n_vars': 4,
            'state_type': 'continuous',
            'action_type': 'discrete',
            'n_actions': 2,
        },
        'lunarlander': {
            'terc_vars': ['x', 'y', 'x_dot', 'y_dot', 'theta', 'theta_dot', 'left_leg', 'right_leg'],
            'n_vars': 8,
            'state_type': 'continuous',
            'action_type': 'discrete',
            'n_actions': 4,
        },
        'pendulum': {
            'terc_vars': ['cos_theta', 'sin_theta', 'theta_dot'],
            'n_vars': 3,
            'state_type': 'continuous',
            'action_type': 'continuous',
            'n_actions': None,  # Continuous
        },
        'skg25': {
            'terc_vars': ['key1', 'key2', 'key3'],
            'n_vars': 3,
            'state_type': 'discrete',
            'action_type': 'discrete',
            'n_actions': 81,
        },
        'skg50': {
            'terc_vars': ['key1', 'key2', 'key3'],
            'n_vars': 3,
            'state_type': 'discrete',
            'action_type': 'discrete',
            'n_actions': 81,
        },
    }

    # Add IPD configurations for TF(N)T with N from 3 to 10
    for n in range(3, 11):
        terc_size = n  # TERC selects last N consecutive actions
        configs[f'ipd_tf{n}t'] = {
            'terc_vars': [f't-{i}' for i in range(1, terc_size + 1)],
            'n_vars': terc_size,
            'state_type': 'discrete',
            'action_type': 'discrete',
            'n_actions': 2,
        }

    return configs.get(env_name)


def get_all_env_names() -> List[str]:
    """Return list of all environment names."""
    base_envs = ['cartpole', 'lunarlander', 'pendulum', 'skg25', 'skg50']
    ipd_envs = [f'ipd_tf{n}t' for n in range(3, 11)]
    return base_envs + ipd_envs


if __name__ == '__main__':
    # Test subset enumeration
    print("Testing subset enumeration...")

    # Test with 3 variables (like Pendulum or SKG)
    print("\n3 variables (e.g., Pendulum TERC):")
    var_names_3 = ['cos_theta', 'sin_theta', 'theta_dot']
    pairs_3 = enumerate_subset_pairs(3, var_names_3)
    print(f"Number of pairs: {len(pairs_3)}")
    for s1, s2, label in pairs_3:
        print(f"  {label}")

    # Test with 4 variables (like CartPole)
    print("\n4 variables (e.g., CartPole TERC):")
    var_names_4 = ['x', 'x_dot', 'theta', 'theta_dot']
    pairs_4 = enumerate_subset_pairs(4, var_names_4)
    print(f"Number of pairs: {len(pairs_4)}")
    for s1, s2, label in pairs_4[:10]:
        print(f"  {label}")
    if len(pairs_4) > 10:
        print(f"  ... and {len(pairs_4) - 10} more")

    # Test with 8 variables (like Lunar Lander)
    print("\n8 variables (e.g., Lunar Lander TERC):")
    var_names_8 = ['x', 'y', 'x_dot', 'y_dot', 'theta', 'theta_dot', 'left_leg', 'right_leg']
    pairs_8 = enumerate_subset_pairs(8, var_names_8)
    print(f"Number of pairs: {len(pairs_8)}")
    print(f"  (Showing only top pairs by label)")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary of subset pairs per environment:")
    print("=" * 60)
    print(f"{'Environment':<20} | {'TERC Vars':<10} | {'# Pairs':<10}")
    print("-" * 60)

    for env_name in get_all_env_names():
        config = get_env_config(env_name)
        if config:
            n_pairs = count_subset_pairs(config['n_vars'])
            print(f"{env_name:<20} | {config['n_vars']:<10} | {n_pairs:<10}")
