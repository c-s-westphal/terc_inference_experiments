#!/usr/bin/env python3
"""
Aggregate individual experiment results and print summary table.

This script combines results from all individual JSON files produced by
the parallelized job array and generates:
1. Aggregated statistics across seeds
2. Speedup comparisons (full vs TERC)
3. Formatted summary tables

Usage:
    python aggregate_results.py --input_dir results/individual --output results/inference_results.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate TERC inference results')
    parser.add_argument('--input_dir', type=str, default='results/individual',
                        help='Directory containing individual JSON results')
    parser.add_argument('--output', type=str, default='results/inference_results.json',
                        help='Output JSON file with aggregated results')
    parser.add_argument('--format', type=str, default='table',
                        choices=['table', 'latex', 'csv'],
                        help='Output format for summary')
    parser.add_argument('--fresh-only', action='store_true',
                        help='Only use freshly trained results (exclude checkpoint-loaded)')
    return parser.parse_args()


def load_all_results(input_dir: Path) -> List[dict]:
    """Load all individual result JSON files."""
    results = []
    json_files = list(input_dir.glob('*.json'))

    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return results


def aggregate_by_config(results: List[dict]) -> dict:
    """
    Group results by (env, state_type) and aggregate across seeds.

    Returns:
        Dictionary with aggregated statistics for each configuration
    """
    grouped = defaultdict(list)

    for r in results:
        key = (r['environment'], r['state_type'])
        grouped[key].append(r)

    aggregated = {}
    for (env, state_type), runs in grouped.items():
        key = f"{env}_{state_type}"

        # Get consistent values
        state_dim = runs[0]['state_dim']
        action_dim = runs[0]['action_dim']
        n_params = runs[0]['n_parameters']

        # Aggregate inference times across seeds
        inference_agg = {}
        timing_keys = list(runs[0]['inference_times'].keys())

        for timing_key in timing_keys:
            means = [r['inference_times'][timing_key]['mean_ms'] for r in runs]
            stds = [r['inference_times'][timing_key]['std_ms'] for r in runs]
            medians = [r['inference_times'][timing_key]['median_ms'] for r in runs]

            inference_agg[timing_key] = {
                'mean_ms': float(np.mean(means)),
                'std_across_seeds_ms': float(np.std(means)),
                'median_ms': float(np.mean(medians)),
                'avg_within_seed_std_ms': float(np.mean(stds)),
                'min_mean_ms': float(np.min(means)),
                'max_mean_ms': float(np.max(means)),
                'n_seeds': len(runs)
            }

        # Aggregate training info if available
        training_agg = {}
        if 'training_info' in runs[0]:
            converged_count = sum(1 for r in runs
                                  if r.get('training_info', {}).get('converged', False))
            episodes = [r['training_info'].get('episodes_trained', 0) for r in runs
                        if 'episodes_trained' in r.get('training_info', {})]
            rewards = [r['training_info'].get('final_avg_reward', 0) for r in runs
                       if 'final_avg_reward' in r.get('training_info', {})]

            training_agg = {
                'converged_count': converged_count,
                'total_runs': len(runs),
                'avg_episodes': float(np.mean(episodes)) if episodes else 0,
                'avg_final_reward': float(np.mean(rewards)) if rewards else 0,
                'std_final_reward': float(np.std(rewards)) if rewards else 0,
            }

        aggregated[key] = {
            'environment': env,
            'state_type': state_type,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'n_parameters': n_params,
            'n_seeds': len(runs),
            'seeds': sorted([r['seed'] for r in runs]),
            'inference_times': inference_agg,
            'training': training_agg
        }

    return aggregated


def compute_speedups(aggregated: dict) -> dict:
    """
    Compute speedup factors comparing full vs terc states.

    Returns:
        Dictionary with speedup statistics for each environment
    """
    speedups = {}

    envs = set(k.rsplit('_', 1)[0] for k in aggregated.keys())

    for env in envs:
        full_key = f"{env}_full"
        terc_key = f"{env}_terc"

        if full_key not in aggregated or terc_key not in aggregated:
            print(f"Warning: Missing data for {env} (full or terc)")
            continue

        full_data = aggregated[full_key]
        terc_data = aggregated[terc_key]

        env_speedups = {
            'full_state_dim': full_data['state_dim'],
            'terc_state_dim': terc_data['state_dim'],
            'state_dim_reduction_pct': 100 * (1 - terc_data['state_dim'] / full_data['state_dim']),
            'full_params': full_data['n_parameters'],
            'terc_params': terc_data['n_parameters'],
            'param_reduction_pct': 100 * (1 - terc_data['n_parameters'] / full_data['n_parameters']),
            'speedups_by_config': {}
        }

        # Speedup for each timing configuration
        for timing_key in full_data['inference_times'].keys():
            full_time = full_data['inference_times'][timing_key]['mean_ms']
            terc_time = terc_data['inference_times'][timing_key]['mean_ms']

            if terc_time > 0:
                speedup = full_time / terc_time
            else:
                speedup = float('inf')

            env_speedups['speedups_by_config'][timing_key] = {
                'speedup': speedup,
                'full_ms': full_time,
                'terc_ms': terc_time,
                'time_saved_ms': full_time - terc_time,
                'time_saved_pct': 100 * (1 - terc_time / full_time) if full_time > 0 else 0
            }

        speedups[env] = env_speedups

    return speedups


def print_summary_table(aggregated: dict, speedups: dict):
    """Print formatted summary table."""
    print("\n" + "=" * 110)
    print("INFERENCE TIME COMPARISON: Full State vs TERC-Selected State")
    print("=" * 110)

    envs = ['cartpole', 'lunarlander', 'pendulum', 'skg25', 'skg50']
    env_display = {
        'cartpole': 'CartPole',
        'lunarlander': 'LunarLander',
        'pendulum': 'Pendulum',
        'skg25': 'SKG-25',
        'skg50': 'SKG-50'
    }

    # Determine which timing configs are available
    sample_key = list(aggregated.keys())[0] if aggregated else None
    if sample_key:
        timing_configs = list(aggregated[sample_key]['inference_times'].keys())
    else:
        timing_configs = ['cpu_batch1', 'cpu_batch64']

    for timing_key in timing_configs:
        parts = timing_key.split('_')
        device = parts[0].upper()
        batch_size = parts[1].replace('batch', '')

        print(f"\n{'='*110}")
        print(f"Device: {device} | Batch Size: {batch_size}")
        print("-" * 110)
        print(f"{'Environment':<12} | {'State Dim':<12} | {'Full (ms)':<15} | "
              f"{'TERC (ms)':<15} | {'Speedup':<8} | {'Params (Full->TERC)':<28}")
        print("-" * 110)

        for env in envs:
            full_key = f"{env}_full"
            terc_key = f"{env}_terc"

            if full_key not in aggregated or terc_key not in aggregated:
                continue

            full_data = aggregated[full_key]
            terc_data = aggregated[terc_key]

            if timing_key not in full_data['inference_times']:
                continue

            full_timing = full_data['inference_times'][timing_key]
            terc_timing = terc_data['inference_times'][timing_key]

            speedup = speedups[env]['speedups_by_config'].get(timing_key, {}).get('speedup', 0)
            param_reduction = speedups[env]['param_reduction_pct']

            dim_str = f"{full_data['state_dim']} -> {terc_data['state_dim']}"
            param_str = f"{full_data['n_parameters']} -> {terc_data['n_parameters']} (-{param_reduction:.0f}%)"

            print(f"{env_display[env]:<12} | "
                  f"{dim_str:<12} | "
                  f"{full_timing['mean_ms']:>6.4f}+/-{full_timing['std_across_seeds_ms']:.4f} | "
                  f"{terc_timing['mean_ms']:>6.4f}+/-{terc_timing['std_across_seeds_ms']:.4f} | "
                  f"{speedup:>6.2f}x | "
                  f"{param_str}")

        print("-" * 110)

    # Print overall summary
    print(f"\n{'='*110}")
    print("SUMMARY BY ENVIRONMENT")
    print("=" * 110)
    print(f"{'Environment':<12} | {'Full Dim':<10} | {'TERC Dim':<10} | "
          f"{'Dim Reduction':<15} | {'Param Reduction':<15} | {'Seeds'}")
    print("-" * 110)

    for env in envs:
        if env not in speedups:
            continue

        s = speedups[env]
        full_key = f"{env}_full"

        n_seeds = aggregated.get(full_key, {}).get('n_seeds', 0)

        print(f"{env_display[env]:<12} | "
              f"{s['full_state_dim']:<10} | "
              f"{s['terc_state_dim']:<10} | "
              f"{s['state_dim_reduction_pct']:>12.1f}% | "
              f"{s['param_reduction_pct']:>13.1f}% | "
              f"{n_seeds}")

    print("=" * 110)


def print_latex_table(aggregated: dict, speedups: dict):
    """Print LaTeX formatted table."""
    print("\n% LaTeX Table")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Inference Time Comparison: Full State vs TERC-Selected State}")
    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    print("Environment & Full Dim & TERC Dim & Full (ms) & TERC (ms) & Speedup \\\\")
    print("\\hline")

    envs = ['cartpole', 'lunarlander', 'pendulum', 'skg25', 'skg50']
    env_display = {
        'cartpole': 'CartPole',
        'lunarlander': 'LunarLander',
        'pendulum': 'Pendulum',
        'skg25': 'SKG-25',
        'skg50': 'SKG-50'
    }

    timing_key = 'cpu_batch1'

    for env in envs:
        full_key = f"{env}_full"
        terc_key = f"{env}_terc"

        if full_key not in aggregated or terc_key not in aggregated:
            continue

        full_data = aggregated[full_key]
        terc_data = aggregated[terc_key]

        if timing_key not in full_data['inference_times']:
            continue

        full_timing = full_data['inference_times'][timing_key]
        terc_timing = terc_data['inference_times'][timing_key]

        speedup = speedups[env]['speedups_by_config'].get(timing_key, {}).get('speedup', 0)

        print(f"{env_display[env]} & "
              f"{full_data['state_dim']} & "
              f"{terc_data['state_dim']} & "
              f"{full_timing['mean_ms']:.4f} & "
              f"{terc_timing['mean_ms']:.4f} & "
              f"{speedup:.2f}x \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


def print_csv(aggregated: dict, speedups: dict):
    """Print CSV formatted output."""
    print("environment,state_type,state_dim,n_params,device,batch_size,mean_ms,std_ms,median_ms")

    for key, data in sorted(aggregated.items()):
        for timing_key, timing in data['inference_times'].items():
            parts = timing_key.split('_')
            device = parts[0]
            batch_size = parts[1].replace('batch', '')

            print(f"{data['environment']},{data['state_type']},{data['state_dim']},"
                  f"{data['n_parameters']},{device},{batch_size},"
                  f"{timing['mean_ms']:.6f},{timing['std_across_seeds_ms']:.6f},"
                  f"{timing['median_ms']:.6f}")


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load all results
    print(f"Loading results from {input_dir}...")
    results = load_all_results(input_dir)
    print(f"Loaded {len(results)} individual result files")

    # Filter for fresh-only if requested
    if args.fresh_only:
        original_count = len(results)
        results = [r for r in results
                   if not r.get('training_info', {}).get('loaded_from_checkpoint', True)]
        print(f"Filtered to {len(results)} freshly-trained results (excluded {original_count - len(results)} checkpoint-loaded)")

    if len(results) == 0:
        print("ERROR: No result files found!")
        print(f"Expected files in: {input_dir.absolute()}")
        return 1

    # Check for missing configurations
    expected_configs = set()
    for env in ['cartpole', 'lunarlander', 'pendulum', 'skg25', 'skg50']:
        for state_type in ['full', 'terc']:
            for seed in range(5):
                expected_configs.add((env, state_type, seed))

    found_configs = set((r['environment'], r['state_type'], r['seed']) for r in results)
    missing = expected_configs - found_configs

    if missing:
        print(f"\nWarning: Missing {len(missing)} configurations:")
        for env, st, seed in sorted(missing)[:10]:
            print(f"  {env}_{st}_seed{seed}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    # Aggregate
    aggregated = aggregate_by_config(results)
    speedups = compute_speedups(aggregated)

    # Save combined results
    output_data = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'n_results': len(results),
            'n_configs': len(aggregated),
            'n_missing': len(missing),
        },
        'aggregated': aggregated,
        'speedups': speedups,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nAggregated results saved to {output_file}")

    # Print output in requested format
    if args.format == 'table':
        print_summary_table(aggregated, speedups)
    elif args.format == 'latex':
        print_latex_table(aggregated, speedups)
    elif args.format == 'csv':
        print_csv(aggregated, speedups)

    return 0


if __name__ == '__main__':
    exit(main())
