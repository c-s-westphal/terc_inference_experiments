#!/usr/bin/env python3
"""
Aggregate Condition 1 validation results and generate plots.

Combines results from all individual JSON files and creates:
1. Combined subplot figures for environment groups
2. Summary table of all results
3. Aggregated JSON with statistics across seeds

Usage:
    python aggregate_and_plot.py
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

# Style configuration
TAB_PALETTE = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
               'tab:pink', 'tab:brown', 'cyan']
NULL_MODEL_COLOR = 'black'
DECOY_COLOR = 'grey'
ALPHA = 0.5
NOT_VERIFIED_COLOR = 'red'
NOT_VERIFIED_ALPHA = 0.15

# Use Arial font
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.it'] = 'Arial:italic'
mpl.rcParams['mathtext.bf'] = 'Arial:bold'


def setup_axes_style(ax):
    """Apply consistent styling to axes."""
    # Remove all spines
    for spine in ax.spines.values():
        spine.set_linewidth(0)

    # White background
    ax.set_facecolor("white")

    # Y-axis grid only
    ax.grid(color='gainsboro', axis='y')
    ax.set_axisbelow(True)

    # Invisible ticks
    ax.tick_params(axis='both', length=0)

    # Tick label font size
    ax.tick_params(axis='both', labelsize=9)


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


def aggregate_by_env(results: List[dict]) -> dict:
    """Group results by environment and aggregate across seeds."""
    grouped = defaultdict(list)

    for r in results:
        grouped[r['environment']].append(r)

    aggregated = {}
    for env, runs in grouped.items():
        # Get consistent values
        H_A_values = [r['H_A'] for r in runs if r['H_A'] is not None]
        H_A_mean = np.mean(H_A_values) if H_A_values else 0
        H_A_std = np.std(H_A_values) if H_A_values else 0

        # Get max_neg_R values
        max_neg_R_values = [r.get('max_neg_R', 0) for r in runs if r.get('max_neg_R') is not None]
        max_neg_R_mean = np.mean(max_neg_R_values) if max_neg_R_values else 0
        max_neg_R_std = np.std(max_neg_R_values) if max_neg_R_values else 0

        # Aggregate pairs across seeds
        pair_labels = [p['label'] for p in runs[0]['pairs']] if runs[0]['pairs'] else []
        pair_stats = {}

        for label in pair_labels:
            neg_Rs = []
            for run in runs:
                for p in run['pairs']:
                    if p['label'] == label and p['neg_R'] is not None:
                        neg_Rs.append(p['neg_R'])
                        break

            if neg_Rs:
                pair_stats[label] = {
                    'mean': float(np.mean(neg_Rs)),
                    'std': float(np.std(neg_Rs)),
                    'min': float(np.min(neg_Rs)),
                    'max': float(np.max(neg_Rs)),
                    'n_samples': len(neg_Rs),
                }

        # Check if condition satisfied across all seeds
        condition_satisfied_list = [r.get('condition1_satisfied') for r in runs
                                    if r.get('condition1_satisfied') is not None]
        all_satisfied = all(condition_satisfied_list) if condition_satisfied_list else None

        aggregated[env] = {
            'H_A_mean': float(H_A_mean),
            'H_A_std': float(H_A_std),
            'max_neg_R_mean': float(max_neg_R_mean),
            'max_neg_R_std': float(max_neg_R_std),
            'margin_mean': float(H_A_mean - max_neg_R_mean),
            'n_seeds': len(runs),
            'n_vars': runs[0].get('n_vars'),
            'terc_vars': runs[0].get('terc_vars'),
            'state_type': runs[0].get('state_type'),
            'action_type': runs[0].get('action_type'),
            'pair_stats': pair_stats,
            'condition1_satisfied': all_satisfied,
            'n_satisfied_seeds': sum(1 for x in condition_satisfied_list if x) if condition_satisfied_list else 0,
        }

    return aggregated


def convert_to_x_notation(label: str) -> str:
    """
    Convert pair label from various notations to X_i notation.

    Examples:
        "{0,1} vs {2,3}" -> "{X_1,X_2} vs {X_3,X_4}"
        "{t-1,t-2} vs {t-3}" -> "{X_1,X_2} vs {X_3}"
        "{x,x_dot} vs {theta}" -> "{X_1,X_2} vs {X_3}"
    """
    # Mapping for gym environment variable names
    gym_var_mappings = {
        # CartPole variables
        'x': 'X_1',
        'x_dot': 'X_2',
        'theta': 'X_3',
        'theta_dot': 'X_4',
        # Pendulum variables
        'cos_theta': 'X_1',
        'sin_theta': 'X_2',
        # theta_dot already mapped above for CartPole, reuse X_3 for Pendulum
    }

    # Check if label contains gym variable names
    has_gym_vars = any(var in label for var in gym_var_mappings.keys())

    if has_gym_vars:
        result = label
        # Sort by length descending to replace longer names first (e.g., theta_dot before theta)
        for var in sorted(gym_var_mappings.keys(), key=len, reverse=True):
            if var in result:
                # For Pendulum, theta_dot should be X_3
                if var == 'theta_dot' and 'cos_theta' in label:
                    result = result.replace(var, 'X_3')
                else:
                    result = result.replace(var, gym_var_mappings[var])
        return result

    # Handle "t-N" notation (IPD environments)
    def replace_t_notation(match):
        idx = int(match.group(1))
        return f"X_{idx}"

    result = re.sub(r't-(\d+)', replace_t_notation, label)

    # Handle plain index notation
    def replace_index(match):
        idx = int(match.group(0))
        return f"X_{idx + 1}"

    # Only replace standalone numbers (not already part of X_N)
    result = re.sub(r'(?<!X_)(?<!X_\{)\b(\d+)\b', replace_index, result)

    return result


def format_pair_label(label: str) -> str:
    """Convert pair label to LaTeX-friendly format with X_i notation."""
    # First convert indices to X_i notation
    label = convert_to_x_notation(label)

    # Split by " vs "
    parts = label.split(' vs ')
    if len(parts) == 2:
        formatted_parts = []
        for part in parts:
            # Extract content between braces
            content = part.strip('{}')
            # Format as math mode with proper subscripts
            content = re.sub(r'X_(\d+)', lambda m: f'X_{{{m.group(1)}}}', content)
            formatted_parts.append(r'$\{' + content + r'\}$')
        return ' vs '.join(formatted_parts)

    return label


def plot_single_env_ax(ax, env_name: str, data: dict, top_n: int = 15, show_ylabel: bool = True):
    """Plot a single environment on the given axes."""

    pair_stats = data['pair_stats']
    H_A = data['H_A_mean']

    if not pair_stats:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title(env_name.replace('_', ' ').title(), fontsize=12)
        setup_axes_style(ax)
        return

    # Sort by mean -R (descending) and take top N
    sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    if len(sorted_pairs) > top_n:
        sorted_pairs = sorted_pairs[:top_n]

    labels = [format_pair_label(p[0]) for p in sorted_pairs]
    means = [p[1]['mean'] for p in sorted_pairs]
    stds = [p[1]['std'] for p in sorted_pairs]

    x = np.arange(len(labels))

    # Color bars based on whether they're below threshold
    bar_colors = ['tab:green' if m < H_A else 'tab:red' for m in means]

    # Vertical bar plot
    ax.bar(x, means, align='center', color=bar_colors, alpha=ALPHA)

    # Error bars
    ax.errorbar(x, means, yerr=stds, fmt='none', lw=1, capsize=5, capthick=1,
                color='black', zorder=10)

    # Threshold line (red dashed, horizontal)
    line_x = [-0.5, len(labels) - 0.5]
    ax.plot(line_x, [H_A, H_A], "--", linewidth=1.5, color='red',
            label=r'$H(A) = {:.2f}$ bits'.format(H_A), zorder=5)

    # Shade the "not verified" region (above threshold)
    y_max = max(max(means) + max(stds) * 1.5 if stds else max(means) * 1.2, H_A * 1.3)
    ax.axhspan(H_A, y_max * 1.1, alpha=NOT_VERIFIED_ALPHA, color=NOT_VERIFIED_COLOR,
               label='Condition 1 not verified')

    # Apply consistent styling
    setup_axes_style(ax)

    # Labels
    ax.set_xlabel(r'Subset pairs $\mathcal{P}_1$ and $\mathcal{P}_2$', fontsize=10)
    if show_ylabel:
        ax.set_ylabel(r'Redundancy (Bits)', fontsize=10)

    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)

    # Y-axis limits
    y_min = min(0, min(means) - max(stds) * 1.5 if stds else min(means) * 1.2)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.5, len(labels) - 0.5)

    # Title
    env_display = env_name.replace('_', ' ').title()
    ax.set_title(env_display, fontsize=12)

    # Legend
    ax.legend(loc='upper right', fontsize=8, frameon=False)


def plot_gym_environments(aggregated: dict, output_dir: Path):
    """Create combined plot for CartPole, LunarLander, Pendulum."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    envs = ['cartpole', 'lunarlander', 'pendulum']
    titles = ['CartPole', 'LunarLander', 'Pendulum']

    for idx, (env, title) in enumerate(zip(envs, titles)):
        ax = axes[idx]
        if env in aggregated:
            plot_single_env_ax(ax, title.lower(), aggregated[env], show_ylabel=(idx == 0))
        else:
            # Blank plot for missing data
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(title, fontsize=12)
            setup_axes_style(ax)
            ax.set_xlabel(r'Subset pairs $\mathcal{P}_1$ and $\mathcal{P}_2$', fontsize=10)
            if idx == 0:
                ax.set_ylabel(r'Redundancy (Bits)', fontsize=10)

    plt.tight_layout()

    output_file = output_dir / "condition1_gym_environments.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved figure: {output_file}")


def plot_skg_environments(aggregated: dict, output_dir: Path):
    """Create combined plot for SKG25 and SKG50."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    envs = ['skg25', 'skg50']
    titles = ['SKG-25', 'SKG-50']

    for idx, (env, title) in enumerate(zip(envs, titles)):
        ax = axes[idx]
        if env in aggregated:
            data = aggregated[env].copy()
            plot_single_env_ax(ax, title, data, show_ylabel=(idx == 0))
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(title, fontsize=12)
            setup_axes_style(ax)

    plt.tight_layout()

    output_file = output_dir / "condition1_skg_environments.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved figure: {output_file}")


def plot_ipd_environments(aggregated: dict, output_dir: Path):
    """Create 2-row subplot for IPD TF(N)T environments."""

    # IPD environments from TF3T to TF10T
    ipd_envs = [f'ipd_tf{n}t' for n in range(3, 11)]
    n_envs = len(ipd_envs)

    # 2 rows, 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, env in enumerate(ipd_envs):
        ax = axes[idx]
        title = f'TF{idx + 3}T'

        if env in aggregated:
            data = aggregated[env]
            plot_single_env_ax(ax, title, data, top_n=10, show_ylabel=(idx % 4 == 0))
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(title, fontsize=12)
            setup_axes_style(ax)
            ax.set_xlabel(r'Subset pairs $\mathcal{P}_1$ and $\mathcal{P}_2$', fontsize=10)
            if idx % 4 == 0:
                ax.set_ylabel(r'Redundancy (Bits)', fontsize=10)

    plt.tight_layout()

    output_file = output_dir / "condition1_ipd_environments.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved figure: {output_file}")


def plot_summary_comparison(aggregated: dict, output_dir: Path):
    """Create summary comparison plot across all environments."""

    envs = sorted(aggregated.keys())
    if not envs:
        return

    # Prepare data
    env_labels = []
    margins = []
    margin_stds = []
    colors = []

    for env in envs:
        data = aggregated[env]
        # Format environment name
        env_display = env.replace('_', ' ').replace('tf', 'TF').replace('ipd', 'IPD')
        env_display = env_display.replace('skg', 'SKG').replace('t', 'T')
        env_labels.append(env_display)
        margins.append(data['margin_mean'])
        # Approximate margin std from H_A and max_neg_R stds
        margin_stds.append(np.sqrt(data['H_A_std']**2 + data['max_neg_R_std']**2))

        if data['condition1_satisfied'] is True:
            colors.append('tab:green')
        elif data['condition1_satisfied'] is False:
            colors.append('tab:red')
        else:
            colors.append(DECOY_COLOR)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(env_labels))

    # Bar plot
    ax.bar(x, margins, align='center', color=colors, alpha=ALPHA)

    # Error bars
    ax.errorbar(x, margins, yerr=margin_stds, fmt='none', lw=1, capsize=5,
                capthick=1, color='black', zorder=10)

    # Zero threshold line (red dashed)
    line_x = [-0.5, len(env_labels) - 0.5]
    ax.plot(line_x, [0, 0], "--", linewidth=1.5, color='red',
            label=r'Violation threshold (margin $= 0$)', zorder=5)

    # Shade the "not verified" region (below zero = negative margin)
    y_min = min(min(margins) - max(margin_stds) * 1.5 if margins else 0, -0.1)
    ax.axhspan(y_min * 1.1, 0, alpha=NOT_VERIFIED_ALPHA, color=NOT_VERIFIED_COLOR,
               label='Condition 1 not verified')

    # Apply consistent styling
    setup_axes_style(ax)

    # Labels
    ax.set_xlabel('Environment', fontsize=12)
    ax.set_ylabel(r'Margin: $H(A) - \max(-R)$ (Bits)', fontsize=12)

    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(env_labels, rotation=60, ha='right', fontsize=9)

    # Y-axis limits
    y_max = max(margins) + max(margin_stds) * 1.5 if margins else 1
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.5, len(env_labels) - 0.5)

    # Legend
    ax.legend(loc='upper right', fontsize=9, frameon=False)

    # Title
    ax.set_title('Condition 1 Validation Summary', fontsize=12)

    plt.tight_layout()

    output_file = output_dir / "condition1_summary.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved summary figure: {output_file}")


def print_summary_table(aggregated: dict):
    """Print summary table of results."""
    print("\n" + "=" * 100)
    print("CONDITION 1 VALIDATION SUMMARY")
    print("=" * 100)
    print(f"{'Environment':<15} | {'TERC Vars':<6} | {'H(A)':<12} | {'Max(-R)':<12} | "
          f"{'Margin':<12} | {'Status':<15}")
    print("-" * 100)

    # Sort by environment name
    for env in sorted(aggregated.keys()):
        data = aggregated[env]

        if data['condition1_satisfied'] is True:
            status = "SATISFIED"
        elif data['condition1_satisfied'] is False:
            status = "VIOLATED"
        else:
            status = "UNDETERMINED"

        print(f"{env:<15} | {data['n_vars']:<6} | "
              f"{data['H_A_mean']:<12.4f} | {data['max_neg_R_mean']:<12.4f} | "
              f"{data['margin_mean']:<12.4f} | {status:<15}")

    print("=" * 100)

    # Overall summary
    all_satisfied = all(d['condition1_satisfied'] for d in aggregated.values()
                        if d['condition1_satisfied'] is not None)
    n_satisfied = sum(1 for d in aggregated.values() if d['condition1_satisfied'] is True)
    n_violated = sum(1 for d in aggregated.values() if d['condition1_satisfied'] is False)
    n_undetermined = sum(1 for d in aggregated.values() if d['condition1_satisfied'] is None)

    print(f"\nTotal environments: {len(aggregated)}")
    print(f"  Satisfied: {n_satisfied}")
    print(f"  Violated: {n_violated}")
    print(f"  Undetermined: {n_undetermined}")

    if n_violated == 0 and n_undetermined == 0:
        print("\n*** CONDITION 1 SATISFIED FOR ALL ENVIRONMENTS! ***")
    elif n_violated > 0:
        violated = [e for e, d in aggregated.items() if d['condition1_satisfied'] is False]
        print(f"\n!!! CONDITION 1 VIOLATED FOR: {violated} !!!")


def main():
    input_dir = Path('condition1_validation/results/individual')
    output_dir = Path('condition1_validation/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading results...")
    results = load_all_results(input_dir)
    print(f"Loaded {len(results)} result files")

    if len(results) == 0:
        print("ERROR: No result files found!")
        print(f"Expected files in: {input_dir.absolute()}")
        return 1

    # Check for missing configurations
    expected_envs = ['cartpole', 'lunarlander', 'pendulum', 'skg25', 'skg50']
    expected_envs += [f'ipd_tf{n}t' for n in range(3, 11)]
    expected_seeds = [0, 1, 2, 3, 4]

    found_configs = set((r['environment'], r['seed']) for r in results)
    expected_configs = set((env, seed) for env in expected_envs for seed in expected_seeds)
    missing = expected_configs - found_configs

    if missing:
        print(f"\nWarning: Missing {len(missing)} configurations:")
        for env, seed in sorted(missing)[:10]:
            print(f"  {env}_seed{seed}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    # Aggregate
    aggregated = aggregate_by_env(results)

    # Generate combined subplot figures
    print("\nGenerating Gym environments plot (CartPole, LunarLander, Pendulum)...")
    plot_gym_environments(aggregated, output_dir)

    print("\nGenerating SKG environments plot...")
    plot_skg_environments(aggregated, output_dir)

    print("\nGenerating IPD environments plot...")
    plot_ipd_environments(aggregated, output_dir)

    # Generate summary plot
    print("\nGenerating summary plot...")
    plot_summary_comparison(aggregated, output_dir)

    # Print summary table
    print_summary_table(aggregated)

    # Save aggregated JSON
    output_json = Path('condition1_validation/results/aggregated_results.json')
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Convert for JSON serialization
    aggregated_json = {}
    for k, v in aggregated.items():
        aggregated_json[k] = {
            key: (val if not isinstance(val, np.floating) else float(val))
            for key, val in v.items()
        }

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'n_results': len(results),
        'n_environments': len(aggregated),
        'aggregated': aggregated_json,
    }

    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2, default=float)
    print(f"\nAggregated results saved to {output_json}")

    return 0


if __name__ == '__main__':
    exit(main())
