#!/usr/bin/env python3
"""Generate jobs_condition1.txt parameter file for SGE job array."""

import os
from pathlib import Path


def main():
    # Base environments
    base_envs = ['cartpole', 'lunarlander', 'pendulum', 'skg25', 'skg50']

    # IPD environments with TF(N)T for N from 3 to 10
    ipd_envs = [f'ipd_tf{n}t' for n in range(3, 11)]

    all_envs = base_envs + ipd_envs
    seeds = [0, 1, 2, 3, 4]

    # Get the scripts directory
    script_dir = Path(__file__).parent
    output_file = script_dir / 'jobs_condition1.txt'

    with open(output_file, 'w') as f:
        for env in all_envs:
            for seed in seeds:
                f.write(f"{env} {seed}\n")

    total_jobs = len(all_envs) * len(seeds)
    print(f"Generated {total_jobs} job configurations to {output_file}")
    print(f"\nJob breakdown:")
    print(f"  Base environments: {len(base_envs)} ({', '.join(base_envs)})")
    print(f"  IPD environments: {len(ipd_envs)} ({', '.join(ipd_envs)})")
    print(f"  Seeds: {len(seeds)} ({', '.join(map(str, seeds))})")
    print(f"  Total: {len(all_envs)} envs x {len(seeds)} seeds = {total_jobs} jobs")


if __name__ == '__main__':
    main()
