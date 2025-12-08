#!/usr/bin/env python3
"""Generate jobs_inference.txt parameter file for SGE job array."""

import os
from pathlib import Path

def main():
    envs = ['cartpole', 'lunarlander', 'pendulum', 'skg25', 'skg50']
    state_types = ['full', 'terc']
    seeds = [0, 1, 2, 3, 4]

    # Get the scripts directory
    script_dir = Path(__file__).parent
    output_file = script_dir / 'jobs_inference.txt'

    with open(output_file, 'w') as f:
        for env in envs:
            for state_type in state_types:
                for seed in seeds:
                    f.write(f"{env} {state_type} {seed}\n")

    total_jobs = len(envs) * len(state_types) * len(seeds)
    print(f"Generated {total_jobs} job configurations to {output_file}")
    print(f"\nJob breakdown:")
    print(f"  Environments: {len(envs)} ({', '.join(envs)})")
    print(f"  State types: {len(state_types)} ({', '.join(state_types)})")
    print(f"  Seeds: {len(seeds)} ({', '.join(map(str, seeds))})")


if __name__ == '__main__':
    main()
