#!/usr/bin/env python3
"""
Single experiment run: train one model and measure inference time.

This script handles a single experiment configuration (env, state_type, seed)
as part of a parallelized job array execution.

Usage:
    python run_single.py --env cartpole --state_type full --seed 0 --device cuda
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.actor_critic import Actor
from models.ppo import PPOPolicy
from training.train_actor_critic import train_actor_critic
from training.train_ppo import train_ppo


def check_gpu_health() -> bool:
    """
    Test GPU health by performing a simple operation.
    Returns True if GPU is healthy, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Try to allocate memory and perform a simple operation
        device = torch.device('cuda')
        test_tensor = torch.randn(100, 100, device=device)
        result = torch.matmul(test_tensor, test_tensor)
        torch.cuda.synchronize()  # Force completion
        del test_tensor, result
        torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        print(f"GPU health check failed: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description='Single TERC inference experiment')
    parser.add_argument('--env', type=str, required=True,
                        choices=['cartpole', 'lunarlander', 'pendulum', 'skg25', 'skg50'],
                        help='Environment name')
    parser.add_argument('--state_type', type=str, required=True,
                        choices=['full', 'terc'],
                        help='State representation type')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'],
                        help='Device for training')
    parser.add_argument('--output_dir', type=str, default='results/individual',
                        help='Directory for output JSON')
    parser.add_argument('--model_dir', type=str, default='models_trained',
                        help='Directory to save/load trained models')
    parser.add_argument('--n_warmup', type=int, default=1000,
                        help='Number of warmup inference calls')
    parser.add_argument('--n_measure', type=int, default=10000,
                        help='Number of measured inference calls')
    parser.add_argument('--force_retrain', action='store_true',
                        help='Force retraining even if model exists')
    return parser.parse_args()


def get_state_dim(env_name: str, state_type: str) -> int:
    """Return state dimension for given environment and state type."""
    dims = {
        'cartpole': {'full': 7, 'terc': 4},
        'lunarlander': {'full': 11, 'terc': 8},
        'pendulum': {'full': 6, 'terc': 3},
        'skg25': {'full': 25, 'terc': 3},
        'skg50': {'full': 50, 'terc': 3},
    }
    return dims[env_name][state_type]


def get_action_dim(env_name: str) -> int:
    """Return action dimension for given environment."""
    dims = {
        'cartpole': 2,      # Discrete: left/right
        'lunarlander': 4,   # Discrete: noop, left, main, right
        'pendulum': 1,      # Continuous: torque
        'skg25': 81,        # Discrete: [-40, 40]
        'skg50': 81,        # Discrete: [-40, 40]
    }
    return dims[env_name]


def measure_inference_time(model: torch.nn.Module, input_dim: int, batch_size: int,
                           device: torch.device, n_warmup: int = 1000,
                           n_measure: int = 10000) -> dict:
    """
    Measure inference time for a model.

    Args:
        model: PyTorch model (Actor or PPOPolicy)
        input_dim: Input dimension
        batch_size: Batch size for inference
        device: Device to run on
        n_warmup: Number of warmup iterations
        n_measure: Number of measurement iterations

    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    model.to(device)

    dummy_input = torch.randn(batch_size, input_dim, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

    # Synchronize if GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_measure):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    times = np.array(times)
    return {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'median_ms': float(np.median(times) * 1000),
        'min_ms': float(np.min(times) * 1000),
        'max_ms': float(np.max(times) * 1000),
        'p5_ms': float(np.percentile(times, 5) * 1000),
        'p95_ms': float(np.percentile(times, 95) * 1000),
    }


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(env_name: str, state_dim: int, action_dim: int) -> torch.nn.Module:
    """Create the appropriate model architecture for the environment."""
    if env_name == 'pendulum':
        return PPOPolicy(state_dim, action_dim, action_scale=2.0)
    else:
        return Actor(state_dim, action_dim)


def train_model(env_name: str, state_type: str, seed: int,
                device: torch.device, verbose: bool = True) -> tuple:
    """
    Train the appropriate model for the given environment.

    Returns:
        model: Trained model (Actor or PPOPolicy)
        training_info: Dictionary with training statistics
    """
    if env_name == 'pendulum':
        policy, info = train_ppo(
            state_type=state_type,
            seed=seed,
            device=device,
            verbose=verbose
        )
        return policy, info
    else:
        actor, info = train_actor_critic(
            env_name=env_name,
            state_type=state_type,
            seed=seed,
            device=device,
            verbose=verbose
        )
        return actor, info


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device with health check
    device = torch.device('cpu')
    if args.device == 'cuda' and torch.cuda.is_available():
        print(f"Checking GPU health: {torch.cuda.get_device_name(0)}...")
        if check_gpu_health():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: GPU health check failed (possible ECC error). Falling back to CPU.")
            print("Consider reporting this node to cluster admins.")
    else:
        print("Using CPU")

    # Get dimensions
    state_dim = get_state_dim(args.env, args.state_type)
    action_dim = get_action_dim(args.env)

    print(f"\n{'='*60}")
    print(f"TERC Inference Experiment")
    print(f"{'='*60}")
    print(f"Environment: {args.env}")
    print(f"State type: {args.state_type}")
    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_dir) / f"{args.env}_{args.state_type}_seed{args.seed}.pt"

    # Training or loading model
    training_info = {}
    if model_path.exists() and not args.force_retrain:
        print(f"Loading existing model from {model_path}")
        model = create_model(args.env, state_dim, action_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        training_info = {'loaded_from_checkpoint': True}
    else:
        print("Training new model...")
        training_start = time.time()

        model, training_info = train_model(
            env_name=args.env,
            state_type=args.state_type,
            seed=args.seed,
            device=device,
            verbose=True
        )

        training_time = time.time() - training_start
        training_info['training_time_seconds'] = training_time
        training_info['loaded_from_checkpoint'] = False

        # Save model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    n_params = count_parameters(model)
    print(f"\nModel parameters: {n_params}")

    # Measure inference time
    print(f"\n{'='*60}")
    print("Measuring Inference Times")
    print(f"{'='*60}")

    results = {
        'environment': args.env,
        'state_type': args.state_type,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'seed': args.seed,
        'n_parameters': n_params,
        'n_warmup': args.n_warmup,
        'n_measure': args.n_measure,
        'training_info': {
            k: v for k, v in training_info.items()
            if k != 'training_rewards'  # Don't save full reward history to JSON
        },
        'inference_times': {},
        'timestamp': datetime.now().isoformat(),
    }

    batch_sizes = [1, 64]
    devices_to_test = ['cpu']
    if torch.cuda.is_available():
        devices_to_test.append('cuda')

    for test_device_name in devices_to_test:
        test_device = torch.device(test_device_name)
        for batch_size in batch_sizes:
            key = f"{test_device_name}_batch{batch_size}"
            print(f"\nMeasuring inference: {key}...")
            print(f"  Warmup iterations: {args.n_warmup}")
            print(f"  Measurement iterations: {args.n_measure}")

            timing = measure_inference_time(
                model=model,
                input_dim=state_dim,
                batch_size=batch_size,
                device=test_device,
                n_warmup=args.n_warmup,
                n_measure=args.n_measure
            )
            results['inference_times'][key] = timing
            print(f"  Mean: {timing['mean_ms']:.4f} ms")
            print(f"  Std:  {timing['std_ms']:.4f} ms")
            print(f"  Median: {timing['median_ms']:.4f} ms")
            print(f"  Range: [{timing['min_ms']:.4f}, {timing['max_ms']:.4f}] ms")

    # Save results
    output_file = Path(args.output_dir) / f"{args.env}_{args.state_type}_seed{args.seed}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")

    # Print summary
    print(f"\nSUMMARY:")
    print(f"  Environment: {args.env}")
    print(f"  State type: {args.state_type} (dim={state_dim})")
    print(f"  Parameters: {n_params}")
    if 'converged' in training_info:
        print(f"  Training converged: {training_info.get('converged', 'N/A')}")
        print(f"  Episodes trained: {training_info.get('episodes_trained', 'N/A')}")
        print(f"  Final avg reward: {training_info.get('final_avg_reward', 'N/A'):.2f}")

    print(f"\n  Inference times (CPU, batch=1): {results['inference_times'].get('cpu_batch1', {}).get('mean_ms', 'N/A'):.4f} ms")
    if 'cuda_batch1' in results['inference_times']:
        print(f"  Inference times (CUDA, batch=1): {results['inference_times']['cuda_batch1']['mean_ms']:.4f} ms")


if __name__ == '__main__':
    main()
