#!/usr/bin/env python3
"""
Mutual Information estimators for different variable types.

Implements:
- Discrete MI: Plug-in estimator using empirical distributions
- KSG estimator: For continuous variables (Kraskov et al., 2004)
- Mixed MI: For continuous X and discrete Y (Ross, 2014 style)
"""

import numpy as np
from scipy.special import digamma, gamma as gamma_func
from scipy.spatial import KDTree
from typing import Union, Tuple
import warnings


def compute_entropy_discrete(data: np.ndarray) -> float:
    """
    Compute entropy for discrete data using plug-in estimator.

    Args:
        data: Array of shape (n_samples,) or (n_samples, n_features)

    Returns:
        Entropy in bits (log base 2)
    """
    if data.ndim > 1:
        # Convert rows to tuple strings for joint entropy
        data = np.array(['_'.join(map(str, row)) for row in data])

    _, counts = np.unique(data, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))


def compute_mi_discrete(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute mutual information for discrete variables.
    I(X;Y) = H(X) + H(Y) - H(X,Y)

    Args:
        x: Array of shape (n_samples,) or (n_samples, n_features)
        y: Array of shape (n_samples,) or (n_samples, n_features)

    Returns:
        Mutual information in bits
    """
    h_x = compute_entropy_discrete(x)
    h_y = compute_entropy_discrete(y)

    # Joint entropy
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    xy = np.hstack([x, y])
    h_xy = compute_entropy_discrete(xy)

    return max(0, h_x + h_y - h_xy)  # Ensure non-negative


def _add_noise(data: np.ndarray, noise_level: float = 1e-10) -> np.ndarray:
    """Add small noise to avoid issues with identical points."""
    return data + np.random.randn(*data.shape) * noise_level


def compute_mi_ksg(x: np.ndarray, y: np.ndarray, k: int = 3, noise: float = 1e-10) -> float:
    """
    Compute mutual information using KSG estimator (Kraskov et al., 2004).
    Implementation of Algorithm 1 (using maximum norm).

    Args:
        x: Array of shape (n_samples,) or (n_samples, n_features_x)
        y: Array of shape (n_samples,) or (n_samples, n_features_y)
        k: Number of nearest neighbors
        noise: Small noise to add to avoid identical points

    Returns:
        Mutual information estimate in nats (natural log base e)
        Multiply by log2(e) ≈ 1.4427 to convert to bits
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Add small noise to avoid issues with identical points
    x = _add_noise(x.astype(np.float64), noise)
    y = _add_noise(y.astype(np.float64), noise)

    n = len(x)
    xy = np.hstack([x, y])

    # Build KD trees using Chebyshev (maximum) norm
    tree_xy = KDTree(xy)
    tree_x = KDTree(x)
    tree_y = KDTree(y)

    # Find k-th nearest neighbor distances in joint space
    # Query k+1 because the point itself is included
    dists_xy, _ = tree_xy.query(xy, k=k+1, p=np.inf)  # Chebyshev norm
    eps = dists_xy[:, -1]  # Distance to k-th neighbor

    # Count neighbors within eps in marginal spaces (strictly less than eps)
    n_x = np.zeros(n)
    n_y = np.zeros(n)

    for i in range(n):
        # Count points with distance < eps[i] (excluding self)
        n_x[i] = len(tree_x.query_ball_point(x[i], eps[i] - 1e-15, p=np.inf)) - 1
        n_y[i] = len(tree_y.query_ball_point(y[i], eps[i] - 1e-15, p=np.inf)) - 1

    # Ensure at least 1 neighbor to avoid log(0)
    n_x = np.maximum(n_x, 1)
    n_y = np.maximum(n_y, 1)

    # KSG estimator (Algorithm 1)
    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)

    return max(0, mi)  # Ensure non-negative


def compute_mi_mixed(x_continuous: np.ndarray, y_discrete: np.ndarray, k: int = 5) -> float:
    """
    Compute MI between continuous X and discrete Y.
    Uses conditional entropy approach: I(X;Y) = H(Y) - H(Y|X)

    Estimates H(Y|X) using k-NN local class frequency estimation.

    Args:
        x_continuous: Continuous array of shape (n_samples,) or (n_samples, n_features)
        y_discrete: Discrete array of shape (n_samples,)
        k: Number of nearest neighbors for local estimation

    Returns:
        Mutual information estimate in bits
    """
    if x_continuous.ndim == 1:
        x_continuous = x_continuous.reshape(-1, 1)

    # Add small noise to continuous variables
    x_continuous = _add_noise(x_continuous.astype(np.float64), 1e-10)

    n = len(x_continuous)

    # H(Y) - entropy of discrete variable
    h_y = compute_entropy_discrete(y_discrete)

    # Build KD tree for continuous space
    tree_x = KDTree(x_continuous)

    # Estimate H(Y|X) using local class frequencies
    h_y_given_x = 0
    effective_k = min(k, n - 1)

    for i in range(n):
        # Find k nearest neighbors (excluding self)
        _, indices = tree_x.query(x_continuous[i], k=effective_k + 1)
        neighbor_indices = indices[1:effective_k + 1]

        # Get class distribution among neighbors
        neighbor_classes = y_discrete[neighbor_indices]
        _, counts = np.unique(neighbor_classes, return_counts=True)
        probs = counts / counts.sum()
        local_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        h_y_given_x += local_entropy

    h_y_given_x /= n

    mi = h_y - h_y_given_x
    return max(0, mi)  # Ensure non-negative


def compute_differential_entropy_knn(x: np.ndarray, k: int = 3) -> float:
    """
    Estimate differential entropy using k-NN method (Kozachenko-Leonenko).

    Args:
        x: Continuous array of shape (n_samples,) or (n_samples, n_features)
        k: Number of nearest neighbors

    Returns:
        Differential entropy estimate in nats
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    x = _add_noise(x.astype(np.float64), 1e-10)
    n, d = x.shape

    tree = KDTree(x)
    dists, _ = tree.query(x, k=k+1)
    rho = dists[:, -1]  # Distance to k-th neighbor

    # Avoid log(0)
    rho = np.maximum(rho, 1e-10)

    # Volume of unit ball in d dimensions
    log_v_d = (d / 2) * np.log(np.pi) - np.log(gamma_func(d / 2 + 1))

    # Kozachenko-Leonenko estimator
    h = d * np.mean(np.log(rho)) + np.log(n - 1) - digamma(k) + log_v_d

    return h


def compute_mi(x: np.ndarray, y: np.ndarray, x_type: str, y_type: str, k: int = 5) -> float:
    """
    Dispatch to appropriate MI estimator based on variable types.

    Args:
        x: First variable
        y: Second variable
        x_type: 'discrete' or 'continuous'
        y_type: 'discrete' or 'continuous'
        k: Number of neighbors for KSG/mixed estimators

    Returns:
        Mutual information estimate in bits
    """
    if x_type == 'discrete' and y_type == 'discrete':
        return compute_mi_discrete(x, y)
    elif x_type == 'continuous' and y_type == 'continuous':
        # KSG returns nats, convert to bits
        return compute_mi_ksg(x, y, k=k) * np.log2(np.e)
    elif x_type == 'continuous' and y_type == 'discrete':
        return compute_mi_mixed(x, y, k=k)
    else:  # x discrete, y continuous
        return compute_mi_mixed(y, x, k=k)


def compute_redundancy(
    states: np.ndarray,
    actions: np.ndarray,
    s1_indices: Tuple[int, ...],
    s2_indices: Tuple[int, ...],
    state_type: str,
    action_type: str,
    k: int = 5
) -> float:
    """
    Compute redundancy: R = I(A; S1 ∪ S2) - I(A; S1) - I(A; S2)
    Return -R (negated, so positive = redundancy)

    Args:
        states: State array of shape (n_samples, n_features)
        actions: Action array of shape (n_samples,) or (n_samples, action_dim)
        s1_indices: Tuple of indices for subset S1
        s2_indices: Tuple of indices for subset S2
        state_type: 'discrete' or 'continuous'
        action_type: 'discrete' or 'continuous'
        k: Number of neighbors for MI estimation

    Returns:
        -R value (positive = redundancy)
    """
    s1 = states[:, list(s1_indices)]
    s2 = states[:, list(s2_indices)]
    s_union = states[:, list(s1_indices) + list(s2_indices)]

    mi_union = compute_mi(s_union, actions, state_type, action_type, k=k)
    mi_s1 = compute_mi(s1, actions, state_type, action_type, k=k)
    mi_s2 = compute_mi(s2, actions, state_type, action_type, k=k)

    R = mi_union - mi_s1 - mi_s2
    return -R  # Negate so positive = redundancy


if __name__ == '__main__':
    # Test the estimators
    np.random.seed(42)

    print("Testing MI estimators...")

    # Test discrete MI
    print("\n1. Discrete MI (independent variables):")
    x_disc = np.random.randint(0, 3, size=1000)
    y_disc = np.random.randint(0, 3, size=1000)
    mi_disc = compute_mi_discrete(x_disc, y_disc)
    print(f"   MI(X,Y) for independent: {mi_disc:.4f} (should be ~0)")

    # Test discrete MI (dependent)
    print("\n2. Discrete MI (Y = X):")
    y_disc_dep = x_disc.copy()
    mi_disc_dep = compute_mi_discrete(x_disc, y_disc_dep)
    h_x = compute_entropy_discrete(x_disc)
    print(f"   MI(X,X) = {mi_disc_dep:.4f}, H(X) = {h_x:.4f} (should be equal)")

    # Test KSG
    print("\n3. KSG MI (independent Gaussians):")
    x_cont = np.random.randn(1000, 2)
    y_cont = np.random.randn(1000, 2)
    mi_ksg = compute_mi_ksg(x_cont, y_cont) * np.log2(np.e)
    print(f"   MI(X,Y) for independent: {mi_ksg:.4f} bits (should be ~0)")

    # Test KSG (correlated)
    print("\n4. KSG MI (correlated Gaussians, rho=0.8):")
    rho = 0.8
    cov = [[1, rho], [rho, 1]]
    xy_corr = np.random.multivariate_normal([0, 0], cov, 1000)
    mi_ksg_corr = compute_mi_ksg(xy_corr[:, 0:1], xy_corr[:, 1:2]) * np.log2(np.e)
    mi_theoretical = -0.5 * np.log2(1 - rho**2)
    print(f"   MI estimate: {mi_ksg_corr:.4f} bits, theoretical: {mi_theoretical:.4f} bits")

    # Test mixed MI
    print("\n5. Mixed MI (continuous X, discrete Y):")
    # Y is determined by sign of X
    x_mix = np.random.randn(1000, 1)
    y_mix = (x_mix[:, 0] > 0).astype(int)
    mi_mixed = compute_mi_mixed(x_mix, y_mix)
    print(f"   MI(X, sign(X)): {mi_mixed:.4f} bits (should be close to H(Y)={compute_entropy_discrete(y_mix):.4f})")

    print("\nAll tests completed!")
