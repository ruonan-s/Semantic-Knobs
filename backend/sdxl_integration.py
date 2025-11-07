# backend/sdxl_integration.py
# Convert concept mixtures to SDXL-ready phrases with gain mapping.

from typing import List, Dict, Tuple
import numpy as np


def normalize_simplex(w: np.ndarray) -> np.ndarray:
    """
    Normalize weights to sum to 1 (simplex constraint).

    Args:
        w: Weight vector (K,)

    Returns:
        Normalized weights that sum to 1
    """
    w = np.maximum(w, 0.0)  # Ensure non-negative
    total = w.sum()
    if total < 1e-10:
        # If all zeros, return uniform
        return np.ones_like(w) / len(w)
    return w / total


def compute_gains(w: np.ndarray, lambda_scale: float = 0.4) -> np.ndarray:
    """
    Compute gains from normalized weights using z-score mapping.

    Formula:
        z_scores = (w - mean_w) / (std_w + eps)
        gains = 1.0 + lambda_scale * z_scores
        gains = clip(gains, 0.7, 1.5)

    Args:
        w: Normalized weight vector (K,)
        lambda_scale: Scaling factor for z-scores (default: 0.4)

    Returns:
        Gains in range [0.7, 1.5]
    """
    mean_w = np.mean(w)
    std_w = np.std(w)

    # Compute z-scores
    z_scores = (w - mean_w) / (std_w + 1e-8)

    # Map to gain range [0.7, 1.5]
    gains = 1.0 + lambda_scale * z_scores
    gains = np.clip(gains, 0.7, 1.5)

    return gains


def concepts_to_sdxl_phrases(
    w: np.ndarray,
    concepts: List[Dict],
    top_k: int = 10,
    num_negatives: int = 3,
    lambda_scale: float = 0.4
) -> Tuple[List[Tuple[str, float]], List[str]]:
    """
    Convert mixture weights to SDXL phrases with gains.

    Algorithm:
        1. Normalize weights to simplex (sum to 1)
        2. Compute gains via z-score mapping: gains = 1.0 + lambda * z_scores
        3. Select Top-K positives (highest weights)
        4. Select deficit negatives (lowest weights below uniform/2, excluding top-K)

    Args:
        w: Weight vector (K,)
        concepts: List of concept dicts with 'label' field
        top_k: Number of positive phrases to return (default: 10)
        num_negatives: Number of negative phrases to return (default: 3)
        lambda_scale: Scaling factor for gain mapping (default: 0.4)

    Returns:
        (positive_phrases, negative_phrases)
        positive_phrases: [(phrase, gain), ...] where gain ∈ [0.7, 1.5]
        negative_phrases: [phrase, phrase, ...]

    Example:
        >>> concepts = [
        ...     {'id': 'c0', 'label': 'cozy'},
        ...     {'id': 'c1', 'label': 'modern'},
        ...     {'id': 'c2', 'label': 'minimalist'},
        ... ]
        >>> w = np.array([0.5, 0.3, 0.2])
        >>> pos, neg = concepts_to_sdxl_phrases(w, concepts, top_k=2, num_negatives=1)
        >>> # pos = [('cozy', 1.4), ('modern', 1.0)] (approximate)
        >>> # neg = ['minimalist']
    """
    K = len(concepts)

    if len(w) != K:
        raise ValueError(f"Weight vector length ({len(w)}) must match number of concepts ({K})")

    if K == 0:
        return [], []

    # Step 1: Normalize weights
    w_norm = normalize_simplex(w)

    # Step 2: Compute gains
    gains = compute_gains(w_norm, lambda_scale=lambda_scale)

    # Step 3: Select Top-K positives
    # Sort by weight descending
    sorted_indices = np.argsort(w_norm)[::-1]

    # Take top K (clamped to available concepts)
    actual_top_k = min(top_k, K)
    top_indices = sorted_indices[:actual_top_k]

    positive_phrases = []
    for idx in top_indices:
        phrase = concepts[idx]['label']
        gain = float(gains[idx])
        positive_phrases.append((phrase, gain))

    # Step 4: Select deficit negatives
    # Deficit = (1/K - w) for concepts below uniform/2
    # IMPORTANT: Exclude concepts already in positive prompts to avoid conflicts
    uniform_weight = 1.0 / K
    deficit_threshold = uniform_weight / 2.0

    # Find concepts with weight below threshold, excluding top-K positives
    deficit_indices = []
    deficits = []

    for idx in range(K):
        if idx not in top_indices and w_norm[idx] < deficit_threshold:
            deficit = uniform_weight - w_norm[idx]
            deficit_indices.append(idx)
            deficits.append(deficit)

    # Sort by deficit descending and take top num_negatives
    if len(deficit_indices) > 0:
        deficit_order = np.argsort(deficits)[::-1]
        actual_num_neg = min(num_negatives, len(deficit_indices))

        negative_phrases = []
        for i in range(actual_num_neg):
            idx = deficit_indices[deficit_order[i]]
            phrase = concepts[idx]['label']
            negative_phrases.append(phrase)
    else:
        negative_phrases = []

    return positive_phrases, negative_phrases


def get_phrase_summary(
    positive_phrases: List[Tuple[str, float]],
    negative_phrases: List[str]
) -> str:
    """
    Create a human-readable summary of phrases and gains.

    Args:
        positive_phrases: List of (phrase, gain) tuples
        negative_phrases: List of negative phrases

    Returns:
        Formatted string summary
    """
    lines = ["Positive phrases:"]
    for phrase, gain in positive_phrases:
        lines.append(f"  {phrase}: gain={gain:.3f}")

    if negative_phrases:
        lines.append("\nNegative phrases:")
        for phrase in negative_phrases:
            lines.append(f"  {phrase}")
    else:
        lines.append("\nNegative phrases: (none)")

    return "\n".join(lines)
