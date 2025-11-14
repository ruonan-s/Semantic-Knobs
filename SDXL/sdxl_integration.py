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

