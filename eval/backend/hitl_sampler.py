"""
HITL Sampler - UCB-Based Multi-Point Composition Sampler

Implements Upper Confidence Bound (UCB) acquisition for selecting 10 diverse
points from the GP utility surface to compose images for ordinal ranking.

Supports multi-modal preferences by finding multiple local maxima with
diversity constraints.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from exploration_GP import PreferenceLearner


@dataclass
class CompositionSample:
    """
    A single image composition with 10 sampled feature points.
    
    Each composition represents a set of aesthetic concepts to be combined
    in a single generated image via weighted attention.
    """
    points: np.ndarray       # (10, 768) - 10 CLIP embeddings
    weights: np.ndarray      # (10,) - attention weights per point
    tag_labels: List[str]    # Labels for each point (for debugging/display)
    tag_indices: List[int]   # Indices into original tag list
    point_ucb_scores: np.ndarray  # UCB score for each point (for analysis)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class HITLSampler:
    """
    UCB-based sampler for 10-point feature compositions.
    
    Instead of sampling from a single Gaussian, queries the GP's utility surface
    to find MULTIPLE local maxima - supporting multi-modal preferences
    (e.g., user likes BOTH "dark/cozy" AND "bright/minimalist").
    
    The sampler uses Upper Confidence Bound (UCB) acquisition:
        UCB(x) = mu(x) + beta * sigma(x)
    
    This balances exploitation (high utility) with exploration (high uncertainty).
    """
    
    def __init__(
        self, 
        preference_gp: PreferenceLearner,
        all_tag_embeddings: np.ndarray,  # (N, 768) all available tag embeddings
        all_tag_labels: List[str],
        beta: float = 2.0,               # UCB exploration coefficient
        diversity_threshold: float = 0.85,  # Cosine similarity threshold for diversity
        weight_temperature: float = 1.0     # Temperature for attention weight softmax
    ):
        """
        Initialize the HITL sampler.
        
        Args:
            preference_gp: PreferenceLearner GP from exploration
            all_tag_embeddings: (N, 768) array of all candidate tag embeddings
            all_tag_labels: List of tag text labels corresponding to embeddings
            beta: UCB exploration coefficient (higher = more exploration)
            diversity_threshold: Max cosine similarity between selected points
            weight_temperature: Temperature for softmax when computing attention weights
        """
        self.gp = preference_gp
        self.embeddings = np.array(all_tag_embeddings)
        self.labels = list(all_tag_labels)
        self.beta = beta
        self.diversity_threshold = diversity_threshold
        self.weight_temperature = weight_temperature
        
        # Precompute normalized embeddings for faster similarity computation
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.normalized_embeddings = self.embeddings / np.maximum(norms, 1e-8)
        
        # Cache for UCB scores (recomputed after GP update)
        self._ucb_cache: Optional[np.ndarray] = None
        self._mu_cache: Optional[np.ndarray] = None
        self._sigma_cache: Optional[np.ndarray] = None
    
    def invalidate_cache(self):
        """Invalidate UCB cache after GP update."""
        self._ucb_cache = None
        self._mu_cache = None
        self._sigma_cache = None
    
    def _compute_ucb_scores(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute UCB scores for all candidate embeddings.
        
        Returns:
            (ucb_scores, mu, sigma) - arrays of shape (N,)
        """
        if self._ucb_cache is not None:
            return self._ucb_cache, self._mu_cache, self._sigma_cache
        
        # Get GP predictions
        mu, sigma = self.gp.predict_utility(self.embeddings)
        
        # UCB acquisition function
        ucb_scores = mu + self.beta * sigma
        
        # Cache results
        self._ucb_cache = ucb_scores
        self._mu_cache = mu
        self._sigma_cache = sigma
        
        return ucb_scores, mu, sigma
    
    def _select_diverse_maxima(
        self, 
        scores: np.ndarray, 
        k: int,
        excluded_indices: Optional[set] = None
    ) -> List[int]:
        """
        Greedy selection: pick highest UCB, then exclude similar points.
        
        This ensures selected points are semantically diverse, supporting
        multi-modal preferences where user may like unrelated aesthetic concepts.
        
        Args:
            scores: UCB scores for all candidates
            k: Number of points to select
            excluded_indices: Indices to exclude from selection
            
        Returns:
            List of selected indices
        """
        selected = []
        available = set(range(len(scores)))
        
        if excluded_indices:
            available -= excluded_indices
        
        for _ in range(k):
            if not available:
                break
            
            # Pick highest scoring available point
            best_idx = max(available, key=lambda i: scores[i])
            selected.append(best_idx)
            available.remove(best_idx)
            
            # Remove points too similar to selected (cosine > threshold)
            to_remove = []
            best_emb_norm = self.normalized_embeddings[best_idx]
            
            for idx in available:
                # Compute cosine similarity using precomputed normalized embeddings
                sim = np.dot(best_emb_norm, self.normalized_embeddings[idx])
                if sim > self.diversity_threshold:
                    to_remove.append(idx)
            
            available -= set(to_remove)
        
        return selected
    
    def _compute_attention_weights(self, ucb_scores: np.ndarray) -> np.ndarray:
        """
        Compute attention weights from UCB scores using softmax.
        
        Higher UCB scores get higher attention weights, scaled by temperature.
        
        Args:
            ucb_scores: UCB scores for selected points
            
        Returns:
            Softmax-normalized attention weights
        """
        # Shift for numerical stability
        scores_shifted = ucb_scores - np.max(ucb_scores)
        exp_scores = np.exp(scores_shifted / self.weight_temperature)
        weights = exp_scores / (np.sum(exp_scores) + 1e-8)
        
        return weights
    
    def sample_composition(
        self, 
        n_points: int = 10,
        excluded_indices: Optional[set] = None
    ) -> CompositionSample:
        """
        Select n_points using UCB acquisition with diversity constraint.
        
        Finds diverse local maxima to support multi-modal preferences
        (e.g., user likes BOTH "dark/cozy" AND "bright/minimalist").
        
        Args:
            n_points: Number of points to select (default 10)
            excluded_indices: Indices to exclude from this sample
            
        Returns:
            CompositionSample with selected points, weights, and metadata
        """
        # Get UCB scores
        ucb_scores, mu, sigma = self._compute_ucb_scores()
        
        # Select diverse maxima
        selected_indices = self._select_diverse_maxima(
            ucb_scores, n_points, excluded_indices
        )
        
        # If we couldn't select enough due to diversity constraints, relax and try again
        if len(selected_indices) < n_points:
            # Fall back to top-K without diversity
            remaining = n_points - len(selected_indices)
            available = set(range(len(ucb_scores))) - set(selected_indices)
            if excluded_indices:
                available -= excluded_indices
            
            sorted_available = sorted(available, key=lambda i: ucb_scores[i], reverse=True)
            selected_indices.extend(sorted_available[:remaining])
        
        # Extract selected data
        selected_indices = selected_indices[:n_points]  # Ensure exactly n_points
        
        points = self.embeddings[selected_indices]
        labels = [self.labels[i] for i in selected_indices]
        point_ucb = ucb_scores[selected_indices]
        
        # Compute attention weights
        weights = self._compute_attention_weights(point_ucb)
        
        return CompositionSample(
            points=points,
            weights=weights,
            tag_labels=labels,
            tag_indices=selected_indices,
            point_ucb_scores=point_ucb
        )
    
    def sample_batch(
        self, 
        batch_size: int = 4, 
        n_points: int = 10,
        ensure_diversity_across_batch: bool = True
    ) -> List[CompositionSample]:
        """
        Sample multiple compositions for a ranking round.
        
        Args:
            batch_size: Number of compositions to generate (default 4)
            n_points: Points per composition (default 10)
            ensure_diversity_across_batch: If True, ensure some diversity
                across compositions by excluding recently selected points
                
        Returns:
            List of CompositionSample objects
        """
        compositions = []
        excluded = set()
        
        for i in range(batch_size):
            sample = self.sample_composition(
                n_points=n_points,
                excluded_indices=excluded if ensure_diversity_across_batch else None
            )
            compositions.append(sample)
            
            # Add top-3 points from this sample to exclusion set
            # This ensures some variation across the batch
            if ensure_diversity_across_batch:
                top_3_idx = np.argsort(sample.point_ucb_scores)[-3:]
                for idx in top_3_idx:
                    excluded.add(sample.tag_indices[idx])
        
        return compositions
    
    def get_utility_statistics(self) -> Dict:
        """
        Get statistics about the current GP utility surface.
        
        Useful for monitoring convergence and understanding preference distribution.
        """
        ucb_scores, mu, sigma = self._compute_ucb_scores()
        
        return {
            "mean_utility": float(np.mean(mu)),
            "std_utility": float(np.std(mu)),
            "max_utility": float(np.max(mu)),
            "min_utility": float(np.min(mu)),
            "mean_uncertainty": float(np.mean(sigma)),
            "max_uncertainty": float(np.max(sigma)),
            "mean_ucb": float(np.mean(ucb_scores)),
            "max_ucb": float(np.max(ucb_scores)),
            "n_candidates": len(mu),
        }
    
    def get_top_k_by_utility(self, k: int = 10) -> List[Tuple[str, float, float]]:
        """
        Get top-K tags by GP utility (not UCB).
        
        Returns:
            List of (tag_label, utility, uncertainty) tuples
        """
        _, mu, sigma = self._compute_ucb_scores()
        
        top_indices = np.argsort(mu)[-k:][::-1]
        
        return [
            (self.labels[i], float(mu[i]), float(sigma[i]))
            for i in top_indices
        ]
