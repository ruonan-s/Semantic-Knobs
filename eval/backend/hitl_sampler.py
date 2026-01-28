"""
HITL Sampler - Continuous Aesthetic Discovery in Reduced CLIP Space

Operates in a PCA-reduced space (e.g., 32 dims) for efficient GP learning,
then reconstructs to 768-dim for SDXL injection.

Sampling Strategies:
1. Centroid Sampling (Exploitation): Points near the current preference mean μ
2. SLERP Interpolation: Semantic bridges between high-utility concepts
3. UCB Perturbations (Exploration): Noise in high-uncertainty directions
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from dim_reducer import EmbeddingReducer, slerp_reduced


@dataclass
class ContinuousComposition:
    """
    A composition of 10 points sampled from continuous embedding space.
    
    Points are stored in FULL 768-dim space (after reconstruction from
    reduced space) for direct injection into SDXL.
    """
    points: np.ndarray           # (10, 768) - 10 CLIP embedding vectors
    points_reduced: np.ndarray   # (10, reduced_dim) - reduced space points
    weights: np.ndarray          # (10,) - attention weights per point
    sampling_strategies: List[str]  # How each point was sampled
    ucb_scores: np.ndarray       # UCB score for each point
    utilities: np.ndarray        # Predicted utility μ(z) for each point
    uncertainties: np.ndarray    # Predicted uncertainty σ(z) for each point


class ContinuousEmbeddingSampler:
    """
    Sample points in reduced CLIP embedding space using GP-guided strategies.
    
    The sampler operates in PCA-reduced space (e.g., 32 dims) where:
    - GP predictions are more accurate (better coverage)
    - Convergence is faster (fewer dimensions)
    - Sampling is more meaningful (captures semantic subspace)
    
    After sampling, points are reconstructed to 768-dim for SDXL.
    """
    
    def __init__(
        self,
        gp,  # Any object with predict_utility(points) -> (mu, sigma)
        reducer: EmbeddingReducer,
        positive_embeddings_768: np.ndarray,   # (N_pos, 768) original embeddings
        negative_embeddings_768: np.ndarray,   # (N_neg, 768)
        neutral_embeddings_768: Optional[np.ndarray] = None,
        beta: float = 2.0,
        perturbation_scale: float = 0.3,       # Larger in reduced space
        n_centroid: int = 4,
        n_slerp: int = 3,
        n_perturbation: int = 3,
    ):
        """
        Initialize the sampler with reduced embeddings.
        
        Args:
            gp: Any object with predict_utility(points) -> (mu, sigma) interface
            reducer: Fitted EmbeddingReducer for dimension reduction
            positive_embeddings_768: Original 768-dim positive embeddings
            negative_embeddings_768: Original 768-dim negative embeddings
            neutral_embeddings_768: Original 768-dim neutral embeddings
            beta: UCB exploration coefficient
            perturbation_scale: Scale for exploration noise (in reduced space)
        """
        self.gp = gp
        self.reducer = reducer
        self.reduced_dim = reducer.dim
        
        # Store original 768-dim embeddings for reconstruction reference
        self._pos_768 = np.array(positive_embeddings_768)
        self._neg_768 = np.array(negative_embeddings_768) if len(negative_embeddings_768) > 0 else np.zeros((0, 768))
        self._neu_768 = np.array(neutral_embeddings_768) if neutral_embeddings_768 is not None and len(neutral_embeddings_768) > 0 else np.zeros((0, 768))
        
        # Reduce embeddings to working space
        self.positive_emb = reducer.reduce(self._pos_768) if len(self._pos_768) > 0 else np.zeros((0, self.reduced_dim))
        self.negative_emb = reducer.reduce(self._neg_768) if len(self._neg_768) > 0 else np.zeros((0, self.reduced_dim))
        self.neutral_emb = reducer.reduce(self._neu_768) if len(self._neu_768) > 0 else np.zeros((0, self.reduced_dim))
        
        self.beta = beta
        self.perturbation_scale = perturbation_scale
        self.n_centroid = n_centroid
        self.n_slerp = n_slerp
        self.n_perturbation = n_perturbation
        
        # Current mean in REDUCED space
        if len(self.positive_emb) > 0:
            self.current_mean = self.positive_emb.mean(axis=0)
        else:
            self.current_mean = np.zeros(self.reduced_dim)
        
        # Covariance in reduced space
        if len(self.positive_emb) > 1:
            centered = self.positive_emb - self.current_mean
            self.current_cov = np.cov(centered.T) + 1e-4 * np.eye(self.reduced_dim)
        else:
            self.current_cov = 0.1 * np.eye(self.reduced_dim)
        
        # Candidate pool in reduced space
        self.all_candidates = self._build_candidate_pool()
        self._ucb_cache = None
        
        print(f"[Sampler] Initialized in {self.reduced_dim}-dim reduced space")
    
    def _build_candidate_pool(self) -> np.ndarray:
        """Build pool of candidate points in reduced space."""
        candidates = []
        
        if len(self.positive_emb) > 0:
            candidates.append(self.positive_emb)
        
        if len(self.neutral_emb) > 0:
            candidates.append(self.neutral_emb)
        
        # Generate SLERP interpolations between positive pairs
        if len(self.positive_emb) >= 2:
            interp_points = []
            n_pos = len(self.positive_emb)
            for i in range(min(n_pos, 5)):
                for j in range(i + 1, min(n_pos, 5)):
                    for t in [0.25, 0.5, 0.75]:
                        interp = slerp_reduced(self.positive_emb[i], self.positive_emb[j], t)
                        interp_points.append(interp)
            if interp_points:
                candidates.append(np.array(interp_points))
        
        if candidates:
            return np.vstack(candidates)
        else:
            return np.random.randn(10, self.reduced_dim) * 0.1
    
    def invalidate_cache(self):
        """Invalidate UCB cache after GP update."""
        self._ucb_cache = None
    
    def update_distribution(self, preferred_points_768: np.ndarray, learning_rate: float = 0.3):
        """
        Update the sampling distribution based on preferred points.
        
        Args:
            preferred_points_768: (K, 768) embeddings from preferred images
            learning_rate: How much to shift toward new preferences
        """
        if len(preferred_points_768) == 0:
            return
        
        # Reduce to working space
        preferred_reduced = self.reducer.reduce(preferred_points_768)
        new_mean = preferred_reduced.mean(axis=0)
        
        self.current_mean = (1 - learning_rate) * self.current_mean + learning_rate * new_mean
        
        # Shrink covariance
        if len(preferred_reduced) > 1:
            centered = preferred_reduced - new_mean
            new_cov = np.cov(centered.T) + 1e-4 * np.eye(self.reduced_dim)
            self.current_cov = (1 - learning_rate) * self.current_cov + learning_rate * new_cov
        else:
            self.current_cov *= (1 - learning_rate * 0.5)
    
    def _compute_ucb(self, points_reduced: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute UCB scores for points in reduced space."""
        mu, sigma = self.gp.predict_utility(points_reduced)
        ucb = mu + self.beta * sigma
        return ucb, mu, sigma
    
    def _apply_repelling_force(self, point: np.ndarray) -> np.ndarray:
        """Apply repelling force from negative embeddings in reduced space."""
        if len(self.negative_emb) == 0:
            return point
        
        point_norm = point / (np.linalg.norm(point) + 1e-8)
        
        repulsion = np.zeros(self.reduced_dim)
        for neg in self.negative_emb:
            neg_norm = neg / (np.linalg.norm(neg) + 1e-8)
            similarity = np.dot(point_norm, neg_norm)
            
            if similarity > 0.3:
                direction = point_norm - neg_norm
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                force = (similarity - 0.3) * 0.5
                repulsion += force * direction
        
        new_point = point + repulsion
        return new_point / (np.linalg.norm(new_point) + 1e-8)
    
    def _sample_centroid_points(self, n: int) -> Tuple[np.ndarray, List[str]]:
        """Sample points near the current preference centroid."""
        points = []
        strategies = []
        
        for _ in range(n):
            noise = np.random.randn(self.reduced_dim) * self.perturbation_scale * 0.5
            point = self.current_mean + noise
            point = self._apply_repelling_force(point)
            points.append(point)
            strategies.append("centroid")
        
        return np.array(points), strategies
    
    def _sample_slerp_points(self, n: int) -> Tuple[np.ndarray, List[str]]:
        """Sample points via SLERP interpolation between high-utility embeddings."""
        points = []
        strategies = []
        
        if len(self.positive_emb) < 2:
            return self._sample_centroid_points(n)
        
        ucb, mu, sigma = self._compute_ucb(self.positive_emb)
        
        top_k = min(5, len(self.positive_emb))
        top_indices = np.argsort(ucb)[-top_k:]
        
        for _ in range(n):
            idx_pair = np.random.choice(top_indices, size=2, replace=False)
            v0 = self.positive_emb[idx_pair[0]]
            v1 = self.positive_emb[idx_pair[1]]
            
            t = np.random.uniform(0.2, 0.8)
            point = slerp_reduced(v0, v1, t)
            point = self._apply_repelling_force(point)
            
            points.append(point)
            strategies.append("slerp")
        
        return np.array(points), strategies
    
    def _sample_perturbation_points(self, n: int) -> Tuple[np.ndarray, List[str]]:
        """Sample points via UCB-guided perturbations."""
        points = []
        strategies = []
        
        ucb, mu, sigma = self._compute_ucb(self.all_candidates)
        
        top_k = min(10, len(self.all_candidates))
        top_indices = np.argsort(ucb)[-top_k:]
        
        for _ in range(n):
            base_idx = np.random.choice(top_indices)
            base_point = self.all_candidates[base_idx]
            
            local_sigma = sigma[base_idx]
            noise_scale = self.perturbation_scale * (1 + local_sigma)
            noise = np.random.randn(self.reduced_dim) * noise_scale
            
            point = base_point + noise
            point = self._apply_repelling_force(point)
            point = point / (np.linalg.norm(point) + 1e-8)
            
            points.append(point)
            strategies.append("perturbation")
        
        return np.array(points), strategies
    
    def sample_composition(self, n_points: int = 10) -> ContinuousComposition:
        """
        Sample a composition of n_points from the reduced embedding space.
        
        After sampling in reduced space, reconstructs to 768-dim for SDXL.
        
        Returns:
            ContinuousComposition with both reduced and 768-dim points
        """
        all_points_reduced = []
        all_strategies = []
        
        n_centroid = min(self.n_centroid, n_points)
        n_slerp = min(self.n_slerp, n_points - n_centroid)
        n_perturbation = n_points - n_centroid - n_slerp
        
        if n_centroid > 0:
            points, strategies = self._sample_centroid_points(n_centroid)
            all_points_reduced.append(points)
            all_strategies.extend(strategies)
        
        if n_slerp > 0:
            points, strategies = self._sample_slerp_points(n_slerp)
            all_points_reduced.append(points)
            all_strategies.extend(strategies)
        
        if n_perturbation > 0:
            points, strategies = self._sample_perturbation_points(n_perturbation)
            all_points_reduced.append(points)
            all_strategies.extend(strategies)
        
        all_points_reduced = np.vstack(all_points_reduced)
        
        # Compute UCB scores in reduced space
        ucb, mu, sigma = self._compute_ucb(all_points_reduced)
        
        # Compute attention weights from utilities
        weights = self._compute_attention_weights(mu)
        
        # RECONSTRUCT to 768-dim for SDXL
        points_768 = self.reducer.reconstruct(all_points_reduced)
        
        return ContinuousComposition(
            points=points_768,
            points_reduced=all_points_reduced,
            weights=weights,
            sampling_strategies=all_strategies,
            ucb_scores=ucb,
            utilities=mu,
            uncertainties=sigma
        )
    
    def _compute_attention_weights(self, utilities: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Compute attention weights from utilities using softmax."""
        utilities_shifted = utilities - np.max(utilities)
        exp_u = np.exp(utilities_shifted / temperature)
        weights = exp_u / (np.sum(exp_u) + 1e-8)
        
        min_weight = 0.05
        weights = np.clip(weights, min_weight, None)
        weights = weights / np.sum(weights)
        
        return weights
    
    def sample_negative_composition(self, n_points: int = 10) -> ContinuousComposition:
        """Sample points for negative prompt (CFG)."""
        if len(self.negative_emb) == 0:
            points_reduced = np.random.randn(n_points, self.reduced_dim) * 0.1
            points_768 = self.reducer.reconstruct(points_reduced)
            return ContinuousComposition(
                points=points_768,
                points_reduced=points_reduced,
                weights=np.ones(n_points) / n_points,
                sampling_strategies=["random"] * n_points,
                ucb_scores=np.zeros(n_points),
                utilities=np.zeros(n_points) - 1.0,
                uncertainties=np.ones(n_points)
            )
        
        points_reduced = []
        strategies = []
        
        for i in range(n_points):
            neg_idx = i % len(self.negative_emb)
            base = self.negative_emb[neg_idx]
            
            noise = np.random.randn(self.reduced_dim) * 0.1
            point = base + noise
            point = point / (np.linalg.norm(point) + 1e-8)
            
            points_reduced.append(point)
            strategies.append("negative")
        
        points_reduced = np.array(points_reduced)
        ucb, mu, sigma = self._compute_ucb(points_reduced)
        weights = np.ones(n_points) / n_points
        
        # Reconstruct to 768-dim
        points_768 = self.reducer.reconstruct(points_reduced)
        
        return ContinuousComposition(
            points=points_768,
            points_reduced=points_reduced,
            weights=weights,
            sampling_strategies=strategies,
            ucb_scores=ucb,
            utilities=mu,
            uncertainties=sigma
        )
    
    def sample_batch(self, batch_size: int = 4, n_points: int = 10) -> List[ContinuousComposition]:
        """Sample a batch of compositions for ranking."""
        compositions = []
        
        for i in range(batch_size):
            exploit_ratio = 1 - (i / batch_size) * 0.5
            
            self.n_centroid = int(n_points * exploit_ratio * 0.4)
            self.n_slerp = int(n_points * 0.3)
            self.n_perturbation = n_points - self.n_centroid - self.n_slerp
            
            comp = self.sample_composition(n_points)
            compositions.append(comp)
        
        self.n_centroid = 4
        self.n_slerp = 3
        self.n_perturbation = 3
        
        return compositions
    
    def get_current_variance(self) -> float:
        """Get the current uncertainty in the preferred region (reduced space)."""
        test_points = [self.current_mean]
        for _ in range(9):
            noise = np.random.randn(self.reduced_dim) * self.perturbation_scale
            test_points.append(self.current_mean + noise)
        
        test_points = np.array(test_points)
        _, sigma = self.gp.predict_utility(test_points)
        
        return float(np.mean(sigma))


# Backward compatibility
CompositionSample = ContinuousComposition
HITLSampler = ContinuousEmbeddingSampler
