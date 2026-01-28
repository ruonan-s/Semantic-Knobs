"""
Repelling Optimizer - GP-Based Preference Learning in Reduced Space

Uses Point-Level Utility Assignment:
- Each composition has 10 feature points
- User ranks 4 compositions → 40 points get utility values
- GP learns u(z) directly from (point, utility) observations

This preserves all feature-level information instead of collapsing to centroids.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from hitl_sampler import ContinuousComposition
from dim_reducer import EmbeddingReducer


@dataclass
class UtilityObservation:
    """A single (point, utility) observation for GP training."""
    point: np.ndarray      # Reduced-dim embedding
    utility: float         # Assigned utility value
    weight: float = 1.0    # Observation weight (attention weight)


class PointLevelUtilityGP:
    """
    GP that learns utility function from individual point observations.
    
    Instead of pairwise comparisons, directly fits u(z) from labeled points.
    Uses scikit-learn's GaussianProcessRegressor for simplicity.
    """
    
    def __init__(self, reduced_dim: int, length_scale: float = 1.0):
        """
        Initialize the utility GP.
        
        Args:
            reduced_dim: Dimensionality of reduced embedding space
            length_scale: RBF kernel length scale
        """
        self.reduced_dim = reduced_dim
        
        # RBF kernel with automatic length scale tuning
        kernel = ConstantKernel(1.0) * RBF(
            length_scale=length_scale,
            length_scale_bounds=(0.1, 10.0)
        ) + WhiteKernel(noise_level=0.1)
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            normalize_y=True,
            alpha=0.1  # Regularization
        )
        
        self.observations: List[UtilityObservation] = []
        self.is_fitted = False
    
    def add_observations(self, obs_list: List[UtilityObservation]):
        """Add utility observations."""
        self.observations.extend(obs_list)
        self.is_fitted = False
    
    def fit(self):
        """Fit GP to all accumulated observations."""
        if len(self.observations) < 3:
            print(f"[UtilityGP] Not enough observations ({len(self.observations)})")
            return
        
        # Prepare training data
        X = np.array([obs.point for obs in self.observations])
        y = np.array([obs.utility for obs in self.observations])
        
        # Fit with sample weights based on attention weights
        # Note: sklearn GP doesn't support sample weights directly,
        # so we duplicate high-weight samples
        weights = np.array([obs.weight for obs in self.observations])
        weights = weights / weights.max()  # Normalize
        
        # Simple weighting: include each point ceil(weight * 3) times
        X_weighted = []
        y_weighted = []
        for i in range(len(X)):
            n_copies = max(1, int(np.ceil(weights[i] * 2)))
            for _ in range(n_copies):
                X_weighted.append(X[i])
                y_weighted.append(y[i])
        
        X_weighted = np.array(X_weighted)
        y_weighted = np.array(y_weighted)
        
        try:
            self.gp.fit(X_weighted, y_weighted)
            self.is_fitted = True
            print(f"[UtilityGP] Fitted on {len(self.observations)} observations "
                  f"({len(X_weighted)} weighted samples)")
        except Exception as e:
            print(f"[UtilityGP] Fit failed: {e}")
    
    def predict_utility(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict utility and uncertainty for points.
        
        Returns:
            (mu, sigma) arrays
        """
        if not self.is_fitted or len(self.observations) < 3:
            # Return prior: zero mean, high uncertainty
            n = len(points) if points.ndim > 1 else 1
            return np.zeros(n), np.ones(n) * 0.5
        
        single = points.ndim == 1
        if single:
            points = points.reshape(1, -1)
        
        mu, sigma = self.gp.predict(points, return_std=True)
        
        return (mu[0], sigma[0]) if single else (mu, sigma)
    
    def clear(self):
        """Clear all observations."""
        self.observations = []
        self.is_fitted = False


class ContinuousSpaceOptimizer:
    """
    GP-based optimizer using point-level utility assignment.
    
    When user ranks 4 images:
    - 1st place: all 10 points get utility = 1.0
    - 2nd place: all 10 points get utility = 0.66
    - 3rd place: all 10 points get utility = 0.33
    - 4th place: all 10 points get utility = 0.0
    
    This preserves feature-level information for GP learning.
    """
    
    def __init__(
        self,
        gp: "PreferenceLearner",  # Keep for backward compatibility but use our GP
        reducer: EmbeddingReducer,
        positive_embeddings_768: np.ndarray,
        negative_embeddings_768: np.ndarray,
        neutral_embeddings_768: Optional[np.ndarray] = None,
        convergence_threshold: float = 0.15,
        n_fit_epochs: int = 50  # Unused but kept for compatibility
    ):
        """Initialize optimizer with point-level utility GP."""
        self.reducer = reducer
        self.reduced_dim = reducer.dim
        
        # Store original 768-dim
        self._pos_768 = np.array(positive_embeddings_768)
        self._neg_768 = np.array(negative_embeddings_768) if len(negative_embeddings_768) > 0 else np.zeros((0, 768))
        self._neu_768 = np.array(neutral_embeddings_768) if neutral_embeddings_768 is not None and len(neutral_embeddings_768) > 0 else np.zeros((0, 768))
        
        # Reduce to working space
        self.positive_emb = reducer.reduce(self._pos_768) if len(self._pos_768) > 0 else np.zeros((0, self.reduced_dim))
        self.negative_emb = reducer.reduce(self._neg_768) if len(self._neg_768) > 0 else np.zeros((0, self.reduced_dim))
        self.neutral_emb = reducer.reduce(self._neu_768) if len(self._neu_768) > 0 else np.zeros((0, self.reduced_dim))
        
        self.convergence_threshold = convergence_threshold
        
        # Point-level utility GP (replaces pairwise GP)
        self.utility_gp = PointLevelUtilityGP(reduced_dim=self.reduced_dim)
        
        # Keep reference to original GP for predict_utility interface
        self._original_gp = gp
        
        # Tracking
        self.all_observations: List[UtilityObservation] = []
        self.round_counts: List[int] = []
        
        # Current centroid in reduced space
        self.current_centroid = self.positive_emb.mean(axis=0) if len(self.positive_emb) > 0 else np.zeros(self.reduced_dim)
        
        print(f"[Optimizer] Initialized with point-level utility GP in {self.reduced_dim}D space")
    
    def seed_gp(self, pos_utility: float = 0.8, neg_utility: float = 0.1):
        """
        Seed GP with initial utility observations.
        
        Positive embeddings → high utility (attractors)
        Negative embeddings → low utility (repellers)
        Neutral embeddings → medium utility with high uncertainty
        """
        initial_obs = []
        
        # Positive tags: high utility
        for pos in self.positive_emb:
            initial_obs.append(UtilityObservation(
                point=pos,
                utility=pos_utility,
                weight=1.0
            ))
        
        # Negative tags: low utility (repellers)
        for neg in self.negative_emb:
            initial_obs.append(UtilityObservation(
                point=neg,
                utility=neg_utility,
                weight=1.0
            ))
        
        # Neutral tags: medium utility (exploration targets)
        for neu in self.neutral_emb:
            initial_obs.append(UtilityObservation(
                point=neu,
                utility=0.5,
                weight=0.5  # Lower weight = more uncertainty
            ))
        
        if initial_obs:
            self.utility_gp.add_observations(initial_obs)
            self.all_observations.extend(initial_obs)
            self.utility_gp.fit()
            print(f"[Optimizer] Seeded GP with {len(initial_obs)} initial observations "
                  f"({len(self.positive_emb)} pos, {len(self.negative_emb)} neg, {len(self.neutral_emb)} neu)")
    
    def update_from_ranking(
        self,
        compositions: List[ContinuousComposition],
        ranking: List[int],
        learning_rate: float = 0.3
    ) -> Dict:
        """
        Update GP from user's ordinal ranking using point-level utility.
        
        Each point in each composition gets a utility based on rank:
        - 1st place: utility = 1.0
        - 2nd place: utility = 0.66
        - 3rd place: utility = 0.33
        - 4th place: utility = 0.0
        """
        n_comps = len(ranking)
        new_obs = []
        
        # Assign utilities based on rank position
        rank_utilities = {
            0: 1.0,    # 1st place
            1: 0.66,   # 2nd place
            2: 0.33,   # 3rd place
            3: 0.0,    # 4th place
        }
        
        # Create observations for ALL points in ALL compositions
        for rank_position, comp_idx in enumerate(ranking):
            comp = compositions[comp_idx]
            utility = rank_utilities.get(rank_position, 0.5)
            
            # Add observation for each of the 10 points
            for i, point in enumerate(comp.points_reduced):
                # Weight by attention weight (high-weight features matter more)
                weight = float(comp.weights[i])
                
                new_obs.append(UtilityObservation(
                    point=point,
                    utility=utility,
                    weight=weight
                ))
        
        # Add to GP and refit
        self.utility_gp.add_observations(new_obs)
        self.all_observations.extend(new_obs)
        self.round_counts.append(len(new_obs))
        
        self.utility_gp.fit()
        
        # Update centroid toward preferred compositions
        best_idx = ranking[0]
        second_idx = ranking[1] if len(ranking) > 1 else ranking[0]
        
        best_centroid = self._get_weighted_centroid_reduced(compositions[best_idx])
        second_centroid = self._get_weighted_centroid_reduced(compositions[second_idx])
        preferred_centroid = 0.7 * best_centroid + 0.3 * second_centroid
        
        self.current_centroid = (1 - learning_rate) * self.current_centroid + \
                                learning_rate * preferred_centroid
        
        # Compute metrics
        metrics = self._compute_metrics()
        metrics["new_observations"] = len(new_obs)
        metrics["total_observations"] = len(self.all_observations)
        
        print(f"[Optimizer] Added {len(new_obs)} point observations (total: {len(self.all_observations)})")
        
        return metrics
    
    def _get_weighted_centroid_reduced(self, composition: ContinuousComposition) -> np.ndarray:
        """Get weighted centroid in reduced space."""
        weights = composition.weights / (composition.weights.sum() + 1e-8)
        centroid = (composition.points_reduced * weights[:, np.newaxis]).sum(axis=0)
        return centroid / (np.linalg.norm(centroid) + 1e-8)
    
    def _compute_metrics(self) -> Dict:
        """Compute convergence metrics using point-level GP."""
        # Test points around current centroid
        test_points = [self.current_centroid]
        for _ in range(19):
            noise = np.random.randn(self.reduced_dim) * 0.2
            point = self.current_centroid + noise
            point = point / (np.linalg.norm(point) + 1e-8)
            test_points.append(point)
        
        test_points = np.array(test_points)
        mu, sigma = self.utility_gp.predict_utility(test_points)
        
        # Also check positive embeddings
        if len(self.positive_emb) > 0:
            pos_mu, pos_sigma = self.utility_gp.predict_utility(self.positive_emb)
            avg_pos_utility = float(np.mean(pos_mu))
        else:
            avg_pos_utility = 0.0
        
        return {
            "gp_variance": float(np.mean(sigma)),
            "centroid_utility": float(mu[0]) if len(mu) > 0 else 0.0,
            "mean_utility": float(np.mean(mu)),
            "max_utility": float(np.max(mu)),
            "positive_avg_utility": avg_pos_utility,
            "is_converged": float(np.mean(sigma)) < self.convergence_threshold
        }
    
    def predict_utility(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict utility for points (interface for sampler)."""
        return self.utility_gp.predict_utility(points)
    
    def is_converged(self) -> bool:
        """Check if preferences have converged."""
        metrics = self._compute_metrics()
        return metrics["gp_variance"] < self.convergence_threshold
    
    def get_current_centroid(self) -> np.ndarray:
        """Get current centroid in reduced space."""
        return self.current_centroid.copy()
    
    def get_current_centroid_768(self) -> np.ndarray:
        """Get current centroid reconstructed to 768-dim."""
        return self.reducer.reconstruct(self.current_centroid)
    
    def get_top_regions(self, n_regions: int = 5) -> List[np.ndarray]:
        """Get top utility regions reconstructed to 768-dim."""
        # Candidates in reduced space
        candidates = []
        if len(self.positive_emb) > 0:
            candidates.extend(list(self.positive_emb))
        
        candidates.append(self.current_centroid)
        for _ in range(20):
            noise = np.random.randn(self.reduced_dim) * 0.3
            point = self.current_centroid + noise
            candidates.append(point / (np.linalg.norm(point) + 1e-8))
        
        candidates = np.array(candidates)
        mu, _ = self.utility_gp.predict_utility(candidates)
        
        top_indices = np.argsort(mu)[-n_regions:][::-1]
        
        # Reconstruct to 768-dim
        top_reduced = candidates[top_indices]
        return [self.reducer.reconstruct(r) for r in top_reduced]


# Backward compatibility
RepellingOptimizer = ContinuousSpaceOptimizer
