"""
Repelling Optimizer - GP-Based Preference Learning with Emergent Repulsion

Instead of manual attraction/repulsion vectors, this optimizer feeds rankings
to the GP as pairwise comparisons. The "repulsion" from negative concepts
becomes an emergent property - the GP learns low utility near negative embeddings.

Key components:
- RankingToPairConverter: Converts ordinal rankings to pairwise comparisons
- RepellingOptimizer: Manages GP fitting and convergence tracking
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from exploration_GP import PreferenceLearner, PreferencePair
from hitl_sampler import CompositionSample


class RankingToPairConverter:
    """
    Convert ordinal rankings to pairwise preference comparisons for GP fitting.
    
    An ordinal ranking like [0, 2, 1, 3] (meaning image 0 is 1st, image 2 is 2nd, etc.)
    generates C(4,2) = 6 pairwise comparisons that the GP can learn from.
    """
    
    def __init__(
        self,
        use_centroids: bool = True,
        use_individual_points: bool = False,
        strength_decay: float = 0.8
    ):
        self.use_centroids = use_centroids
        self.use_individual_points = use_individual_points
        self.strength_decay = strength_decay
    
    def ranking_to_pairs(
        self,
        compositions: List[CompositionSample],
        ranking: List[int]
    ) -> List[PreferencePair]:
        """
        Convert ranking to pairwise comparisons.
        
        Args:
            compositions: List of CompositionSample objects
            ranking: Ordinal ranking e.g., [0, 2, 1, 3] means:
                     - Image 0 is 1st (best)
                     - Image 2 is 2nd
                     - Image 1 is 3rd
                     - Image 3 is 4th (worst)
        
        Returns:
            List of PreferencePair objects for GP fitting
        """
        pairs = []
        n = len(ranking)
        
        for i in range(n):
            for j in range(i + 1, n):
                winner_idx = ranking[i]
                loser_idx = ranking[j]
                
                # Strength based on rank difference
                rank_diff = j - i
                strength = self.strength_decay ** (rank_diff - 1)
                
                if self.use_centroids:
                    winner_centroid = compositions[winner_idx].points.mean(axis=0)
                    loser_centroid = compositions[loser_idx].points.mean(axis=0)
                    
                    pairs.append(PreferencePair(
                        embedding_a=winner_centroid,
                        embedding_b=loser_centroid,
                        strength=strength
                    ))
                
                if self.use_individual_points:
                    winner_comp = compositions[winner_idx]
                    loser_comp = compositions[loser_idx]
                    
                    winner_top = np.argsort(winner_comp.weights)[-3:]
                    loser_top = np.argsort(loser_comp.weights)[-3:]
                    
                    for wi in winner_top:
                        for li in loser_top:
                            pairs.append(PreferencePair(
                                embedding_a=winner_comp.points[wi],
                                embedding_b=loser_comp.points[li],
                                strength=strength * 0.3
                            ))
        
        return pairs
    
    def get_pair_count(self, n_images: int) -> int:
        """Get expected number of pairs from n images."""
        base_pairs = n_images * (n_images - 1) // 2
        if self.use_individual_points:
            return base_pairs * (1 + 9) if self.use_centroids else base_pairs * 9
        return base_pairs


class RepellingOptimizer:
    """
    GP-based preference optimizer with emergent repulsion.
    
    Instead of manual attraction/repulsion vectors, the GP learns the utility
    surface from pairwise comparisons. Negative tags are incorporated by:
    1. Seeding GP with (positive, negative) pairs as initial observations
    2. The GP naturally predicts low utility near negative embeddings
    """
    
    def __init__(
        self,
        preference_gp: PreferenceLearner,
        negative_embeddings: List[np.ndarray],
        pair_converter: Optional[RankingToPairConverter] = None,
        convergence_threshold: float = 0.05,
        n_fit_epochs: int = 50
    ):
        self.gp = preference_gp
        self.negative_embeddings = list(negative_embeddings)
        self.pair_converter = pair_converter or RankingToPairConverter()
        self.convergence_threshold = convergence_threshold
        self.n_fit_epochs = n_fit_epochs
        
        self.all_pairs: List[PreferencePair] = []
        self.round_pair_counts: List[int] = []
    
    def initialize_with_negatives(
        self,
        positive_embeddings: List[np.ndarray],
        strength: float = 0.8
    ):
        """Seed GP with (positive, negative) pairs to establish repulsion."""
        initial_pairs = []
        
        for pos in positive_embeddings:
            for neg in self.negative_embeddings:
                initial_pairs.append(PreferencePair(
                    embedding_a=pos,
                    embedding_b=neg,
                    strength=strength
                ))
        
        if initial_pairs:
            self.all_pairs.extend(initial_pairs)
            self._fit_gp()
            print(f"[RepellingOptimizer] Seeded GP with {len(initial_pairs)} (pos, neg) pairs")
    
    def update_from_ranking(
        self,
        compositions: List[CompositionSample],
        ranking: List[int],
        refit: bool = True
    ) -> Dict:
        """Update GP from ordinal ranking."""
        new_pairs = self.pair_converter.ranking_to_pairs(compositions, ranking)
        
        self.all_pairs.extend(new_pairs)
        self.round_pair_counts.append(len(new_pairs))
        
        if refit and len(self.all_pairs) >= 3:
            self._fit_gp()
        
        metrics = self._compute_metrics(compositions)
        metrics["new_pairs"] = len(new_pairs)
        metrics["total_pairs"] = len(self.all_pairs)
        
        return metrics
    
    def _fit_gp(self):
        """Fit GP to all accumulated pairwise comparisons."""
        if len(self.all_pairs) < 3:
            print(f"[RepellingOptimizer] Not enough pairs to fit ({len(self.all_pairs)})")
            return
        
        self.gp.preference_pairs = []
        self.gp.all_embeddings = []
        self.gp.add_preferences(self.all_pairs)
        
        self.gp.fit(n_epochs=self.n_fit_epochs, verbose=False)
        
        print(f"[RepellingOptimizer] Fitted GP on {len(self.all_pairs)} pairs")
    
    def _compute_metrics(self, compositions: List[CompositionSample]) -> Dict:
        """Compute convergence and quality metrics."""
        all_points = np.vstack([c.points for c in compositions])
        
        mu, sigma = self.gp.predict_utility(all_points)
        
        median_utility = np.median(mu)
        high_utility_mask = mu > median_utility
        
        if np.any(high_utility_mask):
            preferred_variance = float(np.mean(sigma[high_utility_mask]))
        else:
            preferred_variance = float(np.mean(sigma))
        
        return {
            "gp_variance": preferred_variance,
            "mean_utility": float(np.mean(mu)),
            "max_utility": float(np.max(mu)),
            "mean_uncertainty": float(np.mean(sigma)),
            "utility_spread": float(np.max(mu) - np.min(mu)),
        }
    
    def compute_preferred_region_variance(
        self,
        candidate_embeddings: np.ndarray,
        top_fraction: float = 0.25
    ) -> float:
        """Compute average GP uncertainty in high-utility regions."""
        mu, sigma = self.gp.predict_utility(candidate_embeddings)
        
        n_top = max(1, int(len(mu) * top_fraction))
        top_indices = np.argsort(mu)[-n_top:]
        
        return float(np.mean(sigma[top_indices]))
    
    def is_converged(self, candidate_embeddings: np.ndarray) -> bool:
        """Check if preferences have converged."""
        variance = self.compute_preferred_region_variance(candidate_embeddings)
        return variance < self.convergence_threshold
    
    def get_top_preferences(
        self,
        candidate_embeddings: np.ndarray,
        candidate_labels: List[str],
        k: int = 10
    ) -> List[Dict]:
        """Get top-K preferred concepts from GP."""
        mu, sigma = self.gp.predict_utility(candidate_embeddings)
        
        top_indices = np.argsort(mu)[-k:][::-1]
        
        return [
            {
                "label": candidate_labels[i],
                "utility": float(mu[i]),
                "uncertainty": float(sigma[i]),
            }
            for i in top_indices
        ]
    
    def get_statistics(self) -> Dict:
        """Get optimizer statistics."""
        return {
            "total_pairs": len(self.all_pairs),
            "n_rounds": len(self.round_pair_counts),
            "pairs_per_round": self.round_pair_counts,
            "n_negative_embeddings": len(self.negative_embeddings),
            "gp_is_fitted": self.gp.is_fitted,
        }


def create_synthetic_pairs_from_utilities(
    embeddings: np.ndarray,
    utilities: np.ndarray,
    labels: List[str],
    n_pairs_per_tag: int = 5
) -> List[PreferencePair]:
    """
    Create synthetic pairwise comparisons from utility values.
    
    Used to seed GP with exploration utilities as initial observations.
    """
    n = len(utilities)
    sorted_indices = np.argsort(utilities)[::-1]
    
    pairs = []
    for i, winner_idx in enumerate(sorted_indices):
        for j in range(i + 1, min(i + 1 + n_pairs_per_tag, n)):
            loser_idx = sorted_indices[j]
            
            utility_diff = utilities[winner_idx] - utilities[loser_idx]
            strength = min(1.0, utility_diff + 0.5)
            
            pairs.append(PreferencePair(
                embedding_a=embeddings[winner_idx],
                embedding_b=embeddings[loser_idx],
                strength=strength
            ))
    
    return pairs
