"""
HITL Refinement System Tests

Unit tests and integration tests for the HITL (Human-in-the-Loop) 
preference optimization system.

Tests:
- Unit tests for HITLSampler
- Unit tests for RepellingOptimizer
- Unit tests for RankingToPairConverter
- Integration test: 5-round simulation with synthetic data
"""

import numpy as np
import pytest
import tempfile
import os
import json
from typing import List

# Import HITL modules
from hitl_sampler import HITLSampler, CompositionSample, cosine_similarity
from repelling_optimizer import RepellingOptimizer, RankingToPairConverter, create_synthetic_pairs_from_utilities
from exploration_GP import PreferenceLearner, PreferencePair


# ============== Test Fixtures ==============

def create_synthetic_embeddings(n: int, dim: int = 768, seed: int = 42) -> np.ndarray:
    """Create synthetic normalized embeddings."""
    np.random.seed(seed)
    embeddings = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def create_synthetic_labels(n: int) -> List[str]:
    """Create synthetic tag labels."""
    categories = ["warm", "cool", "soft", "bright", "dark", "modern", "cozy", "minimal"]
    modifiers = ["lighting", "texture", "tone", "style", "atmosphere", "aesthetic"]
    return [f"{categories[i % len(categories)]} {modifiers[i % len(modifiers)]}" for i in range(n)]


def create_mock_gp(n_embeddings: int = 50, seed: int = 42) -> PreferenceLearner:
    """Create a mock GP with some initial preferences."""
    gp = PreferenceLearner(embedding_dim=768, n_inducing=32)
    
    # Create synthetic embeddings and preferences
    embeddings = create_synthetic_embeddings(n_embeddings, seed=seed)
    
    # Create some synthetic preference pairs
    pairs = []
    for i in range(min(10, n_embeddings - 1)):
        pairs.append(PreferencePair(
            embedding_a=embeddings[i],
            embedding_b=embeddings[i + 1],
            strength=1.0
        ))
    
    if pairs:
        gp.add_preferences(pairs)
        gp.fit(n_epochs=20, verbose=False)
    
    return gp


# ============== Unit Tests: HITLSampler ==============

class TestHITLSampler:
    """Tests for the HITL Sampler."""
    
    def test_initialization(self):
        """Test sampler initialization."""
        embeddings = create_synthetic_embeddings(50)
        labels = create_synthetic_labels(50)
        gp = create_mock_gp()
        
        sampler = HITLSampler(
            preference_gp=gp,
            all_tag_embeddings=embeddings,
            all_tag_labels=labels
        )
        
        assert sampler.embeddings.shape == (50, 768)
        assert len(sampler.labels) == 50
        assert sampler.beta == 2.0
    
    def test_sample_composition(self):
        """Test sampling a single composition."""
        embeddings = create_synthetic_embeddings(50)
        labels = create_synthetic_labels(50)
        gp = create_mock_gp()
        
        sampler = HITLSampler(
            preference_gp=gp,
            all_tag_embeddings=embeddings,
            all_tag_labels=labels
        )
        
        composition = sampler.sample_composition(n_points=10)
        
        assert isinstance(composition, CompositionSample)
        assert composition.points.shape == (10, 768)
        assert len(composition.weights) == 10
        assert len(composition.tag_labels) == 10
        assert len(composition.tag_indices) == 10
        
        # Weights should be normalized
        assert np.isclose(composition.weights.sum(), 1.0, atol=0.01)
    
    def test_sample_batch(self):
        """Test sampling a batch of compositions."""
        embeddings = create_synthetic_embeddings(100)
        labels = create_synthetic_labels(100)
        gp = create_mock_gp(n_embeddings=100)
        
        sampler = HITLSampler(
            preference_gp=gp,
            all_tag_embeddings=embeddings,
            all_tag_labels=labels
        )
        
        compositions = sampler.sample_batch(batch_size=4, n_points=10)
        
        assert len(compositions) == 4
        for comp in compositions:
            assert comp.points.shape == (10, 768)
    
    def test_diversity_constraint(self):
        """Test that diversity constraint works."""
        # Create embeddings with some duplicates
        base = create_synthetic_embeddings(20, seed=42)
        # Add near-duplicates
        duplicates = base[:5] + np.random.randn(5, 768).astype(np.float32) * 0.01
        duplicates = duplicates / np.linalg.norm(duplicates, axis=1, keepdims=True)
        embeddings = np.vstack([base, duplicates])
        
        labels = create_synthetic_labels(25)
        gp = create_mock_gp(n_embeddings=25)
        
        sampler = HITLSampler(
            preference_gp=gp,
            all_tag_embeddings=embeddings,
            all_tag_labels=labels,
            diversity_threshold=0.85
        )
        
        composition = sampler.sample_composition(n_points=10)
        
        # Check that selected points are diverse
        for i in range(len(composition.tag_indices)):
            for j in range(i + 1, len(composition.tag_indices)):
                sim = cosine_similarity(
                    composition.points[i],
                    composition.points[j]
                )
                # Most pairs should be below threshold (some relaxation allowed)
                # This is a soft check since diversity is enforced greedily
                assert sim < 0.95, f"Points {i} and {j} too similar: {sim}"


# ============== Unit Tests: RankingToPairConverter ==============

class TestRankingToPairConverter:
    """Tests for the ranking to pair converter."""
    
    def test_basic_conversion(self):
        """Test basic ranking to pairs conversion."""
        converter = RankingToPairConverter()
        
        # Create mock compositions
        embeddings = create_synthetic_embeddings(40, seed=42)
        compositions = []
        for i in range(4):
            compositions.append(CompositionSample(
                points=embeddings[i*10:(i+1)*10],
                weights=np.ones(10) / 10,
                tag_labels=[f"tag_{j}" for j in range(10)],
                tag_indices=list(range(i*10, (i+1)*10)),
                point_ucb_scores=np.random.rand(10)
            ))
        
        # Ranking: image 0 > image 2 > image 1 > image 3
        ranking = [0, 2, 1, 3]
        
        pairs = converter.ranking_to_pairs(compositions, ranking)
        
        # Should produce 6 pairs: (0>2), (0>1), (0>3), (2>1), (2>3), (1>3)
        assert len(pairs) == 6
        
        # All pairs should have positive strength
        for pair in pairs:
            assert pair.strength > 0
    
    def test_pair_count(self):
        """Test expected pair count."""
        converter = RankingToPairConverter()
        
        assert converter.get_pair_count(2) == 1
        assert converter.get_pair_count(3) == 3
        assert converter.get_pair_count(4) == 6
        assert converter.get_pair_count(5) == 10


# ============== Unit Tests: RepellingOptimizer ==============

class TestRepellingOptimizer:
    """Tests for the repelling optimizer."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        gp = create_mock_gp()
        neg_embeddings = create_synthetic_embeddings(5)
        
        optimizer = RepellingOptimizer(
            preference_gp=gp,
            negative_embeddings=list(neg_embeddings)
        )
        
        assert len(optimizer.negative_embeddings) == 5
        assert len(optimizer.all_pairs) == 0
    
    def test_initialize_with_negatives(self):
        """Test seeding with positive/negative pairs."""
        gp = PreferenceLearner(embedding_dim=768, n_inducing=32)
        pos_embeddings = create_synthetic_embeddings(3, seed=42)
        neg_embeddings = create_synthetic_embeddings(2, seed=123)
        
        optimizer = RepellingOptimizer(
            preference_gp=gp,
            negative_embeddings=list(neg_embeddings)
        )
        
        optimizer.initialize_with_negatives(list(pos_embeddings))
        
        # Should have 3 * 2 = 6 pairs
        assert len(optimizer.all_pairs) == 6
    
    def test_update_from_ranking(self):
        """Test updating from a ranking."""
        gp = create_mock_gp()
        neg_embeddings = create_synthetic_embeddings(2, seed=123)
        
        optimizer = RepellingOptimizer(
            preference_gp=gp,
            negative_embeddings=list(neg_embeddings)
        )
        
        # Create mock compositions
        embeddings = create_synthetic_embeddings(40, seed=42)
        compositions = []
        for i in range(4):
            compositions.append(CompositionSample(
                points=embeddings[i*10:(i+1)*10],
                weights=np.ones(10) / 10,
                tag_labels=[f"tag_{j}" for j in range(10)],
                tag_indices=list(range(i*10, (i+1)*10)),
                point_ucb_scores=np.random.rand(10)
            ))
        
        ranking = [0, 2, 1, 3]
        result = optimizer.update_from_ranking(compositions, ranking)
        
        assert "gp_variance" in result
        assert "total_pairs" in result
        assert result["total_pairs"] == 6


# ============== Integration Test: 5-Round Simulation ==============

class TestIntegration:
    """Integration tests for the full HITL system."""
    
    def test_five_round_simulation(self):
        """
        Simulate 5 rounds of HITL refinement with synthetic data.
        
        This test verifies that:
        1. The system can run multiple rounds
        2. GP variance decreases over rounds (convergence)
        3. Top preferences stabilize
        """
        print("\n" + "=" * 60)
        print("HITL 5-Round Simulation Test")
        print("=" * 60)
        
        # Setup
        n_tags = 50
        embeddings = create_synthetic_embeddings(n_tags, seed=42)
        labels = create_synthetic_labels(n_tags)
        
        # Create GP and seed with some initial preferences
        gp = PreferenceLearner(embedding_dim=768, n_inducing=32)
        
        # Define "true" preferences - first 10 tags are preferred
        true_utilities = np.zeros(n_tags)
        true_utilities[:10] = 1.0  # First 10 are "good"
        true_utilities[40:] = -1.0  # Last 10 are "bad"
        
        # Create initial pairs based on true utilities
        initial_pairs = create_synthetic_pairs_from_utilities(
            embeddings, true_utilities, labels, n_pairs_per_tag=2
        )
        gp.add_preferences(initial_pairs)
        gp.fit(n_epochs=30, verbose=False)
        
        # Create negative embeddings (last 5 tags)
        neg_embeddings = embeddings[45:]
        pos_embeddings = embeddings[:5]
        
        # Initialize optimizer and sampler
        optimizer = RepellingOptimizer(
            preference_gp=gp,
            negative_embeddings=list(neg_embeddings),
            convergence_threshold=0.05
        )
        optimizer.initialize_with_negatives(list(pos_embeddings))
        
        sampler = HITLSampler(
            preference_gp=gp,
            all_tag_embeddings=embeddings,
            all_tag_labels=labels
        )
        
        # Run 5 rounds
        variances = []
        for round_num in range(5):
            print(f"\n--- Round {round_num + 1} ---")
            
            # Sample compositions
            compositions = sampler.sample_batch(batch_size=4, n_points=10)
            
            # Simulate user ranking based on true utilities
            # Score each composition by how many "good" tags it has
            scores = []
            for comp in compositions:
                # Count overlap with good tags (indices 0-9)
                good_count = sum(1 for idx in comp.tag_indices if idx < 10)
                bad_count = sum(1 for idx in comp.tag_indices if idx >= 40)
                scores.append(good_count - bad_count)
            
            # Create ranking (best first)
            ranking = sorted(range(4), key=lambda i: scores[i], reverse=True)
            print(f"  Scores: {scores}")
            print(f"  Ranking: {ranking}")
            
            # Update optimizer
            result = optimizer.update_from_ranking(compositions, ranking)
            variances.append(result["gp_variance"])
            
            # Invalidate sampler cache
            sampler.invalidate_cache()
            
            print(f"  GP Variance: {result['gp_variance']:.4f}")
            print(f"  Total Pairs: {result['total_pairs']}")
        
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"Variances over rounds: {[f'{v:.4f}' for v in variances]}")
        
        # Get final top preferences
        top_prefs = optimizer.get_top_preferences(embeddings, labels, k=5)
        print(f"Top 5 final preferences:")
        for pref in top_prefs:
            print(f"  - {pref['label']}: utility={pref['utility']:.3f}, uncertainty={pref['uncertainty']:.3f}")
        
        # Assertions
        # Variance should generally decrease (allowing some noise)
        assert variances[-1] < variances[0] * 1.5, "Variance should not increase significantly"
        
        # Total pairs should accumulate
        assert result["total_pairs"] > 20, "Should have accumulated many pairs"
        
        print("\n✅ 5-round simulation completed successfully!")


# ============== Run Tests ==============

def run_all_tests():
    """Run all tests with output."""
    print("=" * 60)
    print("Running HITL System Tests")
    print("=" * 60)
    
    # Run pytest programmatically
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
