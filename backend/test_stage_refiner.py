"""
Test StageRefiner with mock concepts and incidence matrix
"""

import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stage_refiner import StageRefiner, STABILIZE_DEBOUNCE_MS, STABILIZE_COSINE_THRESHOLD


def create_mock_concepts(K=5, d=16):
    """Create mock concepts for testing"""
    np.random.seed(42)

    concepts = []
    concept_states = {}

    for i in range(K):
        # Random centroid (L2 normalized)
        centroid = np.random.randn(d).astype(np.float32)
        centroid = centroid / np.linalg.norm(centroid)

        concept_id = f"concept_{i}"
        concept = {
            'id': concept_id,
            'label': f"Concept {i}",
            'centroid': centroid.tolist(),
            'member_tag_ids': [f"tag_{i}_0", f"tag_{i}_1"]
        }
        concepts.append(concept)

        # Mock state
        concept_states[concept_id] = {
            'like_count': 0,
            'dislike_count': 0,
            'w': 1.0 / K,
            'ema_w': 1.0 / K
        }

    return concepts, concept_states


def create_mock_incidence(image_ids, concept_ids, K):
    """Create mock incidence matrix"""
    np.random.seed(123)

    incidence = {}
    for img_id in image_ids:
        # Each image has 2-4 random concepts
        num_concepts = np.random.randint(2, 5)
        selected = np.random.choice(K, num_concepts, replace=False)

        incidence[img_id] = {}
        for idx in selected:
            incidence[img_id][concept_ids[idx]] = 1

    return incidence


def test_initialization():
    """Test StageRefiner initialization"""
    print("\n" + "=" * 60)
    print("TEST 1: Initialization")
    print("=" * 60)

    K = 5
    d = 16
    concepts, concept_states = create_mock_concepts(K, d)
    image_ids = [f"img_{i}" for i in range(4)]
    concept_ids = [c['id'] for c in concepts]
    incidence = create_mock_incidence(image_ids, concept_ids, K)

    refiner = StageRefiner(
        session_id="test_session",
        stage="impression",
        concepts=concepts,
        concept_states=concept_states,
        image_ids=image_ids,
        incidence_matrix=incidence,
        session_dir=Path("/tmp/test")
    )

    print(f"\n✅ StageRefiner initialized")
    print(f"  K={refiner.K}, d={refiner.d}")
    print(f"  MU shape: {refiner.MU.shape}")
    print(f"  Concept IDs: {refiner.concept_ids}")

    assert refiner.K == K
    assert refiner.d == d
    assert refiner.MU.shape == (K, d)

    print("\n✅ TEST 1 PASSED")


def test_stabilization():
    """Test UI stabilization with debouncing"""
    print("\n" + "=" * 60)
    print("TEST 2: UI Stabilization")
    print("=" * 60)

    K = 5
    d = 16
    concepts, concept_states = create_mock_concepts(K, d)
    image_ids = [f"img_{i}" for i in range(4)]
    concept_ids = [c['id'] for c in concepts]
    incidence = create_mock_incidence(image_ids, concept_ids, K)

    refiner = StageRefiner(
        session_id="test_session",
        stage="impression",
        concepts=concepts,
        concept_states=concept_states,
        image_ids=image_ids,
        incidence_matrix=incidence,
        session_dir=Path("/tmp/test")
    )

    # Test 1: First stabilize (should record)
    w1 = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
    recorded = refiner.on_ui_stabilize(w1)

    print(f"\n  Stabilize 1: recorded={recorded}")
    assert recorded, "First stabilize should record"
    print(f"  ✅ First snapshot recorded")

    # Test 2: Immediate second call (should be debounced)
    w2 = np.array([0.5, 0.25, 0.15, 0.08, 0.02])
    recorded = refiner.on_ui_stabilize(w2)

    print(f"\n  Stabilize 2 (immediate): recorded={recorded}")
    assert not recorded, "Should be debounced"
    print(f"  ✅ Debounce working")

    # Test 3: Wait and call with similar weights (should skip)
    time.sleep((STABILIZE_DEBOUNCE_MS + 100) / 1000.0)  # wait past debounce
    w3 = w1 + np.array([0.001, 0.0, -0.001, 0.0, 0.0])  # very similar
    recorded = refiner.on_ui_stabilize(w3)

    print(f"\n  Stabilize 3 (similar): recorded={recorded}")
    assert not recorded, "Should skip similar weights"
    print(f"  ✅ Similarity threshold working")

    # Test 4: Wait and call with different weights (should record + weak duel)
    time.sleep((STABILIZE_DEBOUNCE_MS + 100) / 1000.0)
    w4 = np.array([0.1, 0.1, 0.3, 0.4, 0.1])  # significantly different
    recorded = refiner.on_ui_stabilize(w4)

    print(f"\n  Stabilize 4 (different): recorded={recorded}")
    assert recorded, "Should record different weights"

    # Check weak duel was added
    assert len(refiner.pbo.duels) > 0, "Should have weak duels"
    weak_duels = [d for d in refiner.pbo.duels if d.strength == 0.5]
    print(f"  ✅ Weak duels recorded: {len(weak_duels)}")

    print("\n✅ TEST 2 PASSED")


def test_favorite_selection():
    """Test favorite image selection with strong duels"""
    print("\n" + "=" * 60)
    print("TEST 3: Favorite Selection")
    print("=" * 60)

    K = 5
    d = 16
    concepts, concept_states = create_mock_concepts(K, d)
    image_ids = [f"img_{i}" for i in range(4)]
    concept_ids = [c['id'] for c in concepts]
    incidence = create_mock_incidence(image_ids, concept_ids, K)

    refiner = StageRefiner(
        session_id="test_session",
        stage="impression",
        concepts=concepts,
        concept_states=concept_states,
        image_ids=image_ids,
        incidence_matrix=incidence,
        session_dir=Path("/tmp/test")
    )

    # Select img_2 as favorite
    favorite = "img_2"
    refiner.on_favorite(favorite, image_ids, incidence)

    # Check candidates were added for all images
    assert len(refiner.image_to_candidate) == len(image_ids)
    print(f"\n  ✅ Image candidates created: {len(refiner.image_to_candidate)}")

    # Check strong duels (should be 3: fav vs each other)
    strong_duels = [d for d in refiner.pbo.duels if d.strength == 1.0]
    assert len(strong_duels) == len(image_ids) - 1
    print(f"  ✅ Strong duels recorded: {len(strong_duels)}")

    # Verify favorite is the "better" in all duels
    fav_cid = refiner.image_to_candidate[favorite]
    for duel in strong_duels:
        assert duel.better_id == fav_cid, f"Favorite should be better in all duels"
    print(f"  ✅ Favorite is winner in all duels")

    print("\n✅ TEST 3 PASSED")


def test_proposal_generation():
    """Test proposal generation"""
    print("\n" + "=" * 60)
    print("TEST 4: Proposal Generation")
    print("=" * 60)

    K = 5
    d = 16
    concepts, concept_states = create_mock_concepts(K, d)
    image_ids = [f"img_{i}" for i in range(4)]
    concept_ids = [c['id'] for c in concepts]
    incidence = create_mock_incidence(image_ids, concept_ids, K)

    refiner = StageRefiner(
        session_id="test_session",
        stage="impression",
        concepts=concepts,
        concept_states=concept_states,
        image_ids=image_ids,
        incidence_matrix=incidence,
        session_dir=Path("/tmp/test")
    )

    # Add some data first
    refiner.on_favorite("img_2", image_ids, incidence)

    # Generate proposals
    proposals = refiner.propose_next_4(
        negatives={concept_ids[0]},  # dislike first concept
        w_current=np.ones(K) / K
    )

    print(f"\n  ✅ Generated {len(proposals)} proposals")
    assert len(proposals) == 4

    # Check each proposal
    for i, w in enumerate(proposals):
        print(f"\n  Proposal {i + 1}:")
        print(f"    w = {w}")
        print(f"    sum = {w.sum():.6f}")
        print(f"    max = {w.max():.3f}")

        # Check simplex
        assert np.isclose(w.sum(), 1.0), "Should be on simplex"
        assert np.all(w >= 0), "Should be non-negative"

    print(f"\n  ✅ All proposals valid")

    print("\n✅ TEST 4 PASSED")


def test_concept_phrases():
    """Test concept-to-phrases conversion"""
    print("\n" + "=" * 60)
    print("TEST 5: Concept Phrases")
    print("=" * 60)

    K = 5
    d = 16
    concepts, concept_states = create_mock_concepts(K, d)
    image_ids = [f"img_{i}" for i in range(4)]
    concept_ids = [c['id'] for c in concepts]
    incidence = create_mock_incidence(image_ids, concept_ids, K)

    refiner = StageRefiner(
        session_id="test_session",
        stage="impression",
        concepts=concepts,
        concept_states=concept_states,
        image_ids=image_ids,
        incidence_matrix=incidence,
        session_dir=Path("/tmp/test")
    )

    # Test mixture
    w = np.array([0.5, 0.25, 0.15, 0.07, 0.03])

    pos_phrases, neg_phrases = refiner.get_concept_phrases(w, top_k=3)

    print(f"\n  Positive phrases (top 3):")
    for phrase, weight in pos_phrases:
        print(f"    {phrase}: {weight:.3f}")

    print(f"\n  Negative phrases:")
    for phrase in neg_phrases:
        print(f"    {phrase}")

    # Checks
    assert len(pos_phrases) == 3, "Should have 3 positives"
    assert len(neg_phrases) <= 3, "Should have ≤ 3 negatives"

    # Top phrase should have highest weight
    assert pos_phrases[0][1] >= pos_phrases[1][1]
    assert pos_phrases[1][1] >= pos_phrases[2][1]

    print(f"\n  ✅ Phrases generated correctly")

    print("\n✅ TEST 5 PASSED")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STAGE REFINER TESTS")
    print("=" * 60)

    try:
        test_initialization()
        test_stabilization()
        test_favorite_selection()
        test_proposal_generation()
        test_concept_phrases()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
