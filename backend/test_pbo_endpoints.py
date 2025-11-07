# backend/test_pbo_endpoints.py
# Test script for PBO API endpoints

import sys
sys.path.insert(0, '..')

import numpy as np
from pathlib import Path
from backend.stage_refiner import StageRefiner
from backend.concept_refinement import Concept, ConceptState

def test_pbo_refiner_creation():
    """Test StageRefiner creation with mock data"""
    print("\n=== Test 1: StageRefiner Creation ===")

    # Create mock concepts
    concepts = [
        {
            'id': 'c0',
            'label': 'cozy',
            'centroid': np.random.randn(128).tolist()
        },
        {
            'id': 'c1',
            'label': 'modern',
            'centroid': np.random.randn(128).tolist()
        },
        {
            'id': 'c2',
            'label': 'minimalist',
            'centroid': np.random.randn(128).tolist()
        },
        {
            'id': 'c3',
            'label': 'warm lighting',
            'centroid': np.random.randn(128).tolist()
        },
    ]

    concept_states = {
        c['id']: {
            'active': True,
            'weight': 0.25,
            'total_positive_feedback': 0,
            'total_negative_feedback': 0
        }
        for c in concepts
    }

    image_ids = ['img_0', 'img_1', 'img_2', 'img_3']
    incidence_matrix = {
        'img_0': {'c0': 2, 'c1': 1},
        'img_1': {'c1': 3, 'c2': 1},
        'img_2': {'c0': 1, 'c3': 2},
        'img_3': {'c2': 2, 'c3': 1}
    }

    session_dir = Path('/tmp/test_pbo_session')
    session_dir.mkdir(parents=True, exist_ok=True)

    # Create StageRefiner
    refiner = StageRefiner(
        session_id='test_session',
        stage='impression',
        concepts=concepts,
        concept_states=concept_states,
        image_ids=image_ids,
        incidence_matrix=incidence_matrix,
        session_dir=session_dir
    )

    print(f"✅ Created StageRefiner with {len(concepts)} concepts")
    print(f"   Session: test_session/impression")
    print(f"   Images: {len(image_ids)}")

    return refiner


def test_stabilize_workflow(refiner: StageRefiner):
    """Test UI stabilization workflow"""
    print("\n=== Test 2: UI Stabilization ===")

    # Test 1: Record first snapshot
    w1 = np.array([0.4, 0.3, 0.2, 0.1])
    recorded = refiner.on_ui_stabilize(w1)
    print(f"Snapshot 1: {'recorded' if recorded else 'skipped'}")
    assert recorded, "First snapshot should be recorded"

    # Test 2: Try immediately (should fail debounce)
    w2 = np.array([0.45, 0.25, 0.2, 0.1])
    recorded = refiner.on_ui_stabilize(w2)
    print(f"Snapshot 2 (immediate): {'recorded' if recorded else 'skipped (debounce)'}")
    assert not recorded, "Should fail debounce"

    # Test 3: Wait and record with significant change
    import time
    time.sleep(0.6)  # Wait > 500ms
    w3 = np.array([0.3, 0.4, 0.2, 0.1])
    recorded = refiner.on_ui_stabilize(w3)
    print(f"Snapshot 3 (after wait): {'recorded' if recorded else 'skipped (threshold)'}")
    assert recorded, "Should be recorded after wait with significant change"

    print("✅ UI stabilization workflow passed")


def test_propose_workflow(refiner: StageRefiner):
    """Test proposal generation"""
    print("\n=== Test 3: Proposal Generation ===")

    # Generate proposals
    proposals = refiner.propose_next_4(fit_first=True)

    print(f"Generated {len(proposals)} proposals:")
    for i, w in enumerate(proposals):
        top_concepts = [refiner.concepts[j]['label'] for j in np.argsort(-w)[:2]]
        print(f"  Proposal {i+1}: top = {top_concepts}, sum = {w.sum():.3f}")

    assert len(proposals) == 4, "Should generate 4 proposals"
    for w in proposals:
        assert abs(w.sum() - 1.0) < 0.01, "Weights should sum to 1"

    print("✅ Proposal generation passed")
    return proposals


def test_favorite_workflow(refiner: StageRefiner):
    """Test favorite selection workflow"""
    print("\n=== Test 4: Favorite Selection ===")

    # Simulate 4 images from a round
    image_ids = ['pbo_img_0', 'pbo_img_1', 'pbo_img_2', 'pbo_img_3']

    # User picks image 2 as favorite
    refiner.on_favorite(
        favorite_image_id='pbo_img_2',
        all_image_ids=image_ids
    )

    print(f"Recorded favorite: pbo_img_2")
    print(f"Strong duels added: {len(image_ids) - 1}")
    print(f"Favorite candidate ID: {refiner.image_to_candidate['pbo_img_2']}")

    assert 'pbo_img_2' in refiner.image_to_candidate, "Favorite should be tracked"
    assert len(refiner.image_to_candidate) == 4, "All images should be tracked"

    print("✅ Favorite selection passed")


def test_full_pbo_cycle():
    """Test complete PBO cycle"""
    print("\n" + "="*60)
    print("Testing Complete PBO Cycle")
    print("="*60)

    # Step 1: Create refiner
    refiner = test_pbo_refiner_creation()

    # Step 2: Test stabilization
    test_stabilize_workflow(refiner)

    # Step 3: Generate proposals
    proposals = test_propose_workflow(refiner)

    # Step 4: (Images would be generated here via SDXL)
    print("\n=== Step 4: Image Generation (skipped - requires SDXL) ===")
    print("In production: SDXLRunner would generate images from proposals")

    # Step 5: Record favorite
    test_favorite_workflow(refiner)

    print("\n" + "="*60)
    print("✅ Complete PBO cycle test PASSED")
    print("="*60)

    # Print final state
    state = refiner.to_dict()
    print(f"\nFinal State:")
    print(f"  Candidates: {state['pbo_state']['num_candidates']}")
    print(f"  Duels: {state['pbo_state']['num_duels']}")
    print(f"  Image candidates: {state['num_image_candidates']}")


def main():
    """Run all tests"""
    try:
        test_full_pbo_cycle()

        print("\n" + "="*60)
        print("🎉 All PBO endpoint tests PASSED!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Start the server: uvicorn server:app --reload --port 8000")
        print("  2. Test endpoints with curl (see STAGE4_QUICK_START.md)")
        print("  3. Integrate with frontend (Stage 5)")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
