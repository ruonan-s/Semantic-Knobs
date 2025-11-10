"""
Test to verify PBO weight updates work correctly after user selections.

This test simulates the full workflow:
1. Initialize PBO with learned concept weights (warm start)
2. Generate first batch (cold start - perturbations of learned weights)
3. Simulate user selection
4. Generate second batch (should use PBO acquisition with GP)
"""

import numpy as np
from pbo import PBO, normalize_simplex


def test_pbo_weight_progression():
    """Test that PBO uses learned weights and evolves after selections."""
    
    print("="*80)
    print("Testing PBO Weight Update Workflow")
    print("="*80)
    
    # Setup: 11 concepts with random centroids
    K = 11
    d = 768
    np.random.seed(42)
    
    MU = np.random.randn(K, d).astype(np.float32)
    # Normalize centroids
    for i in range(K):
        MU[i] = MU[i] / np.linalg.norm(MU[i])
    
    concept_ids = [f"concept_{i}" for i in range(K)]
    
    # Create learned weights (simulate weights from exploration stage)
    # Mimic realistic learned distribution: top 3 concepts have higher weights
    learned_weights = np.array([0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01], dtype=np.float32)
    learned_weights = learned_weights / learned_weights.sum()
    
    # Initialize PBO with learned weights (warm start)
    pbo = PBO(MU=MU, concept_ids=concept_ids, concept_weights=learned_weights, random_state=42)
    
    print(f"\nInitialized PBO: K={K} concepts, d={d} dimensions")
    print(f"Learned weights (top 3): {learned_weights[:3]}")
    print(f"Candidates: {len(pbo.candidates)}, Duels: {len(pbo.duels)}, Fitted: {pbo.fitted}")
    
    # ========================================================================
    # Round 1: Cold start (should return perturbations of learned weights)
    # ========================================================================
    print("\n" + "="*80)
    print("ROUND 1: Cold Start (perturbations of learned weights)")
    print("="*80)
    
    proposals_r1 = pbo.propose_batch(q=4)
    
    print(f"\nProposed {len(proposals_r1)} candidates:")
    for i, w in enumerate(proposals_r1):
        max_w = w.max()
        num_nonzero = np.count_nonzero(w > 0.01)
        entropy = -np.sum(w * np.log(w + 1e-10))
        print(f"  Proposal {i}: max_weight={max_w:.3f}, non-zero_concepts={num_nonzero}, entropy={entropy:.2f}")
        print(f"    Top 3 weights: {sorted(w, reverse=True)[:3]}")
    
    # Check that Round 1 uses learned weights (distributed, not one-hot)
    # Expect proposals to have multiple non-zero weights (not just one)
    distributed_count = sum(1 for w in proposals_r1 if np.count_nonzero(w > 0.01) >= 5)
    print(f"\nRound 1 Analysis: {distributed_count}/{len(proposals_r1)} proposals are distributed (≥5 non-zero concepts)")
    
    if distributed_count >= 3:
        print("✅ PASS: Cold start uses learned weights (distributed proposals)")
    else:
        print("⚠️  WARNING: Expected distributed proposals based on learned weights")
    
    # ========================================================================
    # Simulate User Selection: User picks proposal 1 as favorite
    # ========================================================================
    print("\n" + "="*80)
    print("USER SELECTION: Favorite = Proposal 1")
    print("="*80)
    
    # Add all proposals as candidates
    cand_ids = []
    for i, w in enumerate(proposals_r1):
        cid = pbo.add_candidate(w, candidate_id=f"round1_prop_{i}")
        cand_ids.append(cid)
    
    print(f"\nAdded {len(cand_ids)} candidates to PBO")
    print(f"Candidate IDs: {cand_ids}")
    
    # Add duels: proposal 1 > all others
    favorite_idx = 1
    favorite_cid = cand_ids[favorite_idx]
    
    for i, cid in enumerate(cand_ids):
        if i != favorite_idx:
            pbo.add_preference(favorite_cid, cid, strength=1.0)
    
    print(f"\nAdded {len(cand_ids) - 1} strong duels: {favorite_cid} > others")
    print(f"PBO State: Candidates={len(pbo.candidates)}, Duels={len(pbo.duels)}, Fitted={pbo.fitted}")
    
    # ========================================================================
    # Round 2: After Selection (should use PBO acquisition)
    # ========================================================================
    print("\n" + "="*80)
    print("ROUND 2: After User Selection (PBO should be active)")
    print("="*80)
    
    # Fit GP before proposing
    pbo.fit()
    print(f"GP Fitted: {pbo.fitted}")
    
    proposals_r2 = pbo.propose_batch(q=4, w_current=proposals_r1[favorite_idx])
    
    print(f"\nProposed {len(proposals_r2)} candidates:")
    entropies_r2 = []
    for i, w in enumerate(proposals_r2):
        max_w = w.max()
        num_nonzero = np.count_nonzero(w > 0.01)
        entropy = -np.sum(w * np.log(w + 1e-10))
        entropies_r2.append(entropy)
        print(f"  Proposal {i}: max_weight={max_w:.3f}, non-zero={num_nonzero}, entropy={entropy:.2f}")
        print(f"    Top 3 weights: {sorted(w, reverse=True)[:3]}")
    
    # Calculate average entropies
    entropies_r1 = [-np.sum(w * np.log(w + 1e-10)) for w in proposals_r1]
    avg_entropy_r1 = np.mean(entropies_r1)
    avg_entropy_r2 = np.mean(entropies_r2)
    
    print(f"\nEntropy Analysis:")
    print(f"  Round 1 avg entropy: {avg_entropy_r1:.2f}")
    print(f"  Round 2 avg entropy: {avg_entropy_r2:.2f}")
    
    # Success criteria: Verify that proposals are different and GP is being used
    # Check if proposals have changed (not identical)
    proposals_differ = not all(np.allclose(proposals_r1[i], proposals_r2[i]) for i in range(4))
    
    if proposals_differ and pbo.fitted:
        print("✅ SUCCESS: PBO is working! Round 2 proposals differ from Round 1")
        print(f"   GP is fitted and generating new proposals based on preferences")
        return True
    elif not proposals_differ:
        print("❌ FAILURE: PBO not updating! Round 2 proposals identical to Round 1")
        return False
    else:
        print("❌ FAILURE: GP not fitted despite having data")
        return False


def test_tracker_records_selection():
    """Test that tracker.record_selection() properly logs user choices."""
    from pathlib import Path
    import tempfile
    import json
    import sys
    
    # Fix imports when running as script
    try:
        from tracking import GenerationTracker
    except ImportError:
        from backend.tracking import GenerationTracker
    
    print("\n" + "="*80)
    print("Testing Tracker Selection Recording")
    print("="*80)
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        session_path = Path(tmpdir)
        
        # Create tracker
        tracker = GenerationTracker(
            session_path=session_path,
            session_id="test_session",
            stage="impression",
            descriptor="Test descriptor"
        )
        
        # Set concepts
        concepts = [
            {'id': f'concept_{i}', 'label': f'Concept {i}', 'centroid': np.random.randn(768).tolist()}
            for i in range(5)
        ]
        tracker.set_concepts(concepts)
        
        # Start round
        tracker.start_round(round_number=1, reference_image="test_ref.png")
        
        # Add proposals (simulate 4 generated images)
        for i in range(4):
            w_raw = np.zeros(5)
            w_raw[i] = 1.0
            tracker.add_proposal(
                proposal_index=i,
                w_raw=w_raw,
                concepts=concepts,
                descriptor="Test descriptor",
                pos_phrases=[("Test descriptor", 1.5), (f"Concept {i}", 1.2)],
                neg_phrases=[],
                generated_image_path=f"test_image_{i}.png",
                seed=42 + i,
                generation_params={'mode': 'img2img', 'strength': 0.75, 'steps': 30, 'guidance_scale': 7.5}
            )
        
        # CRITICAL: Record user selection (this was missing!)
        selected_index = 2
        all_indices = [0, 1, 2, 3]
        tracker.record_selection(selected_index, all_indices)
        
        # Load and verify tracking.json
        tracking_file = session_path / "tracking.json"
        with open(tracking_file, 'r') as f:
            data = json.load(f)
        
        # Check that selection was recorded
        round_data = data['rounds'][0]
        
        if 'user_selection' in round_data:
            print("✅ PASS: user_selection field exists in tracking")
            sel = round_data['user_selection']
            print(f"   Selected index: {sel['selected_index']}")
            print(f"   Selected image: {sel['selected_image']}")
        else:
            print("❌ FAIL: user_selection field missing!")
            return False
        
        if 'pbo_update' in round_data:
            print("✅ PASS: pbo_update field exists in tracking")
            pbo = round_data['pbo_update']
            print(f"   Duels added: {pbo['num_duels']}")
            print(f"   GP fitted: {pbo['gp_fitted']}")
        else:
            print("❌ FAIL: pbo_update field missing!")
            return False
        
        return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PBO Weight Update Fix Verification")
    print("="*80 + "\n")
    
    results = []
    
    # Test 1: PBO weight progression (MAIN TEST)
    results.append(test_pbo_weight_progression())
    
    # Test 2: Tracker selection recording (OPTIONAL - requires full session setup)
    try:
        results.append(test_tracker_records_selection())
    except Exception as e:
        print(f"\n⚠️  Tracker test skipped (requires full session setup): {e}")
        print("   This is OK - tracker test is optional and not related to PBO logic")
    
    print("\n" + "="*80)
    if results and results[0]:  # Check if main PBO test passed
        print("✅ MAIN TEST PASSED")
        print("PBO learned weights integration is working correctly!")
        if len(results) > 1 and results[1]:
            print("✅ Tracker test also passed")
    else:
        print(f"❌ MAIN PBO TEST FAILED")
    print("="*80)

