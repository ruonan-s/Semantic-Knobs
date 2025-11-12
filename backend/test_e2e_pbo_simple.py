"""
Simplified End-to-End PBO Test (No CLIP/Torch dependencies)

Tests the complete PBO workflow with mock data:
1. Create mock concepts
2. Simulate UI stabilization
3. Simulate favorite selection
4. Generate proposals
5. Verify convergence
"""

import sys
import numpy as np
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from stage_refiner import StageRefiner
from pbo import PBO, compute_mixture_embedding


def create_mock_concepts(K=8, d=128):
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
            'member_tag_ids': [f"tag_{i}_0", f"tag_{i}_1", f"tag_{i}_2"]
        }
        concepts.append(concept)

        # Mock state
        concept_states[concept_id] = {
            'like_count': 0,
            'dislike_count': 0,
            'w': 1.0 / K,
            'score': 0.0,
            'rank_bonus': 0.0,
            'rank_penalty': 0.0
        }

    return concepts, concept_states


def simulate_user_preferences(concepts, concept_states):
    """Simulate user liking/disliking concepts"""
    # Randomly like 2-3 concepts
    num_likes = np.random.randint(2, 4)
    liked_indices = np.random.choice(len(concepts), num_likes, replace=False)

    # Dislike 1-2 concepts
    remaining = [i for i in range(len(concepts)) if i not in liked_indices]
    num_dislikes = np.random.randint(1, 3)
    disliked_indices = np.random.choice(remaining, num_dislikes, replace=False)

    # Update concept states
    for idx in liked_indices:
        cid = concepts[idx]['id']
        concept_states[cid]['like_count'] += 1

    for idx in disliked_indices:
        cid = concepts[idx]['id']
        concept_states[cid]['dislike_count'] += 1

    # Recompute weights (simple version)
    total_likes = sum(s['like_count'] for s in concept_states.values())
    if total_likes > 0:
        for cid, state in concept_states.items():
            state['w'] = (state['like_count'] + 0.1) / (total_likes + len(concepts) * 0.1)

        # Normalize
        total_w = sum(s['w'] for s in concept_states.values())
        for state in concept_states.values():
            state['w'] /= total_w

    return liked_indices, disliked_indices


def run_e2e_test():
    """Run simplified end-to-end test"""
    print("\n" + "=" * 70)
    print("SIMPLIFIED END-TO-END PBO TEST")
    print("=" * 70)

    session_id = "e2e_simple"
    stage = "impression"
    K = 8
    d = 128
    num_rounds = 4

    print(f"\nConfiguration:")
    print(f"  Concepts: {K}")
    print(f"  Embedding dim: {d}")
    print(f"  Rounds: {num_rounds}")

    # Create mock data
    concepts, concept_states = create_mock_concepts(K, d)
    image_ids = [f"img_{i}" for i in range(4)]

    # Mock incidence matrix
    incidence_matrix = {}
    for img_id in image_ids:
        incidence_matrix[img_id] = {}
        for i, concept in enumerate(concepts):
            if np.random.rand() > 0.5:
                incidence_matrix[img_id][concept['id']] = 1

    print(f"\n✅ Created {K} mock concepts")

    # Create StageRefiner
    refiner = StageRefiner(
        session_id=session_id,
        stage=stage,
        concepts=concepts,
        concept_states=concept_states,
        image_ids=image_ids,
        incidence_matrix=incidence_matrix,
        session_dir=Path(f"/tmp/{session_id}")
    )

    print(f"✅ StageRefiner initialized")
    print(f"  PBO: K={refiner.K}, d={refiner.d}")

    # ========================================================================
    # SIMULATION LOOP
    # ========================================================================

    for round_num in range(num_rounds):
        print(f"\n{'='*70}")
        print(f"ROUND {round_num + 1}/{num_rounds}")
        print(f"{'='*70}")

        # Step 1: User interactions
        print(f"\n[Step 1] Simulating user preferences...")
        liked_idx, disliked_idx = simulate_user_preferences(concepts, concept_states)

        print(f"  Liked: {[concepts[i]['label'] for i in liked_idx]}")
        print(f"  Disliked: {[concepts[i]['label'] for i in disliked_idx]}")

        # Step 2: UI stabilization
        print(f"\n[Step 2] UI stabilization...")

        # Get current weights
        w_ui = np.array([concept_states[c['id']]['w'] for c in concepts])

        # Wait for debounce
        time.sleep(0.6)

        # Record stabilization
        recorded = refiner.on_ui_stabilize(w_ui)

        print(f"  {'✅ Snapshot recorded' if recorded else '⏭️  Snapshot skipped'}")

        # Step 3: Generate proposals
        print(f"\n[Step 3] Generating proposals...")

        negatives = {concepts[i]['id'] for i in disliked_idx}

        proposals = refiner.propose_next_4(
            negatives=negatives,
            w_current=w_ui,
            fit_first=(round_num > 0)
        )

        print(f"  ✅ Generated {len(proposals)} proposals")
        for i, w in enumerate(proposals):
            top_idx = np.argmax(w)
            print(f"    Prop {i+1}: {concepts[top_idx]['label']} ({w[top_idx]:.3f})")

        # Step 4: Select favorite
        print(f"\n[Step 4] Favorite selection...")

        # Create round images
        round_image_ids = [f"pbo_r{round_num}_img_{i}" for i in range(4)]

        # Build incidence for round
        round_incidence = {}
        for i, img_id in enumerate(round_image_ids):
            round_incidence[img_id] = {}
            w_prop = proposals[i]
            for j, concept in enumerate(concepts):
                if w_prop[j] > 0.1:
                    round_incidence[img_id][concept['id']] = max(1, int(w_prop[j] * 10))

        # Pick favorite (image with highest liked concept weight)
        scores = []
        for w_prop in proposals:
            score = sum(w_prop[i] for i in liked_idx)
            scores.append(score)

        favorite_idx = np.argmax(scores)
        favorite_id = round_image_ids[favorite_idx]

        refiner.on_favorite(
            favorite_image_id=favorite_id,
            all_image_ids=round_image_ids,
            incidence_matrix=round_incidence
        )

        print(f"  ✅ Selected img_{favorite_idx} as favorite")

        # Step 5: PBO state
        pbo_state = refiner.pbo.to_dict()
        print(f"\n[Step 5] PBO State:")
        print(f"  Candidates: {pbo_state['num_candidates']}")
        print(f"  Duels: {pbo_state['num_duels']}")
        print(f"  Fitted: {pbo_state['fitted']}")

    # ========================================================================
    # FINAL VERIFICATION
    # ========================================================================
    print(f"\n{'='*70}")
    print("FINAL VERIFICATION")
    print(f"{'='*70}")

    final_state = refiner.pbo.to_dict()

    print(f"\n📊 Final PBO State:")
    print(f"  Total candidates: {final_state['num_candidates']}")
    print(f"  Total duels: {final_state['num_duels']}")
    print(f"  GP fitted: {final_state['fitted']}")

    # Run acceptance checks
    checks = []

    # Check 1: Candidates created
    checks.append(("Candidates created", final_state['num_candidates'] >= num_rounds))

    # Check 2: Duels recorded
    expected_duels = num_rounds * 3  # 3 strong duels per round
    checks.append(("Strong duels recorded", final_state['num_duels'] >= expected_duels))

    # Check 3: GP fitted
    checks.append(("GP fitted successfully", final_state['fitted']))

    # Check 4: Can propose
    try:
        test_proposals = refiner.propose_next_4(fit_first=True)
        checks.append(("Can propose new candidates", len(test_proposals) == 4))

        # Check simplex
        simplex_ok = all(abs(w.sum() - 1.0) < 0.01 for w in test_proposals)
        checks.append(("Proposals on simplex", simplex_ok))

        # Check diversity
        Z = np.array([compute_mixture_embedding(w, refiner.MU) for w in test_proposals])
        max_cos = 0.0
        for i in range(len(Z)):
            for j in range(i+1, len(Z)):
                cos = np.dot(Z[i], Z[j])
                max_cos = max(max_cos, cos)

        checks.append(("Diversity (cos < 0.98)", max_cos < 0.98))

        # Check negatives respected
        neg_concepts = list(negatives) if negatives else []
        if neg_concepts:
            neg_indices = [i for i, c in enumerate(concepts) if c['id'] in neg_concepts]
            avg_neg_weight = np.mean([w[neg_indices].mean() for w in test_proposals])
            checks.append(("Negatives de-emphasized", avg_neg_weight < 0.15))

    except Exception as e:
        checks.append(("Can propose new candidates", False))
        print(f"\n  Error proposing: {e}")

    # Print results
    print(f"\n✅ Acceptance Checks:")
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")

    all_passed = all(passed for _, passed in checks)

    if all_passed:
        print(f"\n{'='*70}")
        print("🎉 ALL TESTS PASSED!")
        print(f"{'='*70}")
        print("\nPBO workflow is functioning correctly.")
        print("Ready for production integration.")
        return True
    else:
        print(f"\n{'='*70}")
        print("⚠️  SOME TESTS FAILED")
        print(f"{'='*70}")
        failed = [name for name, passed in checks if not passed]
        print(f"Failed: {', '.join(failed)}")
        return False


if __name__ == "__main__":
    try:
        success = run_e2e_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
