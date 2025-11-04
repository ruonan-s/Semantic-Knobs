"""
End-to-End Integration Test for PBO Workflow

Simulates complete user journey:
1. Initialize concepts from tags
2. User interacts with tags (like/dislike)
3. UI stabilization → weak duels
4. Generate proposals with PBO
5. User picks favorite → strong duels
6. Repeat for multiple rounds
7. Verify convergence and diversity
"""

import sys
import numpy as np
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from concept_refinement import ConceptRefinementSession, get_or_create_session
from stage_refiner import StageRefiner
from pbo import PBO


def create_mock_image_tags(num_images=4):
    """Create mock tags for testing"""
    # Interior design tags
    tag_pool = [
        "cozy", "comfortable", "warm", "inviting", "soft lighting",
        "modern", "minimalist", "clean lines", "sleek", "contemporary",
        "rustic", "natural materials", "wood", "stone", "earthy",
        "bright", "airy", "spacious", "open", "light-filled",
        "elegant", "sophisticated", "refined", "luxurious", "upscale"
    ]

    image_tags = {}
    for i in range(num_images):
        # Each image gets 6-8 random tags
        num_tags = np.random.randint(6, 9)
        tags = list(np.random.choice(tag_pool, num_tags, replace=False))
        image_tags[f"img_{i}"] = tags

    return image_tags


def test_e2e_workflow():
    """
    Test complete end-to-end PBO workflow
    """
    print("\n" + "=" * 70)
    print("END-TO-END PBO INTEGRATION TEST")
    print("=" * 70)

    # Configuration
    session_id = "e2e_test_session"
    stage = "impression"
    num_rounds = 3

    print(f"\nConfiguration:")
    print(f"  Session: {session_id}")
    print(f"  Stage: {stage}")
    print(f"  Rounds: {num_rounds}")

    # ========================================================================
    # ROUND 0: Initial Setup
    # ========================================================================
    print("\n" + "-" * 70)
    print("ROUND 0: Initial Setup")
    print("-" * 70)

    # Step 1: Initialize concepts from tags
    image_tags = create_mock_image_tags(4)
    image_ids = list(image_tags.keys())

    print(f"\n📊 Generated {len(image_ids)} images with tags:")
    for img_id, tags in image_tags.items():
        print(f"  {img_id}: {', '.join(tags[:3])}... ({len(tags)} total)")

    # Create concept refinement session
    session = get_or_create_session(session_id, stage, image_ids)
    session.initialize_from_tags(image_tags)

    print(f"\n✅ Initialized {len(session.concepts)} concepts")
    for i, concept in enumerate(session.concepts[:5]):
        print(f"  Concept {i}: '{concept.label}' ({len(concept.member_tag_ids)} tags)")

    # Create StageRefiner
    concepts_dict = [
        {
            'id': c.id,
            'label': c.label,
            'centroid': c.centroid,
            'member_tag_ids': c.member_tag_ids
        }
        for c in session.concepts
    ]

    refiner = StageRefiner(
        session_id=session_id,
        stage=stage,
        concepts=concepts_dict,
        concept_states=session.concept_states,
        image_ids=image_ids,
        incidence_matrix=session.incidence_matrix,
        session_dir=Path(f"/tmp/{session_id}")
    )

    print(f"\n✅ Created StageRefiner")
    print(f"  PBO initialized with K={refiner.K} concepts, d={refiner.d} embedding dim")

    # ========================================================================
    # SIMULATION LOOP: Multiple rounds of preference learning
    # ========================================================================

    for round_num in range(num_rounds):
        print("\n" + "=" * 70)
        print(f"ROUND {round_num + 1}/{num_rounds}")
        print("=" * 70)

        # --------------------------------------------------------------------
        # Step 1: User interacts with tags (simulate)
        # --------------------------------------------------------------------
        print(f"\n[Step 1] Simulating user tag interactions...")

        # Pick 2-3 random concepts to like, 1-2 to dislike
        num_likes = np.random.randint(2, 4)
        num_dislikes = np.random.randint(1, 3)

        all_concept_ids = [c['id'] for c in concepts_dict]
        like_concepts = np.random.choice(all_concept_ids, num_likes, replace=False)
        dislike_concepts = np.random.choice(
            [c for c in all_concept_ids if c not in like_concepts],
            num_dislikes,
            replace=False
        )

        # Simulate tag clicks
        for concept_id in like_concepts:
            concept = next(c for c in session.concepts if c.id == concept_id)
            if concept.member_tag_ids:
                tag_id = concept.member_tag_ids[0]
                session.handle_tag_click(tag_id, 'positive')
                print(f"  👍 Liked tag in concept '{concept.label}'")

        for concept_id in dislike_concepts:
            concept = next(c for c in session.concepts if c.id == concept_id)
            if concept.member_tag_ids:
                tag_id = concept.member_tag_ids[0]
                session.handle_tag_click(tag_id, 'negative')
                print(f"  👎 Disliked tag in concept '{concept.label}'")

        # --------------------------------------------------------------------
        # Step 2: UI stabilization (debounced snapshot)
        # --------------------------------------------------------------------
        print(f"\n[Step 2] UI stabilization...")

        # Get current weights
        w_ui = session.get_current_weights_for_pbo()

        # Wait for debounce
        time.sleep(0.6)

        # Record stabilization
        recorded = refiner.on_ui_stabilize(w_ui)

        if recorded:
            print(f"  ✅ Snapshot recorded (candidate {refiner.last_snapshot_cid})")
        else:
            print(f"  ⏭️  Snapshot skipped (debounce/threshold)")

        # --------------------------------------------------------------------
        # Step 3: Generate proposals with PBO
        # --------------------------------------------------------------------
        print(f"\n[Step 3] Generating proposals with PBO...")

        negatives = session.get_negative_concept_ids()

        print(f"  Negative concepts: {len(negatives)}")
        print(f"  Current weights (top 3):")
        top_3_idx = np.argsort(-w_ui)[:3]
        for idx in top_3_idx:
            concept = session.concepts[idx]
            print(f"    {concept.label}: {w_ui[idx]:.3f}")

        # Fit and propose (only if we have enough data)
        fit_first = (round_num > 0)  # Only fit after first round

        proposals = refiner.propose_next_4(
            negatives=negatives,
            w_current=w_ui,
            fit_first=fit_first
        )

        print(f"\n  ✅ Generated {len(proposals)} proposals")
        for i, w in enumerate(proposals):
            top_concept_idx = np.argmax(w)
            top_concept = session.concepts[top_concept_idx]
            print(f"    Proposal {i+1}: dominant='{top_concept.label}' ({w[top_concept_idx]:.3f})")

        # --------------------------------------------------------------------
        # Step 4: User picks favorite (simulate)
        # --------------------------------------------------------------------
        print(f"\n[Step 4] Simulating favorite selection...")

        # Create mock image IDs for this round
        round_image_ids = [f"pbo_round_{round_num}_img_{i}" for i in range(4)]

        # Build incidence matrix for these images (based on proposals)
        round_incidence = {}
        for i, (img_id, w_prop) in enumerate(zip(round_image_ids, proposals)):
            round_incidence[img_id] = {}
            # Distribute tags based on weights
            for j, concept in enumerate(session.concepts):
                if w_prop[j] > 0.1:  # Only include significant concepts
                    round_incidence[img_id][concept.id] = max(1, int(w_prop[j] * 10))

        # Pick "best" proposal (highest weight on liked concepts)
        scores = []
        for w_prop in proposals:
            score = sum(w_prop[session.concepts.index(c)]
                       for c in session.concepts
                       if c.id in like_concepts)
            scores.append(score)

        favorite_idx = np.argmax(scores)
        favorite_image_id = round_image_ids[favorite_idx]

        print(f"  Selected image {favorite_idx} as favorite")

        # Record favorite
        refiner.on_favorite(
            favorite_image_id=favorite_image_id,
            all_image_ids=round_image_ids,
            incidence_matrix=round_incidence
        )

        print(f"  ✅ Recorded {len(round_image_ids) - 1} strong duels")

        # --------------------------------------------------------------------
        # Step 5: Check PBO state
        # --------------------------------------------------------------------
        print(f"\n[Step 5] PBO State:")
        pbo_state = refiner.pbo.to_dict()
        print(f"  Candidates: {pbo_state['num_candidates']}")
        print(f"  Duels: {pbo_state['num_duels']}")
        print(f"  GP fitted: {pbo_state['fitted']}")

        # Get best candidate
        if refiner.pbo.fitted and refiner.pbo.best() is not None:
            w_best = refiner.pbo.best()
            best_concept_idx = np.argmax(w_best)
            best_concept = session.concepts[best_concept_idx]
            print(f"  Current best: '{best_concept.label}' ({w_best[best_concept_idx]:.3f})")

    # ========================================================================
    # FINAL VERIFICATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    final_state = refiner.pbo.to_dict()

    print(f"\n📊 Final PBO State:")
    print(f"  Total candidates: {final_state['num_candidates']}")
    print(f"  Total duels: {final_state['num_duels']}")
    print(f"  Strong duels (favorites): {num_rounds * 3}")  # 3 per round
    print(f"  GP fitted: {final_state['fitted']}")

    # Verify acceptance criteria
    checks = []

    # Check 1: Snapshots recorded
    snapshot_count = len([c for c in refiner.pbo.candidates.values()
                         if 'cand_' in c.id])
    checks.append(("Snapshots recorded", snapshot_count > 0))

    # Check 2: Duels recorded
    checks.append(("Duels recorded", final_state['num_duels'] >= num_rounds * 3))

    # Check 3: GP fitted
    checks.append(("GP fitted successfully", final_state['fitted']))

    # Check 4: Can propose new candidates
    try:
        test_proposals = refiner.propose_next_4(fit_first=True)
        checks.append(("Can propose new candidates", len(test_proposals) == 4))
        checks.append(("Proposals on simplex",
                      all(abs(w.sum() - 1.0) < 0.01 for w in test_proposals)))
    except Exception as e:
        checks.append(("Can propose new candidates", False))
        print(f"  Error: {e}")

    # Check 5: Diversity
    if len(test_proposals) == 4:
        from pbo import compute_mixture_embedding
        Z = np.array([compute_mixture_embedding(w, refiner.MU) for w in test_proposals])
        max_cos = 0.0
        for i in range(len(Z)):
            for j in range(i+1, len(Z)):
                cos = np.dot(Z[i], Z[j])
                max_cos = max(max_cos, cos)
        checks.append(("Proposals are diverse (cos < 0.98)", max_cos < 0.98))

    print(f"\n✅ Acceptance Checks:")
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")

    all_passed = all(passed for _, passed in checks)

    if all_passed:
        print("\n" + "=" * 70)
        print("🎉 END-TO-END TEST PASSED!")
        print("=" * 70)
        print("\nAll systems operational. PBO integration is working correctly.")
        return True
    else:
        print("\n" + "=" * 70)
        print("⚠️  SOME CHECKS FAILED")
        print("=" * 70)
        failed = [name for name, passed in checks if not passed]
        print(f"Failed checks: {', '.join(failed)}")
        return False


if __name__ == "__main__":
    try:
        success = test_e2e_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
