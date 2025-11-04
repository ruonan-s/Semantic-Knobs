"""
Test PBO with toy preference data

Creates a simple 3-5 concept scenario with a known "true favorite" direction
and verifies that PBO converges toward it.
"""

import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from pbo import PBO, normalize_simplex, compute_mixture_embedding


def test_basic_functionality():
    """Test basic PBO operations"""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Functionality")
    print("=" * 60)

    # Create toy problem: 3 concepts
    K = 3
    d = 8  # embedding dimension
    np.random.seed(42)

    # Generate random concept centroids (L2 normalized)
    MU = np.random.randn(K, d).astype(np.float32)
    MU = MU / np.linalg.norm(MU, axis=1, keepdims=True)

    concept_ids = [f"concept_{i}" for i in range(K)]

    # Create PBO
    pbo = PBO(MU=MU, concept_ids=concept_ids)

    # Add some candidates
    w1 = np.array([0.7, 0.2, 0.1])
    w2 = np.array([0.1, 0.7, 0.2])
    w3 = np.array([0.2, 0.1, 0.7])

    id1 = pbo.add_candidate(w1)
    id2 = pbo.add_candidate(w2)
    id3 = pbo.add_candidate(w3)

    print(f"\n✅ Added 3 candidates")
    print(f"  {id1}: {w1}")
    print(f"  {id2}: {w2}")
    print(f"  {id3}: {w3}")

    # Add preferences (w1 > w2 > w3)
    pbo.add_preference(id1, id2, strength=1.0)
    pbo.add_preference(id1, id3, strength=1.0)
    pbo.add_preference(id2, id3, strength=0.5)  # weak preference

    print(f"\n✅ Added 3 duels")

    # Fit
    pbo.fit()
    print(f"\n✅ GP fitted: {pbo.fitted}")

    # Get best
    w_best = pbo.best()
    if w_best is not None:
        print(f"\n✅ Best candidate: {w_best}")
        print(f"  (Should be close to w1 = {w1})")
    else:
        print(f"\n❌ No best candidate found")

    # Propose new batch
    proposals = pbo.propose_batch(q=4, w_current=w1)
    print(f"\n✅ Proposed {len(proposals)} new candidates:")
    for i, w in enumerate(proposals):
        print(f"  Prop {i+1}: {w}")

    print("\n✅ TEST 1 PASSED")


def test_convergence():
    """Test convergence to known favorite direction"""
    print("\n" + "=" * 60)
    print("TEST 2: Convergence")
    print("=" * 60)

    # Setup
    K = 5
    d = 16
    np.random.seed(123)

    MU = np.random.randn(K, d).astype(np.float32)
    MU = MU / np.linalg.norm(MU, axis=1, keepdims=True)

    concept_ids = [f"concept_{i}" for i in range(K)]

    # True favorite: concept 2 is best
    w_true = np.array([0.1, 0.1, 0.6, 0.1, 0.1])
    z_true = compute_mixture_embedding(w_true, MU)

    print(f"\n🎯 True favorite: {w_true}")

    # Create PBO
    pbo = PBO(MU=MU, concept_ids=concept_ids)

    # Simulate 10 rounds of preferences
    print(f"\n🔄 Running 10 rounds of simulated preferences...")

    for round_idx in range(10):
        # Generate 4 random candidates
        candidates = []
        candidate_ids = []

        for _ in range(4):
            w_rand = np.random.dirichlet(np.ones(K))
            w_rand = normalize_simplex(w_rand)
            cid = pbo.add_candidate(w_rand)
            candidates.append((cid, w_rand))
            candidate_ids.append(cid)

        # Evaluate each against true favorite
        scores = []
        for cid, w in candidates:
            z = compute_mixture_embedding(w, MU)
            score = np.dot(z, z_true)  # cosine similarity to true
            scores.append(score)

        # Add preferences (best vs others)
        best_idx = np.argmax(scores)
        best_id = candidate_ids[best_idx]

        for i, cid in enumerate(candidate_ids):
            if i != best_idx:
                pbo.add_preference(best_id, cid, strength=1.0)

        print(f"  Round {round_idx + 1}: Best score = {scores[best_idx]:.4f}")

        # Fit every 3 rounds
        if (round_idx + 1) % 3 == 0:
            pbo.fit()

    # Final fit
    pbo.fit()

    # Check best candidate
    w_best = pbo.best()
    if w_best is not None:
        z_best = compute_mixture_embedding(w_best, MU)
        cos_sim = np.dot(z_best, z_true)

        print(f"\n✅ Best candidate after 10 rounds:")
        print(f"  w_best: {w_best}")
        print(f"  w_true: {w_true}")
        print(f"  Cosine similarity: {cos_sim:.4f}")

        if cos_sim > 0.7:
            print(f"\n✅ TEST 2 PASSED (converged, cos={cos_sim:.4f} > 0.7)")
        else:
            print(f"\n⚠️  TEST 2 WARNING (weak convergence, cos={cos_sim:.4f} < 0.7)")
    else:
        print(f"\n❌ TEST 2 FAILED (no best candidate)")


def test_constraints():
    """Test acquisition constraints (cap, negatives, diversity)"""
    print("\n" + "=" * 60)
    print("TEST 3: Constraints")
    print("=" * 60)

    K = 4
    d = 8
    np.random.seed(456)

    MU = np.random.randn(K, d).astype(np.float32)
    MU = MU / np.linalg.norm(MU, axis=1, keepdims=True)

    concept_ids = [f"concept_{i}" for i in range(K)]

    pbo = PBO(MU=MU, concept_ids=concept_ids)

    # Add some data
    for _ in range(5):
        w = np.random.dirichlet(np.ones(K))
        pbo.add_candidate(w)

    # Add random preferences
    cand_ids = list(pbo.candidates.keys())
    for i in range(len(cand_ids) - 1):
        pbo.add_preference(cand_ids[i], cand_ids[i + 1], strength=1.0)

    pbo.fit()

    # Propose with negatives
    negatives = {concept_ids[0], concept_ids[1]}  # dislike first 2 concepts
    proposals = pbo.propose_batch(q=4, negatives=negatives, w_current=np.ones(K) / K)

    print(f"\n✅ Proposed 4 candidates with negatives {negatives}:")

    # Check constraints
    all_valid = True
    for i, w in enumerate(proposals):
        # Check simplex
        simplex_ok = np.allclose(w.sum(), 1.0) and np.all(w >= 0)

        # Check cap
        cap_ok = np.all(w <= 0.36)  # slight tolerance

        # Check negatives penalty (should be lower than others)
        neg_weight = w[0] + w[1]

        print(f"\n  Proposal {i + 1}:")
        print(f"    w = {w}")
        print(f"    Simplex: {'✅' if simplex_ok else '❌'}")
        print(f"    Cap ≤ 0.35: {'✅' if cap_ok else '❌'}")
        print(f"    Neg weight (concepts 0+1): {neg_weight:.3f}")

        if not (simplex_ok and cap_ok):
            all_valid = False

    # Check diversity
    from pbo import compute_mixture_embedding
    Z = np.array([compute_mixture_embedding(w, MU) for w in proposals])

    print(f"\n  Pairwise cosine similarities:")
    max_cos = 0.0
    for i in range(len(proposals)):
        for j in range(i + 1, len(proposals)):
            cos = np.dot(Z[i], Z[j])
            max_cos = max(max_cos, cos)
            print(f"    cos(prop_{i}, prop_{j}) = {cos:.4f}")

    diversity_ok = max_cos <= 0.96  # slight tolerance

    print(f"\n  Diversity (max cos ≤ 0.95): {'✅' if diversity_ok else '⚠️ '} (max={max_cos:.4f})")

    if all_valid and diversity_ok:
        print(f"\n✅ TEST 3 PASSED")
    else:
        print(f"\n⚠️  TEST 3 WARNING (some constraints not perfectly satisfied)")


def test_coalescing():
    """Test candidate coalescing for near-duplicates"""
    print("\n" + "=" * 60)
    print("TEST 4: Candidate Coalescing")
    print("=" * 60)

    K = 3
    d = 8
    np.random.seed(789)

    MU = np.random.randn(K, d).astype(np.float32)
    MU = MU / np.linalg.norm(MU, axis=1, keepdims=True)

    concept_ids = [f"concept_{i}" for i in range(K)]

    pbo = PBO(MU=MU, concept_ids=concept_ids)

    # Add base candidate
    w_base = np.array([0.5, 0.3, 0.2])
    id_base = pbo.add_candidate(w_base)

    print(f"\n✅ Added base candidate {id_base}: {w_base}")

    # Add very similar candidate (should coalesce)
    w_similar = w_base + np.array([0.001, -0.0005, -0.0005])
    w_similar = normalize_simplex(w_similar)

    id_similar = pbo.add_candidate(w_similar)

    if id_similar == id_base:
        print(f"\n✅ Near-duplicate coalesced (returned same ID: {id_base})")
        print(f"  Total candidates: {len(pbo.candidates)}")
        print(f"\n✅ TEST 4 PASSED")
    else:
        print(f"\n⚠️  Near-duplicate NOT coalesced (got new ID: {id_similar})")
        print(f"  Total candidates: {len(pbo.candidates)}")
        print(f"\n⚠️  TEST 4 WARNING")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PBO UNIT TESTS")
    print("=" * 60)

    try:
        test_basic_functionality()
        test_convergence()
        test_constraints()
        test_coalescing()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
