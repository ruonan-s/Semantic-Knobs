# backend/test_sdxl_integration.py
# Integration tests for Stage 3: SDXL generation from concept mixtures.

import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.sdxl_integration import (
    concepts_to_sdxl_phrases,
    normalize_simplex,
    compute_gains,
    get_phrase_summary
)
from backend.sdxl_runner import SDXLRunner


def test_gain_mapping():
    """Test gain mapping formula produces correct range [0.7, 1.5]"""
    print("\n=== Test 1: Gain Mapping ===")

    # Test with different weight distributions
    test_cases = [
        ("Uniform", np.array([0.2, 0.2, 0.2, 0.2, 0.2])),
        ("Peaked", np.array([0.6, 0.2, 0.1, 0.05, 0.05])),
        ("Bimodal", np.array([0.4, 0.4, 0.1, 0.05, 0.05])),
    ]

    for name, w in test_cases:
        w_norm = normalize_simplex(w)
        gains = compute_gains(w_norm, lambda_scale=0.4)

        print(f"\n{name} distribution:")
        print(f"  Weights: {w_norm}")
        print(f"  Gains: {gains}")
        print(f"  Gain range: [{gains.min():.3f}, {gains.max():.3f}]")

        # Check that gains are in [0.7, 1.5]
        assert gains.min() >= 0.7 - 1e-6, f"Min gain {gains.min()} < 0.7"
        assert gains.max() <= 1.5 + 1e-6, f"Max gain {gains.max()} > 1.5"

    print("\n✅ Gain mapping test passed!")


def test_phrase_selection():
    """Test Top-K selection and deficit-based negatives"""
    print("\n=== Test 2: Phrase Selection ===")

    # Create mock concepts
    concepts = [
        {'id': 'c0', 'label': 'cozy'},
        {'id': 'c1', 'label': 'modern'},
        {'id': 'c2', 'label': 'minimalist'},
        {'id': 'c3', 'label': 'warm lighting'},
        {'id': 'c4', 'label': 'natural materials'},
    ]

    # Test mixture (emphasize cozy + warm)
    w = np.array([0.4, 0.2, 0.15, 0.2, 0.05])

    # Convert to phrases
    pos_phrases, neg_phrases = concepts_to_sdxl_phrases(
        w=w,
        concepts=concepts,
        top_k=4,
        num_negatives=2
    )

    print("\nInput weights:", w)
    print(f"\nPositive phrases ({len(pos_phrases)}):")
    for phrase, gain in pos_phrases:
        print(f"  {phrase}: gain={gain:.3f}")

    print(f"\nNegative phrases ({len(neg_phrases)}):")
    for phrase in neg_phrases:
        print(f"  {phrase}")

    # Assertions
    assert len(pos_phrases) == 4, f"Expected 4 positives, got {len(pos_phrases)}"
    assert len(neg_phrases) <= 2, f"Expected <=2 negatives, got {len(neg_phrases)}"

    # Check that highest weight is first
    assert pos_phrases[0][0] == 'cozy', f"Expected 'cozy' first, got '{pos_phrases[0][0]}'"

    print("\n✅ Phrase selection test passed!")


def test_edge_cases():
    """Test edge cases: single concept, all zeros, etc."""
    print("\n=== Test 3: Edge Cases ===")

    # Single concept
    concepts_1 = [{'id': 'c0', 'label': 'modern'}]
    w_1 = np.array([1.0])
    pos, neg = concepts_to_sdxl_phrases(w_1, concepts_1, top_k=10, num_negatives=3)
    print(f"\nSingle concept: pos={len(pos)}, neg={len(neg)}")
    assert len(pos) == 1
    assert len(neg) == 0  # No deficit negatives when K=1

    # All zeros (should return uniform)
    concepts_3 = [
        {'id': 'c0', 'label': 'a'},
        {'id': 'c1', 'label': 'b'},
        {'id': 'c2', 'label': 'c'},
    ]
    w_3 = np.array([0.0, 0.0, 0.0])
    pos, neg = concepts_to_sdxl_phrases(w_3, concepts_3, top_k=3, num_negatives=1)
    print(f"All zeros: pos={len(pos)}, neg={len(neg)}")
    assert len(pos) == 3

    # More top_k than concepts
    concepts_5 = [
        {'id': 'c0', 'label': 'a'},
        {'id': 'c1', 'label': 'b'},
    ]
    w_5 = np.array([0.6, 0.4])
    pos, neg = concepts_to_sdxl_phrases(w_5, concepts_5, top_k=10, num_negatives=3)
    print(f"top_k > K: pos={len(pos)} (should be 2)")
    assert len(pos) == 2

    print("\n✅ Edge cases test passed!")


def test_sdxl_generation():
    """Test end-to-end SDXL generation from concept mixture"""
    print("\n=== Test 4: SDXL Generation ===")

    # Create mock concepts (with dummy centroids)
    concepts = [
        {'id': 'c0', 'label': 'cozy', 'centroid': np.random.randn(10)},
        {'id': 'c1', 'label': 'modern', 'centroid': np.random.randn(10)},
        {'id': 'c2', 'label': 'minimalist', 'centroid': np.random.randn(10)},
        {'id': 'c3', 'label': 'warm lighting', 'centroid': np.random.randn(10)},
        {'id': 'c4', 'label': 'natural materials', 'centroid': np.random.randn(10)},
    ]

    # Test mixture (emphasize cozy + warm)
    w = np.array([0.4, 0.2, 0.15, 0.2, 0.05])

    # Convert to phrases (for display)
    pos, neg = concepts_to_sdxl_phrases(w, concepts, top_k=4, num_negatives=2)
    print("\n" + get_phrase_summary(pos, neg))

    # Initialize runner
    print("\n[Initializing SDXL Runner...]")
    print("Note: This will download the model if not already cached (~7GB)")
    print("      Set model_id to a local path if you have it pre-downloaded")

    try:
        runner = SDXLRunner(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            device=None,  # Auto-detect (cuda/mps/cpu)
            height=1024,
            width=1024,
            steps=30,
            guidance_scale=7.5
        )

        # Generate single image
        print("\n[Generating single image...]")
        image = runner.generate_from_mixture(
            w=w,
            concepts=concepts,
            seed=42,
            height=1024,
            width=1024,
            steps=30,
            verbose=True
        )

        # Save
        output_path = "/tmp/test_pbo_sdxl_single.png"
        image.save(output_path)
        print(f"\n✅ Single image saved to {output_path}")

        # Generate batch of 4 proposals
        print("\n[Generating batch of 4 proposals...]")

        # Create 4 diverse proposals
        proposals = [
            np.array([0.5, 0.3, 0.1, 0.05, 0.05]),  # Cozy-heavy
            np.array([0.1, 0.5, 0.3, 0.05, 0.05]),  # Modern-heavy
            np.array([0.2, 0.2, 0.3, 0.2, 0.1]),    # Balanced
            np.array([0.15, 0.15, 0.15, 0.4, 0.15]) # Warm lighting focus
        ]

        images = runner.generate_batch_from_proposals(
            proposals=proposals,
            concepts=concepts,
            seed_base=100,
            top_k=4,
            num_negatives=2,
            verbose=False  # Less verbose for batch
        )

        # Save batch
        for i, img in enumerate(images):
            output_path = f"/tmp/test_pbo_sdxl_batch_{i+1}.png"
            img.save(output_path)
            print(f"  Image {i+1} saved to {output_path}")

        print(f"\n✅ Batch generation complete ({len(images)} images)")

        print("\n" + "="*60)
        print("🎉 SDXL integration test PASSED!")
        print("="*60)
        print("\nGenerated images:")
        print("  Single: /tmp/test_pbo_sdxl_single.png")
        print("  Batch:  /tmp/test_pbo_sdxl_batch_[1-4].png")
        print("\nNext steps:")
        print("  1. Visually inspect images")
        print("  2. Verify they reflect weight mixtures")
        print("  3. Check for token truncation warnings")

    except Exception as e:
        print(f"\n❌ SDXL generation failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure conda env 'apl' is activated")
        print("  2. Check that torch and diffusers are installed")
        print("  3. Verify CUDA/MPS availability if using GPU")
        print("  4. Check model download completed successfully")
        raise


def main():
    """Run all tests"""
    print("="*60)
    print("Stage 3: SDXL Integration Tests")
    print("="*60)

    # Run unit tests first
    test_gain_mapping()
    test_phrase_selection()
    test_edge_cases()

    # Run integration test (requires SDXL model)
    print("\n" + "="*60)
    print("IMPORTANT: The following test will:")
    print("  - Download SDXL model (~7GB) if not cached")
    print("  - Require GPU (CUDA/MPS) or will be slow on CPU")
    print("  - Take several minutes to complete")
    print("="*60)

    response = input("\nProceed with SDXL generation test? [y/N]: ")
    if response.lower() in ['y', 'yes']:
        test_sdxl_generation()
    else:
        print("\nSkipping SDXL generation test.")
        print("To run later: python backend/test_sdxl_integration.py")


if __name__ == "__main__":
    main()
