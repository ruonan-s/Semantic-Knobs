"""
Cozy Sweep Helper - Within-Context Semantic Slider

Generate images by interpolating in CONCEPT WEIGHT SPACE (w) between
a neutral baseline and a learned semantic direction (e.g., "cozy").

This is the RIGHT way to do semantic interpolation:
- Work in concept space (not embedding space)
- Use existing SDXLRunner + EmbedFuser pipeline
- Let SDXL handle embeddings correctly

Usage:
    runner = SDXLRunner(...)
    w_neutral = np.ones(K) / K  # or learned baseline
    w_cozy = learned_cozy_weights  # from PBO
    
    sweep = generate_cozy_sweep(
        runner=runner,
        w_neutral=w_neutral,
        w_cozy=w_cozy,
        concepts=concepts,
        descriptor="bedroom",
        alphas=[-0.2, 0.0, 0.2, 0.4]
    )
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import os
from datetime import datetime

from sdxl_runner import SDXLRunner


def generate_cozy_sweep(
    runner: SDXLRunner,
    w_cozy: np.ndarray,
    concepts: List[Dict],
    descriptor: str = "a bedroom interior, wide angle, natural lighting",
    alphas: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    seed: int = 42,
    stage: str | None = "impression",
    output_dir: str | None = None,
    prefix: str = "cozy_sweep",
    verbose: bool = True,
    **kwargs,
) -> List[Tuple[float, Image.Image, str]]:
    """
    Generate a cozy sweep with alpha interpolation.
    
    Formula: e_total = (1-alpha) * e_desc + alpha * e_tags
    
    Where:
        e_desc = embedding of descriptor (fixed)
        e_tags = weighted sum of concept embeddings using w_cozy (fixed relative importances)
        alpha = interpolation factor (0 = pure descriptor, 1 = pure tags)
    
    Key insight: w_cozy is normalized ONCE and stays fixed. Alpha interpolates
    between descriptor and cozy tags.
    
    Args:
        runner: SDXLRunner instance
        w_cozy: Learned cozy weight vector (K,) from PBO (will be normalized once)
        concepts: List of concept dicts with 'label' field
        descriptor: Text that anchors scene type (e.g., "a bedroom interior")
        alphas: Strength factors to test (alpha=0 → descriptor only, alpha=1 → full cozy)
        seed: Base seed (each alpha gets seed + i)
        stage: SDXL stage label for strength config
        output_dir: Directory to save images (None = don't save)
        prefix: Filename prefix for saved images
        verbose: Print progress
        **kwargs: Passed through to SDXLRunner.generate_from_mixture()
    
    Returns:
        List of (alpha, image, filepath) tuples
        
    Example:
        >>> runner = SDXLRunner()
        >>> w_cozy = np.array([0.2, 0.19, 0.18, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02])
        >>> sweep = generate_cozy_sweep(
        ...     runner, w_cozy,
        ...     concepts=bedroom_concepts,
        ...     descriptor="a bedroom interior, wide angle, natural lighting",
        ...     alphas=[0.0, 0.25, 0.5, 0.75, 1.0]
        ... )
        >>> # Show to user in random order, track which alpha they prefer
    """
    K = w_cozy.shape[0]
    assert len(concepts) == K, f"Concept count mismatch: {len(concepts)} vs {K}"
    
    
    
    # Normalize w_cozy ONCE - these are the fixed relative importances
    w_cozy_norm = w_cozy / (w_cozy.sum() + 1e-8)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"COZY SWEEP - Alpha Interpolation")
        print(f"Formula: e = (1-α)*e_desc + α*e_tags")
        print(f"{'='*70}")
        print(f"Descriptor: '{descriptor}'")
        print(f"Concept space dim: {K}")
        print(f"Alphas: {alphas}")
        print(f"Stage: {stage}")
        print(f"{'='*70}\n")
        print(f"Alpha = 0: pure descriptor")
        print(f"Alpha = 1: pure cozy tags")
        print(f"\nTop cozy concepts (fixed relative importances):")
        top_indices = np.argsort(w_cozy_norm)[::-1][:5]
        for idx in top_indices:
            if idx < len(concepts):
                print(f"  {concepts[idx]['label']}: {w_cozy_norm[idx]:.4f}")
    
    # Setup output directory if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate images at each alpha level
    results = []
    
    for i, alpha in enumerate(alphas):
        if verbose:
            print(f"\n[{i+1}/{len(alphas)}] Alpha = {alpha:.2f}")
            print(f"  Formula: e_total = {1-alpha:.2f}*e_desc + {alpha:.2f}*e_tags")
            print(f"  Top 3 cozy concepts (fixed weights):")
            top_3 = np.argsort(w_cozy_norm)[::-1][:3]
            for idx in top_3:
                if idx < len(concepts):
                    print(f"    {concepts[idx]['label']}: {w_cozy_norm[idx]:.3f}")
        
        # Generate image using SDXLRunner
        # w_cozy_norm stays fixed, alpha interpolates between descriptor and tags
        # Fuser computes: e_total = (1-alpha)*e_desc + alpha*(Σ w_cozy_norm[i] * e_tag[i])
        img = runner.generate_from_mixture(
            w=w_cozy_norm,  # Fixed relative importances
            concepts=concepts,
            seed=seed + i,
            descriptor=descriptor,
            alpha=alpha,  # Strength factor
            stage=stage,
            verbose=verbose,
            **kwargs,
        )
        
        # Save if requested
        if output_dir:
            new_folder = os.path.join(output_dir, descriptor)
            os.makedirs(new_folder, exist_ok=True)
            filename = f"{prefix}_{timestamp}_alpha{alpha:+.2f}_seed{seed+i}.png"
            filepath = os.path.join(new_folder, filename)
            img.save(filepath)
            if verbose:
                print(f"  ✓ Saved: {filepath}")
        else:
            filepath = None
        
        results.append((alpha, img, filepath))
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ Generated {len(results)} images")
        print(f"{'='*70}\n")
    
    return results


def generate_cross_context_sweep(
    runner: SDXLRunner,
    w_learned: np.ndarray,
    concepts: List[Dict],
    descriptors: List[str],
    alphas: List[float] = [0.0, 0.5, 1.0],
    seed: int = 42,
    output_dir: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, List[Tuple[float, Image.Image, str]]]:
    """
    Cross-context transfer test.
    
    Test if learned semantic direction (e.g., "cozy") transfers from
    the original context (e.g., bedroom) to new contexts (e.g., living room, cafe).
    
    For each context, alpha interpolates between:
        alpha = 0.0 → pure descriptor (baseline)
        alpha = 1.0 → pure learned cozy tags
    
    Args:
        runner: SDXLRunner instance
        w_learned: Learned personalized weights (e.g., from bedroom PBO)
        concepts: Concept bank (same across contexts)
        descriptors: List of scene descriptors to test
            e.g., ["a bedroom interior", "a living room interior", "a cafe interior"]
        alphas: Interpolation levels (0 = descriptor-only, 1 = full learned)
        seed: Base seed
        output_dir: Output directory (creates subdirs per context)
        verbose: Print progress
        **kwargs: Passed to SDXLRunner
    
    Returns:
        Dict mapping descriptor → list of (alpha, image, filepath) tuples
        
    Example:
        >>> # Learn "cozy" in bedroom context
        >>> w_cozy_bedroom = run_pbo_for_bedroom()
        >>> 
        >>> # Test transfer to other contexts
        >>> results = generate_cross_context_sweep(
        ...     runner,
        ...     w_learned=w_cozy_bedroom,
        ...     concepts=concepts,
        ...     descriptors=[
        ...         "a bedroom interior, wide angle, natural lighting",
        ...         "a living room interior, wide angle, natural lighting",
        ...         "a coffee shop interior, wide angle, natural lighting"
        ...     ],
        ...     alphas=[0.0, 0.5, 1.0]
        ... )
        >>> 
        >>> # Show to user: does "cozy bedroom learning" transfer to living room / cafe?
    """
    all_results = {}
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"CROSS-CONTEXT TRANSFER TEST")
        print(f"Formula: e = (1-α)*e_desc + α*e_tags")
        print(f"{'='*70}")
        print(f"Testing {len(descriptors)} contexts with {len(alphas)} alpha levels")
        print(f"Alpha = 0: pure descriptor (baseline)")
        print(f"Alpha = 1: pure learned cozy tags")
        print(f"{'='*70}\n")
    
    for ctx_idx, descriptor in enumerate(descriptors):
        context_name = f"context_{ctx_idx+1}"
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Context {ctx_idx+1}/{len(descriptors)}: '{descriptor}'")
            print(f"{'='*50}")
        
        # Create context-specific output dir
        ctx_output_dir = None
        if output_dir:
            ctx_output_dir = os.path.join(output_dir, context_name)
        
        # Run sweep for this context
        results = generate_cozy_sweep(
            runner=runner,
            w_cozy=w_learned,
            concepts=concepts,
            descriptor=descriptor,
            alphas=alphas,
            seed=seed + ctx_idx * 100,  # Different seed per context
            output_dir=ctx_output_dir,
            prefix=f"transfer_{context_name}",
            verbose=verbose,
            **kwargs,
        )
        
        all_results[descriptor] = results
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ Cross-context test complete")
        print(f"Tested {len(descriptors)} contexts")
        print(f"{'='*70}\n")
    
    return all_results


def main():
    """
    Example usage of cozy sweep.
    
    CUSTOMIZE THE INPUTS BELOW:
    - w_neutral: Your baseline concept weights
    - w_cozy: Your learned "cozy" weights (from PBO or manual tuning)
    - concepts: Your concept bank
    - descriptor: Scene description
    """
    
    # ========== USER INPUTS ==========
    
    # Example concept bank (REPLACE WITH YOUR ACTUAL CONCEPTS)
    # Note: Remove redundant "bedroom with" prefix since descriptor already says "bedroom"
    
    
    location = "home office"
    concepts = [
        {"id": "c0", "label": f"{location} with large windows"},
        {"id": "c1", "label": f"{location} with natural fibers"},
        {"id": "c2", "label": f"{location} with simple decor"},
        {"id": "c3", "label": f"{location} with cozy seating"},
        {"id": "c4", "label": f"{location} with warm color palette"},
        {"id": "c5", "label": f"{location} with abstract wall art"},
        {"id": "c6", "label": f"{location} with wooden furniture"},
        {"id": "c7", "label": f"{location} with plush cushions"},
        {"id": "c8", "label": f"{location} with sheer curtains"},
        {"id": "c9", "label": f"{location} with indoor plants"},
    ]
    
    K = len(concepts)
    
    # Learned "cozy": example weights (REPLACE WITH YOUR LEARNED WEIGHTS from PBO)
    # Higher weights = more important for "cozy"
    w_cozy = np.array([0.4280, 0.3340, 0.0941, 0.0610, 0.0271, 0.0224, 0.0127, 0.0085, 0.0073, 0.0048])
    
    # Normalize to sum to 1
    w_cozy = w_cozy / w_cozy.sum()
    
    # Scene descriptor (anchor the scene type)
    # Be explicit: scene type, angle, lighting, quality
    descriptor = "f{adjective} {location}"
    
    # Alpha levels to test
    # alpha = 0: descriptor-only baseline (tags essentially off)
    # alpha = 1: descriptor + full learned cozy weights
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Output settings
    output_dir = "outputs/cozy_sweep"
    seed = 42
    
    # ========== END USER INPUTS ==========
    
    print(f"Concept bank: {K} concepts")
    print(f"Cozy weights: {w_cozy}")
    print(f"\nTop cozy concepts:")
    top_indices = np.argsort(w_cozy)[::-1]
    for idx in top_indices:
        print(f"  {concepts[idx]['label']}: {w_cozy[idx]:.4f}")
    
    # Initialize runner
    runner = SDXLRunner(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device=None,  # Auto-detect
        height=1024,
        width=1024,
        steps=30,
        guidance_scale=7.5
    )
    
    # Generate sweep
    results = generate_cozy_sweep(
        runner=runner,
        w_cozy=w_cozy,
        concepts=concepts,
        descriptor=descriptor,
        alphas=alphas,
        seed=seed,
        output_dir=output_dir,
        verbose=True
    )
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for alpha, img, path in results:
        print(f"  Alpha {alpha:.2f}: {path if path else '(not saved)'}")
    print("="*70)
    print("\n✅ Done! For within-context test:")
    print("  1. Show these images to users in random order (don't reveal alpha)")
    print("  2. Ask: 'Rank from least to most cozy'")
    print("  3. Check if ranking correlates with alpha")
    print("\nExpected: Alpha 0 = generic bedroom, Alpha 1 = your learned cozy")


if __name__ == "__main__":
    main()

