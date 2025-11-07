#!/usr/bin/env python3
"""
Example: Stage-Specific Guidance Prompts for SDXL

This script demonstrates how stage guidance prompts are integrated
into the SDXL generation pipeline.
"""

import numpy as np
from backend.sdxl_prompts import get_guidance_prompt, STAGE_GUIDANCE_PROMPTS

# ============================================================================
# Example 1: Getting guidance prompts
# ============================================================================
print("=" * 70)
print("Example 1: Stage Guidance Prompts")
print("=" * 70)

for stage, guidance in STAGE_GUIDANCE_PROMPTS.items():
    print(f"\n{stage.upper()} Stage:")
    print(f"  Guidance: '{guidance}'")
    
    # Show that refinement stages work too
    refinement_stage = f"{stage}_refinement"
    refinement_guidance = get_guidance_prompt(refinement_stage)
    print(f"  {refinement_stage}: '{refinement_guidance}'")

# ============================================================================
# Example 2: Simulated phrase generation
# ============================================================================
print("\n" + "=" * 70)
print("Example 2: Phrase Generation with Guidance")
print("=" * 70)

# Simulate impression refinement
stage = "impression"
guidance = get_guidance_prompt(stage)

# Simulated concept phrases (would come from PBO)
concept_phrases = [
    ("neon_lighting", 1.5),
    ("urban_architecture", 1.2),
    ("dramatic_atmosphere", 1.1),
    ("metallic_textures", 1.0),
    ("glass_surfaces", 0.9),
]

print(f"\nStage: {stage}")
print(f"Guidance: '{guidance}'\n")

print("Phrases WITHOUT guidance:")
for i, (phrase, gain) in enumerate(concept_phrases, 1):
    print(f"  {i}. {phrase}: gain={gain:.3f}")

print("\nPhrases WITH guidance:")
# Prepend guidance
phrases_with_guidance = [(guidance, 1.5)] + concept_phrases
for i, (phrase, gain) in enumerate(phrases_with_guidance, 1):
    if i == 1:
        print(f"  {i}. {phrase}: gain={gain:.3f}  ← STAGE GUIDANCE")
    else:
        print(f"  {i}. {phrase}: gain={gain:.3f}")

# ============================================================================
# Example 3: Full prompt construction
# ============================================================================
print("\n" + "=" * 70)
print("Example 3: Full Prompt Examples")
print("=" * 70)

examples = [
    ("impression", ["neon_lighting", "urban", "dramatic", "modern"]),
    ("spatial", ["symmetrical_layout", "vertical_composition", "depth"]),
    ("objects", ["glass_structures", "metallic_surfaces", "organic"]),
    ("ambient", ["foggy_atmosphere", "soft_lighting", "mysterious"]),
]

for stage, concepts in examples:
    guidance = get_guidance_prompt(stage)
    full_prompt = f"{guidance} {', '.join(concepts)}"
    
    print(f"\n{stage.upper()}:")
    print(f"  Full prompt: '{full_prompt}'")

# ============================================================================
# Example 4: Integration with SDXL Runner (pseudocode)
# ============================================================================
print("\n" + "=" * 70)
print("Example 4: Integration Flow")
print("=" * 70)

print("""
Typical refinement flow with stage guidance:

1. StageRefiner initialized:
   refiner = StageRefiner(session_id, stage="impression", concepts, ...)

2. PBO proposes weight vectors:
   proposals = refiner.propose_next_4()
   # proposals = [w1, w2, w3, w4]

3. Generate images (guidance automatically added):
   images = refiner.generate_images_from_proposals(
       proposals=proposals,
       sdxl_runner=sdxl_runner,
       init_image=reference_image
   )
   
   Internally, for each proposal w:
   
   a) Convert weights to phrases:
      phrases = ["neon_lighting", "urban_architecture", ...]
   
   b) Add stage guidance:
      phrases = ["Update image aesthetic style to", ...phrases]
   
   c) Encode each phrase with CLIP:
      embeds = [CLIP(phrase) for phrase in phrases]
   
   d) Fuse weighted:
      fused_embed = weighted_sum(embeds, gains)
   
   e) Generate with SDXL:
      image = SDXL.img2img(reference, fused_embed)

4. User selects preferred image → PBO learns → Repeat
""")

print("\n" + "=" * 70)
print("✅ Stage guidance is now integrated into SDXL generation!")
print("=" * 70)

