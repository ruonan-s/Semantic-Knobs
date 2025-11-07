# Stage-Specific Guidance Prompts for SDXL

## Overview

Stage-specific guidance prompts have been integrated into the SDXL generation pipeline to provide contextual direction for each refinement stage. These prompts are automatically prepended to the fused concept embeddings.

## Implementation

### Files Modified

1. **`backend/sdxl_prompts.py`** - New file containing stage guidance prompts
2. **`backend/sdxl_runner.py`** - Updated to accept `stage` parameter and prepend guidance
3. **`backend/stage_refiner.py`** - Updated to pass stage information to SDXL runner

### How It Works

```python
# Stage guidance prompts (from sdxl_prompts.py)
STAGE_GUIDANCE_PROMPTS = {
    "impression": "Update image aesthetic style to",
    "spatial": "Update image spatial structure to",
    "objects": "Update image objects and materials to",
    "ambient": "Update image ambient to"
}

# Example flow for impression_refinement:
# 1. PBO proposes weight vector: w = [0.05, 0.50, 0.10, 0.20, 0.05, 0.10]
# 2. Convert to phrases:
#    - "neon_lighting" (gain=1.5)
#    - "urban_architecture" (gain=1.2)
#    - "dramatic_atmosphere" (gain=1.1)
#    - ...
# 3. Prepend guidance:
#    - "Update image aesthetic style to" (gain=1.5)  ← ADDED
#    - "neon_lighting" (gain=1.5)
#    - "urban_architecture" (gain=1.2)
#    - ...
# 4. Fuse all phrases into single embedding
# 5. Generate with SDXL img2img
```

### Phrase List Example

**Before (without guidance):**
```
Positive phrases (10):
  neon_lighting: gain=1.500
  urban_architecture: gain=1.234
  dramatic_atmosphere: gain=1.123
  metallic_textures: gain=1.056
  glass_surfaces: gain=0.987
  modern_design: gain=0.934
  reflective_materials: gain=0.876
  night_scene: gain=0.823
  blue_tones: gain=0.789
  detailed_structures: gain=0.745
```

**After (with guidance):**
```
Positive phrases (11):
  Update image aesthetic style to: gain=1.500  ← STAGE GUIDANCE
  neon_lighting: gain=1.500
  urban_architecture: gain=1.234
  dramatic_atmosphere: gain=1.123
  metallic_textures: gain=1.056
  glass_surfaces: gain=0.987
  modern_design: gain=0.934
  reflective_materials: gain=0.876
  night_scene: gain=0.823
  blue_tones: gain=0.789
  detailed_structures: gain=0.745
```

## Stage-Specific Guidance

### Impression Stage
**Guidance:** `"Update image aesthetic style to"`

**Focus:** Overall visual feel, mood, artistic direction
- Color palettes
- Lighting mood
- Artistic style
- Emotional tone

**Example Result:**
```
"Update image aesthetic style to neon_lighting, urban_architecture, dramatic_atmosphere..."
```

### Spatial Stage
**Guidance:** `"Update image spatial structure to"`

**Focus:** Layout, composition, spatial relationships
- Architectural elements
- Depth and perspective
- Compositional balance
- Spatial organization

**Example Result:**
```
"Update image spatial structure to symmetrical_layout, vertical_composition, layered_depth..."
```

### Objects Stage
**Guidance:** `"Update image objects and materials to"`

**Focus:** Specific objects, materials, textures
- Object types and details
- Material properties
- Surface textures
- Physical characteristics

**Example Result:**
```
"Update image objects and materials to glass_structures, metallic_surfaces, organic_elements..."
```

### Ambient Stage
**Guidance:** `"Update image ambient to"`

**Focus:** Atmosphere, environmental conditions
- Weather effects
- Atmospheric conditions
- Environmental mood
- Ambient lighting

**Example Result:**
```
"Update image ambient to foggy_atmosphere, soft_lighting, mysterious_mood..."
```

## Technical Details

### Gain Value
The guidance prompt is added with **gain=1.5** (maximum gain value), ensuring it has strong influence on the final embedding.

### Embedding Fusion
The guidance is encoded by SDXL's CLIP encoders along with other phrases and weighted-summed:

```python
# Pseudocode for fusion
guidance_embed = CLIP_encode("Update image aesthetic style to")  # (1, 77, 2048)
concept_embeds = [CLIP_encode(phrase) for phrase in concepts]    # List of (1, 77, 2048)

# Weighted sum
fused = (1.5 * guidance_embed + 1.5 * embed_1 + 1.2 * embed_2 + ...) / sum(gains)
```

### Stage Name Handling
The `get_guidance_prompt()` function automatically handles both base stage names and refinement stage names:

```python
get_guidance_prompt("impression")             # → "Update image aesthetic style to"
get_guidance_prompt("impression_refinement")  # → "Update image aesthetic style to"
get_guidance_prompt("spatial")                # → "Update image spatial structure to"
get_guidance_prompt("spatial_refinement")     # → "Update image spatial structure to"
```

## Benefits

1. **Contextual Direction**: SDXL receives clear instructions about what aspect to modify
2. **Stage Consistency**: Each stage focuses on its specific domain
3. **Improved Convergence**: Guidance helps PBO learn more effectively within stage constraints
4. **Better Results**: Generated images align better with stage objectives

## Usage in Code

### Automatic (Recommended)
When using `StageRefiner.generate_images_from_proposals()`, the stage is automatically passed:

```python
refiner = StageRefiner(session_id, stage="impression", concepts=concepts, ...)
proposals = refiner.propose_next_4()
images = refiner.generate_images_from_proposals(proposals, sdxl_runner)
# Stage guidance automatically included
```

### Manual
When calling `SDXLRunner.generate_from_mixture()` directly:

```python
runner = SDXLRunner()
image = runner.generate_from_mixture(
    w=weight_vector,
    concepts=concepts,
    stage="impression",  # Optional: adds guidance
    init_image=reference_image,
    strength=0.75
)
```

### Optional
If no stage is provided, generation works without guidance:

```python
image = runner.generate_from_mixture(
    w=weight_vector,
    concepts=concepts
    # No stage parameter = no guidance
)
```

## Testing

To verify the guidance is being used, look for this in the logs:

```
[SDXLRunner] Converting mixture to phrases (top_k=10, num_negatives=3)...
[SDXLRunner] Added stage guidance: 'Update image aesthetic style to'
  Positive phrases (11):
    Update image aesthetic style to: gain=1.500
    neon_lighting: gain=1.500
    ...
```

## Customization

To modify or add guidance prompts, edit `backend/sdxl_prompts.py`:

```python
STAGE_GUIDANCE_PROMPTS = {
    "impression": "Update image aesthetic style to",
    "spatial": "Update image spatial structure to",
    "objects": "Update image objects and materials to",
    "ambient": "Update image ambient to",
    "custom_stage": "Your custom guidance here"  # Add new stages
}
```

## Summary

Stage-specific guidance prompts provide contextual direction to SDXL, ensuring that refinements focus on the appropriate aspect of the design:
- **Impression**: Aesthetic style and mood
- **Spatial**: Layout and composition
- **Objects**: Materials and objects
- **Ambient**: Atmosphere and environment

This makes the PBO learning more effective and the generated images more aligned with each stage's purpose.

