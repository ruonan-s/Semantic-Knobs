# Changes Summary: Descriptor-Based Prompting

## Overview
Replaced stage-specific guidance prompts with user descriptor for SDXL generation. Added configurable img2img strength values per stage.

## Files Modified

### 1. **Created: `backend/sdxl_config.py`** (NEW FILE)
- Configuration file for stage-specific img2img strength values
- All stages default to `0.75` strength (adjustable)
- Function `get_stage_strength(stage)` to retrieve strength values
- Automatically strips `_refinement` suffix from stage names

**Default Strengths:**
```python
STAGE_IMG2IMG_STRENGTH = {
    "impression": 0.75,
    "spatial": 0.75,
    "objects": 0.75,
    "ambient": 0.75,
}
```

### 2. **Modified: `backend/sdxl_runner.py`**
- **Removed**: Import of `get_guidance_prompt` from `sdxl_prompts.py`
- **Added**: Import of `get_stage_strength` from `sdxl_config.py`
- **Updated** `generate_from_mixture()`:
  - Added `descriptor` parameter (string, optional)
  - Changed `strength` parameter to optional (defaults to config value)
  - Removed guidance prompt prepending logic
  - Added descriptor prepending logic with gain 1.5
  - Strength now auto-loaded from config if not provided

**Old behavior:**
```python
# Prepended: "Update image aesthetic style to" (from guidance prompts)
pos_phrases = [(guidance_prompt, 1.5)] + pos_phrases
```

**New behavior:**
```python
# Prepends: "A comfortable space for reading" (from user descriptor)
if descriptor:
    pos_phrases = [(descriptor, 1.5)] + pos_phrases
```

### 3. **Modified: `backend/server.py`**
Updated 4 locations where `generate_images_from_proposals()` is called:

#### Location 1: `/api/feedback` (line ~1731)
- Uses `descriptor` from session object
- Passes to `generate_images_from_proposals()`

#### Location 2: `/api/generate-stage-refinement` (line ~3406)
- Loads descriptor from `preferences.json`
- Passes to `generate_images_from_proposals()`

#### Location 3: `/api/pbo/generate` (line ~3849)
- Loads descriptor from `preferences.json`
- Passes to `generate_images_from_proposals()`

#### Location 4: `/api/pbo/refine-next-round` (line ~3990)
- Loads descriptor from `preferences.json` (already loading file)
- Passes to `generate_images_from_proposals()`

**All occurrences changed from:**
```python
pil_images = refiner.generate_images_from_proposals(
    ...,
    strength=0.75  # hardcoded
)
```

**To:**
```python
pil_images = refiner.generate_images_from_proposals(
    ...,
    descriptor=descriptor  # from preferences.json or session
)
```

### 4. **Modified: `backend/stage_refiner.py`**
- Updated `generate_images_from_proposals()` docstring
- Clarified that `**kwargs` accepts `descriptor`, `init_image`, `strength`, `verbose`
- Updated comment: stage parameter now "for loading strength from config" instead of "context guidance"
- No functional changes (kwargs already passed through)

### 5. **Not Modified: `backend/sdxl_prompts.py`**
- File still exists but is no longer used
- Can be safely deleted if desired
- Contains the old guidance prompts that are now deprecated

## How It Works Now

### Prompt Construction Flow:
```
1. User provides: "A comfortable space for reading" (saved in preferences.json)
2. PBO generates concept weights: [cozy: 0.5, modern: 0.3, warm: 0.2, ...]
3. Top concepts selected: [("cozy", 1.5), ("warm", 1.2), ("modern", 1.0)]
4. Descriptor prepended: [("A comfortable space for reading", 1.5), ("cozy", 1.5), ...]
5. Embeddings fused with weights
6. SDXL generates image
```

### Strength Configuration:
```
1. Check if strength explicitly provided → use it
2. Else, load from config: get_stage_strength(stage)
3. Config looks up: "impression" → 0.75
4. Pass to img2img pipeline
```

## Benefits

1. **More Context**: User's original intent is directly embedded in prompt
2. **Consistency**: Same descriptor used across all refinement stages
3. **Clarity**: "A comfortable space for reading" is clearer than "Update image aesthetic style to"
4. **Configurability**: Easy to adjust img2img strength per stage without code changes
5. **Semantic**: Descriptor has real semantic meaning vs. instruction words

## Testing Recommendations

### Adjust Strength Values
Edit `backend/sdxl_config.py` to test different strengths:

```python
STAGE_IMG2IMG_STRENGTH = {
    "impression": 0.35,  # Lower = preserve objects, change style only
    "spatial": 0.55,     # Medium = allow structure changes
    "objects": 0.65,     # Higher = allow object replacement
    "ambient": 0.40,     # Low-medium = lighting changes only
}
```

### Verify Descriptor Usage
Check generated images to ensure they follow user descriptor while refining concepts.

## Migration Notes

- No database changes required
- Existing sessions will work (descriptor already in preferences.json)
- Old guidance prompts no longer used but code still exists
- All changes are backward compatible

