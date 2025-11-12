"""
SDXL Generation Configuration

Stage-specific settings for img2img strength and other generation parameters.
These values control how much the generated image deviates from the reference image.

Strength values:
- 0.0-0.3: Minor style tweaks, objects/layout preserved
- 0.3-0.5: Moderate changes, some objects may change
- 0.5-0.7: Significant changes, layout may shift
- 0.7-1.0: Major changes, only rough structure preserved
- 1.0: Effectively txt2img (ignores reference)
"""

# Stage-specific img2img strength values
# Used during refinement stages when a reference image is provided
STAGE_IMG2IMG_STRENGTH = {
    "impression": 0.85,      # Style/aesthetic refinement
    "spatial": 0.8,         # Spatial structure refinement
    "objects": 0.8,         # Objects and materials refinement
    "ambient": 0.8,         # Lighting and atmosphere refinement
}

def get_stage_strength(stage: str, default: float = 0.75) -> float:
    """
    Get the img2img strength for a given stage.
    
    Args:
        stage: Stage name (e.g., "impression", "spatial", "impression_refinement")
               The "_refinement" suffix is automatically stripped.
        default: Default strength if stage not found
    
    Returns:
        Strength value between 0.0 and 1.0
    
    Examples:
        >>> get_stage_strength("impression")
        0.75
        >>> get_stage_strength("impression_refinement")
        0.75
        >>> get_stage_strength("unknown_stage")
        0.75
    """
    # Strip _refinement suffix if present
    base_stage = stage.replace("_refinement", "")
    
    return STAGE_IMG2IMG_STRENGTH.get(base_stage, default)

