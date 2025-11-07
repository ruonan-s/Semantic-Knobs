"""
Stage-specific guidance prompts for SDXL generation.
These are prepended to the fused concept embeddings to provide context.
"""

STAGE_GUIDANCE_PROMPTS = {
    "impression": "Update image aesthetic style to",
    "spatial": "Update image spatial structure to",
    "objects": "Update image objects and materials to",
    "ambient": "Update image ambient to"
}

def get_guidance_prompt(stage: str) -> str:
    """
    Get the guidance prompt for a given stage.
    
    Args:
        stage: Stage name (e.g., "impression", "spatial", "objects", "ambient")
               Can include "_refinement" suffix, which will be stripped.
    
    Returns:
        Guidance prompt string, or empty string if stage not found
    
    Examples:
        >>> get_guidance_prompt("impression")
        "Update image aesthetic style to"
        >>> get_guidance_prompt("impression_refinement")
        "Update image aesthetic style to"
        >>> get_guidance_prompt("spatial")
        "Update image spatial structure to"
    """
    # Strip _refinement suffix if present
    base_stage = stage.replace("_refinement", "")
    
    return STAGE_GUIDANCE_PROMPTS.get(base_stage, "")
