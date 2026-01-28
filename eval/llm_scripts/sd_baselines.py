"""
Stable Diffusion Baselines for Evaluation

This module provides SD-based baseline image generation that uses the model's
native prompt encoding (unlike "ours" which uses custom-fused embeddings).

Three baseline methods:
1. SD Text: txt2img with "{adjective} {location}" prompt
2. SD Tags: txt2img with "{adjective} {location}, {tag1}, ..., {tag10}" prompt  
3. SD Img2Img: img2img with reference image and text prompt
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image
from typing import Optional, List

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
SDXL_DIR = PROJECT_ROOT / "SDXL"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SDXL_DIR))

from SDXL.diffusion_runner import DiffusionRunner

# Global runner instance (cached to avoid reloading model)
_sd_baseline_runner: Optional[DiffusionRunner] = None

# Default negative prompt (same as "ours" for fair comparison)
DEFAULT_NEGATIVE_PROMPT = ", ".join([
    "illustration", "cartoon", "anime", "CGI", "human"
])

# Default generation parameters (consistent with "ours")
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_IMG2IMG_STRENGTH = 0.7


def _get_runner() -> DiffusionRunner:
    """Get or create the shared DiffusionRunner instance."""
    global _sd_baseline_runner
    
    if _sd_baseline_runner is None:
        print("[SD Baselines] Initializing DiffusionRunner...")
        _sd_baseline_runner = DiffusionRunner(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            steps=DEFAULT_STEPS
        )
        # Ensure txt2img pipeline is loaded
        _sd_baseline_runner._ensure_txt2img()
        print(f"[SD Baselines] DiffusionRunner ready on {_sd_baseline_runner.device}")
    
    return _sd_baseline_runner


def generate_sd_text_baseline(
    adjective: str,
    location: str,
    output_path: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = 2026,
    steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
) -> str:
    """
    Generate SD text-only baseline using "{adjective} {location}" as prompt.
    
    This uses the model's native prompt encoding (no custom embedding fusion).
    
    Args:
        adjective: The adjective (e.g., "Cozy", "Lively")
        location: The location (e.g., "Bedroom", "Kitchen")
        output_path: Path to save the generated image
        negative_prompt: Negative prompt for generation
        seed: Random seed for reproducibility
        steps: Number of inference steps
        guidance_scale: Classifier-free guidance scale
    
    Returns:
        Path to the saved image
    """
    runner = _get_runner()
    
    # Build the simple prompt: "{adjective} {location}"
    positive_prompt = f"{adjective} {location}"
    
    # Ensure negative_prompt is a string (not tuple/list)
    if isinstance(negative_prompt, (tuple, list)):
        negative_prompt = ", ".join(str(item) for item in negative_prompt)
    elif not isinstance(negative_prompt, str):
        negative_prompt = str(negative_prompt)
    
    print(f"[SD Text Baseline] Generating with prompt: '{positive_prompt}'")
    print(f"[SD Text Baseline] Seed: {seed}, Steps: {steps}, Guidance: {guidance_scale}")
    
    # Generate using model's native prompt encoding
    image = runner.generate(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        gscale=guidance_scale,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save image
    image.save(output_path)
    print(f"[SD Text Baseline] Saved to: {output_path}")
    
    return output_path


def generate_sd_tags_baseline(
    session_folder: str,
    location: str,
    output_path: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = 2026,
    steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    max_tags: int = 10
) -> str:
    """
    Generate SD tags baseline using "{adjective} {location}, {tag1}, ..., {tag10}" as prompt.
    
    Reads tags from concept_weights.json in the session's impression folder.
    This uses the model's native prompt encoding (no custom embedding fusion).
    
    Args:
        session_folder: Path to the session folder containing concept_weights.json
        location: The location to generate (e.g., "Bedroom", "Kitchen")
        output_path: Path to save the generated image
        negative_prompt: Negative prompt for generation
        seed: Random seed for reproducibility
        steps: Number of inference steps
        guidance_scale: Classifier-free guidance scale
        max_tags: Maximum number of tags to include in prompt
    
    Returns:
        Path to the saved image
    """
    runner = _get_runner()
    
    # Load concept_weights.json to get tags
    concept_weights_path = os.path.join(session_folder, "impression", "concept_weights.json")
    if not os.path.exists(concept_weights_path):
        raise FileNotFoundError(f"concept_weights.json not found: {concept_weights_path}")
    
    with open(concept_weights_path, 'r') as f:
        concept_weights_data = json.load(f)
    
    # Load final_selection.json to get adjective
    final_selection_path = os.path.join(session_folder, "final_selection.json")
    if not os.path.exists(final_selection_path):
        raise FileNotFoundError(f"final_selection.json not found: {final_selection_path}")
    
    with open(final_selection_path, 'r') as f:
        final_selection = json.load(f)
    
    adjective = final_selection.get("adjective", "")
    
    # Extract top N tag labels from concept_weights
    concept_weights = concept_weights_data.get("concept_weights", [])
    if not concept_weights:
        raise ValueError("No concept weights found in concept_weights.json")
    
    # Get up to max_tags tag labels
    tag_labels = [cw.get("label", "") for cw in concept_weights[:max_tags]]
    tag_labels = [t for t in tag_labels if t]  # Filter empty strings
    
    # Build the prompt: "{adjective} {location}, {tag1}, {tag2}, ..., {tag10}"
    tags_str = ", ".join(tag_labels)
    positive_prompt = f"{adjective} {location} with {tags_str}"
    
    # Ensure negative_prompt is a string (not tuple/list)
    if isinstance(negative_prompt, (tuple, list)):
        negative_prompt = ", ".join(str(item) for item in negative_prompt)
    elif not isinstance(negative_prompt, str):
        negative_prompt = str(negative_prompt)
    
    print(f"[SD Tags Baseline] Generating for: {location}")
    print(f"[SD Tags Baseline] Adjective: {adjective}")
    print(f"[SD Tags Baseline] Tags ({len(tag_labels)}): {tag_labels}")
    print(f"[SD Tags Baseline] Full prompt: {positive_prompt[:100]}...")
    print(f"[SD Tags Baseline] Seed: {seed}, Steps: {steps}, Guidance: {guidance_scale}")
    
    # Generate using model's native prompt encoding
    image = runner.generate(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        gscale=guidance_scale,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save image
    image.save(output_path)
    print(f"[SD Tags Baseline] Saved to: {output_path}")
    
    return output_path


def generate_sd_img2img_baseline(
    input_image_path: str,
    adjective: str,
    original_location: str,
    target_location: str,
    output_path: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    strength: float = DEFAULT_IMG2IMG_STRENGTH,
    seed: int = 2026,
    steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
) -> str:
    """
    Generate SD img2img baseline using reference image + text prompt.
    
    This uses the model's native prompt encoding (no custom embedding fusion).
    The prompt format matches the LLM style transfer approach.
    
    Args:
        input_image_path: Path to the reference/input image
        adjective: The adjective (e.g., "Cozy", "Lively")
        original_location: The original location from the reference image
        target_location: The target location to generate
        output_path: Path to save the generated image
        negative_prompt: Negative prompt for generation
        strength: img2img strength (0-1, higher = more deviation from input)
        seed: Random seed for reproducibility
        steps: Number of inference steps
        guidance_scale: Classifier-free guidance scale
    
    Returns:
        Path to the saved image
    
    Raises:
        RuntimeError: If img2img pipeline failed to initialize
    """
    runner = _get_runner()
    
    # Explicitly ensure img2img pipeline is loaded before attempting generation
    print(f"[SD Img2Img Baseline] Ensuring img2img pipeline is loaded...")
    runner._ensure_img2img()
    
    if runner.pipe_i2i is None:
        error_msg = (
            "SDXL Img2Img pipeline failed to initialize. "
            "This could be due to:\n"
            "  - Missing diffusers dependency: pip install diffusers\n"
            "  - Missing StableDiffusionXLImg2ImgPipeline: The img2img pipeline may not be available\n"
            "  - Out of memory: Try reducing batch size or using CPU\n"
            "  - Model download issues: Check internet connection and HuggingFace access"
        )
        print(f"[SD Img2Img Baseline] ERROR: {error_msg}")
        raise RuntimeError(error_msg)
    
    print(f"[SD Img2Img Baseline] Img2Img pipeline ready on {runner.device}")
    
    # Load and preprocess reference image
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    
    print(f"[SD Img2Img Baseline] Loading reference image: {input_image_path}")
    init_image = Image.open(input_image_path).convert("RGB")
    
    # Build the style transfer prompt (similar to LLM approach)
    if target_location.lower() == original_location.lower():
        # Same location - generate another image in same style
        positive_prompt = (
            f"Modify the location in the reference image to {target_location} with the current style and aesthetic"
        )
    else:
        # Different location - transfer style
        positive_prompt = (
            f"Modify the location in the reference image to {target_location} with the current style and aesthetic")
    
    print(f"[SD Img2Img Baseline] Original location: {original_location}")
    print(f"[SD Img2Img Baseline] Target location: {target_location}")
    print(f"[SD Img2Img Baseline] Prompt: {positive_prompt}")
    print(f"[SD Img2Img Baseline] Strength: {strength}, Seed: {seed}")
    
    # Ensure negative_prompt is a string (not tuple/list)
    if isinstance(negative_prompt, (tuple, list)):
        negative_prompt = ", ".join(str(item) for item in negative_prompt)
    elif not isinstance(negative_prompt, str):
        negative_prompt = str(negative_prompt)
    
    print(f"[SD Img2Img Baseline] Negative prompt: {negative_prompt}")
    
    # Generate using model's native prompt encoding with img2img
    try:
        image = runner.generate_img2img(
            init_image=init_image,
            strength=strength,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            gscale=guidance_scale,
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            resize_mode="fit_center_crop"
        )
        
        # Check if we got a mock image (simple heuristic: check if image is mostly gray)
        # Mock images from DiffusionRunner typically have a gray background (#F5F5F5)
        # This is a basic check - a real image is unlikely to be uniform gray
        pixels = list(image.getdata())
        if len(pixels) > 0:
            first_pixel = pixels[0]
            # If all pixels are approximately the same (gray mock image), warn
            sample_size = min(100, len(pixels))
            sample_pixels = pixels[:sample_size]
            all_similar = all(
                abs(sum(p[:3]) - sum(first_pixel[:3])) < 10 
                for p in sample_pixels 
                if len(p) >= 3
            )
            if all_similar and sum(first_pixel[:3]) > 700:  # Very light gray
                print(f"[SD Img2Img Baseline] WARNING: Generated image appears to be a mock/fallback image!")
                print(f"[SD Img2Img Baseline] This suggests the generation may have failed silently.")
        
    except Exception as e:
        error_msg = f"Image generation failed: {str(e)}"
        print(f"[SD Img2Img Baseline] ERROR: {error_msg}")
        raise RuntimeError(error_msg) from e
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save image
    image.save(output_path)
    print(f"[SD Img2Img Baseline] ✅ Saved to: {output_path}")
    
    return output_path


def generate_sd_preferences_baseline(
    session_folder: str,
    location: str,
    output_path: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = 2026,
    steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    max_tags_per_category: int = 10
) -> str:
    """
    Generate SD preferences baseline using structured prompt with positive/negative/neutral features.
    
    Reads tag preferences from tag_preferences.json in the session's impression folder.
    This uses the model's native prompt encoding (no custom embedding fusion).
    
    Prompt format:
        "{adjective} {location}, positive features: {pos_tags}, negative features: {neg_tags}, neutral features: {neutral_tags}"
    
    Args:
        session_folder: Path to the session folder containing tag_preferences.json
        location: The location to generate (e.g., "Bedroom", "Kitchen")
        output_path: Path to save the generated image
        negative_prompt: Negative prompt for generation
        seed: Random seed for reproducibility
        steps: Number of inference steps
        guidance_scale: Classifier-free guidance scale
        max_tags_per_category: Maximum number of tags to include per category
    
    Returns:
        Path to the saved image
    """
    runner = _get_runner()
    
    # Load tag_preferences.json
    tag_prefs_path = os.path.join(session_folder, "impression", "tag_preferences.json")
    if not os.path.exists(tag_prefs_path):
        raise FileNotFoundError(f"tag_preferences.json not found: {tag_prefs_path}")
    
    with open(tag_prefs_path, 'r') as f:
        tag_prefs_data = json.load(f)
    
    # Load final_selection.json to get adjective
    final_selection_path = os.path.join(session_folder, "final_selection.json")
    if not os.path.exists(final_selection_path):
        raise FileNotFoundError(f"final_selection.json not found: {final_selection_path}")
    
    with open(final_selection_path, 'r') as f:
        final_selection = json.load(f)
    
    adjective = final_selection.get("adjective", "")
    
    # Extract tags by category (limit to max_tags_per_category)
    positive_tags = tag_prefs_data.get("positive", [])[:max_tags_per_category]
    negative_tags = tag_prefs_data.get("negative", [])[:max_tags_per_category]
    neutral_tags = tag_prefs_data.get("neutral", [])[:max_tags_per_category]
    
    # Build the structured prompt
    prompt_parts = [f"{adjective} {location}"]
    
    if positive_tags:
        prompt_parts.append(f"positive features: {', '.join(positive_tags)}")
    if negative_tags:
        prompt_parts.append(f"negative features: {', '.join(negative_tags)}")
    if neutral_tags:
        prompt_parts.append(f"neutral features: {', '.join(neutral_tags)}")
    
    positive_prompt = ", ".join(prompt_parts)
    
    # Ensure negative_prompt is a string (not tuple/list)
    if isinstance(negative_prompt, (tuple, list)):
        negative_prompt = ", ".join(str(item) for item in negative_prompt)
    elif not isinstance(negative_prompt, str):
        negative_prompt = str(negative_prompt)
    
    print(f"[SD Prefs Baseline] Generating for: {location}")
    print(f"[SD Prefs Baseline] Adjective: {adjective}")
    print(f"[SD Prefs Baseline] Positive tags ({len(positive_tags)}): {positive_tags}")
    print(f"[SD Prefs Baseline] Negative tags ({len(negative_tags)}): {negative_tags}")
    print(f"[SD Prefs Baseline] Neutral tags ({len(neutral_tags)}): {neutral_tags}")
    print(f"[SD Prefs Baseline] Full prompt: {positive_prompt[:150]}...")
    print(f"[SD Prefs Baseline] Seed: {seed}, Steps: {steps}, Guidance: {guidance_scale}")
    
    # Generate using model's native prompt encoding
    image = runner.generate(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        gscale=guidance_scale,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save image
    image.save(output_path)
    print(f"[SD Prefs Baseline] Saved to: {output_path}")
    
    return output_path


def set_runner(runner: DiffusionRunner) -> None:
    """
    Set an external DiffusionRunner instance to reuse.
    
    This allows sharing the runner with other parts of the system
    to avoid loading multiple model instances.
    
    Args:
        runner: An initialized DiffusionRunner instance
    """
    global _sd_baseline_runner
    _sd_baseline_runner = runner
    print("[SD Baselines] Using external DiffusionRunner instance")


# ============== CLI ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SD baselines")
    parser.add_argument("--type", choices=["text", "tags", "img2img"], required=True,
                       help="Type of baseline to generate")
    parser.add_argument("--adjective", type=str, help="Adjective (e.g., Cozy)")
    parser.add_argument("--location", type=str, help="Location (e.g., Bedroom)")
    parser.add_argument("--session-folder", type=str, help="Session folder for tags baseline")
    parser.add_argument("--input-image", type=str, help="Input image for img2img")
    parser.add_argument("--original-location", type=str, help="Original location for img2img")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output path")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    
    args = parser.parse_args()
    
    try:
        if args.type == "text":
            if not args.adjective or not args.location:
                parser.error("--adjective and --location required for text baseline")
            generate_sd_text_baseline(args.adjective, args.location, args.output, seed=args.seed)
            
        elif args.type == "tags":
            if not args.session_folder or not args.location:
                parser.error("--session-folder and --location required for tags baseline")
            generate_sd_tags_baseline(args.session_folder, args.location, args.output, seed=args.seed)
            
        elif args.type == "img2img":
            if not args.input_image or not args.adjective or not args.location:
                parser.error("--input-image, --adjective, and --location required for img2img baseline")
            orig_loc = args.original_location or args.location
            generate_sd_img2img_baseline(
                args.input_image, args.adjective, orig_loc, args.location, args.output, seed=args.seed
            )
        
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
