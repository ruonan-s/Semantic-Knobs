"""
Evaluation Script: LLM Style Transfer for New Locations

This script generates a style-transferred image when applying user preferences
to a new location. It uses the reference image from the original location's
slider folder as the style reference.

Usage:
    python eval_style_transfer.py

Modify the SESSION_FOLDER and NEW_LOCATION variables below to run on different sessions.
"""

import os
import sys
import json

# Add the eval/llm_scripts directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from style_transfer import generate_image_with_reference
from baseline1 import generate_baseline_image

# ============================================
# CONFIGURATION - Modify these variables
# ============================================
SESSION_FOLDER = "/home/nancy/Semantic-Knobs/eval/session_logs/eval_cozy_bedroom_sample_2026-01-01_22-52-46"
NEW_LOCATION = "kitchen"

# System prompt for photorealistic interior generation
SYSTEM_PROMPT = "Photorealistic interior photo, high fidelity, no people. Full room visible from an experiential, lived-in viewpoint: human eye-level camera placed inside the space, close to furniture with foreground present. Avoid wide-angle, centered, or architectural overview views. Aspect ratio: 1:1."


def generate_style_transfer_for_new_location(
    session_folder: str = SESSION_FOLDER,
    new_location: str = NEW_LOCATION
) -> str:
    """
    Generate a style-transferred image for a new location based on user's
    preferred aesthetic from the original location.
    
    Args:
        session_folder: Path to the session folder containing final_selection.json
        new_location: The new location to apply the style to
    
    Returns:
        Path to the generated style transfer image
    """
    # Load final_selection.json to get adjective and original location
    final_selection_path = os.path.join(session_folder, "final_selection.json")
    if not os.path.exists(final_selection_path):
        raise FileNotFoundError(f"final_selection.json not found in {session_folder}")
    
    with open(final_selection_path, 'r') as f:
        final_selection = json.load(f)
    
    adjective = final_selection.get("adjective", "")
    original_location = final_selection.get("location", "")
    
    print(f"Session folder: {session_folder}")
    print(f"Adjective: {adjective}")
    print(f"Original location: {original_location}")
    print(f"New location: {new_location}")
    
    # Load preferences.json to get exploration selected image
    preferences_path = os.path.join(session_folder, "preferences.json")
    preferences = {}
    if os.path.exists(preferences_path):
        with open(preferences_path, 'r') as f:
            preferences = json.load(f)
    
    # Use exploration selected image from preferences.json
    reference_image_path = None
    exploration_selection = preferences.get("selections", {}).get("impression")
    
    if exploration_selection:
        # Exploration selected image is in the impression folder
        impression_folder = os.path.join(session_folder, "impression")
        ref_path = os.path.join(impression_folder, f"{exploration_selection}.png")
        
        if os.path.exists(ref_path):
            reference_image_path = ref_path
            print(f"Using exploration selected image: {exploration_selection}.png")
        else:
            # Try alternate naming pattern
            ref_path = os.path.join(impression_folder, f"{exploration_selection}_0.png")
            if os.path.exists(ref_path):
                reference_image_path = ref_path
                print(f"Using exploration selected image: {exploration_selection}_0.png")
    
    if not reference_image_path or not os.path.exists(reference_image_path):
        raise FileNotFoundError(f"Exploration selected image not found. Check preferences.json in {session_folder}")
    
    print(f"Reference image: {os.path.basename(reference_image_path)}")
    
    # Build the style transfer prompt
    style_transfer_prompt = (
        f"{SYSTEM_PROMPT} "
        f"This user selected this image as their preferred example of a {adjective} {original_location}. "
        f"Generate a {adjective} {new_location} that matches this user's preferences"
    )
    print(f"Prompt: {style_transfer_prompt}")
    
    # Create output directory for new location
    new_slider_dir = os.path.join(session_folder, "slider", new_location.replace(" ", "_"))
    os.makedirs(new_slider_dir, exist_ok=True)
    
    # Output path for style transfer image
    output_path = os.path.join(new_slider_dir, "llm_style_transfer.png")
    
    # Generate the style transfer image
    print(f"Generating style transfer image...")
    generated_path = generate_image_with_reference(
        input_image_path=reference_image_path,
        text_prompt=style_transfer_prompt,
        output_path=output_path
    )
    
    print(f"Style transfer image saved: {generated_path}")
    
    # Generate the baseline image
    print(f"\nGenerating baseline image...")
    baseline_prompt = f"{adjective} {new_location}"
    print(f"Baseline prompt: {baseline_prompt}")
    
    baseline_path = generate_baseline_image(
        user_input=baseline_prompt,
        output_folder=new_slider_dir,
        output_filename="llm_baseline.png"
    )
    
    print(f"Baseline image saved: {baseline_path}")
    
    return generated_path


if __name__ == "__main__":
    generate_style_transfer_for_new_location()

