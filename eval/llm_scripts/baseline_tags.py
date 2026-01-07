"""
Baseline Tags Script

Generates an image using the learned tags from concept_weights.json.
Prompt format: "{adjective} {location}, {tag1}, {tag2}, ..., {tag10}"

This provides a baseline showing what an LLM can generate with tags alone (no reference image).
"""

import os
import sys
import json
import base64
from io import BytesIO
from PIL import Image

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from util import call_gemini_api_img

# System prompt for photorealistic interior generation
SYSTEM_PROMPT = "A photorealistic interior photo, full room view, high fidelity"


def generate_baseline_tags_image(
    session_folder: str,
    location: str,
    output_folder: str = None,
    output_filename: str = "llm_baseline_tags.png"
) -> str:
    """
    Generate an image using learned tags from concept_weights.json.
    
    The prompt format is: "{adjective} {location}, {tag1}, {tag2}, ..., {tag10}"
    
    Args:
        session_folder: Path to the session folder containing concept_weights.json and final_selection.json
        location: The location to generate (e.g., "bedroom", "kitchen", "cafe")
        output_folder: Folder to save the output image (defaults to slider/{location})
        output_filename: Name of the output image file (defaults to "llm_baseline_tags.png")
    
    Returns:
        Path to the saved image file
    """
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
    
    # Extract top 10 tag labels from concept_weights
    concept_weights = concept_weights_data.get("concept_weights", [])
    if not concept_weights:
        raise ValueError("No concept weights found in concept_weights.json")
    
    # Get up to 10 tag labels
    tag_labels = [cw.get("label", "") for cw in concept_weights[:10]]
    tag_labels = [t for t in tag_labels if t]  # Filter empty strings
    
    # Build the prompt: "{adjective} {location}, {tag1}, {tag2}, ..., {tag10}"
    tags_str = ", ".join(tag_labels)
    user_input = f"{adjective} {location}, {tags_str}"
    
    print(f"[BASELINE_TAGS] Generating image for: {location}")
    print(f"[BASELINE_TAGS] Adjective: {adjective}")
    print(f"[BASELINE_TAGS] Tags: {tag_labels}")
    print(f"[BASELINE_TAGS] Full prompt: {user_input}")
    
    # Determine output folder
    if output_folder is None:
        # Default to slider/{location} folder
        location_folder = location.replace(" ", "_")
        output_folder = os.path.join(session_folder, "slider", location_folder)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Call the image generation API
    response = call_gemini_api_img(user_input, SYSTEM_PROMPT)
    
    if not response:
        raise RuntimeError("Empty response from image generation API")
    
    candidate = response.candidates[0]
    
    if not (getattr(candidate, 'content', None) and getattr(candidate.content, 'parts', None)):
        raise RuntimeError("No image content in API response")
    
    # Process the first image part
    for part in candidate.content.parts:
        if part.inline_data and hasattr(part.inline_data, 'data'):
            raw_data = part.inline_data.data
            try:
                decoded_data = base64.b64decode(raw_data)
                img_data = BytesIO(decoded_data)
            except:
                img_data = BytesIO(raw_data)
            
            # Try different image formats
            img = None
            for fmt in ['PNG', 'JPEG', 'WEBP']:
                try:
                    img_data.seek(0)
                    img = Image.open(img_data)
                    break
                except:
                    continue
            
            if img is None:
                raise ValueError("Could not identify image format")
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            file_path = os.path.join(output_folder, output_filename)
            img.save(file_path, "PNG")
            
            print(f"[BASELINE_TAGS] Image saved to: {file_path}")
            return file_path
    
    raise RuntimeError("No valid image found in response")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate baseline image with learned tags"
    )
    parser.add_argument(
        "session_folder",
        type=str,
        help="Path to the session folder"
    )
    parser.add_argument(
        "location",
        type=str,
        help="Location to generate (e.g., bedroom, kitchen)"
    )
    parser.add_argument(
        "-o", "--output-folder",
        type=str,
        default=None,
        help="Output folder (default: slider/{location} in session folder)"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = generate_baseline_tags_image(
            session_folder=args.session_folder,
            location=args.location,
            output_folder=args.output_folder
        )
        print(f"Success! Generated image saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

