import os
import sys
import base64
from io import BytesIO
from PIL import Image

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from util import call_gemini_api_img, sanitize_folder_name

# ============================================
# PROMPTS - Modify these variables as needed
# ============================================
Adjective = "Inviting"
Location = ["Bathroom", "Bedroom", "Café", "Home Office", "Kitchen", "Livingroom", "Reading Nook", "Restaurant"]
SYSTEM_PROMPT = "Photorealistic interior photo, high fidelity, no people. Full room visible from an experiential, lived-in viewpoint: human eye-level camera placed inside the space, close to furniture with foreground present. Avoid wide-angle, centered, or architectural overview views. Aspect ratio: 1:1."

def generate_baseline_image(
    user_input: str,
    system_prompt: str = SYSTEM_PROMPT,
    adjective: str = None,
    location: str = None,
    output_folder: str = None,
    output_filename: str = None
) -> str:
    """
    Generate a single image using the Gemini image generation model.
    Uses the same model as the exploration stage: gemini-2.0-flash-exp-image-generation
    
    Args:
        user_input: The text prompt describing what to generate
        system_prompt: System instructions for the model
        adjective: The adjective used (e.g., "Cozy", "Focused") - used for folder structure
        location: The location name (e.g., "Café", "Kitchen") - used for folder structure
        output_folder: Folder to save the output image (defaults to baseline_generic_{adjective}/{location})
        output_filename: Name of the output image file (defaults to {adjective}_{location}.png)
    
    Returns:
        Path to the saved image file
    """
    if adjective is None or location is None:
        # Try to parse from user_input (format: "Adjective Location")
        parts = user_input.split(' ', 1)
        if len(parts) == 2:
            adjective = parts[0]
            location = parts[1]
        else:
            # Fallback to old behavior
            adjective = "Focused"
            location = user_input
    
    if output_folder is None:
        base_folder = f"baseline_generic_{adjective}"
        location_folder = location
        output_folder = os.path.join(os.path.dirname(__file__), base_folder, location_folder)
    
    if output_filename is None:
        # Use sanitize_folder_name to convert spaces to underscores
        location_for_filename = sanitize_folder_name(location)
        output_filename = f"{adjective}_{location_for_filename}.png"
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Call the image generation API (same as exploration stage)
    response = call_gemini_api_img(user_input, system_prompt)
    
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
            
            print(f"Image saved to: {file_path}")
            return file_path
    
    raise RuntimeError("No valid image found in response")


if __name__ == "__main__":
    for loc in Location:
        user_input = f"{Adjective} {loc}"
        generate_baseline_image(user_input, adjective=Adjective, location=loc)

