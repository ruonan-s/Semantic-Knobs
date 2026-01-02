"""
Style Transfer Script
Generates a new image based on an input image and text prompt using Gemini's image generation API.
"""

import os
import sys
import time
import base64
import argparse
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)


def generate_image_with_reference(
    input_image_path: str,
    text_prompt: str,
    output_path: str = None,
    max_retries: int = 5,
    base_delay: float = 15.0
) -> str:
    """
    Generate a new image based on an input reference image and a text prompt.
    
    Args:
        input_image_path: Path to the input/reference image
        text_prompt: Text prompt describing the desired output
        output_path: Optional path for the output image. If not provided, 
                     saves to same directory as input with '_styled' suffix
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
    
    Returns:
        Path to the generated output image
    """
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    
    # Determine output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(input_image_path)
        output_path = f"{base}_styled.png"
    
    # Build the contents structure with image + text
    contents = []
    
    # Create user content with text prompt and reference image
    user_parts = [types.Part(text=text_prompt)]
    
    # Load and add the reference image
    print(f"Loading reference image: {input_image_path}")
    with open(input_image_path, 'rb') as f:
        img_data = f.read()
    
    # Determine mime type based on file extension
    ext = os.path.splitext(input_image_path)[1].lower()
    mime_type_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
        '.gif': 'image/gif'
    }
    mime_type = mime_type_map.get(ext, 'image/png')
    
    # Create image part
    image_part = types.Part.from_bytes(
        data=img_data,
        mime_type=mime_type
    )
    user_parts.append(image_part)
    print(f"Added reference image ({mime_type})")
    
    # Create user content with all parts (text + image)
    user_content = types.Content(
        role="user",
        parts=user_parts
    )
    contents.append(user_content)
    
    # Retry loop with exponential backoff
    for attempt in range(max_retries):
        try:
            print(f"Calling Gemini API (attempt {attempt + 1}/{max_retries})...")
            
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            if response and response.candidates:
                candidate = response.candidates[0]
                
                if hasattr(candidate, 'content') and candidate.content.parts:
                    for i, part in enumerate(candidate.content.parts):
                        # Check for image data
                        if hasattr(part, 'inline_data') and part.inline_data is not None:
                            print(f"Processing generated image...")
                            
                            raw_data = part.inline_data.data
                            
                            # Try base64 decode first
                            try:
                                decoded_data = base64.b64decode(raw_data)
                                img_bytes = BytesIO(decoded_data)
                            except:
                                # Fallback to direct bytes
                                img_bytes = BytesIO(raw_data)
                            
                            # Try different image formats
                            img = None
                            for fmt in ['PNG', 'JPEG', 'WEBP']:
                                try:
                                    img_bytes.seek(0)
                                    img = Image.open(img_bytes)
                                    break
                                except:
                                    continue
                            
                            if img is None:
                                raise ValueError("Could not identify image format")
                            
                            # Convert to RGB if necessary
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Save the output image
                            img.save(output_path, "PNG")
                            print(f"Saved generated image to: {output_path}")
                            
                            return output_path
                    
                    print("No image data found in response")
                else:
                    print("No content parts in response")
            else:
                print("Empty response from API")
                
        except Exception as e:
            error_msg = str(e)
            print(f"Error (attempt {attempt + 1}/{max_retries}): {error_msg}")
            
            # Check if it's a retryable error
            if any(code in error_msg for code in ['500', '503', 'INTERNAL', 'UNAVAILABLE', 'overloaded', 'rate limit', 'quota']):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"API overloaded/rate limited. Waiting {delay:.1f} seconds before retry...")
                    time.sleep(delay)
                    continue
            
            # For non-retryable errors or final attempt
            if attempt == max_retries - 1:
                raise e
    
    raise RuntimeError(f"Failed to generate image after {max_retries} attempts")


def main():
    parser = argparse.ArgumentParser(
        description="Style Transfer: Generate a new image based on a reference image and text prompt"
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to the input/reference image"
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt describing the desired style or transformation"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for the generated image (default: input_styled.png)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retry attempts (default: 5)"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = generate_image_with_reference(
            input_image_path=args.input_image,
            text_prompt=args.prompt,
            output_path=args.output,
            max_retries=args.max_retries
        )
        print(f"Success! Generated image saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

