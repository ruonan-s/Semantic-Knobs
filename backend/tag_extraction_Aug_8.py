import os
import json
import re
import time
import base64
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from typing import List

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# focus on general elements (older version)
prompt = '''
    Analyze the attached image and identify its main visual elements. List the primary subjects, key objects, and significant environmental components that define the scene. Focus only on what is visually present. Keep the output concise and use simple language to ensure clarity.

    Format your response as JSON.

    # Output Format

    ```json
    {
        "visual_elements": [
            "A winding dirt path",
            "Tall pine trees",
            "Sunlight filtering through the canopy",
            "A wooden bench",
            "Green ferns and undergrowth"
        ]
    }
    ```
    '''

prompt_impression = '''
    Analyze the attached image and identify Core Impression elements - the fundamental semantic and affective qualities that define the scene's meaning and emotional atmosphere.

    Extract descriptive elements that capture:
    - Function and purpose (what activities this space supports)
    - Style and cultural context (design style, era, cultural setting)
    - Emotional and atmospheric qualities (mood, feeling, psychological impact)

    Format your response as JSON:

    {
    "visual_elements": [
        "for reading/working",
        "modern style", 
        "business style",
        "creative and supportive",
        "professional atmosphere",
        "focused environment",
        "contemporary design"
    ]
    }
    '''

prompt_spatial = '''
    Analyze the attached image and identify Structural Blueprint elements - the specific layout configurations and distinctive spatial relationships that define how this space is organized.

    Extract descriptive elements that capture unique spatial characteristics:
    - Layout types: U-shaped layout, L-shaped layout, galley layout, island layout, peninsula layout, linear layout, corner layout, open plan layout
    - Spatial configurations: central island, breakfast bar, work triangle, parallel counters, wraparound design, corner optimization
    - Vertical features: double-height space, mezzanine level, raised platform, sunken area, split-level design, vaulted ceiling, low ceiling
    - Boundary conditions: fully enclosed room, semi-open space, completely open plan, partial walls, room dividers, glass partitions
    - Connection types: indoor-outdoor flow, seamless transitions, separated zones, defined boundaries, visual connections, physical barriers
    - Scale characteristics: compact footprint, expansive layout, narrow corridor, wide open space, intimate scale, grand proportions

    Focus on the specific layout configuration and how spaces relate to each other.

    Format your response as JSON:

    {
    "visual_elements": [
        "U-shaped layout",
        "central island",
        "indoor-outdoor flow",
        "compact footprint", 
        "breakfast bar",
        "corner optimization",
        "double-height space",
        "seamless transitions"
    ]
    }'''

prompt_ambient = '''
    Analyze the attached image and identify Ambient Medium elements - the lighting, atmospheric conditions, and environmental qualities that define how the scene is illuminated and perceived.

    Extract descriptive elements that capture:
    - Lighting characteristics (quality, color, intensity, direction)
    - Time and weather conditions (time of day, season, weather)
    - Atmospheric mood (environmental feeling, ambient quality)

    Format your response as JSON:

    {
    "visual_elements": [
        "warm lighting",
        "morning of the day",
        "sunset",
        "raindrops", 
        "sad atmosphere",
        "dark room",
        "soft shadows",
        "golden hour"
    ]
    }'''


def extract_visual_elements_from_image(image_path: str, prompt: str, max_retries: int = 3, retry_delay: float = 2.0) -> List[str]:
    """
    Extract visual elements from an image with retry logic.
    
    Args:
        image_path: Path to the image file
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
    
    Returns:
        List of extracted visual elements (tags)
    """
    attempt = 0
    last_error = None
    
    while attempt < max_retries:
        try:
            print(f"\nAttempt {attempt + 1} to extract visual elements from {image_path}")
            
            # Check if file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Read and encode the image
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            # Prepare the message for OpenAI API (GPT-4o vision)
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]}
            ]

            response = openai_client.chat.completions.create(
                model="gpt-4o",  # or another vision-capable model
                messages=messages,
                max_tokens=512
            )

            # Extract JSON from the response
            text = response.choices[0].message.content.strip()
            print(f"Raw response: {text[:200]}...")  # Log first 200 chars
            
            try:
                # Remove markdown code block if present
                text = re.sub(r"```json|```", "", text).strip()
                result = json.loads(text)
            except Exception as e:
                print(f"Failed to parse JSON: {e}\nRaw output: {text}")
                result = {"visual_elements": []}

            if not isinstance(result, dict):
                tags = []
            else:
                tags = result.get("visual_elements", [])
                if not isinstance(tags, list):
                    tags = []
                # Ensure all tags are strings
                tags = [str(tag) for tag in tags if tag]
            
            print(f"Successfully extracted {len(tags)} tags: {tags}")
            return tags
            
        except Exception as e:
            last_error = e
            print(f"Error during extraction attempt {attempt + 1}: {str(e)}")
            
            attempt += 1
            if attempt < max_retries:
                print(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
    
    # If we get here, all attempts failed
    error_msg = f"Failed to extract visual elements after {max_retries} attempts"
    if last_error:
        error_msg += f": {str(last_error)}"
    print(f"Final error: {error_msg}")
    
    # Return empty list instead of raising error to prevent blocking the UI
    return []


# Example usage:
#result = extract_visual_elements_from_image("../sample/cozy_warm/impression/impression_0_0.png")
#print(result)