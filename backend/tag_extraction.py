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

prompt_impression_v1 = '''
    Analyze the attached image and identify Core Impression elements - the visual style indicators and atmosphere-creating elements that define the scene's overall character and feeling.

    Extract descriptive elements that capture:
    - Style characteristics: The visual design approach and aesthetic style visible in the space
    - Atmosphere-creating elements: Specific visual elements (colors, materials, textures, objects) that contribute to the overall mood
    - Design approach: How the space is visually organized and what design philosophy it reflects
    - Functional indicators: Visual cues about what this space is designed for and how it feels to use
    - Overall character: The personality and feeling that the visual elements collectively create

    EXCLUDE lighting, spatial layout, and specific object placement - focus on style, materials, colors, and overall aesthetic approach.

    Format your response as JSON:

    {
    "visual_elements": [
        "scandinavian aesthetic",
        "wood textures add warmth",
        "contemporary minimalism",
        "industrial design elements",
        "bohemian eclectic style",
        "creative workspace energy",
        "soft textiles create coziness",
        "sophisticated urban vibe",
        "curated collected aesthetic"
    ]
    }
    '''

prompt_spatial_v1 = '''
    Analyze the attached image and identify Structural Blueprint elements - the specific layout configurations and distinctive spatial relationships that define how this space is organized.

    Extract 6-9 most prominent descriptive elements that capture specific spatial configurations:
    - Layout patterns: How the space is geometrically arranged 
    - Major functional zones: Where key activity areas are positioned within the overall layout
    - Spatial features: Distinctive architectural or design elements that define the space
    - Built elements: Integrated storage, area separations, and structural features
    - Boundary conditions: How spaces are enclosed, opened, or partially divided
    - Dimensional elements: Platform levels, ceiling heights, width variations, length proportions

    Focus on the specific layout configuration and how spaces relate to each other.

    Format your response as JSON:

    {
    "visual_elements": [
        "L-shaped layout",
        "cooking zone in corner",
        "workspace area against wall",
        "dining area separate",
        "built-in storage integration",
        "raised platform",
        "semi-open space",
        "curved shelving",
        "fully enclosed room"
    ]
    }'''

prompt_objects_v1 = '''
    Analyze the attached image and identify Material World elements - the specific objects, furniture, materials, and surface textures visible in the space.

    Extract 6-9 most prominent descriptive elements that capture visible physical elements:
    - Objects and items: All visible things in the space - furniture, equipment, decor, plants, accessories, etc.
    - Object arrangement: How furniture and items are positioned and grouped relative to each other
    - Equipment and accessories: Tools, technology, decorative items, and functional accessories in the space
    - Material surfaces: The types of materials and their visual/tactile qualities on surfaces
    - Texture qualities: How surfaces appear in terms of smoothness, roughness, finish, and tactile properties
    - Hardware details: Visible mechanical elements, handles, fixtures, and connecting elements

    Focus on tangible, identifiable objects and their material appearance.

    Format your response as JSON:

    {
    "visual_elements": [
        "wooden desk surface",
        "chair pushed under desk",
        "computer monitor",
        "plants clustered near window",
        "white bookshelf",
        "smooth wood grain",
        "lamp positioned at desk corner",
        "ceramic plant pot",
        "books stacked horizontally",
        "dual monitor setup",
        "matte wood finish",
        "fabric seat cushion"
    ]
    }'''

prompt_ambient_v1 = '''
    Analyze the attached image and identify Ambient Medium elements - the lighting, atmospheric conditions, and environmental qualities that define how the scene is illuminated and perceived.

    Extract 6-9 most prominent descriptive elements that capture:
    - Light sources and types: The origins and nature of illumination in the space
    - Light quality and atmosphere: How the lighting behaves and what atmospheric feeling it creates
    - Time and environmental context: What time period, season, or environmental conditions are suggested
    - Atmospheric mood and ambiance: The overall environmental feeling and sensory experience
    - Environmental feeling: The broader atmospheric character and emotional tone of the lighting

    Format your response as JSON:

    {
    "visual_elements": [
        "warm natural daylight",
        "golden hour ambiance",
        "peaceful atmosphere",
        "morning serenity",
        "soft diffuse lighting", 
        "contemplative mood",
        "bright clarity",
        "gentle environmental feeling",
        "cozy warmth",
        "golden hour",
        "inspiring brightness"
    ]
    }'''

prompt_impression = '''
Analyze the attached image and identify Core Impression elements — the foundational style, location, and mood-defining features.

Extract descriptive elements that capture:
- Style characteristics: The visual design approach and aesthetic style visible in the space.
- Location context: Cues about where this space could be situated (e.g., coastal retreat, urban loft, mountain cabin, desert hideaway).
- Atmosphere-creating elements: Specific visual elements (colors, materials, textures, objects) that contribute to the overall mood.
- Design approach: How the space is visually organized and what design philosophy it reflects.
- Functional indicators: Visual cues about what this space is designed for and how it feels to use.
- Overall character: The personality and feeling that the visual elements collectively create.

EXCLUDE lighting specifics, spatial layout, and exact object placement.

Output 6–9 short, distinct tags, one per visual element.
Format as JSON:
{
"visual_elements": [
    "scandinavian aesthetic",
    "beachside retreat location",
    "wood textures add warmth",
    "contemporary minimalism",
    "industrial design accents",
    "creative workspace energy",
    "soft textiles create coziness",
    "sophisticated urban vibe",
    "curated collected aesthetic"
]
}
'''
prompt_spatial = '''
Analyze the attached image and identify Structural Blueprint elements — the specific layout configurations and distinctive spatial relationships that define how this space is organized.

Extract 6–9 short, distinct descriptive elements that capture:
- Layout patterns: How the space is geometrically arranged.
- Major functional zones: Where key activity areas are positioned.
- Spatial features: Distinctive architectural or design elements.
- Built elements: Integrated storage, area separations, and structural features.
- Boundary conditions: How spaces are enclosed, opened, or partially divided.
- Dimensional elements: Platform levels, ceiling heights, width/length proportions.
- Location-aware features visible in the spatial form (e.g., balcony over ocean, mountain-view window wall).

Format as JSON:
{
"visual_elements": [
    "L-shaped layout",
    "cooking zone in corner",
    "workspace area against wall",
    "dining area separate",
    "built-in storage integration",
    "raised platform",
    "semi-open space",
    "curved shelving",
    "balcony over ocean view"
]
}
'''
prompt_objects = '''
Analyze the attached image and identify Material World elements — the physical objects, materials, and arrangement choices that define the space.

Extraction rules:
- For each prominent object, create separate tags for:
  * Object type (e.g., "sofa", "daybed", "hammock", "bookshelf")
  * Material or finish (e.g., "linen fabric", "polished oak", "woven rattan")
  * Form or style descriptor if visually distinctive (e.g., "low platform", "curved form")
- Arrangement and accessory elements should be listed separately (e.g., "sofa facing fireplace", "plants clustered in corner", "stacked art frames on floor").
- Do NOT combine object + material in the same tag.
- Aim for the most visually prominent and style-defining elements.

Output 6–12 short, distinct tags.
Format as JSON:
{
"visual_elements": [
    "sofa",
    "linen fabric",
    "low platform",
    "potted tree",
    "ceramic vase",
    "sofa facing window",
    "plants clustered in corner",
    "stacked art frames on floor"
]
}
'''
prompt_ambient = '''
Analyze the attached image and identify Ambient Medium elements — the lighting, atmospheric conditions, and environmental qualities.

Extraction rules:
- Include at least one tag for the **light source** (e.g., "light through window", "floor lamp glow", "candlelight", "fireplace light").
- Include tags for **light quality** (e.g., "soft diffuse lighting", "warm golden glow", "sharp morning sunlight").
- Include tags for **environmental context** (e.g., "sunset", "overcast winter day", "autumn afternoon").
- Include tags for **mood/atmosphere** (e.g., "peaceful atmosphere", "inviting warmth", "contemplative mood").
- Keep each tag short and concrete — avoid long descriptions.
- Do not output duplicate phrases with small wording changes.

Output 6–9 short, distinct tags.
Format as JSON:
{
"visual_elements": [
    "light through window",
    "soft diffuse lighting",
    "sunset",
    "inviting warmth",
    "table lamp glow",
    "warm golden glow",
    "peaceful atmosphere",
    "autumn afternoon"
]
}
'''


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