import os
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

def sanitize_folder_name(name: str) -> str:
    """Replace invalid filesystem characters with underscores."""
    return re.sub(r'[^\w\-]', '_', name)

def call_gemini_api(user_input: str, system_prompt: str) -> str:
    """
    Calls Gemini API with a single user_input string and system prompt.
    Returns cleaned model.text without markdown fences.
    """
    try:
        # Prepare the final API input for tracking
        api_input = {
            "model": "gemini-2.5-flash",
            "contents": [user_input],
            "system_instruction": system_prompt
        }
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[user_input],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            )
        )
        cleaned = re.sub(r"```json|```", "", response.text).strip()
        return cleaned
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return ""

def call_gemini_api_with_tracking(user_input: str, system_prompt: str) -> tuple[str, str]:
    """
    Calls Gemini API and returns both the response and the final API input structure.
    Returns (response, api_input_str)
    """
    try:
        # Prepare the final API input for tracking
        api_input = {
            "model": "gemini-2.5-flash",
            "contents": [user_input],
            "system_instruction": system_prompt
        }
        # Show complete API input structure for tracking
        api_input_str = f"Model: {api_input['model']}\nContents:\n[0] User Input: {user_input}\nSystem Instruction: {system_prompt}"
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[user_input],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            )
        )
        cleaned = re.sub(r"```json|```", "", response.text).strip()
        return cleaned, api_input_str
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "", ""


def call_gemini_api_with_images(user_input: str, system_prompt: str, reference_images: list = None) -> str:
    """
    Calls Gemini API with text input, system prompt, and optional reference images.
    Returns cleaned model.text without markdown fences.
    """
    try:
        # Build properly structured contents with roles
        contents = []
        
        # Add user input as user content
        user_parts = [types.Part(text=user_input)]
        
        # Add reference images to user content if provided
        if reference_images:
            print(f"Adding {len(reference_images)} reference images to designer call")
            for img_path in reference_images:
                if os.path.exists(img_path):
                    try:
                        # Load image data
                        with open(img_path, 'rb') as f:
                            img_data = f.read()
                        
                        # Add image part to user content
                        image_part = types.Part.from_bytes(
                            data=img_data,
                            mime_type="image/png"
                        )
                        user_parts.append(image_part)
                        print(f"✅ Added reference image to designer: {os.path.basename(img_path)}")
                    except Exception as e:
                        print(f"⚠️ Failed to load reference image {img_path}: {e}")
                else:
                    print(f"⚠️ Reference image not found: {img_path}")
        
        # Create user content with all parts
        user_content = types.Content(
            role="user",
            parts=user_parts
        )
        contents.append(user_content)
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt
            )
        )
        cleaned = re.sub(r"```json|```", "", response.text).strip()
        return cleaned
    except Exception as e:
        print(f"Error calling Gemini API with images: {e}")
        return ""


def extract_json(json_str: str) -> list:
    try:
        data = json.loads(json_str)
        return data.get("outputs", [])
    except json.JSONDecodeError:
        print("Invalid JSON; returning empty list.")
        return []

def call_gemini_api_img(user_input: str, system_prompt: str):
    """
    Calls Gemini Image API with user_input and system prompt as separate parameters.
    Returns the raw response object.
    """
    max_retries = 5
    base_delay = 15.0  # Increased delay for image API
    
    for attempt in range(max_retries):
        try:
            return gemini_client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=[user_input, system_prompt],
                config=types.GenerateContentConfig(response_modalities=['IMAGE','TEXT'])
            )
        except Exception as e:
            error_msg = str(e)
            print(f"Error calling Gemini Image API (attempt {attempt + 1}/{max_retries}): {e}")
            
            # Check if it's a retryable error (500, 503, rate limiting, overload)
            if any(code in error_msg for code in ['500', '503', 'INTERNAL', 'UNAVAILABLE', 'overloaded', 'rate limit', 'quota']):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Image API overloaded/rate limited. Waiting {delay:.1f} seconds before retry...")
                    time.sleep(delay)
                    continue
            
            # For non-retryable errors or final attempt, re-raise
            raise e
    
    raise RuntimeError("Failed to call image API after all retries")



def call_gemini_api_img_with_tracking(user_input: str, system_prompt: str) -> tuple:
    """
    Calls Gemini image generation API with separate user_input and system_prompt and returns both response and API input structure.
    Returns (response, api_input_str)
    """
    try:
        # Prepare the final API input for tracking
        api_input_str = f"Model: gemini-2.0-flash-exp-image-generation\n"
        api_input_str += f"User Input: {user_input}\n"
        api_input_str += f"System Prompt: {system_prompt}\n"
        api_input_str += "Config: response_modalities=['IMAGE', 'TEXT']"
        
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=[user_input, system_prompt],
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE', 'TEXT']  # Model requires both modalities
            )
        )
        return response, api_input_str
    except Exception as e:
        print(f"Error calling Gemini image API: {e}")
        return None, ""

def call_gemini_api_mode3_with_tracking(contents: list) -> tuple:
    """
    Calls Gemini image generation API for Mode 3 (with reference images) and returns both response and API input structure.
    Returns (response, api_input_str)
    """
    try:
        # Prepare the final API input for tracking
        api_input_str = f"Model: gemini-2.0-flash-exp-image-generation\nContents:\n"
        for i, content in enumerate(contents):
            api_input_str += f"[{i}] Role: {content.role}\n"
            for j, part in enumerate(content.parts):
                if hasattr(part, 'text') and part.text:
                    # Show complete text for tracking
                    api_input_str += f"    Part {j}: Text: {part.text}\n"
                elif hasattr(part, 'inline_data') and part.inline_data:
                    api_input_str += f"    Part {j}: Image data (mime_type: {part.inline_data.mime_type})\n"
        api_input_str += "Config: response_modalities=['TEXT', 'IMAGE']"
        
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        return response, api_input_str
    except Exception as e:
        print(f"Error calling Gemini Mode 3 API: {e}")
        return None, ""

def get_retry_settings(stage_name: str, mode: str = "sequential"):
    """
    Get adaptive retry settings based on stage and mode.
    Ambient and final stages get more retries and longer delays due to higher quota usage.
    """
    if stage_name in ['ambient', 'final']:
        # More conservative settings for resource-intensive stages
        return {
            'max_retries': 5,
            'retry_delay': 5.0,
            'backoff_multiplier': 1.5
        }
    else:
        # Standard settings for impression and spatial
        return {
            'max_retries': 3,
            'retry_delay': 2.0,
            'backoff_multiplier': 1.2
        }

def designer_seq(prompt: str, descriptor: str, user_pref: dict, session_folder: str, stage_name: str = "unknown") -> list:
    """
    Sequential mode designer - includes user preferences in API call.
    """
    # Pass descriptor and preferences using explicit tags to match prompt expectations
    parts = [f"[USER DESCRIPTION]{descriptor}[/USER DESCRIPTION]"]
    # Only include preferences block when non-empty to avoid leaking '{}' into copied text
    if isinstance(user_pref, dict) and any(str(v).strip() for v in user_pref.values()):
        parts.append(f"[USER_PREFERENCE]{json.dumps(user_pref)}[/USER_PREFERENCE]")
    user_input = "\n".join(parts)
    settings = get_retry_settings(stage_name)
    max_retries = settings['max_retries']
    retry_delay = settings['retry_delay']
    backoff_multiplier = settings['backoff_multiplier']
    attempt = 0
    
    write_status(session_folder, f"Starting sequential concept design for {stage_name}")
    
    while attempt < max_retries:
        attempt += 1
        current_delay = retry_delay * (backoff_multiplier ** (attempt - 1))
        
        try:
            write_status(session_folder, f"Design attempt {attempt}/{max_retries}")
            
            # Call API
            output, _ = call_gemini_api_with_tracking(user_input, prompt)
            
            # Track designer input snapshot (no system prompt)
            track_prompt(
                session_folder, 
                "DESIGNER", 
                f"{stage_name}_design_sequential",
                "",
                f"Attempt: {attempt}/{max_retries}, Mode: Sequential",
                user_input
            )
            
            if not output:
                write_status(session_folder, f"Empty response from API")
                continue
            
            scenes = extract_json(output)
            if scenes and len(scenes) > 0:
                write_status(session_folder, f"Generated {len(scenes)} concepts")
                return scenes
            else:
                write_status(session_folder, f"No valid scenes in response")
                
        except Exception as e:
            error_msg = str(e)
            write_status(session_folder, f"Design error: {error_msg}")
            
            # Check for quota-related errors
            if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                # Increase delay for quota issues
                current_delay = max(current_delay, 10.0)
                write_status(session_folder, f"Quota limit detected, extending delay to {current_delay}s")
        
        if attempt < max_retries:
            write_status(session_folder, f"Retrying in {current_delay}s...")
            time.sleep(current_delay)
    
    error_msg = f"Failed to generate scenes after {max_retries} attempts"
    write_status(session_folder, error_msg)
    raise RuntimeError(error_msg)


# Keep the original designer function for backward compatibility
def designer(prompt: str, descriptor: str, user_pref: dict, session_folder: str) -> list:
    """
    Legacy designer function - redirects to sequential mode.
    """
    return designer_seq(prompt, descriptor, user_pref, session_folder)

def designer_final_one_concept(
    prompt: str,
    positive_tags: list,
    negative_tags: list,
    descriptor: str,
    user_pref: dict,
    session_folder: str,
    stage_name: str = "final"
) -> dict:
    """
    Designer for final stage - generate ONE concept using accumulated tag feedback.
    Returns a single concept dict.
    """
    # Build user input including accumulated tags for the FINAL one-concept prompt
    user_input = (
        f"[USER DESCRIPTION]{descriptor}[/USER DESCRIPTION]\n"
        f"user_preference = {json.dumps(user_pref)}\n"
        f"[POSITIVE_TAGS]{json.dumps(positive_tags or [])}[/POSITIVE_TAGS]\n"
        f"[NEGATIVE_TAGS]{json.dumps(negative_tags or [])}[/NEGATIVE_TAGS]"
    )

    settings = get_retry_settings(stage_name)
    max_retries = settings['max_retries']
    retry_delay = settings['retry_delay']
    backoff_multiplier = settings['backoff_multiplier']
    attempt = 0

    write_status(session_folder, f"Starting final one-concept design (Progressive)")

    while attempt < max_retries:
        attempt += 1
        current_delay = retry_delay * (backoff_multiplier ** (attempt - 1))

        try:
            write_status(session_folder, f"One-concept design attempt {attempt}/{max_retries}")

            # Call API
            output, _ = call_gemini_api_with_tracking(user_input, prompt)
            
            # Track designer input snapshot (no system prompt)
            track_prompt(
                session_folder,
                "DESIGNER",
                f"{stage_name}_design_progressive_one_concept",
                "",
                f"Attempt: {attempt}/{max_retries}, Mode: Progressive One-Concept",
                user_input
            )

            if not output:
                write_status(session_folder, "Empty response from API")
                continue

            # Parse single concept JSON
            try:
                data = json.loads(output)
            except json.JSONDecodeError:
                # Try to clean potential fence leftovers and retry parsing once
                cleaned = re.sub(r"```json|```", "", output).strip()
                try:
                    data = json.loads(cleaned)
                except Exception:
                    write_status(session_folder, "Invalid JSON response from designer")
                    data = None

            if isinstance(data, dict):
                # Expecting { "output": { ... } }
                if "output" in data and isinstance(data["output"], dict):
                    write_status(session_folder, "Generated one final concept")
                    return data["output"]
                # Fallback: if returns list-like under 'outputs', take first
                if "outputs" in data and isinstance(data["outputs"], list) and data["outputs"]:
                    write_status(session_folder, "Generated one final concept (from outputs[0])")
                    return data["outputs"][0]

            write_status(session_folder, "No valid concept found in response")

        except Exception as e:
            error_msg = str(e)
            write_status(session_folder, f"Design error: {error_msg}")
            if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                current_delay = max(current_delay, 10.0)
                write_status(session_folder, f"Quota limit detected, extending delay to {current_delay}s")

        if attempt < max_retries:
            write_status(session_folder, f"Retrying in {current_delay}s...")
            time.sleep(current_delay)

    error_msg = f"Failed to generate one final concept after {max_retries} attempts"
    write_status(session_folder, error_msg)
    raise RuntimeError(error_msg)
def write_status(folder: str, message: str):
    """Write a status message to the session's status file with timestamp."""
    import datetime
    status_file = os.path.join(folder, "status.txt")
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    
    with open(status_file, "a", encoding="utf-8") as f:
        f.write(formatted_message + "\n")
    
    print(formatted_message)

def initialize_prompt_tracking(session_folder: str, descriptor: str, session_type: str = "unknown"):
    """
    Initialize prompt tracking file with session information.
    
    Args:
        session_folder: Path to session folder
        descriptor: User's original description/input
        session_type: Type of session (sequential, parallel, etc.)
    """
    import datetime
    
    prompt_track_file = os.path.join(session_folder, "prompt_track.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    header = f"""{'=' * 80}
PROMPT TRACKING LOG
Session Started: {timestamp}
Session Folder: {os.path.basename(session_folder)}
Session Type: {session_type}
User Description: {descriptor}
{'=' * 80}

This file tracks all inputs sent to LLMs during this session.
Each entry shows:
- LEVEL: DESIGNER (for JSON concept generation) or GENERATOR (for image generation)
- PURPOSE: Specific stage/mode (e.g., impression_design_sequential, final_generation_mode2)
- LLM API INPUT: The final structured input sent to the LLM API
- ADDITIONAL INFO: Context like attempt numbers, tag counts, reference images, etc.

"""
    
    with open(prompt_track_file, "w", encoding="utf-8") as f:
        f.write(header)

def track_prompt(session_folder: str, level: str, purpose: str, prompt_content: str, additional_info: str = "", llm_input: str = ""):
    """
    Track all prompts sent to LLMs with clear labels.
    
    Args:
        session_folder: Path to session folder
        level: 'DESIGNER' or 'GENERATOR' 
        purpose: Specific purpose like 'impression_design', 'final_mode2_generation', etc.
        prompt_content: The actual prompt sent to LLM (optional, use empty string to skip)
        additional_info: Any additional context (e.g., image paths, tag counts)
        llm_input: The final structured input sent to the LLM API (optional)
    """
    import datetime
    
    prompt_track_file = os.path.join(session_folder, "prompt_track.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    separator = "=" * 80
    header = f"\n{separator}\n[{timestamp}] {level} - {purpose}\n{separator}"
    
    # Only include PROMPT CONTENT section if content is provided
    content_section = ""
    if prompt_content.strip():
        content_section = f"\nPROMPT CONTENT:\n{'-' * 40}\n{prompt_content}\n"
    
    # Add LLM input section if provided
    llm_input_section = ""
    if llm_input:
        llm_input_section = f"\nLLM API INPUT:\n{'-' * 40}\n{llm_input}\n"
    
    additional_section = ""
    if additional_info:
        additional_section = f"\nADDITIONAL INFO:\n{'-' * 40}\n{additional_info}\n"
    
    footer = f"\n{separator}\n"
    
    full_entry = header + content_section + llm_input_section + additional_section + footer
    
    with open(prompt_track_file, "a", encoding="utf-8") as f:
        f.write(full_entry)

def _generate_images(contents: list, folder: str, prefix: str, session_folder: str, mode: str, stage_name: str = "unknown") -> list:
    """
    Common image generation logic with adaptive retry settings.
    """
    settings = get_retry_settings(stage_name)
    max_retries = settings['max_retries']
    retry_delay = settings['retry_delay']
    backoff_multiplier = settings['backoff_multiplier']
    attempt = 0
    
    print(f"🎨 Generating images for {stage_name} ({mode} mode)...")
    write_status(session_folder, f"Starting {mode} image generation for {stage_name}")
    # Find the DESIGN_CONCEPT tag in any of the content items
    scene_json = None
    for content in contents:
        if '[DESIGN_CONCEPT]' in content:
            scene_json = content.split('[DESIGN_CONCEPT]')[1].split('[/DESIGN_CONCEPT]')[0]
            break
    if scene_json:
        scene_data = json.loads(scene_json)
        write_status(session_folder, f"Scene: {scene_data.get('concept_name', 'Unknown')}")
    else:
        write_status(session_folder, "Scene: Unknown (no scene data found)")
    
    while attempt < max_retries:
        current_delay = retry_delay * (backoff_multiplier ** attempt)
        
        try:
            message = f"Calling Gemini API (attempt {attempt + 1}/{max_retries})..."
            write_status(session_folder, message)
            
            # Extract user input and system prompt from contents
            # The last item is typically the system prompt, the rest form the user input
            if len(contents) >= 2:
                user_input = "\n".join(contents[:-1])
                system_prompt = contents[-1]
            else:
                # Fallback if there's only one item
                user_input = contents[0] if contents else ""
                system_prompt = ""
            
            # Call API with tracking
            response, api_input = call_gemini_api_img_with_tracking(user_input, system_prompt)
            
            # Track generator prompt with LLM input only (skip for modes with their own content tracking)
            if mode not in ["final_mode2", "final_mode4"]:
                additional_info = f"Generating 4 images with prefix: {prefix}, Mode: {mode}, Stage: {stage_name}"
                track_prompt(
                    session_folder,
                    "GENERATOR", 
                    f"{stage_name}_generation_{mode}",
                    "",  # Only track LLM API input, not the full prompt
                    additional_info,
                    api_input
                )
            
            if not response:
                write_status(session_folder, f"Empty response from API")
                continue
                
            candidate = response.candidates[0]
            
            # check if image content is present
            if getattr(candidate, 'content', None) and getattr(candidate.content, 'parts', None):
                total_parts = len(candidate.content.parts)
                image_parts = sum(1 for part in candidate.content.parts if part.inline_data and hasattr(part.inline_data, 'data'))
                write_status(session_folder, f"API response received: {total_parts} total parts, {image_parts} image parts")
                
                file_paths = []
                image_count = 0
                for i, part in enumerate(candidate.content.parts):
                    if part.inline_data and hasattr(part.inline_data, 'data'):
                        # Only process the first image to ensure 1 image per scene
                        if image_count >= 1:
                            write_status(session_folder, f"Skipping extra image {i+1} (limiting to 1 image per scene)")
                            continue
                        image_count += 1
                        write_status(session_folder, f"Processing image {i+1}...")
                        
                        raw_data = part.inline_data.data
                        try:
                            # Try base64 decode first
                            import base64
                            decoded_data = base64.b64decode(raw_data)
                            img_data = BytesIO(decoded_data)
                        except:
                            # Fallback to direct bytes
                            img_data = BytesIO(raw_data)
                        
                        # Try different image formats
                        img = None
                        for format in ['PNG', 'JPEG', 'WEBP']:
                            try:
                                img_data.seek(0)
                                img = Image.open(img_data)
                                break
                            except:
                                continue
                        
                        if img is None:
                            raise ValueError("Could not identify image format")
                        
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        filename = f"{prefix}_{image_count-1}.png"
                        file_path = os.path.join(folder, filename)
                        img.save(file_path, "PNG")
                        file_paths.append(file_path)
                        
                        write_status(session_folder, f"Saved: {filename}")
                
                if file_paths:
                    scene_name = scene_data.get('concept_name', 'Unknown') if scene_data else 'Unknown'
                    print(f"🖼️  Generated {len(file_paths)} images for scene '{scene_name}' in {stage_name}")
                    write_status(session_folder, f"Scene '{scene_name}': Successfully generated {len(file_paths)} images!")
                    
                    # Alert if more than 1 image per scene
                    if len(file_paths) > 1:
                        write_status(session_folder, f"⚠️  WARNING: Scene '{scene_name}' generated {len(file_paths)} images (expected 1)")
                    
                    return file_paths
                else:
                    write_status(session_folder, f"No valid images in response")
            else:
                write_status(session_folder, f"No image content in API response")
        
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  Image generation error for {stage_name} (attempt {attempt + 1}): {error_msg}")
            write_status(session_folder, f"Error: {error_msg}")
            
            # Check for quota-related errors and adjust delay
            if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                current_delay = max(current_delay, 15.0)  # Minimum 15s for quota issues
                write_status(session_folder, f"Quota limit detected, extending delay to {current_delay}s")
            elif "timeout" in error_msg.lower():
                current_delay = max(current_delay, 8.0)   # Longer delay for timeouts
        
        attempt += 1
        if attempt < max_retries:
            write_status(session_folder, f"Retrying in {current_delay}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(current_delay)
    
    error_msg = f"Failed to generate valid images after {max_retries} attempts"
    print(f"❌ {error_msg} for {stage_name}")
    write_status(session_folder, error_msg)
    raise RuntimeError(error_msg)

def _generate_single_image_with_tags_parallel(contents: list, folder: str, prefix: str, session_folder: str, mode: str, stage_name: str, scene_index: int, tag_prompt: str = None) -> tuple:
    """
    Generate a single image and immediately extract tags for it.
    Returns (file_paths, tags_dict) where tags_dict maps filename to tag list.
    """
    from tag_extraction import extract_visual_elements_from_image
    
    # First generate the image
    file_paths = _generate_single_image_parallel(contents, folder, prefix, session_folder, mode, stage_name, scene_index)
    
    # Then immediately extract tags for each generated image
    tags_dict = {}
    for img_path in file_paths:
        filename = os.path.basename(img_path)
        write_status(session_folder, f"Extracting tags for {filename} (scene {scene_index})...")
        
        try:
            if tag_prompt:
                tags = extract_visual_elements_from_image(img_path, tag_prompt)
            else:
                # Use default prompt
                tags = extract_visual_elements_from_image(img_path)
            
            tags_dict[filename] = tags
            write_status(session_folder, f"Extracted {len(tags)} tags from {filename} (scene {scene_index})")
            
        except Exception as e:
            error_msg = f"Failed to extract tags from {filename} (scene {scene_index}): {str(e)}"
            write_status(session_folder, error_msg)
            tags_dict[filename] = []
    
    return file_paths, tags_dict

def _generate_single_image_parallel(contents: list, folder: str, prefix: str, session_folder: str, mode: str, stage_name: str, scene_index: int) -> list:
    """
    Generate a single image using the same logic as _generate_images but for parallel execution.
    This function is designed to be called concurrently for multiple scenes.
    """
    settings = get_retry_settings(stage_name)
    max_retries = settings['max_retries']
    retry_delay = settings['retry_delay']
    backoff_multiplier = settings['backoff_multiplier']
    attempt = 0
    
    print(f"🎨 Generating image {scene_index} for {stage_name} ({mode} mode)...")
    write_status(session_folder, f"Starting parallel image generation {scene_index} for {stage_name}")
    
    # Find the DESIGN_CONCEPT tag in any of the content items
    scene_json = None
    for content in contents:
        if '[DESIGN_CONCEPT]' in content:
            scene_json = content.split('[DESIGN_CONCEPT]')[1].split('[/DESIGN_CONCEPT]')[0]
            break
    if scene_json:
        scene_data = json.loads(scene_json)
        write_status(session_folder, f"Scene {scene_index}: {scene_data.get('concept_name', 'Unknown')}")
    else:
        write_status(session_folder, f"Scene {scene_index}: Unknown (no scene data found)")
    
    while attempt < max_retries:
        current_delay = retry_delay * (backoff_multiplier ** attempt)
        
        try:
            message = f"Calling Gemini API for scene {scene_index} (attempt {attempt + 1}/{max_retries})..."
            write_status(session_folder, message)
            
            # Extract user input and system prompt from contents
            if len(contents) >= 2:
                user_input = "\n".join(contents[:-1])
                system_prompt = contents[-1]
            else:
                user_input = contents[0] if contents else ""
                system_prompt = ""
            
            # Call API with tracking
            response, api_input = call_gemini_api_img_with_tracking(user_input, system_prompt)
            
            # Track generator prompt with LLM input only (skip for modes with their own content tracking)
            if mode not in ["final_mode2", "final_mode4"]:
                additional_info = f"Generating 1 image with prefix: {prefix}, Mode: {mode}, Stage: {stage_name}, Scene: {scene_index}"
                track_prompt(
                    session_folder,
                    "GENERATOR", 
                    f"{stage_name}_generation_{mode}_scene_{scene_index}",
                    "",  # Only track LLM API input, not the full prompt
                    additional_info,
                    api_input
                )
            
            if not response:
                write_status(session_folder, f"Empty response from API for scene {scene_index}")
                continue
                
            candidate = response.candidates[0]
            
            # check if image content is present
            if getattr(candidate, 'content', None) and getattr(candidate.content, 'parts', None):
                total_parts = len(candidate.content.parts)
                image_parts = sum(1 for part in candidate.content.parts if part.inline_data and hasattr(part.inline_data, 'data'))
                write_status(session_folder, f"API response received for scene {scene_index}: {total_parts} total parts, {image_parts} image parts")
                
                file_paths = []
                image_count = 0
                for i, part in enumerate(candidate.content.parts):
                    if part.inline_data and hasattr(part.inline_data, 'data'):
                        # Only process the first image to ensure 1 image per scene
                        if image_count >= 1:
                            write_status(session_folder, f"Skipping extra image {i+1} for scene {scene_index} (limiting to 1 image per scene)")
                            continue
                        image_count += 1
                        write_status(session_folder, f"Processing image {i+1} for scene {scene_index}...")
                        
                        raw_data = part.inline_data.data
                        try:
                            # Try base64 decode first
                            import base64
                            decoded_data = base64.b64decode(raw_data)
                            img_data = BytesIO(decoded_data)
                        except:
                            # Fallback to direct bytes
                            img_data = BytesIO(raw_data)
                        
                        # Try different image formats
                        img = None
                        for format in ['PNG', 'JPEG', 'WEBP']:
                            try:
                                img_data.seek(0)
                                img = Image.open(img_data)
                                break
                            except:
                                continue
                        
                        if img is None:
                            raise ValueError("Could not identify image format")
                        
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        filename = f"{prefix}_{image_count-1}.png"
                        file_path = os.path.join(folder, filename)
                        img.save(file_path, "PNG")
                        file_paths.append(file_path)
                        
                        write_status(session_folder, f"Saved: {filename} for scene {scene_index}")
                
                if file_paths:
                    scene_name = scene_data.get('concept_name', 'Unknown') if scene_data else 'Unknown'
                    print(f"🖼️  Generated {len(file_paths)} images for scene '{scene_name}' (index {scene_index}) in {stage_name}")
                    write_status(session_folder, f"Scene {scene_index} '{scene_name}': Successfully generated {len(file_paths)} images!")
                    
                    # Alert if more than 1 image per scene
                    if len(file_paths) > 1:
                        write_status(session_folder, f"⚠️  WARNING: Scene {scene_index} '{scene_name}' generated {len(file_paths)} images (expected 1)")
                    
                    return file_paths
                else:
                    write_status(session_folder, f"No valid images in response for scene {scene_index}")
            else:
                write_status(session_folder, f"No image content in API response for scene {scene_index}")
        
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  Image generation error for scene {scene_index} in {stage_name} (attempt {attempt + 1}): {error_msg}")
            write_status(session_folder, f"Scene {scene_index} Error: {error_msg}")
            
            # Check for quota-related errors and adjust delay
            if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                current_delay = max(current_delay, 15.0)  # Minimum 15s for quota issues
                write_status(session_folder, f"Scene {scene_index}: Quota limit detected, extending delay to {current_delay}s")
            elif "timeout" in error_msg.lower():
                current_delay = max(current_delay, 8.0)   # Longer delay for timeouts
        
        attempt += 1
        if attempt < max_retries:
            write_status(session_folder, f"Scene {scene_index}: Retrying in {current_delay}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(current_delay)
    
    error_msg = f"Failed to generate valid images for scene {scene_index} after {max_retries} attempts"
    print(f"❌ {error_msg} for {stage_name}")
    write_status(session_folder, error_msg)
    raise RuntimeError(error_msg)

def _generate_images_parallel(scenes: list, image_prompt: str, descriptor: str, user_pref: dict, folder: str, prefix_base: str, session_folder: str, mode: str, stage_name: str = "unknown") -> list:
    """
    Generate images for multiple scenes in parallel using ThreadPoolExecutor.
    This significantly speeds up image generation by making concurrent API calls.
    """
    print(f"🚀 Starting parallel image generation for {len(scenes)} scenes in {stage_name} ({mode} mode)...")
    write_status(session_folder, f"Starting parallel image generation for {len(scenes)} scenes in {stage_name}")
    
    results = []
    
    # Prepare the contents for each scene
    scene_contents = []
    for i, scene in enumerate(scenes):
        if mode == "sequential":
            contents = [
                f"[DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]",
                f"[DESCRIPTION]{descriptor}[/DESCRIPTION]",
                f"[USER_PREFERENCE]{json.dumps(user_pref)}[/USER_PREFERENCE]",
                image_prompt
            ]
        elif mode == "parallel":
            contents = [
                f"[DESCRIPTION]{descriptor}[/DESCRIPTION]", 
                f"[DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]",
                image_prompt
            ]
        else:
            # Default to sequential mode
            contents = [
                f"[DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]",
                f"[DESCRIPTION]{descriptor}[/DESCRIPTION]",
                f"[USER_PREFERENCE]{json.dumps(user_pref)}[/USER_PREFERENCE]",
                image_prompt
            ]
        
        scene_contents.append((contents, f"{prefix_base}_{i}"))
    
    # Use ThreadPoolExecutor to generate images in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_scene = {}
        for i, (contents, prefix) in enumerate(scene_contents):
            future = executor.submit(
                _generate_single_image_parallel,
                contents, folder, prefix, session_folder, mode, stage_name, i
            )
            future_to_scene[future] = (i, scene_contents[i][0])  # Store scene index and contents
        
        # Collect results as they complete
        for future in as_completed(future_to_scene):
            scene_index, contents = future_to_scene[future]
            try:
                file_paths = future.result()
                # Find the corresponding scene data
                scene = scenes[scene_index]
                results.append((scene, file_paths))
                write_status(session_folder, f"✅ Completed parallel generation for scene {scene_index}")
            except Exception as e:
                error_msg = f"Scene {scene_index} failed: {str(e)}"
                print(f"❌ {error_msg}")
                write_status(session_folder, error_msg)
                # For now, we'll continue with other scenes. In a production system, you might want to handle this differently
                continue
    
    # Sort results by scene index to maintain order
    results.sort(key=lambda x: scenes.index(x[0]))
    
    print(f"🎉 Parallel image generation completed for {len(results)}/{len(scenes)} scenes in {stage_name}")
    write_status(session_folder, f"Parallel image generation completed: {len(results)}/{len(scenes)} scenes successful")
    
    return results

def _generate_images_with_tags_parallel(scenes: list, image_prompt: str, descriptor: str, user_pref: dict, folder: str, prefix_base: str, session_folder: str, mode: str, stage_name: str = "unknown", tag_prompt: str = None) -> tuple:
    """
    Generate images for multiple scenes in parallel and extract tags immediately after each image is generated.
    This provides the optimal flow: parallel image generation + immediate tag extraction.
    Returns (results, all_visual_tags) where results is list of (scene, files) and all_visual_tags is dict.
    """
    print(f"🚀 Starting parallel image generation with immediate tag extraction for {len(scenes)} scenes in {stage_name} ({mode} mode)...")
    write_status(session_folder, f"Starting parallel image generation with immediate tag extraction for {len(scenes)} scenes in {stage_name}")
    
    results = []
    all_visual_tags = {}
    
    # Prepare the contents for each scene
    scene_contents = []
    for i, scene in enumerate(scenes):
        if mode == "sequential":
            contents = [
                f"[DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]",
                f"[DESCRIPTION]{descriptor}[/DESCRIPTION]",
                f"[USER_PREFERENCE]{json.dumps(user_pref)}[/USER_PREFERENCE]",
                image_prompt
            ]
        elif mode == "parallel":
            contents = [
                f"[DESCRIPTION]{descriptor}[/DESCRIPTION]", 
                f"[DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]",
                image_prompt
            ]
        else:
            # Default to sequential mode
            contents = [
                f"[DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]",
                f"[DESCRIPTION]{descriptor}[/DESCRIPTION]",
                f"[USER_PREFERENCE]{json.dumps(user_pref)}[/USER_PREFERENCE]",
                image_prompt
            ]
        
        scene_contents.append((contents, f"{prefix_base}_{i}"))
    
    # Use ThreadPoolExecutor to generate images and extract tags in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_scene = {}
        for i, (contents, prefix) in enumerate(scene_contents):
            future = executor.submit(
                _generate_single_image_with_tags_parallel,
                contents, folder, prefix, session_folder, mode, stage_name, i, tag_prompt
            )
            future_to_scene[future] = (i, scene_contents[i][0])  # Store scene index and contents
        
        # Collect results as they complete
        for future in as_completed(future_to_scene):
            scene_index, contents = future_to_scene[future]
            try:
                file_paths, tags_dict = future.result()
                # Find the corresponding scene data
                scene = scenes[scene_index]
                results.append((scene, file_paths))
                
                # Add tags to the global tags dictionary
                all_visual_tags.update(tags_dict)
                
                write_status(session_folder, f"✅ Completed parallel generation + tag extraction for scene {scene_index}")
            except Exception as e:
                error_msg = f"Scene {scene_index} failed: {str(e)}"
                print(f"❌ {error_msg}")
                write_status(session_folder, error_msg)
                # For now, we'll continue with other scenes. In a production system, you might want to handle this differently
                continue
    
    # Sort results by scene index to maintain order
    results.sort(key=lambda x: scenes.index(x[0]))
    
    print(f"🎉 Parallel image generation with tag extraction completed for {len(results)}/{len(scenes)} scenes in {stage_name}")
    write_status(session_folder, f"Parallel image generation with tag extraction completed: {len(results)}/{len(scenes)} scenes successful")
    write_status(session_folder, f"Total tags extracted: {sum(len(tags) for tags in all_visual_tags.values())}")
    
    return results, all_visual_tags

def generator_seq(prompt: str, descriptor: str, scene: dict, user_pref: dict, folder: str, prefix: str, session_folder: str, stage_name: str = "unknown") -> list:
    """
    Sequential mode generator - includes user preferences in API call.
    """
    contents = [
        f"[DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]",
        f"[DESCRIPTION]{descriptor}[/DESCRIPTION]",
        f"[USER_PREFERENCE]{json.dumps(user_pref)}[/USER_PREFERENCE]",
        prompt
    ]
    return _generate_images(contents, folder, prefix, session_folder, "sequential", stage_name)

def generator_seq_with_tags(prompt: str, descriptor: str, scene: dict, positive_tags: list, negative_tags: list, folder: str, prefix: str, session_folder: str, stage_name: str = "unknown") -> list:
    """
    Sequential mode generator with tags - includes positive and negative tags in API call.
    """
    contents = [
        f"[DESCRIPTION]{descriptor}[/DESCRIPTION]", 
        f"[DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]",
        f"[POSITIVE_TAGS]{json.dumps(positive_tags)}[/POSITIVE_TAGS]",
        f"[NEGATIVE_TAGS]{json.dumps(negative_tags)}[/NEGATIVE_TAGS]",
        prompt
    ]
    return _generate_images(contents, folder, prefix, session_folder, "sequential", stage_name)



# New parallel generator functions for multiple scenes
def generator_seq_parallel(prompt: str, descriptor: str, scenes: list, user_pref: dict, folder: str, prefix_base: str, session_folder: str, stage_name: str = "unknown") -> list:
    """
    Sequential mode generator for multiple scenes in parallel - includes user preferences in API call.
    This generates all 4 images simultaneously instead of one by one.
    """
    return _generate_images_parallel(scenes, prompt, descriptor, user_pref, folder, prefix_base, session_folder, "sequential", stage_name)


# New optimized parallel generator functions with immediate tag extraction
def generator_seq_parallel_with_tags(prompt: str, descriptor: str, scenes: list, user_pref: dict, folder: str, prefix_base: str, session_folder: str, stage_name: str = "unknown", tag_prompt: str = None) -> tuple:
    """
    Sequential mode generator for multiple scenes in parallel with immediate tag extraction.
    This generates all 4 images simultaneously and extracts tags immediately after each image is generated.
    Returns (results, all_visual_tags) for optimal performance.
    """
    return _generate_images_with_tags_parallel(scenes, prompt, descriptor, user_pref, folder, prefix_base, session_folder, "sequential", stage_name, tag_prompt)



# Keep the original generator function for backward compatibility
def generator(prompt: str, scene: dict, user_pref: dict, folder: str, prefix: str, session_folder: str) -> list:
    """Legacy generator function - redirects to sequential mode."""
    return generator_seq(prompt, scene, user_pref, folder, prefix, session_folder)


def generator_final_mode1(prompt: str, descriptor: str, scene: dict, user_pref: dict, folder: str, prefix: str, session_folder: str, stage_name: str = "final") -> list:
    """
    Final Mode 1: JSON only (current default mode).
    Uses selected images' JSON only.
    """
    contents = [
        f"[DESCRIPTION]{descriptor}[/DESCRIPTION]",
        f"[DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]",
        f"[USER_PREFERENCE]{json.dumps(user_pref)}[/USER_PREFERENCE]",
        prompt
    ]
    return _generate_images(contents, folder, prefix, session_folder, "final_mode1", stage_name)


def generator_final_mode2(narrative_prompt: str, image_prompt: str, descriptor: str, scene: dict, user_pref: dict, tag_data: dict, folder: str, prefix: str, session_folder: str, stage_name: str = "final") -> list:
    """
    Final Mode 2: JSON + Tags.
    Uses selected images' JSON + tag data from preferences.json.
    """
    # Parse preferences for LLM-friendly format
    parsed_prefs = parse_preferences_for_llm({"tags": tag_data})
    
    write_status(session_folder, f"Mode 2 - Parsed tag constraints: {parsed_prefs['tag_instruction']}")
    
    # Generate images using the scene JSON object + tag constraints
    contents = [
        f"[DESCRIPTION]{descriptor}[/DESCRIPTION]",
        f"[DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]",
        f"user_preference = {json.dumps(user_pref)}",
        f"POSITIVE TAGS (prefer to include): {parsed_prefs['positive_tags']}",
        f"NEGATIVE TAGS (prefer to avoid): {parsed_prefs['negative_tags']}",
        f"TAG INSTRUCTION: {parsed_prefs['tag_instruction']}",
        image_prompt
    ]
    
    # Track Mode 2 content structure
    mode2_content = "\n".join(contents)
    track_prompt(
        session_folder,
        "GENERATOR",
        f"{stage_name}_mode2_content_structure",
        mode2_content,
        f"Mode 2 - JSON + Tags content for {scene.get('concept_name', 'Unknown Scene')}"
    )
    
    # Note: Detailed tracking will be done in _generate_images function
    return _generate_images(contents, folder, prefix, session_folder, "final_mode2", stage_name)


def generator_final_mode3(narrative_prompt: str, image_prompt: str, descriptor: str, scene: dict, user_pref: dict, tag_data: dict, reference_images: list, folder: str, prefix: str, session_folder: str, stage_name: str = "final") -> list:
    """
    Final Mode 3: JSON + Tags + Images.
    Uses selected images' JSON + tag data + actual selected images.
    """
    # Parse preferences for LLM-friendly format
    parsed_prefs = parse_preferences_for_llm({"tags": tag_data})
    
    write_status(session_folder, f"Mode 3 - Parsed tag constraints: {parsed_prefs['tag_instruction']}")
    write_status(session_folder, f"Mode 3 - Reference images: {[os.path.basename(img) for img in reference_images]}")
    
    # Generate images using the scene JSON object + tag constraints + reference images
    generator_input = f"""[DESCRIPTION]{descriptor}[/DESCRIPTION]
    [DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]
    user_preference = {json.dumps(user_pref)}
    POSITIVE TAGS (prefer to include): {parsed_prefs['positive_tags']}
    NEGATIVE TAGS (prefer to avoid): {parsed_prefs['negative_tags']}
    TAG INSTRUCTION: {parsed_prefs['tag_instruction']}
    Reference images will be provided to the generator for visual inspiration."""
    
    # Use the working pattern from call_gemini_api_with_images for multiple images
    contents = []
    
    # Create user content with text and images
    user_parts = [types.Part(text=generator_input)]
    
    # Add reference images using the working pattern
    if reference_images:
        print(f"Adding {len(reference_images)} reference images to final generation")
        for img_path in reference_images:
            if os.path.exists(img_path):
                try:
                    # Use the working pattern from call_gemini_api_with_images
                    with open(img_path, 'rb') as f:
                        img_data = f.read()
                    
                    # Create image part with proper structure
                    image_part = types.Part.from_bytes(
                        data=img_data,
                        mime_type="image/png"
                    )
                    user_parts.append(image_part)
                    print(f"✅ Added reference image to generation: {os.path.basename(img_path)}")
                except Exception as e:
                    print(f"⚠️ Failed to load reference image {img_path}: {e}")
    
    # Create user content with all parts (text + images)
    user_content = types.Content(
        role="user",
        parts=user_parts
    )
    contents.append(user_content)
    
    # Add image prompt as system content
    system_content = types.Content(
        role="model", 
        parts=[types.Part(text=image_prompt)]
    )
    contents.append(system_content)
    
    # Generate images using the correct API pattern
    max_retries = 3
    current_delay = 3.0
    
    for attempt in range(max_retries):
        try:
            write_status(session_folder, f"Generating final images with reference images (attempt {attempt + 1}/{max_retries})...")
            
            # Use the proper API call with Content structure
            response, api_input = call_gemini_api_mode3_with_tracking(contents)
            
            # Track Mode 3 prompt (with reference images) after API call
            reference_info = f"Reference images: {[os.path.basename(img) for img in reference_images if os.path.exists(img)]}"
            tag_info = f"Positive tags: {len(parsed_prefs['positive_tags'])}, Negative tags: {len(parsed_prefs['negative_tags'])}"
            scene_info = f"Scene: {scene.get('concept_name', 'Unknown')}"
            additional_info = f"{reference_info}, {tag_info}, {scene_info}"
            
            track_prompt(
                session_folder,
                "GENERATOR", 
                f"{stage_name}_generation_final_mode3",
                "",  # Only track LLM API input, not the generator input
                additional_info,
                api_input
            )
            
            if response:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content.parts:
                    file_paths = []
                    image_count = 0
                    
                    for i, part in enumerate(candidate.content.parts):
                        # Use the correct attribute from the sample: part.inline_data.data
                        if hasattr(part, 'inline_data') and part.inline_data is not None:
                            # Only process the first image to ensure 1 image per scene
                            if image_count >= 1:
                                write_status(session_folder, f"Skipping extra image {i+1} (limiting to 1 image per scene)")
                                continue
                            image_count += 1
                            
                            try:
                                # Use the robust pattern from _generate_images
                                raw_data = part.inline_data.data
                                try:
                                    # Try base64 decode first
                                    import base64
                                    decoded_data = base64.b64decode(raw_data)
                                    img_data = BytesIO(decoded_data)
                                except:
                                    # Fallback to direct bytes
                                    img_data = BytesIO(raw_data)
                                
                                # Try different image formats
                                img = None
                                for format in ['PNG', 'JPEG', 'WEBP']:
                                    try:
                                        img_data.seek(0)
                                        img = Image.open(img_data)
                                        break
                                    except:
                                        continue
                                
                                if img is None:
                                    raise ValueError("Could not identify image format")
                                
                                # Convert to RGB if necessary
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                
                                filename = f"{prefix}_{image_count-1}.png"
                                file_path = os.path.join(folder, filename)
                                img.save(file_path, "PNG")
                                file_paths.append(file_path)
                                
                                write_status(session_folder, f"Saved: {filename}")
                            except Exception as e:
                                print(f"⚠️ Failed to process image part {i}: {e}")
                    
                    if file_paths:
                        print(f"🖼️  Generated {len(file_paths)} images for {stage_name}")
                        write_status(session_folder, f"Successfully generated {len(file_paths)} images!")
                        return file_paths
                    else:
                        write_status(session_folder, f"No valid images in response")
                else:
                    write_status(session_folder, f"No image content in API response")
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  Image generation error for {stage_name} (attempt {attempt + 1}): {error_msg}")
            write_status(session_folder, f"Error: {error_msg}")
            
            if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                current_delay = max(current_delay, 15.0)
                write_status(session_folder, f"Quota limit detected, extending delay to {current_delay}s")
            elif "timeout" in error_msg.lower():
                current_delay = max(current_delay, 8.0)
        
        if attempt < max_retries - 1:
            write_status(session_folder, f"Retrying in {current_delay}s... (attempt {attempt + 2}/{max_retries})")
            time.sleep(current_delay)
    
    error_msg = f"Failed to generate valid images after {max_retries} attempts"
    print(f"❌ {error_msg} for {stage_name}")
    write_status(session_folder, error_msg)
    raise RuntimeError(error_msg)


# Optimized final mode generator functions with parallel generation and immediate tag extraction
def generator_final_mode1_optimized(prompt: str, descriptor: str, scenes: list, user_pref: dict, folder: str, prefix_base: str, session_folder: str, stage_name: str = "final") -> tuple:
    """
    Optimized Final Mode 1: JSON only with parallel generation and immediate tag extraction.
    """
    # Prepare contents for each scene
    scene_contents = []
    for i, scene in enumerate(scenes):
        contents = [
            f"[DESCRIPTION]{descriptor}[/DESCRIPTION]",
            f"[DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]",
            f"[USER_PREFERENCE]{json.dumps(user_pref)}[/USER_PREFERENCE]",
            prompt
        ]
        scene_contents.append((contents, f"{prefix_base}_{i}"))
    
    # Use the optimized parallel generation with immediate tag extraction
    return _generate_images_with_tags_parallel(scenes, prompt, descriptor, user_pref, folder, prefix_base, session_folder, "sequential", stage_name, None)

def generator_final_mode2_optimized(narrative_prompt: str, image_prompt: str, descriptor: str, scenes: list, user_pref: dict, tag_data: dict, folder: str, prefix_base: str, session_folder: str, stage_name: str = "final") -> tuple:
    """
    Optimized Final Mode 2: JSON + Tags with parallel generation and immediate tag extraction.
    """
    # Parse preferences for LLM-friendly format
    parsed_prefs = parse_preferences_for_llm({"tags": tag_data})
    
    write_status(session_folder, f"Mode 2 - Parsed tag constraints: {parsed_prefs['tag_instruction']}")
    
    # Prepare contents for each scene with tag constraints
    scene_contents = []
    for i, scene in enumerate(scenes):
        contents = [
            f"[DESCRIPTION]{descriptor}[/DESCRIPTION]",
            f"[DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]",
            f"[USER_PREFERENCE]{json.dumps(user_pref)}[/USER_PREFERENCE]",
            f"[TAG_CONSTRAINTS]{parsed_prefs['tag_instruction']}[/TAG_CONSTRAINTS]",
            image_prompt
        ]
        scene_contents.append((contents, f"{prefix_base}_{i}"))
    
    # Use the optimized parallel generation with immediate tag extraction
    return _generate_images_with_tags_parallel(scenes, image_prompt, descriptor, user_pref, folder, prefix_base, session_folder, "sequential", stage_name, None)

def generator_final_mode3_optimized(narrative_prompt: str, image_prompt: str, descriptor: str, scenes: list, user_pref: dict, tag_data: dict, reference_images: list, folder: str, prefix_base: str, session_folder: str, stage_name: str = "final") -> tuple:
    """
    Optimized Final Mode 3: JSON + Tags + Images with parallel generation and immediate tag extraction.
    """
    # Parse preferences for LLM-friendly format
    parsed_prefs = parse_preferences_for_llm({"tags": tag_data})
    
    write_status(session_folder, f"Mode 3 - Parsed tag constraints: {parsed_prefs['tag_instruction']}")
    write_status(session_folder, f"Mode 3 - Reference images: {[os.path.basename(img) for img in reference_images]}")
    
    # Prepare contents for each scene with tag constraints and reference images
    scene_contents = []
    for i, scene in enumerate(scenes):
        # Format reference images for the prompt
        ref_images_text = "\n".join([f"Reference {j+1}: {os.path.basename(img)}" for j, img in enumerate(reference_images)])
        
        contents = [
            f"[DESCRIPTION]{descriptor}[/DESCRIPTION]",
            f"[DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]",
            f"[USER_PREFERENCE]{json.dumps(user_pref)}[/USER_PREFERENCE]",
            f"[TAG_CONSTRAINTS]{parsed_prefs['tag_instruction']}[/TAG_CONSTRAINTS]",
            f"[REFERENCE_IMAGES]{ref_images_text}[/REFERENCE_IMAGES]",
            image_prompt
        ]
        scene_contents.append((contents, f"{prefix_base}_{i}"))
    
    # Use the optimized parallel generation with immediate tag extraction
    return _generate_images_with_tags_parallel(scenes, image_prompt, descriptor, user_pref, folder, prefix_base, session_folder, "sequential", stage_name, None)

def generator_final_mode4_optimized(prompt: str, descriptor: str, scenes: list, enhanced_user_pref: dict, folder: str, prefix_base: str, session_folder: str, stage_name: str = "final") -> tuple:
    """
    Optimized Final Mode 4: Enhanced User Preferences with parallel generation and immediate tag extraction.
    """
    write_status(session_folder, f"Mode 4 - Using enhanced user preferences with per-layer tags")
    
    # Format per-layer tag preferences
    tag_instructions = []
    for layer_name, layer_prefs in enhanced_user_pref.items():
        if isinstance(layer_prefs, dict):
            include_tags = layer_prefs.get('prefer_to_include', [])
            avoid_tags = layer_prefs.get('prefer_to_avoid', [])
            
            if include_tags or avoid_tags:
                layer_instruction = f"{layer_name}:"
                if include_tags:
                    layer_instruction += f" Include: {', '.join(include_tags)}"
                if avoid_tags:
                    layer_instruction += f" Avoid: {', '.join(avoid_tags)}"
                tag_instructions.append(layer_instruction)
    
    tag_constraints = "\n".join(tag_instructions) if tag_instructions else "No specific tag constraints"
    write_status(session_folder, f"Mode 4 - Tag constraints: {tag_constraints}")
    
    # Prepare contents for each scene with enhanced preferences
    scene_contents = []
    for i, scene in enumerate(scenes):
        contents = [
            f"[DESCRIPTION]{descriptor}[/DESCRIPTION]",
            f"[DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]",
            f"[ENHANCED_USER_PREFERENCE]{json.dumps(enhanced_user_pref)}[/ENHANCED_USER_PREFERENCE]",
            f"[TAG_CONSTRAINTS]{tag_constraints}[/TAG_CONSTRAINTS]",
            prompt
        ]
        scene_contents.append((contents, f"{prefix_base}_{i}"))
    
    # Use the optimized parallel generation with immediate tag extraction
    return _generate_images_with_tags_parallel(scenes, prompt, descriptor, enhanced_user_pref, folder, prefix_base, session_folder, "sequential", stage_name, None)

def generator_final_mode4(narrative_prompt: str, image_prompt: str, descriptor: str, scene: dict, enhanced_user_pref: dict, folder: str, prefix: str, session_folder: str, stage_name: str = "final") -> list:
    """
    Final Mode 4: Enhanced User Preferences with Per-Layer Tags.
    Uses the comprehensive user preference structure from mode4.py including:
    - Full JSON entries for selected images
    - Per-layer tag preferences (prefer_to_include, prefer_to_avoid)
    - Other preferences for unselected images
    """
    write_status(session_folder, f"Mode 4 - Using enhanced user preferences with per-layer tags")
    
    # Format per-layer tag preferences
    layer_tag_instructions = []
    for layer in ["impression", "spatial", "objects", "ambient"]:
        layer_data = enhanced_user_pref.get(layer, {})
        visual_prefs = layer_data.get("visual_elements_preference", {})
        
        if visual_prefs.get("prefer_to_include") or visual_prefs.get("prefer_to_avoid"):
            layer_name = layer.upper()
            include_tags = visual_prefs.get("prefer_to_include", [])
            avoid_tags = visual_prefs.get("prefer_to_avoid", [])
            
            layer_instruction = f"{layer_name} LAYER:"
            if include_tags:
                layer_instruction += f" Include: {', '.join(include_tags)}"
            if avoid_tags:
                layer_instruction += f" Avoid: {', '.join(avoid_tags)}"
            
            layer_tag_instructions.append(layer_instruction)
    
    # Format other preferences from unselected images
    other_prefs = enhanced_user_pref.get("other_preference", {})
    other_context = []
    if other_prefs.get("preferred"):
        other_context.append(f"Additional preferred elements: {', '.join(other_prefs['preferred'])}")
    if other_prefs.get("avoid"):
        other_context.append(f"Additional elements to avoid: {', '.join(other_prefs['avoid'])}")
    
    # Create comprehensive prompt for mode 4
    contents = [
        f"[DESCRIPTION]{descriptor}[/DESCRIPTION]",
        f"[DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]",
        "",
        "ENHANCED USER PREFERENCES:",
        "Per-layer tag preferences:",
        "\n".join(layer_tag_instructions) if layer_tag_instructions else "No specific layer preferences",
        "",
        "Additional context from user feedback:",
        "\n".join(other_context) if other_context else "No additional context",
        "",
        "INSTRUCTION: Generate images that incorporate the selected elements from each layer (impression, spatial, ambient) while respecting the per-layer tag preferences and additional context.",
        "",
        image_prompt
    ]
    
    # Track Mode 4 content structure
    mode4_content = "\n".join(contents)
    track_prompt(
        session_folder,
        "GENERATOR",
        f"{stage_name}_mode4_content_structure",
        mode4_content,
        f"Mode 4 - Enhanced User Preferences content for {scene.get('concept_name', 'Unknown Scene')}, Layers: {len(layer_tag_instructions)} tag instructions, Context: {len(other_context)} elements"
    )
    
    write_status(session_folder, f"Mode 4 - Generated {len(layer_tag_instructions)} layer-specific tag instructions")
    if other_context:
        write_status(session_folder, f"Mode 4 - Added {len(other_context)} additional context elements")
    
    return _generate_images(contents, folder, prefix, session_folder, "final_mode4", stage_name)


def parse_preferences_for_llm(preferences: dict) -> dict:
    """
    Parse the complex preferences.json format into a simpler, LLM-friendly format.
    
    Args:
        preferences: Original preferences dict with nested tag structure
        
    Returns:
        Simplified dict with positive_tags, negative_tags, and instruction strings
    """
    parsed = {
        "selections": preferences.get("selections", {}),
        "positive_tags": [],
        "negative_tags": [],
        "tag_instruction": "",
        "detailed_constraints": ""
    }
    
    # Extract tags from both parallel and sequential formats
    tags_data = preferences.get("tags", {})
    all_tags = []
    
    # Handle parallel format: {"parallel": [...]}
    if "parallel" in tags_data:
        all_tags.extend(tags_data["parallel"])
    
    # Handle sequential format: {"spatial": [...], "impression": [...], ...}
    for stage in ["impression", "spatial", "objects", "ambient"]:
        if stage in tags_data:
            all_tags.extend(tags_data[stage])
    
    # Separate positive and negative tags
    positive_tags = []
    negative_tags = []
    
    for tag_item in all_tags:
        if isinstance(tag_item, dict):
            tag_name = tag_item.get("tag", "")
            preference = tag_item.get("preference", "")
            source = tag_item.get("source_image", "")
            
            if preference == "positive":
                positive_tags.append(tag_name)
            elif preference == "negative":
                negative_tags.append(tag_name)
    
    # Remove duplicates while preserving order
    parsed["positive_tags"] = list(dict.fromkeys(positive_tags))
    parsed["negative_tags"] = list(dict.fromkeys(negative_tags))
    
    # Create clear instruction strings for LLM
    if positive_tags or negative_tags:
        instructions = []
        
        if positive_tags:
            instructions.append(f"PREFER TO INCLUDE: {', '.join(positive_tags)}")
        
        if negative_tags:
            instructions.append(f"PREFER TO AVOID: {', '.join(negative_tags)}")
        
        parsed["tag_instruction"] = ". ".join(instructions) + "."
        
        # Create detailed constraints for complex prompts
        constraints = []
        if positive_tags:
            constraints.append(f"Preferred elements: {positive_tags}")
        if negative_tags:
            constraints.append(f"Elements to minimize: {negative_tags}")
        
        parsed["detailed_constraints"] = " | ".join(constraints)
    
    return parsed


# Keep the original generator function for backward compatibility

def generator_final_one_concept(
    prompt: str,
    descriptor: str,
    scene: dict,
    user_pref: dict,
    positive_tags: list,
    negative_tags: list,
    folder: str,
    prefix: str,
    session_folder: str,
    stage_name: str = "final"
) -> list:
    """
    Final Progressive Mode (Mode 5): Generate ONE image from ONE concept.
    Includes accumulated tag blocks in generator input.
    """
    contents = [
        f"[DESCRIPTION]{descriptor}[/DESCRIPTION]",
        f"[DESIGN_CONCEPT]{json.dumps(scene)}[/DESIGN_CONCEPT]",
        f"[USER_PREFERENCE]{json.dumps(user_pref)}[/USER_PREFERENCE]",
        f"[POSITIVE_TAGS]{json.dumps(positive_tags or [])}[/POSITIVE_TAGS]",
        f"[NEGATIVE_TAGS]{json.dumps(negative_tags or [])}[/NEGATIVE_TAGS]",
        prompt
    ]
    # Track generator input snapshot for mode 5 with key parts only
    track_prompt(
        session_folder,
        "GENERATOR",
        f"{stage_name}_generation_final_mode5",
        "",
        f"Prefix: {prefix}, Positive tags: {len(positive_tags or [])}, Negative tags: {len(negative_tags or [])}",
        "\n".join(contents[:-1])  # exclude system prompt text
    )
    return _generate_images(contents, folder, prefix, session_folder, "final_mode5", stage_name)
