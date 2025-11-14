import os
import json
from datetime import datetime
from tag_extraction import extract_visual_elements_from_image, prompt, prompt_impression, prompt_spatial, prompt_objects, prompt_ambient
from util import sanitize_folder_name, designer, generator, write_status, designer_seq, generator_seq, generator_seq_parallel, generator_seq_parallel_with_tags, initialize_prompt_tracking
from prompt import (
    IMPRESSION_PROMPT, IMPRESSION_GENERATOR_PROMPT
)

def run_stage_seq(name: str,
              narrative_prompt: str,
              image_prompt: str,
              descriptor: str,
              user_pref: dict,
              session_folder: str) -> tuple[list, dict]:
    """
    Run a single stage of the sequential generation pipeline.
    """
    from tag_extraction import extract_visual_elements_from_image
    
    write_status(session_folder, f"Starting {name.upper()} stage (Sequential)")
    
    # Create stage folder
    folder = os.path.join(session_folder, name)
    os.makedirs(folder, exist_ok=True)
    write_status(session_folder, f"Created folder: {folder}")
    
    # Generate scene descriptions using sequential designer
    write_status(session_folder, f"Generating concepts with user preferences...")
    scenes = designer_seq(narrative_prompt, descriptor, user_pref, session_folder, name)
    
    if not scenes:
        error_msg = f"No scenes generated for {name} stage"
        write_status(session_folder, error_msg)
        raise RuntimeError(error_msg)
    
    write_status(session_folder, f"Generated {len(scenes)} concepts")
    
    # Save scenes JSON
    scenes_file = os.path.join(folder, f"{name}.json")
    with open(scenes_file, "w") as f:
        json.dump(scenes, f, indent=2)
    write_status(session_folder, f"Saved scenes to: {name}.json")
    
    # Generate images for each scene using sequential generator
    results = []
    all_visual_tags = {}
    
    for i, scene in enumerate(scenes):
        concept_name = scene.get('concept_name', f'Scene {i+1}')
        write_status(session_folder, f"Generating images for: {concept_name}")
        
        prefix = f"{name}_{i}"
        
        try:
            files = generator_seq(image_prompt, descriptor,scene, user_pref, folder, prefix, session_folder, name)
            results.append((scene, files))
            write_status(session_folder, f"Generated {len(files)} images for {concept_name}")
            
            # Extract tags for generated images
            write_status(session_folder, f"Extracting visual tags for {concept_name}...")
            
            for img_path in files:
                filename = os.path.basename(img_path)
                write_status(session_folder, f"Analyzing: {filename}")
                
                try:
                    # Use appropriate prompt based on stage
                    if name == "impression":
                        tags = extract_visual_elements_from_image(img_path, prompt_impression)
                    elif name == "spatial":
                        tags = extract_visual_elements_from_image(img_path, prompt_spatial)
                    elif name == "ambient":
                        tags = extract_visual_elements_from_image(img_path, prompt_ambient)
                    elif name == "objects":
                        tags = extract_visual_elements_from_image(img_path, prompt_objects)
                    else:
                        # Use general prompt for other stages
                        tags = extract_visual_elements_from_image(img_path, prompt)
                    all_visual_tags[filename] = tags
                    write_status(session_folder, f"Extracted {len(tags)} tags from {filename}")
                except Exception as e:
                    error_msg = f"Failed to extract tags from {filename}: {str(e)}"
                    write_status(session_folder, error_msg)
                    all_visual_tags[filename] = []
            
        except Exception as e:
            error_msg = f"Failed to generate images for {concept_name}: {str(e)}"
            write_status(session_folder, error_msg)
            continue
    
    # Save visual tags
    if all_visual_tags:
        tags_path = os.path.join(folder, "visual_tags.json")
        with open(tags_path, "w") as tagf:
            json.dump(all_visual_tags, tagf, indent=2)
        total_tags = sum(len(tags) for tags in all_visual_tags.values())
        write_status(session_folder, f"Saved {total_tags} total tags to: visual_tags.json")
    
    write_status(session_folder, f"{name.upper()} stage completed successfully! (Sequential)")
    write_status(session_folder, f"Generated {len(results)} concepts with {sum(len(files) for _, files in results)} total images")
    
    return results, user_pref

def run_stage_seq_parallel(name: str,
              narrative_prompt: str,
              image_prompt: str,
              descriptor: str,
              user_pref: dict,
              session_folder: str) -> tuple[list, dict]:
    """
    Run a single stage of the sequential generation pipeline with parallel image generation.
    This generates all 4 images simultaneously instead of one by one for much faster processing.
    """
    from tag_extraction import extract_visual_elements_from_image
    
    write_status(session_folder, f"Starting {name.upper()} stage (Sequential with Parallel Images)")
    
    # Create stage folder
    folder = os.path.join(session_folder, name)
    os.makedirs(folder, exist_ok=True)
    write_status(session_folder, f"Created folder: {folder}")
    
    # Generate scene descriptions using sequential designer
    write_status(session_folder, f"Generating concepts with user preferences...")
    scenes = designer_seq(narrative_prompt, descriptor, user_pref, session_folder, name)
    
    if not scenes:
        error_msg = f"No scenes generated for {name} stage"
        write_status(session_folder, error_msg)
        raise RuntimeError(error_msg)
    
    write_status(session_folder, f"Generated {len(scenes)} concepts")
    
    # Save scenes JSON
    scenes_file = os.path.join(folder, f"{name}.json")
    with open(scenes_file, "w") as f:
        json.dump(scenes, f, indent=2)
    write_status(session_folder, f"Saved scenes to: {name}.json")
    
    # Generate images for all scenes in parallel using the new parallel generator
    write_status(session_folder, f"Generating images for all {len(scenes)} scenes in parallel...")
    prefix_base = name
    
    try:
        results = generator_seq_parallel(image_prompt, descriptor, scenes, user_pref, folder, prefix_base, session_folder, name)
        write_status(session_folder, f"Successfully generated images for {len(results)} scenes in parallel")
    except Exception as e:
        error_msg = f"Parallel image generation failed: {str(e)}"
        write_status(session_folder, error_msg)
        raise RuntimeError(error_msg)
    
    # Extract tags for generated images (this part remains sequential as it's fast)
    all_visual_tags = {}
    write_status(session_folder, f"Extracting visual tags from generated images...")
    
    for i, (scene, files) in enumerate(results):
        concept_name = scene.get('concept_name', f'Scene {i+1}')
        write_status(session_folder, f"Extracting tags for: {concept_name}")
        
        for img_path in files:
            filename = os.path.basename(img_path)
            write_status(session_folder, f"Analyzing: {filename}")
            
            try:
                # Use appropriate prompt based on stage
                if name == "impression":
                    tags = extract_visual_elements_from_image(img_path, prompt_impression)
                elif name == "spatial":
                    tags = extract_visual_elements_from_image(img_path, prompt_spatial)
                elif name == "ambient":
                    tags = extract_visual_elements_from_image(img_path, prompt_ambient)
                else:
                    tags = extract_visual_elements_from_image(img_path, prompt)
                
                all_visual_tags[filename] = tags
                write_status(session_folder, f"Extracted {len(tags)} tags from {filename}")
                
            except Exception as e:
                error_msg = f"Failed to extract tags from {filename}: {str(e)}"
                write_status(session_folder, error_msg)
                all_visual_tags[filename] = []
    
    # Save visual tags
    if all_visual_tags:
        tags_path = os.path.join(folder, "visual_tags.json")
        with open(tags_path, "w") as tagf:
            json.dump(all_visual_tags, tagf, indent=2)
        total_tags = sum(len(tags) for tags in all_visual_tags.values())
        write_status(session_folder, f"Saved {total_tags} total tags to: visual_tags.json")
    
    write_status(session_folder, f"{name.upper()} stage completed successfully! (Sequential with Parallel Images)")
    write_status(session_folder, f"Generated {len(results)} concepts with {sum(len(files) for _, files in results)} total images")
    
    return results, user_pref

def run_stage_seq_parallel_optimized(name: str,
              narrative_prompt: str,
              image_prompt: str,
              descriptor: str,
              user_pref: dict,
              session_folder: str) -> tuple[list, dict]:
    """
    Run a single stage with optimized parallel image generation and immediate tag extraction.
    This is the fastest approach: parallel image generation + immediate tag extraction.
    """
    write_status(session_folder, f"Starting {name.upper()} stage (Optimized Parallel)")
    
    # Create stage folder
    folder = os.path.join(session_folder, name)
    os.makedirs(folder, exist_ok=True)
    write_status(session_folder, f"Created folder: {folder}")
    
    # Generate scene descriptions using sequential designer
    write_status(session_folder, f"Generating concepts with user preferences...")
    scenes = designer_seq(narrative_prompt, descriptor, user_pref, session_folder, name)
    
    if not scenes:
        error_msg = f"No scenes generated for {name} stage"
        write_status(session_folder, error_msg)
        raise RuntimeError(error_msg)
    
    write_status(session_folder, f"Generated {len(scenes)} concepts")
    
    # Save scenes JSON
    scenes_file = os.path.join(folder, f"{name}.json")
    with open(scenes_file, "w") as f:
        json.dump(scenes, f, indent=2)
    write_status(session_folder, f"Saved scenes to: {name}.json")
    
    # Generate images and extract tags in parallel with immediate tag extraction
    write_status(session_folder, f"Generating images and extracting tags in parallel for all {len(scenes)} scenes...")
    prefix_base = name
    
    # Determine the appropriate tag prompt based on stage
    tag_prompt = None
    if name == "impression":
        tag_prompt = prompt_impression
    elif name == "spatial":
        tag_prompt = prompt_spatial
    elif name == "ambient":
        tag_prompt = prompt_ambient
    elif name == "objects":
        tag_prompt = prompt_objects
    else:
        tag_prompt = prompt  # Use general prompt
    
    try:
        results, all_visual_tags = generator_seq_parallel_with_tags(
            image_prompt, descriptor, scenes, user_pref, folder, prefix_base, session_folder, name, tag_prompt
        )
        write_status(session_folder, f"Successfully generated images and extracted tags for {len(results)} scenes in parallel")
    except Exception as e:
        error_msg = f"Optimized parallel generation failed: {str(e)}"
        write_status(session_folder, error_msg)
        raise RuntimeError(error_msg)
    
    # Save visual tags
    if all_visual_tags:
        tags_path = os.path.join(folder, "visual_tags.json")
        with open(tags_path, "w") as tagf:
            json.dump(all_visual_tags, tagf, indent=2)
        total_tags = sum(len(tags) for tags in all_visual_tags.values())
        write_status(session_folder, f"Saved {total_tags} total tags to: visual_tags.json")
    
    write_status(session_folder, f"{name.upper()} stage completed successfully! (Optimized Parallel)")
    write_status(session_folder, f"Generated {len(results)} concepts with {sum(len(files) for _, files in results)} total images")
    
    return results, user_pref




# Keep the original run_stage function for backward compatibility
def run_stage(name: str,
              narrative_prompt: str,
              image_prompt: str,
              descriptor: str,
              user_pref: dict,
              session_folder: str) -> tuple[list, dict]:
    """
    Legacy run_stage function - redirects to sequential mode.
    """
    return run_stage_seq(name, narrative_prompt, image_prompt, descriptor, user_pref, session_folder)

def main():
    descriptor = input("Enter a descriptor: ")
    safe_desc = sanitize_folder_name(descriptor)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    base = os.path.join(os.getcwd(), "session_folder")
    os.makedirs(base, exist_ok=True)
    session_folder = os.path.join(base, f"{safe_desc}_{timestamp}")
    os.makedirs(session_folder, exist_ok=True)

    # Initialize prompt tracking
    initialize_prompt_tracking(session_folder, descriptor, "main_script")

    # initialize preferences
    user_pref = {
        "impression": "",
        "spatial": "",
        "objects": "",
        "ambient": ""
    }

    # run stages in order
    user_pref = run_stage(
        "impression", 
        IMPRESSION_PROMPT, 
        IMPRESSION_GENERATOR_PROMPT,
        descriptor, 
        user_pref, 
        session_folder
    )
    
    

if __name__ == "__main__":
    main()
