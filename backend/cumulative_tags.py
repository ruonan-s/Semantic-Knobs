import os
import json
from datetime import datetime
from tag_extraction import extract_visual_elements_from_image, prompt_impression, prompt_spatial, prompt_objects, prompt_ambient
from util import sanitize_folder_name, designer, generator_seq_with_tags, write_status, designer_seq
from prompt_tag_acc import (
    IMPRESSION_PROMPT, IMPRESSION_GENERATOR_PROMPT,
    SPATIAL_PROMPT, SPATIAL_GENERATOR_PROMPT,
    OBJECTS_PROMPT, OBJECTS_GENERATOR_PROMPT,
    AMBIENT_PROMPT, AMBIENT_GENERATOR_PROMPT,
    FINAL_PROMPT_TAGS, FINAL_GENERATOR_PROMPT_TAGS,
    FINAL_PROMPT_CUMULATIVE, FINAL_GENERATOR_PROMPT_CUMULATIVE,
    FINAL_PROMPT_CUMULATIVE_TAGS, FINAL_GENERATOR_PROMPT_CUMULATIVE_TAGS,
    FINAL_GENERATOR_PROMPT_CUMULATIVE_IMGS,
    FINAL_PROMPT_one_concept as FINAL_PROMPT_PROGRESSIVE,
    FINAL_GENERATOR_PROMPT_one_concept as FINAL_GENERATOR_PROMPT_PROGRESSIVE
)


def start_cumulative_tags_stage(name: str,
                               narrative_prompt: str,
                               image_prompt: str,
                               descriptor: str,
                               user_pref: dict,
                               session_folder: str) -> dict:
    """
    Start a cumulative tags stage by generating all 4 concepts first, then generating image for first concept.
    
    Process:
    1. Generate all 4 scene descriptions using designer (same as sequential/parallel)
    2. Generate image for first concept (concept 0) 
    3. Extract visual tags from the generated image
    4. Return first concept data and all scenes for subsequent concepts
    """
    write_status(session_folder, f"Starting {name.upper()} stage (Cumulative Tags)")
    
    # Create stage folder
    folder = os.path.join(session_folder, name)
    os.makedirs(folder, exist_ok=True)
    
    # Generate all 4 scene descriptions first (same as sequential/parallel modes)
    write_status(session_folder, f"Generating 4 concepts for {name} stage...")
    scenes = designer_seq(narrative_prompt, descriptor, user_pref, session_folder, name)
    
    # Ensure user_description fields copy the descriptor exactly for cumulative tags mode
    for scene in scenes:
        if isinstance(scene, dict):
            if 'user_description' in scene:
                scene['user_description'] = descriptor
    
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
    
    # Generate image for first concept only
    return generate_concept_with_cumulative_tags(
        name, image_prompt, descriptor, scenes, 0, [], [], folder, session_folder
    )


def generate_concept_with_cumulative_tags(name: str,
                                        image_prompt: str,
                                        descriptor: str,
                                        scenes: list,
                                        concept_index: int,
                                        cumulative_positive_tags: list,
                                        cumulative_negative_tags: list,
                                        folder: str,
                                        session_folder: str) -> dict:
    """
    Generate image for a specific concept using cumulative tags from previous concepts.
    
    Args:
        name: Stage name
        image_prompt: Image generation prompt
        descriptor: User description
        scenes: All 4 scene descriptions (already generated)
        concept_index: Which concept to generate (0-3)
        cumulative_positive_tags: Tags to emphasize from previous concepts
        cumulative_negative_tags: Tags to avoid from previous concepts
        folder: Stage folder path
        session_folder: Session folder path
    """
    if concept_index >= len(scenes):
        raise RuntimeError(f"Concept index {concept_index} out of range for {len(scenes)} scenes")
    
    # Select appropriate tag extraction prompt based on stage
    tag_prompt = {
        'impression': prompt_impression,
        'spatial': prompt_spatial,
        'objects': prompt_objects,  # Using objects prompt for objects stage
        'ambient': prompt_ambient
    }.get(name, prompt_impression)
    
    # Get the specific scene for this concept
    scene = scenes[concept_index]
    concept_name = f"{name}_{concept_index}_{0}"
    write_status(session_folder, f"Processing concept {concept_index + 1}: {concept_name}")
    
    # Generate image with cumulative tags
    write_status(session_folder, f"Generating image for concept {concept_index + 1} with {len(cumulative_positive_tags)} positive and {len(cumulative_negative_tags)} negative tags...")
    
    # ===== DEBUG LOGS: Tag Input to Generator =====
    print(f"🎨 [DEBUG] GENERATOR INPUT:")
    print(f"    Concept: {concept_name}")
    print(f"    Positive Tags: {cumulative_positive_tags}")
    print(f"    Negative Tags: {cumulative_negative_tags}")
    write_status(session_folder, f"[DEBUG] Generator input - Positive: {cumulative_positive_tags}")
    write_status(session_folder, f"[DEBUG] Generator input - Negative: {cumulative_negative_tags}")
    
    images = generator_seq_with_tags(
        image_prompt, 
        descriptor, 
        scene, 
        cumulative_positive_tags, 
        cumulative_negative_tags,
        folder, 
        concept_name, 
        session_folder, 
        name
    )
    
    result = {
        'concept_id': concept_index,
        'concept_name': concept_name,
        'scene_data': scene,
        'cumulative_positive_tags': cumulative_positive_tags.copy(),
        'cumulative_negative_tags': cumulative_negative_tags.copy(),
        'scenes': scenes  # Include all scenes for subsequent concepts
    }
    
    if images:
        image_path = images[0]
        write_status(session_folder, f"Generated image: {os.path.basename(image_path)}")
        
        # Extract visual tags from the generated image
        write_status(session_folder, f"Extracting visual tags from concept {concept_index + 1}...")
        visual_tags = extract_visual_elements_from_image(image_path, tag_prompt)
        
        if visual_tags:
            write_status(session_folder, f"Extracted {len(visual_tags)} visual tags for concept {concept_index + 1}")
        else:
            write_status(session_folder, f"Warning: No visual tags extracted for concept {concept_index + 1}")
        
        result.update({
            'image_path': image_path,
            'visual_tags': visual_tags or []
        })
        
        # Save visual tags
        tags_file = os.path.join(folder, "visual_tags.json")
        all_visual_tags = {}
        if os.path.exists(tags_file):
            with open(tags_file, "r") as f:
                all_visual_tags = json.load(f)
        
        # Store tags using filename as key (same as other modes)
        image_filename = f"{concept_name}.png"
        all_visual_tags[image_filename] = visual_tags or []
        with open(tags_file, "w") as f:
            json.dump(all_visual_tags, f, indent=2)
        
    else:
        write_status(session_folder, f"Warning: No images generated for concept {concept_index + 1}")
        result.update({
            'image_path': None,
            'visual_tags': []
        })
    
    write_status(session_folder, f"Completed concept {concept_index + 1} for {name.upper()} stage")
    
    return result


def update_cumulative_tags(positive_feedback: list, negative_feedback: list, 
                          cumulative_positive: list, cumulative_negative: list) -> tuple[list, list]:
    """
    Update cumulative tags based on user feedback.
    
    Args:
        positive_feedback: List of tags user liked
        negative_feedback: List of tags user disliked
        cumulative_positive: Current cumulative positive tags
        cumulative_negative: Current cumulative negative tags
    
    Returns:
        Updated (cumulative_positive, cumulative_negative) lists
    """
    # Add new positive tags (avoid duplicates)
    updated_positive = cumulative_positive.copy()
    for tag in positive_feedback:
        if tag not in updated_positive:
            updated_positive.append(tag)
    
    # Add new negative tags (avoid duplicates)
    updated_negative = cumulative_negative.copy()
    for tag in negative_feedback:
        if tag not in updated_negative:
            updated_negative.append(tag)
    
    return updated_positive, updated_negative


def save_user_preferences(session_folder: str, stage_name: str, selected_concept: dict, 
                         cumulative_tags: dict, user_pref: dict = None) -> dict:
    """
    Save user preferences when they select a concept in cumulative tags mode.
    
    Args:
        session_folder: Session folder path
        stage_name: Current stage name (impression, spatial, objects, ambient)
        selected_concept: The selected concept data
        cumulative_tags: Current cumulative tags
        user_pref: Existing user preferences (optional)
    
    Returns:
        Updated user preferences dictionary
    """
    if user_pref is None:
        user_pref = {}
    
    # Add the selected concept to user preferences
    user_pref[stage_name] = selected_concept
    
    # Save user preferences to preferences.json
    preferences_file = os.path.join(session_folder, "preferences.json")
    
    # Load existing preferences if file exists
    existing_prefs = {}
    if os.path.exists(preferences_file):
        try:
            with open(preferences_file, 'r') as f:
                existing_prefs = json.load(f)
        except Exception as e:
            write_status(session_folder, f"Warning: Could not load existing preferences: {str(e)}")
    
    # Update with new preferences
    existing_prefs.update(user_pref)
    
    # Add metadata
    existing_prefs['metadata'] = {
        'last_updated': datetime.now().isoformat(),
        'session_type': 'cumulative_tags',
        'stages_completed': list(user_pref.keys()),
        'cumulative_tags_count': len(cumulative_tags.get('positive', [])) + len(cumulative_tags.get('negative', []))
    }
    
    # Save updated preferences
    try:
        with open(preferences_file, 'w') as f:
            json.dump(existing_prefs, f, indent=2)
        write_status(session_folder, f"Saved user preferences for {stage_name} to preferences.json")
    except Exception as e:
        write_status(session_folder, f"Error saving preferences: {str(e)}")
    
    return existing_prefs


def load_user_preferences(session_folder: str) -> dict:
    """
    Load user preferences from preferences.json.
    
    Args:
        session_folder: Session folder path
    
    Returns:
        User preferences dictionary
    """
    preferences_file = os.path.join(session_folder, "preferences.json")
    
    if not os.path.exists(preferences_file):
        return {}
    
    try:
        with open(preferences_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        write_status(session_folder, f"Error loading preferences: {str(e)}")
        return {}


# Note: Pipeline function removed - cumulative tags mode works concept by concept
# through the API endpoints, not as a single pipeline run


def run_cumulative_tags_final_stage(descriptor: str, user_pref: dict, cumulative_tags: dict, 
                                   session_folder: str, mode: str = "mode1") -> list:
    """
    Run the final stage for cumulative tags mode.
    
    Args:
        descriptor: User description
        user_pref: User preferences from selected concepts
        cumulative_tags: Cumulative tags from all stages
        session_folder: Session folder path
        mode: Final generation mode (mode1, mode2, mode3, mode4)
    
    Returns:
        List of (scene, files) tuples
    """
    write_status(session_folder, f"Starting FINAL stage for Cumulative Tags - {mode}")
    
    # Create final folder
    mode_folder_names = {
        "mode1": "final",
        "mode2": "[with Tags]final", 
        "mode3": "[with Imgs]final",
        "mode4": "[Enhanced Prefs]final",
        "mode5": "[Progressive]final"
    }
    
    if mode not in mode_folder_names:
        raise RuntimeError(f"Invalid mode: {mode}")
    
    mode_folder_name = mode_folder_names[mode]
    mode_folder = os.path.join(session_folder, mode_folder_name)
    
    # If folder exists, remove and recreate to ensure clean state
    if os.path.exists(mode_folder):
        import shutil
        shutil.rmtree(mode_folder)
        write_status(session_folder, f"Removed existing {mode_folder_name} folder")
    
    os.makedirs(mode_folder, exist_ok=True)
    
    try:
        if mode == "mode1":
            # Mode 1: Basic final generation
            narrative_prompt, image_prompt = FINAL_PROMPT_CUMULATIVE, FINAL_GENERATOR_PROMPT_CUMULATIVE
            results = run_final_mode1_cumulative(
                narrative_prompt, image_prompt, descriptor, user_pref,
                mode_folder, session_folder
            )
            
        elif mode == "mode2":
            # Mode 2: With cumulative tags
            narrative_prompt, image_prompt = FINAL_PROMPT_CUMULATIVE_TAGS, FINAL_GENERATOR_PROMPT_CUMULATIVE_TAGS
            results = run_final_mode2_cumulative(
                narrative_prompt, image_prompt, descriptor, user_pref, cumulative_tags,
                mode_folder, session_folder
            )
            
        elif mode == "mode3":
            # Mode 3: With cumulative tags and reference images
            narrative_prompt, image_prompt = FINAL_PROMPT_CUMULATIVE_TAGS, FINAL_GENERATOR_PROMPT_CUMULATIVE_IMGS
            results = run_final_mode3_cumulative(
                narrative_prompt, image_prompt, descriptor, user_pref, cumulative_tags,
                mode_folder, session_folder
            )
            
        elif mode == "mode4":
            # Mode 4: Enhanced with cumulative tags
            narrative_prompt, image_prompt = FINAL_PROMPT_CUMULATIVE_TAGS, FINAL_GENERATOR_PROMPT_CUMULATIVE_TAGS
            results = run_final_mode4_cumulative(
                narrative_prompt, image_prompt, descriptor, user_pref, cumulative_tags,
                mode_folder, session_folder
            )
        elif mode == "mode5":
            # Mode 5: Progressive one-concept final for cumulative tags sessions
            results = run_final_mode5_cumulative(
                FINAL_PROMPT_PROGRESSIVE, FINAL_GENERATOR_PROMPT_PROGRESSIVE,
                descriptor, user_pref, cumulative_tags,
                mode_folder, session_folder
            )
        
        write_status(session_folder, f"Final stage {mode} completed with {len(results)} scene results")
        return results
        
    except Exception as e:
        error_msg = f"Failed to generate final stage in {mode}: {str(e)}"
        write_status(session_folder, error_msg)
        raise RuntimeError(error_msg)


def run_final_mode1_cumulative(narrative_prompt: str, image_prompt: str, descriptor: str, 
                              user_pref: dict, mode_folder: str, session_folder: str) -> list:
    """Run final mode 1 for cumulative tags: Basic final generation."""
    from util import generator_final_mode1
    
    write_status(session_folder, f"Starting FINAL Mode 1 - Basic (Cumulative Tags)")
    
    # Generate multiple scene descriptions using designer
    write_status(session_folder, f"Generating concepts...")
    scenes = designer_seq(narrative_prompt, descriptor, user_pref, session_folder, "final")
    
    if not scenes:
        error_msg = f"No scenes generated for final mode 1"
        write_status(session_folder, error_msg)
        raise RuntimeError(error_msg)
    
    write_status(session_folder, f"Generated {len(scenes)} concepts")
    
    # Save scenes JSON
    scenes_file = os.path.join(mode_folder, f"final.json")
    with open(scenes_file, "w") as f:
        json.dump(scenes, f, indent=2)
    write_status(session_folder, f"Saved scenes to: final.json")
    
    # Generate images for each scene
    results = []
    for i, scene in enumerate(scenes):
        concept_name = scene.get('concept_name', f'Scene {i+1}')
        write_status(session_folder, f"Generating images for: {concept_name}")
        
        prefix = f"final_{i}"
        
        try:
            # Use mode 1 generator
            files = generator_final_mode1(
                narrative_prompt, image_prompt, descriptor, scene, user_pref,
                mode_folder, prefix, session_folder, "final"
            )
            
            if files:
                results.append((scene, files))
                write_status(session_folder, f"Generated {len(files)} files for {concept_name}")
            else:
                write_status(session_folder, f"Warning: No files generated for {concept_name}")
                
        except Exception as e:
            write_status(session_folder, f"Error generating {concept_name}: {str(e)}")
            continue
    
    return results


def run_final_mode2_cumulative(narrative_prompt: str, image_prompt: str, descriptor: str, 
                              user_pref: dict, cumulative_tags: dict, mode_folder: str, session_folder: str) -> list:
    """Run final mode 2 for cumulative tags: With cumulative tags."""
    from util import generator_final_mode2
    
    write_status(session_folder, f"Starting FINAL Mode 2 - With Cumulative Tags")
    
    # Generate multiple scene descriptions using designer
    write_status(session_folder, f"Generating concepts with cumulative tags...")
    scenes = designer_seq(narrative_prompt, descriptor, user_pref, session_folder, "final")
    
    if not scenes:
        error_msg = f"No scenes generated for final mode 2"
        write_status(session_folder, error_msg)
        raise RuntimeError(error_msg)
    
    write_status(session_folder, f"Generated {len(scenes)} concepts")
    
    # Save scenes JSON
    scenes_file = os.path.join(mode_folder, f"final.json")
    with open(scenes_file, "w") as f:
        json.dump(scenes, f, indent=2)
    write_status(session_folder, f"Saved scenes to: final.json")
    
    # Generate images for each scene
    results = []
    for i, scene in enumerate(scenes):
        concept_name = scene.get('concept_name', f'Scene {i+1}')
        write_status(session_folder, f"Generating images for: {concept_name}")
        
        prefix = f"final_{i}"
        
        try:
            # Use mode 2 generator with cumulative tags
            files = generator_final_mode2(
                narrative_prompt, image_prompt, descriptor, scene, user_pref, cumulative_tags,
                mode_folder, prefix, session_folder, "final"
            )
            
            if files:
                results.append((scene, files))
                write_status(session_folder, f"Generated {len(files)} files for {concept_name}")
            else:
                write_status(session_folder, f"Warning: No files generated for {concept_name}")
                
        except Exception as e:
            write_status(session_folder, f"Error generating {concept_name}: {str(e)}")
            continue
    
    return results


def run_final_mode3_cumulative(narrative_prompt: str, image_prompt: str, descriptor: str, 
                              user_pref: dict, cumulative_tags: dict, mode_folder: str, session_folder: str) -> list:
    """Run final mode 3 for cumulative tags: With cumulative tags and reference images."""
    from util import generator_final_mode3
    
    write_status(session_folder, f"Starting FINAL Mode 3 - With Cumulative Tags and Reference Images")
    
    # Generate multiple scene descriptions using designer
    write_status(session_folder, f"Generating concepts with cumulative tags and reference images...")
    scenes = designer_seq(narrative_prompt, descriptor, user_pref, session_folder, "final")
    
    if not scenes:
        error_msg = f"No scenes generated for final mode 3"
        write_status(session_folder, error_msg)
        raise RuntimeError(error_msg)
    
    write_status(session_folder, f"Generated {len(scenes)} concepts")
    
    # Save scenes JSON
    scenes_file = os.path.join(mode_folder, f"final.json")
    with open(scenes_file, "w") as f:
        json.dump(scenes, f, indent=2)
    write_status(session_folder, f"Saved scenes to: final.json")
    
    # Get reference images from user preferences
    reference_images = []
    for stage_name in ['impression', 'spatial', 'objects', 'ambient']:
        if stage_name in user_pref:
            selected_concept = user_pref[stage_name]
            if isinstance(selected_concept, dict) and 'concept_name' in selected_concept:
                # Extract image path from concept data
                concept_name = selected_concept['concept_name']
                img_path = os.path.join(session_folder, stage_name, f"{concept_name}.png")
                if os.path.exists(img_path):
                    reference_images.append(img_path)
                    write_status(session_folder, f"Added reference image: {stage_name}/{concept_name}.png")
                else:
                    write_status(session_folder, f"Warning: Reference image not found: {img_path}")
    
    # Generate images for each scene
    results = []
    for i, scene in enumerate(scenes):
        concept_name = scene.get('concept_name', f'Scene {i+1}')
        write_status(session_folder, f"Generating images for: {concept_name}")
        
        prefix = f"final_{i}"
        
        try:
            # Use mode 3 generator with cumulative tags and reference images
            files = generator_final_mode3(
                narrative_prompt, image_prompt, descriptor, scene, user_pref, cumulative_tags, reference_images,
                mode_folder, prefix, session_folder, "final"
            )
            
            if files:
                results.append((scene, files))
                write_status(session_folder, f"Generated {len(files)} files for {concept_name}")
            else:
                write_status(session_folder, f"Warning: No files generated for {concept_name}")
                
        except Exception as e:
            write_status(session_folder, f"Error generating {concept_name}: {str(e)}")
            continue
    
    return results


def run_final_mode4_cumulative(narrative_prompt: str, image_prompt: str, descriptor: str, 
                              user_pref: dict, cumulative_tags: dict, mode_folder: str, session_folder: str) -> list:
    """Run final mode 4 for cumulative tags: Enhanced with cumulative tags."""
    from util import generator_final_mode4
    
    write_status(session_folder, f"Starting FINAL Mode 4 - Enhanced with Cumulative Tags")
    
    # Generate multiple scene descriptions using designer
    write_status(session_folder, f"Generating enhanced concepts with cumulative tags...")
    scenes = designer_seq(narrative_prompt, descriptor, user_pref, session_folder, "final")
    
    if not scenes:
        error_msg = f"No scenes generated for final mode 4"
        write_status(session_folder, error_msg)
        raise RuntimeError(error_msg)
    
    write_status(session_folder, f"Generated {len(scenes)} concepts")
    
    # Save scenes JSON
    scenes_file = os.path.join(mode_folder, f"final.json")
    with open(scenes_file, "w") as f:
        json.dump(scenes, f, indent=2)
    write_status(session_folder, f"Saved scenes to: final.json")
    
    # Generate images for each scene
    results = []
    for i, scene in enumerate(scenes):
        concept_name = scene.get('concept_name', f'Scene {i+1}')
        write_status(session_folder, f"Generating images for: {concept_name}")
        
        prefix = f"final_{i}"
        
        try:
            # Use mode 4 generator with enhanced cumulative tags
            files = generator_final_mode4(
                narrative_prompt, image_prompt, descriptor, scene, user_pref, cumulative_tags,
                mode_folder, prefix, session_folder, "final"
            )
            
            if files:
                results.append((scene, files))
                write_status(session_folder, f"Generated {len(files)} files for {concept_name}")
            else:
                write_status(session_folder, f"Warning: No files generated for {concept_name}")
                
        except Exception as e:
            write_status(session_folder, f"Error generating {concept_name}: {str(e)}")
            continue
    
    return results





def run_final_mode5_cumulative(narrative_prompt: str, image_prompt: str, descriptor: str, 
                               user_pref: dict, cumulative_tags: dict, mode_folder: str, session_folder: str) -> list:
    """Run final mode 5 (Progressive one-concept) for cumulative tags sessions."""
    from util import designer_final_one_concept, generator_final_one_concept
    from tag_extraction import extract_visual_elements_from_image, prompt as general_prompt

    write_status(session_folder, f"Starting FINAL Mode 5 - Progressive (Cumulative Tags)")

    images_results = []
    all_visual_tags = {}
    concepts = []

    # Progressive accumulators
    accumulated_positive = []
    accumulated_negative = []

    for i in range(4):
        write_status(session_folder, f"Mode5 Iteration {i+1}: designing with {len(accumulated_positive)} positive and {len(accumulated_negative)} negative tags")

        concept = designer_final_one_concept(
            narrative_prompt,
            accumulated_positive,
            accumulated_negative,
            descriptor,
            user_pref,
            session_folder,
            "final"
        )
        concepts.append(concept)

        prefix = f"final_{i}"
        files = generator_final_one_concept(
            image_prompt,
            descriptor,
            concept,
            user_pref,
            accumulated_positive,
            accumulated_negative,
            mode_folder,
            prefix,
            session_folder,
            "final"
        )

        # Append result tuple and extract tags
        images_results.append((concept, files))
        for file_path in files:
            if file_path.endswith('.png'):
                filename = os.path.basename(file_path)
                write_status(session_folder, f"Extracting visual tags for {filename}")
                try:
                    tags = extract_visual_elements_from_image(file_path, general_prompt)
                except Exception:
                    tags = []
                all_visual_tags[filename] = tags

                # NOTE: UI is responsible for providing feedback to accumulate; here we can
                # passively add positives from extracted tags if desired. Keeping accumulators unchanged
                # to align with external feedback workflow.

    # Save scenes JSON
    scenes_file = os.path.join(mode_folder, "final.json")
    with open(scenes_file, "w") as f:
        json.dump(concepts, f, indent=2)

    # Save visual tags
    if all_visual_tags:
        tags_path = os.path.join(mode_folder, "visual_tags.json")
        with open(tags_path, "w") as tagf:
            json.dump(all_visual_tags, tagf, indent=2)
        total_tags = sum(len(tags) for tags in all_visual_tags.values())
        write_status(session_folder, f"Saved {total_tags} total tags to: visual_tags.json")

    write_status(session_folder, f"FINAL Mode 5 (Progressive) completed successfully!")
    return images_results

