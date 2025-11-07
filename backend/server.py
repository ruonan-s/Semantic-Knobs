import os
import uuid
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import run_stage, run_stage_seq, run_stage_seq_parallel, run_stage_seq_parallel_optimized
from cumulative_tags import start_cumulative_tags_stage, generate_concept_with_cumulative_tags, update_cumulative_tags, run_cumulative_tags_final_stage
from util import designer_seq
from tag_extraction import extract_visual_elements_from_image
from util import sanitize_folder_name, generator, designer, write_status, generator_final_mode1, generator_final_mode2, generator_final_mode3, generator_final_mode4, generator_final_mode1_optimized, generator_final_mode2_optimized, generator_final_mode3_optimized, generator_final_mode4_optimized, initialize_prompt_tracking
from mode4 import generate_user_preference
from tag_extraction import prompt as general_prompt
from prompt import (
    IMPRESSION_PROMPT, IMPRESSION_GENERATOR_PROMPT,
    SPATIAL_PROMPT, SPATIAL_GENERATOR_PROMPT,
    OBJECTS_PROMPT, OBJECTS_GENERATOR_PROMPT,
    AMBIENT_PROMPT, AMBIENT_GENERATOR_PROMPT,
    FINAL_PROMPT, FINAL_GENERATOR_PROMPT,
    FINAL_PROMPT_TAGS, FINAL_GENERATOR_PROMPT_TAGS,
    FINAL_GENERATOR_PROMPT_IMGS
)
from prompt_refinement import (
    IMPRESSION_REFINEMENT_PROMPT,
    SPATIAL_REFINEMENT_PROMPT,
    OBJECTS_REFINEMENT_PROMPT,
    AMBIENT_REFINEMENT_PROMPT
)
# Progressive final prompts (one concept) from prompt_tag_acc
 
import json
from fastapi.responses import JSONResponse
from datetime import datetime

# Import concept refinement module
from concept_refinement import get_or_create_session as get_refinement_session

# Define your stages and prompts
STAGES = [
    "impression", "impression_refinement",
    "spatial", "spatial_refinement",
    "objects", "objects_refinement",
    "ambient", "ambient_refinement",
    "final"
]
# Sequential prompts
PROMPTS = {
    'impression': (IMPRESSION_PROMPT, IMPRESSION_GENERATOR_PROMPT),
    'impression_refinement': (IMPRESSION_REFINEMENT_PROMPT, IMPRESSION_GENERATOR_PROMPT),
    'spatial': (SPATIAL_PROMPT, SPATIAL_GENERATOR_PROMPT),
    'spatial_refinement': (SPATIAL_REFINEMENT_PROMPT, SPATIAL_GENERATOR_PROMPT),
    'objects': (OBJECTS_PROMPT, OBJECTS_GENERATOR_PROMPT),
    'objects_refinement': (OBJECTS_REFINEMENT_PROMPT, OBJECTS_GENERATOR_PROMPT),
    'ambient': (AMBIENT_PROMPT, AMBIENT_GENERATOR_PROMPT),
    'ambient_refinement': (AMBIENT_REFINEMENT_PROMPT, AMBIENT_GENERATOR_PROMPT),
    'final': (FINAL_PROMPT, FINAL_GENERATOR_PROMPT)
}

# Refinement stages (used to identify if a stage is refinement)
REFINEMENT_STAGES = {
    "impression_refinement", "spatial_refinement", 
    "objects_refinement", "ambient_refinement"
}


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Where we store session folders & serve images from them
SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)
app.mount("/sessions", StaticFiles(directory=SESSIONS_DIR), name="sessions")

# In-memory session store
sessions = {}

# --- Models ---
class GenerateRequest(BaseModel):
    descriptor: str

class ImageItem(BaseModel):
    id: str
    url: str

class GenerateResponse(BaseModel):
    session_id: str
    stage: str
    images: list[ImageItem]

class CumulativeTagsRequest(BaseModel):
    descriptor: str

class CumulativeTagsConceptRequest(BaseModel):
    session_id: str
    concept_index: int

class TagFeedbackRequest(BaseModel):
    session_id: str
    stage: str
    concept_index: int
    positive_tags: list[str]
    negative_tags: list[str]

class CumulativeTagsResponse(BaseModel):
    session_id: str
    stage: str
    concept: dict
    total_concepts: int

@app.post("/api/generate")
def generate(req: GenerateRequest):
    # Create session folder with descriptive name and mode prefix
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_descriptor = sanitize_folder_name(req.descriptor)
    folder_name = f"[seq]_{safe_descriptor}_{timestamp}"
    session_id = folder_name  # Use folder name as session ID for URL consistency
    session_folder = os.path.join("sessions", folder_name)
    os.makedirs(session_folder, exist_ok=True)
    
    print(f"📁 Created sequential session: {session_id}")
    print(f"🎯 Descriptor: {req.descriptor}")
    
    # Store session info
    sessions[session_id] = {
        'folder': session_folder,
        'descriptor': req.descriptor,
        'user_pref': {}
    }
    
    # Initialize prompt tracking
    initialize_prompt_tracking(session_folder, req.descriptor, "sequential")
    
    try:
        # Start with impression stage
        stage = 'impression'
        narrative_prompt, image_prompt = PROMPTS[stage]
        
        results, user_pref = run_stage_seq_parallel_optimized(
            stage,
            narrative_prompt, 
            image_prompt,
            req.descriptor,
            {},
            session_folder
        )
        
        # Update session with new user_pref
        sessions[session_id]['user_pref'] = user_pref
        
        # Prepare response
        images = []
        for i, (scene, files) in enumerate(results):
            for file in files:
                if file.endswith('.png'):
                    # Use the actual filename (without extension) as ID to match the initial generation
                    image_id = os.path.splitext(os.path.basename(file))[0]
                    image_url = f"/sessions/{session_id}/{stage}/{os.path.basename(file)}"
                    images.append({"id": image_id, "url": image_url})
        
        return {
            "session_id": session_id,
            "stage": stage,
            "images": images
        }
        
    except Exception as e:
        print(f"Error in generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Sequential with Parallel Images generation endpoint ---
@app.post("/api/generate-fast")
def generate_fast(req: GenerateRequest):
    """Generate using sequential mode but with parallel image generation for much faster processing."""
    # Create session folder with descriptive name and mode prefix
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_descriptor = sanitize_folder_name(req.descriptor)
    folder_name = f"[fast]_{safe_descriptor}_{timestamp}"
    session_id = folder_name  # Use folder name as session ID for URL consistency
    session_folder = os.path.join("sessions", folder_name)
    os.makedirs(session_folder, exist_ok=True)
    
    print(f"📁 Created fast session: {session_id}")
    print(f"🎯 Descriptor: {req.descriptor}")
    
    # Store session info
    sessions[session_id] = {
        'folder': session_folder,
        'descriptor': req.descriptor,
        'user_pref': {}
    }
    
    # Initialize prompt tracking
    initialize_prompt_tracking(session_folder, req.descriptor, "sequential_parallel_images")
    
    try:
        # Start with impression stage
        stage = 'impression'
        narrative_prompt, image_prompt = PROMPTS[stage]
        
        results, user_pref = run_stage_seq_parallel_optimized(
            stage,
            narrative_prompt, 
            image_prompt,
            req.descriptor,
            {},
            session_folder
        )
        
        # Update session with new user_pref
        sessions[session_id]['user_pref'] = user_pref
        
        # Prepare response
        images = []
        for i, (scene, files) in enumerate(results):
            for file in files:
                if file.endswith('.png'):
                    # Use the actual filename (without extension) as ID to match the initial generation
                    image_id = os.path.splitext(os.path.basename(file))[0]
                    image_url = f"/sessions/{session_id}/{stage}/{os.path.basename(file)}"
                    images.append({"id": image_id, "url": image_url})
        
        return {
            "session_id": session_id,
            "stage": stage,
            "images": images
        }
        
    except Exception as e:
        print(f"Error in generate-fast endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Cumulative Tags Mode Endpoints ---
@app.post("/api/generate-cumulative-tags")
def generate_cumulative_tags(req: CumulativeTagsRequest):
    """Start a new cumulative tags session and generate the first concept."""
    try:
        # Create session folder with descriptive name and mode prefix
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_descriptor = sanitize_folder_name(req.descriptor)
        folder_name = f"[seqtag]_{safe_descriptor}_{timestamp}"
        session_id = folder_name
        session_folder = os.path.join("sessions", folder_name)
        os.makedirs(session_folder, exist_ok=True)
        
        print(f"📁 Created cumulative tags session: {session_id}")
        print(f"🎯 Descriptor: {req.descriptor}")
        
        # Store session info
        sessions[session_id] = {
            'folder': session_folder,
            'descriptor': req.descriptor,
            'mode': 'cumulative_tags',
            'current_stage': 'impression',
            'current_concept_index': 0,
            'cumulative_tags': {
                'impression': {'positive': [], 'negative': []},
                'spatial': {'positive': [], 'negative': []},
                'objects': {'positive': [], 'negative': []},
                'ambient': {'positive': [], 'negative': []}
            },
            'user_pref': {},
            'scenes': {}
        }
        
        # Initialize prompt tracking
        initialize_prompt_tracking(session_folder, req.descriptor, "cumulative_tags")
        
        # Generate all 4 concepts first, then first image for impression stage
        from prompt_tag_acc import IMPRESSION_PROMPT, IMPRESSION_GENERATOR_PROMPT
        stage = 'impression'
        
        result = start_cumulative_tags_stage(
            stage,
            IMPRESSION_PROMPT,
            IMPRESSION_GENERATOR_PROMPT,
            req.descriptor,
            {},
            session_folder
        )
        
        # Store scenes and current concept in session
        sessions[session_id]['scenes'][stage] = result['scenes']
        
        # Prepare response - format as single image like sequential mode
        concept_data = {
            'concept_id': result['concept_id'],
            'concept_name': result['concept_name'],
            'image_url': f"/sessions/{session_id}/{stage}/{os.path.basename(result['image_path'])}" if result['image_path'] else None,
            'visual_tags': result['visual_tags'],
            'scene_data': result['scene_data']
        }
        
        # Return in same format as sequential mode with single image
        return {
            "session_id": session_id,
            "stage": stage,
            "images": [{"id": result['concept_name'], "url": concept_data['image_url']}] if result['image_path'] else []
        }
        
    except Exception as e:
        print(f"Error in cumulative tags generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cumulative-tags-feedback")
def cumulative_tags_feedback(req: TagFeedbackRequest):
    """Process tag feedback and generate next concept with accumulated feedback."""
    try:
        session_id = req.session_id
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        current_stage = req.stage
        
        # ===== DEBUG LOGS: Tag Recording Flow =====
        print(f"🏷️  [DEBUG] CUMULATIVE TAG FEEDBACK RECEIVED:")
        print(f"    Session ID: {session_id}")
        print(f"    Stage: {current_stage}")
        print(f"    Concept Index: {req.concept_index}")
        print(f"    Positive Tags Received: {req.positive_tags}")
        print(f"    Negative Tags Received: {req.negative_tags}")
        print(f"    Current Session Cumulative Tags: {session['cumulative_tags'][current_stage]}")
        
        # Update cumulative tags for this stage
        stage_tags = session['cumulative_tags'][current_stage]
        updated_positive, updated_negative = update_cumulative_tags(
            req.positive_tags,
            req.negative_tags,
            stage_tags['positive'],
            stage_tags['negative']
        )
        
        session['cumulative_tags'][current_stage]['positive'] = updated_positive
        session['cumulative_tags'][current_stage]['negative'] = updated_negative
        
        print(f"🏷️  [DEBUG] AFTER UPDATE:")
        print(f"    Updated Positive Tags: {updated_positive}")
        print(f"    Updated Negative Tags: {updated_negative}")
        print(f"    Session State Updated: {session['cumulative_tags'][current_stage]}")
        
        # Write session folder for debugging
        session_folder = session['folder']
        write_status(session_folder, f"[DEBUG] Tag feedback received: +{len(req.positive_tags)} positive, +{len(req.negative_tags)} negative")
        write_status(session_folder, f"[DEBUG] Cumulative tags now: +{len(updated_positive)} positive, +{len(updated_negative)} negative")
        
        print(f"Updated cumulative tags for {current_stage}: +{len(updated_positive)} positive, +{len(updated_negative)} negative")
        
        # Check if we need to generate next concept
        next_concept_index = req.concept_index + 1
        
        if next_concept_index < 4:  # We have 4 concepts total (0, 1, 2, 3)
            # Generate next concept with accumulated tags
            from prompt_tag_acc import (
                IMPRESSION_PROMPT, IMPRESSION_GENERATOR_PROMPT,
                SPATIAL_PROMPT, SPATIAL_GENERATOR_PROMPT,
                OBJECTS_PROMPT, OBJECTS_GENERATOR_PROMPT,
                AMBIENT_PROMPT, AMBIENT_GENERATOR_PROMPT
            )
            
            stage_prompts = {
                'impression': (IMPRESSION_PROMPT, IMPRESSION_GENERATOR_PROMPT),
                'spatial': (SPATIAL_PROMPT, SPATIAL_GENERATOR_PROMPT),
                'objects': (OBJECTS_PROMPT, OBJECTS_GENERATOR_PROMPT),
                'ambient': (AMBIENT_PROMPT, AMBIENT_GENERATOR_PROMPT)
            }
            
            narrative_prompt, image_prompt = stage_prompts[current_stage]
            
            result = generate_concept_with_cumulative_tags(
                current_stage,
                image_prompt,
                session['descriptor'],
                session['scenes'][current_stage],
                next_concept_index,
                updated_positive,
                updated_negative,
                os.path.join(session['folder'], current_stage),
                session['folder']
            )
            
            # Update session state
            session['current_concept_index'] = next_concept_index
            
            # Return next concept in same format as initial generation
            return {
                "session_id": session_id,
                "stage": current_stage,
                "images": [{"id": result['concept_name'], "url": f"/sessions/{session_id}/{current_stage}/{os.path.basename(result['image_path'])}"}] if result['image_path'] else [],
                "concept_index": next_concept_index,
                "total_concepts": 4,
                "has_next": next_concept_index < 3
            }
        else:
            # All concepts for this stage completed
            return {
                "session_id": session_id,
                "stage": current_stage,
                "status": "stage_complete",
                "message": "All concepts for this stage completed"
            }
        
    except Exception as e:
        print(f"Error processing cumulative tags feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cumulative-tags-next-stage")
def cumulative_tags_next_stage(req: dict):
    """Start the next stage in cumulative tags mode."""
    try:
        session_id = req.get('session_id')
        selected_concept = req.get('selected_concept')
        
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        current_stage = session['current_stage']
        
        # Require a selection before proceeding to the next stage
        if not selected_concept:
            raise HTTPException(400, "Please select one image before proceeding to the next stage.")

        # Update user preferences with selected concept
        if selected_concept:
            # selected_concept is the image ID (e.g., "impression_0_0")
            # We need to find the full concept JSON from the stored scenes
            if current_stage in session.get('scenes', {}):
                scenes = session['scenes'][current_stage]
                # Extract concept index from image ID (e.g., "impression_0_0" -> index 0)
                try:
                    concept_index = int(selected_concept.split('_')[1])
                    if 0 <= concept_index < len(scenes):
                        # Store the full concept JSON instead of just the image ID
                        session['user_pref'][current_stage] = scenes[concept_index]
                        print(f"✅ Stored full concept for {current_stage}: {scenes[concept_index].get('concept_name', 'Unknown')}")
                        
                        # Write to status file for debugging
                        session_folder = session['folder']
                        write_status(session_folder, f"[DEBUG] Stored user preference for {current_stage}: {scenes[concept_index].get('concept_name', 'Unknown')}")
                        write_status(session_folder, f"[DEBUG] Full concept JSON: {json.dumps(scenes[concept_index], indent=2)}")
                        
                        # Save user preferences to file
                        user_pref_file = os.path.join(session_folder, "user_preference.json")
                        with open(user_pref_file, 'w') as f:
                            json.dump(session['user_pref'], f, indent=2)
                        write_status(session_folder, f"Saved user preferences for {current_stage}")

                        # Also update preferences.json to mirror parallel/sequential structure
                        preferences_path = os.path.join(session_folder, "preferences.json")
                        # Load or init
                        preferences_data = {}
                        if os.path.exists(preferences_path):
                            try:
                                with open(preferences_path, 'r') as pf:
                                    preferences_data = json.load(pf) or {}
                            except Exception:
                                preferences_data = {}
                        selections = preferences_data.get('selections', {})
                        tags_struct = preferences_data.get('tags', {})
                        # Update selection id for current stage
                        selections[current_stage] = selected_concept
                        # Flatten all cumulative tags into parallel list
                        parallel_tags = []
                        for stage_name, tag_sets in session['cumulative_tags'].items():
                            for tag in tag_sets.get('positive', []):
                                parallel_tags.append({
                                    'tag': tag,
                                    'preference': 'positive',
                                    'source_image': stage_name
                                })
                            for tag in tag_sets.get('negative', []):
                                parallel_tags.append({
                                    'tag': tag,
                                    'preference': 'negative',
                                    'source_image': stage_name
                                })
                        tags_struct['parallel'] = parallel_tags
                        preferences_data['selections'] = selections
                        preferences_data['tags'] = tags_struct
                        with open(preferences_path, 'w') as pf:
                            json.dump(preferences_data, pf, indent=2)
                        write_status(session_folder, f"Updated preferences.json with selection for {current_stage} and cumulative tags")
                    else:
                        print(f"⚠️ Concept index {concept_index} out of range for {current_stage}")
                        session['user_pref'][current_stage] = selected_concept  # Fallback to image ID
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Error parsing concept index from {selected_concept}: {e}")
                    session['user_pref'][current_stage] = selected_concept  # Fallback to image ID
            else:
                print(f"⚠️ No scenes found for {current_stage}, storing image ID")
                session['user_pref'][current_stage] = selected_concept  # Fallback to image ID
        
        # Move to next stage
        stage_order = ['impression', 'spatial', 'objects', 'ambient']
        current_stage_index = stage_order.index(current_stage)
        
        if current_stage_index >= len(stage_order) - 1:
            # All stages completed, transition to mode selection
            write_status(session['folder'], "All stages completed! Transitioning to mode selection for final stage.")
            return {"status": "complete", "message": "All stages completed", "next_stage": "mode-selection"}
        
        next_stage = stage_order[current_stage_index + 1]
        session['current_stage'] = next_stage
        session['current_concept_index'] = 0  # Reset concept index for new stage
        
        # Clear cumulative tags for the new stage
        session['cumulative_tags'][next_stage] = {'positive': [], 'negative': []}
        
        # Get appropriate prompts for the next stage
        from prompt_tag_acc import (
            SPATIAL_PROMPT, SPATIAL_GENERATOR_PROMPT,
            OBJECTS_PROMPT, OBJECTS_GENERATOR_PROMPT,
            AMBIENT_PROMPT, AMBIENT_GENERATOR_PROMPT
        )
        
        stage_prompts = {
            'spatial': (SPATIAL_PROMPT, SPATIAL_GENERATOR_PROMPT),
            'objects': (OBJECTS_PROMPT, OBJECTS_GENERATOR_PROMPT),
            'ambient': (AMBIENT_PROMPT, AMBIENT_GENERATOR_PROMPT)
        }
        
        if next_stage in stage_prompts:
            narrative_prompt, image_prompt = stage_prompts[next_stage]
            
            # Generate all 4 concepts for next stage, then first image
            result = start_cumulative_tags_stage(
                next_stage,
                narrative_prompt,
                image_prompt,
                session['descriptor'],
                session['user_pref'],
                session['folder']
            )
            
            # Store scenes for the new stage
            session['scenes'][next_stage] = result['scenes']
            
            # Return first concept of next stage
            return {
                "session_id": session_id,
                "stage": next_stage,
                "images": [{"id": result['concept_name'], "url": f"/sessions/{session_id}/{next_stage}/{os.path.basename(result['image_path'])}"}] if result['image_path'] else []
            }
        else:
            return {"status": "complete", "message": "All stages completed", "next_stage": "mode-selection"}
        
    except Exception as e:
        print(f"Error in cumulative tags next stage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cumulative-tags-select-concept")
def cumulative_tags_select_concept(req: dict):
    """Handle concept selection in cumulative tags mode and save user preferences."""
    try:
        session_id = req.get('session_id')
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        current_stage = req.get('stage')
        selected_concept_index = req.get('concept_index')
        selected_image_id = req.get('selected_image_id')
        
        if not all([session_id, current_stage, selected_concept_index, selected_image_id]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        # Load the selected concept data
        stage_folder = os.path.join(session['folder'], current_stage)
        json_path = os.path.join(stage_folder, f"{current_stage}.json")
        
        if not os.path.exists(json_path):
            raise HTTPException(status_code=404, detail=f"Stage data not found for {current_stage}")
        
        with open(json_path, 'r') as f:
            scenes = json.load(f)
        
        if selected_concept_index is None or selected_concept_index >= len(scenes):
            raise HTTPException(status_code=400, detail="Invalid concept index")
        
        selected_concept = scenes[selected_concept_index]
        
        # Load current user preferences
        from cumulative_tags import load_user_preferences, save_user_preferences
        user_pref = load_user_preferences(session['folder'])
        
        # Save the selection to user preferences
        updated_prefs = save_user_preferences(
            session['folder'],
            current_stage,
            selected_concept,
            session['cumulative_tags'],
            user_pref
        )
        
        # Save cumulative tags to file for final stage
        cumulative_tags_file = os.path.join(session['folder'], "cumulative_tags.json")
        with open(cumulative_tags_file, 'w') as f:
            json.dump(session['cumulative_tags'], f, indent=2)
        
        # Also maintain preferences.json in the same structure used by parallel/sequential modes
        preferences_path = os.path.join(session['folder'], "preferences.json")
        preferences_data = {}
        if os.path.exists(preferences_path):
            try:
                with open(preferences_path, 'r') as pf:
                    preferences_data = json.load(pf) or {}
            except Exception:
                preferences_data = {}

        # Ensure required keys
        selections = preferences_data.get('selections', {})
        tags_struct = preferences_data.get('tags', {})

        # Update selection for the current stage with the selected image id (e.g., impression_0_0)
        selections[current_stage] = selected_image_id

        # Flatten cumulative tags into a parallel-style list
        parallel_tags = []
        for stage_name, tag_sets in session['cumulative_tags'].items():
            for tag in tag_sets.get('positive', []):
                parallel_tags.append({
                    'tag': tag,
                    'preference': 'positive',
                    'source_image': stage_name
                })
            for tag in tag_sets.get('negative', []):
                parallel_tags.append({
                    'tag': tag,
                    'preference': 'negative',
                    'source_image': stage_name
                })

        tags_struct['parallel'] = parallel_tags

        # Persist updated preferences.json
        preferences_data['selections'] = selections
        preferences_data['tags'] = tags_struct
        with open(preferences_path, 'w') as pf:
            json.dump(preferences_data, pf, indent=2)
        write_status(session['folder'], f"Updated preferences.json with selection for {current_stage} and cumulative tags")

        write_status(session['folder'], f"User selected concept {selected_concept_index} for {current_stage}")
        write_status(session['folder'], f"Saved preferences and cumulative tags")
        
        return {
            "session_id": session_id,
            "stage": current_stage,
            "selected_concept": selected_concept,
            "message": f"Concept {selected_concept_index} selected for {current_stage}",
            "preferences_saved": True
        }
        
    except Exception as e:
        print(f"Error processing concept selection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Tags endpoint ---
class TagsRequest(BaseModel):
    session_id: str
    stage: str
    image_id: str

class TagsResponse(BaseModel):
    tags: list[str]

@app.post("/api/tags", response_model=TagsResponse)
def get_tags(req: TagsRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    try:
        # Handle different final mode folders
        if req.stage == 'final':
            # Check which final mode folder exists
            base_folder = session['folder']
            possible_folders = ['final', '[with Tags]final', '[with Imgs]final', '[Enhanced Prefs]final']
            stage_folder = None
            
            for folder_name in possible_folders:
                test_folder = os.path.join(base_folder, folder_name)
                if os.path.exists(test_folder):
                    stage_folder = test_folder
                    break
            
            if stage_folder is None:
                print(f"No final mode folder found in {base_folder}")
                return {"tags": []}
        else:
            # Regular stage folder
            stage_folder = os.path.join(session['folder'], req.stage)
        
        visual_tags_path = os.path.join(stage_folder, "visual_tags.json")
        
        print(f"Looking for visual tags at: {visual_tags_path}")
        print(f"Requested image_id: {req.image_id}")
        
        if not os.path.exists(visual_tags_path):
            print(f"visual_tags.json not found at {visual_tags_path}")
            return {"tags": []}
        
        with open(visual_tags_path, 'r') as f:
            visual_tags_data = json.load(f)
        
        print(f"Available files in visual_tags.json: {list(visual_tags_data.keys())}")
        
        # Now the image_id should exactly match the filename (without extension)
        # req.image_id is like "impression_0_0", so we need "impression_0_0.png"
        image_filename = f"{req.image_id}.png"
        
        print(f"Looking for filename: {image_filename}")
        
        if image_filename in visual_tags_data:
            tags = visual_tags_data[image_filename]
            print(f"Found {len(tags)} tags for {image_filename}: {tags}")
            return {"tags": tags}
        else:
            print(f"No tags found for {image_filename}")
            print(f"Available files: {list(visual_tags_data.keys())}")
            return {"tags": []}
            
    except Exception as e:
        print(f"Error loading tags: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {"tags": []}


# --- Concept Refinement Endpoints ---
class ConceptInitRequest(BaseModel):
    session_id: str
    stage: str
    image_ids: list[str]

class ConceptInitResponse(BaseModel):
    success: bool
    concepts: list[dict]
    categorized: dict
    image_effects: dict
    incidence_matrix: dict
    tag_preferences: dict

@app.post("/api/concepts/init", response_model=ConceptInitResponse)
def init_concepts(req: ConceptInitRequest):
    """Initialize concepts from image tags for a stage"""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    try:
        # Load tags for all images
        stage_folder = os.path.join(session['folder'], req.stage)
        visual_tags_path = os.path.join(stage_folder, "visual_tags.json")
        
        if not os.path.exists(visual_tags_path):
            print(f"visual_tags.json not found at {visual_tags_path}")
            return {
                "success": False,
                "concepts": [],
                "categorized": {"positive": [], "neutral": [], "negative": []},
                "image_effects": {},
                "incidence_matrix": {}
            }
        
        with open(visual_tags_path, 'r') as f:
            visual_tags_data = json.load(f)
        
        # Build image_tags dict
        image_tags = {}
        for image_id in req.image_ids:
            image_filename = f"{image_id}.png"
            if image_filename in visual_tags_data:
                image_tags[image_id] = visual_tags_data[image_filename]
            else:
                image_tags[image_id] = []
        
        # Get or create refinement session
        refinement_session = get_refinement_session(
            req.session_id, 
            req.stage, 
            req.image_ids
        )
        
        # Initialize from tags
        if not refinement_session.initialized:
            refinement_session.initialize_from_tags(image_tags)
        
        # Return current state
        state_dict = refinement_session.to_dict()
        
        return {
            "success": True,
            "concepts": state_dict['concepts'],
            "categorized": state_dict['categorized'],
            "image_effects": state_dict['image_effects'],
            "incidence_matrix": state_dict['incidence_matrix'],
            "tag_preferences": state_dict.get('tag_preferences', {})
        }
        
    except Exception as e:
        print(f"Error initializing concepts: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(500, str(e))


class ConceptInteractionRequest(BaseModel):
    session_id: str
    stage: str
    tag_id: str
    preference: str  # 'positive' or 'negative'

class ConceptInteractionResponse(BaseModel):
    success: bool
    concepts: list[dict]
    categorized: dict
    image_effects: dict
    tag_preferences: dict

@app.post("/api/concepts/interact", response_model=ConceptInteractionResponse)
def interact_with_concept(req: ConceptInteractionRequest):
    """Handle tag like/dislike interaction"""
    from concept_refinement import refinement_sessions
    
    key = f"{req.session_id}_{req.stage}"
    if key not in refinement_sessions:
        raise HTTPException(404, "Refinement session not found")
    
    try:
        refinement_session = refinement_sessions[key]
        
        # Handle interaction
        refinement_session.handle_tag_click(req.tag_id, req.preference)
        
        # Return updated state
        state_dict = refinement_session.to_dict()
        
        return {
            "success": True,
            "concepts": state_dict['concepts'],
            "categorized": state_dict['categorized'],
            "image_effects": state_dict['image_effects'],
            "tag_preferences": state_dict.get('tag_preferences', {})
        }
        
    except Exception as e:
        print(f"Error handling interaction: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(500, str(e))


class ConceptRankingRequest(BaseModel):
    session_id: str
    stage: str
    positive_concept_ids: list[str]
    negative_concept_ids: list[str]

class ConceptRankingResponse(BaseModel):
    success: bool
    concepts: list[dict]
    categorized: dict
    image_effects: dict
    tag_preferences: dict

@app.post("/api/concepts/rank", response_model=ConceptRankingResponse)
def update_concept_rankings(req: ConceptRankingRequest):
    """Update concept rankings from drag-and-drop"""
    from concept_refinement import refinement_sessions
    
    key = f"{req.session_id}_{req.stage}"
    if key not in refinement_sessions:
        raise HTTPException(404, "Refinement session not found")
    
    try:
        refinement_session = refinement_sessions[key]
        
        # Update rankings
        refinement_session.update_rankings(
            req.positive_concept_ids,
            req.negative_concept_ids
        )
        
        # Return updated state
        state_dict = refinement_session.to_dict()
        
        return {
            "success": True,
            "concepts": state_dict['concepts'],
            "categorized": state_dict['categorized'],
            "image_effects": state_dict['image_effects'],
            "tag_preferences": state_dict.get('tag_preferences', {})
        }
        
    except Exception as e:
        print(f"Error updating rankings: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(500, str(e))


class ImageSelectionRequest(BaseModel):
    session_id: str
    stage: str
    image_id: str
    boost_amount: float = 0.5

class ImageSelectionResponse(BaseModel):
    success: bool
    concepts: list[dict]
    categorized: dict
    image_effects: dict
    tag_preferences: dict

@app.post("/api/concepts/select-image", response_model=ImageSelectionResponse)
def handle_image_selection(req: ImageSelectionRequest):
    """Handle image selection to boost concept weights"""
    from concept_refinement import refinement_sessions
    
    key = f"{req.session_id}_{req.stage}"
    if key not in refinement_sessions:
        raise HTTPException(404, "Refinement session not found")
    
    try:
        refinement_session = refinement_sessions[key]
        
        # Handle image selection
        refinement_session.handle_image_selection(
            req.image_id,
            req.boost_amount
        )
        
        # Return updated state
        state_dict = refinement_session.to_dict()
        
        return {
            "success": True,
            "concepts": state_dict['concepts'],
            "categorized": state_dict['categorized'],
            "image_effects": state_dict['image_effects'],
            "tag_preferences": state_dict.get('tag_preferences', {})
        }
        
    except Exception as e:
        print(f"Error handling image selection: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(500, str(e))


# --- Feedback endpoint ---
class FeedbackRequest(BaseModel):
    session_id: str
    stage: str
    selected_image_id: str | None  # Allow None for parallel-to-final transitions
    preferences: dict

class FeedbackResponse(BaseModel):
    next_stage: str | None
    images: list[ImageItem]


# --- Progressive Final Mode (Mode 5) Endpoints ---
class FinalProgressiveStartRequest(BaseModel):
    session_id: str

@app.post("/api/generate-final-progressive")
def generate_final_progressive(req: FinalProgressiveStartRequest):
    """Start progressive final mode (iteration 1)."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Reuse generate_final logic for mode5 startup but only run first iteration
    # Build user_pref from selections in preferences saved in memory
    descriptor = session['descriptor']
    base_folder = session['folder']

    # Expect the UI to have called /api/generate-final/mode5 as well for folder creation
    mode_folder_name = "[Progressive]final"
    mode_folder = os.path.join(base_folder, mode_folder_name)
    if os.path.exists(mode_folder):
        import shutil
        shutil.rmtree(mode_folder)
    os.makedirs(mode_folder, exist_ok=True)

    # Build user_pref from existing selections on disk if available via preferences.json
    user_pref = {}
    # Best-effort: load each stage json and pick index from session selections if present
    selections = session.get('selections') or {}
    for stage_name in ['impression', 'spatial', 'objects', 'ambient']:
        if stage_name in selections:
            selected_image_id = selections[stage_name]
            stage_folder = os.path.join(base_folder, stage_name)
            json_path = os.path.join(stage_folder, f"{stage_name}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path) as f:
                        scenes = json.load(f)
                    idx = int(selected_image_id.split('_')[1])
                    if isinstance(scenes, list) and idx < len(scenes):
                        user_pref[stage_name] = scenes[idx]
                except Exception:
                    pass

    # Initialize state
    from util import designer_final_one_concept, generator_final_one_concept
    from tag_extraction import extract_visual_elements_from_image, prompt as general_prompt

    progressive_state = {
        'iteration': 1,
        'accumulated_positive': [],
        'accumulated_negative': [],
        'history': []
    }
    session['final_progressive'] = progressive_state

    # Iteration 1
    concept = designer_final_one_concept(
        "",
        [],
        [],
        descriptor,
        user_pref,
        base_folder,
        "final"
    )

    prefix = "final_0"
    files = generator_final_one_concept(
        "",
        descriptor,
        concept,
        user_pref,
        [],
        [],
        mode_folder,
        prefix,
        base_folder,
        "final"
    )

    images = []
    all_visual_tags = {}
    image_url = None
    if files:
        for file_path in files:
            if file_path.endswith('.png'):
                image_id = os.path.splitext(os.path.basename(file_path))[0]
                image_url = f"/sessions/{req.session_id}/{mode_folder_name}/{os.path.basename(file_path)}"
                images.append({"id": image_id, "url": image_url})
                try:
                    tags = extract_visual_elements_from_image(file_path, general_prompt)
                except Exception:
                    tags = []
                all_visual_tags[os.path.basename(file_path)] = tags
                progressive_state['history'].append({
                    'iteration': 1,
                    'concept': concept,
                    'image_file': os.path.basename(file_path),
                    'visual_tags': tags
                })

    # Save concept list and tags
    scenes_file = os.path.join(mode_folder, "final.json")
    with open(scenes_file, "w") as f:
        json.dump([concept], f, indent=2)
    tags_path = os.path.join(mode_folder, "visual_tags.json")
    with open(tags_path, "w") as tagf:
        json.dump(all_visual_tags, tagf, indent=2)

    return {
        "session_id": req.session_id,
        "stage": "final_progressive",
        "images": images
    }


class FinalProgressiveFeedbackRequest(BaseModel):
    session_id: str
    positive_tags: list[str]
    negative_tags: list[str]

@app.post("/api/final-progressive-feedback")
def final_progressive_feedback(req: FinalProgressiveFeedbackRequest):
    """Submit feedback tags and run the next progressive iteration (up to 4)."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    descriptor = session['descriptor']
    base_folder = session['folder']
    mode_folder_name = "[Progressive]final"
    mode_folder = os.path.join(base_folder, mode_folder_name)
    os.makedirs(mode_folder, exist_ok=True)

    progressive_state = session.get('final_progressive')
    if not progressive_state:
        raise HTTPException(400, "Progressive session not initialized")

    # Accumulate feedback
    pos = progressive_state.get('accumulated_positive', [])
    neg = progressive_state.get('accumulated_negative', [])
    for t in req.positive_tags:
        if t not in pos:
            pos.append(t)
    for t in req.negative_tags:
        if t not in neg:
            neg.append(t)
    progressive_state['accumulated_positive'] = pos
    progressive_state['accumulated_negative'] = neg

    iteration = progressive_state.get('iteration', 1)
    if iteration >= 4:
        return {"message": "Progressive mode already completed"}

    # Build user_pref from disk selections if available
    user_pref = {}
    selections = session.get('selections') or {}
    for stage_name in ['impression', 'spatial', 'objects', 'ambient']:
        if stage_name in selections:
            selected_image_id = selections[stage_name]
            stage_folder = os.path.join(base_folder, stage_name)
            json_path = os.path.join(stage_folder, f"{stage_name}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path) as f:
                        scenes = json.load(f)
                    idx = int(selected_image_id.split('_')[1])
                    if isinstance(scenes, list) and idx < len(scenes):
                        user_pref[stage_name] = scenes[idx]
                except Exception:
                    pass

    from util import designer_final_one_concept, generator_final_one_concept
    from tag_extraction import extract_visual_elements_from_image, prompt as general_prompt

    # Next iteration index
    next_iter_index = iteration
    concept = designer_final_one_concept(
        "",
        pos,
        neg,
        descriptor,
        user_pref,
        base_folder,
        "final"
    )

    prefix = f"final_{next_iter_index}"
    files = generator_final_one_concept(
        "",
        descriptor,
        concept,
        user_pref,
        pos,
        neg,
        mode_folder,
        prefix,
        base_folder,
        "final"
    )

    # Append to saved concepts
    scenes_file = os.path.join(mode_folder, "final.json")
    existing_concepts = []
    if os.path.exists(scenes_file):
        try:
            with open(scenes_file, 'r') as f:
                existing_concepts = json.load(f)
        except Exception:
            existing_concepts = []
    existing_concepts.append(concept)
    with open(scenes_file, 'w') as f:
        json.dump(existing_concepts, f, indent=2)

    images = []
    tags_path = os.path.join(mode_folder, "visual_tags.json")
    all_visual_tags = {}
    if os.path.exists(tags_path):
        try:
            with open(tags_path, 'r') as f:
                all_visual_tags = json.load(f)
        except Exception:
            all_visual_tags = {}

    if files:
        for file_path in files:
            if file_path.endswith('.png'):
                image_id = os.path.splitext(os.path.basename(file_path))[0]
                image_url = f"/sessions/{req.session_id}/{mode_folder_name}/{os.path.basename(file_path)}"
                images.append({"id": image_id, "url": image_url})
                try:
                    tags = extract_visual_elements_from_image(file_path, general_prompt)
                except Exception:
                    tags = []
                all_visual_tags[os.path.basename(file_path)] = tags
                progressive_state['history'].append({
                    'iteration': next_iter_index + 1,
                    'concept': concept,
                    'image_file': os.path.basename(file_path),
                    'visual_tags': tags
                })

    with open(tags_path, 'w') as tagf:
        json.dump(all_visual_tags, tagf, indent=2)

    # Increment iteration
    progressive_state['iteration'] = iteration + 1

    return {
        "session_id": req.session_id,
        "stage": "final_progressive",
        "iteration": progressive_state['iteration'],
        "images": images
    }

class GenerationError(Exception):
    pass

@app.exception_handler(GenerationError)
async def generation_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": str(exc)}
    )

def safe_generate_images(stage, prompt, scene, folder, prefix):
    try:
        return generator(prompt, scene, folder, prefix)
    except Exception as e:
        raise GenerationError(f"Failed to generate images for {stage}: {str(e)}")

def handle_parallel_to_final(req: FeedbackRequest, session: dict, folder: str, descriptor: str):
    """Handle the transition from parallel stages to final stage."""
    try:
        write_status(folder, "Starting FINAL stage with collected preferences...")
        
        # Build user_pref from the collected selections
        user_pref = {}
        selections = req.preferences.get('selections', {})
        
        # Load scene data for each selected image
        for stage_name in ['impression', 'spatial', 'objects', 'ambient']:
            if stage_name in selections:
                selected_image_id = selections[stage_name]
                
                # Load the scene data for this selection
                stage_folder = os.path.join(folder, stage_name)
                json_path = os.path.join(stage_folder, f"{stage_name}.json")
                
                if os.path.exists(json_path):
                    with open(json_path) as f:
                        scenes = json.load(f)
                    
                    # Extract index from image ID (e.g., "impression_0_0" -> 0)
                    try:
                        selected_index = int(selected_image_id.split('_')[1])
                        if isinstance(scenes, list) and selected_index < len(scenes):
                            user_pref[stage_name] = scenes[selected_index]
                            write_status(folder, f"Loaded {stage_name} preference: {scenes[selected_index].get('concept_name', 'unknown')}")
                    except (ValueError, IndexError) as e:
                        write_status(folder, f"Failed to parse selection for {stage_name}: {str(e)}")
        
        # Generate final stage
        narrative_prompt, image_prompt = PROMPTS['final']  # Use sequential prompts for final
        write_status(folder, "Generating final stage with all preferences...")
        
        results, _ = run_stage_seq_parallel_optimized(  # Use optimized function for final stage
            'final',
            narrative_prompt,
            image_prompt, 
            descriptor,
            user_pref,
            folder
        )
        
        # Prepare response
        images = []
        for i, (scene, files) in enumerate(results):
            for file in files:
                if file.endswith('.png'):
                    image_id = os.path.splitext(os.path.basename(file))[0]
                    image_url = f"/sessions/{req.session_id}/final/{os.path.basename(file)}"
                    images.append({"id": image_id, "url": image_url})
        
        write_status(folder, f"Final stage completed with {len(images)} images!")
        
        return {
            "next_stage": None,  # Final stage
            "images": images
        }
        
    except Exception as e:
        error_msg = f"Failed to generate final stage: {str(e)}"
        write_status(folder, error_msg)
        raise HTTPException(500, error_msg)


def run_refinement_stage(stage_name: str,
                         refinement_prompt: str,
                         image_prompt: str,
                         descriptor: str,
                         selected_json: dict,
                         tag_preferences: dict,
                         session_folder: str) -> list:
    """
    Run a refinement stage with the selected JSON and tag preferences.
    This generates 4 refined concepts based on user preferences WITHOUT tag extraction.
    
    Args:
        stage_name: Name of the refinement stage (e.g., 'impression_refinement')
        refinement_prompt: The refinement prompt template
        image_prompt: The image generation prompt
        descriptor: Original user description
        selected_json: The selected interpretation JSON from previous exploration stage
        tag_preferences: Dictionary with 'positive' and 'negative' tag lists
        session_folder: Path to session folder
    
    Returns:
        list: Results with (scene, files) tuples
    """
    write_status(session_folder, f"Starting {stage_name.upper()} stage (Refinement)")
    
    # Create stage folder
    folder = os.path.join(session_folder, stage_name)
    os.makedirs(folder, exist_ok=True)
    write_status(session_folder, f"Created folder: {folder}")
    
    # Format the refinement prompt with inputs
    formatted_prompt = refinement_prompt.replace(
        '[USER DESCRIPTION]\n<user text>\n[/USER DESCRIPTION]',
        f'[USER DESCRIPTION]\n{descriptor}\n[/USER DESCRIPTION]'
    )
    
    formatted_prompt = formatted_prompt.replace(
        '[SELECTED_INTERPRETATION_JSON]\n<one of the 4 exploration outputs, full JSON>\n[/SELECTED_INTERPRETATION_JSON]',
        f'[SELECTED_INTERPRETATION_JSON]\n{json.dumps(selected_json, indent=2)}\n[/SELECTED_INTERPRETATION_JSON]'
    )
    
    # Format tag preferences
    positive_tags = tag_preferences.get('positive', [])[:5]  # Top 5
    negative_tags = tag_preferences.get('negative', [])[:3]  # Top 3
    tag_prefs_json = {
        "positive": positive_tags,
        "negative": negative_tags
    }
    
    formatted_prompt = formatted_prompt.replace(
        '[TAG PREFERENCES]\n{\n  "positive": ["P1","P2","P3","P4","P5"],   // ordered; left = higher priority\n  "negative": ["N1","N2","N3"]              // ordered; left = stricter constraint\n}\n[/TAG PREFERENCES]',
        f'[TAG PREFERENCES]\n{json.dumps(tag_prefs_json, indent=2)}\n[/TAG PREFERENCES]'
    )
    
    write_status(session_folder, f"Generating refined concepts with user preferences...")
    write_status(session_folder, f"Positive concepts: {positive_tags}")
    write_status(session_folder, f"Negative concepts: {negative_tags}")
    
    # Generate scene descriptions using sequential designer
    scenes = designer_seq(formatted_prompt, descriptor, {}, session_folder, stage_name)
    
    if not scenes:
        error_msg = f"No scenes generated for {stage_name} stage"
        write_status(session_folder, error_msg)
        raise RuntimeError(error_msg)
    
    write_status(session_folder, f"Generated {len(scenes)} refined concepts")
    
    # Save scenes JSON
    scenes_file = os.path.join(folder, f"{stage_name}.json")
    with open(scenes_file, "w") as f:
        json.dump(scenes, f, indent=2)
    write_status(session_folder, f"Saved scenes to: {stage_name}.json")
    
    # Generate images WITHOUT tag extraction (use generator_seq_parallel instead of generator_seq_parallel_with_tags)
    write_status(session_folder, f"Generating images for all {len(scenes)} refined concepts...")
    prefix_base = stage_name
    
    from util import generator_seq_parallel
    
    try:
        results = generator_seq_parallel(
            image_prompt, descriptor, scenes, {}, folder, prefix_base, session_folder, stage_name
        )
        write_status(session_folder, f"Successfully generated images for {len(results)} refined concepts")
    except Exception as e:
        error_msg = f"Refinement stage image generation failed: {str(e)}"
        write_status(session_folder, error_msg)
        raise RuntimeError(error_msg)
    
    write_status(session_folder, f"{stage_name.upper()} stage completed successfully!")
    write_status(session_folder, f"Generated {len(results)} concepts with {sum(len(files) for _, files in results)} total images")
    
    return results


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    print(f"🔍 DEBUG - Feedback request: stage={req.stage}, selected_image_id={req.selected_image_id}")
    
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    descriptor = session['descriptor']
    # user_pref might not exist for parallel sessions
    user_pref = session.get('user_pref', {})
    folder = session['folder']

    # Save preferences to file
    preferences_file = os.path.join(folder, "preferences.json")
    with open(preferences_file, 'w') as f:
        json.dump(req.preferences, f, indent=2)
        print(f"Saved preferences to {preferences_file}")

    # Handle parallel-to-final transition - REDIRECT TO MODE SELECTION
    if req.stage == 'parallel-to-final':
        return {"next_stage": "mode-selection", "images": []}

    # Update preferences with selected image
    current_stage = req.stage
    selected_image_id = req.selected_image_id
    
    # Update session user_pref for sequential mode
    # IMPORTANT: If this is a refinement stage, store in the BASE stage key to replace it
    if current_stage.endswith('_refinement'):
        base_stage = current_stage.replace('_refinement', '')
        pref_key = base_stage
        print(f"[REFINEMENT] Storing selection from '{current_stage}' as '{pref_key}' (replaces base stage)")
    else:
        pref_key = current_stage
    
    # Load the full concept JSON, not just the ID
    stage_folder = os.path.join(folder, current_stage)
    json_file = os.path.join(stage_folder, f"{current_stage}.json")
    
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                all_scenes = json.load(f)
            
            # Extract scene index from image ID (e.g., "impression_refinement_2_0" -> 2)
            if selected_image_id and '_' in selected_image_id:
                parts = selected_image_id.split('_')
                if len(parts) >= 2:
                    # Get the second-to-last part as the concept index
                    scene_idx = int(parts[-2])
                    if 0 <= scene_idx < len(all_scenes):
                        # Store full concept JSON
                        user_pref[pref_key] = all_scenes[scene_idx]
                        print(f"✅ Stored full concept for '{pref_key}': {all_scenes[scene_idx].get('concept_name', 'Unknown')}")
                    else:
                        # Fallback: just store ID
                        user_pref[pref_key] = selected_image_id
                        print(f"⚠️  Scene index {scene_idx} out of range, storing ID only")
                else:
                    user_pref[pref_key] = selected_image_id
            else:
                user_pref[pref_key] = selected_image_id
        except Exception as e:
            print(f"⚠️  Error loading scene JSON: {e}")
            user_pref[pref_key] = selected_image_id
    else:
        # Fallback: just store ID
        user_pref[pref_key] = selected_image_id
        print(f"⚠️  JSON file not found for {current_stage}, storing ID only")
    
    session['user_pref'] = user_pref
    
    # Debug: Print current user_pref state
    print(f"[USER_PREF] Current state:")
    for key, val in user_pref.items():
        if isinstance(val, dict):
            print(f"  {key}: {val.get('concept_name', 'Unknown concept')}")
        else:
            print(f"  {key}: {val}")
    
    # IMPORTANT: Update preferences.json with the full user_pref structure
    # This ensures all concept JSONs (including refinements) are persisted
    try:
        # Load existing preferences
        if os.path.exists(preferences_file):
            with open(preferences_file, 'r') as f:
                preferences_data = json.load(f)
        else:
            preferences_data = req.preferences if req.preferences else {}
        
        # Update with descriptor if not present
        if 'descriptor' not in preferences_data:
            preferences_data['descriptor'] = descriptor
        
        # Update selections tracking (image IDs for UI)
        if 'selections' not in preferences_data:
            preferences_data['selections'] = {}
        preferences_data['selections'][current_stage] = selected_image_id
        
        # Update user_pref (full concept JSONs for generation)
        if 'user_pref' not in preferences_data:
            preferences_data['user_pref'] = {}
        preferences_data['user_pref'] = user_pref.copy()
        
        # Track refinement history (helpful for debugging)
        if 'refinement_history' not in preferences_data:
            preferences_data['refinement_history'] = []
        if current_stage.endswith('_refinement'):
            preferences_data['refinement_history'].append({
                'stage': current_stage,
                'base_stage': pref_key,
                'selected_id': selected_image_id,
                'concept_name': user_pref[pref_key].get('concept_name', 'Unknown') if isinstance(user_pref.get(pref_key), dict) else 'N/A'
            })
        
        # Save updated preferences
        with open(preferences_file, 'w') as f:
            json.dump(preferences_data, f, indent=2)
        
        print(f"✅ Saved complete preferences to {preferences_file}")
        print(f"   - user_pref keys: {list(user_pref.keys())}")
        print(f"   - selections: {preferences_data['selections']}")
        
    except Exception as e:
        print(f"⚠️  Error updating preferences.json: {e}")
        import traceback
        traceback.print_exc()
    
    # Transition logic for sequential
    try:
        # find current stage index
        current_idx = None
        for idx, stage_name in enumerate(STAGES):
            if stage_name == current_stage:
                current_idx = idx
                break
        
        if current_idx is None:
            raise HTTPException(400, f"Invalid stage: {current_stage}")
        
        # Check if this is the last refinement stage (before mode-selection)
        if current_stage == 'ambient_refinement':
            return {"next_stage": "mode-selection", "images": []}
        
        if current_idx + 1 < len(STAGES):
            next_stage = STAGES[current_idx + 1]
            narrative_prompt, image_prompt = PROMPTS[next_stage]
            
            print(f"Generating next stage: {next_stage}")
            
            # Check if next stage is a refinement stage
            if next_stage in REFINEMENT_STAGES:
                # Refinement stage: need selected JSON and tag preferences
                # Get the base exploration stage (e.g., 'impression' from 'impression_refinement')
                base_stage = next_stage.replace('_refinement', '')
                
                # Use user_pref directly - it already has the correct (refined if done) concept!
                # This is simpler and more reliable than parsing image IDs
                selected_json = user_pref.get(base_stage)
                
                if not selected_json:
                    raise HTTPException(400, f"No concept selected for {base_stage} stage. Please complete {base_stage} stage first.")
                
                # Validate it's a dict with required fields
                if not isinstance(selected_json, dict):
                    raise HTTPException(400, f"Invalid concept data for {base_stage} stage")
                
                print(f"[REFINEMENT INPUT] Using {base_stage} concept: {selected_json.get('concept_name', 'Unknown')}")
                
                # Get tag preferences from concept refinement system
                # First, collect image IDs from the base exploration stage
                exploration_stage_folder = os.path.join(folder, base_stage)
                exploration_images = []
                if os.path.exists(exploration_stage_folder):
                    for file in os.listdir(exploration_stage_folder):
                        if file.endswith('.png'):
                            img_id = os.path.splitext(file)[0]
                            exploration_images.append(img_id)
                
                # Access the concept refinement session
                refinement_session = get_refinement_session(req.session_id, base_stage, exploration_images)
                
                # Get categorized concepts
                categorized = refinement_session.get_categorized_concepts()
                positive_concept_ids = categorized.get('positive', [])
                negative_concept_ids = categorized.get('negative', [])
                
                # Sort by weight (score) and get top 5 positive and top 3 negative
                positive_with_weights = []
                for concept_id in positive_concept_ids:
                    if concept_id in refinement_session.concept_states:
                        weight = refinement_session.concept_states[concept_id].score
                        positive_with_weights.append((concept_id, weight))
                
                negative_with_weights = []
                for concept_id in negative_concept_ids:
                    if concept_id in refinement_session.concept_states:
                        weight = refinement_session.concept_states[concept_id].score
                        negative_with_weights.append((concept_id, weight))
                
                # Sort by weight (descending for positive, ascending for negative)
                positive_with_weights.sort(key=lambda x: x[1], reverse=True)
                negative_with_weights.sort(key=lambda x: x[1])
                
                # Get labels for top concepts
                positive_tags = []
                for concept_id, _ in positive_with_weights[:5]:
                    concept = next((c for c in refinement_session.concepts if c.id == concept_id), None)
                    if concept:
                        positive_tags.append(concept.label)
                
                negative_tags = []
                for concept_id, _ in negative_with_weights[:3]:
                    concept = next((c for c in refinement_session.concepts if c.id == concept_id), None)
                    if concept:
                        negative_tags.append(concept.label)
                
                tag_preferences = {
                    'positive': positive_tags,
                    'negative': negative_tags
                }
                
                print(f"Refinement stage tag preferences: {tag_preferences}")
                
                # Save concept preferences to preferences.json
                try:
                    if os.path.exists(preferences_file):
                        with open(preferences_file, 'r') as f:
                            preferences_data = json.load(f)
                    else:
                        preferences_data = {}
                    
                    if 'concept_preferences' not in preferences_data:
                        preferences_data['concept_preferences'] = {}
                    
                    preferences_data['concept_preferences'][base_stage] = tag_preferences
                    
                    with open(preferences_file, 'w') as f:
                        json.dump(preferences_data, f, indent=2)
                    
                    print(f"✅ Saved concept preferences for {base_stage} to preferences.json")
                except Exception as e:
                    print(f"⚠️  Error saving concept preferences: {e}")
                
                # Generate refinement stage using PBO + SDXL
                write_status(folder, f"🔄 Starting PBO refinement for {base_stage}...")
                
                # Initialize PBO with tag cluster concepts
                visual_tags_path = os.path.join(exploration_stage_folder, "visual_tags.json")
                if not os.path.exists(visual_tags_path):
                    raise HTTPException(
                        404,
                        f"Visual tags not found for {base_stage}. Cannot run refinement without tags."
                    )
                
                with open(visual_tags_path, 'r') as f:
                    visual_tags_data = json.load(f)
                
                # Build image_tags dict
                image_tags = {}
                for image_id in exploration_images:
                    image_filename = f"{image_id}.png"
                    if image_filename in visual_tags_data:
                        image_tags[image_id] = visual_tags_data[image_filename]
                    else:
                        image_tags[image_id] = []
                
                # Initialize concepts if not already done
                if not refinement_session.initialized:
                    write_status(folder, "🔨 Clustering tags into concepts...")
                    refinement_session.initialize_from_tags(image_tags)
                    write_status(folder, f"✅ Created {len(refinement_session.concepts)} tag cluster concepts")
                
                # Initialize StageRefiner with PBO
                refiner = get_or_create_pbo_refiner(
                    session_id=req.session_id,
                    stage=base_stage
                )
                
                write_status(folder, f"🎲 Generating 4 PBO proposals...")
                
                # Propose 4 weight mixtures
                proposals = refiner.propose_next_4(
                    negatives=None,
                    w_current=None,
                    fit_first=True
                )
                
                write_status(folder, "🎨 Generating images with SDXL...")
                
                # Load the selected favorite image as reference
                selected_image_id = req.selected_image_id  # From feedback request
                print(f"[PBO] Looking for reference image with ID: {selected_image_id}")
                
                selected_image_path = None
                if selected_image_id:
                    # Build expected image path from selected_image_id
                    # e.g., "impression_2_0" -> "impression/impression_2_0.png"
                    image_filename = f"{selected_image_id}.png"
                    selected_image_path = os.path.join(exploration_stage_folder, image_filename)
                    
                    print(f"[PBO] Checking path: {selected_image_path}")
                    if not os.path.exists(selected_image_path):
                        # Try alternative: maybe it's just the stage + index
                        # e.g., "impression_2" -> "impression_2_0.png"
                        alt_filename = f"{selected_image_id}_0.png"
                        alt_path = os.path.join(exploration_stage_folder, alt_filename)
                        print(f"[PBO] Trying alternative: {alt_path}")
                        if os.path.exists(alt_path):
                            selected_image_path = alt_path
                        else:
                            selected_image_path = None
                
                reference_image = None
                if selected_image_path and os.path.exists(selected_image_path):
                    from PIL import Image as PILImage
                    reference_image = PILImage.open(selected_image_path)
                    write_status(folder, f"📷 Using reference image: {os.path.basename(selected_image_path)}")
                    print(f"[PBO] ✅ Loaded reference image: {selected_image_path}")
                else:
                    write_status(folder, f"⚠️ Reference image not found for ID '{selected_image_id}', using txt2img mode")
                    print(f"[PBO] ⚠️ Reference image not found. Selected ID: {selected_image_id}")
                
                # Generate images using SDXL
                sdxl_runner = get_sdxl_runner()
                pil_images = refiner.generate_images_from_proposals(
                    proposals=proposals,
                    sdxl_runner=sdxl_runner,
                    seed_base=42,
                    verbose=False,
                    init_image=reference_image,
                    descriptor=descriptor  # User description from session
                )
                
                # Save images to Round 1 folder
                refinement_folder = os.path.join(folder, next_stage)
                round_1_folder = os.path.join(refinement_folder, "round_1")
                os.makedirs(round_1_folder, exist_ok=True)
                
                results = []
                for idx, pil_img in enumerate(pil_images):
                    # Save to round_1 folder
                    image_filename = f"image_{idx}.png"
                    image_path = os.path.join(round_1_folder, image_filename)
                    pil_img.save(image_path)
                    
                    # Also save with legacy naming for compatibility
                    legacy_filename = f"{next_stage}_{idx}_0.png"
                    legacy_path = os.path.join(refinement_folder, legacy_filename)
                    pil_img.save(legacy_path)
                    
                    write_status(folder, f"💾 Saved {image_filename}")
                    
                    # Create result tuple compatible with existing code
                    concept_data = {
                        "concept_name": f"PBO Mixture {idx+1}",
                        "weight_vector": proposals[idx].tolist()
                    }
                    results.append((concept_data, [legacy_path]))  # Use legacy path for compatibility
                
                # Save Round 1 weight vectors
                weights_file = os.path.join(round_1_folder, "weights.json")
                with open(weights_file, "w") as f:
                    json.dump({
                        "round": 1,
                        "proposals": [p.tolist() for p in proposals],
                        "concept_labels": [c['label'] for c in refiner.concepts],
                        "reference_image": selected_image_id
                    }, f, indent=2)
                
                write_status(folder, f"✅ PBO refinement complete! Generated {len(results)} images")
            else:
                # Regular exploration stage: use standard generation
                results, user_pref = run_stage_seq_parallel_optimized(
                    next_stage,
                    narrative_prompt,
                    image_prompt,
                    descriptor,
                    user_pref,
                    folder
                )
            
            # Prepare response
            images = []
            for i, (scene, files) in enumerate(results):
                for file in files:
                    if file.endswith('.png'):
                        # Use the actual filename (without extension) as ID to match the initial generation
                        image_id = os.path.splitext(os.path.basename(file))[0]
                        image_url = f"/sessions/{req.session_id}/{next_stage}/{os.path.basename(file)}"
                        images.append({"id": image_id, "url": image_url})
            
            print(f"Generated {len(images)} images for {next_stage}")
            print(f"Generated image IDs: {[img['id'] for img in images]}")
            return {"next_stage": next_stage, "images": images}
        else:
            print("Reached final stage")
            return {"next_stage": None, "images": []}
            
    except Exception as e:
        print(f"Error in feedback endpoint: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mode-selection/{session_id}")
def get_mode_selection(session_id: str):
    """Get available final generation modes for a session."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    # Check if this is a cumulative tags session
    is_cumulative_tags = (
        session.get('mode') == 'cumulative_tags' or
        session_id.startswith("[cumtag]") or
        session_id.startswith("[seqtag]")
    )
    
    if is_cumulative_tags:
        # Return modes specific to cumulative tags
        return {
            "modes": [
                {
                    "id": "mode1",
                    "name": "Basic Final Generation", 
                    "description": "Generate final images using basic cumulative tags approach"
                },
                {
                    "id": "mode2", 
                    "name": "With Cumulative Tags",
                    "description": "Generate final images incorporating all accumulated tag preferences"
                },
                {
                    "id": "mode3", 
                    "name": "With Tags + Reference Images", 
                    "description": "Generate final images using cumulative tags and reference images from selected concepts"
                },
                {
                    "id": "mode4",
                    "name": "Enhanced Cumulative Tags",
                    "description": "Generate final images with enhanced cumulative tags processing"
                }
            ]
        }
    else:
        # Return modes for regular sessions
        return {
            "modes": [
                {
                    "id": "mode1",
                    "name": "only img JSON", 
                    "description": "Selected images' JSON only (current default)"
                },
                {
                    "id": "mode2", 
                    "name": "img JSON + Tags",
                    "description": "Selected images' JSON + tag preferences"
                },
                {
                    "id": "mode3", 
                    "name": "img JSON + Tags + Images", 
                    "description": "Selected images' JSON + tag preferences + actual selected images"
                },
                {
                    "id": "mode4",
                    "name": "Enhanced User Preferences",
                    "description": "Per-layer tag preferences with comprehensive user feedback structure"
                }
            ]
        }


@app.post("/api/generate-final/{mode}")
def generate_final(mode: str, req: FeedbackRequest):
    """Generate final images using the specified mode."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    descriptor = session['descriptor']
    base_folder = session['folder']
    
    # Create mode-specific folder
    mode_folder_names = {
        "mode1": "final",
        "mode2": "[with Tags]final", 
        "mode3": "[with Imgs]final",
        "mode4": "[Enhanced Prefs]final"
    }
    
    if mode not in mode_folder_names:
        raise HTTPException(400, f"Invalid mode: {mode}")
    
    mode_folder_name = mode_folder_names[mode]
    mode_folder = os.path.join(base_folder, mode_folder_name)
    
    # If folder exists, remove and recreate to ensure clean state
    if os.path.exists(mode_folder):
        import shutil
        shutil.rmtree(mode_folder)
        write_status(base_folder, f"Removed existing {mode_folder_name} folder")
    
    os.makedirs(mode_folder, exist_ok=True)
    
    try:
        write_status(base_folder, f"Starting FINAL stage - {mode_folder_names[mode]}...")
        
        # Build user_pref from the collected selections
        user_pref = {}
        selections = req.preferences.get('selections', {})
        
        # Check if this is an uploaded session
        is_uploaded = session.get('uploaded', False)
        
        if is_uploaded:
            # For uploaded sessions, load scene data directly from preferences
            write_status(base_folder, "Using uploaded session data...")
            
            # Load scene data for each selected image  
            for stage_name in ['impression', 'spatial', 'objects', 'ambient']:
                if stage_name in selections:
                    selected_image_id = selections[stage_name]
                    
                    # Load the scene data for this selection
                    stage_folder = os.path.join(base_folder, stage_name)
                    json_path = os.path.join(stage_folder, f"{stage_name}.json")
                    
                    if os.path.exists(json_path):
                        with open(json_path) as f:
                            scenes = json.load(f)
                        
                        # Extract index from image ID (e.g., "impression_0_0" -> 0)
                        try:
                            selected_index = int(selected_image_id.split('_')[1])
                            if isinstance(scenes, list) and selected_index < len(scenes):
                                user_pref[stage_name] = scenes[selected_index]
                                write_status(base_folder, f"Loaded {stage_name} preference: {scenes[selected_index].get('concept_name', 'unknown')}")
                        except (ValueError, IndexError) as e:
                            write_status(base_folder, f"Failed to parse selection for {stage_name}: {str(e)}")
            
            # Get descriptor from session or preferences 
            descriptor = session.get('descriptor', 'Uploaded session')
        else:
            # For regular sessions, load scene data for each selected image
            for stage_name in ['impression', 'spatial', 'objects', 'ambient']:
                if stage_name in selections:
                    selected_image_id = selections[stage_name]
                    
                    # Load the scene data for this selection
                    stage_folder = os.path.join(base_folder, stage_name)
                    json_path = os.path.join(stage_folder, f"{stage_name}.json")
                    
                    if os.path.exists(json_path):
                        with open(json_path) as f:
                            scenes = json.load(f)
                        
                        # Extract index from image ID (e.g., "impression_0_0" -> 0)
                        try:
                            selected_index = int(selected_image_id.split('_')[1])
                            if isinstance(scenes, list) and selected_index < len(scenes):
                                user_pref[stage_name] = scenes[selected_index]
                                write_status(base_folder, f"Loaded {stage_name} preference: {scenes[selected_index].get('concept_name', 'unknown')}")
                        except (ValueError, IndexError) as e:
                            write_status(base_folder, f"Failed to parse selection for {stage_name}: {str(e)}")
        
        # Always use sequential mode prompts
        if mode == "mode1":
            # Mode 1: JSON only (current default)
            narrative_prompt, image_prompt = FINAL_PROMPT, FINAL_GENERATOR_PROMPT
            
            # Use optimized final mode 1 workflow (same as mode 2 but without tags)
            write_status(base_folder, f"DEBUG: Starting OPTIMIZED Mode 1 generation with {len(user_pref)} user preferences")
            results = run_final_mode1_optimized(
                narrative_prompt, image_prompt, descriptor, user_pref,
                mode_folder, base_folder
            )
            write_status(base_folder, f"DEBUG: Mode 1 completed with {len(results)} scene results")
            
        elif mode == "mode2":
            # Mode 2: JSON + Tags - Use custom workflow with tags
            narrative_prompt, image_prompt = FINAL_PROMPT_TAGS, FINAL_GENERATOR_PROMPT_TAGS
            
            # Get tag data
            tag_data = req.preferences.get('tags', {})
            
            # Debug logging for tag preferences
            write_status(base_folder, f"DEBUG: Tag data received: {json.dumps(tag_data, indent=2)}")
            if 'parallel' in tag_data or 'spatial' in tag_data:
                tags_list = tag_data.get('parallel', []) + tag_data.get('spatial', [])
                negative_tags = [t['tag'] for t in tags_list if t.get('preference') == 'negative']
                positive_tags = [t['tag'] for t in tags_list if t.get('preference') == 'positive']
                write_status(base_folder, f"DEBUG: Negative tags to avoid: {negative_tags}")
                write_status(base_folder, f"DEBUG: Positive tags to include: {positive_tags}")
            
            # Use optimized final mode 2 workflow
            write_status(base_folder, f"DEBUG: Starting OPTIMIZED Mode 2 generation with {len(user_pref)} user preferences")
            results = run_final_mode2_optimized(
                narrative_prompt, image_prompt, descriptor, user_pref, tag_data,
                mode_folder, base_folder
            )
            write_status(base_folder, f"DEBUG: Mode 2 completed with {len(results)} scene results")
            
        elif mode == "mode3":
            # Mode 3: JSON + Tags + Images
            narrative_prompt, image_prompt = FINAL_PROMPT_TAGS, FINAL_GENERATOR_PROMPT_IMGS
            
            # Get tag data
            tag_data = req.preferences.get('tags', {})
            
            # Debug logging for tag preferences
            write_status(base_folder, f"DEBUG: Tag data received: {json.dumps(tag_data, indent=2)}")
            if 'parallel' in tag_data or 'spatial' in tag_data:
                tags_list = tag_data.get('parallel', []) + tag_data.get('spatial', [])
                negative_tags = [t['tag'] for t in tags_list if t.get('preference') == 'negative']
                positive_tags = [t['tag'] for t in tags_list if t.get('preference') == 'positive']
                write_status(base_folder, f"DEBUG: Negative tags to avoid: {negative_tags}")
                write_status(base_folder, f"DEBUG: Positive tags to include: {positive_tags}")
            
            # Get reference images
            reference_images = []
            for stage_name in ['impression', 'spatial', 'objects', 'ambient']:
                if stage_name in selections:
                    selected_image_id = selections[stage_name]
                    # Construct image path
                    img_path = os.path.join(base_folder, stage_name, f"{selected_image_id}.png")
                    if os.path.exists(img_path):
                        reference_images.append(img_path)
                        write_status(base_folder, f"Added reference image: {stage_name}/{selected_image_id}.png")
                    else:
                        write_status(base_folder, f"Warning: Reference image not found: {img_path}")
            
            # Use optimized final mode 3 workflow
            write_status(base_folder, f"DEBUG: Starting OPTIMIZED Mode 3 generation with {len(reference_images)} reference images")
            results = run_final_mode3_optimized(
                narrative_prompt, image_prompt, descriptor, user_pref, tag_data, reference_images,
                mode_folder, base_folder
            )
            write_status(base_folder, f"DEBUG: Mode 3 completed with {len(results)} scene results")
            
        elif mode == "mode4":
            # Mode 4: Enhanced User Preferences with Per-Layer Tags
            narrative_prompt, image_prompt = FINAL_PROMPT_TAGS, FINAL_GENERATOR_PROMPT_TAGS
            
            # Generate enhanced user preferences using mode4.py
            try:
                # Construct file paths for mode4.generate_user_preference
                impression_path = os.path.join(base_folder, "impression", "impression.json")
                spatial_path = os.path.join(base_folder, "spatial", "spatial.json")
                objects_path = os.path.join(base_folder, "objects", "objects.json")
                ambient_path = os.path.join(base_folder, "ambient", "ambient.json")
                preferences_path = None
                
                # Validate that required JSON files exist
                missing_files = []
                for name, path in [("impression", impression_path), ("spatial", spatial_path), ("objects", objects_path), ("ambient", ambient_path)]:
                    if not os.path.exists(path):
                        missing_files.append(f"{name}: {path}")
                
                if missing_files:
                    error_msg = f"Missing required JSON files for Mode 4: {', '.join(missing_files)}"
                    write_status(base_folder, f"ERROR: {error_msg}")
                    raise HTTPException(500, error_msg)
                
                # Create a temporary preferences file with correct structure for mode4.py
                import tempfile
                
                # Transform req.preferences to match mode4.py expected structure
                tag_data = req.preferences.get('tags', {})
                
                # Ensure tags are in the "parallel" format that mode4.py expects
                if 'parallel' not in tag_data and (isinstance(tag_data, dict) and tag_data):
                    # If tags exist but not in "parallel" format, wrap them
                    if isinstance(tag_data, list):
                        # If tag_data is a list, put it under "parallel"
                        formatted_tags = {"parallel": tag_data}
                    else:
                        # If tag_data is a dict, check for known keys and consolidate
                        all_tags = []
                        for key in ['parallel', 'spatial', 'impression', 'ambient']:
                            if key in tag_data and isinstance(tag_data[key], list):
                                all_tags.extend(tag_data[key])
                        formatted_tags = {"parallel": all_tags} if all_tags else tag_data
                else:
                    formatted_tags = tag_data
                
                # Create properly formatted preferences for mode4.py
                mode4_preferences = {
                    "selections": req.preferences.get('selections', {}),
                    "tags": formatted_tags
                }
                
                write_status(base_folder, f"DEBUG: Mode4 preferences structure: {json.dumps(mode4_preferences, indent=2)}")
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                    json.dump(mode4_preferences, temp_file, indent=2)
                    preferences_path = temp_file.name
                
                # Generate enhanced user preferences
                write_status(base_folder, f"DEBUG: Calling generate_user_preference with files: {impression_path}, {spatial_path}, {objects_path}, {ambient_path}, {preferences_path}")
                enhanced_user_pref = generate_user_preference(
                    impression_path, spatial_path, objects_path, ambient_path, preferences_path
                )
                write_status(base_folder, f"DEBUG: Successfully generated enhanced user preferences")
                
                # Clean up temporary file
                os.unlink(preferences_path)
                
                write_status(base_folder, f"DEBUG: Generated enhanced user preferences with {len(enhanced_user_pref)} layers")
                
                # Use optimized final mode 4 workflow
                results = run_final_mode4_optimized(
                    narrative_prompt, image_prompt, descriptor, enhanced_user_pref,
                    mode_folder, base_folder
                )
                write_status(base_folder, f"DEBUG: Mode 4 completed with {len(results)} scene results")
                
            except Exception as e:
                write_status(base_folder, f"ERROR: Failed to generate enhanced user preferences: {str(e)}")
                raise HTTPException(500, f"Mode 4 preference generation failed: {str(e)}")

        elif mode == "mode5":
            # Mode 5: Progressive one-concept generation over 4 iterations
            from util import designer_final_one_concept, generator_final_one_concept
            from tag_extraction import extract_visual_elements_from_image, prompt as general_prompt

            write_status(base_folder, f"Starting FINAL Mode 5 - Progressive (4 iterations)")

            # Build user_pref from selections (same logic as above already populated)
            # user_pref variable is already built earlier in this function

            # Initialize progressive state
            progressive_state = {
                'iteration': 1,
                'accumulated_positive': [],
                'accumulated_negative': [],
                'history': []
            }

            # Persist state to session memory
            session['final_progressive'] = progressive_state

            # Ensure mode folder is ready
            os.makedirs(mode_folder, exist_ok=True)

            images = []
            concepts = []
            mode5_results = []
            all_visual_tags = {}

            # Run 4 iterations synchronously
            for i in range(4):
                pos = progressive_state['accumulated_positive']
                neg = progressive_state['accumulated_negative']

                write_status(base_folder, f"Mode5 Iteration {i+1}: designing with {len(pos)} positive and {len(neg)} negative tags")
                concept = designer_final_one_concept(
                    "",
                    pos,
                    neg,
                    descriptor,
                    user_pref,
                    base_folder,
                    "final"
                )
                concepts.append(concept)

                prefix = f"final_{i}"
                files = generator_final_one_concept(
                    "",
                    descriptor,
                    concept,
                    user_pref,
                    pos,
                    neg,
                    mode_folder,
                    prefix,
                    base_folder,
                    "final"
                )

                # Attach image
                for file_path in files:
                    if file_path.endswith('.png'):
                        image_id = os.path.splitext(os.path.basename(file_path))[0]
                        image_url = f"/sessions/{req.session_id}/{mode_folder_name}/{os.path.basename(file_path)}"
                        images.append({"id": image_id, "url": image_url})
                
                # collect results for common response builder
                mode5_results.append((concept, files))

                # Extract tags (first image only if present)
                if files:
                    file_path = files[0]
                    try:
                        tags = extract_visual_elements_from_image(file_path, general_prompt)
                    except Exception:
                        tags = []
                    all_visual_tags[os.path.basename(file_path)] = tags

                    # Record history (no auto-accumulation here)
                    progressive_state['history'].append({
                        'iteration': i+1,
                        'concept': concept,
                        'image_file': os.path.basename(file_path),
                        'visual_tags': tags
                    })

                        # For this generate-final/{mode5} endpoint, we won't auto-accumulate without user feedback.
                        # Accumulation will happen via /api/final-progressive-feedback.

            # Save concepts JSON
            scenes_file = os.path.join(mode_folder, "final.json")
            with open(scenes_file, "w") as f:
                json.dump(concepts, f, indent=2)

            # Save visual tags
            tags_path = os.path.join(mode_folder, "visual_tags.json")
            with open(tags_path, "w") as tagf:
                json.dump(all_visual_tags, tagf, indent=2)
            
            # Use common response path by setting results
            results = mode5_results
        
        # Prepare response - handle results as (scene, files) tuples like original workflow
        images = []
        if isinstance(results, list):
            for i, (scene, files) in enumerate(results):
                for file_path in files:
                    if file_path.endswith('.png'):
                        image_id = os.path.splitext(os.path.basename(file_path))[0]
                        image_url = f"/sessions/{req.session_id}/{mode_folder_name}/{os.path.basename(file_path)}"
                        images.append({"id": image_id, "url": image_url})
        
        write_status(base_folder, f"Final stage completed with {len(images)} images!")
        
        return {
            "next_stage": "final",
        "images": images,
        "mode": mode,
        "mode_name": mode_folder_names[mode]
        }
        
    except Exception as e:
        error_msg = f"Failed to generate final stage in {mode}: {str(e)}"
        write_status(base_folder, error_msg)
        raise HTTPException(500, error_msg)


@app.post("/api/generate-final-cumulative-tags/{mode}")
def generate_final_cumulative_tags(mode: str, req: FeedbackRequest):
    """Generate final images for cumulative tags mode using the specified mode."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    descriptor = session['descriptor']
    base_folder = session['folder']
    
    # Check if this is a cumulative tags session
    if not (req.session_id.startswith("[cumtag]") or req.session_id.startswith("[seqtag]") or session.get('mode') == 'cumulative_tags'):
        raise HTTPException(400, "This endpoint is only for cumulative tags sessions")
    
    try:
        write_status(base_folder, f"Starting FINAL stage for Cumulative Tags - {mode}...")
        
        # Load user preferences from session
        user_pref = session.get('user_pref', {})
        
        # If no user preferences in session, try to load from file
        if not user_pref:
            user_pref_file = os.path.join(base_folder, "user_preference.json")
            if os.path.exists(user_pref_file):
                try:
                    with open(user_pref_file, 'r') as f:
                        user_pref = json.load(f)
                    write_status(base_folder, f"Loaded user preferences from file with {len(user_pref)} stages")
                except Exception as e:
                    write_status(base_folder, f"Warning: Could not load user preferences from file: {str(e)}")
            else:
                write_status(base_folder, "Warning: No user preferences found in session or file")

        # Ensure a preferences.json mirror exists for consistency (seqtag sessions)
        try:
            preferences_mirror = {
                "selections": session.get('selections', {}),
                "user_pref": user_pref,
                "tags": session.get('cumulative_tags', {})
            }
            with open(os.path.join(base_folder, "preferences.json"), 'w') as pf:
                json.dump(preferences_mirror, pf, indent=2)
            write_status(base_folder, "Wrote preferences.json mirror for cumulative tags session")
        except Exception as e:
            write_status(base_folder, f"Warning: Failed to write preferences.json mirror: {str(e)}")
        
        # Load cumulative tags
        cumulative_tags_file = os.path.join(base_folder, "cumulative_tags.json")
        cumulative_tags = {}
        if os.path.exists(cumulative_tags_file):
            with open(cumulative_tags_file, 'r') as f:
                cumulative_tags = json.load(f)
            write_status(base_folder, f"Loaded cumulative tags with {len(cumulative_tags)} stages")
        else:
            write_status(base_folder, "Warning: No cumulative tags file found")
        
        # Progressive mode5 removed

        # Run the cumulative tags final stage (modes 1-4)
        results = run_cumulative_tags_final_stage(
            descriptor, user_pref, cumulative_tags, base_folder, mode
        )
        
        # Convert results to image items for response
        images = []
        mode_folder_names = {
            "mode1": "final",
            "mode2": "[with Tags]final",
            "mode3": "[with Imgs]final",
            "mode4": "[Enhanced Prefs]final",
            "mode5": "[Progressive]final"
        }
        mode_folder_name = mode_folder_names.get(mode, "final")
        for scene, files in results:
            for file_path in files:
                if os.path.exists(file_path):
                    image_id = os.path.splitext(os.path.basename(file_path))[0]
                    image_url = f"/sessions/{req.session_id}/{mode_folder_name}/{os.path.basename(file_path)}"
                    images.append({"id": image_id, "url": image_url})
        
        write_status(base_folder, f"Final stage for cumulative tags completed with {len(images)} images!")
        
        return {
            "next_stage": "final",
            "images": images,
            "mode": mode,
            "mode_name": f"Cumulative Tags Final - {mode}"
        }
        
    except Exception as e:
        error_msg = f"Failed to generate final stage for cumulative tags in {mode}: {str(e)}"
        write_status(base_folder, error_msg)
        raise HTTPException(500, error_msg)


def run_final_mode2(narrative_prompt: str, image_prompt: str, descriptor: str, user_pref: dict, tag_data: dict, mode_folder: str, session_folder: str):
    """
    Run final mode 2: JSON + Tags with proper multi-scene workflow
    """
    from tag_extraction import extract_visual_elements_from_image
    
    write_status(session_folder, f"Starting FINAL Mode 2 - JSON + Tags")
    
    # Generate multiple scene descriptions using designer
    write_status(session_folder, f"Generating concepts with tags...")
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
    all_visual_tags = {}
    
    for i, scene in enumerate(scenes):
        concept_name = scene.get('concept_name', f'Scene {i+1}')
        write_status(session_folder, f"Generating images for: {concept_name}")
        
        prefix = f"final_{i}"
        
        try:
            # Use mode 2 generator with tags
            files = generator_final_mode2(
                narrative_prompt, image_prompt, descriptor, scene, user_pref, tag_data,
                mode_folder, prefix, session_folder, "final"
            )
            results.append((scene, files))
            write_status(session_folder, f"Generated {len(files)} images for {concept_name}")
            
            # Extract tags for generated images
            write_status(session_folder, f"Extracting visual tags for {concept_name}...")
            
            for img_path in files:
                filename = os.path.basename(img_path)
                write_status(session_folder, f"Analyzing: {filename}")
                
                try:
                    # Use general prompt for final generation modes
                    tags = extract_visual_elements_from_image(img_path, general_prompt)
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
        tags_path = os.path.join(mode_folder, "visual_tags.json")
        with open(tags_path, "w") as tagf:
            json.dump(all_visual_tags, tagf, indent=2)
        total_tags = sum(len(tags) for tags in all_visual_tags.values())
        write_status(session_folder, f"Saved {total_tags} total tags to: visual_tags.json")
    
    write_status(session_folder, f"FINAL Mode 2 completed successfully!")
    return results


def run_final_mode3(narrative_prompt: str, image_prompt: str, descriptor: str, user_pref: dict, tag_data: dict, reference_images: list, mode_folder: str, session_folder: str):
    """
    Run final mode 3: JSON + Tags + Images with proper multi-scene workflow
    """
    from tag_extraction import extract_visual_elements_from_image
    
    write_status(session_folder, f"Starting FINAL Mode 3 - JSON + Tags + Images")
    
    # Generate multiple scene descriptions using designer
    write_status(session_folder, f"Generating concepts with tags and reference images...")
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
    
    # Generate images for each scene
    results = []
    all_visual_tags = {}
    
    for i, scene in enumerate(scenes):
        concept_name = scene.get('concept_name', f'Scene {i+1}')
        write_status(session_folder, f"Generating images for: {concept_name}")
        
        prefix = f"final_{i}"
        
        try:
            # Use mode 3 generator with tags and reference images
            files = generator_final_mode3(
                narrative_prompt, image_prompt, descriptor, scene, user_pref, tag_data, reference_images,
                mode_folder, prefix, session_folder, "final"
            )
            results.append((scene, files))
            write_status(session_folder, f"Generated {len(files)} images for {concept_name}")
            
            # Extract tags for generated images
            write_status(session_folder, f"Extracting visual tags for {concept_name}...")
            
            for img_path in files:
                filename = os.path.basename(img_path)
                write_status(session_folder, f"Analyzing: {filename}")
                
                try:
                    # Use general prompt for final generation modes
                    tags = extract_visual_elements_from_image(img_path, general_prompt)
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
        tags_path = os.path.join(mode_folder, "visual_tags.json")
        with open(tags_path, "w") as tagf:
            json.dump(all_visual_tags, tagf, indent=2)
        total_tags = sum(len(tags) for tags in all_visual_tags.values())
        write_status(session_folder, f"Saved {total_tags} total tags to: visual_tags.json")
    
    write_status(session_folder, f"FINAL Mode 3 completed successfully!")
    return results


# Optimized final mode functions with parallel generation and immediate tag extraction
def run_final_mode1_optimized(narrative_prompt: str, image_prompt: str, descriptor: str, user_pref: dict, mode_folder: str, session_folder: str):
    """Run optimized final mode 1 generation (JSON only) using parallel generation and immediate tag extraction"""
    from tag_extraction import extract_visual_elements_from_image
    
    write_status(session_folder, f"Starting OPTIMIZED FINAL Mode 1 - JSON only")
    
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
    
    # Generate images and extract tags in parallel with immediate tag extraction
    write_status(session_folder, f"Generating images and extracting tags in parallel for all {len(scenes)} scenes...")
    prefix_base = "final"
    
    try:
        results, all_visual_tags = generator_final_mode1_optimized(
            image_prompt, descriptor, scenes, user_pref, mode_folder, prefix_base, session_folder, "final"
        )
        write_status(session_folder, f"Successfully generated images and extracted tags for {len(results)} scenes in parallel")
    except Exception as e:
        error_msg = f"Optimized final mode 1 generation failed: {str(e)}"
        write_status(session_folder, error_msg)
        raise RuntimeError(error_msg)
    
    # Save visual tags
    if all_visual_tags:
        tags_path = os.path.join(mode_folder, "visual_tags.json")
        with open(tags_path, "w") as tagf:
            json.dump(all_visual_tags, tagf, indent=2)
        total_tags = sum(len(tags) for tags in all_visual_tags.values())
        write_status(session_folder, f"Saved {total_tags} total tags to: visual_tags.json")
    
    write_status(session_folder, f"OPTIMIZED FINAL Mode 1 completed successfully!")
    return results

def run_final_mode2_optimized(narrative_prompt: str, image_prompt: str, descriptor: str, user_pref: dict, tag_data: dict, mode_folder: str, session_folder: str):
    """Run optimized final mode 2 generation (JSON + Tags) using parallel generation and immediate tag extraction"""
    write_status(session_folder, f"Starting OPTIMIZED FINAL Mode 2 - JSON + Tags")
    
    # Generate multiple scene descriptions using designer
    write_status(session_folder, f"Generating concepts with tags...")
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
    
    # Generate images and extract tags in parallel with immediate tag extraction
    write_status(session_folder, f"Generating images and extracting tags in parallel for all {len(scenes)} scenes...")
    prefix_base = "final"
    
    try:
        results, all_visual_tags = generator_final_mode2_optimized(
            narrative_prompt, image_prompt, descriptor, scenes, user_pref, tag_data, mode_folder, prefix_base, session_folder, "final"
        )
        write_status(session_folder, f"Successfully generated images and extracted tags for {len(results)} scenes in parallel")
    except Exception as e:
        error_msg = f"Optimized final mode 2 generation failed: {str(e)}"
        write_status(session_folder, error_msg)
        raise RuntimeError(error_msg)
    
    # Save visual tags
    if all_visual_tags:
        tags_path = os.path.join(mode_folder, "visual_tags.json")
        with open(tags_path, "w") as tagf:
            json.dump(all_visual_tags, tagf, indent=2)
        total_tags = sum(len(tags) for tags in all_visual_tags.values())
        write_status(session_folder, f"Saved {total_tags} total tags to: visual_tags.json")
    
    write_status(session_folder, f"OPTIMIZED FINAL Mode 2 completed successfully!")
    return results

def run_final_mode3_optimized(narrative_prompt: str, image_prompt: str, descriptor: str, user_pref: dict, tag_data: dict, reference_images: list, mode_folder: str, session_folder: str):
    """Run optimized final mode 3 generation (JSON + Tags + Images) using parallel generation and immediate tag extraction"""
    write_status(session_folder, f"Starting OPTIMIZED FINAL Mode 3 - JSON + Tags + Images")
    
    # Generate multiple scene descriptions using designer
    write_status(session_folder, f"Generating concepts with tags and reference images...")
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
    
    # Generate images and extract tags in parallel with immediate tag extraction
    write_status(session_folder, f"Generating images and extracting tags in parallel for all {len(scenes)} scenes...")
    prefix_base = "final"
    
    try:
        results, all_visual_tags = generator_final_mode3_optimized(
            narrative_prompt, image_prompt, descriptor, scenes, user_pref, tag_data, reference_images, mode_folder, prefix_base, session_folder, "final"
        )
        write_status(session_folder, f"Successfully generated images and extracted tags for {len(results)} scenes in parallel")
    except Exception as e:
        error_msg = f"Optimized final mode 3 generation failed: {str(e)}"
        write_status(session_folder, error_msg)
        raise RuntimeError(error_msg)
    
    # Save visual tags
    if all_visual_tags:
        tags_path = os.path.join(mode_folder, "visual_tags.json")
        with open(tags_path, "w") as tagf:
            json.dump(all_visual_tags, tagf, indent=2)
        total_tags = sum(len(tags) for tags in all_visual_tags.values())
        write_status(session_folder, f"Saved {total_tags} total tags to: visual_tags.json")
    
    write_status(session_folder, f"OPTIMIZED FINAL Mode 3 completed successfully!")
    return results

def run_final_mode4_optimized(narrative_prompt: str, image_prompt: str, descriptor: str, enhanced_user_pref: dict, mode_folder: str, session_folder: str):
    """Run optimized final mode 4 generation (Enhanced User Preferences) using parallel generation and immediate tag extraction"""
    write_status(session_folder, f"Starting OPTIMIZED FINAL Mode 4 - Enhanced User Preferences")
    
    # Generate multiple scene descriptions using designer
    write_status(session_folder, f"Generating concepts with enhanced user preferences...")
    scenes = designer_seq(narrative_prompt, descriptor, enhanced_user_pref, session_folder, "final")
    
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
    
    # Generate images and extract tags in parallel with immediate tag extraction
    write_status(session_folder, f"Generating images and extracting tags in parallel for all {len(scenes)} scenes...")
    prefix_base = "final"
    
    try:
        results, all_visual_tags = generator_final_mode4_optimized(
            image_prompt, descriptor, scenes, enhanced_user_pref, mode_folder, prefix_base, session_folder, "final"
        )
        write_status(session_folder, f"Successfully generated images and extracted tags for {len(results)} scenes in parallel")
    except Exception as e:
        error_msg = f"Optimized final mode 4 generation failed: {str(e)}"
        write_status(session_folder, error_msg)
        raise RuntimeError(error_msg)
    
    # Save visual tags
    if all_visual_tags:
        tags_path = os.path.join(mode_folder, "visual_tags.json")
        with open(tags_path, "w") as tagf:
            json.dump(all_visual_tags, tagf, indent=2)
        total_tags = sum(len(tags) for tags in all_visual_tags.values())
        write_status(session_folder, f"Saved {total_tags} total tags to: visual_tags.json")
    
    write_status(session_folder, f"OPTIMIZED FINAL Mode 4 completed successfully!")
    return results

def run_final_mode1(narrative_prompt: str, image_prompt: str, descriptor: str, user_pref: dict, mode_folder: str, session_folder: str):
    """Run final mode 1 generation (JSON only) using the full Designer -> Generator workflow"""
    
    # Designer stage
    scenes = designer_seq(narrative_prompt, descriptor, user_pref, session_folder, "final")
    
    if not scenes:
        write_status(session_folder, "❌ No scenes generated by designer")
        return []
    
    write_status(session_folder, f"✅ Designer generated {len(scenes)} scenes")
    
    # Save designer output
    scenes_file = os.path.join(mode_folder, f"final.json")
    with open(scenes_file, "w") as f:
        json.dump(scenes, f, indent=2)
    write_status(session_folder, f"💾 Saved designer output to final.json")
    
    # Generator stage - generate images for each scene
    results = []
    all_visual_tags = {}
    
    for i, scene in enumerate(scenes):
        write_status(session_folder, f"🎨 Generating images for scene {i+1}/{len(scenes)}: {scene.get('concept_name', 'Unknown')}")
        
        # Generate images using final mode 1 generator
        files = generator_final_mode1(
            image_prompt, descriptor, scene, user_pref, 
            mode_folder, f"final_{i}", session_folder
        )
        
        if files:
            results.append((scene, files))
            write_status(session_folder, f"✅ Generated {len(files)} images for scene {i+1}")
            
            # Extract visual tags from generated images
            for file_path in files:
                if os.path.exists(file_path):
                    # Use general prompt for final generation modes
                    visual_tags = extract_visual_elements_from_image(file_path, general_prompt)
                    image_key = os.path.basename(file_path)
                    all_visual_tags[image_key] = visual_tags
                    write_status(session_folder, f"🏷️ Extracted tags for {os.path.basename(file_path)}")
        else:
            write_status(session_folder, f"❌ Failed to generate images for scene {i+1}")
    
    # Save visual tags
    tags_file = os.path.join(mode_folder, "visual_tags.json")
    with open(tags_file, "w") as f:
        json.dump(all_visual_tags, f, indent=2)
    write_status(session_folder, f"💾 Saved visual tags to visual_tags.json")
    
    write_status(session_folder, f"🎉 Mode 1 generation completed! Generated {len(results)} scenes with images and tags")
    return results


def run_final_mode4(narrative_prompt: str, image_prompt: str, descriptor: str, enhanced_user_pref: dict, mode_folder: str, session_folder: str):
    """
    Run final mode 4: Enhanced User Preferences with Per-Layer Tags
    Uses the comprehensive user preference structure from mode4.py
    """
    from tag_extraction import extract_visual_elements_from_image
    
    write_status(session_folder, f"Starting FINAL Mode 4 - Enhanced User Preferences with Per-Layer Tags")
    
    # Generate multiple scene descriptions using designer
    write_status(session_folder, f"Generating concepts with enhanced user preferences...")
    scenes = designer_seq(narrative_prompt, descriptor, enhanced_user_pref, session_folder, "final")
    
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
    all_visual_tags = {}
    
    for i, scene in enumerate(scenes):
        concept_name = scene.get('concept_name', f'Scene {i+1}')
        write_status(session_folder, f"Generating images for: {concept_name}")
        
        prefix = f"final_{i}"
        
        try:
            # Use mode 4 generator with enhanced user preferences
            files = generator_final_mode4(
                narrative_prompt, image_prompt, descriptor, scene, enhanced_user_pref,
                mode_folder, prefix, session_folder, "final"
            )
            results.append((scene, files))
            write_status(session_folder, f"Generated {len(files)} images for {concept_name}")
            
            # Extract tags for generated images
            write_status(session_folder, f"Extracting visual tags for {concept_name}...")
            
            for img_path in files:
                filename = os.path.basename(img_path)
                write_status(session_folder, f"Analyzing: {filename}")
                
                try:
                    # Use general prompt for final generation modes
                    tags = extract_visual_elements_from_image(img_path, general_prompt)
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
        tags_path = os.path.join(mode_folder, "visual_tags.json")
        with open(tags_path, "w") as tagf:
            json.dump(all_visual_tags, tagf, indent=2)
        total_tags = sum(len(tags) for tags in all_visual_tags.values())
        write_status(session_folder, f"Saved {total_tags} total tags to: visual_tags.json")
    
    write_status(session_folder, f"FINAL Mode 4 completed successfully!")
    return results


@app.post("/api/upload-session")
async def upload_session(request: Request):
    """Handle session folder upload and validation."""
    try:
        form = await request.form()
        files = form.getlist("files")
        folder_name = form.get("folderName")
        
        if not files or not folder_name:
            raise HTTPException(400, "No files or folder name provided")
        
        # Create session folder in backend sessions directory
        session_folder = os.path.join("sessions", folder_name)
        os.makedirs(session_folder, exist_ok=True)
        
        # Process uploaded files
        file_map = {}
        for file in files:
            # Get the relative path from the uploaded file
            file_path = file.filename
            if not file_path:
                continue
                
            # Remove the root folder name from path if present
            path_parts = file_path.split('/')
            if len(path_parts) > 1 and path_parts[0] == folder_name:
                relative_path = '/'.join(path_parts[1:])
            else:
                relative_path = file_path
            
            # Create the full path in session folder
            full_path = os.path.join(session_folder, relative_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Save the file
            with open(full_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            
            file_map[relative_path] = full_path
        
        # Validate required files
        required_files = [
            'impression/impression.json',
            'spatial/spatial.json', 
            'ambient/ambient.json',
            'preferences.json'
        ]
        
        missing_files = []
        for required_file in required_files:
            if required_file not in file_map:
                missing_files.append(required_file)
        
        if missing_files:
            # Clean up the created folder
            import shutil
            shutil.rmtree(session_folder, ignore_errors=True)
            raise HTTPException(400, f"Missing required files: {', '.join(missing_files)}")
        
        # Load preferences.json
        preferences_path = os.path.join(session_folder, "preferences.json")
        with open(preferences_path, 'r') as f:
            preferences = json.load(f)
        
        # Determine session type from folder name or preferences
        session_type = "parallel" if "[para]" in folder_name else "sequential"
        
        # Store session info
        session_id = folder_name
        sessions[session_id] = {
            'folder': session_folder,
            'descriptor': f"Uploaded session - {folder_name}",
            'user_pref': {},
            'uploaded': True,
            'session_type': session_type
        }
        
        return {
            "session_id": session_id,
            "preferences": preferences,
            "message": f"Session folder '{folder_name}' uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to upload session: {str(e)}"
        print(f"Upload error: {error_msg}")
        raise HTTPException(500, error_msg)


# --- Status endpoint ---
class StatusResponse(BaseModel):
    status: str
    messages: list[str]

@app.get("/api/status/{session_id}")
def get_status(session_id: str):
    # URL decode the session_id
    import urllib.parse
    decoded_session_id = urllib.parse.unquote(session_id)
    
    print(f"DEBUG: Raw session_id: {session_id}")
    print(f"DEBUG: Decoded session_id: {decoded_session_id}")
    print(f"DEBUG: Available sessions: {list(sessions.keys())}")
    
    # Try both encoded and decoded versions
    session = sessions.get(session_id) or sessions.get(decoded_session_id)
    
    if not session:
        # If session not in memory, try to find it on disk
        sessions_dir = "./sessions"
        if os.path.exists(os.path.join(sessions_dir, decoded_session_id)):
            # Session exists on disk but not in memory - register it
            session_folder = os.path.join(sessions_dir, decoded_session_id)
            print(f"DEBUG: Found session on disk, registering: {decoded_session_id}")
            
            # Determine session type
            session_type = "parallel" if "[para]" in decoded_session_id else "sequential"
            
            # Register the session
            sessions[decoded_session_id] = {
                'folder': session_folder,
                'descriptor': f"Recovered session - {decoded_session_id}",
                'user_pref': {},
                'uploaded': True,
                'session_type': session_type
            }
            session = sessions[decoded_session_id]
        else:
            print(f"DEBUG: Session not found on disk either: {decoded_session_id}")
            raise HTTPException(404, f"Session not found: {decoded_session_id}")
    
    # Get the latest status messages from the session
    messages = []
    status_file = os.path.join(session['folder'], "status.txt")
    
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            messages = [line.strip() for line in f.readlines()]
    
    return {
        "status": "complete" if not messages else "in_progress",
        "messages": messages
    }

# --- JSON endpoint ---
class JsonRequest(BaseModel):
    session_id: str
    stage: str
    image_id: str

class JsonResponse(BaseModel):
    json_data: dict

@app.post("/api/json", response_model=JsonResponse)
def get_json(req: JsonRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    try:
        # Handle different final mode folders
        if req.stage == 'final':
            # Check which final mode folder exists
            base_folder = session['folder']
            possible_folders = ['final', '[with Tags]final', '[with Imgs]final', '[Enhanced Prefs]final']
            stage_folder = None
            
            for folder_name in possible_folders:
                test_folder = os.path.join(base_folder, folder_name)
                if os.path.exists(test_folder):
                    stage_folder = test_folder
                    break
            
            if stage_folder is None:
                print(f"No final mode folder found in {base_folder}")
                return {"json_data": {}}
                
            # For final stages, the JSON file is always named "final.json"
            stage_json_path = os.path.join(stage_folder, "final.json")
        else:
            # Regular stage folder
            stage_folder = os.path.join(session['folder'], req.stage)
            stage_json_path = os.path.join(stage_folder, f"{req.stage}.json")
        
        print(f"Looking for stage JSON at: {stage_json_path}")
        print(f"Requested image_id: {req.image_id}")
        
        if not os.path.exists(stage_json_path):
            print(f"Stage JSON file not found at {stage_json_path}")
            return {"json_data": {}}
        
        with open(stage_json_path, 'r') as f:
            stage_data = json.load(f)
        
        # Extract the index from the image ID (e.g., "impression_0_0" -> index 0)
        # The image_id should match the filename pattern
        if '_' in req.image_id:
            # For image_id like "impression_0_0", get the first index (0)
            parts = req.image_id.split('_')
            if len(parts) >= 2:
                try:
                    image_index = int(parts[1])  # Gets "0" from "impression_0_0"
                except ValueError:
                    image_index = 0
            else:
                image_index = 0
        else:
            image_index = 0
        
        print(f"Extracted image index: {image_index}")
        
        # Get the JSON data for the specific image
        if isinstance(stage_data, list):
            if image_index < len(stage_data):
                json_data = stage_data[image_index]
            else:
                json_data = {}
        elif isinstance(stage_data, dict) and 'outputs' in stage_data:
            outputs = stage_data['outputs']
            if image_index < len(outputs):
                json_data = outputs[image_index]
            else:
                json_data = {}
        else:
            json_data = stage_data
        
        print(f"Returning JSON data for image {req.image_id}")
        return {"json_data": json_data}
        
    except Exception as e:
        print(f"Error loading JSON: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {"json_data": {}}


# ============================================================================
# TEST STAGE REFINEMENT - New Feature
# ============================================================================

class SessionListResponse(BaseModel):
    sessions: list[dict]

class LoadSessionRequest(BaseModel):
    session_path: str

class LoadSessionResponse(BaseModel):
    session_id: str
    session_name: str
    stages: list[dict]
    preferences: dict

class LoadStageDataRequest(BaseModel):
    session_path: str
    stage: str

class LoadStageDataResponse(BaseModel):
    images: list[ImageItem]
    tags: dict
    stage_json: list  # List of concept dictionaries, one per image
    preferences: dict


@app.get("/api/list-sessions", response_model=SessionListResponse)
def list_sessions():
    """List all available session folders"""
    try:
        sessions_list = []
        
        if not os.path.exists(SESSIONS_DIR):
            return {"sessions": []}
        
        for folder_name in sorted(os.listdir(SESSIONS_DIR), reverse=True):
            folder_path = os.path.join(SESSIONS_DIR, folder_name)
            
            # Skip files and debug_logs
            if not os.path.isdir(folder_path) or folder_name == "debug_logs" or folder_name == "test_refinement":
                continue
            
            # Get folder metadata
            stat_info = os.stat(folder_path)
            timestamp = datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            # Count completed stages
            stages_completed = []
            for stage in ["impression", "spatial", "objects", "ambient", "final"]:
                stage_folder = os.path.join(folder_path, stage)
                if os.path.exists(stage_folder):
                    stages_completed.append(stage)
            
            sessions_list.append({
                "name": folder_name,
                "path": folder_name,  # Relative to SESSIONS_DIR
                "timestamp": timestamp,
                "stages_completed": stages_completed,
                "stage_count": len(stages_completed)
            })
        
        return {"sessions": sessions_list}
        
    except Exception as e:
        print(f"Error listing sessions: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(500, str(e))


@app.post("/api/load-session", response_model=LoadSessionResponse)
def load_session(req: LoadSessionRequest):
    """Load session metadata and available stages"""
    try:
        session_path = os.path.join(SESSIONS_DIR, req.session_path)
        
        if not os.path.exists(session_path):
            raise HTTPException(404, f"Session not found: {req.session_path}")
        
        # Load preferences if exists
        preferences = {}
        prefs_file = os.path.join(session_path, "preferences.json")
        if os.path.exists(prefs_file):
            with open(prefs_file, "r") as f:
                preferences = json.load(f)
        
        # Scan for available stages
        stages_info = []
        for stage in ["impression", "spatial", "objects", "ambient"]:
            stage_folder = os.path.join(session_path, stage)
            
            if not os.path.exists(stage_folder):
                continue
            
            # Count images
            image_files = [f for f in os.listdir(stage_folder) if f.endswith('.png')]
            
            # Check for tags
            tags_file = os.path.join(stage_folder, "visual_tags.json")
            has_tags = os.path.exists(tags_file)
            
            # Check for JSON
            json_file = os.path.join(stage_folder, f"{stage}.json")
            has_json = os.path.exists(json_file)
            
            # Check if refinement exists
            refinement_folder = os.path.join(session_path, f"{stage}_refinement")
            has_refinement = os.path.exists(refinement_folder)
            
            stages_info.append({
                "stage_name": stage,
                "has_images": len(image_files) > 0,
                "image_count": len(image_files),
                "has_tags": has_tags,
                "has_json": has_json,
                "has_refinement": has_refinement,
                "can_refine": has_tags and has_json and len(image_files) > 0
            })
        
        return {
            "session_id": req.session_path,  # Use folder name as session ID
            "session_name": req.session_path,
            "stages": stages_info,
            "preferences": preferences
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error loading session: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(500, str(e))


@app.post("/api/load-stage-data", response_model=LoadStageDataResponse)
def load_stage_data(req: LoadStageDataRequest):
    """Load images, tags, and JSON for a specific stage"""
    try:
        session_path = os.path.join(SESSIONS_DIR, req.session_path)
        stage_folder = os.path.join(session_path, req.stage)
        
        if not os.path.exists(stage_folder):
            raise HTTPException(404, f"Stage folder not found: {req.stage}")
        
        # Load images
        images = []
        image_files = sorted([f for f in os.listdir(stage_folder) if f.endswith('.png')])
        
        for img_file in image_files:
            # Extract concept index from filename: {stage}_{concept_idx}_{variant_idx}.png
            parts = img_file.replace('.png', '').split('_')
            if len(parts) >= 3:
                concept_idx = parts[-2]
                variant_idx = parts[-1]
                image_id = f"{req.stage}_{concept_idx}_{variant_idx}"
                
                images.append({
                    "id": image_id,
                    "url": f"/sessions/{req.session_path}/{req.stage}/{img_file}"
                })
        
        # Load tags
        tags = {}
        tags_file = os.path.join(stage_folder, "visual_tags.json")
        if os.path.exists(tags_file):
            with open(tags_file, "r") as f:
                tags = json.load(f)
        
        # Load stage JSON
        stage_json = {}
        json_file = os.path.join(stage_folder, f"{req.stage}.json")
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                stage_json = json.load(f)
        
        # Load preferences
        preferences = {}
        prefs_file = os.path.join(session_path, "preferences.json")
        if os.path.exists(prefs_file):
            with open(prefs_file, "r") as f:
                preferences = json.load(f)
        
        # Create session entry for concept system endpoints
        # This allows /api/concepts/init and /api/tags to work
        sessions[req.session_path] = {
            'folder': session_path,
            'descriptor': preferences.get('descriptor', ''),
            'mode': 'test-stage-refinement'
        }
        
        return {
            "images": images,
            "tags": tags,
            "stage_json": stage_json,
            "preferences": preferences
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error loading stage data: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(500, str(e))


class GenerateStageRefinementRequest(BaseModel):
    session_path: str
    stage: str
    selected_concept_index: int
    positive_concept_labels: list[str]
    negative_concept_labels: list[str]
    descriptor: str

class GenerateStageRefinementResponse(BaseModel):
    success: bool
    images: list[ImageItem]
    stage_json: list[dict]
    refinement_folder: str


@app.post("/api/generate-stage-refinement", response_model=GenerateStageRefinementResponse)
def generate_stage_refinement(req: GenerateStageRefinementRequest):
    """
    Generate 4 refinement images using PBO + SDXL.
    
    NEW IMPLEMENTATION:
    - Uses tag cluster concepts from visual_tags.json
    - PBO proposes 4 weight mixtures
    - SDXL generates images from fused embeddings
    """
    try:
        session_path = os.path.join(SESSIONS_DIR, req.session_path)
        session_id = req.session_path
        
        if not os.path.exists(session_path):
            raise HTTPException(404, f"Session not found: {req.session_path}")
        
        refinement_stage = f"{req.stage}_refinement"
        
        write_status(session_path, f"🔄 Starting PBO refinement for {req.stage}...")
        
        # Step 1: Get image IDs from the base stage
        stage_folder = os.path.join(session_path, req.stage)
        json_file = os.path.join(stage_folder, f"{req.stage}.json")
        
        if not os.path.exists(json_file):
            raise HTTPException(404, f"Stage JSON not found: {req.stage}")
        
        with open(json_file, "r") as f:
            stage_concepts = json.load(f)
        
        # Get image IDs (e.g., impression_0, impression_1, impression_2, impression_3)
        image_ids = [f"{req.stage}_{i}" for i in range(len(stage_concepts))]
        
        write_status(session_path, f"📊 Loading {len(image_ids)} images from {req.stage} stage")
        
        # Step 2: Initialize PBO with tag cluster concepts
        write_status(session_path, "🧠 Initializing PBO with tag cluster concepts...")
        
        visual_tags_path = os.path.join(stage_folder, "visual_tags.json")
        if not os.path.exists(visual_tags_path):
            raise HTTPException(
                404,
                f"Visual tags not found. The {req.stage} stage needs visual tags for refinement."
            )
        
        with open(visual_tags_path, 'r') as f:
            visual_tags_data = json.load(f)
        
        # Build image_tags dict
        image_tags = {}
        for image_id in image_ids:
            image_filename = f"{image_id}_0.png"
            if image_filename in visual_tags_data:
                image_tags[image_id] = visual_tags_data[image_filename]
            else:
                write_status(session_path, f"⚠️ No tags found for {image_id}")
                image_tags[image_id] = []
        
        total_tags = sum(len(tags) for tags in image_tags.values())
        write_status(session_path, f"📋 Loaded {total_tags} visual tags")
        
        # Initialize ConceptRefinementSession (clusters tags)
        refinement_session = get_refinement_session(
            session_id,
            req.stage,
            image_ids
        )
        
        if not refinement_session.initialized:
            write_status(session_path, "🔨 Clustering tags into concepts using K-means...")
            refinement_session.initialize_from_tags(image_tags)
            write_status(session_path, f"✅ Created {len(refinement_session.concepts)} tag cluster concepts")
        else:
            write_status(session_path, f"✅ Using existing {len(refinement_session.concepts)} concepts")
        
        # Initialize StageRefiner (creates PBO with MU matrix)
        refiner = get_or_create_pbo_refiner(
            session_id=session_id,
            stage=req.stage
        )
        
        concept_labels = [c['label'] for c in refiner.concepts]
        write_status(session_path, f"🎯 PBO initialized with concepts: {', '.join(concept_labels[:5])}...")
        
        # Step 3: Use PBO to propose 4 weight mixtures
        write_status(session_path, "🎲 Generating 4 optimized weight mixtures...")
        
        proposals = refiner.propose_next_4(
            negatives=None,
            w_current=None,
            fit_first=True
        )
        
        write_status(session_path, f"✅ Generated {len(proposals)} proposals")
        
        # Step 4: Generate images using SDXL
        write_status(session_path, "🎨 Generating images with SDXL...")
        
        # Load the selected image as reference
        selected_image_path = os.path.join(stage_folder, f"{req.stage}_{req.selected_concept_index}_0.png")
        reference_image = None
        
        if os.path.exists(selected_image_path):
            from PIL import Image as PILImage
            reference_image = PILImage.open(selected_image_path)
            write_status(session_path, f"📷 Using reference image: {os.path.basename(selected_image_path)}")
        else:
            write_status(session_path, "⚠️ Selected image not found, using txt2img mode")
        
        # Load descriptor from preferences.json
        descriptor = None
        preferences_file = os.path.join(session_path, "preferences.json")
        if os.path.exists(preferences_file):
            with open(preferences_file, 'r') as f:
                prefs = json.load(f)
                descriptor = prefs.get('descriptor')
        
        # Initialize tracking
        from backend.tracking import create_tracker
        tracker = create_tracker(
            session_path=Path(session_path),
            session_id=session_id,
            stage=req.stage,
            descriptor=descriptor or "No descriptor"
        )
        tracker.set_concepts(refiner.concepts)
        tracker.start_round(
            round_number=1,
            reference_image=os.path.basename(selected_image_path) if reference_image else None
        )
        
        sdxl_runner = get_sdxl_runner()
        
        # Prepare image paths for tracking
        image_paths_for_tracking = [
            f"{refinement_stage}/round_1/image_{i}.png" for i in range(len(proposals))
        ]
        
        # Generate images from proposals
        pil_images = refiner.generate_images_from_proposals(
            proposals=proposals,
            sdxl_runner=sdxl_runner,
            seed_base=42,
            verbose=False,
            init_image=reference_image,
            descriptor=descriptor,  # User description from preferences
            tracker=tracker,  # Track all generation details
            generated_image_paths=image_paths_for_tracking  # Image paths for tracking
        )
        
        # Create refinement folder structure (Round 1)
        refinement_folder_path = os.path.join(session_path, refinement_stage)
        round_1_folder = os.path.join(refinement_folder_path, "round_1")
        os.makedirs(round_1_folder, exist_ok=True)
        
        # Save images to Round 1 folder
        images = []
        for idx, pil_img in enumerate(pil_images):
            # Save to round_1 folder
            image_filename = f"image_{idx}.png"
            image_path = os.path.join(round_1_folder, image_filename)
            pil_img.save(image_path)
            
            # Also save with legacy naming for compatibility
            legacy_filename = f"{refinement_stage}_{idx}_0.png"
            legacy_path = os.path.join(refinement_folder_path, legacy_filename)
            pil_img.save(legacy_path)
            
            write_status(session_path, f"💾 Saved image {idx + 1}/4: {image_filename}")
            
            # Use legacy format for now
            image_id = f"{refinement_stage}_{idx}_0"
            image_url = f"/sessions/{req.session_path}/{refinement_stage}/{legacy_filename}"
            
            images.append({
                "id": image_id,
                "url": image_url
            })
        
        # Save Round 1 weight vectors
        reference_image_id = f"{req.stage}_{req.selected_concept_index}_0"
        weights_data = {
            "round": 1,
            "proposals": [p.tolist() for p in proposals],
            "concept_labels": concept_labels,
            "reference_image": reference_image_id,
            "selected_concept_index": req.selected_concept_index,
            "positive_labels": req.positive_concept_labels,
            "negative_labels": req.negative_concept_labels
        }
        
        weights_file = os.path.join(round_1_folder, "weights.json")
        with open(weights_file, "w") as f:
            json.dump(weights_data, f, indent=2)
        
        # Update preferences
        prefs_file = os.path.join(session_path, "preferences.json")
        preferences = {}
        if os.path.exists(prefs_file):
            with open(prefs_file, "r") as f:
                preferences = json.load(f)
        
        if req.stage not in preferences:
            preferences[req.stage] = {}
        
        preferences[req.stage]["selected_concept_index"] = req.selected_concept_index
        preferences[req.stage]["positive_labels"] = req.positive_concept_labels
        preferences[req.stage]["negative_labels"] = req.negative_concept_labels
        preferences[req.stage]["refinement_method"] = "pbo_sdxl"
        
        with open(prefs_file, "w") as f:
            json.dump(preferences, f, indent=2)
        
        write_status(session_path, f"✅ PBO refinement complete! Generated 4 images using SDXL")
        
        # Create dummy stage_json for compatibility (frontend expects this)
        stage_json = [
            {"concept_name": f"PBO Mixture {i+1}", "weight_vector": proposals[i].tolist()}
            for i in range(len(proposals))
        ]
        
        return {
            "success": True,
            "images": images,
            "stage_json": stage_json,
            "refinement_folder": refinement_stage
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error in PBO refinement: {str(e)}"
        print(error_msg)
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        # Try to write error to status
        try:
            if 'session_path' in locals():
                write_status(session_path, f"❌ Error: {error_msg}")
        except:
            pass

        raise HTTPException(status_code=500, detail=error_msg)


# ============================================================================
# PBO (Preference-Based Optimization) Endpoints - Stage 4
# ============================================================================

from stage_refiner import StageRefiner
from sdxl_runner import SDXLRunner
import numpy as np
from pathlib import Path
from typing import Optional, Dict as TypingDict

# Global singletons
_sdxl_runner: Optional[SDXLRunner] = None
_pbo_refiners: TypingDict[str, StageRefiner] = {}


def get_sdxl_runner() -> SDXLRunner:
    """Get or create global SDXL runner singleton."""
    global _sdxl_runner
    if _sdxl_runner is None:
        print("[PBO] Initializing SDXL Runner...")
        _sdxl_runner = SDXLRunner(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            device=None,  # Auto-detect (cuda/mps/cpu)
            height=1024,
            width=1024,
            steps=30,
            guidance_scale=7.5
        )
        print("[PBO] SDXL Runner initialized")
    return _sdxl_runner


def get_or_create_pbo_refiner(session_id: str, stage: str) -> StageRefiner:
    """
    Get or create StageRefiner for PBO session/stage.

    This integrates with existing ConceptRefinementSession to reuse concepts.
    """
    key = f"{session_id}:{stage}"

    if key not in _pbo_refiners:
        print(f"[PBO] Creating new StageRefiner for {session_id}/{stage}")

        # Get existing concept refinement session (which has the concepts)
        concept_session = get_refinement_session(session_id, stage, [])

        if not concept_session.initialized:
            raise HTTPException(
                status_code=400,
                detail=f"Concept session for {session_id}/{stage} not initialized. "
                       f"Run concept refinement first."
            )

        # Convert ConceptRefinementSession data to StageRefiner format
        concepts = []
        for concept in concept_session.concepts:
            concepts.append({
                'id': concept.id,
                'label': concept.label,
                'centroid': concept.centroid.tolist() if hasattr(concept.centroid, 'tolist') else concept.centroid
            })

        # Convert concept_states
        concept_states = {}
        for cid, state in concept_session.concept_states.items():
            concept_states[cid] = {
                'active': True,  # All concepts are active by default
                'weight': state.w,
                'total_positive_feedback': state.like_count,
                'total_negative_feedback': state.dislike_count
            }

        # Get session directory
        session_dir = Path(SESSIONS_DIR) / session_id / stage
        session_dir.mkdir(parents=True, exist_ok=True)

        # Create StageRefiner
        _pbo_refiners[key] = StageRefiner(
            session_id=session_id,
            stage=stage,
            concepts=concepts,
            concept_states=concept_states,
            image_ids=concept_session.image_ids,
            incidence_matrix=concept_session.incidence_matrix,
            session_dir=session_dir
        )

        print(f"[PBO] StageRefiner created with {len(concepts)} concepts")

    return _pbo_refiners[key]


# --- Pydantic Models ---

class StabilizeRequest(BaseModel):
    session_id: str
    stage: str
    w_ui: list[float]  # Current UI weights


class StabilizeResponse(BaseModel):
    snapshot_recorded: bool
    candidate_id: Optional[str]
    message: str


class ProposeRequest(BaseModel):
    session_id: str
    stage: str
    negatives: Optional[list[str]] = None  # Concept IDs to avoid
    w_current: Optional[list[float]] = None  # Current UI weights for seeding


class ProposeResponse(BaseModel):
    proposals: list[list[float]]  # 4 weight vectors
    proposal_ids: list[str]
    message: str


class GenerateRequest_PBO(BaseModel):
    session_id: str
    stage: str
    proposals: list[list[float]]  # From /api/pbo/propose
    seed_base: int = 42


class GenerateResponse_PBO(BaseModel):
    image_paths: list[str]
    proposals: list[list[float]]
    round_number: int
    message: str


class FavoriteRequest(BaseModel):
    session_id: str
    stage: str
    favorite_image_id: str
    all_image_ids: list[str]


class FavoriteResponse(BaseModel):
    duels_added: int
    favorite_candidate_id: str
    message: str


class InitRefinementRequest(BaseModel):
    session_id: str
    stage: str
    image_ids: list[str]  # Image IDs from the base stage (e.g., impression)


class InitRefinementResponse(BaseModel):
    success: bool
    num_concepts: int
    concept_labels: list[str]
    message: str


# --- PBO Endpoints ---

@app.post("/api/pbo/init-refinement", response_model=InitRefinementResponse)
async def pbo_init_refinement(request: InitRefinementRequest):
    """
    Initialize PBO refinement for a stage using tag cluster concepts.
    
    Flow:
    1. Load visual tags from base stage
    2. Initialize ConceptRefinementSession (clusters tags into concepts)
    3. Initialize StageRefiner with concept centroids (MU matrix)
    4. Ready for PBO propose/generate/favorite cycle
    
    This bridges the tag clustering system with PBO optimization.
    """
    try:
        session = sessions.get(request.session_id)
        if not session:
            raise HTTPException(404, f"Session not found: {request.session_id}")
        
        print(f"[PBO Init] Initializing refinement for {request.session_id}/{request.stage}")
        
        # Step 1: Load visual tags
        stage_folder = os.path.join(session['folder'], request.stage)
        visual_tags_path = os.path.join(stage_folder, "visual_tags.json")
        
        if not os.path.exists(visual_tags_path):
            raise HTTPException(
                404, 
                f"Visual tags not found at {visual_tags_path}. "
                f"Run the base stage first to extract tags."
            )
        
        with open(visual_tags_path, 'r') as f:
            visual_tags_data = json.load(f)
        
        # Build image_tags dict
        image_tags = {}
        for image_id in request.image_ids:
            image_filename = f"{image_id}.png"
            if image_filename in visual_tags_data:
                image_tags[image_id] = visual_tags_data[image_filename]
            else:
                print(f"[PBO Init] Warning: No tags found for {image_id}")
                image_tags[image_id] = []
        
        total_tags = sum(len(tags) for tags in image_tags.values())
        print(f"[PBO Init] Loaded {total_tags} tags from {len(image_tags)} images")
        
        # Step 2: Initialize ConceptRefinementSession (clusters tags)
        refinement_session = get_refinement_session(
            request.session_id,
            request.stage,
            request.image_ids
        )
        
        if not refinement_session.initialized:
            print(f"[PBO Init] Clustering tags into concepts...")
            refinement_session.initialize_from_tags(image_tags)
            print(f"[PBO Init] Created {len(refinement_session.concepts)} concepts")
        else:
            print(f"[PBO Init] Using existing {len(refinement_session.concepts)} concepts")
        
        # Step 3: Initialize StageRefiner (creates PBO with MU matrix)
        refiner = get_or_create_pbo_refiner(
            session_id=request.session_id,
            stage=request.stage
        )
        
        concept_labels = [c['label'] for c in refiner.concepts]
        
        print(f"[PBO Init] ✅ Refinement initialized with {len(refiner.concepts)} concepts")
        print(f"[PBO Init] Concept labels: {', '.join(concept_labels[:5])}...")
        
        return InitRefinementResponse(
            success=True,
            num_concepts=len(refiner.concepts),
            concept_labels=concept_labels,
            message=f"Initialized PBO refinement with {len(refiner.concepts)} tag cluster concepts"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PBO Init] Error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pbo/stabilize", response_model=StabilizeResponse)
async def pbo_stabilize(request: StabilizeRequest):
    """
    Record stabilized UI weights as weak duel (debounced).

    Called when user's slider adjustments stabilize (after 500ms debounce).
    """
    try:
        refiner = get_or_create_pbo_refiner(
            session_id=request.session_id,
            stage=request.stage
        )

        w_ui = np.array(request.w_ui)
        recorded = refiner.on_ui_stabilize(w_ui)

        return StabilizeResponse(
            snapshot_recorded=recorded,
            candidate_id=refiner.last_snapshot_cid if recorded else None,
            message="Snapshot recorded" if recorded else "Debounce/threshold not met"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[PBO] Error in stabilize: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pbo/propose", response_model=ProposeResponse)
async def pbo_propose(request: ProposeRequest):
    """
    Generate 4 new concept mixtures using PBO.

    Called when user clicks "Generate Next 4 (PBO)" button.
    """
    try:
        refiner = get_or_create_pbo_refiner(
            session_id=request.session_id,
            stage=request.stage
        )

        negatives = set(request.negatives) if request.negatives else None
        w_current = np.array(request.w_current) if request.w_current else None

        proposals = refiner.propose_next_4(
            negatives=negatives,
            w_current=w_current,
            fit_first=True
        )

        return ProposeResponse(
            proposals=[w.tolist() for w in proposals],
            proposal_ids=[f"pbo_prop_{i}" for i in range(len(proposals))],
            message=f"Generated {len(proposals)} proposals"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[PBO] Error in propose: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pbo/generate", response_model=GenerateResponse_PBO)
async def pbo_generate(request: GenerateRequest_PBO):
    """
    Generate images from concept mixtures using SDXL.

    Called after /api/pbo/propose to actually generate the images.
    """
    try:
        refiner = get_or_create_pbo_refiner(
            session_id=request.session_id,
            stage=request.stage
        )

        # Load descriptor from preferences.json
        descriptor = None
        session_path = Path(SESSIONS_DIR) / request.session_id
        preferences_file = session_path / "preferences.json"
        if preferences_file.exists():
            with open(preferences_file, 'r') as f:
                prefs = json.load(f)
                descriptor = prefs.get('descriptor')
        
        # Get SDXL runner
        sdxl_runner = get_sdxl_runner()

        # Generate images
        proposals_np = [np.array(w) for w in request.proposals]
        images = refiner.generate_images_from_proposals(
            proposals=proposals_np,
            sdxl_runner=sdxl_runner,
            seed_base=request.seed_base,
            verbose=False,  # Less verbose for API
            descriptor=descriptor  # User description from preferences
        )

        # Save images
        session_dir = Path(SESSIONS_DIR) / request.session_id / request.stage
        session_dir.mkdir(parents=True, exist_ok=True)

        # Find next PBO round number
        pbo_rounds = list(session_dir.glob("pbo_round_*"))
        round_number = len(pbo_rounds)
        round_dir = session_dir / f"pbo_round_{round_number}"
        round_dir.mkdir(exist_ok=True)

        # Save images and build paths
        image_paths = []
        for i, img in enumerate(images):
            filename = f"image_{i}.png"
            file_path = round_dir / filename
            img.save(file_path)

            # Return relative path for frontend
            rel_path = f"/sessions/{request.session_id}/{request.stage}/pbo_round_{round_number}/{filename}"
            image_paths.append(rel_path)

        return GenerateResponse_PBO(
            image_paths=image_paths,
            proposals=request.proposals,
            round_number=round_number,
            message=f"Generated {len(images)} images in round {round_number}"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[PBO] Error in generate: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


class RefineNextRoundRequest(BaseModel):
    session_id: str
    stage: str  # base stage (e.g., "impression")
    selected_image_id: str  # Selected from current round
    all_image_ids: list[str]  # All images in current round
    round_number: int

class RefineNextRoundResponse(BaseModel):
    success: bool
    image_paths: list[str]
    round_number: int
    message: str


@app.post("/api/pbo/refine-next-round", response_model=RefineNextRoundResponse)
async def pbo_refine_next_round(request: RefineNextRoundRequest):
    """
    Complete refinement iteration: record selection + propose + generate.
    
    IMPORTANT: Always uses the ORIGINAL reference image from exploration stage,
    not the selected refinement image.
    """
    try:
        # Get refiner
        refiner = get_or_create_pbo_refiner(
            session_id=request.session_id,
            stage=request.stage
        )
        
        print(f"[PBO Refine] Round {request.round_number} → {request.round_number + 1}")
        print(f"[PBO Refine] Selected: {request.selected_image_id}")
        
        # Step 1: Record selection in tracking (CRITICAL: must come before starting new round)
        # Load tracker to record selection from previous round
        session = sessions.get(request.session_id)
        if not session:
            raise HTTPException(404, f"Session not found: {request.session_id}")
        
        session_folder = session['folder']
        
        # Get descriptor
        preferences_file = os.path.join(session_folder, "preferences.json")
        descriptor = None
        if os.path.exists(preferences_file):
            with open(preferences_file, 'r') as f:
                prefs = json.load(f)
                descriptor = prefs.get('descriptor')
        
        from backend.tracking import create_tracker
        tracker_for_selection = create_tracker(
            session_path=Path(session_folder),
            session_id=request.session_id,
            stage=request.stage,
            descriptor=descriptor or "No descriptor"
        )
        
        # Record the selection from the current round (before starting new round)
        # Find index of selected image in all_image_ids
        try:
            selected_index = request.all_image_ids.index(request.selected_image_id)
        except ValueError:
            selected_index = 0  # Fallback
        
        all_indices = list(range(len(request.all_image_ids)))
        tracker_for_selection.record_selection(selected_index, all_indices)
        print(f"[PBO Refine] ✅ Recorded selection in tracking: index {selected_index}")
        
        # Step 2: Load the actual weight vectors from the current round
        # (Don't rely on incidence_matrix - use the actual proposals!)
        refinement_stage = f"{request.stage}_refinement"
        refinement_folder = os.path.join(session_folder, refinement_stage)
        current_round_folder = os.path.join(refinement_folder, f"round_{request.round_number}")
        weights_file = os.path.join(current_round_folder, "weights.json")
        
        if not os.path.exists(weights_file):
            raise HTTPException(400, f"Weights file not found for round {request.round_number}")
        
        with open(weights_file, 'r') as f:
            weights_data = json.load(f)
        
        proposals_from_round = [np.array(w, dtype=np.float32) for w in weights_data['proposals']]
        print(f"[PBO Refine] Loaded {len(proposals_from_round)} weight vectors from round {request.round_number}")
        
        # Step 3: Record selection as PBO preference using ACTUAL weight vectors
        # Add each proposal as a candidate, then add duels
        candidate_ids = []
        for i, (img_id, w) in enumerate(zip(request.all_image_ids, proposals_from_round)):
            cand_id = refiner.pbo.add_candidate(w, candidate_id=f"round{request.round_number}_img{i}")
            candidate_ids.append(cand_id)
            refiner.image_to_candidate[img_id] = cand_id
        
        # Add strong duels: selected > others
        favorite_index = request.all_image_ids.index(request.selected_image_id)
        favorite_cand_id = candidate_ids[favorite_index]
        
        duels_added = 0
        for i, cand_id in enumerate(candidate_ids):
            if i != favorite_index:
                refiner.pbo.add_preference(favorite_cand_id, cand_id, strength=1.0)
                duels_added += 1
        
        print(f"[PBO Refine] Recorded {duels_added} duels in PBO (favorite: {favorite_cand_id})")
        
        # Step 4: Propose new weight mixtures
        proposals = refiner.propose_next_4(
            negatives=None,
            w_current=None,
            fit_first=True
        )
        print(f"[PBO Refine] Proposed {len(proposals)} new mixtures")
        
        # Step 5: Load ORIGINAL reference image (from exploration stage, not refinement)
        # (session_folder and descriptor already loaded above)
        
        # Get the originally selected image from exploration stage
        reference_image_id = None
        
        if os.path.exists(preferences_file):
            with open(preferences_file, 'r') as f:
                prefs = json.load(f)
                # Get the selection from base stage
                selections = prefs.get('selections', {})
                reference_image_id = selections.get(request.stage)  # e.g., "impression_2_0"
        
        if not reference_image_id:
            raise HTTPException(400, f"No reference image found for {request.stage} stage")
        
        print(f"[PBO Refine] Using ORIGINAL reference: {reference_image_id}")
        
        # Load reference image
        stage_folder = os.path.join(session_folder, request.stage)
        reference_image_path = os.path.join(stage_folder, f"{reference_image_id}.png")
        
        if not os.path.exists(reference_image_path):
            reference_image_path = os.path.join(stage_folder, f"{reference_image_id}_0.png")
        
        reference_image = None
        if os.path.exists(reference_image_path):
            from PIL import Image as PILImage
            reference_image = PILImage.open(reference_image_path)
            print(f"[PBO Refine] ✅ Loaded reference: {os.path.basename(reference_image_path)}")
        else:
            print(f"[PBO Refine] ⚠️ Reference not found: {reference_image_path}")
        
        # Step 6: Start new round in tracking (reuse tracker from earlier)
        # Set concepts if not already set
        if not tracker_for_selection.data.get("concepts"):
            tracker_for_selection.set_concepts(refiner.concepts)
        
        # Start new round
        tracker_for_selection.start_round(
            round_number=request.round_number + 1,
            reference_image=os.path.basename(reference_image_path) if reference_image else None
        )
        
        # Step 7: Generate images with SDXL
        # (refinement_stage and refinement_folder already defined in Step 2)
        round_folder = os.path.join(refinement_folder, f"round_{request.round_number + 1}")
        
        # Prepare image paths for tracking
        image_paths_for_tracking = [
            f"{refinement_stage}/round_{request.round_number + 1}/image_{i}.png" 
            for i in range(len(proposals))
        ]
        
        sdxl_runner = get_sdxl_runner()
        pil_images = refiner.generate_images_from_proposals(
            proposals=proposals,
            sdxl_runner=sdxl_runner,
            seed_base=42 + request.round_number,
            verbose=False,
            init_image=reference_image,
            descriptor=descriptor,  # User description from preferences
            tracker=tracker_for_selection,  # Track all generation details
            generated_image_paths=image_paths_for_tracking  # Image paths for tracking
        )
        
        # Step 7: Save images
        os.makedirs(round_folder, exist_ok=True)
        
        image_paths = []
        for idx, pil_img in enumerate(pil_images):
            image_filename = f"image_{idx}.png"
            image_path = os.path.join(round_folder, image_filename)
            pil_img.save(image_path)
            
            # Return relative path
            rel_path = f"/sessions/{request.session_id}/{refinement_stage}/round_{request.round_number + 1}/{image_filename}"
            image_paths.append(rel_path)
        
        # Step 8: Save weight vectors for this round
        weights_file = os.path.join(round_folder, "weights.json")
        with open(weights_file, "w") as f:
            json.dump({
                "round": request.round_number + 1,
                "proposals": [p.tolist() for p in proposals],
                "concept_labels": [c['label'] for c in refiner.concepts],
                "reference_image": reference_image_id
            }, f, indent=2)
        
        print(f"[PBO Refine] ✅ Round {request.round_number + 1} complete: {len(image_paths)} images")
        
        return RefineNextRoundResponse(
            success=True,
            image_paths=image_paths,
            round_number=request.round_number + 1,
            message=f"Generated round {request.round_number + 1} using PBO"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PBO Refine] Error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pbo/record-refinement-favorite", response_model=FavoriteResponse)
async def pbo_record_refinement_favorite(request: FavoriteRequest):
    """
    Record user's favorite from refinement round.
    
    This updates the PBO model so next proposals are informed by this selection.
    Call this after each refinement round before requesting new proposals.
    """
    try:
        refiner = get_or_create_pbo_refiner(
            session_id=request.session_id,
            stage=request.stage
        )
        
        # Map refinement image IDs to proposals
        # The image_ids should be like: [image_0, image_1, image_2, image_3]
        # These correspond to the last generated proposals
        
        # For now, we'll create proxy duels based on position
        # In a full implementation, you'd track which proposal generated which image
        favorite_idx = int(request.favorite_image_id.split('_')[-1])  # Extract index
        
        print(f"[PBO] Recording favorite refinement image: {request.favorite_image_id} (index {favorite_idx})")
        
        # Record duels (this will guide next proposals)
        refiner.on_favorite(
            favorite_image_id=request.favorite_image_id,
            all_image_ids=request.all_image_ids
        )
        
        # Update tracking with user selection
        session = sessions.get(request.session_id)
        if session:
            session_folder = session['folder']
            tracking_file = os.path.join(session_folder, "tracking.json")
            
            if os.path.exists(tracking_file):
                try:
                    from backend.tracking import GenerationTracker
                    # Load tracker to update with selection
                    tracker = GenerationTracker.__new__(GenerationTracker)
                    tracker.session_path = Path(session_folder)
                    tracker.tracking_file = Path(tracking_file)
                    with open(tracking_file, 'r') as f:
                        tracker.data = json.load(f)
                    
                    # Record selection
                    all_indices = [int(img_id.split('_')[-1]) for img_id in request.all_image_ids]
                    tracker.record_selection(favorite_idx, all_indices)
                    print(f"[Tracking] Recorded selection: {request.favorite_image_id}")
                except Exception as e:
                    print(f"[Tracking] Warning: Could not update tracking: {e}")
        
        return FavoriteResponse(
            duels_added=len(request.all_image_ids) - 1,
            favorite_candidate_id=refiner.image_to_candidate.get(request.favorite_image_id, "unknown"),
            message=f"Recorded refinement favorite, ready for next round"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PBO] Error recording refinement favorite: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pbo/favorite", response_model=FavoriteResponse)
async def pbo_favorite(request: FavoriteRequest):
    """
    Record user's favorite image selection (strong duels).

    Called when user picks their favorite among the generated images.
    """
    try:
        refiner = get_or_create_pbo_refiner(
            session_id=request.session_id,
            stage=request.stage
        )

        refiner.on_favorite(
            favorite_image_id=request.favorite_image_id,
            all_image_ids=request.all_image_ids
        )

        return FavoriteResponse(
            duels_added=len(request.all_image_ids) - 1,
            favorite_candidate_id=refiner.image_to_candidate[request.favorite_image_id],
            message=f"Recorded {len(request.all_image_ids) - 1} strong duels"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[PBO] Error in favorite: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))