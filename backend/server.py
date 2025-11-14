import os
import numpy as np
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from main import run_stage_seq_parallel_optimized
from util import sanitize_folder_name, write_status, initialize_prompt_tracking
from prompt import (
    IMPRESSION_PROMPT, IMPRESSION_GENERATOR_PROMPT
)

import json
from fastapi.responses import JSONResponse
from datetime import datetime

# Import concept refinement module
from concept_refinement import get_or_create_session as get_refinement_session

# Import PBO and SDXL integration
from sdxl_runner import SDXLRunner
from stage_refiner import StageRefiner

# Global variables for SDXL and PBO caching
_sdxl_runner = None
_pbo_refiners = {}


def clear_pbo_cache_for_session(session_id: str) -> int:
    """
    Clear all PBO refiner cache entries for a given session.
    
    This should be called when:
    - Uploading a new session
    - Loading an existing session
    - Auto-recovering a session from disk
    
    Returns:
        Number of cache entries cleared
    """
    keys_to_remove = [key for key in _pbo_refiners.keys() if key.startswith(f"{session_id}:")]
    
    for key in keys_to_remove:
        del _pbo_refiners[key]
    
    if keys_to_remove:
        print(f"[Cache] Cleared {len(keys_to_remove)} PBO refiner(s) for session '{session_id}'")
        print(f"[Cache] Removed keys: {keys_to_remove}")
    
    return len(keys_to_remove)


def clear_pbo_refinement_rounds(session_id: str) -> dict:
    """
    Clear all PBO refinement round folders and tracking data for a session.
    
    This removes:
    - All round_N folders in refinement stages (e.g., impression_refinement/round_1/, round_2/, etc.)
    - tracking.json in refinement stages
    
    Returns:
        Dictionary with counts of what was cleared
    """
    import shutil
    
    session = sessions.get(session_id)
    if not session:
        # Try to find session folder on disk
        session_folder = os.path.join(SESSIONS_DIR, session_id)
        if not os.path.exists(session_folder):
            print(f"[Clear Rounds] Session not found: {session_id}")
            return {"error": "Session not found"}
    else:
        session_folder = session['folder']
    
    cleared = {
        "rounds_cleared": 0,
        "tracking_cleared": 0,
        "stages_processed": []
    }
    
    # Process each refinement stage
    refinement_stages = ["impression_refinement", "spatial_refinement", "objects_refinement", "ambient_refinement"]
    
    for refinement_stage in refinement_stages:
        refinement_folder = os.path.join(session_folder, refinement_stage)
        
        if not os.path.exists(refinement_folder):
            continue
        
        stage_cleared = False
        
        # Remove all round_N folders
        for item in os.listdir(refinement_folder):
            item_path = os.path.join(refinement_folder, item)
            if os.path.isdir(item_path) and item.startswith("round_"):
                try:
                    shutil.rmtree(item_path)
                    cleared["rounds_cleared"] += 1
                    stage_cleared = True
                    print(f"[Clear Rounds] Removed: {refinement_stage}/{item}")
                except Exception as e:
                    print(f"[Clear Rounds] Failed to remove {item_path}: {e}")
        
        # Remove tracking.json
        tracking_file = os.path.join(refinement_folder, "tracking.json")
        if os.path.exists(tracking_file):
            try:
                os.remove(tracking_file)
                cleared["tracking_cleared"] += 1
                stage_cleared = True
                print(f"[Clear Rounds] Removed: {refinement_stage}/tracking.json")
            except Exception as e:
                print(f"[Clear Rounds] Failed to remove {tracking_file}: {e}")
        
        if stage_cleared:
            cleared["stages_processed"].append(refinement_stage)
    
    if cleared["rounds_cleared"] > 0 or cleared["tracking_cleared"] > 0:
        print(f"[Clear Rounds] Summary for '{session_id}':")
        print(f"  Rounds cleared: {cleared['rounds_cleared']}")
        print(f"  Tracking files cleared: {cleared['tracking_cleared']}")
        print(f"  Stages processed: {cleared['stages_processed']}")
    else:
        print(f"[Clear Rounds] No refinement rounds found for '{session_id}'")
    
    return cleared


# Define your stages and prompts
STAGES = [
    "impression", "impression_refinement"
]
# Sequential prompts (refinement stages use PBO+SDXL, not Gemini prompts)
PROMPTS = {
    'impression': (IMPRESSION_PROMPT, IMPRESSION_GENERATOR_PROMPT)
}

# Refinement stages (used to identify if a stage is refinement)
REFINEMENT_STAGES = {
    "impression_refinement"
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

class TagFeedbackRequest(BaseModel):
    session_id: str
    stage: str
    concept_index: int
    positive_tags: list[str]
    negative_tags: list[str]

class TagsRequest(BaseModel):
    session_id: str
    stage: str
    image_id: str
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

class TagsResponse(BaseModel):
    tags: list[str]

@app.post("/api/tags", response_model=TagsResponse)
def get_tags(req: TagsRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    try:
        # Get stage folder
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
            # Save initial concept weights (uniform weights before any interactions)
            refinement_session.save_concept_weights(session['folder'])
            print(f"[CONCEPTS INIT] Saved initial concept weights for {req.stage}")
        
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
        
        # NOTE: Weights auto-save is disabled for performance (saves happen on generation/refinement)
        # If needed, uncomment the lines below to save on every interaction:
        # if req.session_id in sessions:
        #     session_folder = sessions[req.session_id]['folder']
        #     refinement_session.save_concept_weights(session_folder)
        
        # Return updated state
        state_dict = refinement_session.to_dict()
        
        # Minimal logging for speed
        print(f"[API] Returning {len(state_dict['concepts'])} concepts")
        
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
        
        # Save updated weights to disk
        if req.session_id in sessions:
            session_folder = sessions[req.session_id]['folder']
            refinement_session.save_concept_weights(session_folder)
        
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
    tag_weights: dict | None = None  # Optional: tag name -> weight mapping

class FeedbackResponse(BaseModel):
    next_stage: str | None
    images: list[ImageItem]


# --- Progressive Final Mode (Mode 5) Endpoints ---
class FinalProgressiveStartRequest(BaseModel):
    session_id: str
class FinalProgressiveFeedbackRequest(BaseModel):
    session_id: str
    positive_tags: list[str]
    negative_tags: list[str]
class GenerationError(Exception):
    pass

@app.exception_handler(GenerationError)
async def generation_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": str(exc)}
    )

@app.post("/api/feedback", response_model=FeedbackResponse)
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

    # Save tag weights to selection.json if provided
    if req.tag_weights:
        selection_file = os.path.join(folder, "selection.json")
        selection_data = {
            "image_id": req.selected_image_id,
            "stage": req.stage,
            "tag_weights": req.tag_weights
        }
        with open(selection_file, 'w') as f:
            json.dump(selection_data, f, indent=2)
        print(f"✅ Saved selection with {len(req.tag_weights)} tag weights to {selection_file}")

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
    
    # Save learned concept weights for the current stage (sequential/parallel pipeline)
    try:
        from concept_refinement import get_refinement_session as get_ref_session_for_save
        # For refinement stages, save under the refinement stage name
        # For base stages, save under the base stage name
        refinement_session = get_ref_session_for_save(req.session_id, current_stage, [])
        if refinement_session.initialized:
            refinement_session.save_concept_weights(folder)
            print(f"💾 [SEQUENTIAL/PARALLEL] Saved concept weights for {current_stage}")
    except Exception as e:
        print(f"⚠️  Could not save concept weights: {e}")
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

        if current_idx + 1 < len(STAGES):
            next_stage = STAGES[current_idx + 1]
            
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
                
                # Collect image IDs from the base exploration stage
                exploration_stage_folder = os.path.join(folder, base_stage)
                exploration_images = []
                if os.path.exists(exploration_stage_folder):
                    for file in os.listdir(exploration_stage_folder):
                        if file.endswith('.png'):
                            img_id = os.path.splitext(file)[0]
                            exploration_images.append(img_id)
                
                # Access the concept refinement session
                refinement_session = get_refinement_session(req.session_id, base_stage, exploration_images)
                
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
                    
                    # Load learned weights from base stage (warm start)
                    weights_loaded = refinement_session.load_concept_weights_from_base_stage(folder)
                    if weights_loaded:
                        write_status(folder, f"🔥 Warm start: Loaded learned weights from base stage")
                    else:
                        write_status(folder, f"🆕 Cold start: Using uniform weights (no previous weights found)")
                
                # Initialize StageRefiner with PBO
                # Force recreate to pick up the latest learned weights
                refiner = get_or_create_pbo_refiner(
                    session_id=req.session_id,
                    stage=base_stage,
                    force_recreate=True  # Always recreate to use latest weights
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
                
                # Initialize tracking
                from backend.tracking import create_tracker
                tracker = create_tracker(
                    session_path=Path(folder),
                    session_id=req.session_id,
                    stage=next_stage,  # Use refinement stage name, not base stage
                    descriptor=descriptor or "No descriptor"
                )
                tracker.set_concepts(refiner.concepts)
                tracker.start_round(
                    round_number=1,
                    reference_image=os.path.basename(selected_image_path) if selected_image_path else None
                )
                
                # Prepare image paths for tracking
                image_paths_for_tracking = [
                    f"{next_stage}/round_1/image_{i}.png" for i in range(len(proposals))
                ]
                
                # Generate images using SDXL
                sdxl_runner = get_sdxl_runner()
                pil_images = refiner.generate_images_from_proposals(
                    proposals=proposals,
                    sdxl_runner=sdxl_runner,
                    seed_base=42,
                    verbose=False,
                    init_image=reference_image,
                    descriptor=descriptor,  # User description from session
                    tracker=tracker,  # Track all generation details
                    generated_image_paths=image_paths_for_tracking
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
                # Extract concept index from selected_image_id (e.g., "impression_2_0" -> 2)
                selected_concept_index = None
                if selected_image_id:
                    try:
                        parts = selected_image_id.split('_')
                        if len(parts) >= 2:
                            selected_concept_index = int(parts[-2])
                    except (ValueError, IndexError):
                        pass
                
                weights_data = {
                    "round": 1,
                    "proposals": [p.tolist() for p in proposals],
                    "concept_labels": [c['label'] for c in refiner.concepts],
                    "reference_image": selected_image_id,
                    "selected_concept_index": selected_concept_index
                }
                
                weights_file = os.path.join(round_1_folder, "weights.json")
                with open(weights_file, "w") as f:
                    json.dump(weights_data, f, indent=2)
                
                write_status(folder, f"✅ PBO refinement complete! Generated {len(results)} images")
            else:
                # Regular exploration stage: use standard generation
                narrative_prompt, image_prompt = PROMPTS[next_stage]
                
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


# --- Status endpoint ---
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

        # Validate required files (simplified for impression-only)
        # Only impression.json is required; preferences.json is optional (created during exploration)
        required_files = [
            'impression/impression.json'
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

        # Load preferences.json if it exists (optional)
        preferences_path = os.path.join(session_folder, "preferences.json")
        preferences = {}
        if os.path.exists(preferences_path):
            with open(preferences_path, 'r') as f:
                preferences = json.load(f)
        else:
            # Create empty preferences.json
            preferences = {
                "descriptor": f"Uploaded session - {folder_name}",
                "selections": {},
                "user_pref": {}
            }
            with open(preferences_path, 'w') as f:
                json.dump(preferences, f, indent=2)

        # Store session info
        session_id = folder_name
        sessions[session_id] = {
            'folder': session_folder,
            'descriptor': preferences.get('descriptor', f"Uploaded session - {folder_name}"),
            'user_pref': preferences.get('user_pref', {}),
            'uploaded': True
        }
        
        # Clear any cached PBO refiners and refinement rounds for this session
        clear_pbo_cache_for_session(session_id)
        clear_pbo_refinement_rounds(session_id)

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
            
            # Clear any cached PBO refiners and refinement rounds for this recovered session
            clear_pbo_cache_for_session(decoded_session_id)
            clear_pbo_refinement_rounds(decoded_session_id)
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
        # Get stage folder and JSON path
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
            for stage in ["impression", "impression_refinement"]:
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
        
        # Clear any cached PBO refiners and refinement rounds for this session
        clear_pbo_cache_for_session(req.session_path)
        clear_pbo_refinement_rounds(req.session_path)
        
        # Load preferences if exists
        preferences = {}
        prefs_file = os.path.join(session_path, "preferences.json")
        if os.path.exists(prefs_file):
            with open(prefs_file, "r") as f:
                preferences = json.load(f)
        
        # Scan for available stages
        stages_info = []
        for stage in ["impression"]:
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
    descriptor: str

class GenerateStageRefinementResponse(BaseModel):
    success: bool
    images: list[ImageItem]
    stage_json: list[dict]
    refinement_folder: str
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


def get_or_create_pbo_refiner(session_id: str, stage: str, force_recreate: bool = False) -> StageRefiner:
    """
    Get or create StageRefiner for PBO session/stage.

    This integrates with existing ConceptRefinementSession to reuse concepts.
    
    Args:
        session_id: Session ID
        stage: Stage name (e.g., 'impression')
        force_recreate: If True, recreate even if cached (to pick up updated weights)
    """
    key = f"{session_id}:{stage}"

    if force_recreate and key in _pbo_refiners:
        print(f"\n[REFINER CACHE] Force recreating StageRefiner for {session_id}/{stage}")
        print(f"  Reason: Picking up updated weights")
        del _pbo_refiners[key]

    if key not in _pbo_refiners:
        print(f"\n[REFINER CACHE] Creating NEW StageRefiner for {session_id}/{stage}")

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

        # Convert concept_states for warm start
        concept_states = {}
        for cid, state in concept_session.concept_states.items():
            concept_states[cid] = {
                'active': True,  # All concepts are active by default
                'w': state.w,  # Use 'w' key to match ConceptState in concept_refinement.py
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

        refiner = _pbo_refiners[key]
        print(f"[REFINER CACHE] ✅ StageRefiner created and cached")
        print(f"  Concepts: {len(concepts)}")
        print(f"  PBO state: candidates={len(refiner.pbo.candidates)}, duels={len(refiner.pbo.duels)}, fitted={refiner.pbo.fitted}")
    else:
        print(f"\n[REFINER CACHE] Using CACHED StageRefiner for {session_id}/{stage}")
        refiner = _pbo_refiners[key]
        print(f"  PBO state: candidates={len(refiner.pbo.candidates)}, duels={len(refiner.pbo.duels)}, fitted={refiner.pbo.fitted}")

    return _pbo_refiners[key]




class RefineNextRoundRequest(BaseModel):
    session_id: str
    stage: str  # base stage (e.g., "impression")
    selected_image_id: str  # Selected from current round
    all_image_ids: list[str]  # All images in current round
    round_number: int
    injected_tag: Optional[str] = None  # Optional custom tag to inject
    injected_emphasis: Optional[str] = None  # 'high', 'mid', or 'low'

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
        print("\n" + "=" * 80)
        print(f"[PBO REFINE NEXT ROUND] ENDPOINT CALLED")
        print("=" * 80)
        print(f"  Session: {request.session_id}")
        print(f"  Stage: {request.stage}")
        print(f"  Round: {request.round_number} → {request.round_number + 1}")
        print(f"  Selected: {request.selected_image_id}")
        print(f"  All images: {request.all_image_ids}")
        
        # Get refiner
        refiner = get_or_create_pbo_refiner(
            session_id=request.session_id,
            stage=request.stage
        )
        
        print(f"\n[PBO STATE] Before recording selection:")
        print(f"  candidates: {len(refiner.pbo.candidates)}")
        print(f"  duels: {len(refiner.pbo.duels)}")
        print(f"  fitted: {refiner.pbo.fitted}")
        
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
        
        # IMPORTANT: Use refinement stage for tracking, not base stage
        refinement_stage = f"{request.stage}_refinement"
        
        from backend.tracking import create_tracker
        tracker_for_selection = create_tracker(
            session_path=Path(session_folder),
            session_id=request.session_id,
            stage=refinement_stage,  # Use refinement stage, not base stage
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
        # refinement_stage already defined above
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
        
        print(f"\n[PBO Refine] ✅ Recorded selection:")
        print(f"  Candidates added: {len(candidate_ids)}")
        print(f"  Duels added: {duels_added}")
        print(f"  Favorite: {favorite_cand_id}")
        
        print(f"\n[PBO STATE] After recording selection:")
        print(f"  candidates: {len(refiner.pbo.candidates)}")
        print(f"  duels: {len(refiner.pbo.duels)}")
        print(f"  fitted: {refiner.pbo.fitted}")
        
        # Step 3.5: Handle tag injection if provided
        if request.injected_tag and request.injected_tag.strip():
            print(f"\n[TAG INJECTION] Injecting custom tag: '{request.injected_tag}' with {request.injected_emphasis} emphasis")
            refiner.inject_custom_tag(
                tag_text=request.injected_tag.strip(),
                emphasis=request.injected_emphasis or 'mid'
            )
            print(f"[TAG INJECTION] ✅ Tag injected successfully")
            print(f"  New concept count: {refiner.K}")
        
        # Step 4: Propose new weight mixtures
        print(f"\n[PBO Refine] Proposing new mixtures with fit_first=True...")
        proposals = refiner.propose_next_4(
            negatives=None,
            w_current=None,
            fit_first=True
        )
        
        print(f"\n[PBO STATE] After propose:")
        print(f"  candidates: {len(refiner.pbo.candidates)}")
        print(f"  duels: {len(refiner.pbo.duels)}")
        print(f"  fitted: {refiner.pbo.fitted}")
        print(f"  Proposed {len(proposals)} new mixtures")
        
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
        
        # IMPORTANT: If this is the first time creating tracking.json (moving from Round 1 to Round 2),
        # we need to backfill Round 1's data so it appears in selection history
        if request.round_number == 1 and len(tracker_for_selection.data.get("rounds", [])) == 0:
            print(f"[PBO Refine] Backfilling Round 1 data for selection history...")
            # Start Round 1 first
            tracker_for_selection.start_round(
                round_number=1,
                reference_image=os.path.basename(reference_image_path) if reference_image else None
            )
            # Record Round 1's proposals (from the old flow round_1/ folder)
            round_1_weights_file = os.path.join(refinement_folder, "round_1", "weights.json")
            if os.path.exists(round_1_weights_file):
                with open(round_1_weights_file, 'r') as f:
                    round_1_weights = json.load(f)
                    # Add proposals to tracking with ALL required fields
                    for i in range(4):
                        w_raw = np.array(round_1_weights["proposals"][i], dtype=np.float32)
                        
                        # Normalize weights
                        from backend.sdxl_integration import normalize_simplex, compute_gains
                        w_norm = normalize_simplex(w_raw)
                        
                        # Compute gains and statistics
                        gains = compute_gains(w_norm)
                        mean_w = float(np.mean(w_norm))
                        std_w = float(np.std(w_norm))
                        z_scores = (w_norm - mean_w) / (std_w + 1e-8)
                        
                        # Build concept breakdown with ALL required fields
                        concept_breakdown = []
                        for idx, concept in enumerate(refiner.concepts):
                            # Compute rank (1 = highest weight)
                            rank = int(np.where(np.argsort(w_norm)[::-1] == idx)[0][0]) + 1
                            
                            concept_breakdown.append({
                                "concept_id": refiner.concept_ids[idx],
                                "label": concept["label"],
                                "weight_raw": float(w_raw[idx]),
                                "weight_normalized": float(w_norm[idx]),
                                "z_score": float(z_scores[idx]),
                                "gain_before_clip": float(1.0 + 0.4 * z_scores[idx]),
                                "gain_after_clip": float(gains[idx]),
                                "rank": rank,
                                "included_positive": False,  # Not available for backfill
                                "included_negative": False   # Not available for backfill
                            })
                        
                        from datetime import datetime as dt
                        tracker_for_selection.data["rounds"][0]["proposals"].append({
                            "proposal_index": i,
                            "seed": 42 + i,
                            "generated_image": f"impression_refinement/round_1/image_{i}.png",
                            "generated_at": dt.now().isoformat(),
                            
                            # Weight statistics
                            "weight_statistics": {
                                "raw_weights": [float(x) for x in w_raw],
                                "normalized_weights": [float(x) for x in w_norm],
                                "mean": mean_w,
                                "std": std_w,
                                "min": float(w_norm.min()),
                                "max": float(w_norm.max())
                            },
                            
                            # Concept breakdown
                            "concept_breakdown": concept_breakdown,
                            
                            # Prompt composition (not available for backfill)
                            "prompt_composition": {
                                "positive_phrases": [],
                                "negative_phrases": []
                            },
                            
                            # Generation params with ALL required fields
                            "generation_params": {
                                "mode": "img2img",
                                "strength": 0.65,
                                "steps": 27,
                                "guidance_scale": 7.5
                            }
                        })
            # Record Round 1's selection (the one user just made)
            tracker_for_selection.record_selection(selected_index, all_indices)
            print(f"[PBO Refine] ✅ Backfilled Round 1 with selection index {selected_index}")
        
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
        weights_data = {
            "round": request.round_number + 1,
            "proposals": [p.tolist() for p in proposals],
            "concept_labels": [c['label'] for c in refiner.concepts],
            "reference_image": reference_image_id
        }
        
        # Add injected tag info if provided
        if request.injected_tag and request.injected_tag.strip():
            emphasis_weights = {'high': 0.5, 'mid': 0.3, 'low': 0.1}
            weights_data["injected_tag"] = {
                "tag": request.injected_tag.strip(),
                "emphasis": request.injected_emphasis or 'mid',
                "weight": emphasis_weights.get(request.injected_emphasis or 'mid', 0.3),
                "round": request.round_number + 1
            }
        
        with open(weights_file, "w") as f:
            json.dump(weights_data, f, indent=2)
        
        print(f"\n[PBO Refine] ✅ Round {request.round_number + 1} complete:")
        print(f"  Generated: {len(image_paths)} images")
        print(f"  Saved weights to: {weights_file}")
        print("=" * 80 + "\n")
        
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


class RefineFromWeightsRequest(BaseModel):
    session_id: str
    stage: str  # base stage (e.g., "impression")
    weights: list[float]  # Historical weight vector to refine from
    round_number: int

class RefineFromWeightsResponse(BaseModel):
    success: bool
    image_paths: list[str]
    round_number: int
    message: str


@app.post("/api/pbo/refine-from-weights", response_model=RefineFromWeightsResponse)
async def pbo_refine_from_weights(request: RefineFromWeightsRequest):
    """
    Generate new round using historical weights as a starting point.
    This allows users to revisit previous selections and explore nearby regions.
    """
    try:
        print("\n" + "=" * 80)
        print(f"[PBO REFINE FROM WEIGHTS] ENDPOINT CALLED")
        print("=" * 80)
        print(f"  Session: {request.session_id}")
        print(f"  Stage: {request.stage}")
        print(f"  Weights shape: {len(request.weights)}")
        print(f"  Round: {request.round_number} → {request.round_number + 1}")
        
        # Get refiner
        refiner = get_or_create_pbo_refiner(
            session_id=request.session_id,
            stage=request.stage
        )
        
        # Convert weights to numpy array
        import numpy as np
        historical_weights = np.array(request.weights, dtype=np.float32)
        
        # Add the historical weights as a candidate to the PBO
        # (this will inform the GP about this region of interest)
        cid = refiner.pbo.add_candidate(
            w=historical_weights,
            candidate_id="historical_selection"
        )
        print(f"[PBO] Added historical weights as candidate: {cid}")
        
        # Fit the GP
        refiner.pbo.fit()
        print(f"[PBO] GP fitted with {len(refiner.pbo.candidates)} candidates, {len(refiner.pbo.duels)} duels")
        
        # Propose new batch using local_around to explore near the historical weights
        # This gives 4 diverse proposals around the historical selection
        from backend.pbo import local_around
        proposals = []
        for i in range(4):
            w_local = local_around(historical_weights, alpha_scale=30.0 + i*10, top_k=10, rng=refiner.pbo.rng)
            proposals.append(w_local)
        
        print(f"[PBO] Generated 4 local proposals around historical weights")
        
        # Get session info
        session = sessions.get(request.session_id)
        if not session:
            raise HTTPException(404, f"Session not found: {request.session_id}")
        
        session_folder = session['folder']
        preferences_file = os.path.join(session_folder, "preferences.json")
        
        # Get the original reference image
        reference_image_id = None
        if os.path.exists(preferences_file):
            with open(preferences_file, 'r') as f:
                prefs = json.load(f)
                selections = prefs.get('selections', {})
                reference_image_id = selections.get(request.stage)
        
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
        
        # Get descriptor
        descriptor = None
        if os.path.exists(preferences_file):
            with open(preferences_file, 'r') as f:
                prefs = json.load(f)
                descriptor = prefs.get('descriptor')
        
        # Create tracker and start new round
        from backend.tracking import create_tracker
        tracker = create_tracker(
            session_path=Path(session_folder),
            session_id=request.session_id,
            stage=request.stage,
            descriptor=descriptor or "No descriptor"
        )
        
        if not tracker.data.get("concepts"):
            tracker.set_concepts(refiner.concepts)
        
        tracker.start_round(
            round_number=request.round_number + 1,
            reference_image=os.path.basename(reference_image_path) if reference_image else None
        )
        
        # Generate images with SDXL
        refinement_stage = f"{request.stage}_refinement"
        refinement_folder = os.path.join(session_folder, refinement_stage)
        round_folder = os.path.join(refinement_folder, f"round_{request.round_number + 1}")
        os.makedirs(round_folder, exist_ok=True)
        
        # Generate images using refiner
        print(f"[StageRefiner] Generating 4 images from proposals...")
        pil_images = refiner.generate_images_batch(
            weight_vectors=proposals,
            reference_image=reference_image
        )
        
        # Save images
        image_paths = []
        for i, pil_img in enumerate(pil_images):
            image_filename = f"image_{i}.png"
            image_path = os.path.join(round_folder, image_filename)
            pil_img.save(image_path)
            
            # Return relative path for frontend
            relative_path = f"/sessions/{request.session_id}/{refinement_stage}/round_{request.round_number + 1}/{image_filename}"
            image_paths.append(relative_path)
        
        print(f"[StageRefiner] ✅ Generated {len(pil_images)} images")
        
        # Record proposals in tracking
        tracker.record_proposals(
            proposals=[{
                'weights': w.tolist(),
                'image_path': f"round_{request.round_number + 1}/image_{i}.png"
            } for i, w in enumerate(proposals)]
        )
        
        # Save weights
        weights_data = {
            "round": request.round_number + 1,
            "proposals": [p.tolist() for p in proposals],
            "concept_labels": [c['label'] for c in refiner.concepts],
            "reference_image": reference_image_id,
            "source": "historical_weights"
        }
        
        weights_file = os.path.join(round_folder, "weights.json")
        with open(weights_file, "w") as f:
            json.dump(weights_data, f, indent=2)
        
        print(f"[PBO Refine] ✅ Round {request.round_number + 1} complete:")
        print(f"  Generated: {len(image_paths)} images")
        print(f"  Saved weights to: {weights_file}")
        print("=" * 80 + "\n")
        
        return RefineFromWeightsResponse(
            success=True,
            image_paths=image_paths,
            round_number=request.round_number + 1,
            message=f"Generated round {request.round_number + 1} from historical weights"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PBO Refine From Weights] Error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pbo/debug-state")
async def pbo_debug_state(session_id: str, stage: str):
    """
    Diagnostic endpoint to inspect current PBO state.
    
    Returns detailed information about the PBO refiner state including:
    - Number of candidates and duels
    - GP fitting status
    - Concept weights
    - Recent candidates
    """
    try:
        key = f"{session_id}:{stage}"
        
        if key not in _pbo_refiners:
            return {
                "error": f"No PBO refiner found for {session_id}/{stage}",
                "cached_sessions": list(_pbo_refiners.keys())
            }
        
        refiner = _pbo_refiners[key]
        pbo = refiner.pbo
        
        # Get recent candidates
        recent_candidates = []
        for cid, cand in list(pbo.candidates.items())[-10:]:  # Last 10
            recent_candidates.append({
                "id": cand.id,
                "top_3_concepts": [(refiner.concepts[i]['label'], float(cand.w[i])) 
                                   for i in np.argsort(-cand.w)[:3]]
            })
        
        # Get recent duels
        recent_duels = []
        for duel in list(pbo.duels)[-10:]:  # Last 10
            recent_duels.append({
                "better": duel.better_id,
                "worse": duel.worse_id,
                "strength": duel.strength
            })
        
        return {
            "session_id": session_id,
            "stage": stage,
            "pbo_state": {
                "num_candidates": len(pbo.candidates),
                "num_duels": len(pbo.duels),
                "fitted": pbo.fitted,
                "K": pbo.K,
                "d": pbo.d
            },
            "concept_weights": {
                "sum": float(pbo.concept_weights.sum()),
                "top_5": [(refiner.concepts[i]['label'], float(pbo.concept_weights[i])) 
                          for i in np.argsort(-pbo.concept_weights)[:5]]
            },
            "recent_candidates": recent_candidates,
            "recent_duels": recent_duels,
            "cache_key": key
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }