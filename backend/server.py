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
from fastapi.exceptions import RequestValidationError
from datetime import datetime

# Import concept refinement module
from concept_refinement import get_or_create_session as get_refinement_session

# Import PBO and SDXL integration (backend version - for refinement)
from sdxl_runner import SDXLRunner  # Backend version - no alpha, for refinement
from stage_refiner import StageRefiner

# Import SDXL folder modules for slider generation (has alpha support)
import sys
SDXL_PATH = os.path.join(os.path.dirname(__file__), '..', 'SDXL')
sys.path.insert(0, SDXL_PATH)
from slider_generator import generate_cozy_sweep
# Import SDXL folder's embed fuser for slider (has fuse_with_alpha)
import importlib.util
_sdxl_embed_fuser_spec = importlib.util.spec_from_file_location("sdxl_embed_fuser_slider", os.path.join(SDXL_PATH, "sdxl_embed_fuser.py"))
_sdxl_embed_fuser_module = importlib.util.module_from_spec(_sdxl_embed_fuser_spec)
_sdxl_embed_fuser_spec.loader.exec_module(_sdxl_embed_fuser_module)
SDXLEmbedFuserSlider = _sdxl_embed_fuser_module.SDXLEmbedFuser

# Global variables for SDXL and PBO caching
_sdxl_runner = None  # Backend's runner for refinement
_slider_fuser = None  # SDXL folder's fuser for slider (shares pipeline with _sdxl_runner)
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


# ============================================================================
# Selection History Management
# ============================================================================
def init_selection_history(refinement_folder: str, concept_labels: list) -> dict:
    """
    Initialize selection_history.json when refinement folder is created.
    
    Args:
        refinement_folder: Path to the refinement stage folder (e.g., .../impression_refinement)
        concept_labels: List of concept labels for weight vector interpretation
    
    Returns:
        The initialized selection history dict
    """
    history_file = os.path.join(refinement_folder, "selection_history.json")
    
    history = {
        "created_at": datetime.now().isoformat(),
        "concept_labels": concept_labels,
        "selections": []  # List of {round, image_name, image_path, weights, selected_at}
    }
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"[Selection History] ✅ Initialized: {history_file}")
    return history


def record_selection(refinement_folder: str, round_number: int, image_index: int, 
                     image_name: str, weights: list) -> dict:
    """
    Record a selection in selection_history.json.
    
    Args:
        refinement_folder: Path to the refinement stage folder
        round_number: The round number (1-indexed)
        image_index: Index of selected image in the round (0-3)
        image_name: Name of the selected image file (e.g., "image_2.png")
        weights: The weight vector for the selected image
    
    Returns:
        The updated selection history dict
    """
    history_file = os.path.join(refinement_folder, "selection_history.json")
    
    # Load existing history or create new
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = {
            "created_at": datetime.now().isoformat(),
            "concept_labels": [],
            "selections": []
        }
    
    # Build image path relative to session
    stage_name = os.path.basename(refinement_folder)
    image_path = f"{stage_name}/round_{round_number}/{image_name}"
    
    # Add selection entry
    selection_entry = {
        "round": round_number,
        "image_index": image_index,
        "image_name": image_name,
        "image_path": image_path,
        "weights": weights,
        "selected_at": datetime.now().isoformat()
    }
    
    # Check if we already have a selection for this round (update it)
    existing_idx = None
    for i, s in enumerate(history["selections"]):
        if s["round"] == round_number:
            existing_idx = i
            break
    
    if existing_idx is not None:
        history["selections"][existing_idx] = selection_entry
        print(f"[Selection History] Updated round {round_number}: {image_name}")
    else:
        history["selections"].append(selection_entry)
        print(f"[Selection History] Added round {round_number}: {image_name}")
    
    # Sort by round number
    history["selections"].sort(key=lambda x: x["round"])
    
    # Save
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


def get_selection_history(refinement_folder: str) -> dict:
    """
    Get the selection history for a refinement stage.
    
    Args:
        refinement_folder: Path to the refinement stage folder
    
    Returns:
        The selection history dict, or empty structure if not found
    """
    history_file = os.path.join(refinement_folder, "selection_history.json")
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    
    return {
        "created_at": None,
        "concept_labels": [],
        "selections": []
    }


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

# Add validation error handler to log details
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"\n⚠️ VALIDATION ERROR on {request.url.path}")
    print(f"  Errors: {exc.errors()}")
    try:
        body = await request.body()
        print(f"  Body: {body[:500].decode() if body else 'empty'}")
    except:
        print(f"  Body: <could not read>")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
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
    adjective: str = ""
    location: str = ""

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
    print(f"   Adjective: {req.adjective}, Location: {req.location}")
    
    # Store session info
    sessions[session_id] = {
        'folder': session_folder,
        'descriptor': req.descriptor,
        'adjective': req.adjective,
        'location': req.location,
        'user_pref': {}
    }
    
    # Create initial final_selection.json with basic info
    initial_final_selection = {
        "adjective": req.adjective,
        "location": req.location,
        "descriptor": req.descriptor,
        "session_id": session_id,
        "created_at": datetime.now().isoformat()
    }
    final_selection_path = os.path.join(session_folder, "final_selection.json")
    with open(final_selection_path, "w") as f:
        json.dump(initial_final_selection, f, indent=2)
    print(f"📄 Created initial final_selection.json")
    
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
    print(f"🔍 DEBUG - Feedback request:")
    print(f"  session_id: {req.session_id}")
    print(f"  stage: {req.stage}")
    print(f"  selected_image_id: {req.selected_image_id}")
    print(f"  preferences type: {type(req.preferences)}, keys: {list(req.preferences.keys()) if req.preferences else 'None'}")
    print(f"  tag_weights: {req.tag_weights is not None}")
    
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
                
                # Initialize selection_history.json
                init_selection_history(
                    refinement_folder=refinement_folder,
                    concept_labels=[c['label'] for c in refiner.concepts]
                )
                
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
        
        # Get descriptor from preferences.json
        preferences_file = os.path.join(session_folder, "preferences.json")
        descriptor = None
        if os.path.exists(preferences_file):
            with open(preferences_file, 'r') as f:
                prefs = json.load(f)
                descriptor = prefs.get('descriptor')
        
        # Get location from final_selection.json
        final_selection_file = os.path.join(session_folder, "final_selection.json")
        location = None
        if os.path.exists(final_selection_file):
            with open(final_selection_file, 'r') as f:
                final_selection = json.load(f)
                location = final_selection.get('location')
        
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
        
        # Record selection in selection_history.json
        selected_weights = proposals_from_round[favorite_index].tolist()
        record_selection(
            refinement_folder=refinement_folder,
            round_number=request.round_number,
            image_index=favorite_index,
            image_name=f"image_{favorite_index}.png",
            weights=selected_weights
        )
        
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
            location=location,  # Location for txt2img tag prefixing
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
    current_round_image_ids: list[str] = []  # IDs of current round images to record preference against

class RefineFromWeightsResponse(BaseModel):
    success: bool
    image_paths: list[str]
    round_number: int
    message: str


@app.post("/api/pbo/refine-from-weights", response_model=RefineFromWeightsResponse)
async def pbo_refine_from_weights(request: RefineFromWeightsRequest):
    """
    Generate new round using historical weights as a starting point.
    
    When user clicks a previous selection:
    1. Record preference: historical_selection > all current round images
    2. Generate new proposals around the historical weights
    3. Use PBO to learn from this preference
    """
    try:
        print("\n" + "=" * 80)
        print(f"[PBO REFINE FROM WEIGHTS] ENDPOINT CALLED")
        print("=" * 80)
        print(f"  Session: {request.session_id}")
        print(f"  Stage: {request.stage}")
        print(f"  Weights shape: {len(request.weights)}")
        print(f"  Round: {request.round_number} → {request.round_number + 1}")
        print(f"  Current round images to compare against: {request.current_round_image_ids}")
        
        # Get refiner
        refiner = get_or_create_pbo_refiner(
            session_id=request.session_id,
            stage=request.stage
        )
        
        # Convert weights to numpy array
        import numpy as np
        historical_weights = np.array(request.weights, dtype=np.float32)
        
        # Step 1: Add the historical weights as a candidate and record preferences
        historical_cid = refiner.pbo.add_candidate(
            w=historical_weights,
            candidate_id=f"historical_round{request.round_number}"
        )
        print(f"[PBO] Added historical weights as candidate: {historical_cid}")
        
        # Step 2: If we have current round images, load their weights and record preferences
        # (historical selection beats all current round images)
        session = sessions.get(request.session_id)
        if not session:
            raise HTTPException(404, f"Session not found: {request.session_id}")
        
        session_folder = session['folder']
        refinement_stage = f"{request.stage}_refinement"
        refinement_folder = os.path.join(session_folder, refinement_stage)
        
        duels_added = 0
        if request.current_round_image_ids:
            # Load weights for current round
            current_round_folder = os.path.join(refinement_folder, f"round_{request.round_number}")
            weights_file = os.path.join(current_round_folder, "weights.json")
            
            if os.path.exists(weights_file):
                with open(weights_file, 'r') as f:
                    current_weights_data = json.load(f)
                
                current_proposals = [np.array(w, dtype=np.float32) for w in current_weights_data['proposals']]
                
                # Add each current image as a candidate and record preference
                for i, (img_id, w) in enumerate(zip(request.current_round_image_ids, current_proposals)):
                    current_cid = refiner.pbo.add_candidate(
                        w=w, 
                        candidate_id=f"round{request.round_number}_img{i}"
                    )
                    refiner.image_to_candidate[img_id] = current_cid
                    
                    # Historical selection beats this current image
                    refiner.pbo.add_preference(historical_cid, current_cid, strength=1.0)
                    duels_added += 1
                
                print(f"[PBO] Recorded {duels_added} preferences: historical > current round images")
        
        # Step 3: Fit the GP with the new preferences
        refiner.pbo.fit()
        print(f"[PBO] GP fitted with {len(refiner.pbo.candidates)} candidates, {len(refiner.pbo.duels)} duels")
        
        # Step 4: Propose new batch - use PBO's propose method to explore around the preferred region
        # The GP now knows the user prefers the historical weights region
        proposals = refiner.propose_next_4(
            negatives=None,
            w_current=historical_weights,  # Start from historical weights
            fit_first=False  # Already fitted above
        )
        
        print(f"[PBO] Generated {len(proposals)} new proposals informed by preference")
        
        # Get preferences file
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
        
        # Get descriptor from preferences.json
        descriptor = None
        if os.path.exists(preferences_file):
            with open(preferences_file, 'r') as f:
                prefs = json.load(f)
                descriptor = prefs.get('descriptor')
        
        # Get location from final_selection.json
        final_selection_file = os.path.join(session_folder, "final_selection.json")
        location = None
        if os.path.exists(final_selection_file):
            with open(final_selection_file, 'r') as f:
                final_selection = json.load(f)
                location = final_selection.get('location')
        
        # Create tracker and start new round
        from backend.tracking import create_tracker
        tracker = create_tracker(
            session_path=Path(session_folder),
            session_id=request.session_id,
            stage=refinement_stage,  # Use refinement stage, not base stage
            descriptor=descriptor or "No descriptor"
        )
        
        if not tracker.data.get("concepts"):
            tracker.set_concepts(refiner.concepts)
        
        tracker.start_round(
            round_number=request.round_number + 1,
            reference_image=os.path.basename(reference_image_path) if reference_image else None
        )
        
        # Generate images with SDXL
        round_folder = os.path.join(refinement_folder, f"round_{request.round_number + 1}")
        os.makedirs(round_folder, exist_ok=True)
        
        # Prepare image paths for tracking
        image_paths_for_tracking = [
            f"{refinement_stage}/round_{request.round_number + 1}/image_{i}.png" 
            for i in range(len(proposals))
        ]
        
        # Generate images using the correct method
        print(f"[StageRefiner] Generating {len(proposals)} images from proposals...")
        sdxl_runner = get_sdxl_runner()
        pil_images = refiner.generate_images_from_proposals(
            proposals=proposals,
            sdxl_runner=sdxl_runner,
            seed_base=42 + request.round_number,
            verbose=False,
            init_image=reference_image,
            descriptor=descriptor,
            location=location,  # Location for txt2img tag prefixing
            tracker=tracker,
            generated_image_paths=image_paths_for_tracking
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
        
        # Save weights
        weights_data = {
            "round": request.round_number + 1,
            "proposals": [p.tolist() for p in proposals],
            "concept_labels": [c['label'] for c in refiner.concepts],
            "reference_image": reference_image_id,
            "source": "historical_weights",
            "historical_round": request.round_number,
            "duels_from_historical": duels_added
        }
        
        weights_file = os.path.join(round_folder, "weights.json")
        with open(weights_file, "w") as f:
            json.dump(weights_data, f, indent=2)
        
        # Record selection in selection_history.json
        # This marks that the historical image was "selected" over current round
        record_selection(
            refinement_folder=refinement_folder,
            round_number=request.round_number,
            image_index=-1,  # Special marker for historical selection
            image_name="historical_selection",
            weights=historical_weights.tolist()
        )
        
        print(f"\n[PBO Refine] ✅ Round {request.round_number + 1} complete:")
        print(f"  Generated: {len(image_paths)} images")
        print(f"  Preferences recorded: historical > {duels_added} current images")
        print(f"  Saved weights to: {weights_file}")
        print("=" * 80 + "\n")
        
        return RefineFromWeightsResponse(
            success=True,
            image_paths=image_paths,
            round_number=request.round_number + 1,
            message=f"Generated round {request.round_number + 1} from historical weights (recorded {duels_added} preferences)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PBO Refine From Weights] Error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Selection History API
# ============================================================================
class SelectionHistoryRequest(BaseModel):
    session_id: str
    stage: str  # base stage (e.g., "impression")


@app.post("/api/pbo/selection-history")
def get_pbo_selection_history(request: SelectionHistoryRequest):
    """
    Get the selection history for a refinement stage.
    
    Returns all selected images with their weights for display in the UI.
    """
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {request.session_id}")
    
    session_folder = session['folder']
    refinement_stage = f"{request.stage}_refinement"
    refinement_folder = os.path.join(session_folder, refinement_stage)
    
    if not os.path.exists(refinement_folder):
        return {
            "session_id": request.session_id,
            "stage": refinement_stage,
            "selections": [],
            "concept_labels": []
        }
    
    history = get_selection_history(refinement_folder)
    
    # Add full image URLs for frontend display
    selections_with_urls = []
    for sel in history.get("selections", []):
        sel_copy = sel.copy()
        sel_copy["image_url"] = f"/sessions/{request.session_id}/{sel['image_path']}"
        selections_with_urls.append(sel_copy)
    
    return {
        "session_id": request.session_id,
        "stage": refinement_stage,
        "selections": selections_with_urls,
        "concept_labels": history.get("concept_labels", [])
    }


# ============================================================================
# Save Final Selection API
# ============================================================================
class SaveFinalSelectionRequest(BaseModel):
    session_id: str
    stage: str  # base stage (e.g., "impression")
    weights: list[float]
    image_path: str  # Path to the selected image
    round_number: int  # Which round this selection came from
    is_historical: bool = False  # Whether this is from a historical selection

class SaveFinalSelectionResponse(BaseModel):
    success: bool
    message: str
    file_path: str


@app.post("/api/save-final-selection", response_model=SaveFinalSelectionResponse)
def save_final_selection(request: SaveFinalSelectionRequest):
    """
    Save the final selection from refinement to final_selection.json in the session folder.
    
    This records:
    - Concept names (labels) with their weights
    - The selected image path
    - Metadata about the selection
    """
    try:
        print("\n" + "=" * 80)
        print(f"[SAVE FINAL SELECTION] ENDPOINT CALLED")
        print("=" * 80)
        print(f"  Session: {request.session_id}")
        print(f"  Stage: {request.stage}")
        print(f"  Round: {request.round_number}")
        print(f"  Image: {request.image_path}")
        print(f"  Is Historical: {request.is_historical}")
        print(f"  Weights count: {len(request.weights)}")
        
        # Get session
        session = sessions.get(request.session_id)
        if not session:
            raise HTTPException(404, f"Session not found: {request.session_id}")
        
        session_folder = session['folder']
        
        # Load concept labels from the refinement stage
        refinement_stage = f"{request.stage}_refinement"
        refinement_folder = os.path.join(session_folder, refinement_stage)
        
        # Try to get concept labels from the weights.json of any round
        concept_labels = []
        round_folder = os.path.join(refinement_folder, f"round_{request.round_number}")
        weights_file = os.path.join(round_folder, "weights.json")
        
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                weights_data = json.load(f)
                concept_labels = weights_data.get("concept_labels", [])
        else:
            # Try round 1 as fallback
            round_1_weights = os.path.join(refinement_folder, "round_1", "weights.json")
            if os.path.exists(round_1_weights):
                with open(round_1_weights, 'r') as f:
                    weights_data = json.load(f)
                    concept_labels = weights_data.get("concept_labels", [])
        
        # Build the concepts array with label and weight
        concepts = []
        for i, weight in enumerate(request.weights):
            if i < len(concept_labels):
                concepts.append({
                    "id": f"c{i}",
                    "label": concept_labels[i],
                    "weight": float(weight)
                })
            else:
                concepts.append({
                    "id": f"c{i}",
                    "label": f"concept_{i}",
                    "weight": float(weight)
                })
        
        # Sort by weight descending for readability
        concepts_sorted = sorted(concepts, key=lambda x: x["weight"], reverse=True)
        
        # Load existing final_selection.json to preserve adjective and location
        final_selection_path = os.path.join(session_folder, "final_selection.json")
        existing_data = {}
        if os.path.exists(final_selection_path):
            with open(final_selection_path, 'r') as f:
                existing_data = json.load(f)
        
        # Get adjective and location from existing file or session
        adjective = existing_data.get("adjective", session.get("adjective", ""))
        location = existing_data.get("location", session.get("location", ""))
        descriptor = existing_data.get("descriptor", session.get("descriptor", ""))
        
        # Build final selection data
        from datetime import datetime
        final_selection = {
            "saved_at": datetime.now().isoformat(),
            "session_id": request.session_id,
            "adjective": adjective,
            "location": location,
            "descriptor": descriptor,
            "stage": request.stage,
            "round_number": request.round_number,
            "is_historical_selection": request.is_historical,
            "image_path": request.image_path,
            "concepts": concepts_sorted,
            "weights_raw": request.weights,
            "summary": {
                "total_concepts": len(concepts),
                "top_3_concepts": [c["label"] for c in concepts_sorted[:3]],
                "top_3_weights": [c["weight"] for c in concepts_sorted[:3]]
            }
        }
        
        # Save to session folder (not inside impression_refinement)
        with open(final_selection_path, "w") as f:
            json.dump(final_selection, f, indent=2)
        
        print(f"\n[SAVE FINAL SELECTION] ✅ Saved to: {final_selection_path}")
        print(f"  Top concepts: {[c['label'] for c in concepts_sorted[:3]]}")
        print("=" * 80 + "\n")
        
        return SaveFinalSelectionResponse(
            success=True,
            message=f"Final selection saved with {len(concepts)} concepts",
            file_path=final_selection_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SAVE FINAL SELECTION] Error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== SLIDER GENERATION ENDPOINT ==============

class SliderGenerateRequest(BaseModel):
    session_id: str
    location: str = ""  # If empty, use original location from final_selection.json

class SliderImageItem(BaseModel):
    alpha: float
    url: str

class SliderData(BaseModel):
    slider_type: str  # 'current', 'exploration', or 'refinement'
    adjective: str
    location: str
    descriptor: str
    images: list[SliderImageItem]

class SliderGenerateResponse(BaseModel):
    success: bool
    sliders: list[SliderData]

@app.post("/api/generate-slider", response_model=SliderGenerateResponse)
def generate_slider(request: SliderGenerateRequest):
    """
    Generate three semantic slider sweeps:
    1. Current slider: descriptor → refinement weights [0.0, 0.25, 0.5, 0.75, 1.0, 1.0withref] (6 images)
    2. Slider 1: descriptor → exploration concept_weights [0.0, 0.25, 0.5, 0.75] (4 images, text2text)
    3. Slider 2: exploration weights → refinement weights [0.0, 0.25, 0.5, 0.75] (4 images, text2text)
    """
    try:
        print("\n" + "=" * 80)
        print(f"[SLIDER GENERATION] Starting for session: {request.session_id}")
        print("=" * 80)
        
        # Get session - try memory first, then disk
        session = sessions.get(request.session_id)
        if not session:
            # Try to find session folder on disk
            session_folder = os.path.join(SESSIONS_DIR, request.session_id)
            if not os.path.exists(session_folder):
                raise HTTPException(404, f"Session not found: {request.session_id}")
            
            # Create session entry from disk
            session = {
                'folder': session_folder,
                'descriptor': request.session_id,
                'user_pref': {}
            }
            sessions[request.session_id] = session
            print(f"  Loaded session from disk: {session_folder}")
        
        session_folder = session['folder']
        
        # Load final_selection.json
        final_selection_path = os.path.join(session_folder, "final_selection.json")
        if not os.path.exists(final_selection_path):
            raise HTTPException(404, "final_selection.json not found. Please save a selection first.")
        
        with open(final_selection_path, 'r') as f:
            final_selection = json.load(f)
        
        # Load concept_weights.json from exploration stage
        concept_weights_path = os.path.join(session_folder, "impression", "concept_weights.json")
        if not os.path.exists(concept_weights_path):
            raise HTTPException(404, "concept_weights.json not found in impression stage. Please complete exploration first.")
        
        with open(concept_weights_path, 'r') as f:
            concept_weights_data = json.load(f)
        
        # Load preferences.json for initial descriptor
        preferences_file = os.path.join(session_folder, "preferences.json")
        if not os.path.exists(preferences_file):
            raise HTTPException(404, "preferences.json not found.")
        
        with open(preferences_file, 'r') as f:
            prefs = json.load(f)
        
        # Get adjective and location
        adjective = final_selection.get("adjective", "")
        location = request.location if request.location else final_selection.get("location", "")
        descriptor = f"{adjective} {location}"
        initial_descriptor = prefs.get("descriptor", descriptor)  # Use from preferences if available
        
        print(f"  Adjective: {adjective}")
        print(f"  Location: {location}")
        print(f"  Descriptor: {descriptor}")
        print(f"  Initial descriptor: {initial_descriptor}")
        
        # Get concepts and weights from final_selection (refinement)
        concepts_data_refinement = final_selection.get("concepts", [])
        weights_raw_refinement = final_selection.get("weights_raw", [])
        
        if not concepts_data_refinement or not weights_raw_refinement:
            raise HTTPException(400, "No concepts or weights found in final_selection.json")
        
        # Get concept weights from exploration
        concept_weights_exploration = concept_weights_data.get("concept_weights", [])
        if not concept_weights_exploration:
            raise HTTPException(400, "No concept weights found in concept_weights.json")
        
        # Build concept mapping: label -> weight for exploration
        exploration_weight_map = {cw["label"]: cw["weight"] for cw in concept_weights_exploration}
        
        # Build concepts list with format: "{location} with {label}"
        concepts_refinement = []
        for c in concepts_data_refinement:
            concepts_refinement.append({
                "id": c["id"],
                "label": f"{location} with {c['label']}"
            })
        
        # Build exploration concepts (match refinement concepts by label)
        concepts_exploration = []
        exploration_weights_list = []
        for c in concepts_data_refinement:
            label = c['label']
            exploration_weight = exploration_weight_map.get(label, 0.0)
            concepts_exploration.append({
                "id": c["id"],
                "label": f"{location} with {label}"
            })
            exploration_weights_list.append(exploration_weight)
        
        # Convert weights to numpy arrays
        w_refinement = np.array(weights_raw_refinement)
        w_exploration = np.array(exploration_weights_list)
        
        # Normalize weights
        w_refinement_norm = w_refinement / (w_refinement.sum() + 1e-8)
        w_exploration_norm = w_exploration / (w_exploration.sum() + 1e-8)
        
        print(f"  Refinement concepts count: {len(concepts_refinement)}")
        print(f"  Exploration concepts count: {len(concepts_exploration)}")
        print(f"  Refinement weights shape: {w_refinement_norm.shape}")
        print(f"  Exploration weights shape: {w_exploration_norm.shape}")
        
        # Load reference image from exploration stage (for current slider 6th image)
        reference_image = None
        is_original_location = not request.location  # Empty location means original
        
        if is_original_location:
            selections = prefs.get('selections', {})
            reference_image_id = selections.get('impression')
            
            if reference_image_id:
                # Load reference image from impression stage
                impression_folder = os.path.join(session_folder, 'impression')
                reference_image_path = os.path.join(impression_folder, f"{reference_image_id}.png")
                
                if not os.path.exists(reference_image_path):
                    # Try fallback: {image_id}_0.png
                    reference_image_path = os.path.join(impression_folder, f"{reference_image_id}_0.png")
                
                if os.path.exists(reference_image_path):
                    from PIL import Image as PILImage
                    reference_image = PILImage.open(reference_image_path)
                    print(f"  ✅ Loaded reference image: {os.path.basename(reference_image_path)}")
                else:
                    print(f"  ⚠️ Reference image not found: {reference_image_id}")
            else:
                print(f"  ⚠️ No reference image ID found in preferences.json")
        
        # Create output directory for sliders
        slider_output_dir = os.path.join(session_folder, "slider", location.replace(" ", "_"))
        os.makedirs(slider_output_dir, exist_ok=True)
        
        # Get or create the backend's SDXL runner (for refinement - we reuse its pipeline)
        global _sdxl_runner, _slider_fuser
        if _sdxl_runner is None:
            print("  Initializing SDXLRunner (backend version for refinement)...")
            _sdxl_runner = SDXLRunner(
                model_id="stabilityai/stable-diffusion-xl-base-1.0",
                device=None,
                height=1024,
                width=1024,
                steps=30,
                guidance_scale=7.5
            )
        
        # Create slider fuser from the SDXL folder (has fuse_with_alpha) - reuses same pipeline
        if _slider_fuser is None and _sdxl_runner.runner.pipe is not None:
            print("  Creating slider fuser (SDXL folder version with alpha support)...")
            _slider_fuser = SDXLEmbedFuserSlider(_sdxl_runner.runner.pipe, device=_sdxl_runner.runner.device)
        
        if _slider_fuser is None:
            raise HTTPException(500, "SDXL pipeline not available for slider generation")
        
        import torch
        #neg_phrases = ["illustration", "anime", "cartoon", "drawing", "painted", "digital art", "concept art", "people", "person", "human", "man", "woman", "face", "body", "portrait"]
        neg_phrases = ["illustration", "painted", "drawing", "cartoon", "anime", "isometric", "diorama", "miniature", "3D render", "CGI", "concept art", "stylized", "toon shading", "people", "person", "human"]
        seed_base = 42
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_sliders = []
        
        # ========== SLIDER 1: Current slider (descriptor → refinement weights) ==========
        print(f"\n{'='*80}")
        print(f"[SLIDER 1] Current slider: descriptor → refinement weights")
        print(f"{'='*80}")
        
        alphas_current = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # Get top-K concepts for refinement
        top_k = 10
        sorted_indices = np.argsort(w_refinement_norm)[::-1]
        actual_top_k = min(top_k, len(concepts_refinement))
        top_indices = sorted_indices[:actual_top_k]
        
        tag_phrases = [concepts_refinement[idx]['label'] for idx in top_indices]
        tag_weights = np.array([float(w_refinement_norm[idx]) for idx in top_indices])
        
        print(f"  Generating {len(alphas_current)} images with alpha interpolation...")
        print(f"  Descriptor: '{descriptor}'")
        print(f"  Top concepts: {tag_phrases[:3]}")
        
        results_current = []
        prompt_embeds_alpha_1 = None
        
        for i, alpha in enumerate(alphas_current):
            print(f"\n  [{i+1}/{len(alphas_current)}] Alpha = {alpha:.2f}")
            
            prompt_embeds, pooled, neg_embeds, neg_pooled = _slider_fuser.fuse_with_alpha(
                descriptor=descriptor,
                tag_phrases=tag_phrases,
                tag_weights=tag_weights,
                alpha=alpha,
                neg_phrases=neg_phrases,
                max_negatives=20
            )
            
            if alpha == 1.0:
                prompt_embeds_alpha_1 = (prompt_embeds, pooled, neg_embeds, neg_pooled)
            
            generator = torch.Generator(device=_sdxl_runner.runner.device).manual_seed(seed_base + i)
            
            image = _sdxl_runner.runner.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                pooled_prompt_embeds=pooled,
                negative_pooled_prompt_embeds=neg_pooled,
                height=1024,
                width=1024,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator
            ).images[0]
            
            results_current.append((alpha, image, None))
        
        # Generate 6th image with reference (alpha=1.0, img2img) - only for original location
        if reference_image is not None and is_original_location and prompt_embeds_alpha_1 is not None:
            print(f"\n  [6/6] Alpha = 1.0 (with reference image, img2img)")
            
            prompt_embeds_ref, pooled_ref, neg_embeds_ref, neg_pooled_ref = prompt_embeds_alpha_1
            
            from backend.sdxl_config import get_stage_strength
            strength = get_stage_strength('impression')
            
            print(f"  Using img2img mode with strength={strength}")
            
            generator = torch.Generator(device=_sdxl_runner.runner.device).manual_seed(seed_base + 5)
            
            image_6 = _sdxl_runner.runner.generate_embeds_img2img(
                init_image=reference_image,
                strength=strength,
                prompt_embeds=prompt_embeds_ref,
                negative_prompt_embeds=neg_embeds_ref,
                pooled_prompt_embeds=pooled_ref,
                negative_pooled_prompt_embeds=neg_pooled_ref,
                seed=seed_base + 5,
                steps=30,
                gscale=7.5,
                height=1024,
                width=1024
            )
            
            results_current.append((1.0, image_6, "ref"))
            print(f"  ✅ Generated 6th image with reference")
        
        # Save current slider images
        images_current = []
        for alpha, img, ref_flag in results_current:
            if ref_flag == "ref":
                filename = f"current_alphaRef_{alpha:.2f}_{timestamp}.png"
            else:
                filename = f"current_alpha_{alpha:.2f}_{timestamp}.png"
            filepath = os.path.join(slider_output_dir, filename)
            img.save(filepath)
            
            rel_path = os.path.relpath(filepath, "sessions")
            url = f"/sessions/{rel_path}"
            images_current.append(SliderImageItem(alpha=alpha, url=url))
            print(f"  Saved: {filepath}")
        
        all_sliders.append(SliderData(
            slider_type="current",
            adjective=adjective,
            location=location,
            descriptor=descriptor,
            images=images_current
        ))
        
        # ========== SLIDER 2: Descriptor → exploration weights ==========
        print(f"\n{'='*80}")
        print(f"[SLIDER 2] Descriptor → exploration concept_weights")
        print(f"{'='*80}")
        
        alphas_exploration = [0.0, 0.25, 0.5, 0.75]
        
        # Get top-K concepts for exploration
        sorted_indices_expl = np.argsort(w_exploration_norm)[::-1]
        actual_top_k_expl = min(top_k, len(concepts_exploration))
        top_indices_expl = sorted_indices_expl[:actual_top_k_expl]
        
        tag_phrases_expl = [concepts_exploration[idx]['label'] for idx in top_indices_expl]
        tag_weights_expl = np.array([float(w_exploration_norm[idx]) for idx in top_indices_expl])
        
        print(f"  Generating {len(alphas_exploration)} images with alpha interpolation...")
        print(f"  Descriptor: '{descriptor}'")
        print(f"  Top concepts: {tag_phrases_expl[:3]}")
        
        results_exploration = []
        
        for i, alpha in enumerate(alphas_exploration):
            print(f"\n  [{i+1}/{len(alphas_exploration)}] Alpha = {alpha:.2f}")
            
            prompt_embeds, pooled, neg_embeds, neg_pooled = _slider_fuser.fuse_with_alpha(
                descriptor=descriptor,
                tag_phrases=tag_phrases_expl,
                tag_weights=tag_weights_expl,
                alpha=alpha,
                neg_phrases=neg_phrases,
                max_negatives=20
            )
            
            generator = torch.Generator(device=_sdxl_runner.runner.device).manual_seed(seed_base + 100 + i)
            
            image = _sdxl_runner.runner.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                pooled_prompt_embeds=pooled,
                negative_pooled_prompt_embeds=neg_pooled,
                height=1024,
                width=1024,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator
            ).images[0]
            
            results_exploration.append((alpha, image, None))
        
        # Save exploration slider images
        images_exploration = []
        for alpha, img, ref_flag in results_exploration:
            filename = f"exploration_alpha_{alpha:.2f}_{timestamp}.png"
            filepath = os.path.join(slider_output_dir, filename)
            img.save(filepath)
            
            rel_path = os.path.relpath(filepath, "sessions")
            url = f"/sessions/{rel_path}"
            images_exploration.append(SliderImageItem(alpha=alpha, url=url))
            print(f"  Saved: {filepath}")
        
        all_sliders.append(SliderData(
            slider_type="exploration",
            adjective=adjective,
            location=location,
            descriptor=descriptor,
            images=images_exploration
        ))
        
        # ========== SLIDER 3: Exploration weights → refinement weights ==========
        print(f"\n{'='*80}")
        print(f"[SLIDER 3] Exploration weights → refinement weights")
        print(f"{'='*80}")
        
        alphas_refinement = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # For this slider, we interpolate between two weight vectors
        # At alpha=0: use exploration weights
        # At alpha=1: use refinement weights
        # We'll use the same top-K concepts from refinement
        
        print(f"  Generating {len(alphas_refinement)} images with weight interpolation...")
        print(f"  Descriptor: '{descriptor}'")
        print(f"  Interpolating between exploration and refinement weights")
        
        results_refinement = []
        
        for i, alpha in enumerate(alphas_refinement):
            print(f"\n  [{i+1}/{len(alphas_refinement)}] Alpha = {alpha:.2f}")
            
            # Interpolate weights: w = (1-alpha) * w_exploration + alpha * w_refinement
            w_interpolated = (1 - alpha) * w_exploration_norm + alpha * w_refinement_norm
            w_interpolated = w_interpolated / (w_interpolated.sum() + 1e-8)  # Renormalize
            
            # Get top-K concepts using interpolated weights
            sorted_indices_interp = np.argsort(w_interpolated)[::-1]
            actual_top_k_interp = min(top_k, len(concepts_refinement))
            top_indices_interp = sorted_indices_interp[:actual_top_k_interp]
            
            tag_phrases_interp = [concepts_refinement[idx]['label'] for idx in top_indices_interp]
            tag_weights_interp = np.array([float(w_interpolated[idx]) for idx in top_indices_interp])
            
            # For weight interpolation slider, we use alpha=1.0 to fully apply the interpolated weights
            prompt_embeds, pooled, neg_embeds, neg_pooled = _slider_fuser.fuse_with_alpha(
                descriptor=descriptor,
                tag_phrases=tag_phrases_interp,
                tag_weights=tag_weights_interp,
                alpha=1.0,  # Always use full strength for interpolated weights
                neg_phrases=neg_phrases,
                max_negatives=20
            )
            
            generator = torch.Generator(device=_sdxl_runner.runner.device).manual_seed(seed_base + 200 + i)
            
            image = _sdxl_runner.runner.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                pooled_prompt_embeds=pooled,
                negative_pooled_prompt_embeds=neg_pooled,
                height=1024,
                width=1024,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator
            ).images[0]
            
            results_refinement.append((alpha, image, None))
        
        # Save refinement slider images
        images_refinement = []
        for alpha, img, ref_flag in results_refinement:
            filename = f"refinement_alpha_{alpha:.2f}_{timestamp}.png"
            filepath = os.path.join(slider_output_dir, filename)
            img.save(filepath)
            
            rel_path = os.path.relpath(filepath, "sessions")
            url = f"/sessions/{rel_path}"
            images_refinement.append(SliderImageItem(alpha=alpha, url=url))
            print(f"  Saved: {filepath}")
        
        all_sliders.append(SliderData(
            slider_type="refinement",
            adjective=adjective,
            location=location,
            descriptor=descriptor,
            images=images_refinement
        ))
        
        print(f"\n[SLIDER GENERATION] ✅ Generated {len(all_sliders)} sliders")
        print(f"  - Current slider: {len(images_current)} images")
        print(f"  - Exploration slider: {len(images_exploration)} images")
        print(f"  - Refinement slider: {len(images_refinement)} images")
        print("=" * 80 + "\n")
        
        return SliderGenerateResponse(
            success=True,
            sliders=all_sliders
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SLIDER GENERATION] Error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))