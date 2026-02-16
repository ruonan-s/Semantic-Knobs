"""
Evaluation Prototype Server

A lightweight server that extends the main backend server with eval-specific
endpoints for the evaluation prototype that skips refinement.
"""

import os
import sys
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add backend to path to import from main server
BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
PROJECT_ROOT = BACKEND_DIR.parent
SDXL_DIR = PROJECT_ROOT / "SDXL"

# Add paths for imports
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SDXL_DIR))  # For SDXL internal imports like diffusion_runner

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
import random
import glob as glob_module
from datetime import datetime

# Import eval utilities
from eval_utils import (
    generate_final_selection_from_exploration,
    copy_predefined_session,
    list_predefined_sessions,
    validate_session_for_eval,
    create_eval_session_log
)

# Add utils path for histogram generation
EVAL_UTILS_DIR = Path(__file__).parent.parent / "utils"
sys.path.insert(0, str(EVAL_UTILS_DIR))
from Histograms import generate_histograms

# Import from main backend
from concept_refinement import get_or_create_session as get_refinement_session, refinement_sessions

# Import GP exploration system
from gp_session import get_or_create_gp_session, gp_exploration_sessions, GPExplorationSession

# Import HITL refinement system (V1 - composition-level GP)
from hitl_session import HITLRefinementSession

# Import HITL refinement system V2 (tag-level GP) - NEW
from hitl_session_v2 import HITLRefinementSessionV2

# Import Slot-based refinement system
from Refinement_steps import SlotRefinementSession, SlotRefinementConfig

# Import Tag GP Refinement system
from tag_gp_refiner import TagGPRefiner, GPRefinerConfig

# ============== Mode Toggle ==============
# Set to True to use GP-based preference learning, False for original softmax approach
USE_GP_EXPLORATION = True

# Import SD baselines for baseline image generation
LLM_SCRIPTS_PATH = Path(__file__).parent.parent / "llm_scripts"
sys.path.insert(0, str(LLM_SCRIPTS_PATH))
from sd_baselines import (
    generate_sd_text_baseline,
    generate_sd_img2img_baseline,
    set_runner as set_sd_baseline_runner
)

# Create FastAPI app for eval
app = FastAPI(title="Semantic Knobs Eval Prototype")

# Thread pool for running blocking image generation without blocking the event loop
# max_workers=1 ensures GPU operations don't conflict, but event loop stays responsive
generation_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="img_gen")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000"],  # Eval frontend on 3001
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
EVAL_DIR = Path(__file__).parent.parent
PREDEFINED_INPUT_DIR = EVAL_DIR / "predefined_input"
SESSION_LOGS_DIR = EVAL_DIR / "session_logs"
BACKEND_SESSIONS_DIR = BACKEND_DIR / "sessions"
LLM_SCRIPTS_DIR = EVAL_DIR / "llm_scripts"  # Parent of all baseline_generic_* folders

# Ensure directories exist
os.makedirs(PREDEFINED_INPUT_DIR, exist_ok=True)
os.makedirs(SESSION_LOGS_DIR, exist_ok=True)

# Mount static files for serving images
app.mount("/predefined", StaticFiles(directory=str(PREDEFINED_INPUT_DIR)), name="predefined")
app.mount("/session_logs", StaticFiles(directory=str(SESSION_LOGS_DIR)), name="session_logs")
app.mount("/llm_scripts", StaticFiles(directory=str(LLM_SCRIPTS_DIR)), name="llm_scripts")

# Also mount backend sessions for compatibility
if BACKEND_SESSIONS_DIR.exists():
    app.mount("/sessions", StaticFiles(directory=str(BACKEND_SESSIONS_DIR)), name="sessions")

# In-memory session store for eval
eval_sessions = {}


# ============== Models ==============

class PredefinedSessionInfo(BaseModel):
    name: str
    path: str
    has_impression: bool
    has_images: bool
    adjective: str
    location: str
    descriptor: str
    valid: bool


class ListPredefinedResponse(BaseModel):
    sessions: List[PredefinedSessionInfo]


class LoadSessionRequest(BaseModel):
    session_name: str
    user_id: Optional[str] = None


class LoadSessionResponse(BaseModel):
    success: bool
    session_id: str
    session_folder: str
    descriptor: str
    adjective: str
    location: str
    images: List[dict]


class SkipToSliderRequest(BaseModel):
    session_id: str
    selected_image_id: Optional[str] = None


class SkipToSliderResponse(BaseModel):
    success: bool
    message: str
    next_stage: str


class ImageItem(BaseModel):
    id: str
    url: str


class ConceptWeight(BaseModel):
    concept_id: str
    label: str
    weight: float


class EvalStatusResponse(BaseModel):
    session_id: str
    stage: str
    concepts_initialized: bool
    concept_count: int


class ManualTagPoolRequest(BaseModel):
    session_id: str
    stage: str = "impression"


class SaveManualWeightsRequest(BaseModel):
    session_id: str
    selected_tags: List[str]
    weights: Dict[str, float]
    selected_image_id: Optional[str] = None


# ============== Endpoints ==============

@app.get("/api/eval/predefined-sessions", response_model=ListPredefinedResponse)
def list_predefined():
    """List all available predefined sessions for evaluation."""
    sessions = list_predefined_sessions(str(PREDEFINED_INPUT_DIR))
    return ListPredefinedResponse(
        sessions=[PredefinedSessionInfo(**s) for s in sessions]
    )


@app.post("/api/eval/load-session", response_model=LoadSessionResponse)
def load_session(request: LoadSessionRequest):
    """
    Load a predefined session for evaluation.
    
    Copies the predefined session to session_logs and initializes it.
    """
    try:
        predefined_folder = PREDEFINED_INPUT_DIR / request.session_name
        
        if not predefined_folder.exists():
            raise HTTPException(404, f"Predefined session not found: {request.session_name}")
        
        # Validate session
        validation = validate_session_for_eval(str(predefined_folder))
        if not validation["valid"]:
            missing_items = ", ".join([m["description"] for m in validation["missing"]])
            raise HTTPException(400, f"Invalid session - missing: {missing_items}")
        
        # Copy to session_logs
        new_session_folder = copy_predefined_session(
            str(predefined_folder),
            str(SESSION_LOGS_DIR),
            request.user_id
        )
        
        session_id = os.path.basename(new_session_folder)
        
        # Load session metadata
        final_selection_path = os.path.join(new_session_folder, "final_selection.json")
        with open(final_selection_path, 'r') as f:
            final_selection = json.load(f)
        
        adjective = final_selection.get("adjective", "")
        location = final_selection.get("location", "")
        descriptor = final_selection.get("descriptor", f"{adjective} {location}".strip())
        
        # Load images from impression folder
        impression_folder = os.path.join(new_session_folder, "impression")
        images = []
        if os.path.exists(impression_folder):
            for filename in sorted(os.listdir(impression_folder)):
                if filename.endswith('.png'):
                    image_id = filename.replace('.png', '')
                    images.append({
                        "id": image_id,
                        "url": f"/session_logs/{session_id}/impression/{filename}"
                    })
        
        # Store in memory
        eval_sessions[session_id] = {
            "folder": new_session_folder,
            "descriptor": descriptor,
            "adjective": adjective,
            "location": location,
            "user_pref": {},
            "user_id": request.user_id
        }
        
        # Log session start
        create_eval_session_log(
            new_session_folder,
            request.user_id or "anonymous",
            "session_start",
            {"predefined_session": request.session_name}
        )
        
        print(f"[EVAL] Loaded session: {session_id}")
        print(f"  Descriptor: {descriptor}")
        print(f"  Images: {len(images)}")
        
        return LoadSessionResponse(
            success=True,
            session_id=session_id,
            session_folder=new_session_folder,
            descriptor=descriptor,
            adjective=adjective,
            location=location,
            images=images
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[EVAL] Error loading session: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/eval/skip-to-slider", response_model=SkipToSliderResponse)
def skip_to_slider(request: SkipToSliderRequest):
    """
    Skip refinement and go directly to slider generation.
    
    Uses exploration concept_weights.json to generate final_selection.json,
    then returns signal to navigate to slider generation stage.
    """
    try:
        session = eval_sessions.get(request.session_id)
        if not session:
            raise HTTPException(404, f"Session not found: {request.session_id}")
        
        session_folder = session["folder"]
        use_gp = session.get("use_gp", USE_GP_EXPLORATION)
        
        # Save concept weights before generating final_selection
        key = f"{request.session_id}_impression"
        
        if use_gp and key in gp_exploration_sessions:
            # IMPORTANT: Trigger GP fitting before saving weights
            # This trains the GP on all accumulated tag preferences
            gp_session = gp_exploration_sessions[key]
            print(f"[EVAL] Triggering GP fitting for {request.session_id}...")
            gp_session.process_round_manually()
            
            # Now save the trained weights
            gp_session.save_raw_tag_weights(session_folder)
            print(f"[EVAL] Saved GP raw tag weights (top-10 with dedup) for {request.session_id}")
            
            # Save tag preferences (positive/negative/neutral) for preferences baseline
            tag_prefs = gp_session.get_tag_preferences()
            positive_tags = []
            negative_tags = []
            neutral_tags = []
            
            for tag_id, pref in tag_prefs.items():
                if tag_id in gp_session.raw_tags:
                    tag_text = gp_session.raw_tags[tag_id].text
                    if pref == 'positive':
                        positive_tags.append(tag_text)
                    elif pref == 'negative':
                        negative_tags.append(tag_text)
                    else:
                        neutral_tags.append(tag_text)
            
            tag_preferences_data = {
                "positive": positive_tags,
                "negative": negative_tags,
                "neutral": neutral_tags
            }
            
            tag_prefs_path = os.path.join(session_folder, "impression", "tag_preferences.json")
            with open(tag_prefs_path, 'w') as f:
                json.dump(tag_preferences_data, f, indent=2)
            print(f"[EVAL] Saved tag preferences: {len(positive_tags)} positive, {len(negative_tags)} negative, {len(neutral_tags)} neutral")
        elif key in refinement_sessions:
            # Save original refinement weights
            refinement_session = refinement_sessions[key]
            refinement_session.save_concept_weights(session_folder)
            print(f"[EVAL] Saved exploration concept weights for {request.session_id}")
        
        # Generate final_selection.json from exploration weights
        final_selection = generate_final_selection_from_exploration(
            session_folder,
            exploration_stage="impression"
        )
        
        # Update preferences.json with selection if provided
        if request.selected_image_id:
            preferences_path = os.path.join(session_folder, "preferences.json")
            preferences = {}
            if os.path.exists(preferences_path):
                with open(preferences_path, 'r') as f:
                    preferences = json.load(f)
            
            preferences["selections"] = preferences.get("selections", {})
            preferences["selections"]["impression"] = request.selected_image_id
            preferences["descriptor"] = session.get("descriptor", "")
            
            with open(preferences_path, 'w') as f:
                json.dump(preferences, f, indent=2)
        
        # Log event
        create_eval_session_log(
            session_folder,
            session.get("user_id", "anonymous"),
            "skip_to_slider",
            {
                "selected_image_id": request.selected_image_id,
                "concept_count": final_selection.get("summary", {}).get("total_concepts", 0),
                "gp_mode": use_gp
            }
        )
        
        mode_str = "GP" if use_gp else "original"
        print(f"[EVAL] Skipping refinement for session: {request.session_id}")
        print(f"  Using {mode_str} exploration weights directly for slider generation")
        
        return SkipToSliderResponse(
            success=True,
            message=f"Generated final_selection from {mode_str} exploration weights. Ready for slider generation.",
            next_stage="slider_generation"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[EVAL] Error in skip_to_slider: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/eval/status/{session_id}")
def get_eval_status(session_id: str):
    """Get the status of an evaluation session."""
    session = eval_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {session_id}")
    
    use_gp = session.get("use_gp", USE_GP_EXPLORATION)
    key = f"{session_id}_impression"
    
    # Check concept system state
    if use_gp and key in gp_exploration_sessions:
        gp_session = gp_exploration_sessions[key]
        concepts_initialized = gp_session.initialized
        concept_count = len(gp_session.get_concepts())
    elif key in refinement_sessions:
        concepts_initialized = True
        concept_count = len(refinement_sessions[key].concepts)
    else:
        concepts_initialized = False
        concept_count = 0
    
    return {
        "session_id": session_id,
        "stage": "impression",
        "concepts_initialized": concepts_initialized,
        "concept_count": concept_count,
        "gp_mode": use_gp
    }


def _extract_positive_tag_pool(session_id: str, session_folder: str, stage: str = "impression") -> List[str]:
    """
    Build manual-tag pool from explicit positive clicks, with concept_weights fallback.
    """
    tags: List[str] = []
    key = f"{session_id}_{stage}"

    if key in gp_exploration_sessions:
        gp_session = gp_exploration_sessions[key]
        tag_prefs = gp_session.get_tag_preferences()
        for tag_id, pref in tag_prefs.items():
            if pref == "positive" and tag_id in gp_session.raw_tags:
                text = gp_session.raw_tags[tag_id].text.strip()
                if text:
                    tags.append(text)

    # Fallback for older sessions / no explicit positive clicks
    if not tags:
        concept_weights_path = Path(session_folder) / stage / "concept_weights.json"
        if concept_weights_path.exists():
            with open(concept_weights_path, "r") as f:
                cw_data = json.load(f)
            for cw in cw_data.get("concept_weights", []):
                label = str(cw.get("label", "")).strip()
                category = str(cw.get("category", "")).lower()
                if label and (not category or category == "positive"):
                    tags.append(label)

    # Deduplicate while preserving order
    deduped: List[str] = []
    seen = set()
    for tag in tags:
        key_tag = tag.lower()
        if key_tag in seen:
            continue
        seen.add(key_tag)
        deduped.append(tag)

    return deduped


@app.post("/api/eval/manual-tag-pool")
def get_manual_tag_pool(request: ManualTagPoolRequest):
    """Return positive tag pool for manual user customization stage."""
    session = eval_sessions.get(request.session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {request.session_id}")

    pool = _extract_positive_tag_pool(
        session_id=request.session_id,
        session_folder=session["folder"],
        stage=request.stage
    )

    return {"success": True, "tags": pool, "count": len(pool)}


@app.post("/api/eval/save-manual-weights")
def save_manual_weights(request: SaveManualWeightsRequest):
    """Persist manual 10-tag selection and weights for user_customized baseline."""
    session = eval_sessions.get(request.session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {request.session_id}")

    selected_tags = [str(t).strip() for t in request.selected_tags if str(t).strip()]
    if len(selected_tags) != 10:
        raise HTTPException(400, "Exactly 10 selected tags are required")

    if len(set(tag.lower() for tag in selected_tags)) != 10:
        raise HTTPException(400, "Selected tags must be unique")

    validated_weights: Dict[str, float] = {}
    for tag in selected_tags:
        if tag not in request.weights:
            raise HTTPException(400, f"Missing weight for tag: {tag}")
        value = request.weights[tag]
        if not isinstance(value, (int, float)):
            raise HTTPException(400, f"Weight must be numeric for tag: {tag}")
        w = float(value)
        if w < 0.0 or w > 1.0:
            raise HTTPException(400, f"Weight out of range [0,1] for tag: {tag}")
        validated_weights[tag] = w

    stage_dir = Path(session["folder"]) / "impression"
    stage_dir.mkdir(parents=True, exist_ok=True)
    out_path = stage_dir / "user_manual_weights.json"

    payload = {
        "stage": "impression",
        "session_id": request.session_id,
        "timestamp": datetime.now().isoformat(),
        "selected_image_id": request.selected_image_id,
        "selected_tags": selected_tags,
        "weights": validated_weights
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    create_eval_session_log(
        session["folder"],
        session.get("user_id", "anonymous"),
        "save_manual_weights",
        {
            "selected_image_id": request.selected_image_id,
            "tag_count": len(selected_tags),
            "weights_file": str(out_path)
        }
    )

    return {"success": True, "path": str(out_path), "tag_count": len(selected_tags)}


# ============== Proxy endpoints to main backend ==============
# These endpoints proxy to the main backend functionality

@app.post("/api/concepts/init")
async def proxy_concepts_init(request: dict):
    """Initialize concepts - proxied from main backend or GP system."""
    session_id = request.get("session_id")
    stage = request.get("stage", "impression")
    image_ids = request.get("image_ids", [])
    use_gp = request.get("use_gp", USE_GP_EXPLORATION)  # Can override via request
    
    session = eval_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {session_id}")
    
    session_folder = session["folder"]
    stage_folder = os.path.join(session_folder, stage)
    visual_tags_path = os.path.join(stage_folder, "visual_tags.json")
    
    if not os.path.exists(visual_tags_path):
        return {"success": False, "concepts": [], "categorized": {}, "image_effects": {}, "incidence_matrix": {}, "tag_preferences": {}}
    
    with open(visual_tags_path, 'r') as f:
        visual_tags_data = json.load(f)
    
    # Build image_tags dict
    image_tags = {}
    for image_id in image_ids:
        filename = f"{image_id}.png"
        if filename in visual_tags_data:
            tags = visual_tags_data[filename]
            if isinstance(tags, list):
                image_tags[image_id] = tags
            elif isinstance(tags, dict):
                flat_tags = []
                for category in ["overall_style", "material_world", "ambient_medium", "visual_elements"]:
                    flat_tags.extend(tags.get(category, []))
                image_tags[image_id] = flat_tags
    
    if use_gp:
        # Use GP-based exploration
        print(f"[EVAL] Using GP exploration mode for session {session_id}")
        gp_session = get_or_create_gp_session(session_id, stage, image_ids)
        
        if not gp_session.initialized:
            gp_session.initialize_from_tags(image_tags)
            # Save initial weights
            gp_session.save_raw_tag_weights(session_folder)
        
        state_dict = gp_session.to_dict()
        
        # Store GP mode flag in session
        session["use_gp"] = True
    else:
        # Use original refinement session
        print(f"[EVAL] Using original refinement mode for session {session_id}")
        from concept_refinement import (
            build_concepts, compute_weights, RawTag, Concept, ConceptState
        )
        
        key = f"{session_id}_{stage}"
        refinement_session = get_refinement_session(session_id, stage, image_ids)
        
        if not refinement_session.initialized:
            refinement_session.initialize_from_tags(image_tags)
            # Save initial weights
            refinement_session.save_concept_weights(session_folder)
        
        state_dict = refinement_session.to_dict()
        
        # Store GP mode flag in session
        session["use_gp"] = False
    
    return {
        "success": True,
        "concepts": state_dict["concepts"],
        "categorized": state_dict["categorized"],
        "image_effects": state_dict["image_effects"],
        "incidence_matrix": state_dict.get("incidence_matrix", {}),
        "tag_preferences": state_dict.get("tag_preferences", {}),
        "gp_mode": use_gp
    }


@app.post("/api/concepts/interact")
async def proxy_concepts_interact(request: dict):
    """Handle tag interaction - proxied from main backend or GP system."""
    session_id = request.get("session_id")
    stage = request.get("stage", "impression")
    tag_id = request.get("tag_id")
    preference = request.get("preference")  # 'positive' or 'negative'
    
    session = eval_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {session_id}")
    
    key = f"{session_id}_{stage}"
    use_gp = session.get("use_gp", USE_GP_EXPLORATION)
    
    if use_gp and key in gp_exploration_sessions:
        # Use GP session
        gp_session = gp_exploration_sessions[key]
        gp_session.handle_tag_click(tag_id, preference)
        
        # Note: GP doesn't refit on every tag click for performance
        # Refitting happens on image selection or explicit process_round
        
        # Save updated weights
        gp_session.save_raw_tag_weights(session["folder"])
        
        state_dict = gp_session.to_dict()
    else:
        # Use original refinement session
        if key not in refinement_sessions:
            raise HTTPException(404, "Refinement session not found")
        
        refinement_session = refinement_sessions[key]
        refinement_session.handle_tag_click(tag_id, preference)
        
        # Save updated weights
        refinement_session.save_concept_weights(session["folder"])
        
        state_dict = refinement_session.to_dict()
    
    return {
        "success": True,
        "concepts": state_dict["concepts"],
        "categorized": state_dict["categorized"],
        "image_effects": state_dict["image_effects"],
        "tag_preferences": state_dict.get("tag_preferences", {}),
        "gp_mode": use_gp
    }


@app.post("/api/concepts/select-image")
async def proxy_select_image(request: dict):
    """Handle image selection - proxied from main backend or GP system."""
    session_id = request.get("session_id")
    stage = request.get("stage", "impression")
    image_id = request.get("image_id")
    boost_amount = request.get("boost_amount", 0.5)
    
    session = eval_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {session_id}")
    
    key = f"{session_id}_{stage}"
    use_gp = session.get("use_gp", USE_GP_EXPLORATION)
    
    if use_gp and key in gp_exploration_sessions:
        # Use GP session - this triggers GP fitting
        gp_session = gp_exploration_sessions[key]
        gp_session.handle_image_selection(image_id, boost_amount)
        
        # Save updated weights
        gp_session.save_raw_tag_weights(session["folder"])
        
        state_dict = gp_session.to_dict()
    else:
        # Use original refinement session
        if key not in refinement_sessions:
            raise HTTPException(404, "Refinement session not found")
        
        refinement_session = refinement_sessions[key]
        refinement_session.handle_image_selection(image_id, boost_amount)
        
        # Save updated weights
        refinement_session.save_concept_weights(session["folder"])
        
        state_dict = refinement_session.to_dict()
    
    return {
        "success": True,
        "concepts": state_dict["concepts"],
        "categorized": state_dict["categorized"],
        "image_effects": state_dict["image_effects"],
        "tag_preferences": state_dict.get("tag_preferences", {}),
        "gp_mode": use_gp
    }


@app.post("/api/tags")
async def proxy_get_tags(request: dict):
    """Get tags for an image - proxied from main backend."""
    session_id = request.get("session_id")
    stage = request.get("stage", "impression")
    image_id = request.get("image_id")
    
    session = eval_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {session_id}")
    
    session_folder = session["folder"]
    stage_folder = os.path.join(session_folder, stage)
    visual_tags_path = os.path.join(stage_folder, "visual_tags.json")
    
    if not os.path.exists(visual_tags_path):
        return {"tags": []}
    
    with open(visual_tags_path, 'r') as f:
        visual_tags_data = json.load(f)
    
    filename = f"{image_id}.png"
    if filename in visual_tags_data:
        tags = visual_tags_data[filename]
        if isinstance(tags, list):
            return {"tags": tags}
        elif isinstance(tags, dict):
            flat_tags = []
            for category in ["overall_style", "material_world", "ambient_medium", "visual_elements"]:
                flat_tags.extend(tags.get(category, []))
            return {"tags": flat_tags}
    
    return {"tags": []}


def _generate_hitl_v2_image(
    session_folder: str,
    adjective: str,
    target_location: str,
    output_path: str,
    seed: int = 2026,
    source_mode: str = "ours"
) -> bool:
    """
    Generate an image with the same HITL V2 attention-control pipeline used for "ours".
    Source can be:
    - "ours": refined_preferences_v2.json
    - "user_customized": impression/user_manual_weights.json
    
    Args:
        session_folder: Path to session folder
        adjective: Style adjective (e.g., "Calm")
        target_location: Target location (e.g., "Home Office", "Bedroom")
        output_path: Where to save the generated image
        seed: Random seed for generation
        source_mode: "ours" or "user_customized"
        
    Returns:
        True if generation succeeded, False otherwise
    """
    import numpy as np
    
    from hitl_fuser import HITLCompositionFuser, generate_with_hooks
    from hitl_sampler import CompositionSample
    
    if source_mode == "ours":
        prefs_path = os.path.join(session_folder, "refined_preferences_v2.json")
        if not os.path.exists(prefs_path):
            print(f"[HITL V2 EVAL] refined_preferences_v2.json not found at {prefs_path}")
            return False
        with open(prefs_path, "r") as f:
            refined_data = json.load(f)
        final_tags = refined_data.get("tags", [])
        weights_dict = refined_data.get("weights", {})
        if not final_tags:
            print("[HITL V2 EVAL] No tags in refined_preferences_v2.json")
            return False
        tag_labels = list(final_tags)
        tag_weights = [float(weights_dict.get(tag, 0.1)) for tag in tag_labels]
    elif source_mode == "user_customized":
        manual_path = os.path.join(session_folder, "impression", "user_manual_weights.json")
        if not os.path.exists(manual_path):
            print(f"[HITL V2 EVAL] user_manual_weights.json not found at {manual_path}")
            return False
        with open(manual_path, "r") as f:
            manual_data = json.load(f)
        selected_tags = manual_data.get("selected_tags", [])
        weights_dict = manual_data.get("weights", {})
        if not selected_tags:
            print("[HITL V2 EVAL] No selected_tags in user_manual_weights.json")
            return False
        tag_labels = list(selected_tags)
        tag_weights = [float(weights_dict.get(tag, 0.1)) for tag in tag_labels]
    else:
        print(f"[HITL V2 EVAL] Unknown source_mode: {source_mode}")
        return False
    
    # Compute cross-attention weights using same method as refinement stage
    # (softmax scaled to range around 1.0)
    mus = np.array(tag_weights)
    exp_mu = np.exp(mus - np.max(mus))
    softmax = exp_mu / (exp_mu.sum() + 1e-8)
    attn_weights = 0.5 + softmax * len(mus)
    attn_weights = attn_weights * (len(mus) / attn_weights.sum())
    
    print(f"\n{'='*80}")
    print(f"[HITL V2 EVAL] Generating with refinement-stage method ({source_mode})")
    print(f"  Adjective: {adjective}")
    print(f"  Location: {target_location}")
    print(f"  Tags ({len(tag_labels)}): {tag_labels[:5]}...")
    print(f"  Attention weights: {[f'{w:.3f}' for w in attn_weights[:5]]}...")
    print(f"{'='*80}")
    
    # Build base prompt: "{adjective} {location}" — same as refinement stage
    base_prompt = f"{adjective} {target_location}"
    neg_phrases = [
        "illustration", "plan view", "bird's-eye view",
        "cartoon", "anime", "isometric", "sketch",
        "low quality", "blurry", "text", "watermark", "human"
    ]
    
    # Get or create SDXL runner (reuse global)
    global _eval_sdxl_runner
    if '_eval_sdxl_runner' not in globals() or _eval_sdxl_runner is None:
        from SDXL.sdxl_runner import SDXLRunner
        print("  Initializing SDXL runner...")
        _eval_sdxl_runner = SDXLRunner(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            device=None,
            height=1024,
            width=1024,
            steps=30,
            guidance_scale=7.5
        )
    
    pipe = _eval_sdxl_runner.runner.pipe
    device = _eval_sdxl_runner.runner.device
    
    # Create fuser — same as refinement stage
    hitl_fuser = HITLCompositionFuser(pipe=pipe, device=device)
    
    # Create CompositionSample — identical to hitl_session_v2._generate_images
    comp_sample = CompositionSample(
        points=np.zeros((len(tag_labels), 768)),  # Placeholder - not used in generation
        weights=np.array(attn_weights),
        tag_labels=tag_labels,
        tag_indices=list(range(len(tag_labels))),
        point_ucb_scores=np.array(tag_weights),
    )
    
    # Fuse composition — identical to refinement stage
    prompt_embeds, pooled, neg_embeds, neg_pooled, attn_controller = hitl_fuser.fuse_composition(
        comp_sample,
        base_prompt=base_prompt,
        neg_phrases=neg_phrases
    )
    
    # --- Boost location tokens in attention map ---
    # Tags learned from the original location (e.g., Living Room) can dominate 
    # cross-attention when transferred to a new location (e.g., Bathroom).
    # Boosting the target location tokens ensures the model strongly attends to the
    # correct room type, while tags influence style/aesthetics.
    tags_str = ", ".join(tag_labels)
    full_prompt = f"{base_prompt} with features: {tags_str}"
    
    # Map both the full base_prompt and just the location to their token indices
    concepts_to_boost = [target_location]
    location_map, _ = hitl_fuser._tokenize_and_map(full_prompt, concepts_to_boost)
    
    LOCATION_BOOST = 3.0  # Location tokens get 3x attention (tags max at ~1.5x)
    
    boosted_count = 0
    for concept in concepts_to_boost:
        if concept in location_map:
            for idx in location_map[concept]:
                attn_controller.token_weight_map.weights[idx] = LOCATION_BOOST
                boosted_count += 1
    
    print(f"  Location boost: boosted {boosted_count} tokens for '{target_location}' to {LOCATION_BOOST}x")
    
    # Generate with hooks — identical to refinement stage
    image = generate_with_hooks(
        pipe,
        prompt_embeds, pooled,
        neg_embeds, neg_pooled,
        attn_controller,
        seed=seed
    )
    
    image.save(output_path)
    print(f"  Saved: {os.path.basename(output_path)}")
    
    return True


def _generate_slider_sync(session_id: str, location: str, session: dict):
    """
    Synchronous slider generation - runs in thread pool to avoid blocking event loop.
    
    Uses HITL V2 refined tags/weights for "ours" image, then generates baselines.
    """
    import numpy as np
    import torch
    from datetime import datetime
    
    session_folder = session["folder"]
    
    try:
        # Check for HITL V2 refinement output (ours)
        v2_prefs_path = os.path.join(session_folder, "refined_preferences_v2.json")
        has_v2_refinement = os.path.exists(v2_prefs_path)
        manual_weights_path = os.path.join(session_folder, "impression", "user_manual_weights.json")
        has_manual_weights = os.path.exists(manual_weights_path)
        
        if has_v2_refinement:
            print("\n" + "=" * 80)
            print(f"[EVAL SLIDER] Generating with HITL V2 refinement (fuse_composition + generate_with_hooks)")
            print(f"[EVAL SLIDER] Session: {session_id}")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print(f"[EVAL SLIDER] ERROR: No refined_preferences_v2.json found")
            print(f"[EVAL SLIDER] Session: {session_id}")
            print("=" * 80)
            return {"error": True, "status_code": 404, "message": "refined_preferences_v2.json not found. HITL refinement must be completed first."}

        if not has_manual_weights:
            print("\n" + "=" * 80)
            print(f"[EVAL SLIDER] ERROR: No user_manual_weights.json found")
            print(f"[EVAL SLIDER] Session: {session_id}")
            print("=" * 80)
            return {"error": True, "status_code": 404, "message": "user_manual_weights.json not found. Complete manual tag customization before evaluation."}
        
        # Load final_selection.json for adjective/location
        final_selection_path = os.path.join(session_folder, "final_selection.json")
        if not os.path.exists(final_selection_path):
            return {"error": True, "status_code": 404, "message": "final_selection.json not found"}
        
        with open(final_selection_path, 'r') as f:
            final_selection = json.load(f)
        
        # Get adjective and location
        adjective = final_selection.get("adjective", "")
        target_location = location if location else final_selection.get("location", "")
        descriptor = f"{adjective} {target_location}"
        
        print(f"  Adjective: {adjective}")
        print(f"  Location: {target_location}")
        print(f"  Descriptor: {descriptor}")
        
        # Load preferences.json (needed for style transfer baseline)
        preferences_path = os.path.join(session_folder, "preferences.json")
        prefs = {}
        if os.path.exists(preferences_path):
            with open(preferences_path, 'r') as f:
                prefs = json.load(f)
        
        # Create output directory
        slider_output_dir = os.path.join(session_folder, "slider", target_location.replace(" ", "_"))
        os.makedirs(slider_output_dir, exist_ok=True)
        
        # Import SDXL components (paths already set at module level)
        from SDXL.sdxl_runner import SDXLRunner
        from SDXL.sdxl_embed_fuser import SDXLEmbedFuser as SDXLEmbedFuserSlider
        
        # Get or create SDXL runner (cached globally)
        global _eval_sdxl_runner, _eval_slider_fuser
        
        if '_eval_sdxl_runner' not in globals() or _eval_sdxl_runner is None:
            print("  Initializing SDXL runner...")
            _eval_sdxl_runner = SDXLRunner(
                model_id="stabilityai/stable-diffusion-xl-base-1.0",
                device=None,
                height=1024,
                width=1024,
                steps=30,
                guidance_scale=7.5
            )
        
        if '_eval_slider_fuser' not in globals() or _eval_slider_fuser is None:
            if _eval_sdxl_runner.runner.pipe is not None:
                print("  Creating slider fuser...")
                _eval_slider_fuser = SDXLEmbedFuserSlider(
                    _eval_sdxl_runner.runner.pipe, 
                    device=_eval_sdxl_runner.runner.device
                )
                # Share runner with SD baselines module to avoid loading model twice
                set_sd_baseline_runner(_eval_sdxl_runner.runner)
        
        if _eval_sdxl_runner is None or _eval_sdxl_runner.runner.pipe is None:
            return {"error": True, "status_code": 500, "message": "SDXL pipeline not available"}
        
        # ========== Generate "ours" image using HITL V2 method ==========
        # Same generation as refinement stage, just with target_location swapped in
        seed_base = 2026
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ours_filename = f"eval_alpha_1.00_{timestamp}.png"
        ours_filepath = os.path.join(slider_output_dir, ours_filename)
        
        success = _generate_hitl_v2_image(
            session_folder=session_folder,
            adjective=adjective,
            target_location=target_location,
            output_path=ours_filepath,
            seed=seed_base
        )
        
        if not success:
            return {"error": True, "status_code": 500, "message": "Failed to generate HITL V2 image"}

        # ========== Generate "user_customized" image using same HITL V2 method ==========
        user_customized_path = os.path.join(slider_output_dir, "user_customized.png")
        success_user_customized = _generate_hitl_v2_image(
            session_folder=session_folder,
            adjective=adjective,
            target_location=target_location,
            output_path=user_customized_path,
            seed=seed_base + 1,
            source_mode="user_customized"
        )

        if not success_user_customized:
            return {"error": True, "status_code": 500, "message": "Failed to generate user_customized image"}
        
        # Build response for the "ours" image
        rel_path = os.path.relpath(ours_filepath, SESSION_LOGS_DIR)
        slider_images = [{"alpha": 1.0, "url": f"/session_logs/{rel_path}"}]
        
        # Log success
        create_eval_session_log(
            session_folder,
            session.get("user_id", "anonymous"),
            "slider_generation_complete",
            {"success": True, "image_count": 2, "method": "hitl_v2"}
        )
        
        print(f"\n  ✅ Generated HITL V2 images for eval slider (ours + user_customized)")
        
        # ========== SD BASELINE TEXT (text-only baseline) ==========
        # Generate image using only "{adjective} {location}" as prompt
        # Uses model's native prompt encoding (no custom embedding fusion)
        print(f"\n{'='*80}")
        print(f"[SD BASELINE TEXT] Generating text-only baseline")
        print(f"{'='*80}")
        
        try:
            sd_text_output = os.path.join(slider_output_dir, "sd_baseline_text.png")
            generate_sd_text_baseline(
                adjective=adjective,
                location=target_location,
                output_path=sd_text_output,
                seed=2026
            )
            print(f"  ✅ SD text baseline saved: sd_baseline_text.png")
        except Exception as e:
            print(f"  ⚠️ SD text baseline generation failed: {str(e)}")
        
        # ========== SD STYLE TRANSFER (img2img baseline) ==========
        # Uses exploration selected image as reference for img2img
        # Uses model's native prompt encoding (no custom embedding fusion)
        original_location = final_selection.get("location", "")
        
        print(f"\n{'='*80}")
        print(f"[SD STYLE TRANSFER] Generating img2img baseline")
        print(f"{'='*80}")
        print(f"  Original location: {original_location}")
        print(f"  Target location: {target_location}")
        
        # Use exploration selected image from preferences.json
        reference_image_path = None
        exploration_selection = prefs.get("selections", {}).get("impression")
        
        if exploration_selection:
            # Exploration selected image is in the impression folder
            impression_folder = os.path.join(session_folder, "impression")
            ref_path = os.path.join(impression_folder, f"{exploration_selection}.png")
            
            if os.path.exists(ref_path):
                reference_image_path = ref_path
                print(f"  Using exploration selected image: {exploration_selection}.png")
            else:
                # Try alternate naming pattern
                ref_path = os.path.join(impression_folder, f"{exploration_selection}_0.png")
                if os.path.exists(ref_path):
                    reference_image_path = ref_path
                    print(f"  Using exploration selected image: {exploration_selection}_0.png")
                else:
                    print(f"  ⚠️ Exploration selected image not found: {exploration_selection}")
        else:
            print(f"  ⚠️ No exploration selection found in preferences.json")
        
        if reference_image_path and os.path.exists(reference_image_path):
            print(f"  Reference image: {os.path.basename(reference_image_path)}")
            
            # Output path for SD img2img baseline
            sd_img2img_output = os.path.join(slider_output_dir, "sd_style_transfer.png")
            
            try:
                generate_sd_img2img_baseline(
                    input_image_path=reference_image_path,
                    adjective=adjective,
                    original_location=original_location,
                    target_location=target_location,
                    output_path=sd_img2img_output,
                    seed=2026
                )
                print(f"  ✅ SD img2img baseline saved: sd_style_transfer.png")
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"  ❌ SD img2img baseline generation FAILED: {str(e)}")
                print(f"  Error details:\n{error_details}")
                # Log the error to the session
                create_eval_session_log(
                    session_folder,
                    session.get("user_id", "anonymous"),
                    "sd_img2img_baseline_error",
                    {
                        "error": str(e),
                        "reference_image": reference_image_path,
                        "target_location": target_location,
                        "traceback": error_details
                    }
                )
        else:
            print(f"  ⚠️ Reference image not found, skipping img2img baseline")
        
        return {
            "success": True,
            "sliders": [{
                "slider_type": "current",
                "adjective": adjective,
                "location": target_location,
                "descriptor": descriptor,
                "images": slider_images
            }]
        }
        
    except Exception as e:
        print(f"[EVAL SLIDER] Error: {e}")
        import traceback
        traceback.print_exc()
        
        create_eval_session_log(
            session_folder,
            session.get("user_id", "anonymous"),
            "slider_generation_error",
            {"error": str(e)}
        )
        
        return {"error": True, "status_code": 500, "message": str(e)}


@app.post("/api/generate-slider")
async def generate_slider_eval(request: dict):
    """
    Generate ONE slider for eval prototype using EXPLORATION weights.
    
    This endpoint uses a thread pool to run the blocking image generation,
    keeping the event loop responsive for other requests.
    """
    session_id = request.get("session_id")
    location = request.get("location", "")
    
    session = eval_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {session_id}")
    
    session_folder = session["folder"]
    
    # Log start
    create_eval_session_log(
        session_folder,
        session.get("user_id", "anonymous"),
        "slider_generation_start",
        {"location": location, "mode": "exploration_weights"}
    )
    
    # Run the blocking generation in a thread pool to keep event loop responsive
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        generation_executor,
        lambda: _generate_slider_sync(session_id, location, session)
    )
    
    # Check if the sync function returned an error
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(result.get("status_code", 500), result.get("message", "Generation failed"))
    
    return result


# Global SDXL runner for eval (cached to avoid reloading)
_eval_sdxl_runner = None
_eval_slider_fuser = None


# ============== GP-specific endpoints ==============

@app.post("/api/concepts/process-round")
async def process_gp_round(request: dict):
    """
    Manually trigger GP processing of current interaction state.
    
    This is useful when you want to update the GP model without selecting an image.
    Only has effect in GP mode.
    """
    session_id = request.get("session_id")
    stage = request.get("stage", "impression")
    
    session = eval_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {session_id}")
    
    key = f"{session_id}_{stage}"
    use_gp = session.get("use_gp", USE_GP_EXPLORATION)
    
    if not use_gp or key not in gp_exploration_sessions:
        return {
            "success": False,
            "message": "GP mode not active for this session",
            "gp_mode": False
        }
    
    gp_session = gp_exploration_sessions[key]
    concepts = gp_session.process_round_manually()
    
    # Save updated weights
    gp_session.save_raw_tag_weights(session["folder"])
    
    state_dict = gp_session.to_dict()
    
    return {
        "success": True,
        "concepts": state_dict["concepts"],
        "categorized": state_dict["categorized"],
        "image_effects": state_dict["image_effects"],
        "tag_preferences": state_dict.get("tag_preferences", {}),
        "gp_mode": True,
        "n_concepts": len(concepts)
    }


# ============== HITL Refinement Endpoints ==============

# In-memory store for HITL refinement sessions
# Now uses V2 (tag-level GP) instead of V1 (composition-level GP)
hitl_sessions: Dict[str, HITLRefinementSessionV2] = {}


class HITLInitRequest(BaseModel):
    session_id: str
    base_prompt: Optional[str] = None
    negative_phrases: Optional[List[str]] = None


class HITLInitResponse(BaseModel):
    success: bool
    round_count: int
    is_initialized: bool
    is_converged: bool
    top_concepts: List[dict]


@app.post("/api/hitl/initialize", response_model=HITLInitResponse)
def initialize_hitl(request: HITLInitRequest):
    """
    Initialize HITL refinement from exploration outputs.
    
    IDEMPOTENT: If session already exists (browser refresh), reload state.
    This endpoint is called after exploration stage to begin refinement.
    """
    session = eval_sessions.get(request.session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {request.session_id}")
    
    session_folder = session["folder"]
    use_gp = session.get("use_gp", USE_GP_EXPLORATION)
    
    # IMPORTANT: Save tag_preferences.json from GP session before HITL init
    # This file is required by HITLRefinementSession.initialize_from_exploration()
    key = f"{request.session_id}_impression"
    tag_prefs_path = os.path.join(session_folder, "impression", "tag_preferences.json")
    
    if not os.path.exists(tag_prefs_path):
        if use_gp and key in gp_exploration_sessions:
            gp_session = gp_exploration_sessions[key]
            print(f"[HITL] Triggering GP fitting before HITL init for {request.session_id}...")
            gp_session.process_round_manually()
            
            # Save raw tag weights (concept_weights.json)
            gp_session.save_raw_tag_weights(session_folder)
            print(f"[HITL] Saved GP raw tag weights for {request.session_id}")
            
            # Save tag preferences (positive/negative/neutral)
            tag_prefs = gp_session.get_tag_preferences()
            positive_tags = []
            negative_tags = []
            neutral_tags = []
            
            for tag_id, pref in tag_prefs.items():
                if tag_id in gp_session.raw_tags:
                    tag_text = gp_session.raw_tags[tag_id].text
                    if pref == 'positive':
                        positive_tags.append(tag_text)
                    elif pref == 'negative':
                        negative_tags.append(tag_text)
                    else:
                        neutral_tags.append(tag_text)
            
            tag_preferences_data = {
                "positive": positive_tags,
                "negative": negative_tags,
                "neutral": neutral_tags
            }
            
            with open(tag_prefs_path, 'w') as f:
                json.dump(tag_preferences_data, f, indent=2)
            print(f"[HITL] Saved tag preferences: {len(positive_tags)} positive, {len(negative_tags)} negative, {len(neutral_tags)} neutral")
        else:
            # No GP session - create empty tag_preferences.json
            print(f"[HITL] Warning: No GP session found, creating empty tag_preferences.json")
            tag_preferences_data = {"positive": [], "negative": [], "neutral": []}
            os.makedirs(os.path.dirname(tag_prefs_path), exist_ok=True)
            with open(tag_prefs_path, 'w') as f:
                json.dump(tag_preferences_data, f, indent=2)
    
    # Determine base prompt from session
    base_prompt = request.base_prompt
    if not base_prompt:
        # Use "{adjective} {location}" as base prompt for HITL generation
        adjective = session.get("adjective", "")
        location = session.get("location", "")
        if adjective and location:
            base_prompt = f"{adjective} {location}"
        else:
            base_prompt = session.get("descriptor", location or "interior design")
        print(f"[HITL] Using base prompt: {base_prompt}")
    
    negative_phrases = request.negative_phrases or [
        "illustration","plan view", "bird's-eye view", "cartoon", "anime", 
        "isometric", "diorama", "miniature", "3D render", "CGI", 
        "concept art", "stylized", "toon shading", "human"
    ]
    
    # Initialize SDXL runner if not already available
    # This is needed for HITL image generation
    global _eval_sdxl_runner, _eval_slider_fuser
    
    if _eval_sdxl_runner is None:
        print("[HITL] Initializing SDXL runner for image generation...")
        from SDXL.sdxl_runner import SDXLRunner
        _eval_sdxl_runner = SDXLRunner(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            device=None,
            height=1024,
            width=1024
        )
        print("[HITL] SDXL runner initialized")
    
    pipe = None
    if _eval_sdxl_runner is not None and _eval_sdxl_runner.runner is not None:
        pipe = _eval_sdxl_runner.runner.pipe
    
    try:
        # Use V2 (tag-level GP) for HITL refinement
        hitl = HITLRefinementSessionV2.load_or_create(
            session_id=request.session_id,
            session_folder=session_folder,
            pipe=pipe,
            base_prompt=base_prompt,
            negative_prompt=", ".join(negative_phrases) if negative_phrases else "",
        )
        hitl_sessions[request.session_id] = hitl
        
        # Log event
        create_eval_session_log(
            session_folder,
            session.get("user_id", "anonymous"),
            "hitl_initialize",
            {
                "round_count": hitl.round_count,
                "is_restored": hitl.round_count > 0,
                "base_prompt": base_prompt,
                "version": "v2_tag_level_gp"
            }
        )
        
        # Get top tags for response
        top_tags = []
        if hitl.refiner:
            sorted_tags = sorted(hitl.refiner.tags.values(), key=lambda t: t.mu, reverse=True)[:5]
            top_tags = [
                {"label": t.text, "utility": round(t.mu, 3), "uncertainty": round(t.sigma, 3)}
                for t in sorted_tags
            ]
        
        return HITLInitResponse(
            success=True,
            round_count=hitl.round_count,
            is_initialized=hitl.is_initialized,
            is_converged=hitl.is_converged,
            top_concepts=top_tags
        )
    except Exception as e:
        print(f"[HITL] Error initializing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


class HITLGenerateRequest(BaseModel):
    session_id: str
    num_images: int = 4


class HITLGenerateResponse(BaseModel):
    success: bool
    round_number: int
    images: List[dict]
    compositions: List[dict]
    best_picks: Optional[List[dict]] = None


@app.post("/api/hitl/generate-round", response_model=HITLGenerateResponse)
async def generate_hitl_round(request: HITLGenerateRequest):
    """
    Generate images for ordinal ranking.
    
    Uses UCB acquisition to select diverse points from GP utility surface,
    then generates images with attention-weighted fusion.
    """
    hitl = hitl_sessions.get(request.session_id)
    if not hitl:
        raise HTTPException(404, f"HITL session not found: {request.session_id}")
    
    if not hitl.is_initialized:
        raise HTTPException(400, "HITL session not initialized")
    
    # Let user continue as many rounds as they want - no convergence blocking
    # They can finalize whenever they're satisfied
    
    try:
        # Run in thread pool for non-blocking generation
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            generation_executor,
            lambda: hitl.generate_round()
        )
        
        compositions, image_paths = result
        
        # Build response with image URLs
        images = []
        for i, path in enumerate(image_paths):
            # Convert path to URL
            rel_path = os.path.relpath(path, SESSION_LOGS_DIR)
            url = f"/session_logs/{rel_path}"
            images.append({
                "id": f"round_{hitl.round_count}_img_{i}",
                "url": url,
                "path": path
            })
        
        # Build compositions summary (V2 uses CompositionV2 objects)
        comp_summaries = []
        for comp in compositions:
            comp_summaries.append({
                "option_id": comp.option_id,
                "strategy": comp.strategy,
                "tags": comp.tag_labels[:5],  # Top 5 tags
                "weights": comp.weights[:5],
                "mus": comp.mus[:5],
            })
        
        # Use refiner.current_round which is incremented inside generate_round_options()
        # hitl.round_count only updates after record_ranking, so it lags by 1
        current_round = hitl.refiner.current_round if hitl.refiner else hitl.round_count
        
        # Build best picks with URLs for gallery
        best_picks = hitl.get_best_picks_list()
        for pick in best_picks:
            if pick.get("image_path"):
                try:
                    rel = os.path.relpath(pick["image_path"], SESSION_LOGS_DIR)
                    pick["url"] = f"/session_logs/{rel}"
                except ValueError:
                    pick["url"] = None
        
        return HITLGenerateResponse(
            success=True,
            round_number=current_round,
            images=images,
            compositions=comp_summaries,
            best_picks=best_picks if best_picks else None
        )
    except Exception as e:
        print(f"[HITL] Error generating round: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


class HITLRankRequest(BaseModel):
    session_id: str
    round_number: int
    ranking: List[int]  # [1st_idx, 2nd_idx, 3rd_idx, 4th_idx]


class HITLRankResponse(BaseModel):
    success: bool
    round_number: int
    gp_variance: float
    is_converged: bool
    next_round_ready: bool
    total_pairs: int


@app.post("/api/hitl/submit-ranking", response_model=HITLRankResponse)
def submit_hitl_ranking(request: HITLRankRequest):
    """
    Submit ordinal ranking and update tag utilities via tag-level GP.
    
    V2: Updates individual tag μ (utility) and σ (uncertainty) based on
    pairwise comparisons extracted from the ranking.
    """
    hitl = hitl_sessions.get(request.session_id)
    if not hitl:
        raise HTTPException(404, f"HITL session not found: {request.session_id}")
    
    try:
        result = hitl.record_ranking(request.ranking)
        
        session = eval_sessions.get(request.session_id)
        if session:
            create_eval_session_log(
                session["folder"],
                session.get("user_id", "anonymous"),
                "hitl_ranking",
                {
                    "round": result["round"],
                    "ranking": request.ranking,
                    "image_variance": result.get("image_variance", 0),
                    "beta": result.get("beta", 0),
                    "tags_updated": result.get("tags_updated", 0),
                    "is_converged": result["is_converged"]
                }
            )
        
        return HITLRankResponse(
            success=True,
            round_number=result["round"],
            gp_variance=result.get("image_variance", 0),  # Use image variance for V2
            is_converged=result["is_converged"],
            next_round_ready=not result["is_converged"],
            total_pairs=result.get("comparisons_made", 6)  # 6 comparisons from 4 options
        )
    except Exception as e:
        print(f"[HITL] Error recording ranking: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


class HITLFinalizeRequest(BaseModel):
    session_id: str


class HITLFinalizeResponse(BaseModel):
    success: bool
    final_selection_path: str
    rounds_completed: int
    total_pairs: int


@app.post("/api/hitl/finalize", response_model=HITLFinalizeResponse)
def finalize_hitl(request: HITLFinalizeRequest):
    """
    Finalize refinement and export best tags with attention weights.
    
    V2: Outputs top 10 tags with softmax-normalized weights based on
    learned tag utilities (μ values).
    """
    hitl = hitl_sessions.get(request.session_id)
    if not hitl:
        raise HTTPException(404, f"HITL session not found: {request.session_id}")
    
    try:
        output = hitl.finalize()
        
        session = eval_sessions.get(request.session_id)
        if session:
            create_eval_session_log(
                session["folder"],
                session.get("user_id", "anonymous"),
                "hitl_finalize",
                {
                    "rounds_completed": hitl.round_count,
                    "total_comparisons": hitl.round_count * 6,  # 6 per round
                    "final_tags": output.get("tags", [])[:5],
                    "version": "v2_tag_level_gp"
                }
            )
        
        # Get output path
        output_path = str(hitl.session_folder / "refined_preferences_v2.json")
        
        result = HITLFinalizeResponse(
            success=True,
            final_selection_path=output_path,
            rounds_completed=hitl.round_count,
            total_pairs=hitl.round_count * 6  # 6 pairwise comparisons per round
        )
        
        # Clean up session
        del hitl_sessions[request.session_id]
        
        return result
    except Exception as e:
        print(f"[HITL] Error finalizing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/hitl/status/{session_id}")
def get_hitl_status(session_id: str):
    """Get the status of an HITL refinement session."""
    hitl = hitl_sessions.get(session_id)
    if not hitl:
        return {
            "exists": False,
            "session_id": session_id
        }
    
    return {
        "exists": True,
        "session_id": session_id,
        **hitl.get_status()
    }


class HITLRollbackRequest(BaseModel):
    session_id: str
    target_round: int


@app.post("/api/hitl/rollback")
def rollback_hitl(request: HITLRollbackRequest):
    """
    Roll back to a previous round's state.
    
    The user preferred round X's output over current rounds.
    Restores tag states, injects preference bonus, and resumes from that round.
    """
    hitl = hitl_sessions.get(request.session_id)
    if not hitl:
        raise HTTPException(404, f"HITL session not found: {request.session_id}")
    
    try:
        result = hitl.rollback_to_round(request.target_round)
        
        session = eval_sessions.get(request.session_id)
        if session:
            create_eval_session_log(
                session["folder"],
                session.get("user_id", "anonymous"),
                "hitl_rollback",
                {
                    "from_round": result["from_round"],
                    "to_round": result["to_round"],
                    "tags_boosted": result["tags_boosted"][:5],
                    "tags_penalized": result["tags_penalized"][:5],
                }
            )
        
        return {
            "success": True,
            **result,
        }
    except Exception as e:
        print(f"[HITL] Error rolling back: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/hitl/best-picks/{session_id}")
def get_hitl_best_picks(session_id: str):
    """Get the best (1st-ranked) images from each round for the gallery."""
    hitl = hitl_sessions.get(session_id)
    if not hitl:
        raise HTTPException(404, f"HITL session not found: {session_id}")
    
    picks = hitl.get_best_picks_list()
    
    # Add URLs for each pick
    for pick in picks:
        if pick.get("image_path"):
            try:
                rel_path = os.path.relpath(pick["image_path"], SESSION_LOGS_DIR)
                pick["url"] = f"/session_logs/{rel_path}"
            except ValueError:
                pick["url"] = None
    
    return {
        "session_id": session_id,
        "best_picks": picks,
    }


# ============== Slot-Based Refinement Endpoints ==============

# In-memory store for slot refinement sessions
slot_refinement_sessions: Dict[str, SlotRefinementSession] = {}
gp_refinement_sessions: Dict[str, TagGPRefiner] = {}


class SlotRefineInitRequest(BaseModel):
    session_id: str


class SlotRefineInitResponse(BaseModel):
    status: str
    stage: str
    deduplication: dict
    slot_creation: dict


class SlotRoundRequest(BaseModel):
    session_id: str


class SlotRoundResponse(BaseModel):
    round_num: int
    stage: str
    round_type: str
    images: List[str]
    focus_slot: Optional[str] = None
    compositions: Optional[List[dict]] = None
    weight_configs: Optional[List[dict]] = None
    slots_status: Optional[List[dict]] = None
    current_weights: Optional[dict] = None


class SlotFeedbackRequest(BaseModel):
    session_id: str
    selected_idx: int


class SlotFeedbackResponse(BaseModel):
    stage: str
    is_complete: bool
    slots_status: Optional[List[dict]] = None
    current_weights: Optional[dict] = None
    newly_resolved: Optional[List[str]] = None
    eliminations: Optional[List] = None
    weight_updates: Optional[dict] = None
    max_change: Optional[float] = None


class SlotFinalizeRequest(BaseModel):
    session_id: str


class SlotFinalizeResponse(BaseModel):
    base_prompt: str
    final_tags: List[dict]
    final_prompt: str
    summary: dict
    description: str


@app.post("/api/slot-refinement/initialize", response_model=SlotRefineInitResponse)
def initialize_slot_refinement(request: SlotRefineInitRequest):
    """
    Initialize slot-based refinement from exploration outputs.
    
    Runs Stage 1 (Deduplication) and Stage 2 (Semantic Slots via LLM).
    After this, the session is ready for elimination rounds.
    """
    global _eval_sdxl_runner
    
    # Get session from eval_sessions or find from session_logs
    session = eval_sessions.get(request.session_id)
    if session:
        session_folder = Path(session["folder"])
    else:
        # Try to find session folder directly in session_logs
        session_folder = SESSION_LOGS_DIR / request.session_id
        if not session_folder.exists():
            raise HTTPException(404, f"Session not found: {request.session_id}")
        print(f"[SlotRefine] Found session folder directly: {session_folder}")
    
    # Check for existing session (idempotent)
    if request.session_id in slot_refinement_sessions:
        session = slot_refinement_sessions[request.session_id]
        return SlotRefineInitResponse(
            status="already_initialized",
            stage=session.stage.value,
            deduplication={
                "original_count": len(session.raw_tags),
                "deduplicated_count": session.dedup_result.deduplicated_count if session.dedup_result else 0,
                "duplicates_merged": session.dedup_result.duplicates_removed if session.dedup_result else []
            },
            slot_creation={
                "num_slots": len(session.slots),
                "slots": [
                    {"name": s.name, "description": s.description, "tags": s.tags, "importance": s.importance}
                    for s in session.slots
                ],
                "reasoning": ""
            }
        )
    
    # IMPORTANT: Save tag_preferences.json from GP session if it doesn't exist
    tag_prefs_path = session_folder / "impression" / "tag_preferences.json"
    
    if not tag_prefs_path.exists():
        # Try to get from GP session and save
        key = f"{request.session_id}_impression"
        gp_session = gp_exploration_sessions.get(key)
        
        if gp_session:
            # Save tag preferences (positive/negative/neutral)
            tag_prefs = gp_session.get_tag_preferences()
            positive_tags = [tag for tag, pref in tag_prefs.items() if pref == "positive"]
            negative_tags = [tag for tag, pref in tag_prefs.items() if pref == "negative"]
            neutral_tags = [tag for tag, pref in tag_prefs.items() if pref is None]
            
            # Get descriptor from session
            descriptor = ""
            if session:
                descriptor = f"{session.get('adjective', '')} {session.get('location', '')}".strip()
            
            tag_preferences_data = {
                "positive": positive_tags,
                "negative": negative_tags,
                "neutral": neutral_tags,
                "descriptor": descriptor
            }
            
            tag_prefs_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tag_prefs_path, 'w') as f:
                json.dump(tag_preferences_data, f, indent=2)
            print(f"[SlotRefine API] Saved tag_preferences.json: {len(positive_tags)} positive, {len(negative_tags)} negative")
        else:
            # No GP session - check for concept_weights.json and convert
            concept_weights_path = session_folder / "impression" / "concept_weights.json"
            if concept_weights_path.exists():
                with open(concept_weights_path) as f:
                    cw_data = json.load(f)
                
                # Extract positive/negative from concept_weights
                positive_tags = [
                    cw["label"] for cw in cw_data.get("concept_weights", [])
                    if cw.get("category") == "positive"
                ]
                negative_tags = [
                    cw["label"] for cw in cw_data.get("concept_weights", [])
                    if cw.get("category") == "negative"
                ]
                
                # Extract descriptor from session_id
                session_id = cw_data.get("session_id", request.session_id)
                descriptor = session_id.replace("eval_", "").split("_Sample")[0].replace("_", " ")
                
                tag_preferences_data = {
                    "positive": positive_tags,
                    "negative": negative_tags,
                    "neutral": [],
                    "descriptor": descriptor
                }
                
                with open(tag_prefs_path, 'w') as f:
                    json.dump(tag_preferences_data, f, indent=2)
                print(f"[SlotRefine API] Created tag_preferences.json from concept_weights: {len(positive_tags)} positive, {len(negative_tags)} negative")
            else:
                raise HTTPException(404, f"No GP session and no concept_weights.json found")
    
    print(f"[SlotRefine API] Using: {tag_prefs_path}")
    
    # Initialize SDXL if needed
    pipe = None
    if _eval_sdxl_runner is None:
        try:
            from SDXL.sdxl_runner import SDXLRunner
            print("[SlotRefine API] Initializing SDXL runner...")
            _eval_sdxl_runner = SDXLRunner(
                model_id="stabilityai/stable-diffusion-xl-base-1.0",
                height=512,
                width=512
            )
            pipe = _eval_sdxl_runner.pipe
            print("[SlotRefine API] SDXL runner initialized")
        except Exception as e:
            print(f"[SlotRefine API] Warning: Could not initialize SDXL: {e}")
    else:
        pipe = _eval_sdxl_runner.pipe
    
    # Create session
    session = SlotRefinementSession(
        session_id=request.session_id,
        session_folder=str(session_folder),
        pipe=pipe,
        sdxl_runner=_eval_sdxl_runner
    )
    
    # Initialize from exploration
    try:
        result = session.initialize_from_exploration(str(tag_prefs_path))
    except Exception as e:
        raise HTTPException(500, f"Failed to initialize slot refinement: {e}")
    
    slot_refinement_sessions[request.session_id] = session
    
    return SlotRefineInitResponse(
        status=result["status"],
        stage=result["stage"],
        deduplication=result["deduplication"],
        slot_creation=result["slot_creation"]
    )


@app.post("/api/slot-refinement/generate-round", response_model=SlotRoundResponse)
async def generate_slot_round(request: SlotRoundRequest):
    """
    Generate images for the next round.
    
    Returns image paths and round metadata.
    """
    session = slot_refinement_sessions.get(request.session_id)
    if not session:
        raise HTTPException(404, f"Slot refinement session not found: {request.session_id}")
    
    if session.is_complete:
        raise HTTPException(400, "Session already complete")
    
    try:
        result = session.generate_round()
    except Exception as e:
        print(f"[SlotRefine API] Error generating round: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Failed to generate round: {e}")
    
    return SlotRoundResponse(
        round_num=result["round_num"],
        stage=result["stage"],
        round_type=result["round_type"],
        images=result["images"],
        focus_slot=result.get("focus_slot"),
        compositions=result.get("compositions"),
        weight_configs=result.get("weight_configs"),
        slots_status=result.get("slots_status"),
        current_weights=result.get("current_weights")
    )


@app.post("/api/slot-refinement/submit-feedback", response_model=SlotFeedbackResponse)
def submit_slot_feedback(request: SlotFeedbackRequest):
    """
    Submit user selection for current round.
    
    Updates slot scores or weights based on stage.
    """
    session = slot_refinement_sessions.get(request.session_id)
    if not session:
        raise HTTPException(404, f"Slot refinement session not found: {request.session_id}")
    
    try:
        result = session.submit_feedback(request.selected_idx)
    except Exception as e:
        print(f"[SlotRefine API] Error processing feedback: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Failed to process feedback: {e}")
    
    return SlotFeedbackResponse(
        stage=result["stage"],
        is_complete=session.is_complete,
        slots_status=result.get("slots_status"),
        current_weights=result.get("current_weights"),
        newly_resolved=result.get("newly_resolved"),
        eliminations=result.get("eliminations"),
        weight_updates=result.get("weight_updates"),
        max_change=result.get("max_change")
    )


@app.post("/api/slot-refinement/finalize", response_model=SlotFinalizeResponse)
def finalize_slot_refinement(request: SlotFinalizeRequest):
    """
    Finalize refinement and get final weighted tags.
    
    Saves refined_preferences.json to session folder.
    """
    session = slot_refinement_sessions.get(request.session_id)
    if not session:
        raise HTTPException(404, f"Slot refinement session not found: {request.session_id}")
    
    try:
        result = session.finalize()
    except Exception as e:
        print(f"[SlotRefine API] Error finalizing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Failed to finalize: {e}")
    
    # Clean up session
    del slot_refinement_sessions[request.session_id]
    
    return SlotFinalizeResponse(
        base_prompt=result["base_prompt"],
        final_tags=result["final_tags"],
        final_prompt=result["final_prompt"],
        summary=result["summary"],
        description=result.get("description", "")
    )


@app.get("/api/slot-refinement/status/{session_id}")
def get_slot_refinement_status(session_id: str):
    """Get the status of a slot refinement session."""
    session = slot_refinement_sessions.get(session_id)
    if not session:
        return {"exists": False, "session_id": session_id}
    
    return {
        "exists": True,
        "session_id": session_id,
        **session.get_status()
    }


# ============== Tag GP Refinement Endpoints ==============
# New principled GP-based refinement with ranking feedback

class GPRefineInitRequest(BaseModel):
    session_id: str


class GPRefineInitResponse(BaseModel):
    status: str
    positive_tags: int
    neutral_tags: int
    total_tags: int
    categories: dict


class GPRoundResponse(BaseModel):
    round_num: int
    beta: float
    max_overlap: float
    options: List[dict]
    images: Optional[List[str]] = None  # Image paths if generated


class GPRankingRequest(BaseModel):
    session_id: str
    ranking: List[int]  # Option IDs from best to worst, e.g. [2, 0, 3, 1]


class GPRankingResponse(BaseModel):
    round_num: int
    pairwise_comparisons: int
    top_tags: List[dict]
    is_complete: bool


class GPFinalResponse(BaseModel):
    final_tags: List[str]
    weights: Dict[str, float]
    rounds_completed: int
    total_comparisons: int


@app.post("/api/gp-refinement/initialize", response_model=GPRefineInitResponse)
def initialize_gp_refinement(request: GPRefineInitRequest):
    """
    Initialize GP-based tag refinement from exploration outputs.
    
    Uses positive and neutral tags from exploration, assigns prior utilities
    based on whether tags appeared in selected images.
    """
    # Get session info
    session = eval_sessions.get(request.session_id)
    if session:
        session_folder = Path(session["folder"])
    else:
        session_folder = SESSION_LOGS_DIR / request.session_id
        if not session_folder.exists():
            raise HTTPException(404, f"Session not found: {request.session_id}")
    
    # Check for existing session (idempotent)
    if request.session_id in gp_refinement_sessions:
        refiner = gp_refinement_sessions[request.session_id]
        from tag_gp_refiner import TagCategory
        return GPRefineInitResponse(
            status="already_initialized",
            positive_tags=len(refiner.positive_tag_ids),
            neutral_tags=len(refiner.neutral_tag_ids),
            total_tags=len(refiner.tags),
            categories={
                cat.value: sum(1 for t in refiner.tags.values() if t.category == cat)
                for cat in TagCategory
            }
        )
    
    # Load tag preferences
    tag_prefs_path = session_folder / "impression" / "tag_preferences.json"
    concept_weights_path = session_folder / "impression" / "concept_weights.json"
    
    positive_tags = []
    neutral_tags = []
    selected_image_tags = set()
    
    if tag_prefs_path.exists():
        with open(tag_prefs_path) as f:
            data = json.load(f)
        positive_tags = data.get("positive", [])
        neutral_tags = data.get("neutral", [])
        # Note: selected_image_tags would need to be tracked during exploration
    elif concept_weights_path.exists():
        with open(concept_weights_path) as f:
            data = json.load(f)
        for cw in data.get("concept_weights", []):
            if cw.get("category") == "positive" or cw.get("score", 0) > 0.5:
                positive_tags.append(cw["label"])
            elif cw.get("category") == "neutral" or -0.5 <= cw.get("score", 0) <= 0.5:
                neutral_tags.append(cw["label"])
    else:
        # Try GP session
        key = f"{request.session_id}_impression"
        gp_session = gp_exploration_sessions.get(key)
        if gp_session:
            tag_prefs = gp_session.get_tag_preferences()
            positive_tags = [tag for tag, pref in tag_prefs.items() if pref == "positive"]
            neutral_tags = [tag for tag, pref in tag_prefs.items() if pref is None]
        else:
            raise HTTPException(404, "No tag preferences found")
    
    # Assume first half of positive tags were in selected images (heuristic)
    # In real usage, this should come from exploration tracking
    selected_image_tags = set(positive_tags[:len(positive_tags)//2])
    
    # Create refiner
    refiner = TagGPRefiner()
    
    # Set up logging
    refiner.set_logger(request.session_id, session_folder)
    
    result = refiner.initialize_from_exploration(
        positive_tags=positive_tags,
        neutral_tags=neutral_tags,
        selected_image_tags=selected_image_tags,
    )
    
    gp_refinement_sessions[request.session_id] = refiner
    
    # Save state
    state_path = session_folder / "gp_refinement" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    refiner.save_state(state_path)
    
    print(f"[GP Refine] Initialized: {len(positive_tags)} positive, {len(neutral_tags)} neutral tags")
    
    return GPRefineInitResponse(
        status=result["status"],
        positive_tags=result["positive_tags"],
        neutral_tags=result["neutral_tags"],
        total_tags=result["total_tags"],
        categories=result["categories"]
    )


@app.post("/api/gp-refinement/generate-round")
async def generate_gp_round(request: GPRefineInitRequest):
    """
    Generate 4 options for the current round with cross-attention weighted images.
    
    Each option contains 10 tags with different selection strategies:
    - Option 0: Exploitation (highest mean μ)
    - Option 1: Exploration (highest uncertainty σ)
    - Option 2: UCB Balanced (μ + β × σ)
    - Option 3: Challenger (top tags with swaps)
    
    Images are generated using SDXL with cross-attention weighting based on tag utilities.
    """
    global _eval_sdxl_runner
    
    refiner = gp_refinement_sessions.get(request.session_id)
    if not refiner:
        raise HTTPException(404, f"GP refinement session not found: {request.session_id}")
    
    if refiner.is_complete:
        raise HTTPException(400, "Refinement already complete (6 rounds)")
    
    # Generate options (tag selections)
    options = refiner.generate_round_options()
    
    # Setup SDXL pipeline if not already done
    if refiner.pipe is None and _eval_sdxl_runner is not None:
        # Get base prompt from session
        session = eval_sessions.get(request.session_id)
        if session:
            base_prompt = f"{session.get('adjective', '')} {session.get('location', '')}".strip()
        else:
            # Extract from session_id (e.g., "eval_Cozy_Bedroom_Sample_2026...")
            parts = request.session_id.replace("eval_", "").split("_Sample")[0]
            base_prompt = parts.replace("_", " ")
        
        refiner.set_sdxl_pipeline(
            pipe=_eval_sdxl_runner.pipe,
            base_prompt=base_prompt,
            image_height=512,
            image_width=512,
            num_inference_steps=30,
            guidance_scale=7.5,
        )
    
    # Generate images if pipeline is available
    image_paths = []
    if refiner.pipe is not None:
        try:
            session_folder = SESSION_LOGS_DIR / request.session_id
            round_dir = session_folder / "gp_refinement" / f"round_{refiner.current_round}"
            
            # Run image generation in thread pool to not block event loop
            image_paths = await asyncio.get_event_loop().run_in_executor(
                generation_executor,
                lambda: refiner.generate_round_images(round_dir)
            )
            
            print(f"[GP Refine] Generated {len(image_paths)} images for round {refiner.current_round}")
        except Exception as e:
            print(f"[GP Refine] Image generation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[GP Refine] No SDXL pipeline available, returning options without images")
    
    return GPRoundResponse(
        round_num=refiner.current_round,
        beta=round(refiner.beta, 2),
        max_overlap=round(refiner.max_overlap, 2),
        options=[opt.to_dict() for opt in options],
        images=image_paths if image_paths else None
    )


@app.post("/api/gp-refinement/submit-ranking", response_model=GPRankingResponse)
def submit_gp_ranking(request: GPRankingRequest):
    """
    Submit user ranking for current round.
    
    Ranking is a list of option IDs from best to worst.
    E.g., [2, 0, 3, 1] means Option 2 is best, Option 1 is worst.
    
    This generates 6 pairwise comparisons and updates tag utilities.
    """
    refiner = gp_refinement_sessions.get(request.session_id)
    if not refiner:
        raise HTTPException(404, f"GP refinement session not found: {request.session_id}")
    
    try:
        result = refiner.submit_ranking(request.ranking)
    except ValueError as e:
        raise HTTPException(400, str(e))
    
    # Get top tags for response
    sorted_tags = sorted(
        refiner.tags.values(),
        key=lambda t: t.mu,
        reverse=True
    )[:10]
    
    top_tags = [
        {"text": t.text, "mu": round(t.mu, 3), "sigma": round(t.sigma, 3)}
        for t in sorted_tags
    ]
    
    # Save state
    session_folder = SESSION_LOGS_DIR / request.session_id
    state_path = session_folder / "gp_refinement" / "state.json"
    refiner.save_state(state_path)
    
    return GPRankingResponse(
        round_num=result.round_num,
        pairwise_comparisons=len(result.pairwise_comparisons),
        top_tags=top_tags,
        is_complete=refiner.is_complete
    )


@app.post("/api/gp-refinement/finalize", response_model=GPFinalResponse)
def finalize_gp_refinement(request: GPRefineInitRequest):
    """
    Finalize GP refinement and get final weighted tags.
    
    Returns top 10 tags with softmax-normalized weights.
    Saves refined_preferences.json to session folder.
    """
    refiner = gp_refinement_sessions.get(request.session_id)
    if not refiner:
        raise HTTPException(404, f"GP refinement session not found: {request.session_id}")
    
    final_tags, weights = refiner.get_final_selection(n_tags=10, use_softmax=True)
    
    # Save to refined_preferences.json
    session_folder = SESSION_LOGS_DIR / request.session_id
    output_path = session_folder / "gp_refinement" / "refined_preferences.json"
    
    full_result = refiner.get_final_result()
    
    output_data = {
        "final_tags": [
            {"tag": tag, "weight": weights[tag], "usage": "cross_attention_map_scaling"}
            for tag in final_tags
        ],
        "all_tag_details": full_result["all_tag_details"],
        "summary": {
            "rounds_completed": full_result["rounds_completed"],
            "total_comparisons": full_result["total_comparisons"],
        },
        "round_history": full_result["round_history"],
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"[GP Refine] Saved final result to {output_path}")
    
    # Clean up session
    del gp_refinement_sessions[request.session_id]
    
    return GPFinalResponse(
        final_tags=final_tags,
        weights=weights,
        rounds_completed=full_result["rounds_completed"],
        total_comparisons=full_result["total_comparisons"]
    )


@app.get("/api/gp-refinement/status/{session_id}")
def get_gp_refinement_status(session_id: str):
    """Get the status of a GP refinement session."""
    refiner = gp_refinement_sessions.get(session_id)
    if not refiner:
        return {"exists": False, "session_id": session_id}
    
    sorted_tags = sorted(
        refiner.tags.values(),
        key=lambda t: t.mu,
        reverse=True
    )[:10]
    
    return {
        "exists": True,
        "session_id": session_id,
        "current_round": refiner.current_round,
        "max_rounds": refiner.config.max_rounds,
        "is_complete": refiner.is_complete,
        "total_tags": len(refiner.tags),
        "beta": round(refiner.beta, 2) if refiner.current_round > 0 else 2.0,
        "top_tags": [
            {"text": t.text, "mu": round(t.mu, 3), "sigma": round(t.sigma, 3)}
            for t in sorted_tags
        ]
    }


# ============== Evaluation Ranking Endpoints ==============

@app.get("/api/eval/session-logs")
def list_session_logs():
    """
    List available session logs that have slider directories.
    These are sessions that have already been processed and have generated images.
    """
    session_logs = []
    
    if SESSION_LOGS_DIR.exists():
        for session_folder in sorted(SESSION_LOGS_DIR.iterdir(), reverse=True):
            if session_folder.is_dir():
                slider_dir = session_folder / "slider"
                if slider_dir.exists() and slider_dir.is_dir():
                    # Get available locations in this session
                    locations = [loc.name for loc in slider_dir.iterdir() if loc.is_dir()]
                    
                    # Load session metadata if available
                    final_selection_path = session_folder / "final_selection.json"
                    adjective = ""
                    location = ""
                    if final_selection_path.exists():
                        try:
                            with open(final_selection_path, 'r') as f:
                                fs = json.load(f)
                                adjective = fs.get("adjective", "")
                                location = fs.get("location", "")
                        except:
                            pass
                    
                    session_logs.append({
                        "name": session_folder.name,
                        "path": str(session_folder),
                        "locations": locations,
                        "adjective": adjective,
                        "location": location,
                        "descriptor": f"{adjective} {location}".strip()
                    })
    
    return {"session_logs": session_logs}


@app.get("/api/eval/locations")
def get_available_locations():
    """
    Return all locations from baseline_generic_* folders.
    These are the locations that can be evaluated.
    Locations are the same across all adjectives, so we pick the first baseline folder.
    """
    locations = []
    
    # Find first baseline_generic_* folder to get locations
    if LLM_SCRIPTS_DIR.exists():
        baseline_folders = [f for f in LLM_SCRIPTS_DIR.iterdir() 
                          if f.is_dir() and f.name.startswith("baseline_generic_")]
        
        if baseline_folders:
            # Use first baseline folder to get location list
            first_baseline = baseline_folders[0]
            for loc_folder in sorted(first_baseline.iterdir()):
                if loc_folder.is_dir():
                    # Check if there's a baseline image
                    baseline_images = list(loc_folder.glob("*.png"))
                    if baseline_images:
                        locations.append({
                            "name": loc_folder.name,
                            "baseline_image": baseline_images[0].name
                        })
    
    return {"locations": locations}


class ComparisonImagesRequest(BaseModel):
    session_log: str
    location: str
    is_initial_round: bool = False


@app.post("/api/eval/get-comparison-images")
def get_comparison_images(request: ComparisonImagesRequest):
    """
    Get 4 comparison images for a location, randomized.

    Conditions:
    - slider/{location}/user_customized.png (manual user-selected tags/weights, HITL V2 pipeline)
    - slider/{location}/eval_alpha_1.00_*.png (ours)
    - slider/{location}/sd_baseline_text.png (SD text-only baseline)
    - slider/{location}/sd_style_transfer.png (SD img2img style transfer)

    Returns images in randomized order with a mapping to identify them later.
    """
    print(f"[GET-COMPARISON] Request: session_log={request.session_log}, location={request.location}, is_initial={request.is_initial_round}")
    
    session_folder = SESSION_LOGS_DIR / request.session_log
    if not session_folder.exists():
        raise HTTPException(404, f"Session log not found: {request.session_log}")
    
    # Find slider folder for this location (handle case sensitivity)
    slider_dir = session_folder / "slider"
    if not slider_dir.exists():
        raise HTTPException(404, f"Slider directory not found: {slider_dir}")
    
    location_folder = None
    location_folder_name = None
    
    for folder in slider_dir.iterdir():
        if folder.is_dir() and folder.name.lower().replace("_", " ") == request.location.lower().replace("_", " "):
            location_folder = folder
            location_folder_name = folder.name
            break
    
    if not location_folder or not location_folder.exists():
        raise HTTPException(404, f"Slider folder not found for location: {request.location}. Available: {[f.name for f in slider_dir.iterdir() if f.is_dir()]}")
    
    print(f"[GET-COMPARISON] Found slider folder: {location_folder_name}")
    
    # Find user_customized
    user_customized_path = location_folder / "user_customized.png"
    if not user_customized_path.exists():
        raise HTTPException(404, f"No user_customized.png found for: {request.location}")
    user_customized_url = f"/session_logs/{request.session_log}/slider/{location_folder_name}/user_customized.png"

    # Find SD text-only baseline (sd_baseline_text.png)
    baseline_text_path = location_folder / "sd_baseline_text.png"
    if not baseline_text_path.exists():
        raise HTTPException(404, f"No sd_baseline_text.png found for: {request.location}")
    baseline_url = f"/session_logs/{request.session_log}/slider/{location_folder_name}/sd_baseline_text.png"
    
    # Find alpha_1.00 image (ours - personalized with custom embeddings)
    alpha_images = list(location_folder.glob("eval_alpha_1.00_*.png"))
    if not alpha_images:
        raise HTTPException(404, f"No eval_alpha_1.00 image found for: {request.location}")
    
    alpha_image_name = alpha_images[0].name
    alpha_url = f"/session_logs/{request.session_log}/slider/{location_folder_name}/{alpha_image_name}"
    
    # Find SD img2img style transfer (sd_style_transfer.png)
    style_transfer_path = location_folder / "sd_style_transfer.png"
    if not style_transfer_path.exists():
        raise HTTPException(404, f"No sd_style_transfer.png found for: {request.location}")
    style_transfer_url = f"/session_logs/{request.session_log}/slider/{location_folder_name}/sd_style_transfer.png"
    
    # Build images list with identifiers (4 images)
    images = [
        {"id": "user_customized", "url": user_customized_url, "filename": "user_customized.png", "type": "user_customized"},
        {"id": "baseline", "url": baseline_url, "filename": "sd_baseline_text.png", "type": "baseline"},
        {"id": "alpha", "url": alpha_url, "filename": alpha_image_name, "type": "personalized"},
        {"id": "third", "url": style_transfer_url, "filename": "sd_style_transfer.png", "type": "style_transfer"}
    ]
    
    # Randomize order
    random.shuffle(images)
    
    # Create position mapping (which position each image ended up in)
    position_mapping = {img["id"]: idx for idx, img in enumerate(images)}
    
    print(f"[GET-COMPARISON] Success! Returning 4 images for {request.location}")
    
    return {
        "images": images,
        "position_mapping": position_mapping,
        "location": request.location,
        "is_initial_round": request.is_initial_round
    }


class SaveRankingRequest(BaseModel):
    session_log: str
    location: str
    rankings: Dict[str, dict]  # {"1": {"image": "filename", "score": 6}, "2": {...}, ...}


@app.post("/api/eval/save-ranking")
def save_ranking(request: SaveRankingRequest):
    """
    Save ranking for a location to rank_order.json in the session folder.
    """
    session_folder = SESSION_LOGS_DIR / request.session_log
    if not session_folder.exists():
        raise HTTPException(404, f"Session log not found: {request.session_log}")
    
    rank_order_path = session_folder / "rank_order.json"
    
    # Load existing or create new
    if rank_order_path.exists():
        with open(rank_order_path, 'r') as f:
            rank_order = json.load(f)
    else:
        rank_order = {
            "session_log": request.session_log,
            "rankings": {}
        }
    
    # Update ranking for this location
    rank_order["rankings"][request.location] = request.rankings
    
    # Save back
    with open(rank_order_path, 'w') as f:
        json.dump(rank_order, f, indent=2)
    
    print(f"[EVAL] Saved ranking for {request.location}: {request.rankings}")
    
    # Check if all 8 locations have been ranked - generate histograms automatically
    num_ranked = len(rank_order["rankings"])
    print(f"[EVAL] Total locations ranked: {num_ranked}/8")
    
    if num_ranked >= 6:
        print(f"[EVAL] ✅ All 6 locations ranked! Generating histograms automatically...")
        try:
            histogram_result = generate_histograms(str(rank_order_path), show=False)
            print(f"[EVAL] ✅ Histograms generated successfully!")
            print(f"[EVAL]   - rank_histogram.png saved to {session_folder}")
            print(f"[EVAL]   - average_rank.png saved to {session_folder}")
            print(f"[EVAL]   - score_by_rank.png saved to {session_folder}")
        except Exception as e:
            import traceback
            print(f"[EVAL] ❌ Failed to generate histograms: {e}")
            traceback.print_exc()
    
    return {
        "success": True,
        "location": request.location,
        "rankings": request.rankings,
        "total_ranked": num_ranked
    }


class InitRankingSessionRequest(BaseModel):
    session_log: str


@app.post("/api/eval/init-ranking-session")
def init_ranking_session(request: InitRankingSessionRequest):
    """
    Initialize rank_order.json for a session, preserving existing rankings if present.
    This allows resuming interrupted ranking sessions.
    Also loads the session into eval_sessions so generation endpoints work.
    """
    session_folder = SESSION_LOGS_DIR / request.session_log
    if not session_folder.exists():
        raise HTTPException(404, f"Session log not found: {request.session_log}")
    
    rank_order_path = session_folder / "rank_order.json"
    
    # Only create if doesn't exist - preserve existing rankings for resume
    if not rank_order_path.exists():
        rank_order = {
            "session_log": request.session_log,
            "rankings": {}
        }
        with open(rank_order_path, 'w') as f:
            json.dump(rank_order, f, indent=2)
        print(f"[EVAL] Created new rank_order.json for {request.session_log}")
    else:
        print(f"[EVAL] Preserving existing rank_order.json for {request.session_log}")
    
    # Load session into eval_sessions so generation endpoints work
    # This is needed for resuming sessions that weren't started in this server instance
    if request.session_log not in eval_sessions:
        # Load session metadata from final_selection.json
        final_selection_path = session_folder / "final_selection.json"
        adjective = ""
        location = ""
        descriptor = ""
        
        if final_selection_path.exists():
            try:
                with open(final_selection_path, 'r') as f:
                    final_selection = json.load(f)
                    adjective = final_selection.get("adjective", "")
                    location = final_selection.get("location", "")
                    descriptor = final_selection.get("descriptor", f"{adjective} {location}".strip())
            except Exception as e:
                print(f"[EVAL] Warning: Could not load final_selection.json: {e}")
        
        eval_sessions[request.session_log] = {
            "folder": str(session_folder),
            "descriptor": descriptor,
            "adjective": adjective,
            "location": location,
            "user_pref": {},
            "user_id": "resumed"
        }
        print(f"[EVAL] Loaded session into eval_sessions: {request.session_log}")
    
    # Return existing rankings count for frontend
    with open(rank_order_path, 'r') as f:
        rank_order = json.load(f)
    
    return {
        "success": True,
        "session_log": request.session_log,
        "existing_rankings_count": len(rank_order.get("rankings", {}))
    }


@app.get("/api/eval/get-rankings/{session_log}")
def get_rankings(session_log: str):
    """
    Get current rankings for a session.
    """
    session_folder = SESSION_LOGS_DIR / session_log
    if not session_folder.exists():
        raise HTTPException(404, f"Session log not found: {session_log}")
    
    rank_order_path = session_folder / "rank_order.json"
    
    if rank_order_path.exists():
        with open(rank_order_path, 'r') as f:
            rank_order = json.load(f)
        return rank_order
    else:
        return {
            "session_log": session_log,
            "rankings": {}
        }


@app.get("/api/eval/session-locations/{session_log}")
def get_session_locations(session_log: str):
    """
    Get locations that have been generated for this session.
    Used for resuming interrupted ranking sessions to determine which
    locations already have slider images.
    """
    session_folder = SESSION_LOGS_DIR / session_log
    if not session_folder.exists():
        raise HTTPException(404, f"Session log not found: {session_log}")
    
    slider_dir = session_folder / "slider"
    
    generated = []
    if slider_dir.exists():
        for loc_folder in slider_dir.iterdir():
            if loc_folder.is_dir():
                # Check if it has the required images (eval_alpha_1.00_*.png)
                has_alpha = list(loc_folder.glob("eval_alpha_1.00_*.png"))
                if has_alpha:
                    generated.append(loc_folder.name)
    
    print(f"[EVAL] Session {session_log} has {len(generated)} generated locations: {generated}")
    
    return {"generated_locations": generated}


# ============== Health check ==============

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "eval_server",
        "predefined_sessions": len(list_predefined_sessions(str(PREDEFINED_INPUT_DIR))),
        "active_sessions": len(eval_sessions),
        "gp_exploration_enabled": USE_GP_EXPLORATION,
        "active_gp_sessions": len(gp_exploration_sessions)
    }


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("EVALUATION PROTOTYPE SERVER")
    print("=" * 60)
    print(f"Predefined input: {PREDEFINED_INPUT_DIR}")
    print(f"Session logs: {SESSION_LOGS_DIR}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8001)

