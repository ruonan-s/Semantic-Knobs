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

# ============== Mode Toggle ==============
# Set to True to use GP-based preference learning, False for original softmax approach
USE_GP_EXPLORATION = True

# Import style transfer and baseline for LLM-based generation when applying to new locations
LLM_SCRIPTS_PATH = Path(__file__).parent.parent / "llm_scripts"
sys.path.insert(0, str(LLM_SCRIPTS_PATH))
from style_transfer import generate_image_with_reference
from baseline1 import generate_baseline_image
from baseline_tags import generate_baseline_tags_image

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


def _generate_slider_sync(session_id: str, location: str, session: dict):
    """
    Synchronous slider generation - runs in thread pool to avoid blocking event loop.
    Uses exploration selected image for style transfer (no need for rank #1 reference).
    """
    import numpy as np
    import torch
    from datetime import datetime
    
    session_folder = session["folder"]
    
    try:
        print("\n" + "=" * 80)
        print(f"[EVAL SLIDER] Generating slider with EXPLORATION weights")
        print(f"[EVAL SLIDER] Session: {session_id}")
        print("=" * 80)
        
        # Load exploration concept_weights.json
        concept_weights_path = os.path.join(session_folder, "impression", "concept_weights.json")
        if not os.path.exists(concept_weights_path):
            return {"error": True, "status_code": 404, "message": "concept_weights.json not found in impression stage"}
        
        with open(concept_weights_path, 'r') as f:
            concept_weights_data = json.load(f)
        
        # Load final_selection.json for adjective/location
        final_selection_path = os.path.join(session_folder, "final_selection.json")
        if not os.path.exists(final_selection_path):
            return {"error": True, "status_code": 404, "message": "final_selection.json not found"}
        
        with open(final_selection_path, 'r') as f:
            final_selection = json.load(f)
        
        # Load preferences.json
        preferences_path = os.path.join(session_folder, "preferences.json")
        prefs = {}
        if os.path.exists(preferences_path):
            with open(preferences_path, 'r') as f:
                prefs = json.load(f)
        
        # Get adjective and location
        adjective = final_selection.get("adjective", "")
        target_location = location if location else final_selection.get("location", "")
        descriptor = f"{adjective} {target_location}"
        
        # Check if this is a café/coffeeshop context - add photorealistic prefix to tags only
        target_location_lower = target_location.lower()
        is_cafe_context = (
            "café" in target_location_lower or 
            "cafe" in target_location_lower or 
            "coffeeshop" in target_location_lower or 
            "coffee shop" in target_location_lower
        )
        
        print(f"  Adjective: {adjective}")
        print(f"  Location: {target_location}")
        print(f"  Descriptor: {descriptor}")
        
        # Get exploration weights
        concept_weights = concept_weights_data.get("concept_weights", [])
        if not concept_weights:
            return {"error": True, "status_code": 400, "message": "No concept weights found"}
        
        # Build concepts with location prefix
        concepts = []
        weights = []
        for cw in concept_weights:
            if is_cafe_context:
                concepts.append({
                    "id": cw.get("concept_id", ""),
                    "label": f"photorealistic {target_location} with {cw['label']}"
                })
            else:
                concepts.append({
                    "id": cw.get("concept_id", ""),
                    "label": f"{target_location} with {cw['label']}"
                })
            weights.append(cw.get("weight", 0.0))
        
        w_exploration = np.array(weights)
        w_exploration_norm = w_exploration / (w_exploration.sum() + 1e-8)
        
        print(f"  Exploration concepts: {len(concepts)}")
        print(f"  Weights sum: {w_exploration.sum():.4f}")
        
        # Get top-K concepts
        top_k = 10
        sorted_indices = np.argsort(w_exploration_norm)[::-1]
        actual_top_k = min(top_k, len(concepts))
        top_indices = sorted_indices[:actual_top_k]
        
        tag_phrases = [concepts[idx]['label'] for idx in top_indices]
        tag_weights = np.array([float(w_exploration_norm[idx]) for idx in top_indices])
        
        print(f"  Top {actual_top_k} concepts: {[c.split(' with ')[-1] for c in tag_phrases[:3]]}")
        
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
        
        if _eval_slider_fuser is None:
            return {"error": True, "status_code": 500, "message": "SDXL pipeline not available"}
        
        # Generate images with alpha interpolation
        neg_phrases = ["illustration", "cartoon", "anime", "human"]
        
        # Only generate alpha=1.0 for evaluation (skip intermediate values)
        alphas = [1.0]
        seed_base = 2026
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = []
        
        print(f"\n  Generating {len(alphas)} image(s) (alpha=1.0 only)...")
        
        for i, alpha in enumerate(alphas):
            print(f"  [{i+1}/{len(alphas)}] Alpha = {alpha:.2f}")
            
            prompt_embeds, pooled, neg_embeds, neg_pooled = _eval_slider_fuser.fuse_with_alpha(
                descriptor=descriptor,
                tag_phrases=tag_phrases,
                tag_weights=tag_weights,
                alpha=alpha,
                neg_phrases=neg_phrases,
                max_negatives=20
            )
            
            generator = torch.Generator(device=_eval_sdxl_runner.runner.device).manual_seed(seed_base + i)
            
            image = _eval_sdxl_runner.runner.pipe(
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
            
            results.append((alpha, image, None))
        
        # Save images and build response
        slider_images = []
        for alpha, img, ref_flag in results:
            filename = f"eval_alpha_{alpha:.2f}_{timestamp}.png"
            
            filepath = os.path.join(slider_output_dir, filename)
            img.save(filepath)
            
            # URL relative to session_logs
            rel_path = os.path.relpath(filepath, SESSION_LOGS_DIR)
            url = f"/session_logs/{rel_path}"
            
            slider_images.append({"alpha": alpha, "url": url})
            print(f"  Saved: {filename}")
        
        # Log success
        create_eval_session_log(
            session_folder,
            session.get("user_id", "anonymous"),
            "slider_generation_complete",
            {"success": True, "image_count": len(slider_images)}
        )
        
        print(f"\n  ✅ Generated {len(slider_images)} images for eval slider")
        
        # ========== LLM BASELINE TAGS (for all locations) ==========
        # Generate image using learned tags: "{adjective} {location}, {tag1}, {tag2}, ..."
        print(f"\n{'='*80}")
        print(f"[LLM BASELINE TAGS] Generating baseline with learned tags")
        print(f"{'='*80}")
        
        try:
            baseline_tags_path = generate_baseline_tags_image(
                session_folder=session_folder,
                location=target_location,
                output_folder=slider_output_dir
            )
            print(f"  ✅ Baseline tags image saved: {os.path.basename(baseline_tags_path)}")
        except Exception as e:
            print(f"  ⚠️ Baseline tags generation failed: {str(e)}")
        
        # ========== LLM STYLE TRANSFER (for ALL locations) ==========
        # Uses exploration selected image as reference for style transfer
        original_location = final_selection.get("location", "")
        is_new_location = location and location != original_location
        
        print(f"\n{'='*80}")
        print(f"[LLM STYLE TRANSFER] Generating style transfer image")
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
            
            # Build the style transfer prompt based on whether it's original or new location
            if is_new_location:
                style_transfer_prompt = (
                    f"This user selected this image as their preferred example of a {adjective} {original_location}. "
                    f"Generate a {adjective} {target_location} that matches this user's personal aesthetic"
                )
            else:
                # For original location, generate another image in the same style
                style_transfer_prompt = (
                    f"This user selected this image as their preferred example of a {adjective} {target_location}. "
                    f"Generate another {adjective} {target_location} that matches this user's personal aesthetic"
                )
            print(f"  Prompt: {style_transfer_prompt}")
            
            # Output path for style transfer image
            style_transfer_output = os.path.join(slider_output_dir, "llm_style_transfer.png")
            
            try:
                generated_path = generate_image_with_reference(
                    input_image_path=reference_image_path,
                    text_prompt=style_transfer_prompt,
                    output_path=style_transfer_output
                )
                print(f"  ✅ Style transfer image saved: {os.path.basename(generated_path)}")
            except Exception as e:
                print(f"  ⚠️ Style transfer failed: {str(e)}")
        else:
            print(f"  ⚠️ Reference image not found")
        
        # NOTE: Baseline image for all locations comes from baseline_generic folder
        # No need to generate llm_baseline.png - get-comparison-images uses baseline_generic
        
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
    
    For initial round (bedroom):
    - baseline_generic_{adjective}/{Location}/{Adjective}_{Location}.png (generic LLM baseline)
    - slider/{location}/eval_alpha_1.00_*.png (SDXL personalized)
    - slider/{location}/eval_alphaRef_1.00_*.png (SDXL with reference)
    - slider/{location}/llm_baseline_tags.png (LLM with learned tags)
    
    For other locations:
    - baseline_generic_{adjective}/{Location}/{Adjective}_{Location}.png (generic LLM baseline)
    - slider/{location}/eval_alpha_1.00_*.png (SDXL personalized)
    - slider/{location}/llm_style_transfer.png (LLM style transfer)
    - slider/{location}/llm_baseline_tags.png (LLM with learned tags)
    
    Returns images in randomized order with a mapping to identify them later.
    """
    print(f"[GET-COMPARISON] Request: session_log={request.session_log}, location={request.location}, is_initial={request.is_initial_round}")
    
    session_folder = SESSION_LOGS_DIR / request.session_log
    if not session_folder.exists():
        raise HTTPException(404, f"Session log not found: {request.session_log}")
    
    # Load adjective from final_selection.json
    final_selection_path = session_folder / "final_selection.json"
    if not final_selection_path.exists():
        raise HTTPException(404, f"final_selection.json not found in session: {request.session_log}")
    
    with open(final_selection_path, 'r') as f:
        final_selection = json.load(f)
    
    adjective = final_selection.get("adjective", "")
    if not adjective:
        raise HTTPException(400, f"No adjective found in final_selection.json for session: {request.session_log}")
    
    print(f"[GET-COMPARISON] Session adjective: {adjective}")
    
    # Construct dynamic baseline folder path: baseline_generic_{adjective}
    baseline_generic_folder = LLM_SCRIPTS_DIR / f"baseline_generic_{adjective}"
    if not baseline_generic_folder.exists():
        available_baseline_folders = [f.name for f in LLM_SCRIPTS_DIR.iterdir() 
                                      if f.is_dir() and f.name.startswith("baseline_generic_")]
        raise HTTPException(404, f"Baseline folder not found: baseline_generic_{adjective}. Available: {available_baseline_folders}")
    
    # Find baseline image for the location (handle case sensitivity)
    baseline_folder = None
    baseline_folder_name = None
    available_locations = [f.name for f in baseline_generic_folder.iterdir() if f.is_dir()]
    print(f"[GET-COMPARISON] Available locations in baseline_generic_{adjective}: {available_locations}")
    
    for folder in baseline_generic_folder.iterdir():
        if folder.is_dir() and folder.name.lower().replace("_", " ") == request.location.lower().replace("_", " "):
            baseline_folder = folder
            baseline_folder_name = folder.name
            break
    
    if not baseline_folder or not baseline_folder.exists():
        raise HTTPException(404, f"Baseline location not found: {request.location}. Available: {available_locations}")
    
    baseline_images = list(baseline_folder.glob("*.png"))
    if not baseline_images:
        raise HTTPException(404, f"No baseline image found for: {request.location}")
    
    baseline_image_name = baseline_images[0].name
    baseline_url = f"/llm_scripts/baseline_generic_{adjective}/{baseline_folder_name}/{baseline_image_name}"
    
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
    
    # Find alpha_1.00 image
    alpha_images = list(location_folder.glob("eval_alpha_1.00_*.png"))
    if not alpha_images:
        raise HTTPException(404, f"No eval_alpha_1.00 image found for: {request.location}")
    
    alpha_image_name = alpha_images[0].name
    alpha_url = f"/session_logs/{request.session_log}/slider/{location_folder_name}/{alpha_image_name}"
    
    # Find third image: llm_style_transfer.png (same for all locations)
    style_transfer_path = location_folder / "llm_style_transfer.png"
    if not style_transfer_path.exists():
        raise HTTPException(404, f"No llm_style_transfer.png found for: {request.location}")
    third_image_name = "llm_style_transfer.png"
    third_image_type = "style_transfer"
    
    third_url = f"/session_logs/{request.session_log}/slider/{location_folder_name}/{third_image_name}"
    
    # Find fourth image: llm_baseline_tags.png (LLM with learned tags)
    baseline_tags_path = location_folder / "llm_baseline_tags.png"
    if not baseline_tags_path.exists():
        raise HTTPException(404, f"No llm_baseline_tags.png found for: {request.location}")
    
    tags_url = f"/session_logs/{request.session_log}/slider/{location_folder_name}/llm_baseline_tags.png"
    
    # Build images list with identifiers (4 images)
    images = [
        {"id": "baseline", "url": baseline_url, "filename": baseline_image_name, "type": "baseline"},
        {"id": "alpha", "url": alpha_url, "filename": alpha_image_name, "type": "personalized"},
        {"id": "third", "url": third_url, "filename": third_image_name, "type": third_image_type},
        {"id": "tags", "url": tags_url, "filename": "llm_baseline_tags.png", "type": "baseline_tags"}
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
    
    if num_ranked >= 8:
        print(f"[EVAL] ✅ All 8 locations ranked! Generating histograms automatically...")
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

