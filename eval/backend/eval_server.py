"""
Evaluation Prototype Server

A lightweight server that extends the main backend server with eval-specific
endpoints for the evaluation prototype that skips refinement.
"""

import os
import sys
from pathlib import Path

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
from typing import Optional, List
import json

# Import eval utilities
from eval_utils import (
    generate_final_selection_from_exploration,
    copy_predefined_session,
    list_predefined_sessions,
    validate_session_for_eval,
    create_eval_session_log
)

# Import from main backend
from concept_refinement import get_or_create_session as get_refinement_session, refinement_sessions

# Import style transfer and baseline for LLM-based generation when applying to new locations
LLM_SCRIPTS_PATH = Path(__file__).parent.parent / "llm_scripts"
sys.path.insert(0, str(LLM_SCRIPTS_PATH))
from style_transfer import generate_image_with_reference
from baseline1 import generate_baseline_image

# Create FastAPI app for eval
app = FastAPI(title="Semantic Knobs Eval Prototype")

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

# Ensure directories exist
os.makedirs(PREDEFINED_INPUT_DIR, exist_ok=True)
os.makedirs(SESSION_LOGS_DIR, exist_ok=True)

# Mount static files for serving images
app.mount("/predefined", StaticFiles(directory=str(PREDEFINED_INPUT_DIR)), name="predefined")
app.mount("/session_logs", StaticFiles(directory=str(SESSION_LOGS_DIR)), name="session_logs")

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
        
        # Save concept weights before generating final_selection
        # This ensures the exploration weights are saved
        key = f"{request.session_id}_impression"
        if key in refinement_sessions:
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
                "concept_count": final_selection.get("summary", {}).get("total_concepts", 0)
            }
        )
        
        print(f"[EVAL] Skipping refinement for session: {request.session_id}")
        print(f"  Using exploration weights directly for slider generation")
        
        return SkipToSliderResponse(
            success=True,
            message="Generated final_selection from exploration weights. Ready for slider generation.",
            next_stage="slider_generation"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[EVAL] Error in skip_to_slider: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/eval/status/{session_id}", response_model=EvalStatusResponse)
def get_eval_status(session_id: str):
    """Get the status of an evaluation session."""
    session = eval_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {session_id}")
    
    # Check concept system state
    key = f"{session_id}_impression"
    concepts_initialized = key in refinement_sessions
    concept_count = 0
    if concepts_initialized:
        concept_count = len(refinement_sessions[key].concepts)
    
    return EvalStatusResponse(
        session_id=session_id,
        stage="impression",
        concepts_initialized=concepts_initialized,
        concept_count=concept_count
    )


# ============== Proxy endpoints to main backend ==============
# These endpoints proxy to the main backend functionality

@app.post("/api/concepts/init")
async def proxy_concepts_init(request: dict):
    """Initialize concepts - proxied from main backend."""
    from concept_refinement import (
        build_concepts, compute_weights, RawTag, Concept, ConceptState
    )
    
    session_id = request.get("session_id")
    stage = request.get("stage", "impression")
    image_ids = request.get("image_ids", [])
    
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
    
    # Get or create refinement session (must pass image_ids)
    key = f"{session_id}_{stage}"
    refinement_session = get_refinement_session(session_id, stage, image_ids)
    
    if not refinement_session.initialized:
        refinement_session.initialize_from_tags(image_tags)
        # Save initial weights
        refinement_session.save_concept_weights(session_folder)
    
    state_dict = refinement_session.to_dict()
    
    return {
        "success": True,
        "concepts": state_dict["concepts"],
        "categorized": state_dict["categorized"],
        "image_effects": state_dict["image_effects"],
        "incidence_matrix": state_dict.get("incidence_matrix", {}),
        "tag_preferences": state_dict.get("tag_preferences", {})
    }


@app.post("/api/concepts/interact")
async def proxy_concepts_interact(request: dict):
    """Handle tag interaction - proxied from main backend."""
    session_id = request.get("session_id")
    stage = request.get("stage", "impression")
    tag_id = request.get("tag_id")
    preference = request.get("preference")  # 'positive' or 'negative'
    
    session = eval_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {session_id}")
    
    key = f"{session_id}_{stage}"
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
        "tag_preferences": state_dict.get("tag_preferences", {})
    }


@app.post("/api/concepts/select-image")
async def proxy_select_image(request: dict):
    """Handle image selection - proxied from main backend."""
    session_id = request.get("session_id")
    stage = request.get("stage", "impression")
    image_id = request.get("image_id")
    boost_amount = request.get("boost_amount", 0.5)
    
    session = eval_sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session not found: {session_id}")
    
    key = f"{session_id}_{stage}"
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
        "tag_preferences": state_dict.get("tag_preferences", {})
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


@app.post("/api/generate-slider")
async def generate_slider_eval(request: dict):
    """
    Generate ONE slider for eval prototype using EXPLORATION weights.
    
    This is a simplified version that:
    - Uses exploration weights (not refinement weights)
    - Generates only the "current" slider with 6 images
    - Alpha values: 0.0, 0.25, 0.5, 0.75, 1.0, 1.0+reference
    """
    import numpy as np
    import torch
    from datetime import datetime
    from PIL import Image as PILImage
    
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
    
    try:
        print("\n" + "=" * 80)
        print(f"[EVAL SLIDER] Generating slider with EXPLORATION weights")
        print(f"[EVAL SLIDER] Session: {session_id}")
        print("=" * 80)
        
        # Load exploration concept_weights.json
        concept_weights_path = os.path.join(session_folder, "impression", "concept_weights.json")
        if not os.path.exists(concept_weights_path):
            raise HTTPException(404, "concept_weights.json not found in impression stage")
        
        with open(concept_weights_path, 'r') as f:
            concept_weights_data = json.load(f)
        
        # Load final_selection.json for adjective/location
        final_selection_path = os.path.join(session_folder, "final_selection.json")
        if not os.path.exists(final_selection_path):
            raise HTTPException(404, "final_selection.json not found")
        
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
        
        print(f"  Adjective: {adjective}")
        print(f"  Location: {target_location}")
        print(f"  Descriptor: {descriptor}")
        
        # Get exploration weights
        concept_weights = concept_weights_data.get("concept_weights", [])
        if not concept_weights:
            raise HTTPException(400, "No concept weights found")
        
        # Build concepts with location prefix
        concepts = []
        weights = []
        for cw in concept_weights:
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
        
        # Load reference image
        reference_image = None
        is_original_location = not location
        
        if is_original_location:
            selections = prefs.get('selections', {})
            ref_image_id = selections.get('impression')
            
            if ref_image_id:
                impression_folder = os.path.join(session_folder, 'impression')
                ref_path = os.path.join(impression_folder, f"{ref_image_id}.png")
                
                if not os.path.exists(ref_path):
                    ref_path = os.path.join(impression_folder, f"{ref_image_id}_0.png")
                
                if os.path.exists(ref_path):
                    reference_image = PILImage.open(ref_path)
                    print(f"  Reference image: {os.path.basename(ref_path)}")
        
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
            raise HTTPException(500, "SDXL pipeline not available")
        
        # Generate images with alpha interpolation
        neg_phrases = ["illustration", "painted", "drawing", "cartoon", "anime", 
                       "isometric", "diorama", "miniature", "3D render", "CGI", 
                       "concept art", "stylized", "toon shading", 
                       "people", "person", "human"]
        
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        seed_base = 30
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = []
        prompt_embeds_alpha_1 = None
        
        print(f"\n  Generating {len(alphas)} images...")
        
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
            
            if alpha == 1.0:
                prompt_embeds_alpha_1 = (prompt_embeds, pooled, neg_embeds, neg_pooled)
            
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
        
        # Generate 6th image with reference (img2img at alpha=1.0)
        if reference_image is not None and is_original_location and prompt_embeds_alpha_1 is not None:
            print(f"  [6/6] Alpha = 1.0 (with reference image)")
            
            prompt_embeds_ref, pooled_ref, neg_embeds_ref, neg_pooled_ref = prompt_embeds_alpha_1
            
            from backend.sdxl_config import get_stage_strength
            strength = get_stage_strength('impression')
            
            generator = torch.Generator(device=_eval_sdxl_runner.runner.device).manual_seed(seed_base + 5)
            
            image_6 = _eval_sdxl_runner.runner.generate_embeds_img2img(
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
            
            results.append((1.0, image_6, "ref"))
            print(f"  Generated 6th image with reference")
        
        # Save images and build response
        slider_images = []
        for alpha, img, ref_flag in results:
            if ref_flag == "ref":
                filename = f"eval_alphaRef_{alpha:.2f}_{timestamp}.png"
            else:
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
        
        # ========== LLM STYLE TRANSFER (for new locations only) ==========
        original_location = final_selection.get("location", "")
        is_new_location = location and location != original_location
        
        if is_new_location:
            print(f"\n{'='*80}")
            print(f"[LLM STYLE TRANSFER] Applying style transfer to new location")
            print(f"{'='*80}")
            print(f"  Original location: {original_location}")
            print(f"  New location: {location}")
            
            # Find reference image from original location's slider folder
            original_slider_dir = os.path.join(session_folder, "slider", original_location.replace(" ", "_"))
            reference_image_path = None
            
            if os.path.exists(original_slider_dir):
                # Look for *alphaRef_1.00*.png patterns
                import glob
                patterns = [
                    os.path.join(original_slider_dir, "*alphaRef_1.00*.png"),
                    os.path.join(original_slider_dir, "*current_alphaRef_1.00*.png"),
                    os.path.join(original_slider_dir, "eval_alphaRef_1.00*.png")
                ]
                
                for pattern in patterns:
                    matches = glob.glob(pattern)
                    if matches:
                        reference_image_path = matches[0]
                        break
            
            if reference_image_path and os.path.exists(reference_image_path):
                print(f"  Reference image: {os.path.basename(reference_image_path)}")
                
                # Build the style transfer prompt
                style_transfer_prompt = (
                    f"This user selected this image as their preferred example of a {adjective} {original_location}. "
                    f"Generate a {adjective} {location} that matches this user's personal aesthetic"
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
                print(f"  ⚠️ Reference image not found in {original_slider_dir}")
            
            # ========== LLM BASELINE (for new locations) ==========
            print(f"\n[LLM BASELINE] Generating baseline image for new location")
            baseline_prompt = f"{adjective} {location}"
            print(f"  Prompt: {baseline_prompt}")
            
            try:
                baseline_output = os.path.join(slider_output_dir, "llm_baseline.png")
                generated_baseline = generate_baseline_image(
                    user_input=baseline_prompt,
                    output_folder=slider_output_dir,
                    output_filename="llm_baseline.png"
                )
                print(f"  ✅ Baseline image saved: llm_baseline.png")
            except Exception as e:
                print(f"  ⚠️ Baseline generation failed: {str(e)}")
        
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
        
    except HTTPException:
        raise
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
        
        raise HTTPException(500, str(e))


# Global SDXL runner for eval (cached to avoid reloading)
_eval_sdxl_runner = None
_eval_slider_fuser = None


# ============== Health check ==============

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "eval_server",
        "predefined_sessions": len(list_predefined_sessions(str(PREDEFINED_INPUT_DIR))),
        "active_sessions": len(eval_sessions)
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

