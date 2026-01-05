"""
Evaluation Prototype Utilities

Utility functions for the evaluation prototype that skips refinement
and uses exploration weights directly for slider generation.
"""

import os
import json
import shutil
from datetime import datetime
from typing import Optional, Dict, Any, List


def generate_final_selection_from_exploration(
    session_folder: str,
    exploration_stage: str = "impression"
) -> Dict[str, Any]:
    """
    Generate final_selection.json using exploration concept_weights.json.
    
    This function creates the same format expected by the slider generation
    endpoint, but uses exploration weights instead of refinement weights.
    
    Args:
        session_folder: Path to the session folder
        exploration_stage: Name of the exploration stage (default: "impression")
    
    Returns:
        The final_selection data dictionary
    
    Raises:
        FileNotFoundError: If required files are missing
    """
    # Load concept_weights.json from exploration stage
    concept_weights_path = os.path.join(session_folder, exploration_stage, "concept_weights.json")
    if not os.path.exists(concept_weights_path):
        raise FileNotFoundError(f"concept_weights.json not found at {concept_weights_path}")
    
    with open(concept_weights_path, 'r') as f:
        concept_weights_data = json.load(f)
    
    # Load existing final_selection.json for adjective/location
    final_selection_path = os.path.join(session_folder, "final_selection.json")
    existing_data = {}
    if os.path.exists(final_selection_path):
        with open(final_selection_path, 'r') as f:
            existing_data = json.load(f)
    
    # Extract adjective, location, descriptor
    adjective = existing_data.get("adjective", "")
    location = existing_data.get("location", "")
    descriptor = existing_data.get("descriptor", f"{adjective} {location}".strip())
    session_id = existing_data.get("session_id", os.path.basename(session_folder))
    
    # Build concepts array from concept_weights
    concept_weights = concept_weights_data.get("concept_weights", [])
    concepts = []
    weights_raw = []
    
    for cw in concept_weights:
        concepts.append({
            "id": cw.get("concept_id", f"c{len(concepts)}"),
            "label": cw.get("label", f"concept_{len(concepts)}"),
            "weight": cw.get("weight", 0.0)
        })
        weights_raw.append(cw.get("weight", 0.0))
    
    # Sort by weight descending
    concepts_sorted = sorted(concepts, key=lambda x: x["weight"], reverse=True)
    
    # Build final selection data
    final_selection = {
        "saved_at": datetime.now().isoformat(),
        "session_id": session_id,
        "adjective": adjective,
        "location": location,
        "descriptor": descriptor,
        "stage": exploration_stage,
        "round_number": 0,  # No refinement rounds
        "is_historical_selection": False,
        "image_path": None,  # No specific image selected in eval mode
        "concepts": concepts_sorted,
        "weights_raw": weights_raw,
        "eval_mode": True,  # Flag to indicate eval prototype
        "summary": {
            "total_concepts": len(concepts),
            "top_3_concepts": [c["label"] for c in concepts_sorted[:3]],
            "top_3_weights": [c["weight"] for c in concepts_sorted[:3]]
        }
    }
    
    # Save to session folder
    with open(final_selection_path, "w") as f:
        json.dump(final_selection, f, indent=2)
    
    print(f"[EVAL] Generated final_selection.json from exploration weights")
    print(f"  Path: {final_selection_path}")
    print(f"  Concepts: {len(concepts)}")
    print(f"  Top 3: {[c['label'] for c in concepts_sorted[:3]]}")
    
    return final_selection


def copy_predefined_session(
    predefined_folder: str,
    session_logs_folder: str,
    user_id: Optional[str] = None
) -> str:
    """
    Copy a predefined session to session_logs for a new evaluation run.
    
    Args:
        predefined_folder: Path to the predefined session folder
        session_logs_folder: Path to the session_logs directory
        user_id: Optional user identifier for the session name
    
    Returns:
        Path to the new session folder
    """
    if not os.path.exists(predefined_folder):
        raise FileNotFoundError(f"Predefined session not found: {predefined_folder}")
    
    # Create session name with user_id and timestamp
    predefined_name = os.path.basename(predefined_folder)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if user_id:
        session_name = f"eval_{user_id}_{predefined_name}_{timestamp}"
    else:
        session_name = f"eval_{predefined_name}_{timestamp}"
    
    new_session_folder = os.path.join(session_logs_folder, session_name)
    
    # Copy the entire folder
    shutil.copytree(predefined_folder, new_session_folder)
    
    print(f"[EVAL] Copied predefined session to session_logs")
    print(f"  From: {predefined_folder}")
    print(f"  To: {new_session_folder}")
    
    return new_session_folder


def list_predefined_sessions(predefined_input_folder: str) -> List[Dict[str, Any]]:
    """
    List all available predefined sessions.
    
    Args:
        predefined_input_folder: Path to the predefined_input directory
    
    Returns:
        List of session info dictionaries
    """
    sessions = []
    
    if not os.path.exists(predefined_input_folder):
        return sessions
    
    for name in os.listdir(predefined_input_folder):
        folder_path = os.path.join(predefined_input_folder, name)
        if not os.path.isdir(folder_path):
            continue
        
        # Check for required files
        impression_folder = os.path.join(folder_path, "impression")
        has_impression = os.path.exists(impression_folder)
        has_images = has_impression and any(
            f.endswith('.png') for f in os.listdir(impression_folder)
        ) if has_impression else False
        
        # Load metadata if available
        final_selection_path = os.path.join(folder_path, "final_selection.json")
        metadata = {}
        if os.path.exists(final_selection_path):
            try:
                with open(final_selection_path, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                pass
        
        sessions.append({
            "name": name,
            "path": folder_path,
            "has_impression": has_impression,
            "has_images": has_images,
            "adjective": metadata.get("adjective", ""),
            "location": metadata.get("location", ""),
            "descriptor": metadata.get("descriptor", name),
            "valid": has_impression and has_images
        })
    
    return sessions


def validate_session_for_eval(session_folder: str) -> Dict[str, Any]:
    """
    Validate that a session has all required files for evaluation.
    
    Args:
        session_folder: Path to the session folder
    
    Returns:
        Validation result with status and any missing items
    """
    required_files = [
        ("impression/impression.json", "Impression concepts JSON"),
        ("impression/visual_tags.json", "Visual tags JSON"),
        ("final_selection.json", "Final selection JSON (for adjective/location)"),
    ]
    
    required_images = True  # At least one image in impression folder
    
    missing = []
    
    for file_path, description in required_files:
        full_path = os.path.join(session_folder, file_path)
        if not os.path.exists(full_path):
            missing.append({"file": file_path, "description": description})
    
    # Check for images
    impression_folder = os.path.join(session_folder, "impression")
    has_images = False
    if os.path.exists(impression_folder):
        has_images = any(f.endswith('.png') for f in os.listdir(impression_folder))
    
    if not has_images:
        missing.append({"file": "impression/*.png", "description": "Impression images"})
    
    return {
        "valid": len(missing) == 0,
        "missing": missing,
        "session_folder": session_folder
    }


def create_eval_session_log(
    session_folder: str,
    user_id: str,
    event_type: str,
    data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an evaluation event to the session's eval_log.json.
    
    Args:
        session_folder: Path to the session folder
        user_id: User identifier
        event_type: Type of event (e.g., "session_start", "exploration_complete", "slider_generated")
        data: Optional additional data to log
    """
    log_path = os.path.join(session_folder, "eval_log.json")
    
    # Load existing log or create new
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = {
            "user_id": user_id,
            "session_folder": session_folder,
            "created_at": datetime.now().isoformat(),
            "events": []
        }
    
    # Add event
    event = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "data": data or {}
    }
    log_data["events"].append(event)
    
    # Save
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)


