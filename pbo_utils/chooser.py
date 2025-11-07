
# pbo_min/ui/chooser.py
from pathlib import Path
from typing import List, Dict, Any, Optional
from config import NON_INTERACTIVE, NON_INTERACTIVE_CHOICE

def save_round_images(images, out_dir: Path, round_idx: int, prompt_data: Optional[List[Dict[str, Any]]] = None):
    """
    Save round images and optionally their prompt metadata.
    
    Args:
        images: List of PIL Images to save
        out_dir: Output directory
        round_idx: Round index for naming
        prompt_data: Optional list of prompt metadata dicts for each image
    """
    round_dir = out_dir / f"round_{round_idx:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        p = round_dir / f"candidate_{i}.png"
        img.save(p)
        paths.append(p)
    
    # Save prompt metadata if provided
    if prompt_data is not None:
        from utils.prompt_recorder import PromptRecorder
        recorder = PromptRecorder(out_dir)
        recorder.record_round_prompts(round_idx, prompt_data)
    
    return round_dir, paths

def save_single_image(img, out_dir: Path, round_idx: int, candidate_idx: int, prompt_data: Optional[Dict[str, Any]] = None):
    """
    Save a single image and optionally its prompt metadata.
    
    Args:
        img: PIL Image to save
        out_dir: Output directory
        round_idx: Round index for naming
        candidate_idx: Candidate index within the round
        prompt_data: Optional prompt metadata dict for the image
    """
    round_dir = out_dir / f"round_{round_idx:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    
    img_path = round_dir / f"candidate_{candidate_idx}.png"
    img.save(img_path)
    
    # Save prompt metadata if provided
    if prompt_data is not None:
        from utils.prompt_recorder import PromptRecorder
        recorder = PromptRecorder(out_dir)
        recorder.record_single_prompt(round_idx, candidate_idx, prompt_data)
    
    return round_dir, img_path

def ask_user_choice(images_paths, round_dir: Path) -> int:
    print(f"\n[Round {round_dir.name}] Candidates saved:")
    for i, p in enumerate(images_paths):
        print(f"  [{i}] {p}")
    if NON_INTERACTIVE:
        print(f"[AUTO] NON_INTERACTIVE is True → selecting {NON_INTERACTIVE_CHOICE}")
        return max(0, min(len(images_paths)-1, NON_INTERACTIVE_CHOICE))
    while True:
        try:
            idx = int(input(f"Pick the best [0..{len(images_paths)-1}]: ").strip())
            if 0 <= idx < len(images_paths):
                return idx
        except Exception:
            pass
        print("Invalid input, try again.")
