"""
Tracking utilities for SDXL generation pipeline.

Logs all inputs, transformations, and outputs for debugging and analysis.
Creates tracking.json in each session folder.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


class GenerationTracker:
    """
    Tracks SDXL generation pipeline for debugging and analysis.
    
    Records:
    - Descriptor and concepts
    - Weight transformations (raw → normalized → gains)
    - Prompt composition (descriptor + concepts)
    - Reference images
    - User selections and PBO updates
    """
    
    def __init__(self, session_path: Path, session_id: str, stage: str, descriptor: str):
        self.session_path = Path(session_path)
        self.tracking_file = self.session_path / "tracking.json"
        
        # Initialize or load tracking data
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                "session_id": session_id,
                "descriptor": descriptor,
                "stage": stage,
                "created_at": datetime.now().isoformat(),
                "concepts": [],
                "rounds": []
            }
    
    def set_concepts(self, concepts: List[Dict]) -> None:
        """
        Store concept definitions.
        
        Args:
            concepts: List of concept dicts with 'id', 'label', 'centroid'
        """
        self.data["concepts"] = [
            {
                "id": c["id"],
                "label": c["label"],
                "centroid_shape": np.array(c["centroid"]).shape if "centroid" in c else None,
                "source": "tag_cluster"
            }
            for c in concepts
        ]
        self._save()
    
    def start_round(
        self,
        round_number: int,
        reference_image: Optional[str] = None
    ) -> None:
        """Start a new generation round."""
        self.data["rounds"].append({
            "round_number": round_number,
            "reference_image": reference_image,
            "started_at": datetime.now().isoformat(),
            "proposals": []
        })
        self._save()
    
    def add_proposal(
        self,
        proposal_index: int,
        w_raw: np.ndarray,
        concepts: List[Dict],
        descriptor: Optional[str],
        pos_phrases: List[Tuple[str, float]],
        neg_phrases: List[str],
        generated_image_path: str,
        seed: int,
        generation_params: Dict[str, Any]
    ) -> None:
        """
        Record a single proposal's complete pipeline.
        
        Args:
            proposal_index: Index of this proposal (0-3)
            w_raw: Raw weight vector
            concepts: List of concept dicts
            descriptor: User descriptor (if used)
            pos_phrases: List of (phrase, gain) tuples
            neg_phrases: List of negative phrase strings
            generated_image_path: Path to generated image
            seed: Random seed used
            generation_params: Dict with strength, steps, guidance_scale, etc.
        """
        try:
            from backend.sdxl_integration import normalize_simplex, compute_gains
        except ImportError:
            from sdxl_integration import normalize_simplex, compute_gains
        
        # Get current round
        if not self.data["rounds"]:
            self.start_round(1)
        current_round = self.data["rounds"][-1]
        
        # Compute weight transformations
        w_norm = normalize_simplex(w_raw.copy())
        gains = compute_gains(w_norm)
        
        mean_w = float(np.mean(w_norm))
        std_w = float(np.std(w_norm))
        z_scores = (w_norm - mean_w) / (std_w + 1e-8)
        
        # Build concept breakdown
        concept_breakdown = []
        for i, concept in enumerate(concepts):
            concept_breakdown.append({
                "concept_id": concept["id"],
                "label": concept["label"],
                "weight_raw": float(w_raw[i]),
                "weight_normalized": float(w_norm[i]),
                "z_score": float(z_scores[i]),
                "gain_before_clip": float(1.0 + 0.4 * z_scores[i]),
                "gain_after_clip": float(gains[i]),
                "rank": int(np.where(np.argsort(w_norm)[::-1] == i)[0][0]) + 1
            })
        
        # Determine which concepts are in positive/negative phrases
        pos_labels = [p[0] for p in pos_phrases if p[0] != descriptor]
        neg_labels = neg_phrases if neg_phrases else []
        
        for concept_info in concept_breakdown:
            concept_info["included_positive"] = concept_info["label"] in pos_labels
            concept_info["included_negative"] = concept_info["label"] in neg_labels
        
        # Extract gains from pos_phrases for normalization
        pos_gains = [g for _, g in pos_phrases]
        total_gain = sum(pos_gains) if pos_gains else 1.0
        
        # Build prompt composition
        prompt_composition = {
            "positive_phrases": [],
            "negative_phrases": neg_labels
        }
        
        for phrase, gain in pos_phrases:
            phrase_info = {
                "text": phrase,
                "gain_original": float(gain),
                "gain_normalized": float(gain / total_gain) if total_gain > 0 else 0.0,
                "is_descriptor": phrase == descriptor
            }
            prompt_composition["positive_phrases"].append(phrase_info)
        
        # Create proposal record
        proposal_record = {
            "proposal_index": proposal_index,
            "seed": seed,
            "generated_image": generated_image_path,
            "generated_at": datetime.now().isoformat(),
            
            "weight_statistics": {
                "raw_weights": [float(x) for x in w_raw],
                "normalized_weights": [float(x) for x in w_norm],
                "mean": float(mean_w),
                "std": float(std_w),
                "min": float(w_norm.min()),
                "max": float(w_norm.max())
            },
            
            "concept_breakdown": concept_breakdown,
            "prompt_composition": prompt_composition,
            "generation_params": generation_params
        }
        
        current_round["proposals"].append(proposal_record)
        self._save()
    
    def record_selection(
        self,
        selected_index: int,
        all_indices: List[int]
    ) -> None:
        """
        Record user selection and PBO update.
        
        Args:
            selected_index: Index of selected image (0-3)
            all_indices: All image indices in this round
        """
        if not self.data["rounds"]:
            return
        
        current_round = self.data["rounds"][-1]
        
        # Record selection
        current_round["user_selection"] = {
            "selected_index": selected_index,
            "selected_image": current_round["proposals"][selected_index]["generated_image"],
            "selection_timestamp": datetime.now().isoformat()
        }
        
        # Record PBO duels
        duels = []
        for idx in all_indices:
            if idx != selected_index:
                duels.append({
                    "winner_index": selected_index,
                    "loser_index": idx,
                    "strength": 1.0,
                    "type": "strong_duel"
                })
        
        current_round["pbo_update"] = {
            "duels_added": duels,
            "num_duels": len(duels),
            "gp_fitted": True
        }
        
        self._save()
    
    def add_round_summary(
        self,
        weight_changes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add summary of how weights changed after user selection.
        
        Args:
            weight_changes: Dict with weight evolution info
        """
        if not self.data["rounds"]:
            return
        
        current_round = self.data["rounds"][-1]
        current_round["summary"] = {
            "total_proposals": len(current_round["proposals"]),
            "weight_changes": weight_changes,
            "completed_at": datetime.now().isoformat()
        }
        
        self._save()
    
    def get_tracking_data(self) -> Dict:
        """Get complete tracking data."""
        return self.data
    
    def _save(self) -> None:
        """Save tracking data to file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        # Also save human-readable summary
        self._save_readable_summary()
    
    def _save_readable_summary(self) -> None:
        """Generate and save human-readable tracking summary."""
        readable_file = self.session_path / "tracking_readable.txt"
        
        with open(readable_file, 'w') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("SDXL GENERATION TRACKING - HUMAN READABLE SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Session: {self.data['session_id']}\n")
            f.write(f"Stage: {self.data['stage']}\n")
            f.write(f"Descriptor: {self.data['descriptor']}\n")
            f.write(f"Created: {self.data['created_at']}\n")
            f.write(f"Total Rounds: {len(self.data['rounds'])}\n\n")
            
            # Concept Evolution Section
            f.write("="*80 + "\n")
            f.write("CONCEPT WEIGHT EVOLUTION\n")
            f.write("="*80 + "\n\n")
            
            for concept in self.data['concepts']:
                f.write(f"Concept: {concept['label']} (ID: {concept['id']})\n")
                f.write("-" * 60 + "\n")
                
                # Track this concept across all rounds
                for round_idx, round_data in enumerate(self.data['rounds'], 1):
                    f.write(f"\n  Round {round_idx}:\n")
                    
                    for proposal in round_data['proposals']:
                        # Find this concept in the proposal
                        concept_info = None
                        for c in proposal['concept_breakdown']:
                            if c['concept_id'] == concept['id']:
                                concept_info = c
                                break
                        
                        if concept_info:
                            f.write(f"    Proposal {proposal['proposal_index']}:\n")
                            f.write(f"      Weight (raw):        {concept_info['weight_raw']:.4f}\n")
                            f.write(f"      Weight (normalized): {concept_info['weight_normalized']:.4f}\n")
                            f.write(f"      Z-score:            {concept_info['z_score']:.4f}\n")
                            f.write(f"      Gain (final):       {concept_info['gain_after_clip']:.4f}\n")
                            f.write(f"      Rank:               {concept_info['rank']}\n")
                            f.write(f"      In positive prompt: {concept_info['included_positive']}\n")
                            f.write(f"      In negative prompt: {concept_info['included_negative']}\n")
                    
                    # Show which proposal was selected
                    if 'user_selection' in round_data:
                        sel_idx = round_data['user_selection']['selected_index']
                        f.write(f"\n  → SELECTED: Proposal {sel_idx}\n")
                
                f.write("\n" + "="*60 + "\n\n")
            
            # Round Summary Section
            f.write("\n" + "="*80 + "\n")
            f.write("ROUND SUMMARIES\n")
            f.write("="*80 + "\n\n")
            
            for round_data in self.data['rounds']:
                round_num = round_data['round_number']
                f.write(f"Round {round_num}:\n")
                f.write("-" * 60 + "\n")
                f.write(f"Reference Image: {round_data.get('reference_image', 'None')}\n")
                f.write(f"Started: {round_data['started_at']}\n\n")
                
                # Each proposal
                for proposal in round_data['proposals']:
                    f.write(f"  Image {proposal['proposal_index']}: {proposal['generated_image']}\n")
                    f.write(f"    Seed: {proposal['seed']}\n")
                    f.write(f"    Generated: {proposal['generated_at']}\n")
                    
                    # Generation params
                    params = proposal['generation_params']
                    f.write(f"    Mode: {params['mode']}, Strength: {params.get('strength', 'N/A')}\n")
                    f.write(f"    Steps: {params['steps']}, Guidance: {params['guidance_scale']}\n\n")
                    
                    # Prompt composition
                    f.write("    Prompt Composition:\n")
                    comp = proposal['prompt_composition']
                    
                    f.write("      Positive phrases:\n")
                    for phrase in comp['positive_phrases']:
                        marker = " [DESCRIPTOR]" if phrase['is_descriptor'] else ""
                        f.write(f"        - {phrase['text']}: gain={phrase['gain_normalized']:.3f}{marker}\n")
                    
                    if comp['negative_phrases']:
                        f.write("      Negative phrases:\n")
                        for phrase in comp['negative_phrases']:
                            f.write(f"        - {phrase}\n")
                    
                    # Weight stats
                    stats = proposal['weight_statistics']
                    f.write(f"\n    Weight Statistics:\n")
                    f.write(f"      Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}\n")
                    f.write(f"      Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                    
                    # Top concepts by weight
                    f.write(f"\n    Top Concepts by Weight:\n")
                    sorted_concepts = sorted(
                        proposal['concept_breakdown'],
                        key=lambda x: x['weight_normalized'],
                        reverse=True
                    )[:5]
                    
                    for c in sorted_concepts:
                        status = "✓ positive" if c['included_positive'] else ("✗ negative" if c['included_negative'] else "  not used")
                        f.write(f"      {c['rank']}. {c['label']:20s} weight={c['weight_normalized']:.4f} gain={c['gain_after_clip']:.3f} [{status}]\n")
                    
                    f.write("\n")
                
                # User selection
                if 'user_selection' in round_data:
                    sel = round_data['user_selection']
                    f.write(f"\n  ★ SELECTED: Image {sel['selected_index']} - {sel['selected_image']}\n")
                    f.write(f"    Timestamp: {sel['selection_timestamp']}\n")
                
                # PBO update
                if 'pbo_update' in round_data:
                    pbo = round_data['pbo_update']
                    f.write(f"\n  PBO Update:\n")
                    f.write(f"    Duels added: {pbo['num_duels']}\n")
                    f.write(f"    GP fitted: {pbo['gp_fitted']}\n")
                    for duel in pbo['duels_added']:
                        f.write(f"      Winner: {duel['winner_index']} > Loser: {duel['loser_index']} (strength: {duel['strength']})\n")
                
                f.write("\n" + "="*60 + "\n\n")
    
    def print_summary(self) -> None:
        """Print human-readable summary."""
        print(f"\n{'='*80}")
        print(f"GENERATION TRACKING SUMMARY")
        print(f"{'='*80}")
        print(f"Session: {self.data['session_id']}")
        print(f"Stage: {self.data['stage']}")
        print(f"Descriptor: {self.data['descriptor']}")
        print(f"Concepts: {len(self.data['concepts'])}")
        print(f"Rounds: {len(self.data['rounds'])}")
        
        for round_data in self.data["rounds"]:
            print(f"\n{'-'*80}")
            print(f"Round {round_data['round_number']}")
            print(f"Reference: {round_data.get('reference_image', 'None')}")
            print(f"Proposals: {len(round_data['proposals'])}")
            
            if "user_selection" in round_data:
                sel = round_data["user_selection"]
                print(f"Selected: Index {sel['selected_index']} - {sel['selected_image']}")
            
            if "pbo_update" in round_data:
                pbo = round_data["pbo_update"]
                print(f"PBO Duels Added: {pbo['num_duels']}")
        
        print(f"{'='*80}\n")


def create_tracker(
    session_path: Path,
    session_id: str,
    stage: str,
    descriptor: str
) -> GenerationTracker:
    """
    Create or load generation tracker for a session.
    
    Args:
        session_path: Path to session folder
        session_id: Session ID
        stage: Stage name (e.g., "impression", "spatial")
        descriptor: User descriptor
    
    Returns:
        GenerationTracker instance
    """
    return GenerationTracker(session_path, session_id, stage, descriptor)


def extract_proposal_data(
    w: np.ndarray,
    concepts: List[Dict],
    descriptor: Optional[str],
    pos_phrases: List[Tuple[str, float]],
    neg_phrases: List[str],
    seed: int,
    **generation_params
) -> Dict[str, Any]:
    """
    Extract all data for a proposal (helper for manual tracking).
    
    Returns dict suitable for JSON serialization.
    """
    try:
        from backend.sdxl_integration import normalize_simplex, compute_gains
    except ImportError:
        from sdxl_integration import normalize_simplex, compute_gains
    
    w_norm = normalize_simplex(w.copy())
    gains = compute_gains(w_norm)
    mean_w = float(np.mean(w_norm))
    std_w = float(np.std(w_norm))
    
    return {
        "weight_vector": {
            "raw": [float(x) for x in w],
            "normalized": [float(x) for x in w_norm],
            "mean": mean_w,
            "std": std_w,
            "gains": [float(x) for x in gains]
        },
        "descriptor": descriptor,
        "positive_phrases": [(p, float(g)) for p, g in pos_phrases],
        "negative_phrases": neg_phrases,
        "seed": seed,
        "generation_params": generation_params
    }

