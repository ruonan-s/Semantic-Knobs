"""
Slot-Based Refinement Session Manager

Orchestrates the complete 4-stage refinement pipeline with SDXL integration:
1. Deduplication (automatic)
2. Semantic Slots (automatic, via Gemini)
3. Elimination (3-5 user rounds)
4. Weight Refinement (2-3 user rounds)
"""

import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict, List
from enum import Enum

from .deduplication import deduplicate_tags, DeduplicationResult
from .semantic_slots import create_semantic_slots_sync, SemanticSlot, SlotCreationResult
from .slots_based_elimination import (
    SlotBasedElimination, 
    RoundConfig, 
    RoundFeedback,
    RoundType,
    ImageConfig
)
from .attention_gradient import (
    AttentionGradientRefiner,
    RefinementRoundConfig,
    RefinementFeedback
)


class RefinementStage(Enum):
    """Current stage in the refinement pipeline."""
    INIT = "init"
    DEDUPLICATION = "deduplication"
    SLOT_CREATION = "slot_creation"
    ELIMINATION = "elimination"
    WEIGHT_REFINEMENT = "weight_refinement"
    COMPLETE = "complete"


@dataclass
class SlotRefinementConfig:
    """Configuration for the refinement session."""
    base_prompt: str
    elimination_max_rounds: int = 6
    refinement_max_rounds: int = 4
    confidence_threshold: float = 0.75
    learning_rate: float = 0.12
    convergence_threshold: float = 0.04
    image_width: int = 512
    image_height: int = 512
    num_inference_steps: int = 30
    negative_prompt: str = "illustration, painted, drawing, cartoon, anime, isometric, diorama, miniature, 3D render, CGI, concept art, stylized, toon shading, human"


@dataclass
class SessionState:
    """Serializable session state."""
    session_id: str
    stage: str
    base_prompt: str
    raw_tags: List[str]
    dedup_result: Optional[Dict] = None
    slots: Optional[List[Dict]] = None
    elimination_state: Optional[Dict] = None
    refinement_state: Optional[Dict] = None
    round_images: List[Dict] = field(default_factory=list)
    final_result: Optional[Dict] = None
    created_at: str = ""
    updated_at: str = ""


class SlotRefinementSession:
    """
    Complete session manager for slot-based refinement.
    
    Usage:
        session = SlotRefinementSession(session_id, session_folder, pipe=sdxl_pipe)
        
        # Initialize (Stages 1 & 2 - automatic)
        init_result = session.initialize_from_exploration(tag_preferences_path)
        
        # User interaction loop (Stages 3 & 4)
        while not session.is_complete:
            round_data = session.generate_round()
            # ... show images to user, get selection ...
            result = session.submit_feedback(selected_idx)
        
        final = session.finalize()
    """
    
    def __init__(
        self,
        session_id: str,
        session_folder: str,
        pipe: Any = None,
        sdxl_runner: Any = None,
        config: Optional[SlotRefinementConfig] = None
    ):
        self.session_id = session_id
        self.session_folder = Path(session_folder)
        self.pipe = pipe
        self.sdxl_runner = sdxl_runner
        self.config = config or SlotRefinementConfig(base_prompt="")
        
        # State
        self.stage = RefinementStage.INIT
        self.raw_tags: List[str] = []
        self.dedup_result: Optional[DeduplicationResult] = None
        self.slots: List[SemanticSlot] = []
        
        # Stage managers
        self.elimination: Optional[SlotBasedElimination] = None
        self.refiner: Optional[AttentionGradientRefiner] = None
        
        # Current round tracking
        self.current_round_config: Optional[Any] = None
        self.current_round_images: List[str] = []
        self.round_count = 0
        
        # Attention map storage
        self.attention_maps_cache: Dict[str, Dict[str, float]] = {}
        
        # Paths
        self.slot_folder = self.session_folder / "slot_refinement"
        self.slot_folder.mkdir(parents=True, exist_ok=True)
        self.state_file = self.slot_folder / "state.json"
        
        # Try to restore state
        if self.state_file.exists():
            self._load_state()
    
    @property
    def is_complete(self) -> bool:
        return self.stage == RefinementStage.COMPLETE
    
    def initialize_from_exploration(self, tag_preferences_path: str) -> Dict:
        """
        Initialize the session from exploration output.
        Runs Stage 1 (Deduplication) and Stage 2 (Slot Creation).
        
        Args:
            tag_preferences_path: Path to tag_preferences.json or concept_weights.json
            
        Returns:
            Initialization summary with slots created
        """
        print(f"[SlotRefine] Initializing from {tag_preferences_path}")
        
        # Load positive tags from exploration
        with open(tag_preferences_path) as f:
            prefs = json.load(f)
        
        # Handle both tag_preferences.json and concept_weights.json formats
        if "positive" in prefs:
            # tag_preferences.json format
            self.raw_tags = prefs.get("positive", [])
            self.config.base_prompt = prefs.get("descriptor", self.config.base_prompt)
        elif "concept_weights" in prefs:
            # concept_weights.json format - extract positive tags
            self.raw_tags = [
                cw["label"] for cw in prefs["concept_weights"]
                if cw.get("category") == "positive" or cw.get("score", 0) > 0.5
            ]
            # Try to extract descriptor from session_id
            session_id = prefs.get("session_id", "")
            if session_id:
                # e.g. "eval_Calm_Home_Office_Sample_2026-01-28_03-58-05" -> "Calm Home Office"
                parts = session_id.replace("eval_", "").split("_Sample")[0].replace("_", " ")
                self.config.base_prompt = parts or self.config.base_prompt
            print(f"[SlotRefine] Loaded {len(self.raw_tags)} positive tags from concept_weights.json")
        else:
            raise ValueError("Unrecognized file format - need 'positive' or 'concept_weights' key")
        
        if not self.raw_tags:
            raise ValueError("No positive tags found in exploration output")
        
        print(f"[SlotRefine] Loaded {len(self.raw_tags)} positive tags")
        
        # Stage 1: Deduplication
        self.stage = RefinementStage.DEDUPLICATION
        self.dedup_result = deduplicate_tags(self.raw_tags)
        
        print(f"[SlotRefine] Deduplication: {self.dedup_result.original_count} → {self.dedup_result.deduplicated_count}")
        
        # Stage 2: Semantic Slot Creation via LLM
        self.stage = RefinementStage.SLOT_CREATION
        slot_result = create_semantic_slots_sync(
            base_prompt=self.config.base_prompt,
            tags=self.dedup_result.unique_tags
        )
        self.slots = slot_result.slots
        
        print(f"[SlotRefine] Created {len(self.slots)} semantic slots")
        
        # Initialize Stage 3: Elimination
        self.elimination = SlotBasedElimination(
            base_prompt=self.config.base_prompt,
            slots=self.slots,
            max_rounds=self.config.elimination_max_rounds,
            confidence_threshold=self.config.confidence_threshold
        )
        
        self.stage = RefinementStage.ELIMINATION
        self._save_state()
        
        return {
            "status": "initialized",
            "stage": self.stage.value,
            "deduplication": {
                "original_count": self.dedup_result.original_count,
                "deduplicated_count": self.dedup_result.deduplicated_count,
                "duplicates_merged": self.dedup_result.duplicates_removed
            },
            "slot_creation": {
                "num_slots": len(self.slots),
                "slots": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "tags": s.tags,
                        "importance": s.importance
                    }
                    for s in self.slots
                ],
                "reasoning": slot_result.reasoning
            }
        }
    
    def generate_round(self) -> Dict:
        """
        Generate the next round of images.
        
        Returns:
            Round configuration with image paths and metadata
        """
        self.round_count += 1
        
        if self.stage == RefinementStage.ELIMINATION:
            return self._generate_elimination_round()
        elif self.stage == RefinementStage.WEIGHT_REFINEMENT:
            return self._generate_refinement_round()
        else:
            return {"error": f"Cannot generate round in stage: {self.stage.value}"}
    
    def _generate_elimination_round(self) -> Dict:
        """Generate an elimination round."""
        
        round_config = self.elimination.generate_round()
        self.current_round_config = round_config
        
        # Create round folder
        round_folder = self.slot_folder / f"round_{self.round_count}"
        round_folder.mkdir(exist_ok=True)
        
        # Generate images
        image_paths = []
        compositions = []
        
        for img_config in round_config.images:
            prompt = img_config.prompt
            
            # Generate image using SDXL
            image_path = self._generate_image(
                prompt=prompt,
                output_path=round_folder / f"image_{img_config.image_id}.png"
            )
            
            image_paths.append(str(image_path))
            compositions.append({
                "image_id": img_config.image_id,
                "slot_selections": img_config.slot_selections,
                "prompt": prompt,
                "strategy": img_config.strategy
            })
        
        self.current_round_images = image_paths
        self._save_state()
        
        return {
            "round_num": self.round_count,
            "stage": "elimination",
            "round_type": round_config.round_type.value,
            "focus_slot": round_config.focus_slot,
            "images": image_paths,
            "compositions": compositions,
            "slots_status": self._get_slots_status()
        }
    
    def _generate_refinement_round(self) -> Dict:
        """Generate a weight refinement round."""
        
        round_config = self.refiner.generate_round()
        self.current_round_config = round_config
        
        # Create round folder
        round_folder = self.slot_folder / f"round_{self.round_count}"
        round_folder.mkdir(exist_ok=True)
        
        # Generate images
        image_paths = []
        weight_configs = []
        
        for img_data in round_config.images:
            prompt = img_data["prompt"]
            
            # Generate image using SDXL with weighted attention
            image_path = self._generate_image_with_weights(
                prompt=self.config.base_prompt,
                tags=list(img_data["weights"].keys()),
                weights=list(img_data["weights"].values()),
                output_path=round_folder / f"image_{img_data['image_id']}.png"
            )
            
            image_paths.append(str(image_path))
            weight_configs.append({
                "image_id": img_data["image_id"],
                "weights": img_data["weights"],
                "strategy": img_data["strategy"]
            })
        
        self.current_round_images = image_paths
        self._save_state()
        
        return {
            "round_num": self.round_count,
            "stage": "refinement",
            "round_type": "weight_optimization",
            "images": image_paths,
            "weight_configs": weight_configs,
            "current_weights": self.refiner.state.weights
        }
    
    def _generate_image(self, prompt: str, output_path: Path) -> Path:
        """Generate a single image using SDXL."""
        
        full_prompt = f"{self.config.base_prompt} with features: {prompt.replace(self.config.base_prompt + ', ', '')}"
        
        print(f"[SlotRefine] Generating image: {full_prompt[:80]}...")
        
        if self.sdxl_runner:
            # Use SDXLRunner
            images = self.sdxl_runner.generate(
                prompt=full_prompt,
                negative_prompt=self.config.negative_prompt,
                height=self.config.image_height,
                width=self.config.image_width,
                num_inference_steps=self.config.num_inference_steps,
                num_images=1
            )
            if images:
                images[0].save(str(output_path))
                return output_path
        
        elif self.pipe:
            # Use pipeline directly
            result = self.pipe(
                prompt=full_prompt,
                negative_prompt=self.config.negative_prompt,
                height=self.config.image_height,
                width=self.config.image_width,
                num_inference_steps=self.config.num_inference_steps
            )
            if result.images:
                result.images[0].save(str(output_path))
                return output_path
        
        # Fallback: create placeholder
        print(f"[SlotRefine] Warning: No SDXL pipeline, creating placeholder")
        self._create_placeholder_image(output_path, prompt)
        return output_path
    
    def _generate_image_with_weights(
        self,
        prompt: str,
        tags: List[str],
        weights: List[float],
        output_path: Path
    ) -> Path:
        """Generate image with weighted cross-attention."""
        
        # Build weighted prompt
        tag_parts = []
        for tag, weight in zip(tags, weights):
            if weight >= 1.1:
                tag_parts.append(f"({tag}:{weight:.2f})")
            elif weight >= 0.7:
                tag_parts.append(tag)
            else:
                tag_parts.append(f"[{tag}:{weight:.2f}]")
        
        full_prompt = f"{prompt} with features: {', '.join(tag_parts)}"
        
        print(f"[SlotRefine] Generating weighted image...")
        
        if self.sdxl_runner:
            images = self.sdxl_runner.generate(
                prompt=full_prompt,
                negative_prompt=self.config.negative_prompt,
                height=self.config.image_height,
                width=self.config.image_width,
                num_inference_steps=self.config.num_inference_steps,
                num_images=1
            )
            if images:
                images[0].save(str(output_path))
                return output_path
        
        elif self.pipe:
            result = self.pipe(
                prompt=full_prompt,
                negative_prompt=self.config.negative_prompt,
                height=self.config.image_height,
                width=self.config.image_width,
                num_inference_steps=self.config.num_inference_steps
            )
            if result.images:
                result.images[0].save(str(output_path))
                return output_path
        
        # Fallback
        print(f"[SlotRefine] Warning: No SDXL pipeline, creating placeholder")
        self._create_placeholder_image(output_path, full_prompt)
        return output_path
    
    def _create_placeholder_image(self, path: Path, text: str):
        """Create a placeholder image for testing without SDXL."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            img = Image.new('RGB', (self.config.image_width, self.config.image_height), color='#2a2a2a')
            draw = ImageDraw.Draw(img)
            
            # Draw text
            wrapped = text[:60] + "..." if len(text) > 60 else text
            draw.text((20, 20), f"Round {self.round_count}", fill='white')
            draw.text((20, 50), wrapped, fill='#888888')
            
            img.save(str(path))
        except Exception as e:
            print(f"[SlotRefine] Could not create placeholder: {e}")
    
    def submit_feedback(
        self,
        selected_idx: int,
        attention_maps: Optional[List[Dict[str, float]]] = None
    ) -> Dict:
        """
        Submit user feedback for the current round.
        
        Args:
            selected_idx: Index of user-selected image (0-3)
            attention_maps: Optional attention maps for each image
            
        Returns:
            Feedback processing result
        """
        
        if not self.current_round_config:
            return {"error": "No active round"}
        
        # Default attention maps if not provided
        if attention_maps is None:
            attention_maps = self._get_default_attention_maps()
        
        if self.stage == RefinementStage.ELIMINATION:
            return self._process_elimination_feedback(selected_idx, attention_maps)
        elif self.stage == RefinementStage.WEIGHT_REFINEMENT:
            return self._process_refinement_feedback(selected_idx, attention_maps)
        else:
            return {"error": f"Cannot process feedback in stage: {self.stage.value}"}
    
    def _process_elimination_feedback(
        self,
        selected_idx: int,
        attention_maps: List[Dict[str, float]]
    ) -> Dict:
        """Process feedback for elimination round."""
        
        feedback = RoundFeedback(
            selected_image_idx=selected_idx,
            attention_maps=attention_maps
        )
        
        result = self.elimination.process_feedback(self.current_round_config, feedback)
        
        # Check if elimination is complete
        if result["is_complete"]:
            self._transition_to_refinement()
        
        self._save_state()
        
        return {
            "stage": "elimination",
            "newly_resolved": result["newly_resolved"],
            "eliminations": result["eliminations"],
            "is_stage_complete": result["is_complete"],
            "current_winners": self.elimination.state.get_current_winners(),
            "next_stage": self.stage.value,
            "slots_status": self._get_slots_status()
        }
    
    def _process_refinement_feedback(
        self,
        selected_idx: int,
        attention_maps: List[Dict[str, float]]
    ) -> Dict:
        """Process feedback for refinement round."""
        
        feedback = RefinementFeedback(
            selected_image_idx=selected_idx,
            attention_maps=attention_maps
        )
        
        result = self.refiner.process_feedback(self.current_round_config, feedback)
        
        # Check if refinement is complete
        if result["is_converged"] or self.refiner.current_round >= self.config.refinement_max_rounds:
            self.stage = RefinementStage.COMPLETE
        
        self._save_state()
        
        return {
            "stage": "refinement",
            "weight_updates": result["weight_updates"],
            "max_change": result["max_change"],
            "is_converged": result["is_converged"],
            "is_complete": self.stage == RefinementStage.COMPLETE,
            "current_weights": result["current_weights"]
        }
    
    def _transition_to_refinement(self):
        """Transition from elimination to refinement stage."""
        
        # Get winning tags from elimination
        winning_tags = self.elimination.get_final_tags()
        
        # Get initial weights based on slot importance
        initial_weights = {}
        for slot_state in self.elimination.state.slot_states.values():
            if slot_state.current_winner:
                initial_weights[slot_state.current_winner] = slot_state.slot.importance
        
        # Initialize refinement
        self.refiner = AttentionGradientRefiner(
            base_prompt=self.config.base_prompt,
            tags=winning_tags,
            initial_weights=initial_weights,
            learning_rate=self.config.learning_rate,
            convergence_threshold=self.config.convergence_threshold,
            max_rounds=self.config.refinement_max_rounds
        )
        
        self.stage = RefinementStage.WEIGHT_REFINEMENT
        print(f"[SlotRefine] Transitioned to refinement with {len(winning_tags)} tags")
    
    def _get_default_attention_maps(self) -> List[Dict[str, float]]:
        """Generate default attention maps when not provided."""
        
        # For simplicity, assign uniform attention
        all_tags = set()
        if self.stage == RefinementStage.ELIMINATION:
            for slot_state in self.elimination.state.slot_states.values():
                all_tags.update(slot_state.tag_scores.keys())
        elif self.stage == RefinementStage.WEIGHT_REFINEMENT:
            all_tags = set(self.refiner.tags)
        
        default_map = {tag: 0.5 for tag in all_tags}
        return [default_map.copy() for _ in range(4)]
    
    def _get_slots_status(self) -> List[Dict]:
        """Get current status of all slots."""
        if not self.elimination:
            return []
        
        status = []
        for name, state in self.elimination.state.slot_states.items():
            status.append({
                "name": name,
                "winner": state.current_winner,
                "confidence": round(state.confidence, 2),
                "is_resolved": state.is_resolved,
                "remaining_tags": [t for t, s in state.tag_scores.items() if s >= 0.2]
            })
        return status
    
    def finalize(self) -> Dict:
        """Get final result after pipeline completion."""
        
        if self.stage != RefinementStage.COMPLETE:
            # Allow early finalization
            if self.stage == RefinementStage.WEIGHT_REFINEMENT:
                pass  # OK to finalize during refinement
            elif self.stage == RefinementStage.ELIMINATION:
                # Transition to complete with current state
                self._transition_to_refinement()
            else:
                return {"error": f"Cannot finalize in stage: {self.stage.value}"}
        
        # Get final weights
        if self.refiner:
            final_result = self.refiner.get_final_result()
        else:
            # Use elimination results with default weights
            winning_tags = self.elimination.get_final_tags()
            final_result = {
                "base_prompt": self.config.base_prompt,
                "optimized_tags": {tag: 1.0 for tag in winning_tags},
                "final_prompt": f"{self.config.base_prompt} with features: {', '.join(winning_tags)}",
                "rounds_used": self.round_count
            }
        
        # Save final result
        output_path = self.slot_folder / "refined_preferences.json"
        
        output = {
            "base_prompt": self.config.base_prompt,
            "final_tags": [
                {
                    "tag": tag,
                    "attn_map_weight": weight,
                    "usage": "cross_attention_map_scaling"
                }
                for tag, weight in final_result["optimized_tags"].items()
            ],
            "final_prompt": final_result["final_prompt"],
            "summary": {
                "original_tags": len(self.raw_tags),
                "after_deduplication": self.dedup_result.deduplicated_count if self.dedup_result else 0,
                "semantic_slots": len(self.slots),
                "final_tags": len(final_result["optimized_tags"]),
                "total_rounds": self.round_count
            },
            "description": "Attention weights for cross-attention map scaling in SDXL"
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"[SlotRefine] Saved final result to {output_path}")
        
        self.stage = RefinementStage.COMPLETE
        self._save_state()
        
        return output
    
    def get_status(self) -> Dict:
        """Get current session status."""
        return {
            "session_id": self.session_id,
            "stage": self.stage.value,
            "round_count": self.round_count,
            "is_complete": self.is_complete,
            "slots_status": self._get_slots_status() if self.elimination else [],
            "current_weights": self.refiner.state.weights if self.refiner else None
        }
    
    def _save_state(self):
        """Save session state to disk."""
        
        state = {
            "session_id": self.session_id,
            "stage": self.stage.value,
            "base_prompt": self.config.base_prompt,
            "raw_tags": self.raw_tags,
            "round_count": self.round_count,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save deduplication result
        if self.dedup_result:
            state["dedup_result"] = {
                "original_count": self.dedup_result.original_count,
                "deduplicated_count": self.dedup_result.deduplicated_count,
                "unique_tags": self.dedup_result.unique_tags,
                "duplicates_removed": self.dedup_result.duplicates_removed
            }
        
        # Save slots
        if self.slots:
            state["slots"] = [
                {
                    "name": s.name,
                    "description": s.description,
                    "tags": s.tags,
                    "importance": s.importance
                }
                for s in self.slots
            ]
        
        # Save elimination state
        if self.elimination:
            state["elimination"] = {
                "slot_states": {
                    name: {
                        "current_winner": ss.current_winner,
                        "confidence": ss.confidence,
                        "is_resolved": ss.is_resolved,
                        "tag_scores": ss.tag_scores
                    }
                    for name, ss in self.elimination.state.slot_states.items()
                },
                "round_history": self.elimination.state.round_history
            }
        
        # Save refinement state
        if self.refiner:
            state["refinement"] = {
                "weights": self.refiner.state.weights,
                "history": self.refiner.state.history,
                "round_history": self.refiner.round_history
            }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load session state from disk."""
        
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            
            self.session_id = state.get("session_id", self.session_id)
            self.stage = RefinementStage(state.get("stage", "init"))
            self.config.base_prompt = state.get("base_prompt", "")
            self.raw_tags = state.get("raw_tags", [])
            self.round_count = state.get("round_count", 0)
            
            # Restore slots
            if "slots" in state:
                self.slots = [
                    SemanticSlot(
                        name=s["name"],
                        description=s["description"],
                        tags=s["tags"],
                        importance=s.get("importance", 0.5)
                    )
                    for s in state["slots"]
                ]
            
            # Restore elimination
            if "elimination" in state and self.slots:
                self.elimination = SlotBasedElimination(
                    base_prompt=self.config.base_prompt,
                    slots=self.slots,
                    max_rounds=self.config.elimination_max_rounds
                )
                
                # Restore slot states
                for name, ss_data in state["elimination"]["slot_states"].items():
                    if name in self.elimination.state.slot_states:
                        ss = self.elimination.state.slot_states[name]
                        ss.current_winner = ss_data.get("current_winner")
                        ss.confidence = ss_data.get("confidence", 0.0)
                        ss.is_resolved = ss_data.get("is_resolved", False)
                        ss.tag_scores = ss_data.get("tag_scores", ss.tag_scores)
                
                self.elimination.state.round_history = state["elimination"].get("round_history", [])
            
            # Restore refinement
            if "refinement" in state:
                winning_tags = list(state["refinement"]["weights"].keys())
                self.refiner = AttentionGradientRefiner(
                    base_prompt=self.config.base_prompt,
                    tags=winning_tags,
                    initial_weights=state["refinement"]["weights"]
                )
                self.refiner.state.history = state["refinement"].get("history", [])
                self.refiner.round_history = state["refinement"].get("round_history", [])
            
            print(f"[SlotRefine] Restored session state: stage={self.stage.value}, rounds={self.round_count}")
            
        except Exception as e:
            print(f"[SlotRefine] Error loading state: {e}")
