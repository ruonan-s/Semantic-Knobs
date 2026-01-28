"""
Complete Pipeline: Integrating all 4 stages
"""

from dataclasses import dataclass
from typing import Callable, Optional
import json


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline."""
    base_prompt: str
    api_key: str
    llm_model: str = "openai-gpt-4o"
    elimination_max_rounds: int = 5
    refinement_max_rounds: int = 3
    confidence_threshold: float = 0.75
    learning_rate: float = 0.12
    convergence_threshold: float = 0.04


@dataclass
class PipelineState:
    """Current state of the pipeline."""
    stage: str  # "deduplication", "slot_creation", "elimination", "refinement", "complete"
    raw_tags: list[str]
    deduplicated_tags: Optional[list[str]] = None
    slots: Optional[list[SemanticSlot]] = None
    elimination_state: Optional[SlotBasedElimination] = None
    refinement_state: Optional[AttentionGradientRefiner] = None
    final_result: Optional[dict] = None


class TagRefinementPipeline:
    """
    Complete pipeline for tag refinement.
    
    Usage:
        pipeline = TagRefinementPipeline(config, raw_positive_tags)
        
        # Stage 1 & 2: Automatic (no user interaction)
        await pipeline.initialize()
        
        # Stage 3 & 4: User interaction loop
        while not pipeline.is_complete():
            round_config = pipeline.get_next_round()
            # ... show images to user, get selection and attention ...
            pipeline.process_feedback(selected_idx, attention_maps)
        
        result = pipeline.get_final_result()
    """
    
    def __init__(self, config: PipelineConfig, raw_positive_tags: list[str]):
        self.config = config
        self.state = PipelineState(
            stage="deduplication",
            raw_tags=raw_positive_tags
        )
    
    async def initialize(self) -> dict:
        """
        Run Stage 1 (Deduplication) and Stage 2 (Slot Creation).
        These don't require user interaction.
        
        Returns summary of initialization.
        """
        # Stage 1: Deduplication
        dedup_result = deduplicate_tags(self.state.raw_tags)
        self.state.deduplicated_tags = dedup_result.unique_tags
        self.state.stage = "slot_creation"
        
        # Stage 2: LLM Slot Creation
        slot_result = await create_semantic_slots(
            base_prompt=self.config.base_prompt,
            tags=self.state.deduplicated_tags,
            api_key=self.config.api_key,
            model=self.config.llm_model
        )
        self.state.slots = slot_result.slots
        
        # Initialize Stage 3: Elimination
        self.state.elimination_state = SlotBasedElimination(
            base_prompt=self.config.base_prompt,
            slots=self.state.slots,
            max_rounds=self.config.elimination_max_rounds,
            confidence_threshold=self.config.confidence_threshold
        )
        self.state.stage = "elimination"
        
        return {
            "deduplication": {
                "original_count": dedup_result.original_count,
                "deduplicated_count": dedup_result.deduplicated_count,
                "duplicates_merged": dedup_result.duplicates_removed
            },
            "slot_creation": {
                "num_slots": len(slot_result.slots),
                "slots": [
                    {"name": s.name, "tags": s.tags, "importance": s.importance}
                    for s in slot_result.slots
                ],
                "reasoning": slot_result.reasoning
            }
        }
    
    def initialize_sync(self) -> dict:
        """Synchronous version of initialize."""
        # Stage 1: Deduplication
        dedup_result = deduplicate_tags(self.state.raw_tags)
        self.state.deduplicated_tags = dedup_result.unique_tags
        self.state.stage = "slot_creation"
        
        # Stage 2: LLM Slot Creation
        slot_result = create_semantic_slots_sync(
            base_prompt=self.config.base_prompt,
            tags=self.state.deduplicated_tags,
            api_key=self.config.api_key,
            model=self.config.llm_model
        )
        self.state.slots = slot_result.slots
        
        # Initialize Stage 3: Elimination
        self.state.elimination_state = SlotBasedElimination(
            base_prompt=self.config.base_prompt,
            slots=self.state.slots,
            max_rounds=self.config.elimination_max_rounds,
            confidence_threshold=self.config.confidence_threshold
        )
        self.state.stage = "elimination"
        
        return {
            "deduplication": {
                "original_count": dedup_result.original_count,
                "deduplicated_count": dedup_result.deduplicated_count,
                "duplicates_merged": dedup_result.duplicates_removed
            },
            "slot_creation": {
                "num_slots": len(slot_result.slots),
                "slots": [
                    {"name": s.name, "tags": s.tags, "importance": s.importance}
                    for s in slot_result.slots
                ],
                "reasoning": slot_result.reasoning
            }
        }
    
    def get_next_round(self) -> dict:
        """
        Get the next round configuration.
        
        Returns dict with:
        - stage: current stage
        - round_num: round number within stage
        - round_type: type of round
        - images: list of image configs to generate
        """
        if self.state.stage == "elimination":
            round_config = self.state.elimination_state.generate_round()
            return {
                "stage": "elimination",
                "round_num": round_config.round_num,
                "round_type": round_config.round_type.value,
                "focus_slot": round_config.focus_slot,
                "images": [
                    {
                        "image_id": img.image_id,
                        "slot_selections": img.slot_selections,
                        "prompt": img.prompt,
                        "strategy": img.strategy
                    }
                    for img in round_config.images
                ],
                "_round_config": round_config  # For internal use
            }
        
        elif self.state.stage == "refinement":
            round_config = self.state.refinement_state.generate_round()
            return {
                "stage": "refinement",
                "round_num": round_config.round_num,
                "round_type": "weight_optimization",
                "images": round_config.images,
                "_round_config": round_config
            }
        
        else:
            return {"stage": self.state.stage, "error": "No rounds available"}
    
    def process_feedback(
        self,
        selected_image_idx: int,
        attention_maps: list[dict[str, float]],
        round_config: dict = None
    ) -> dict:
        """
        Process user feedback for current round.
        
        Args:
            selected_image_idx: Index of user-selected image (0-3)
            attention_maps: Cross-attention maps for each image
            round_config: The round config (if not provided, uses last generated)
            
        Returns dict with update summary and next steps.
        """
        if self.state.stage == "elimination":
            feedback = RoundFeedback(
                selected_image_idx=selected_image_idx,
                attention_maps=attention_maps
            )
            
            # Get the actual round config object
            if round_config and "_round_config" in round_config:
                config = round_config["_round_config"]
            else:
                # This shouldn't happen in normal flow
                raise ValueError("Round config required for processing feedback")
            
            result = self.state.elimination_state.process_feedback(config, feedback)
            
            # Check if elimination is complete
            if result["is_complete"]:
                self._transition_to_refinement()
            
            return {
                "stage": "elimination",
                "newly_resolved": result["newly_resolved"],
                "eliminations": result["eliminations"],
                "is_stage_complete": result["is_complete"],
                "current_winners": self.state.elimination_state.state.get_current_winners()
            }
        
        elif self.state.stage == "refinement":
            feedback = RefinementFeedback(
                selected_image_idx=selected_image_idx,
                attention_maps=attention_maps
            )
            
            if round_config and "_round_config" in round_config:
                config = round_config["_round_config"]
            else:
                raise ValueError("Round config required for processing feedback")
            
            result = self.state.refinement_state.process_feedback(config, feedback)
            
            # Check if refinement is complete
            if result["is_converged"] or self.state.refinement_state.current_round >= self.config.refinement_max_rounds:
                self.state.stage = "complete"
                self.state.final_result = self.state.refinement_state.get_final_result()
            
            return {
                "stage": "refinement",
                "weight_updates": result["weight_updates"],
                "max_change": result["max_change"],
                "is_converged": result["is_converged"],
                "is_stage_complete": self.state.stage == "complete"
            }
        
        return {"error": f"Cannot process feedback in stage: {self.state.stage}"}
    
    def _transition_to_refinement(self):
        """Transition from elimination to refinement stage."""
        # Get winning tags from elimination
        winning_tags = self.state.elimination_state.get_final_tags()
        
        # Get initial weights based on slot importance
        initial_weights = {}
        for slot_state in self.state.elimination_state.state.slot_states.values():
            if slot_state.current_winner:
                initial_weights[slot_state.current_winner] = slot_state.slot.importance
        
        # Initialize refinement
        self.state.refinement_state = AttentionGradientRefiner(
            base_prompt=self.config.base_prompt,
            tags=winning_tags,
            initial_weights=initial_weights,
            learning_rate=self.config.learning_rate,
            convergence_threshold=self.config.convergence_threshold,
            max_rounds=self.config.refinement_max_rounds
        )
        
        self.state.stage = "refinement"
    
    def is_complete(self) -> bool:
        """Check if pipeline is complete."""
        return self.state.stage == "complete"
    
    def get_final_result(self) -> dict:
        """Get final result after pipeline completion."""
        if not self.is_complete():
            return {"error": "Pipeline not complete"}
        
        return {
            "base_prompt": self.config.base_prompt,
            "final_tags": self.state.final_result["optimized_tags"],
            "final_prompt": self.state.final_result["final_prompt"],
            "summary": {
                "original_tags": len(self.state.raw_tags),
                "after_deduplication": len(self.state.deduplicated_tags),
                "semantic_slots": len(self.state.slots),
                "final_tags": len(self.state.final_result["optimized_tags"]),
                "elimination_rounds": len(self.state.elimination_state.state.round_history),
                "refinement_rounds": len(self.state.refinement_state.round_history)
            }
        }
    
    def get_current_status(self) -> dict:
        """Get current pipeline status."""
        status = {
            "stage": self.state.stage,
            "is_complete": self.is_complete()
        }
        
        if self.state.stage == "elimination" and self.state.elimination_state:
            elim_state = self.state.elimination_state.get_current_results()
            status["elimination"] = {
                "rounds_completed": elim_state["rounds_completed"],
                "slots_resolved": sum(
                    1 for s in elim_state["slots"].values() if s["is_resolved"]
                ),
                "total_slots": len(elim_state["slots"]),
                "current_winners": {
                    k: v["winner"] for k, v in elim_state["slots"].items()
                }
            }
        
        if self.state.stage == "refinement" and self.state.refinement_state:
            status["refinement"] = {
                "rounds_completed": self.state.refinement_state.current_round,
                "current_weights": self.state.refinement_state.state.weights
            }
        
        return status