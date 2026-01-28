"""
Stage 3: Slot-Based Elimination
Run head-to-head competitions within slots to find winning tags.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict
from enum import Enum
import numpy as np

# Import SemanticSlot from local module
from .semantic_slots import SemanticSlot


class RoundType(Enum):
    EXPLORATION = "exploration"           # Show diverse combinations
    HEAD_TO_HEAD = "head_to_head"         # Focus on one uncertain slot
    VALIDATION = "validation"             # Confirm current winners


@dataclass
class ImageConfig:
    """Configuration for a single generated image."""
    image_id: int
    slot_selections: dict[str, str]   # slot_name → selected_tag
    prompt: str
    strategy: str                      # Description of why this config


@dataclass
class RoundConfig:
    """Configuration for a single user round."""
    round_num: int
    round_type: RoundType
    focus_slot: Optional[str]          # For HEAD_TO_HEAD rounds
    images: list[ImageConfig]


@dataclass
class RoundFeedback:
    """User feedback from a round."""
    selected_image_idx: int
    attention_maps: list[dict[str, float]]  # Per image: tag → attention score


@dataclass
class SlotState:
    """Current state of a semantic slot."""
    slot: SemanticSlot
    tag_scores: dict[str, float]       # tag → cumulative score
    current_winner: Optional[str]
    confidence: float                  # 0-1, how certain we are
    is_resolved: bool                  # True if winner is final
    
    @classmethod
    def from_slot(cls, slot: SemanticSlot) -> 'SlotState':
        """Initialize state from a SemanticSlot."""
        initial_scores = {tag: 1.0 for tag in slot.tags}
        return cls(
            slot=slot,
            tag_scores=initial_scores,
            current_winner=slot.tags[0] if slot.tags else None,
            confidence=0.0 if len(slot.tags) > 1 else 1.0,
            is_resolved=len(slot.tags) <= 1
        )


@dataclass
class EliminationState:
    """Complete state of the elimination phase."""
    base_prompt: str
    slot_states: dict[str, SlotState]  # slot_name → state
    round_history: list[dict]
    current_round: int
    
    def get_unresolved_slots(self) -> list[SlotState]:
        """Get slots that still need resolution."""
        return [
            state for state in self.slot_states.values()
            if not state.is_resolved
        ]
    
    def get_current_winners(self) -> dict[str, str]:
        """Get current winner from each slot."""
        return {
            name: state.current_winner
            for name, state in self.slot_states.items()
            if state.current_winner
        }
    
    def is_complete(self) -> bool:
        """Check if all slots are resolved."""
        return all(state.is_resolved for state in self.slot_states.values())


class SlotBasedElimination:
    """
    Manages the slot-based elimination process.
    """
    
    def __init__(
        self,
        base_prompt: str,
        slots: list[SemanticSlot],
        max_rounds: int = 5,
        confidence_threshold: float = 0.75
    ):
        self.base_prompt = base_prompt
        self.max_rounds = max_rounds
        self.confidence_threshold = confidence_threshold
        
        # Initialize state
        self.state = EliminationState(
            base_prompt=base_prompt,
            slot_states={slot.name: SlotState.from_slot(slot) for slot in slots},
            round_history=[],
            current_round=0
        )
        
        # Sort slots by importance for prioritized resolution
        self.slot_priority = sorted(
            slots, 
            key=lambda s: (len(s.tags) > 1, s.importance), 
            reverse=True
        )
    
    def generate_round(self) -> RoundConfig:
        """
        Generate the next round configuration.
        
        Strategy:
        - Round 1: Exploration (diverse combinations to establish baseline)
        - Rounds 2+: Head-to-head for most uncertain slot
        - Final round: Validation of all winners
        """
        self.state.current_round += 1
        round_num = self.state.current_round
        
        unresolved = self.state.get_unresolved_slots()
        
        # Round 1: Exploration
        if round_num == 1:
            return self._generate_exploration_round(round_num)
        
        # All resolved: Validation
        if not unresolved:
            return self._generate_validation_round(round_num)
        
        # Find most uncertain slot
        most_uncertain = min(unresolved, key=lambda s: s.confidence)
        
        # If multiple tags remain, do head-to-head
        active_tags = [t for t, score in most_uncertain.tag_scores.items() if score > 0]
        
        if len(active_tags) > 1:
            return self._generate_head_to_head_round(round_num, most_uncertain)
        else:
            # Slot resolved, move to next
            most_uncertain.is_resolved = True
            most_uncertain.confidence = 1.0
            return self.generate_round()  # Recurse to next slot
    
    def _generate_exploration_round(self, round_num: int) -> RoundConfig:
        """Generate exploration round with diverse slot combinations."""
        
        images = []
        slot_names = list(self.state.slot_states.keys())
        
        for img_idx in range(4):
            selections = {}
            
            for slot_name in slot_names:
                slot_state = self.state.slot_states[slot_name]
                tags = list(slot_state.tag_scores.keys())
                
                if len(tags) == 1:
                    selections[slot_name] = tags[0]
                else:
                    # Rotate through tags across images
                    tag_idx = (img_idx + hash(slot_name)) % len(tags)
                    selections[slot_name] = tags[tag_idx]
            
            images.append(ImageConfig(
                image_id=img_idx,
                slot_selections=selections,
                prompt=self._build_prompt(selections),
                strategy=f"exploration_combo_{img_idx}"
            ))
        
        return RoundConfig(
            round_num=round_num,
            round_type=RoundType.EXPLORATION,
            focus_slot=None,
            images=images
        )
    
    def _generate_head_to_head_round(
        self, 
        round_num: int, 
        focus_slot_state: SlotState
    ) -> RoundConfig:
        """Generate head-to-head round for a specific slot."""
        
        focus_slot_name = focus_slot_state.slot.name
        active_tags = [t for t, score in focus_slot_state.tag_scores.items() if score > 0]
        
        # Get baseline selections for other slots (current winners)
        baseline = {}
        for name, state in self.state.slot_states.items():
            if name != focus_slot_name:
                baseline[name] = state.current_winner
        
        images = []
        
        # Create one image per competing tag (up to 4)
        for img_idx, tag in enumerate(active_tags[:4]):
            selections = baseline.copy()
            selections[focus_slot_name] = tag
            
            images.append(ImageConfig(
                image_id=img_idx,
                slot_selections=selections,
                prompt=self._build_prompt(selections),
                strategy=f"test_{focus_slot_name}={tag}"
            ))
        
        # If fewer than 4 tags, add variations
        while len(images) < 4 and len(active_tags) >= 2:
            # Add image without this slot (test if slot is needed)
            if len(images) == len(active_tags):
                selections = baseline.copy()
                # Don't include focus slot
                images.append(ImageConfig(
                    image_id=len(images),
                    slot_selections=selections,
                    prompt=self._build_prompt(selections),
                    strategy=f"without_{focus_slot_name}"
                ))
            else:
                # Duplicate best-scoring tag with slight variation in other slot
                best_tag = max(active_tags, key=lambda t: focus_slot_state.tag_scores[t])
                selections = baseline.copy()
                selections[focus_slot_name] = best_tag
                images.append(ImageConfig(
                    image_id=len(images),
                    slot_selections=selections,
                    prompt=self._build_prompt(selections),
                    strategy=f"confirm_{focus_slot_name}={best_tag}"
                ))
            
            if len(images) >= 4:
                break
        
        return RoundConfig(
            round_num=round_num,
            round_type=RoundType.HEAD_TO_HEAD,
            focus_slot=focus_slot_name,
            images=images
        )
    
    def _generate_validation_round(self, round_num: int) -> RoundConfig:
        """Generate validation round to confirm all winners."""
        
        current_winners = self.state.get_current_winners()
        
        images = []
        
        # Image A: All current winners
        images.append(ImageConfig(
            image_id=0,
            slot_selections=current_winners.copy(),
            prompt=self._build_prompt(current_winners),
            strategy="all_winners"
        ))
        
        # Images B-D: Swap one slot to runner-up
        swappable_slots = [
            (name, state) for name, state in self.state.slot_states.items()
            if len([t for t, s in state.tag_scores.items() if s > 0]) > 1
        ]
        
        for img_idx, (slot_name, slot_state) in enumerate(swappable_slots[:3], start=1):
            selections = current_winners.copy()
            
            # Find runner-up
            sorted_tags = sorted(
                slot_state.tag_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            if len(sorted_tags) > 1:
                runner_up = sorted_tags[1][0]
                selections[slot_name] = runner_up
            
            images.append(ImageConfig(
                image_id=img_idx,
                slot_selections=selections,
                prompt=self._build_prompt(selections),
                strategy=f"swap_{slot_name}_to_runnerup"
            ))
        
        # Fill remaining slots if needed
        while len(images) < 4:
            images.append(ImageConfig(
                image_id=len(images),
                slot_selections=current_winners.copy(),
                prompt=self._build_prompt(current_winners),
                strategy="duplicate_winners"
            ))
        
        return RoundConfig(
            round_num=round_num,
            round_type=RoundType.VALIDATION,
            focus_slot=None,
            images=images
        )
    
    def _build_prompt(self, selections: dict[str, str]) -> str:
        """Build generation prompt from slot selections."""
        tags = [tag for tag in selections.values() if tag]
        return f"{self.base_prompt}, {', '.join(tags)}"
    
    def process_feedback(
        self, 
        round_config: RoundConfig, 
        feedback: RoundFeedback
    ) -> dict:
        """
        Process user feedback and update slot states.
        
        Returns dict with:
        - updates: what changed
        - newly_resolved: slots that are now resolved
        - eliminations: tags that were eliminated
        """
        
        selected_idx = feedback.selected_image_idx
        selected_image = round_config.images[selected_idx]
        selected_attention = feedback.attention_maps[selected_idx]
        
        updates = {}
        newly_resolved = []
        eliminations = []
        
        if round_config.round_type == RoundType.EXPLORATION:
            # Update all slots based on what was in selected image
            for slot_name, selected_tag in selected_image.slot_selections.items():
                slot_state = self.state.slot_states[slot_name]
                
                if slot_state.is_resolved:
                    continue
                
                # Boost selected tag, penalize others
                attention_score = selected_attention.get(selected_tag, 0.5)
                
                for tag in slot_state.tag_scores:
                    if tag == selected_tag:
                        slot_state.tag_scores[tag] += 0.3 + 0.2 * attention_score
                    else:
                        # Check if this tag was in any rejected image
                        in_rejected = any(
                            round_config.images[i].slot_selections.get(slot_name) == tag
                            for i in range(4) if i != selected_idx
                        )
                        if in_rejected:
                            slot_state.tag_scores[tag] -= 0.15
                
                # Update winner
                slot_state.current_winner = max(
                    slot_state.tag_scores.items(),
                    key=lambda x: x[1]
                )[0]
                
                # Update confidence based on score gap
                scores = list(slot_state.tag_scores.values())
                if len(scores) > 1:
                    scores.sort(reverse=True)
                    gap = scores[0] - scores[1]
                    slot_state.confidence = min(gap / 2, 1.0)
        
        elif round_config.round_type == RoundType.HEAD_TO_HEAD:
            focus_slot_name = round_config.focus_slot
            slot_state = self.state.slot_states[focus_slot_name]
            
            # Determine which tag won
            winning_tag = selected_image.slot_selections.get(focus_slot_name)
            
            if winning_tag:
                # Strong boost to winner
                attention_score = selected_attention.get(winning_tag, 0.5)
                slot_state.tag_scores[winning_tag] += 0.5 + 0.3 * attention_score
                
                # Penalize losers
                for img in round_config.images:
                    if img.image_id != selected_idx:
                        losing_tag = img.slot_selections.get(focus_slot_name)
                        if losing_tag and losing_tag in slot_state.tag_scores:
                            slot_state.tag_scores[losing_tag] -= 0.3
                            
                            # Eliminate if score drops too low
                            if slot_state.tag_scores[losing_tag] < 0.2:
                                eliminations.append((focus_slot_name, losing_tag))
                
                # Update winner and confidence
                slot_state.current_winner = winning_tag
                
                active_tags = [t for t, s in slot_state.tag_scores.items() if s >= 0.2]
                if len(active_tags) == 1:
                    slot_state.is_resolved = True
                    slot_state.confidence = 1.0
                    newly_resolved.append(focus_slot_name)
                else:
                    scores = [slot_state.tag_scores[t] for t in active_tags]
                    scores.sort(reverse=True)
                    gap = scores[0] - scores[1] if len(scores) > 1 else 1.0
                    slot_state.confidence = min(0.5 + gap / 2, 0.95)
                    
                    # Resolve if confidence is high enough
                    if slot_state.confidence >= self.confidence_threshold:
                        slot_state.is_resolved = True
                        newly_resolved.append(focus_slot_name)
            
            else:
                # User selected image without this slot - slot may not be needed
                # Reduce importance but don't eliminate
                slot_state.slot.importance *= 0.8
        
        elif round_config.round_type == RoundType.VALIDATION:
            if selected_idx == 0:
                # User confirmed all winners - we're done
                for slot_state in self.state.slot_states.values():
                    slot_state.is_resolved = True
                    slot_state.confidence = 1.0
            else:
                # User preferred a swap - update that slot
                selected_config = selected_image
                for slot_name, tag in selected_config.slot_selections.items():
                    current_winner = self.state.slot_states[slot_name].current_winner
                    if tag != current_winner:
                        # User preferred the swap
                        self.state.slot_states[slot_name].current_winner = tag
                        self.state.slot_states[slot_name].tag_scores[tag] += 0.3
                        self.state.slot_states[slot_name].confidence = 0.6  # Reset confidence
                        self.state.slot_states[slot_name].is_resolved = False
        
        # Record in history
        self.state.round_history.append({
            "round_num": round_config.round_num,
            "round_type": round_config.round_type.value,
            "focus_slot": round_config.focus_slot,
            "selected_idx": selected_idx,
            "selected_tags": selected_image.slot_selections,
            "newly_resolved": newly_resolved,
            "eliminations": eliminations
        })
        
        return {
            "updates": updates,
            "newly_resolved": newly_resolved,
            "eliminations": eliminations,
            "is_complete": self.state.is_complete()
        }
    
    def get_current_results(self) -> dict:
        """Get current state of all slots."""
        return {
            "base_prompt": self.base_prompt,
            "slots": {
                name: {
                    "winner": state.current_winner,
                    "confidence": state.confidence,
                    "is_resolved": state.is_resolved,
                    "remaining_tags": [t for t, s in state.tag_scores.items() if s >= 0.2],
                    "scores": state.tag_scores
                }
                for name, state in self.state.slot_states.items()
            },
            "current_prompt": self._build_prompt(self.state.get_current_winners()),
            "rounds_completed": len(self.state.round_history),
            "is_complete": self.state.is_complete()
        }
    
    def get_final_tags(self) -> list[str]:
        """Get the final list of winning tags."""
        return [
            state.current_winner 
            for state in self.state.slot_states.values()
            if state.current_winner
        ]