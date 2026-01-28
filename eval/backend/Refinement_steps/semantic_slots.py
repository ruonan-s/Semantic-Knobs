"""
Stage 2: Semantic Slot Creation via LLM API
Group deduplicated tags into semantic slots for head-to-head competition.
"""

import json
import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path

# Add backend paths for imports
_current_dir = Path(__file__).parent.resolve()
_eval_backend = _current_dir.parent
_main_backend = _current_dir.parent.parent.parent / 'backend'

for _path in [str(_eval_backend), str(_main_backend)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from util import call_gemini_api
except ImportError as e:
    print(f"[SemanticSlots] Warning: Could not import call_gemini_api: {e}")
    # Fallback: define a stub for testing without backend
    def call_gemini_api(user_input: str, system_prompt: str) -> str:
        raise NotImplementedError("call_gemini_api not available - check backend path")


@dataclass
class SemanticSlot:
    """A semantic category containing competing tag alternatives."""
    name: str                           # Slot name (e.g., "lighting", "greenery")
    description: str                    # What this slot controls
    tags: List[str]                     # Competing tags in this slot
    importance: float = 0.5             # How critical is this slot (0-1)
    current_winner: Optional[str] = None
    confidence: float = 0.0


@dataclass
class SlotCreationResult:
    """Result of LLM slot creation."""
    slots: List[SemanticSlot]
    unassigned_tags: List[str] = field(default_factory=list)
    reasoning: str = ""


def create_slot_creation_prompt(base_prompt: str, tags: List[str]) -> str:
    """
    Create the prompt for LLM to generate semantic slots.
    """
    
    prompt = f"""You are helping design a system for personalized interior/environment generation. 

## Context
A user wants to create images matching the description: "{base_prompt}"

Through an exploration phase, we identified these positive tags that the user likes:
{json.dumps(tags, indent=2)}

## Your Task
Group these tags into **semantic slots**. Each slot represents a specific design dimension where the tags are **alternative approaches to achieve the same goal**.

### Key Principles:

1. **Tags within a slot are alternatives, not complements**
   - CORRECT: "natural light" and "bright atmosphere" in same slot (both about lighting quality)
   - WRONG: "natural light" and "hanging plants" in same slot (different design dimensions)

2. **Tags across slots work together**
   - The final result will use ONE tag from EACH slot
   - Example: "natural light" (from lighting slot) + "hanging plants" (from greenery slot)

3. **Preserve meaningful distinctions**
   - "hanging plants" vs "lush greenery" should be in SAME slot (both about plants, user will choose)
   - "open layout" vs "cozy seating" should be in DIFFERENT slots (layout vs furniture)

## Output Format
Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "slots": [
    {{
      "name": "lighting",
      "description": "How the space is illuminated and the quality of light",
      "tags": ["natural light", "bright atmosphere", "abundant natural light"]
    }},
    {{
      "name": "greenery", 
      "description": "How plants and natural elements are incorporated",
      "tags": ["hanging plants", "lush greenery", "integrated nature elements"]
    }}
  ],
  "unassigned_tags": [],
  "reasoning": "Brief explanation of grouping logic"
}}

## Requirements
- Create 6-10 slots (aim for meaningful groupings, not too granular)
- Every tag should be assigned to exactly one slot (or listed in unassigned_tags)
- Slot names should be concise and descriptive
- Each slot should have 1-4 tags

Now analyze the tags and create semantic slots:"""

    return prompt


def calculate_slot_importance(
    slot_tags: List[str],
    tag_frequencies: Dict[str, int]
) -> float:
    """
    Importance = average frequency of tags in this slot,
    normalized to 0-1 range.
    """
    if not slot_tags:
        return 0.3
    
    slot_freq = sum(tag_frequencies.get(tag, 1) for tag in slot_tags)
    avg_freq = slot_freq / len(slot_tags)
    
    # Normalize: assume max reasonable frequency is ~5
    importance = min(avg_freq / 5, 1.0)
    
    # Floor at 0.3 (nothing is unimportant)
    return max(importance, 0.3)


def add_importance_to_slots(
    slots: List[SemanticSlot],
    tag_frequencies: Dict[str, int]
) -> List[SemanticSlot]:
    """Add importance scores based on frequency data."""
    
    for slot in slots:
        slot_freq = sum(tag_frequencies.get(tag, 1) for tag in slot.tags)
        avg_freq = slot_freq / len(slot.tags) if slot.tags else 0
        slot.importance = round(min(max(avg_freq / 5, 0.3), 1.0), 2)
    
    return slots

    
def parse_llm_response(response_text: str) -> SlotCreationResult:
    """Parse LLM response into SlotCreationResult."""
    
    # Clean response
    json_str = response_text.strip()
    
    # Extract JSON from response (handle markdown code blocks)
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0]
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0]
    
    json_str = json_str.strip()
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[SemanticSlots] JSON parse error: {e}")
        print(f"[SemanticSlots] Raw response: {response_text[:500]}")
        # Return empty result on parse failure
        return SlotCreationResult(
            slots=[],
            unassigned_tags=[],
            reasoning="Parse error"
        )
    
    slots = []
    for slot_data in data.get("slots", []):
        slots.append(SemanticSlot(
            name=slot_data.get("name", "unknown"),
            description=slot_data.get("description", ""),
            tags=slot_data.get("tags", []),
            importance=slot_data.get("importance", 0.5)
        ))
    
    return SlotCreationResult(
        slots=slots,
        unassigned_tags=data.get("unassigned_tags", []),
        reasoning=data.get("reasoning", "")
    )


def create_semantic_slots_sync(
    base_prompt: str,
    tags: List[str],
    tag_frequencies: Dict[str, int] = None
) -> SlotCreationResult:
    """
    Call Gemini API to create semantic slots from tags.
    
    Args:
        base_prompt: The user's base description (e.g., "refreshing cafe")
        tags: List of deduplicated positive tags
        tag_frequencies: Optional frequency data for importance calculation
        
    Returns:
        SlotCreationResult with semantic slots
    """
    
    prompt = create_slot_creation_prompt(base_prompt, tags)
    system_prompt = "You are an expert interior design AI. Output valid JSON only, no markdown formatting."
    
    print(f"[SemanticSlots] Calling Gemini API for {len(tags)} tags...")
    
    response_text = call_gemini_api(prompt, system_prompt)
    
    if not response_text:
        print("[SemanticSlots] Empty response from Gemini API")
        return SlotCreationResult(slots=[], unassigned_tags=tags, reasoning="API error")
    
    result = parse_llm_response(response_text)
    
    # Add importance scores from frequency data
    if tag_frequencies:
        result.slots = add_importance_to_slots(result.slots, tag_frequencies)
    
    # Validate: ensure all tags are assigned
    assigned_tags = set()
    for slot in result.slots:
        assigned_tags.update(slot.tags)
    assigned_tags.update(result.unassigned_tags)
    
    missing = set(tags) - assigned_tags
    if missing:
        print(f"[SemanticSlots] Warning: {len(missing)} tags not assigned: {missing}")
        result.unassigned_tags.extend(list(missing))
    
    print(f"[SemanticSlots] Created {len(result.slots)} slots")
    for slot in result.slots:
        print(f"  - {slot.name}: {slot.tags}")
    
    return result


# Async version if needed
async def create_semantic_slots(
    base_prompt: str,
    tags: List[str],
    tag_frequencies: Dict[str, int] = None
) -> SlotCreationResult:
    """Async wrapper - just calls sync version for now."""
    return create_semantic_slots_sync(base_prompt, tags, tag_frequencies)
