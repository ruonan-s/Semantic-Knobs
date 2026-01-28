"""
Refinement Pipeline - 4-Stage Tag Optimization

Transforms raw positive tags from exploration into optimized weighted tags:
    Raw tags → Deduplicated → Semantic slots → Winning tags → Weighted tags

Stages:
    1. Deduplication - Remove near-duplicate phrasings (automatic)
    2. Semantic Slots - Group tags into design dimensions via LLM (automatic)
    3. Elimination - Head-to-head competition per slot (3-5 user rounds)
    4. Refinement - Attention gradient weight optimization (2-3 user rounds)
"""

from .deduplication import (
    TagGroup,
    DeduplicationResult,
    deduplicate_tags,
    are_duplicates,
    normalize_tag
)

from .semantic_slots import (
    SemanticSlot,
    SlotCreationResult,
    create_semantic_slots_sync,
    create_slot_creation_prompt,
    add_importance_to_slots
)

from .slots_based_elimination import (
    RoundType,
    ImageConfig,
    RoundConfig,
    RoundFeedback,
    SlotState,
    EliminationState,
    SlotBasedElimination
)

from .attention_gradient import (
    WeightState,
    RefinementRoundConfig,
    RefinementFeedback,
    AttentionGradientRefiner
)

from .refinement_session import (
    RefinementStage,
    SlotRefinementConfig,
    SlotRefinementSession
)

__all__ = [
    # Stage 1
    'TagGroup',
    'DeduplicationResult', 
    'deduplicate_tags',
    'are_duplicates',
    'normalize_tag',
    
    # Stage 2
    'SemanticSlot',
    'SlotCreationResult',
    'create_semantic_slots_sync',
    'create_slot_creation_prompt',
    'add_importance_to_slots',
    
    # Stage 3
    'RoundType',
    'ImageConfig',
    'RoundConfig',
    'RoundFeedback',
    'SlotState',
    'EliminationState',
    'SlotBasedElimination',
    
    # Stage 4
    'WeightState',
    'RefinementRoundConfig',
    'RefinementFeedback',
    'AttentionGradientRefiner',
    
    # Session Manager
    'RefinementStage',
    'SlotRefinementConfig',
    'SlotRefinementSession',
]
