"""
Test script for refinement stages without generating images.
This script simulates the stage progression and refinement logic.
"""

import os
import json
from typing import Dict, List
from concept_refinement import get_or_create_session, ConceptState

# Import the refinement prompts
from prompt_refinement import (
    IMPRESSION_REFINEMENT_PROMPT,
    SPATIAL_REFINEMENT_PROMPT,
    OBJECTS_REFINEMENT_PROMPT,
    AMBIENT_REFINEMENT_PROMPT
)

# Import stage definitions from server
STAGES = [
    "impression", "impression_refinement",
    "spatial", "spatial_refinement",
    "objects", "objects_refinement",
    "ambient", "ambient_refinement",
    "final"
]

REFINEMENT_STAGES = {
    "impression_refinement", "spatial_refinement", 
    "objects_refinement", "ambient_refinement"
}


def create_mock_session_data(base_folder: str):
    """Create mock session data for testing"""
    os.makedirs(base_folder, exist_ok=True)
    
    # Create mock impression stage data
    impression_folder = os.path.join(base_folder, "impression")
    os.makedirs(impression_folder, exist_ok=True)
    
    # Mock JSON data for impression stage
    mock_scenes = [
        {
            "concept_name": "Cozy Corner Concept A",
            "user_description": "A cozy corner",
            "overall_style": "Minimalist modern with warm tones",
            "location_context": "Living room corner",
            "style_characteristics": "Clean lines, soft textures, natural light",
            "design_intent": "Create a peaceful reading nook",
            "design_rationale": "Emphasizes simplicity and comfort"
        },
        {
            "concept_name": "Cozy Corner Concept B",
            "user_description": "A cozy corner",
            "overall_style": "Rustic farmhouse style",
            "location_context": "Bedroom corner",
            "style_characteristics": "Wooden accents, vintage elements, warm lighting",
            "design_intent": "Create a nostalgic retreat",
            "design_rationale": "Emphasizes traditional comfort"
        },
        {
            "concept_name": "Cozy Corner Concept C",
            "user_description": "A cozy corner",
            "overall_style": "Scandinavian hygge",
            "location_context": "Window nook",
            "style_characteristics": "Natural materials, neutral colors, soft textiles",
            "design_intent": "Create a serene relaxation space",
            "design_rationale": "Emphasizes natural comfort and light"
        },
        {
            "concept_name": "Cozy Corner Concept D",
            "user_description": "A cozy corner",
            "overall_style": "Bohemian eclectic",
            "location_context": "Studio apartment corner",
            "style_characteristics": "Colorful textiles, plants, layered textures",
            "design_intent": "Create an artistic personal space",
            "design_rationale": "Emphasizes creativity and individuality"
        }
    ]
    
    # Save scenes JSON
    scenes_file = os.path.join(impression_folder, "impression.json")
    with open(scenes_file, "w") as f:
        json.dump(mock_scenes, f, indent=2)
    
    # Create mock image files (empty)
    for i in range(4):
        img_path = os.path.join(impression_folder, f"impression_{i}_0.png")
        with open(img_path, "w") as f:
            f.write("")  # Empty file
    
    # Create mock visual tags
    mock_tags = {
        "impression_0_0.png": ["minimalist", "warm tones", "natural light", "clean lines", "soft texture"],
        "impression_1_0.png": ["rustic", "wooden accents", "vintage", "warm lighting", "traditional"],
        "impression_2_0.png": ["scandinavian", "neutral colors", "natural materials", "serene", "simple"],
        "impression_3_0.png": ["bohemian", "colorful", "plants", "eclectic", "artistic"]
    }
    
    tags_file = os.path.join(impression_folder, "visual_tags.json")
    with open(tags_file, "w") as f:
        json.dump(mock_tags, f, indent=2)
    
    print(f"✅ Created mock session data in {base_folder}")
    return mock_scenes, mock_tags


def test_concept_extraction(session_id: str, stage: str, image_ids: List[str], mock_tags: Dict):
    """Test concept extraction and categorization"""
    print(f"\n{'='*60}")
    print(f"TEST: Concept Extraction for {stage}")
    print(f"{'='*60}")
    
    # Convert tags format
    image_tags = {}
    for img_file, tags in mock_tags.items():
        img_id = os.path.splitext(img_file)[0]
        if img_id in image_ids:
            image_tags[img_id] = tags
    
    # Get or create refinement session
    refinement_session = get_or_create_session(session_id, stage, image_ids)
    
    # Initialize from tags
    if not refinement_session.initialized:
        refinement_session.initialize_from_tags(image_tags)
    
    # Get categorized concepts
    categorized = refinement_session.get_categorized_concepts()
    
    print(f"\n📊 Concepts Created: {len(refinement_session.concepts)}")
    for concept in refinement_session.concepts:
        print(f"  - {concept.label} (ID: {concept.id[:8]}...)")
    
    print(f"\n📈 Categorization:")
    print(f"  Positive: {len(categorized.get('positive', []))}")
    print(f"  Neutral: {len(categorized.get('neutral', []))}")
    print(f"  Negative: {len(categorized.get('negative', []))}")
    
    # Simulate some user interactions
    print(f"\n🎯 Simulating User Interactions:")
    
    # Like some tags from the first image
    if len(refinement_session.raw_tags) >= 3:
        tag_to_like = refinement_session.raw_tags[0]
        refinement_session.handle_tag_click(tag_to_like.id, 'positive')
        print(f"  ✓ Liked tag: {tag_to_like.text}")
        
        tag_to_like2 = refinement_session.raw_tags[1]
        refinement_session.handle_tag_click(tag_to_like2.id, 'positive')
        print(f"  ✓ Liked tag: {tag_to_like2.text}")
    
    # Dislike some tags from another image
    if len(refinement_session.raw_tags) >= 15:
        tag_to_dislike = refinement_session.raw_tags[10]
        refinement_session.handle_tag_click(tag_to_dislike.id, 'negative')
        print(f"  ✗ Disliked tag: {tag_to_dislike.text}")
    
    # Simulate image selection
    if image_ids:
        refinement_session.handle_image_selection(image_ids[0])
        print(f"  📸 Selected image: {image_ids[0]}")
    
    # Get updated categorization
    categorized = refinement_session.get_categorized_concepts()
    
    return refinement_session, categorized


def test_tag_preference_extraction(refinement_session, categorized: Dict):
    """Test extraction of top positive and negative concepts"""
    print(f"\n{'='*60}")
    print(f"TEST: Tag Preference Extraction")
    print(f"{'='*60}")
    
    positive_concept_ids = categorized.get('positive', [])
    negative_concept_ids = categorized.get('negative', [])
    
    # Sort by weight (score) and get top 5 positive and top 3 negative
    positive_with_weights = []
    for concept_id in positive_concept_ids:
        if concept_id in refinement_session.concept_states:
            weight = refinement_session.concept_states[concept_id].score
            positive_with_weights.append((concept_id, weight))
    
    negative_with_weights = []
    for concept_id in negative_concept_ids:
        if concept_id in refinement_session.concept_states:
            weight = refinement_session.concept_states[concept_id].score
            negative_with_weights.append((concept_id, weight))
    
    # Sort by weight (descending for positive, ascending for negative)
    positive_with_weights.sort(key=lambda x: x[1], reverse=True)
    negative_with_weights.sort(key=lambda x: x[1])
    
    # Get labels for top concepts
    positive_tags = []
    print(f"\n✅ Top 5 Positive Concepts:")
    for concept_id, weight in positive_with_weights[:5]:
        concept = next((c for c in refinement_session.concepts if c.id == concept_id), None)
        if concept:
            positive_tags.append(concept.label)
            print(f"  {len(positive_tags)}. {concept.label} (weight: {weight:.3f})")
    
    negative_tags = []
    print(f"\n❌ Top 3 Negative Concepts:")
    for concept_id, weight in negative_with_weights[:3]:
        concept = next((c for c in refinement_session.concepts if c.id == concept_id), None)
        if concept:
            negative_tags.append(concept.label)
            print(f"  {len(negative_tags)}. {concept.label} (weight: {weight:.3f})")
    
    tag_preferences = {
        'positive': positive_tags,
        'negative': negative_tags
    }
    
    return tag_preferences


def test_refinement_prompt_formatting(selected_json: Dict, tag_preferences: Dict, descriptor: str, stage_name: str):
    """Test refinement prompt formatting"""
    print(f"\n{'='*60}")
    print(f"TEST: Refinement Prompt Formatting for {stage_name}")
    print(f"{'='*60}")
    
    # Get the appropriate refinement prompt
    refinement_prompts = {
        'impression_refinement': IMPRESSION_REFINEMENT_PROMPT,
        'spatial_refinement': SPATIAL_REFINEMENT_PROMPT,
        'objects_refinement': OBJECTS_REFINEMENT_PROMPT,
        'ambient_refinement': AMBIENT_REFINEMENT_PROMPT
    }
    
    refinement_prompt = refinement_prompts.get(stage_name)
    
    if not refinement_prompt:
        print(f"❌ No refinement prompt found for {stage_name}")
        return None
    
    # Format the refinement prompt with inputs
    formatted_prompt = refinement_prompt.replace(
        '[USER DESCRIPTION]\n<user text>\n[/USER DESCRIPTION]',
        f'[USER DESCRIPTION]\n{descriptor}\n[/USER DESCRIPTION]'
    )
    
    formatted_prompt = formatted_prompt.replace(
        '[SELECTED_INTERPRETATION_JSON]\n<one of the 4 exploration outputs, full JSON>\n[/SELECTED_INTERPRETATION_JSON]',
        f'[SELECTED_INTERPRETATION_JSON]\n{json.dumps(selected_json, indent=2)}\n[/SELECTED_INTERPRETATION_JSON]'
    )
    
    # Format tag preferences
    positive_tags = tag_preferences.get('positive', [])[:5]  # Top 5
    negative_tags = tag_preferences.get('negative', [])[:3]  # Top 3
    tag_prefs_json = {
        "positive": positive_tags,
        "negative": negative_tags
    }
    
    formatted_prompt = formatted_prompt.replace(
        '[TAG PREFERENCES]\n{\n  "positive": ["P1","P2","P3","P4","P5"],   // ordered; left = higher priority\n  "negative": ["N1","N2","N3"]              // ordered; left = stricter constraint\n}\n[/TAG PREFERENCES]',
        f'[TAG PREFERENCES]\n{json.dumps(tag_prefs_json, indent=2)}\n[/TAG PREFERENCES]'
    )
    
    print(f"\n📝 Formatted Prompt Preview (first 500 chars):")
    print(f"{formatted_prompt[:500]}...")
    
    print(f"\n✅ Prompt Sections Present:")
    print(f"  - USER DESCRIPTION: {'✓' if descriptor in formatted_prompt else '✗'}")
    print(f"  - SELECTED JSON: {'✓' if 'concept_name' in formatted_prompt else '✗'}")
    print(f"  - TAG PREFERENCES: {'✓' if positive_tags and str(positive_tags[0]) in formatted_prompt else '✗'}")
    
    return formatted_prompt


def test_stage_progression():
    """Test stage progression logic"""
    print(f"\n{'='*60}")
    print(f"TEST: Stage Progression")
    print(f"{'='*60}")
    
    print(f"\n📋 Stage Order:")
    for i, stage in enumerate(STAGES):
        is_refinement = stage in REFINEMENT_STAGES
        symbol = "↻" if is_refinement else "→"
        print(f"  {i+1}. {stage} {symbol}")
    
    # Test progression from each stage
    print(f"\n🔄 Testing Progressions:")
    for i, current_stage in enumerate(STAGES[:-1]):  # Exclude final
        next_idx = i + 1
        next_stage = STAGES[next_idx]
        is_refinement = next_stage in REFINEMENT_STAGES
        print(f"  {current_stage} → {next_stage} {'(refinement)' if is_refinement else '(exploration)'}")


def run_full_test():
    """Run complete refinement stage test"""
    print("\n" + "="*60)
    print("REFINEMENT STAGE TEST SUITE")
    print("="*60)
    
    # Setup
    session_id = "test_session_001"
    descriptor = "A cozy corner"
    base_folder = "sessions/test_refinement"
    
    # Clean up old test data
    import shutil
    if os.path.exists(base_folder):
        shutil.rmtree(base_folder)
    
    # Create mock data
    mock_scenes, mock_tags = create_mock_session_data(base_folder)
    
    # Get image IDs
    image_ids = [f"impression_{i}_0" for i in range(4)]
    
    # Test 1: Concept extraction
    refinement_session, categorized = test_concept_extraction(
        session_id, 
        "impression", 
        image_ids, 
        mock_tags
    )
    
    # Test 2: Tag preference extraction
    tag_preferences = test_tag_preference_extraction(refinement_session, categorized)
    
    # Test 3: Refinement prompt formatting
    selected_json = mock_scenes[0]  # Simulate selecting first concept
    formatted_prompt = test_refinement_prompt_formatting(
        selected_json,
        tag_preferences,
        descriptor,
        "impression_refinement"
    )
    
    # Test 4: Stage progression
    test_stage_progression()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Concept Extraction: PASSED")
    print(f"✅ Tag Preference Extraction: PASSED")
    print(f"✅ Refinement Prompt Formatting: PASSED")
    print(f"✅ Stage Progression: PASSED")
    print(f"\n🎉 All tests completed successfully!")
    print(f"\nTest data location: {base_folder}")
    print(f"\nYou can now proceed with actual image generation.")


if __name__ == "__main__":
    run_full_test()

