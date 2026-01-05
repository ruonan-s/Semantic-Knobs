#!/usr/bin/env python3
"""
Test script to verify GP exploration integration with slider generation.

This script tests:
1. GP system can process interactions and produce concepts
2. GP weights can be saved in format compatible with slider generation
3. The saved format matches expected slider generation input
"""

import os
import sys
import json
import tempfile
import numpy as np

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from exploration_GP import (
    AdaptivePreferenceSystem,
    RawTag,
    InteractionRound,
    DisplayConcept
)
from gp_session import GPExplorationSession, get_or_create_gp_session


def create_mock_embedding(seed: int, dim: int = 768) -> np.ndarray:
    """Create a mock CLIP embedding."""
    np.random.seed(seed)
    emb = np.random.randn(dim)
    return (emb / np.linalg.norm(emb)).astype(np.float32)


def test_gp_system_basic():
    """Test basic GP system functionality."""
    print("=" * 60)
    print("TEST 1: Basic GP System Functionality")
    print("=" * 60)
    
    # Create system
    system = AdaptivePreferenceSystem(embedding_dim=768, n_inducing=16)
    
    # Create mock tags for 4 images
    tag_texts = [
        "warm lighting", "cozy atmosphere", "soft textures",
        "cool tones", "minimalist", "modern design",
        "rustic charm", "natural materials", "earth colors",
        "industrial look", "metal surfaces", "concrete walls"
    ]
    
    images = []
    for img_idx in range(4):
        img_tags = []
        for tag_idx in range(3):
            text = tag_texts[img_idx * 3 + tag_idx]
            img_tags.append(RawTag(
                id=f"tag_{img_idx}_{tag_idx}",
                text=text,
                embedding=create_mock_embedding(img_idx * 100 + tag_idx),
                source_image_idx=img_idx
            ))
        images.append(img_tags)
    
    # Create interaction with preferences
    tag_states = {}
    for img in images:
        for tag in img:
            # Like "warm" and "cozy" things
            if "warm" in tag.text or "cozy" in tag.text:
                tag_states[tag.id] = "liked"
            # Dislike "industrial" and "metal"
            elif "industrial" in tag.text or "metal" in tag.text:
                tag_states[tag.id] = "disliked"
            else:
                tag_states[tag.id] = "neutral"
    
    interaction = InteractionRound(
        images=images,
        selected_image_idx=0,  # Select first image
        tag_states=tag_states
    )
    
    # Process interaction
    concepts = system.process_interaction(interaction, refit=True, verbose=True)
    
    print(f"\nGenerated {len(concepts)} concepts")
    for c in concepts[:5]:
        print(f"  {c.representative_tag}: utility={c.mean_utility:.3f}, category={c.category}")
    
    # Test weight computation
    weights = system.get_concept_weights()
    print(f"\nWeight distribution (sum={sum(weights.values()):.4f}):")
    for cid, w in list(weights.items())[:5]:
        print(f"  {cid}: {w:.4f}")
    
    assert len(concepts) > 0, "Should have generated concepts"
    assert abs(sum(weights.values()) - 1.0) < 0.001, "Weights should sum to 1"
    
    print("\n✅ TEST 1 PASSED")
    return system


def test_concept_weights_format(system: AdaptivePreferenceSystem):
    """Test that saved concept weights match expected format."""
    print("\n" + "=" * 60)
    print("TEST 2: Concept Weights Format Compatibility")
    print("=" * 60)
    
    # Save to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        weights_file = system.save_concept_weights(tmpdir, "impression")
        
        assert os.path.exists(weights_file), f"Weights file should exist at {weights_file}"
        
        # Load and verify format
        with open(weights_file, 'r') as f:
            data = json.load(f)
        
        print(f"Saved to: {weights_file}")
        print(f"Keys in file: {list(data.keys())}")
        
        # Check required fields
        assert "concept_weights" in data, "Should have 'concept_weights' key"
        assert "stage" in data, "Should have 'stage' key"
        assert "num_concepts" in data, "Should have 'num_concepts' key"
        
        concept_weights = data["concept_weights"]
        assert len(concept_weights) > 0, "Should have at least one concept"
        
        # Check each concept has required fields for slider generation
        required_fields = ["concept_id", "label", "weight"]
        for cw in concept_weights:
            for field in required_fields:
                assert field in cw, f"Concept missing required field: {field}"
        
        print(f"\nConcept weights ({len(concept_weights)} concepts):")
        for cw in concept_weights[:5]:
            print(f"  {cw['concept_id']}: {cw['label']} = {cw['weight']:.4f}")
        
        # Check weights sum to ~1
        total_weight = sum(cw["weight"] for cw in concept_weights)
        print(f"\nTotal weight: {total_weight:.4f}")
        assert abs(total_weight - 1.0) < 0.001, f"Weights should sum to 1, got {total_weight}"
        
        # Check GP mode flag
        assert data.get("gp_mode") == True, "Should have gp_mode=True flag"
    
    print("\n✅ TEST 2 PASSED")


def test_gp_session_wrapper():
    """Test the GPExplorationSession wrapper class."""
    print("\n" + "=" * 60)
    print("TEST 3: GP Session Wrapper")
    print("=" * 60)
    
    # Create session
    session = GPExplorationSession(
        session_id="test_session",
        stage="impression",
        image_ids=["img_0", "img_1", "img_2", "img_3"],
        embedding_dim=768,
        n_inducing=16
    )
    
    # Note: We can't test initialize_from_tags without CLIP model
    # So we test the internal structures
    
    # Manually add some tags
    for img_idx in range(4):
        session.image_tags[f"img_{img_idx}"] = []
        for tag_idx in range(3):
            tag = RawTag(
                id=f"tag_impression_img_{img_idx}_{tag_idx}",
                text=f"tag_{img_idx}_{tag_idx}",
                embedding=create_mock_embedding(img_idx * 100 + tag_idx),
                source_image_idx=img_idx
            )
            session.raw_tags[tag.id] = tag
            session.tag_to_image[tag.id] = f"img_{img_idx}"
            session.tag_states[tag.id] = "neutral"
            session.image_tags[f"img_{img_idx}"].append(tag)
            session.gp_system.all_tags[tag.id] = tag
    
    session.initialized = True
    
    # Test tag interaction
    session.handle_tag_click("tag_impression_img_0_0", "positive")
    assert session.tag_states["tag_impression_img_0_0"] == "liked"
    
    session.handle_tag_click("tag_impression_img_3_2", "negative")
    assert session.tag_states["tag_impression_img_3_2"] == "disliked"
    
    # Test image selection (triggers GP processing)
    session.handle_image_selection("img_0")
    assert session.selected_image_id == "img_0"
    
    # Test to_dict
    state_dict = session.to_dict()
    assert "concepts" in state_dict
    assert "categorized" in state_dict
    assert "tag_preferences" in state_dict
    
    print(f"Concepts in state: {len(state_dict['concepts'])}")
    print(f"Categorized: {state_dict['categorized']}")
    
    # Test save_concept_weights
    with tempfile.TemporaryDirectory() as tmpdir:
        weights_file = session.save_concept_weights(tmpdir)
        assert os.path.exists(weights_file), "Weights file should be created"
        
        with open(weights_file, 'r') as f:
            data = json.load(f)
        
        print(f"Saved {data['num_concepts']} concepts")
    
    print("\n✅ TEST 3 PASSED")


def test_slider_compatibility():
    """Test that GP output is compatible with slider generation expectations."""
    print("\n" + "=" * 60)
    print("TEST 4: Slider Generation Compatibility")
    print("=" * 60)
    
    # Create a mock concept_weights.json as GP would produce
    mock_weights = {
        "stage": "impression",
        "session_id": "test_session",
        "timestamp": "2026-01-02T10:00:00",
        "num_concepts": 5,
        "gp_mode": True,
        "n_preference_pairs": 20,
        "is_fitted": True,
        "concept_weights": [
            {"concept_id": "concept_0", "label": "warm lighting", "weight": 0.35},
            {"concept_id": "concept_1", "label": "cozy atmosphere", "weight": 0.25},
            {"concept_id": "concept_2", "label": "soft textures", "weight": 0.20},
            {"concept_id": "concept_3", "label": "natural materials", "weight": 0.15},
            {"concept_id": "concept_4", "label": "industrial look", "weight": 0.05},
        ]
    }
    
    # Simulate slider generation logic (from eval_server.py)
    concept_weights = mock_weights.get("concept_weights", [])
    assert len(concept_weights) > 0, "Should have concept weights"
    
    concepts = []
    weights = []
    target_location = "bedroom"
    
    for cw in concept_weights:
        concepts.append({
            "id": cw.get("concept_id", ""),
            "label": f"{target_location} with {cw['label']}"
        })
        weights.append(cw.get("weight", 0.0))
    
    w_exploration = np.array(weights)
    w_exploration_norm = w_exploration / (w_exploration.sum() + 1e-8)
    
    # Get top-K
    top_k = 10
    sorted_indices = np.argsort(w_exploration_norm)[::-1]
    actual_top_k = min(top_k, len(concepts))
    top_indices = sorted_indices[:actual_top_k]
    
    tag_phrases = [concepts[idx]['label'] for idx in top_indices]
    tag_weights = np.array([float(w_exploration_norm[idx]) for idx in top_indices])
    
    print("Slider generation input:")
    for phrase, weight in zip(tag_phrases, tag_weights):
        print(f"  {phrase}: {weight:.4f}")
    
    assert len(tag_phrases) == 5, "Should have 5 tag phrases"
    assert abs(tag_weights.sum() - 1.0) < 0.001, "Weights should sum to 1"
    assert "warm lighting" in tag_phrases[0], "Top concept should be in first position"
    
    print("\n✅ TEST 4 PASSED")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GP INTEGRATION TEST SUITE")
    print("=" * 60)
    
    try:
        # Test 1: Basic GP functionality
        system = test_gp_system_basic()
        
        # Test 2: Concept weights format
        test_concept_weights_format(system)
        
        # Test 3: GP session wrapper
        test_gp_session_wrapper()
        
        # Test 4: Slider compatibility
        test_slider_compatibility()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

