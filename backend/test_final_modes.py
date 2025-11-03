#!/usr/bin/env python3
"""
Test script for final stage generation modes (Mode 1, Mode 2, Mode 3, Mode 4)
with proper prompt tracking.
"""

import os
import json
from datetime import datetime
from util import sanitize_folder_name, initialize_prompt_tracking, generator_final_mode1, generator_final_mode2, generator_final_mode3, generator_final_mode4, designer_seq
from prompt import FINAL_PROMPT, FINAL_GENERATOR_PROMPT, FINAL_PROMPT_TAGS, FINAL_GENERATOR_PROMPT_TAGS, FINAL_GENERATOR_PROMPT_IMGS
from mode4 import generate_user_preference

def test_final_modes():
    """Test all final generation modes with prompt tracking."""
    
    # Test configuration
    descriptor = "A quiet and clean space for working and studying"
    
    # Mock user preferences (as would come from previous stages)
    user_pref = {
        "impression": {
            "concept_name": "Focused Productivity Haven",
            "core_strategy": "Create a clean, organized environment that promotes concentration and mental clarity",
            "technical_details": {
                "primary_purpose": "Work and study with minimal distractions",
                "intended_experience": "Calm focus and mental clarity",
                "user_needs": "Organization, quiet, clean aesthetics",
                "environmental_character": "Minimal, organized, peaceful",
                "activity_priority": "Reading, writing, computer work"
            },
            "foundational_identity": "A sanctuary for productive work and learning"
        },
        "spatial": {
            "concept_name": "Organized Workspace Layout",
            "core_strategy": "Optimize layout for workflow efficiency and visual calm",
            "technical_details": {
                "room_dimensions": "Medium-sized room, 12ft x 14ft",
                "spatial_layout": "L-shaped desk configuration with storage",
                "architectural_features": "Large window, built-in shelving, clean lines",
                "object_placement": "Desk near window, storage along walls, minimal decoration"
            }
        },
        "ambient": {
            "concept_name": "Productive Ambiance",
            "core_strategy": "Create calm, focused atmosphere with natural elements",
            "technical_details": {
                "lighting_character": "Natural daylight with warm task lighting",
                "color_palette": "Neutral colors with natural wood accents",
                "atmospheric_mood": "Calm, focused, organized",
                "sensory_experience": "Quiet, clean, minimal visual distractions"
            }
        }
    }
    
    # Mock tag data (as would come from user preferences)
    tag_data = {
        "parallel": [
            {"tag": "Wooden desk", "preference": "positive", "source_image": "impression_0_0"},
            {"tag": "Large windows", "preference": "positive", "source_image": "spatial_1_0"},
            {"tag": "Natural light", "preference": "positive", "source_image": "ambient_2_0"},
            {"tag": "Clutter", "preference": "negative", "source_image": "impression_3_0"},
            {"tag": "Dark colors", "preference": "negative", "source_image": "ambient_1_0"}
        ]
    }
    
    # Mock reference images for Mode 3
    reference_images = [
        "test_impression.png",  # These would be real paths in actual use
        "test_spatial.png",
        "test_ambient.png"
    ]
    
    # Create test session folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_desc = sanitize_folder_name(descriptor)
    session_folder = os.path.join("test_sessions", f"final_modes_test_{safe_desc}_{timestamp}")
    os.makedirs(session_folder, exist_ok=True)
    
    # Initialize prompt tracking
    initialize_prompt_tracking(session_folder, descriptor, "final_modes_test")
    
    print(f"🧪 Testing final modes in: {session_folder}")
    print(f"📝 Descriptor: {descriptor}")
    print("=" * 60)
    
    # First generate a scene using designer (needed for all modes)
    print("🎨 Generating scene concepts...")
    scenes = designer_seq(FINAL_PROMPT, descriptor, user_pref, session_folder, "final")
    
    if not scenes:
        print("❌ Failed to generate scenes")
        return
    
    scene = scenes[0]  # Use first generated scene
    print(f"✅ Generated scene: {scene.get('concept_name', 'Unknown')}")
    print()
    
    # Test Mode 1: JSON only
    print("🔄 Testing Mode 1 (JSON only)...")
    try:
        mode1_folder = os.path.join(session_folder, "mode1")
        os.makedirs(mode1_folder, exist_ok=True)
        
        files = generator_final_mode1(
            FINAL_GENERATOR_PROMPT, descriptor, scene, user_pref, 
            mode1_folder, "mode1_test", session_folder, "final"
        )
        print(f"✅ Mode 1 completed: {len(files) if files else 0} images generated")
    except Exception as e:
        print(f"❌ Mode 1 failed: {e}")
    print()
    
    # Test Mode 2: JSON + Tags
    print("🔄 Testing Mode 2 (JSON + Tags)...")
    try:
        mode2_folder = os.path.join(session_folder, "mode2")
        os.makedirs(mode2_folder, exist_ok=True)
        
        files = generator_final_mode2(
            FINAL_PROMPT_TAGS, FINAL_GENERATOR_PROMPT_TAGS, descriptor,
            scene, user_pref, tag_data,
            mode2_folder, "mode2_test", session_folder, "final"
        )
        print(f"✅ Mode 2 completed: {len(files) if files else 0} images generated")
    except Exception as e:
        print(f"❌ Mode 2 failed: {e}")
    print()
    
    # Test Mode 3: JSON + Tags + Images
    print("🔄 Testing Mode 3 (JSON + Tags + Images)...")
    try:
        mode3_folder = os.path.join(session_folder, "mode3")
        os.makedirs(mode3_folder, exist_ok=True)
        
        files = generator_final_mode3(
            FINAL_PROMPT_TAGS, FINAL_GENERATOR_PROMPT_IMGS, descriptor,
            scene, user_pref, tag_data, reference_images,
            mode3_folder, "mode3_test", session_folder, "final"
        )
        print(f"✅ Mode 3 completed: {len(files) if files else 0} images generated")
    except Exception as e:
        print(f"❌ Mode 3 failed: {e}")
    print()
    
    # Test Mode 4: Enhanced User Preferences
    print("🔄 Testing Mode 4 (Enhanced User Preferences)...")
    try:
        mode4_folder = os.path.join(session_folder, "mode4")
        os.makedirs(mode4_folder, exist_ok=True)
        
        # Create mock data files for mode4 testing
        test_impression_data = [user_pref["impression"]]
        test_spatial_data = [user_pref["spatial"]]  
        test_ambient_data = [user_pref["ambient"]]
        test_preferences = {
            "selections": {
                "impression": "impression_0_0",
                "spatial": "spatial_0_0", 
                "ambient": "ambient_0_0"
            },
            "tags": tag_data
        }
        
        # Create temporary test files
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='_impression.json', delete=False) as f:
            json.dump(test_impression_data, f, indent=2)
            impression_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_spatial.json', delete=False) as f:
            json.dump(test_spatial_data, f, indent=2)
            spatial_path = f.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='_ambient.json', delete=False) as f:
            json.dump(test_ambient_data, f, indent=2)
            ambient_path = f.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='_preferences.json', delete=False) as f:
            json.dump(test_preferences, f, indent=2)
            preferences_path = f.name
        
        # Generate enhanced user preferences
        enhanced_user_pref = generate_user_preference(
            impression_path, spatial_path, ambient_path, preferences_path
        )
        
        # Clean up temporary files
        for path in [impression_path, spatial_path, ambient_path, preferences_path]:
            os.unlink(path)
        
        files = generator_final_mode4(
            FINAL_PROMPT_TAGS, FINAL_GENERATOR_PROMPT_TAGS, descriptor,
            scene, enhanced_user_pref,
            mode4_folder, "mode4_test", session_folder, "final"
        )
        print(f"✅ Mode 4 completed: {len(files) if files else 0} images generated")
    except Exception as e:
        print(f"❌ Mode 4 failed: {e}")
    print()
    
    print("=" * 60)
    print(f"📊 Test completed! Check prompt tracking in: {session_folder}/prompt_track.txt")
    print(f"📁 Generated files in: {session_folder}")

if __name__ == "__main__":
    test_final_modes() 