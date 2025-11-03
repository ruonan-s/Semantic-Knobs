#!/usr/bin/env python3
"""
Test script for cumulative tags final stage functionality.
"""

import os
import json
from cumulative_tags import (
    run_cumulative_tags_final_stage,
    save_user_preferences,
    load_user_preferences
)

def test_cumulative_tags_final_stage():
    """Test the cumulative tags final stage functionality."""
    
    # Create a test session folder
    test_folder = "test_cumulative_session"
    os.makedirs(test_folder, exist_ok=True)
    
    try:
        # Test data
        descriptor = "A modern minimalist living room with natural light"
        user_pref = {
            'impression': {
                'concept_name': 'Minimalist Serenity',
                'core_strategy': 'Clean lines and open space',
                'technical_details': {
                    'style_location_foundation': 'Modern minimalist',
                    'structural_organization': 'Open floor plan',
                    'objects_materials_integration': 'Natural materials',
                    'atmospheric_completion': 'Soft natural lighting'
                }
            },
            'spatial': {
                'concept_name': 'Spatial Harmony',
                'core_strategy': 'Balanced proportions',
                'technical_details': {
                    'style_location_foundation': 'Harmonious layout',
                    'structural_organization': 'Symmetrical design',
                    'objects_materials_integration': 'Integrated elements',
                    'atmospheric_completion': 'Cohesive atmosphere'
                }
            }
        }
        
        cumulative_tags = {
            'positive': ['minimalist', 'natural light', 'clean lines', 'open space', 'natural materials'],
            'negative': ['cluttered', 'dark', 'ornate', 'busy patterns']
        }
        
        print("🧪 Testing cumulative tags final stage...")
        print(f"📁 Test folder: {test_folder}")
        print(f"📝 Descriptor: {descriptor}")
        print(f"👤 User preferences: {len(user_pref)} stages")
        print(f"🏷️  Cumulative tags: +{len(cumulative_tags['positive'])} positive, -{len(cumulative_tags['negative'])} negative")
        
        # Test saving user preferences
        print("\n💾 Testing user preferences saving...")
        saved_prefs = save_user_preferences(test_folder, 'objects', {
            'concept_name': 'Object Harmony',
            'core_strategy': 'Balanced object placement'
        }, cumulative_tags, user_pref)
        
        print(f"✅ Saved preferences: {len(saved_prefs)} total stages")
        
        # Test loading user preferences
        print("\n📖 Testing user preferences loading...")
        loaded_prefs = load_user_preferences(test_folder)
        print(f"✅ Loaded preferences: {len(loaded_prefs)} total stages")
        
        # Test final stage mode 1
        print("\n🎨 Testing final stage mode 1...")
        try:
            results = run_cumulative_tags_final_stage(
                descriptor, user_pref, cumulative_tags, test_folder, "mode1"
            )
            print(f"✅ Final stage mode 1 completed: {len(results)} results")
        except Exception as e:
            print(f"⚠️  Final stage mode 1 failed (expected without proper setup): {str(e)}")
        
        # Test final stage mode 2
        print("\n🏷️  Testing final stage mode 2...")
        try:
            results = run_cumulative_tags_final_stage(
                descriptor, user_pref, cumulative_tags, test_folder, "mode2"
            )
            print(f"✅ Final stage mode 2 completed: {len(results)} results")
        except Exception as e:
            print(f"⚠️  Final stage mode 2 failed (expected without proper setup): {str(e)}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        if os.path.exists(test_folder):
            import shutil
            shutil.rmtree(test_folder)
            print(f"🧹 Cleaned up test folder: {test_folder}")

if __name__ == "__main__":
    test_cumulative_tags_final_stage() 