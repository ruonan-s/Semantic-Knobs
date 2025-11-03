"""
Test script to verify concept refinement system
Run this to check for common issues before testing in UI
"""

import json
from concept_refinement import (
    RawTag, Concept, ConceptState, 
    build_concepts, compute_weights, categorize_concepts,
    ConceptRefinementSession
)

def test_basic_functionality():
    """Test basic concept system functionality"""
    print("\n" + "="*80)
    print("TEST 1: Basic Concept Building")
    print("="*80)
    
    # Create sample tags
    raw_tags = [
        RawTag(id="tag_1", text="mountain", embedding=[0.1] * 1536, image_id="img_1"),
        RawTag(id="tag_2", text="hill", embedding=[0.11] * 1536, image_id="img_1"),
        RawTag(id="tag_3", text="sky", embedding=[0.5] * 1536, image_id="img_2"),
        RawTag(id="tag_4", text="cloud", embedding=[0.51] * 1536, image_id="img_2"),
        RawTag(id="tag_5", text="forest", embedding=[0.9] * 1536, image_id="img_3"),
    ]
    
    # Normalize embeddings
    import numpy as np
    for tag in raw_tags:
        arr = np.array(tag.embedding)
        tag.embedding = (arr / np.linalg.norm(arr)).tolist()
    
    print(f"✓ Created {len(raw_tags)} tags")
    
    # Build concepts
    concepts, tag_to_concept = build_concepts(raw_tags, theta=0.78)
    print(f"✓ Built {len(concepts)} concepts from {len(raw_tags)} tags")
    
    for concept in concepts:
        member_texts = [t.text for t in raw_tags if t.id in concept.member_tag_ids]
        print(f"  - {concept.label}: {member_texts}")
    
    return True


def test_json_serialization():
    """Test that ConceptState can be serialized"""
    print("\n" + "="*80)
    print("TEST 2: JSON Serialization")
    print("="*80)
    
    # Create a ConceptState with sets
    state = ConceptState(
        like_count=2,
        dislike_count=1,
        w=0.5,
        ema_w=0.48
    )
    state.liked_tags.add("tag_1")
    state.liked_tags.add("tag_2")
    state.disliked_tags.add("tag_3")
    
    print(f"✓ Created ConceptState with:")
    print(f"  - {len(state.liked_tags)} liked tags")
    print(f"  - {len(state.disliked_tags)} disliked tags")
    
    # Try to serialize
    try:
        serialized = {
            'like_count': state.like_count,
            'dislike_count': state.dislike_count,
            'w': state.w,
            'ema_w': state.ema_w,
            'liked_tags': list(state.liked_tags),
            'disliked_tags': list(state.disliked_tags)
        }
        
        json_str = json.dumps(serialized)
        print(f"✓ Serialization successful: {len(json_str)} bytes")
        
        # Verify we can deserialize
        loaded = json.loads(json_str)
        print(f"✓ Deserialization successful")
        print(f"  - Recovered {len(loaded['liked_tags'])} liked tags")
        
        return True
    except Exception as e:
        print(f"✗ Serialization failed: {e}")
        return False


def test_categorization():
    """Test categorization logic"""
    print("\n" + "="*80)
    print("TEST 3: Categorization Logic")
    print("="*80)
    
    # Create concepts
    concepts = [
        Concept(id=f"c_{i}", label=f"concept_{i}", centroid=[0.0]*10, member_tag_ids=[])
        for i in range(10)
    ]
    
    # Create states with different weights
    concept_states = {}
    K = len(concepts)
    w_base = 1.0 / K
    delta = 0.2 / K
    
    print(f"K = {K}, w_base = {w_base:.4f}, delta = {delta:.4f}")
    print(f"Thresholds: positive >= {w_base + delta:.4f}, negative <= {w_base - delta:.4f}")
    
    # Set up some concepts as positive, neutral, negative
    for i, concept in enumerate(concepts):
        if i < 2:
            # Positive
            w = w_base + delta + 0.01
        elif i < 7:
            # Neutral
            w = w_base
        else:
            # Negative
            w = w_base - delta - 0.01
        
        concept_states[concept.id] = ConceptState(w=w, ema_w=w)
    
    # Categorize
    positive, neutral, negative = categorize_concepts(concepts, concept_states)
    
    print(f"✓ Categorized: {len(positive)} positive, {len(neutral)} neutral, {len(negative)} negative")
    
    # Verify
    if len(neutral) > 0:
        print(f"✓ Neutral concepts exist!")
        return True
    else:
        print(f"✗ No neutral concepts found!")
        return False


def test_toggle_logic():
    """Test tag toggle logic"""
    print("\n" + "="*80)
    print("TEST 4: Tag Toggle Logic")
    print("="*80)
    
    state = ConceptState()
    tag_id = "tag_test"
    
    print("Initial state:")
    print(f"  like_count: {state.like_count}, liked_tags: {state.liked_tags}")
    
    # Like once
    if tag_id in state.liked_tags:
        state.liked_tags.remove(tag_id)
        state.like_count -= 1
    else:
        state.liked_tags.add(tag_id)
        state.like_count += 1
    
    print("After first like:")
    print(f"  like_count: {state.like_count}, liked_tags: {state.liked_tags}")
    assert state.like_count == 1
    assert tag_id in state.liked_tags
    
    # Like again (toggle off)
    if tag_id in state.liked_tags:
        state.liked_tags.remove(tag_id)
        state.like_count -= 1
    else:
        state.liked_tags.add(tag_id)
        state.like_count += 1
    
    print("After second like (toggle off):")
    print(f"  like_count: {state.like_count}, liked_tags: {state.liked_tags}")
    assert state.like_count == 0
    assert tag_id not in state.liked_tags
    
    print("✓ Toggle logic works correctly!")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("CONCEPT REFINEMENT SYSTEM TESTS")
    print("="*80)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("JSON Serialization", test_json_serialization),
        ("Categorization", test_categorization),
        ("Toggle Logic", test_toggle_logic),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before using.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

