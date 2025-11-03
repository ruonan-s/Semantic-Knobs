"""
Test script to debug CLIP clustering behavior
"""
import torch
import clip
import numpy as np

# Load CLIP
print("Loading CLIP ViT-L/14...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
print(f"Loaded on {device}")

def get_similarity(text1, text2):
    """Get cosine similarity between two texts"""
    with torch.no_grad():
        tokens = clip.tokenize([text1, text2], truncate=True).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        embeddings = features.cpu().numpy()
    
    sim = np.dot(embeddings[0], embeddings[1])
    return float(sim)

# Test common visual tag pairs
print("\n" + "="*60)
print("TESTING VISUAL TAG SIMILARITIES (CLIP ViT-L/14)")
print("="*60)

# Test cases: (tag1, tag2, expected_behavior)
test_cases = [
    # Should merge (similar concepts)
    ("blue sky", "azure sky", "SHOULD MERGE"),
    ("blue sky", "clear sky", "SHOULD MERGE"),
    ("red car", "crimson car", "SHOULD MERGE"),
    ("tall building", "skyscraper", "SHOULD MERGE"),
    ("green tree", "verdant tree", "SHOULD MERGE"),
    
    # Should NOT merge (different concepts)
    ("blue sky", "blue car", "SHOULD NOT MERGE"),
    ("blue sky", "red sky", "SHOULD NOT MERGE"),
    ("tall building", "tall tree", "SHOULD NOT MERGE"),
    ("red car", "red carpet", "SHOULD NOT MERGE"),
    ("white cloud", "white snow", "SHOULD NOT MERGE"),
    
    # Edge cases
    ("blue", "azure", "EDGE CASE"),
    ("sky", "clouds", "EDGE CASE"),
    ("car", "vehicle", "EDGE CASE"),
]

current_threshold = 0.85

print(f"\nCurrent threshold: {current_threshold}")
print(f"Similarities >= {current_threshold} will merge\n")

merge_count = 0
no_merge_count = 0
correct_merges = 0
incorrect_merges = 0

for tag1, tag2, expected in test_cases:
    sim = get_similarity(tag1, tag2)
    will_merge = sim >= current_threshold
    
    # Check if behavior matches expectation
    status = "✓" if (
        (expected == "SHOULD MERGE" and will_merge) or
        (expected == "SHOULD NOT MERGE" and not will_merge)
    ) else "✗"
    
    if expected == "SHOULD MERGE":
        if will_merge:
            correct_merges += 1
        else:
            incorrect_merges += 1
    elif expected == "SHOULD NOT MERGE":
        if not will_merge:
            correct_merges += 1
        else:
            incorrect_merges += 1
    
    merge_str = "WILL MERGE" if will_merge else "SEPARATE"
    
    print(f"{status} '{tag1}' ↔ '{tag2}'")
    print(f"   Similarity: {sim:.4f} → {merge_str} ({expected})")
    
    if will_merge:
        merge_count += 1
    else:
        no_merge_count += 1

print("\n" + "="*60)
print(f"SUMMARY with threshold={current_threshold}")
print("="*60)
print(f"Total pairs tested: {len(test_cases)}")
print(f"Would merge: {merge_count}")
print(f"Would stay separate: {no_merge_count}")
print(f"Correct behavior: {correct_merges}/{len([t for t in test_cases if t[2] != 'EDGE CASE'])}")
print(f"Incorrect behavior: {incorrect_merges}")

# Suggest better threshold
print("\n" + "="*60)
print("THRESHOLD RECOMMENDATIONS")
print("="*60)

similarities = [get_similarity(t[0], t[1]) for t in test_cases]
should_merge_sims = [get_similarity(t[0], t[1]) for t in test_cases if t[2] == "SHOULD MERGE"]
should_not_merge_sims = [get_similarity(t[0], t[1]) for t in test_cases if t[2] == "SHOULD NOT MERGE"]

if should_merge_sims and should_not_merge_sims:
    min_should_merge = min(should_merge_sims)
    max_should_not_merge = max(should_not_merge_sims)
    
    print(f"Min similarity for 'should merge': {min_should_merge:.4f}")
    print(f"Max similarity for 'should not merge': {max_should_not_merge:.4f}")
    
    if min_should_merge > max_should_not_merge:
        optimal = (min_should_merge + max_should_not_merge) / 2
        print(f"\n✓ Good separation exists!")
        print(f"  Recommended threshold: {optimal:.4f}")
    else:
        print(f"\n⚠ Overlap detected! No perfect threshold exists.")
        print(f"  Consider using threshold: {min_should_merge:.4f} (prioritize merging similar)")
        print(f"  Or threshold: {max_should_not_merge:.4f} (prioritize keeping different separate)")

print("\n" + "="*60)
print("To adjust threshold, edit backend/concept_refinement.py:")
print(f"  Current: THETA_MERGE = {current_threshold}")
print("="*60)

