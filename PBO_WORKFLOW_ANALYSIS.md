# PBO Workflow Analysis

## Current Implementation Flow

### Stage 1: Exploration (Impression)
```
User likes/dislikes tags
  ↓
handle_tag_click() → updates like_count/dislike_count
  ↓
compute_weights() → calculates concept weights via softmax
  Formula: score = a*likes - b*dislikes + rank_bonus - rank_penalty
  Then: weights = softmax(scores/τ)
  ↓
save_concept_weights() → saves to impression/concept_weights.json
```

**Saved Data Format:**
```json
{
  "concept_weights": [
    {
      "concept_id": "c0",
      "label": "warm lighting",
      "weight": 0.234,
      "score": 2.5,
      "like_count": 5,
      "dislike_count": 1
    },
    ...
  ]
}
```

### Stage 2: Refinement Initialization (Impression Refinement)
```
User clicks "Generate Refinement"
  ↓
get_or_create_pbo_refiner()
  ↓
get_refinement_session() → clusters tags into concepts
  ↓
load_concept_weights_from_base_stage() → loads impression/concept_weights.json
  Matches concepts by LABEL
  Transfers: w (weight), score
  ↓
StageRefiner.__init__()
  Extracts concept_weights from concept_states
  concept_weights[i] = concept_states[cid].get('w', 1/K)
  Normalizes to sum=1
  ↓
PBO.__init__(concept_weights=concept_weights)
  Stores as self.concept_weights
  Sorts by weight for cold start (self.sorted_indices)
```

**Verification Point:**
```python
# backend/stage_refiner.py:88-102
concept_weights = np.array([
    concept_states.get(cid, {}).get('w', 1.0 / self.K)
    for cid in self.concept_ids
], dtype=np.float32)

# Normalize weights to sum to 1
if concept_weights.sum() > 0:
    concept_weights = concept_weights / concept_weights.sum()
else:
    concept_weights = np.ones(self.K, dtype=np.float32) / self.K
```

### Stage 3: Round 1 - Cold Start Proposals
```
propose_next_4() called
  ↓
if not self.fitted or len(self.candidates) < 2:
  COLD START MODE
  ↓
  w_learned = self.concept_weights.copy()
  w_learned = w_learned / (w_learned.sum() + EPS)  # Normalize again
  ↓
  Generate 4 proposals:
    [1] Learned Baseline: w_learned (as-is)
    [2] Top-Heavy: amplify top-3 (×1.5), dampen rest (×0.5)
    [3] Diversified: reduce top-3 (×0.7), boost mid-tier (×1.8)
    [4] Smoothed: 70% learned + 30% uniform
  ↓
  Each proposal normalized via normalize_simplex()
```

**CRITICAL: Cold Start Logic**
```python
# backend/pbo.py:549-605
# Get learned weights (normalized)
w_learned = self.concept_weights.copy()
w_learned = w_learned / (w_learned.sum() + EPS)

# Strategy 1: Learned Baseline - use weights directly from exploration
w1 = w_learned.copy()
w1 = normalize_simplex(w1)  # Explicit normalization
proposals.append(w1)
```

### Stage 4: Weight → SDXL Generation
```
generate_images_from_proposals(proposals)
  ↓
For each weight vector w:
  concepts_to_sdxl_phrases(w, concepts)
    ↓
    normalize_simplex(w)  # Ensure sum=1
    ↓
    compute_gains(w_norm)  # z-score mapping
      z_scores = (w - mean) / std
      gains = 1.0 + 0.4*z_scores
      gains = clip(gains, 0.7, 1.5)
    ↓
    Select top-K positives (by weight)
    Select deficit negatives (w < uniform/2)
    ↓
    Returns: [(phrase, gain), ...], [neg_phrases]
  ↓
  fuse_weighted_phrases() → weighted embeddings
  ↓
  generate_embeds_img2img() → SDXL generation
```

### Stage 5: User Selection → Learning
```
User selects favorite image
  ↓
/api/pbo/refine-next-round
  ↓
Load actual weight vectors from round_N/weights.json
  ↓
Add all 4 weight vectors as candidates
  ↓
Add strong duels: favorite ≻ others (strength=1.0)
  ↓
refiner.pbo.fit() → Fit GP on duels
  Convert duels to utility scores via Copeland:
    utility = logit(wins/(wins+losses))
  Fit GP(Z, utility) where Z = mixture_embeddings
  ↓
propose_next_4(fit_first=True) → ROUND 2+ MODE
  Uses GP posterior to generate:
    [A] Anchor: best posterior mean
    [B] Local refinement: Dirichlet around best
    [C] Diverse: high uncertainty, far from A/B
    [D] Thompson/EI: optimistic sampling
```

## Potential Issues

### Issue 1: Multiple Normalizations
The weights go through multiple normalizations:
1. `compute_weights()` → softmax(scores/τ) 
2. `StageRefiner.__init__` → normalize to sum=1
3. `PBO.__init__` → stored as-is
4. `propose_batch()` → normalize again: `w_learned / (w_learned.sum() + EPS)`
5. Each proposal → `normalize_simplex()`
6. `concepts_to_sdxl_phrases()` → normalize again

**Question:** Are the relative magnitudes preserved correctly?

### Issue 2: Weight Transfer by Label Matching
```python
# backend/concept_refinement.py:1104-1120
weight_map = {w['label']: w for w in weights_data['concept_weights']}

for concept in self.concepts:
    if concept.label in weight_map:
        prev_weight_data = weight_map[concept.label]
        state.w = prev_weight_data.get('weight', state.w)
```

**Question:** Do concept labels match exactly between exploration and refinement?

### Issue 3: Gain Computation
```python
# backend/sdxl_integration.py:26-52
mean_w = np.mean(w)
std_w = np.std(w)
z_scores = (w - mean_w) / (std_w + 1e-8)
gains = 1.0 + 0.4 * z_scores
gains = np.clip(gains, 0.7, 1.5)
```

**Question:** Does z-score normalization distort the learned preferences?
- If all weights are similar → low std → z-scores compressed → gains all ~1.0
- If weights are varied → high std → z-scores spread → gains vary 0.7-1.5

### Issue 4: Top-K Selection in SDXL
```python
# backend/sdxl_runner.py:145
pos_phrases, neg_phrases = concepts_to_sdxl_phrases(
    w=w,
    concepts=concepts,
    top_k=9999,  # Effectively all concepts!
    num_negatives=3
)
```

**MAJOR ISSUE:** `top_k=9999` means ALL concepts are included in positive prompt!
This dilutes the learned preferences.

## Recommendations

### 1. Set Appropriate top_k
```python
# Should be much smaller, e.g., 10
pos_phrases, neg_phrases = concepts_to_sdxl_phrases(
    w=w,
    concepts=concepts,
    top_k=10,  # Only top 10 concepts
    num_negatives=5
)
```

### 2. Verify Weight Preservation
Add debug logging to check if learned weights are preserved:
```python
print(f"[DEBUG] Learned weights (top-3): {w_learned[top_3_indices]}")
print(f"[DEBUG] After normalization (top-3): {w1[top_3_indices]}")
print(f"[DEBUG] After SDXL projection (top-3): {w_sdxl[top_3_indices]}")
```

### 3. Check Concept Matching
Verify that concepts from exploration match refinement:
```python
print(f"[DEBUG] Exploration concepts: {[c['label'] for c in exploration_concepts]}")
print(f"[DEBUG] Refinement concepts: {[c['label'] for c in refinement_concepts]}")
print(f"[DEBUG] Matched: {matched_count}/{len(concepts)}")
```

### 4. Reduce Unnecessary Normalizations
The weight vector should already be normalized after `compute_weights()`.
Remove redundant normalizations in the pipeline.

### 5. Adjust Gain Computation
Consider using absolute weights instead of z-scores:
```python
# Option 1: Direct gain from weight
gains = 0.7 + 0.8 * w_norm  # Range [0.7, 1.5] for w_norm ∈ [0, 1]

# Option 2: Power transformation
gains = w_norm ** 0.5  # Square root to reduce variance
```

## Next Steps

1. Check if `top_k=9999` is the actual issue (likely!)
2. Add debug logging to trace weight values through pipeline
3. Verify concept label matching
4. Simplify normalization pipeline
5. Test with different gain computation methods

