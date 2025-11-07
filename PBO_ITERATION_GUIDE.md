# PBO Iterative Refinement Guide

## Overview

This guide explains how PBO learns from user preferences across multiple rounds and how to implement the frontend iteration loop.

---

## How PBO Works (Not Random!)

### 1. Embedding Space Navigation
Each proposed image represents a **specific weight mixture** over tag cluster concepts:

```
w = [0.3, 0.2, 0.15, 0.1, 0.05, ...]  # weights for K concepts
z = L2_normalize(w @ MU)               # embedding (768-dim CLIP space)
```

You're navigating in **embedding space** where similar concepts are close together.

### 2. Learning from Preferences

**Round 1 (Cold Start):**
- No preferences yet → explores corners + center of the simplex
- Like trying coffee with: 100% milk, 100% sugar, 100% coffee, or 33% each

**Round 2+ (GP-Guided):**
When you pick a favorite, PBO records **duels**:
```
favorite > image_1
favorite > image_2  
favorite > image_3
```

Then fits a **Gaussian Process** that learns:
- Which embeddings you prefer
- Where to explore next
- Uncertainty estimates

### 3. Acquisition Strategies (4 per batch)

Each proposal uses a different strategy:

1. **Thompson Sampling**: Sample from GP posterior (exploit best predictions)
2. **Expected Improvement**: Where GP predicts improvement over current best
3. **Variance Sampling**: High uncertainty areas (explore unknowns)
4. **Diversity**: Maximize distance from other proposals (avoid redundancy)

This ensures you get **diverse, informative proposals** each round!

---

## API Endpoints for Iteration

### Round 1: Initial Refinement

```bash
# After user selects favorite from impression stage
POST /api/feedback
{
  "session_id": "session_123",
  "stage": "impression",
  "selected_image_id": "impression_2_0",
  "preferences": {...}
}

# Backend automatically:
# 1. Clusters tags into concepts
# 2. Initializes PBO
# 3. Proposes 4 weight mixtures (cold start)
# 4. Generates 4 images using SDXL img2img
# 
# Returns: 4 refinement images
```

### Round 2+: Iterative Refinement

**Step 1: Record Favorite**
```bash
POST /api/pbo/record-refinement-favorite
{
  "session_id": "session_123",
  "stage": "impression",  # base stage, not impression_refinement
  "favorite_image_id": "refinement_round_1_image_2",
  "all_image_ids": [
    "refinement_round_1_image_0",
    "refinement_round_1_image_1", 
    "refinement_round_1_image_2",
    "refinement_round_1_image_3"
  ]
}

# This records duels in PBO:
# refinement_round_1_image_2 > others
```

**Step 2: Propose New Mixtures**
```bash
POST /api/pbo/propose
{
  "session_id": "session_123",
  "stage": "impression",
  "negatives": null,
  "w_current": null
}

# Returns 4 new weight mixtures informed by previous rounds
Response: {
  "proposals": [
    [0.25, 0.30, 0.15, 0.10, ...],  # 4 weight vectors
    [0.20, 0.35, 0.12, 0.08, ...],
    [0.28, 0.25, 0.18, 0.09, ...],
    [0.22, 0.32, 0.14, 0.07, ...]
  ],
  "proposal_ids": ["prop_0", "prop_1", "prop_2", "prop_3"]
}
```

**Step 3: Generate Images**
```bash
POST /api/pbo/generate
{
  "session_id": "session_123",
  "stage": "impression",
  "proposals": [[...], [...], [...], [...]],  # from step 2
  "seed_base": 42
}

# Returns 4 new images
Response: {
  "image_paths": [
    "/sessions/session_123/impression/pbo_round_2/image_0.png",
    "/sessions/session_123/impression/pbo_round_2/image_1.png",
    "/sessions/session_123/impression/pbo_round_2/image_2.png",
    "/sessions/session_123/impression/pbo_round_2/image_3.png"
  ],
  "round_number": 2
}
```

**Repeat Steps 1-3** until user is satisfied!

---

## Frontend Implementation (React Example)

```javascript
// State
const [round, setRound] = useState(1);
const [images, setImages] = useState([]);
const [proposals, setProposals] = useState([]);
const [canRefineMore, setCanRefineMore] = useState(false);

// After round 1 (from /api/feedback)
useEffect(() => {
  if (images.length === 4) {
    setCanRefineMore(true);  // Show "Refine More" button
  }
}, [images]);

// User selects favorite
async function onSelectFavorite(favoriteId) {
  const allIds = images.map(img => img.id);
  
  // Record favorite
  await fetch('/api/pbo/record-refinement-favorite', {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
      stage: 'impression',
      favorite_image_id: favoriteId,
      all_image_ids: allIds
    })
  });
  
  setCanRefineMore(true);
}

// User clicks "Refine More"
async function onRefineMore() {
  setCanRefineMore(false);
  setRound(round + 1);
  
  // Step 1: Propose new mixtures
  const proposeRes = await fetch('/api/pbo/propose', {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
      stage: 'impression',
      negatives: null,
      w_current: null
    })
  });
  const { proposals: newProposals } = await proposeRes.json();
  setProposals(newProposals);
  
  // Step 2: Generate images
  const genRes = await fetch('/api/pbo/generate', {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
      stage: 'impression',
      proposals: newProposals,
      seed_base: 42 + round  // Different seed per round
    })
  });
  const { image_paths } = await genRes.json();
  
  // Update images
  setImages(image_paths.map((path, i) => ({
    id: `round_${round}_image_${i}`,
    url: path
  })));
  
  setCanRefineMore(true);  // Enable next iteration
}
```

---

## UI Flow

```
Round 1:
  [Image 0] [Image 1] [Image 2] [Image 3]
  User clicks Image 2 (favorite)
  → "Refine More" button appears

Click "Refine More":
  → Backend: Record duels (Image 2 > others)
  → Backend: Fit GP on all duels
  → Backend: Propose 4 new mixtures using acquisition
  → Backend: Generate 4 new images

Round 2:
  [Image 0] [Image 1] [Image 2] [Image 3]  (NEW images)
  User clicks Image 1 (favorite)
  → "Refine More" button appears

Click "Refine More":
  → Backend: Record duels (Round2_Image1 > others)
  → Backend: Re-fit GP with MORE data
  → Backend: Propose 4 BETTER mixtures
  → Backend: Generate 4 new images

Round 3, 4, 5...
  → Images get progressively closer to user's ideal
  → GP becomes more confident
  → Proposals exploit learned preferences
```

---

## Convergence

PBO typically converges in **3-7 rounds**:

- **Rounds 1-2**: Exploration (GP learning broad preferences)
- **Rounds 3-5**: Refinement (GP narrowing down)
- **Rounds 5+**: Fine-tuning (small adjustments)

### When to Stop?

**Option 1: User decides** - Stop button + "Use this image"

**Option 2: Auto-stop** - When GP variance drops below threshold:
```python
if pbo.gp_variance < 0.1:  # High confidence
    print("Converged!")
```

**Option 3: Max rounds** - Stop after N rounds (e.g., 10)

---

## Verification: Is It Learning?

To verify PBO is learning (not random), check:

### 1. GP is fitting
```
[PBO PROPOSE] Fitting GP on 12 duels...  ← Should increase each round
[GP] Fitted with 12 samples
```

### 2. Acquisition uses predictions
```
  Strategy 1/4: thompson      ← GP-guided
  Strategy 2/4: ei            ← GP-guided  
  Strategy 3/4: variance      ← GP-guided
  Strategy 4/4: diverse       ← Geometry-based
```

### 3. Proposals differ across rounds
Round 1:
```
proposals[0] = [0.25, 0.25, 0.25, 0.25]  # Uniform (cold start)
```

Round 2:
```
proposals[0] = [0.45, 0.30, 0.15, 0.10]  # Learned to emphasize concepts 0, 1
```

Round 3:
```
proposals[0] = [0.60, 0.25, 0.10, 0.05]  # Further emphasis on concept 0
```

---

## Testing PBO Iteration

### Manual Test (cURL)

```bash
# Round 1
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "stage": "impression", "selected_image_id": "impression_2_0", ...}'

# User picks refinement favorite
curl -X POST http://localhost:8000/api/pbo/record-refinement-favorite \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "stage": "impression",
    "favorite_image_id": "round_1_image_2",
    "all_image_ids": ["round_1_image_0", "round_1_image_1", "round_1_image_2", "round_1_image_3"]
  }'

# Round 2: Propose
curl -X POST http://localhost:8000/api/pbo/propose \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "stage": "impression"}'

# Round 2: Generate
curl -X POST http://localhost:8000/api/pbo/generate \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "stage": "impression",
    "proposals": [[...], [...], [...], [...]],
    "seed_base": 43
  }'

# Repeat...
```

---

## Summary

✅ **PBO is learning properly** - Uses GP + acquisition functions, not random

✅ **Endpoints are ready** - Just need frontend loop

✅ **Each round improves** - GP learns from duels → proposes better mixtures

✅ **Navigating embedding space** - Converges to user's ideal concept mixture

🎯 **Next Step**: Implement frontend iteration using the endpoints above!


