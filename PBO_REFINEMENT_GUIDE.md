# PBO Refinement Guide

## Overview

This guide explains how to use the new PBO (Preference-Based Optimization) refinement system that uses tag cluster concepts with SDXL image generation.

## System Architecture

### Impression Stage (Base)
1. User input → Gemini generates 4 concepts
2. Gemini generates 4 images  
3. **Extract visual tags** from images → saved to `impression/visual_tags.json`
4. User selects favorite image

### Impression Refinement Stage (PBO)
1. **Cluster tags into concepts** (K-means on CLIP embeddings)
2. Each concept has a **centroid** (average embedding of tags in cluster)
3. Initialize **PBO** with concept centroids → creates MU matrix (K concepts × d dimensions)
4. **Iterative refinement loop**:
   - PBO proposes 4 weight mixtures: `w^1, w^2, w^3, w^4`
   - Convert to embeddings: `z = L2_normalize(w @ MU)`
   - SDXL generates 4 images using fused embeddings
   - User picks favorite
   - PBO learns from preference → updates GP model
   - Repeat

---

## API Endpoints

### 1. Initialize Refinement
**POST** `/api/pbo/init-refinement`

Loads visual tags, clusters them into concepts, and initializes PBO.

**Request:**
```json
{
  "session_id": "session_123",
  "stage": "impression",
  "image_ids": ["impression_0", "impression_1", "impression_2", "impression_3"]
}
```

**Response:**
```json
{
  "success": true,
  "num_concepts": 15,
  "concept_labels": ["warm lighting", "wooden furniture", "cozy atmosphere", ...],
  "message": "Initialized PBO refinement with 15 tag cluster concepts"
}
```

**What it does:**
- Loads `visual_tags.json` from impression stage
- Runs K-means clustering on tag CLIP embeddings
- Creates concepts with centroids (MU matrix)
- Initializes StageRefiner with PBO

---

### 2. Propose Next 4 Mixtures
**POST** `/api/pbo/propose`

Generates 4 new weight mixtures using PBO acquisition.

**Request:**
```json
{
  "session_id": "session_123",
  "stage": "impression",
  "negatives": null,
  "w_current": null
}
```

**Response:**
```json
{
  "proposals": [
    [0.3, 0.2, 0.15, 0.1, 0.05, ...],  // weights for 15 concepts
    [0.25, 0.25, 0.2, 0.1, 0.05, ...],
    [0.4, 0.15, 0.15, 0.1, 0.05, ...],
    [0.2, 0.3, 0.15, 0.15, 0.05, ...]
  ],
  "proposal_ids": ["prop_0", "prop_1", "prop_2", "prop_3"],
  "message": "Generated 4 proposals"
}
```

**What it does:**
- PBO uses Gaussian Process to predict which mixtures user will prefer
- Uses acquisition strategies: Thompson sampling, EI, variance, diversity
- Returns 4 weight vectors (simplex: sum to 1)

---

### 3. Generate Images from Proposals
**POST** `/api/pbo/generate`

Converts weight mixtures to SDXL images.

**Request:**
```json
{
  "session_id": "session_123",
  "stage": "impression",
  "proposals": [
    [0.3, 0.2, 0.15, ...],
    [0.25, 0.25, 0.2, ...],
    ...
  ],
  "seed_base": 42
}
```

**Response:**
```json
{
  "image_paths": [
    "/sessions/session_123/impression/pbo_round_0/image_0.png",
    "/sessions/session_123/impression/pbo_round_0/image_1.png",
    "/sessions/session_123/impression/pbo_round_0/image_2.png",
    "/sessions/session_123/impression/pbo_round_0/image_3.png"
  ],
  "proposals": [...],
  "round_number": 0,
  "message": "Generated 4 images in round 0"
}
```

**What it does:**
- For each weight vector `w`: computes `z = L2_normalize(w @ MU)`
- SDXL fuses concept embeddings using weights
- Generates image with fused embedding
- Saves to `pbo_round_N/` directory

---

### 4. Record Favorite Selection
**POST** `/api/pbo/favorite`

Records user's favorite image selection as strong duels.

**Request:**
```json
{
  "session_id": "session_123",
  "stage": "impression",
  "favorite_image_id": "image_2",
  "all_image_ids": ["image_0", "image_1", "image_2", "image_3"]
}
```

**Response:**
```json
{
  "duels_added": 3,
  "favorite_candidate_id": "cand_abc123",
  "message": "Recorded 3 strong duels"
}
```

**What it does:**
- Creates strong preference duels: favorite > others
- PBO learns: updates Gaussian Process model
- Next proposals will be better informed

---

## Complete Workflow Example

### Step 1: Run Impression Stage (Existing Flow)
```bash
# User runs impression stage
POST /api/generate-fast
{
  "descriptor": "A cozy reading nook",
  "mode": "fast"
}

# System:
# - Generates 4 concepts
# - Generates 4 images
# - Extracts visual tags → visual_tags.json
```

### Step 2: User Selects Favorite
```bash
# User picks impression_2 as favorite
POST /api/feedback
{
  "session_id": "session_123",
  "stage": "impression",
  "choice": "impression_2"
}
```

### Step 3: Initialize PBO Refinement
```bash
POST /api/pbo/init-refinement
{
  "session_id": "session_123",
  "stage": "impression",
  "image_ids": ["impression_0", "impression_1", "impression_2", "impression_3"]
}

# System clusters tags:
# - 42 tags → K-means → 15 concepts
# - Creates MU matrix (15 × 512 CLIP dimensions)
# - Initializes PBO GP model
```

### Step 4: Iterative Refinement Loop

**Round 1:**
```bash
# Propose 4 new mixtures
POST /api/pbo/propose
{
  "session_id": "session_123",
  "stage": "impression"
}
# Returns 4 weight vectors

# Generate images
POST /api/pbo/generate
{
  "session_id": "session_123",
  "stage": "impression",
  "proposals": [[...], [...], [...], [...]]
}
# Returns 4 images

# User picks favorite (e.g., image_1)
POST /api/pbo/favorite
{
  "session_id": "session_123",
  "stage": "impression",
  "favorite_image_id": "image_1",
  "all_image_ids": ["image_0", "image_1", "image_2", "image_3"]
}
# PBO learns from preference
```

**Round 2, 3, 4...** (repeat):
```bash
POST /api/pbo/propose  → 4 new proposals
POST /api/pbo/generate → 4 new images
POST /api/pbo/favorite → user picks favorite
```

---

## Key Concepts

### Tag Cluster Concepts
- Visual tags extracted from images (e.g., "warm lighting", "wooden table")
- K-means clusters similar tags together
- Each cluster = one concept
- Concept centroid = average CLIP embedding of tags in cluster

### Weight Mixtures
- `w` = vector of weights for each concept
- Simplex constraint: all weights ≥ 0, sum to 1
- Example: `w = [0.3, 0.2, 0.15, 0.1, 0.05, ...]`
  - 30% of concept 1 (warm lighting)
  - 20% of concept 2 (wooden furniture)
  - etc.

### Embedding Fusion
- Each concept has centroid embedding `μ_k` (CLIP vector)
- MU matrix = stack of all centroids (K × d)
- Mixture embedding: `z = L2_normalize(w @ MU)`
- This fuses concepts according to weights

### PBO Learning
- Gaussian Process learns user preferences
- Input: weight mixture `w`
- Output: predicted preference score
- Uses cosine-RBF kernel on embedding space
- Acquisition strategies find promising new mixtures

---

## File Structure

```
sessions/
  session_123/
    impression/
      impression.json          # 4 original concepts
      visual_tags.json         # Extracted tags
      impression_0_0.png       # Original images
      impression_1_0.png
      impression_2_0.png
      impression_3_0.png
      pbo_round_0/            # First PBO refinement
        image_0.png
        image_1.png
        image_2.png
        image_3.png
      pbo_round_1/            # Second PBO refinement
        image_0.png
        ...
```

---

## Troubleshooting

### Error: "Visual tags not found"
**Solution:** Run the impression stage first. The system needs `visual_tags.json`.

### Error: "Concept session not initialized"
**Solution:** Call `/api/pbo/init-refinement` before using other PBO endpoints.

### Error: "CLIP model not found"
**Solution:** Install CLIP: `pip install git+https://github.com/openai/CLIP.git`

### SDXL is slow
**Solution:** 
- Use GPU: Check `torch.cuda.is_available()`
- Reduce steps: Default is 30, try 20
- Use smaller model: Try `stable-diffusion-xl-base-0.9`

---

## Next Steps

Once PBO refinement works for impression stage, extend to:
- Spatial refinement
- Objects refinement  
- Ambient refinement
- Final stage refinement

The same flow applies to all stages!


