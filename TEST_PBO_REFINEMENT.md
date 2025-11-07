# Testing PBO Refinement

## Quick Test Commands

### 1. Start the Server

```bash
cd /home/nancy/Exploration+Refinement/backend
conda activate apl
uvicorn server:app --reload --port 8000 --host 0.0.0.0
```

---

## Test Sequence

### Step 1: Generate Impression Stage (if needed)

First, you need a session with visual tags. If you already have one, skip this step.

```bash
curl -X POST http://localhost:8000/api/generate-fast \
  -H "Content-Type: application/json" \
  -d '{
    "descriptor": "A cozy reading nook with warm lighting",
    "mode": "fast"
  }'
```

Note the `session_id` from the response.

---

### Step 2: Initialize PBO Refinement

```bash
curl -X POST http://localhost:8000/api/pbo/init-refinement \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "stage": "impression",
    "image_ids": ["impression_0", "impression_1", "impression_2", "impression_3"]
  }'
```

**Expected Response:**
```json
{
  "success": true,
  "num_concepts": 15,
  "concept_labels": ["warm lighting", "wooden furniture", ...],
  "message": "Initialized PBO refinement with 15 tag cluster concepts"
}
```

**What to check:**
- ✅ `success` is `true`
- ✅ `num_concepts` > 0 (typically 5-50)
- ✅ `concept_labels` contains meaningful tag clusters

---

### Step 3: Propose Next 4 Mixtures

```bash
curl -X POST http://localhost:8000/api/pbo/propose \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "stage": "impression",
    "negatives": null,
    "w_current": null
  }'
```

**Expected Response:**
```json
{
  "proposals": [
    [0.25, 0.20, 0.15, 0.10, ...],
    [0.30, 0.18, 0.12, 0.08, ...],
    [0.22, 0.22, 0.16, 0.09, ...],
    [0.28, 0.19, 0.14, 0.11, ...]
  ],
  "proposal_ids": ["prop_0", "prop_1", "prop_2", "prop_3"],
  "message": "Generated 4 proposals"
}
```

**What to check:**
- ✅ 4 proposal arrays
- ✅ Each array length matches `num_concepts`
- ✅ Each array sums to ~1.0 (simplex constraint)
- ✅ All values ≥ 0

---

### Step 4: Generate Images (Requires SDXL)

**Note:** This step requires SDXL model and GPU. It will be slow on CPU (~2-3 min per image).

```bash
curl -X POST http://localhost:8000/api/pbo/generate \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "stage": "impression",
    "proposals": [
      [0.25, 0.20, 0.15, 0.10, ...],
      [0.30, 0.18, 0.12, 0.08, ...],
      [0.22, 0.22, 0.16, 0.09, ...],
      [0.28, 0.19, 0.14, 0.11, ...]
    ],
    "seed_base": 42
  }'
```

**Copy the proposals from Step 3 response!**

**Expected Response:**
```json
{
  "image_paths": [
    "/sessions/YOUR_SESSION_ID/impression/pbo_round_0/image_0.png",
    "/sessions/YOUR_SESSION_ID/impression/pbo_round_0/image_1.png",
    "/sessions/YOUR_SESSION_ID/impression/pbo_round_0/image_2.png",
    "/sessions/YOUR_SESSION_ID/impression/pbo_round_0/image_3.png"
  ],
  "proposals": [...],
  "round_number": 0,
  "message": "Generated 4 images in round 0"
}
```

**What to check:**
- ✅ 4 image paths returned
- ✅ Files exist on disk
- ✅ Images are viewable (1024×1024 PNG)

---

### Step 5: Record Favorite

```bash
curl -X POST http://localhost:8000/api/pbo/favorite \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "stage": "impression",
    "favorite_image_id": "image_2",
    "all_image_ids": ["image_0", "image_1", "image_2", "image_3"]
  }'
```

**Expected Response:**
```json
{
  "duels_added": 3,
  "favorite_candidate_id": "cand_abc123",
  "message": "Recorded 3 strong duels"
}
```

**What to check:**
- ✅ `duels_added` = 3 (favorite vs 3 others)
- ✅ `favorite_candidate_id` returned

---

### Step 6: Iterate (Repeat Steps 3-5)

Now repeat propose → generate → favorite to continue refinement.

The PBO model learns from each favorite selection, so proposals should get better over time!

---

## Debugging

### Check Server Logs

Watch for these log messages:

```
[PBO Init] Initializing refinement for session_123/impression
[PBO Init] Loaded 42 tags from 4 images
[PBO Init] Clustering tags into concepts...
[PBO Init] Created 15 concepts
[PBO Init] ✅ Refinement initialized with 15 concepts
```

### Check Visual Tags File

```bash
cat sessions/YOUR_SESSION_ID/impression/visual_tags.json
```

Should contain tags for each image:
```json
{
  "impression_0_0.png": ["warm lighting", "wooden table", "cozy atmosphere", ...],
  "impression_1_0.png": ["modern design", "white walls", ...],
  ...
}
```

### Check Concepts

After init, the ConceptRefinementSession stores clustered concepts. Check server logs for:
```
[K-MEANS CLUSTERING] Building concepts from 42 tags
  Running K-means with K=15 clusters...
  Created 15 non-empty clusters
```

### Verify CLIP Model

```bash
conda activate apl
python -c "import clip; print(clip.available_models())"
```

Should show: `['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', ...]`

### Verify SDXL

```bash
conda activate apl
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

For GPU: Should print `CUDA available: True`

---

## Common Errors

### "Visual tags not found"
**Cause:** Impression stage not run yet
**Fix:** Run `/api/generate-fast` first

### "Concept session not initialized"  
**Cause:** Didn't call `/api/pbo/init-refinement`
**Fix:** Call init endpoint first

### "No module named 'clip'"
**Cause:** CLIP not installed
**Fix:** `pip install git+https://github.com/openai/CLIP.git`

### SDXL OOM (Out of Memory)
**Cause:** GPU memory insufficient
**Fix:** 
- Reduce image size (512×512 instead of 1024×1024)
- Use CPU (slow but works)
- Close other GPU processes

---

## Performance Expectations

### With GPU (NVIDIA RTX 3090):
- Init: ~2-5 seconds
- Propose: ~0.1 seconds  
- Generate: ~10-20 seconds (4 images)
- Favorite: ~0.1 seconds

### With CPU:
- Init: ~5-10 seconds
- Propose: ~0.2 seconds
- Generate: ~8-12 minutes (4 images) ⚠️
- Favorite: ~0.1 seconds

---

## Success Criteria

The system works correctly if:

1. ✅ Init returns concepts with meaningful labels
2. ✅ Proposals are valid simplexes (sum to 1, all ≥ 0)
3. ✅ Images generate successfully
4. ✅ Favorite selection records duels
5. ✅ Subsequent proposals are different (PBO learns)

---

## Next Steps

After verifying the basic flow works:

1. Test with existing session data
2. Verify proposals improve over iterations
3. Test error handling (missing files, invalid inputs)
4. Benchmark performance on your hardware
5. Integrate with frontend UI


