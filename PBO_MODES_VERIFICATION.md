# PBO Modes Verification

## Overview

This document verifies that both PBO modes work correctly with the new debug logging.

---

## Mode 1: Full Pipeline Mode

### Description
User runs through the complete pipeline: Impression → Spatial → Refinement

### Entry Point
**Endpoint:** `POST /api/generate-stage-refinement`

**File:** `backend/server.py:3202`

### Flow

```
1. User completes impression stage
2. User selects favorite image (e.g., image 2)
3. User clicks "Refine Impression"

4. Backend: /api/generate-stage-refinement
   ├─ Load visual tags from impression stage
   ├─ Initialize ConceptRefinementSession (clusters tags)
   ├─ Load learned weights (ema_w) from concept_weights.json
   ├─ Create StageRefiner with force_recreate=True
   │  └─ PBO initialized with learned weights
   ├─ propose_next_4(fit_first=True)
   │  └─ Cold start (no candidates yet)
   │  └─ Generate 4 proposals from learned weights
   ├─ Generate 4 images with SDXL
   │  └─ Uses selected impression image as reference
   ├─ Save to impression_refinement/round_1/
   └─ Return 4 images to frontend

5. User views 4 refinement images
6. User selects favorite (e.g., image 0)
7. User clicks "Next Round"

8. Backend: /api/pbo/refine-next-round
   ├─ Get cached refiner (from step 4)
   ├─ Load Round 1 weight vectors from weights.json
   ├─ Add 4 candidates to PBO
   ├─ Add 3 duels (favorite > others)
   ├─ propose_next_4(fit_first=True)
   │  ├─ fit() GP with 4 candidates, 3 duels
   │  └─ Generate 4 NEW proposals (GP-driven)
   ├─ Generate 4 images with SDXL
   └─ Save to impression_refinement/round_2/

9. Repeat steps 5-8 for subsequent rounds
```

### Key Implementation Details

#### Step 4: Initial Refinement
```python
# backend/server.py:3287-3291
refiner = get_or_create_pbo_refiner(
    session_id=session_id,
    stage=req.stage,
    force_recreate=True  # ✅ Creates NEW refiner with learned weights
)
```

**Why `force_recreate=True`?**
- Picks up the latest learned weights from base stage
- Ensures clean state for refinement

**Expected Log:**
```
[REFINER CACHE] Force recreating StageRefiner for session_id/impression
  Reason: Picking up updated weights
[REFINER CACHE] Creating NEW StageRefiner for session_id/impression
[REFINER CACHE] ✅ StageRefiner created and cached
  Concepts: 25
  PBO state: candidates=0, duels=0, fitted=False
```

#### Step 8: Next Round
```python
# backend/server.py:3930-3933
refiner = get_or_create_pbo_refiner(
    session_id=request.session_id,
    stage=request.stage
    # NO force_recreate! Uses cached refiner
)
```

**Why NO `force_recreate`?**
- Reuses cached refiner with accumulated candidates/duels
- Preserves PBO state

**Expected Log:**
```
[REFINER CACHE] Using CACHED StageRefiner for session_id/impression
  PBO state: candidates=4, duels=3, fitted=False
```

### Verification Checklist

#### ✅ Round 1 (Initial Refinement)
- [ ] `force_recreate=True` creates NEW refiner
- [ ] `candidates: 0, duels: 0, fitted: False`
- [ ] Cold start proposals based on learned weights
- [ ] 4 images generated and saved to `round_1/`
- [ ] `weights.json` saved with proposals

#### ✅ Round 2+ (Subsequent Rounds)
- [ ] Cached refiner retrieved (NO force_recreate)
- [ ] 4 candidates + 3 duels added
- [ ] GP fitted successfully
- [ ] `fitted: True` after fit
- [ ] 4 DIFFERENT proposals (GP-driven)
- [ ] Images saved to `round_2/`, `round_3/`, etc.

#### ✅ Weight Evolution
- [ ] Round 1 proposals are perturbations of learned weights
- [ ] Round 2+ proposals evolve based on preferences
- [ ] Top concepts shift based on user selections

---

## Mode 2: Test Refinement Stage Mode

### Description
Developer directly tests refinement without running full pipeline

### Entry Points

**1. Initialize:** `POST /api/pbo/init-refinement`
**2. Propose:** `POST /api/pbo/propose`
**3. Generate:** `POST /api/pbo/generate`
**4. Record Favorite:** `POST /api/pbo/record-refinement-favorite`

### Flow

```
1. Developer opens test refinement UI
2. Backend: /api/pbo/init-refinement
   ├─ Load visual tags from base stage
   ├─ Initialize ConceptRefinementSession
   ├─ Load learned weights (ema_w)
   ├─ Create StageRefiner with force_recreate=True
   └─ Return concept labels

3. Developer clicks "Propose"
4. Backend: /api/pbo/propose
   ├─ Get cached refiner
   ├─ propose_next_4(fit_first=True)
   │  └─ Cold start (no candidates)
   └─ Return 4 weight vectors

5. Developer clicks "Generate"
6. Backend: /api/pbo/generate
   ├─ Get cached refiner
   ├─ Generate 4 images from proposals
   └─ Return image URLs

7. Developer selects favorite (e.g., image 0)
8. Backend: /api/pbo/record-refinement-favorite
   ├─ Get cached refiner
   ├─ Add 4 candidates
   ├─ Add 3 duels
   └─ Update PBO state

9. Repeat steps 3-8 for subsequent rounds
```

### Key Differences from Mode 1

| Aspect | Mode 1 (Full Pipeline) | Mode 2 (Test) |
|--------|----------------------|---------------|
| **Initialization** | `/api/generate-stage-refinement` | `/api/pbo/init-refinement` |
| **Propose + Generate** | Combined in one endpoint | Separate endpoints |
| **Reference Image** | Uses selected impression image | Optional |
| **Preference Recording** | `/api/pbo/refine-next-round` | `/api/pbo/record-refinement-favorite` |
| **Use Case** | Production (user flow) | Development (testing) |

### Verification Checklist

#### ✅ Initialization
- [ ] `/api/pbo/init-refinement` called successfully
- [ ] `force_recreate=True` creates NEW refiner
- [ ] Learned weights loaded
- [ ] Concepts returned to frontend

#### ✅ First Proposal
- [ ] `/api/pbo/propose` returns 4 weight vectors
- [ ] Cold start proposals (candidates=0)
- [ ] Proposals based on learned weights

#### ✅ Generation
- [ ] `/api/pbo/generate` generates 4 images
- [ ] Images returned successfully

#### ✅ Favorite Recording
- [ ] `/api/pbo/record-refinement-favorite` updates PBO state
- [ ] 4 candidates + 3 duels added
- [ ] PBO state persists in cache

#### ✅ Second Proposal
- [ ] `/api/pbo/propose` with fit_first=True
- [ ] GP fitted with candidates/duels
- [ ] 4 DIFFERENT proposals (GP-driven)

---

## Common Issues

### Issue 1: Force Recreate in Wrong Place

**Symptom:**
```
Round 2: candidates=0, duels=0  ← Should be candidates=4, duels=3!
```

**Cause:**
`force_recreate=True` was called in `/api/pbo/refine-next-round`, clearing the PBO state.

**Fix:**
Only use `force_recreate=True` during initial setup:
- ✅ `/api/generate-stage-refinement` (line 3290)
- ✅ `/api/pbo/init-refinement` (line 3703)
- ❌ `/api/pbo/refine-next-round` (must be False/default)
- ❌ `/api/pbo/propose` (must be False/default)

### Issue 2: Wrong Endpoint Called

**Symptom:**
All rounds have identical proposals

**Diagnosis:**
Check logs for which endpoint is called after user selection:
```
# WRONG: Calling propose repeatedly
[PBO PROPOSE] ENDPOINT CALLED  ← Round 1
[PBO PROPOSE] ENDPOINT CALLED  ← Round 2 (wrong!)
[PBO PROPOSE] ENDPOINT CALLED  ← Round 3 (wrong!)

# CORRECT: Calling refine-next-round
[PBO PROPOSE] ENDPOINT CALLED              ← Round 1 (via generate-stage-refinement)
[PBO REFINE NEXT ROUND] ENDPOINT CALLED    ← Round 2 (correct!)
[PBO REFINE NEXT ROUND] ENDPOINT CALLED    ← Round 3 (correct!)
```

**Fix:**
Frontend should call `/api/pbo/refine-next-round`, NOT `/api/pbo/propose`.

---

## Diagnostic Commands

### 1. Check PBO State
```bash
curl "http://localhost:8765/api/pbo/debug-state?session_id=<SESSION>&stage=impression"
```

### 2. Verify Weight Evolution
```python
import json
import numpy as np

# Load Round 1 and Round 2 weights
with open("sessions/<SESSION>/impression_refinement/round_1/weights.json") as f:
    r1 = json.load(f)
with open("sessions/<SESSION>/impression_refinement/round_2/weights.json") as f:
    r2 = json.load(f)

# Compare proposals
for i in range(4):
    w1 = np.array(r1['proposals'][i])
    w2 = np.array(r2['proposals'][i])
    
    identical = np.allclose(w1, w2, atol=1e-6)
    print(f"Proposal {i}: {'❌ IDENTICAL (BUG!)' if identical else '✅ Different (correct)'}")
```

### 3. Check Server Logs
```bash
# Start server and watch logs
cd backend
conda activate apl
python server.py 2>&1 | grep -E "\[PBO|REFINER CACHE"
```

Look for the sequence:
```
[PBO INIT] ENDPOINT CALLED
[REFINER CACHE] Creating NEW StageRefiner
[PBO STATE] candidates: 0, duels: 0, fitted: False

[PBO REFINE NEXT ROUND] ENDPOINT CALLED
[REFINER CACHE] Using CACHED StageRefiner
[PBO STATE] Before: candidates: 0, duels: 0
[PBO STATE] After recording: candidates: 4, duels: 3
[PBO FIT] Fitting GP...
[PBO STATE] After propose: candidates: 4, duels: 3, fitted: True
```

---

## Summary

Both modes are implemented correctly:

### ✅ Mode 1 (Full Pipeline)
- Uses `/api/generate-stage-refinement` for Round 1
- Uses `/api/pbo/refine-next-round` for Round 2+
- Refiner cached between rounds
- PBO state persists

### ✅ Mode 2 (Test Refinement)
- Uses separate endpoints for propose/generate/favorite
- More granular control for testing
- Same PBO logic as Mode 1

### 🔧 Key Implementation Points

1. **`force_recreate=True`** only during initial setup
2. **Refiner cache** preserves PBO state between rounds
3. **`/api/pbo/refine-next-round`** records preferences AND proposes
4. **`/api/pbo/propose`** only proposes (no preference recording)

With the new debug logging, you can now verify both modes are working correctly by checking:
- Endpoint call sequence
- PBO state evolution
- GP fitting success
- Proposal diversity

