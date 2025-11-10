# PBO Debugging Guide

## Overview

This guide explains how to debug and trace PBO (Preferential Bayesian Optimization) behavior in the system. Comprehensive logging has been added to track the entire PBO lifecycle.

---

## Debug Logging

### What's Logged

All PBO endpoints now log detailed information:

#### 1. **`/api/pbo/init-refinement`** - Initialization
```
================================================================================
[PBO INIT] ENDPOINT CALLED
================================================================================
  Session: <session_id>
  Stage: <stage>
  Image IDs: [...]

[PBO INIT] ✅ Refinement initialized
  Concepts: 25
  Concept labels: coastal retreat location, inviting serene vibe, ...

[PBO STATE] After initialization:
  candidates: 0
  duels: 0
  fitted: False
  concept_weights sum: 1.000000
================================================================================
```

#### 2. **`/api/pbo/propose`** - Proposal Generation
```
================================================================================
[PBO PROPOSE] ENDPOINT CALLED
================================================================================
  Session: <session_id>
  Stage: <stage>
  Negatives: None
  w_current provided: False

[PBO STATE] Before propose:
  candidates: 0
  duels: 0
  fitted: False

[PBO STATE] After propose:
  candidates: 0
  duels: 0
  fitted: False
  Generated 4 proposals
================================================================================
```

#### 3. **`/api/pbo/refine-next-round`** - Next Round with Preference Recording
```
================================================================================
[PBO REFINE NEXT ROUND] ENDPOINT CALLED
================================================================================
  Session: <session_id>
  Stage: <stage>
  Round: 1 → 2
  Selected: impression_refinement_0_0
  All images: [...]

[PBO STATE] Before recording selection:
  candidates: 0
  duels: 0
  fitted: False

[PBO Refine] ✅ Recorded selection:
  Candidates added: 4
  Duels added: 3
  Favorite: round1_img0

[PBO STATE] After recording selection:
  candidates: 4
  duels: 3
  fitted: False

[PBO Refine] Proposing new mixtures with fit_first=True...

[PBO STATE] After propose:
  candidates: 4
  duels: 3
  fitted: True
  Proposed 4 new mixtures

[PBO Refine] ✅ Round 2 complete:
  Generated: 4 images
  Saved weights to: .../round_2/weights.json
================================================================================
```

#### 4. **Refiner Cache** - Tracking Refiner Lifecycle
```
[REFINER CACHE] Creating NEW StageRefiner for session_id/stage
[REFINER CACHE] ✅ StageRefiner created and cached
  Concepts: 25
  PBO state: candidates=0, duels=0, fitted=False

[REFINER CACHE] Using CACHED StageRefiner for session_id/stage
  PBO state: candidates=4, duels=3, fitted=True
```

---

## Diagnostic Endpoint

### **GET `/api/pbo/debug-state`**

Inspect PBO state at any time.

**Parameters:**
- `session_id`: Session ID
- `stage`: Stage name (e.g., "impression")

**Example:**
```bash
curl "http://localhost:8765/api/pbo/debug-state?session_id=[fast]_A_refreshing_space_by_the_sea_2025-11-07_02-33-39&stage=impression"
```

**Response:**
```json
{
  "session_id": "[fast]_A_refreshing_space_by_the_sea_2025-11-07_02-33-39",
  "stage": "impression",
  "pbo_state": {
    "num_candidates": 4,
    "num_duels": 3,
    "fitted": true,
    "K": 25,
    "d": 768
  },
  "concept_weights": {
    "sum": 1.0,
    "top_5": [
      ["coastal retreat location", 0.305],
      ["inviting serene vibe", 0.276],
      ["open airy ambiance", 0.172],
      ["natural wood elements", 0.066],
      ["plants enhance tranquility", 0.033]
    ]
  },
  "recent_candidates": [
    {
      "id": "round1_img0",
      "top_3_concepts": [
        ["coastal retreat location", 0.305],
        ["inviting serene vibe", 0.276],
        ["open airy ambiance", 0.172]
      ]
    },
    ...
  ],
  "recent_duels": [
    {"better": "round1_img0", "worse": "round1_img1", "strength": 1.0},
    {"better": "round1_img0", "worse": "round1_img2", "strength": 1.0},
    {"better": "round1_img0", "worse": "round1_img3", "strength": 1.0}
  ],
  "cache_key": "session_id:stage"
}
```

---

## Common Issues & Diagnosis

### Issue 1: **Proposals are Identical Across Rounds**

**Symptoms:**
- Round 1, 2, 3, ... all have identical weight vectors
- Images look very similar

**Diagnosis:**
Check the logs for:
```
[PBO STATE] After recording selection:
  candidates: 0  ← Should NOT be 0!
  duels: 0       ← Should NOT be 0!
```

**Root Cause:**
Preferences are not being recorded between rounds.

**Check:**
1. Is `/api/pbo/refine-next-round` being called?
2. Or is frontend calling `/api/pbo/propose` repeatedly (wrong!)?

**Fix:**
Frontend should call `/api/pbo/refine-next-round` after user selects an image, NOT `/api/pbo/propose`.

---

### Issue 2: **GP Never Fitted**

**Symptoms:**
- `fitted: False` in all rounds
- Cold start proposals in every round

**Diagnosis:**
Check logs for:
```
[PBO FIT] Fitting GP with X candidates, Y duels
```

If this message is missing, `fit()` is not being called or is failing.

**Root Cause:**
- Not enough candidates (need ≥2) or duels (need ≥1)
- GP fitting failed (sklearn/scipy issues)

**Check:**
```python
# In pbo.py::fit()
if len(self.candidates) < 2 or len(self.duels) == 0:
    print(f"[PBO] Not enough data to fit...")
    return
```

---

### Issue 3: **Refiner Cache Issues**

**Symptoms:**
- State resets unexpectedly
- Candidates/duels disappear

**Diagnosis:**
Check for:
```
[REFINER CACHE] Force recreating StageRefiner for session_id/stage
```

**Root Cause:**
`force_recreate=True` is being called unnecessarily, clearing the PBO state.

**Where force_recreate is used:**
- `/api/pbo/init-refinement` (line 3703) - ✅ Correct (initialization)
- Other endpoints should use `force_recreate=False` (default)

---

## Two Modes: Full Pipeline vs Test Refinement

### Mode 1: **Full Pipeline**

User runs through all stages:
1. Impression stage → generates images, user selects
2. Spatial stage → generates images, user selects
3. **Refinement** → PBO refinement on selected stage

**Key Files:**
- `frontend/src/components/PipelineController.tsx`
- `backend/server.py::run_stage_refinement()`

**Flow:**
```
1. User completes impression stage
2. User clicks "Refine Impression"
3. Backend: /api/generate-stage-refinement
   → Initializes PBO with learned weights from impression
   → Generates Round 1 images (cold start)
4. User selects image → clicks "Next Round"
5. Backend: /api/pbo/refine-next-round
   → Records preference
   → Fits GP
   → Generates Round 2 images (GP-driven)
```

---

### Mode 2: **Test Refinement Stage**

Developer directly tests refinement without running full pipeline:
1. Open test UI for refinement
2. Backend initializes from existing stage data

**Key Files:**
- `frontend/src/components/TestRefinementStage.tsx`
- `backend/server.py::pbo_init_refinement()`

**Flow:**
```
1. Developer opens test refinement UI
2. Backend: /api/pbo/init-refinement
   → Loads visual tags from base stage
   → Initializes PBO with learned weights
3. Developer: /api/pbo/propose + /api/pbo/generate
   → Generates Round 1 images (cold start)
4. Developer selects image
5. Backend: /api/pbo/record-refinement-favorite
   → Records preference
6. Developer: /api/pbo/propose + /api/pbo/generate
   → Generates Round 2 images (GP-driven)
```

---

## Expected PBO State Evolution

### Round 1 (Cold Start):
```
candidates: 0
duels: 0
fitted: False
→ Generates 4 proposals based on learned weights (perturbations)
```

### After User Selects Image 0:
```
candidates: 4  ← Added round1_img{0,1,2,3}
duels: 3       ← Added (img0 > img1), (img0 > img2), (img0 > img3)
fitted: False  ← Not yet fitted
```

### Round 2 (GP-Driven):
```
GP fitted: True  ← After fit()
→ Generates 4 DIFFERENT proposals:
  A: Anchor/Exploit (w_best from GP)
  B: Local Refinement (Dirichlet around w_best)
  C: Uncertainty-Diverse (high σ)
  D: Thompson Sampling (high upside)
```

### After User Selects Image 1:
```
candidates: 8  ← Added round2_img{0,1,2,3}
duels: 6       ← Added 3 more duels
fitted: False  ← Reset before next fit
```

### Round 3+ (GP-Driven):
```
GP refitted with more data
→ Better proposals based on accumulated preferences
```

---

## Verification Checklist

### ✅ Cold Start (Round 1)
- [ ] `candidates: 0, duels: 0, fitted: False`
- [ ] 4 proposals generated based on learned weights
- [ ] Top-3 concepts are consistent with exploration stage

### ✅ Preference Recording
- [ ] After selection: `candidates: 4, duels: 3`
- [ ] Favorite candidate recorded correctly
- [ ] Duels show `favorite > others`

### ✅ GP Fitting
- [ ] `[PBO FIT] Fitting GP with X candidates, Y duels` appears
- [ ] `fitted: True` after fit
- [ ] Log shows learned kernel parameters

### ✅ Round 2+ Proposals
- [ ] Proposals are DIFFERENT from Round 1
- [ ] 4 distinct strategies: Anchor, Local, Diverse, Thompson
- [ ] Diversity check shows varied cosine similarities

---

## Quick Test Script

```python
import requests

BASE_URL = "http://localhost:8765"
SESSION_ID = "[fast]_A_refreshing_space_by_the_sea_2025-11-07_02-33-39"
STAGE = "impression"

# 1. Check PBO state
response = requests.get(f"{BASE_URL}/api/pbo/debug-state", params={
    "session_id": SESSION_ID,
    "stage": STAGE
})
print(response.json())

# 2. Propose batch
response = requests.post(f"{BASE_URL}/api/pbo/propose", json={
    "session_id": SESSION_ID,
    "stage": STAGE,
    "negatives": None,
    "w_current": None
})
proposals = response.json()["proposals"]
print(f"Got {len(proposals)} proposals")

# 3. Check state again
response = requests.get(f"{BASE_URL}/api/pbo/debug-state", params={
    "session_id": SESSION_ID,
    "stage": STAGE
})
print(response.json()["pbo_state"])
```

---

## Summary

With the new debug logging, you can:
1. ✅ Trace which endpoints are being called
2. ✅ Monitor PBO state evolution (candidates, duels, fitted)
3. ✅ Verify GP is being fitted correctly
4. ✅ Inspect proposals and understand why they differ (or don't)
5. ✅ Diagnose cache issues and state resets

**Next Steps:**
- Run through both modes and check server logs
- Use `/api/pbo/debug-state` to inspect state at any time
- Compare logged PBO state with expected behavior

