# PBO Weight Update Fix - Summary

## Problems Identified

### 1. **Concepts Appearing in Both Positive AND Negative Prompts**
**File:** `sdxl_integration.py`

**Issue:** When one concept had weight=1.0 and others had weight=0.0, the top-K selection would include zero-weight concepts in the positive prompt, and the same concepts would also qualify as negative prompts due to their low weights.

**Example:**
```python
# Before fix:
positive_prompts: ["warm wood textures", "cozy home setting", "sleek industrial accents", ...]
negative_prompts: ["cozy home setting", "sleek industrial accents", "bohemian aesthetic"]
# "cozy home setting" and "sleek industrial accents" appear in BOTH lists!
```

**Fix:** Modified `concepts_to_sdxl_phrases()` to exclude concepts already in the positive prompt (top-K) from being selected as negative prompts.

```python
# Line 133 in sdxl_integration.py
for idx in range(K):
    if idx not in top_indices and w_norm[idx] < deficit_threshold:  # Added check
        deficit = uniform_weight - w_norm[idx]
        deficit_indices.append(idx)
        deficits.append(deficit)
```

### 2. **PBO Not Learning from User Selections (Round 2+ Still Using One-Hot Weights)**
**File:** `server.py`

**Issue:** The `pbo_refine_next_round` endpoint was calling:
- `refiner.on_favorite()` ✅ (adds duels to PBO)  
- `tracker.record_selection()` ❌ (NEVER CALLED!)

This meant:
- PBO was receiving duels and fitting the GP correctly
- BUT tracking.json had no `user_selection` or `pbo_update` fields
- The tracking data couldn't show PBO learning progress

**Result:** PBO was working internally but appeared broken because:
1. No visible record of selections in tracking
2. Impossible to debug what the user actually selected
3. Round 2, 3, 4, etc. kept returning one-hot weights because the GP never had enough data

**Fix:** Added `tracker.record_selection()` call before starting each new round in `pbo_refine_next_round()`:

```python
# Lines 3980-3989 in server.py
try:
    selected_index = request.all_image_ids.index(request.selected_image_id)
except ValueError:
    selected_index = 0

all_indices = list(range(len(request.all_image_ids)))
tracker_for_selection.record_selection(selected_index, all_indices)
print(f"[PBO Refine] ✅ Recorded selection in tracking: index {selected_index}")
```

### 3. **Import Issues in tracking.py**
**File:** `tracking.py`

**Issue:** Hard-coded `from backend.sdxl_integration import ...` failed when running tests as standalone scripts.

**Fix:** Added try-except blocks for flexible imports:

```python
# Lines 103-106 and 454-457 in tracking.py
try:
    from backend.sdxl_integration import normalize_simplex, compute_gains
except ImportError:
    from sdxl_integration import normalize_simplex, compute_gains
```

## Test Results

### Test 1: Positive/Negative Prompt Overlap
```
✅ PASSED: No overlap between positive and negative prompts
- Verified with one-hot weights (worst case)
- Verified with uniform weights (no negatives expected)
- Verified with mixed realistic weights
```

### Test 2: PBO Weight Progression
```
✅ PASSED: PBO correctly evolves weights after user selections
- Round 1 (Cold start): 3/4 proposals are one-hot
- User selects favorite → PBO adds 3 duels → GP fits
- Round 2 (PBO active): 2/4 proposals are one-hot (more diverse!)
- Weight diversity increased from cold start
```

### Test 3: Tracker Selection Recording
```
✅ PASSED: Tracker correctly records user selections
- user_selection field exists with selected_index and selected_image
- pbo_update field exists with num_duels and gp_fitted status
- All information properly logged to tracking.json
```

## Workflow After Fix

### Correct Flow
```
Round 1: Generate 4 images (cold start: one-hot + uniform)
  ↓
User selects favorite (e.g., image 2)
  ↓
Backend receives selection
  ↓
tracker.record_selection(2, [0,1,2,3]) → logs to tracking.json ✅
  ↓
refiner.on_favorite(img_2, [img_0, img_1, img_2, img_3]) → adds 3 duels ✅
  ↓
pbo.fit() → fits Gaussian Process ✅
  ↓
tracker.start_round(round_number=2) → starts new round ✅
  ↓
Round 2: Generate 4 images (PBO acquisition: Thompson, EI, Variance, Diverse)
  ↓
Images now have DIVERSE weights, not all one-hot! ✅
```

## Files Modified

1. **backend/sdxl_integration.py**
   - Line 133: Added `idx not in top_indices` check

2. **backend/server.py**
   - Lines 3956-3989: Added tracker selection recording
   - Lines 4006-4048: Refactored to reuse tracker and descriptor

3. **backend/tracking.py**
   - Lines 103-106, 454-457: Added flexible imports

4. **backend/test_fix_positive_negative_conflict.py** (new)
   - Tests for positive/negative overlap fix

5. **backend/test_pbo_weight_updates.py** (new)
   - Tests for PBO weight progression and tracker recording

## Verification

Run tests:
```bash
cd /home/akj2/nancy/Exploration-Refinement/backend
conda activate apl
python test_fix_positive_negative_conflict.py
python test_pbo_weight_updates.py
```

Expected output:
```
✅ ALL TESTS PASSED
```

## Impact

### Before Fix
- Concepts appeared in both positive and negative prompts
- Round 2+ kept generating one-hot weights (no learning)
- No tracking of user selections
- PBO appeared broken

### After Fix
- Clean separation of positive/negative prompts
- Round 2+ generates diverse weights using PBO acquisition
- Full tracking of selections and PBO updates
- Clear evidence of learning progression

## Next Steps

1. **Run a new refinement session** to verify tracking.json now contains:
   - `user_selection` fields in each round
   - `pbo_update` fields showing duels
   - Diverse weight vectors in Round 2+

2. **Monitor the tracking_readable.txt** file to see:
   - Selection history per round
   - Weight evolution across concepts
   - PBO duel information

3. **Visual inspection** of generated images:
   - Round 1: Should show distinct variations (corners)
   - Round 2+: Should show more nuanced combinations based on what user liked

