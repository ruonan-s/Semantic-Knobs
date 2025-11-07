# PBO Weight Repetition Fix

## Problem

Round 3 was generating images identical to Round 2 because the PBO model was proposing the exact same weight vectors in every round, despite user selections being made.

### Root Cause Analysis

The issue was in how the `pbo_refine_next_round` endpoint handled user preferences:

1. **Weight Vector Reconstruction Failed**: The `on_favorite()` method in `stage_refiner.py` tried to reconstruct weight vectors from the `incidence_matrix`, which only contains data from the exploration stage
2. **Refinement Image IDs Not in Matrix**: Refinement round images (e.g., `"impression_refinement/round_2/image_0"`) were not in the incidence_matrix, so all reconstructed weights were zero
3. **Zero Weights = No Learning**: The PBO model was receiving candidates with all-zero weights, providing no useful signal for the Gaussian Process to learn from
4. **GP Cold Start**: With meaningless training data, the GP model stayed in "cold start" mode, always returning the same corner+center proposals (one-hot vectors)

### Evidence

From `tracking_readable.txt`:
- Round 2: Proposals were `[1,0,0,...]`, `[0,1,0,...]`, `[0,0,1,...]`, `[0.09,0.09,...]`
- Round 3: Proposals were **identical**: `[1,0,0,...]`, `[0,1,0,...]`, `[0,0,1,...]`, `[0.09,0.09,...]`

User selections were being recorded in tracking but not used by PBO.

## Solution

Modified `backend/server.py` in the `pbo_refine_next_round` endpoint to:

1. **Load Actual Weight Vectors**: Read the `weights.json` file from the current round to get the actual proposal weight vectors used to generate the images
2. **Add Candidates Directly**: Add these actual weight vectors as PBO candidates instead of reconstructing from incidence_matrix
3. **Create Duels**: Add strong duels between the selected favorite and other candidates using the actual weights

### Code Changes

**File**: `backend/server.py`

**Location**: Lines 3991-4025 (in `pbo_refine_next_round` endpoint)

**Key Changes**:
```python
# Step 2: Load the actual weight vectors from the current round
refinement_stage = f"{request.stage}_refinement"
refinement_folder = os.path.join(session_folder, refinement_stage)
current_round_folder = os.path.join(refinement_folder, f"round_{request.round_number}")
weights_file = os.path.join(current_round_folder, "weights.json")

with open(weights_file, 'r') as f:
    weights_data = json.load(f)

proposals_from_round = [np.array(w, dtype=np.float32) for w in weights_data['proposals']]

# Step 3: Record selection as PBO preference using ACTUAL weight vectors
candidate_ids = []
for i, (img_id, w) in enumerate(zip(request.all_image_ids, proposals_from_round)):
    cand_id = refiner.pbo.add_candidate(w, candidate_id=f"round{request.round_number}_img{i}")
    candidate_ids.append(cand_id)
    refiner.image_to_candidate[img_id] = cand_id

# Add strong duels: selected > others
favorite_index = request.all_image_ids.index(request.selected_image_id)
favorite_cand_id = candidate_ids[favorite_index]

duels_added = 0
for i, cand_id in enumerate(candidate_ids):
    if i != favorite_index:
        refiner.pbo.add_preference(favorite_cand_id, cand_id, strength=1.0)
        duels_added += 1
```

**Additional**: Added `import numpy as np` at the top of `server.py`

## Testing

Created `diagnose_pbo_fitting.py` to verify PBO behavior:
- Confirmed that with proper weight vectors, the GP fits correctly
- Confirmed that proposals change significantly after learning from duels
- Verified diversity metrics improve with real weight data

## Expected Behavior After Fix

With this fix:
1. ✅ Each round's actual proposal weights are preserved and used for PBO learning
2. ✅ User selections provide meaningful training signal to the GP model
3. ✅ The GP model fits successfully after each round
4. ✅ Subsequent rounds propose diverse, optimized weight vectors based on learned preferences
5. ✅ Image generation explores the concept space more effectively

## Notes

- The fix preserves the original `on_favorite()` method in `stage_refiner.py` for exploration stage use
- Refinement rounds now bypass incidence_matrix reconstruction and use direct weight vectors
- The PBO refiner state persists in-memory during the server session (but is lost on restart)
- Future improvement: Consider persisting PBO state to disk for robustness across server restarts

## Related Files

- `backend/server.py` - Modified endpoint
- `backend/stage_refiner.py` - Original `on_favorite()` method (unchanged)
- `backend/pbo.py` - PBO implementation (unchanged)
- `backend/tracking.py` - Selection tracking (works correctly)

