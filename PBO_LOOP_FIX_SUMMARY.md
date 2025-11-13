# PBO Loop Fix Summary

## Issues Fixed

### Issue 1: Loop Skipping (Round 1 → Round 3)
**Problem:** After completing Round 1, the system would jump directly to Round 3, skipping Round 2 entirely.

**Root Cause:** 
- The tracking system was saving data to the wrong `tracking.json` file
- Tracker was created with `stage=request.stage` (e.g., "impression") instead of `stage=refinement_stage` (e.g., "impression_refinement")
- This caused Round 2 data to be saved to `/impression/tracking.json` instead of `/impression_refinement/tracking.json`
- The frontend reads from `/impression_refinement/tracking.json`, which only had Round 1, so it thought the next round should be Round 2
- But the backend was already on Round 3 because it was tracking rounds in the wrong file

**Fix Applied:**
- **File:** `backend/server.py`, Line 3968-3975
- Changed tracker creation to use `refinement_stage` instead of `request.stage`:
  ```python
  # IMPORTANT: Use refinement stage for tracking, not base stage
  refinement_stage = f"{request.stage}_refinement"
  
  from backend.tracking import create_tracker
  tracker_for_selection = create_tracker(
      session_path=Path(session_folder),
      session_id=request.session_id,
      stage=refinement_stage,  # Use refinement stage, not base stage
      descriptor=descriptor or "No descriptor"
  )
  ```
- Also removed redundant `refinement_stage` definition at line 3991 (already defined at 3968)

### Issue 2: Tag Weights Limited to Top 10
**Problem:** Only the top 10 concepts by weight were being included in SDXL prompts, even when users had 32+ concepts.

**Root Cause:**
- `top_k` parameter defaulted to 10 in multiple places
- This was intentional to avoid token budget overflow, but user wants all concepts included until natural SDXL token limit (77 tokens)

**Fix Applied:**
- **File:** `backend/sdxl_integration.py`, Line 58
  - Changed default `top_k` from 10 to 9999 in `concepts_to_sdxl_phrases()`
  
- **File:** `backend/sdxl_runner.py`, Line 94
  - Changed default `top_k` from 10 to 9999 in `generate_from_mixture()`
  
- **File:** `backend/sdxl_embed_fuser.py`, Line 106
  - Changed default `max_positives` from 10 to 9999 in `fuse_weighted_phrases()`

**Result:** 
- All concepts will now be included in prompts (sorted by weight)
- SDXL's tokenizer will naturally truncate at 77 tokens
- Token budget warning is still displayed if concepts exceed limit

## Testing Recommendations

### Test 1: Round Tracking
1. Start a new session and complete impression stage
2. Click "Refine More" to generate Round 1
3. Select an image from Round 1 and click "Refine More"
4. Verify Round 2 is generated (not Round 3)
5. Check `/impression_refinement/tracking.json` contains both Round 1 and Round 2
6. Continue to Round 3 and verify proper numbering

### Test 2: All Concepts in Prompts  
1. Start a session with 32 concepts
2. Generate refinement images
3. Check backend logs for phrase count - should see all 32 concepts (or until token limit)
4. Previously would only see "Top 10" phrases

### Issue 3: Selection History Not Displaying
**Problem:** Previous round selections were not appearing in the history panel on the left side of the UI.

**Root Cause:**
- Frontend was requesting tracking data from wrong path: `/impression/tracking.json`
- Should request from: `/impression_refinement/tracking.json`
- The tracking data exists in the refinement folder, but the frontend was looking in the base stage folder

**Fix Applied:**
- **File:** `frontend/src/components/RefinementIterationControls.jsx`, Line 78
- Changed tracking URL from `/${stage}/tracking.json` to `/${stage}_refinement/tracking.json`:
  ```javascript
  // Old (incorrect):
  const trackingUrl = `/sessions/${sessionId}/${stage}/tracking.json`;
  
  // New (correct):
  const trackingUrl = `/sessions/${sessionId}/${stage}_refinement/tracking.json`;
  ```

### Issue 5: PBO Loop Starting from Round 2 Instead of Round 1
**Problem:** When clicking "Refine More" on initial refinement images, system treated them as Round 2 instead of Round 1, causing "round_2/weights.json not found" error.

**Root Cause:**
- Initial refinement images use legacy naming: `impression_refinement_0_0`, `impression_refinement_1_0`, etc.
- Frontend regex only matched new format: `round_X_image_Y`
- When regex failed to match legacy format, it fell back to incorrect state value

**Fix Applied:**
- **File:** `frontend/src/components/RefinementIterationControls.jsx`, Line 248-258
- Added detection for legacy format and explicitly treat as Round 1:
  ```javascript
  // Old (incorrect):
  const imageRoundMatch = firstImageId.match(/round_(\d+)_/);
  const actualImageRound = imageRoundMatch ? parseInt(imageRoundMatch[1]) : round;
  
  // New (correct):
  let actualImageRound;
  if (imageRoundMatch) {
    // New format: round_X_image_Y
    actualImageRound = parseInt(imageRoundMatch[1]);
  } else if (firstImageId.match(/^(impression|spatial|objects|ambient)_refinement_\d+_\d+$/)) {
    // Legacy format from initial refinement: treat as Round 1
    actualImageRound = 1;
  } else {
    // Fallback to state
    actualImageRound = round;
  }
  ```

**Result:**
- Initial refinement images now correctly identified as Round 1
- "Refine More" properly transitions from Round 1 → Round 2

### Issue 4: Uniform Tag Weights in Round 1 ⭐ CRITICAL
**Problem:** All 4 proposals in Round 1 had identical uniform weights (all concepts weighted equally at 1/K), completely ignoring the learned concept weights from user tag interactions.

**Root Cause:**
- When converting concept states for StageRefiner, `backend/server.py` used wrong dictionary key
- Set `concept_states[cid]['weight'] = state.w`  ❌
- But `StageRefiner.__init__` looks for `concept_states[cid]['ema_w']`  
- Key mismatch caused all concepts to fall back to default value of `1/K` (uniform)
- This made all Round 1 proposals meaningless (no diversity, no learning from user preferences)

**Fix Applied:**
- **File:** `backend/server.py`, Line 3535 AND `backend/stage_refiner.py`, Line 89
- Fixed key mismatch - both now use `'w'` to match `ConceptState`:
  ```python
  # server.py - Old (incorrect):
  concept_states[cid] = {
      'weight': state.w,  # Wrong key!
      ...
  }
  
  # server.py - New (correct):
  concept_states[cid] = {
      'w': state.w,  # Correct key matching ConceptState
      ...
  }
  
  # stage_refiner.py - Old (incorrect):
  concept_weights = np.array([
      concept_states.get(cid, {}).get('ema_w', 1.0 / self.K)  # Wrong key!
      ...
  ])
  
  # stage_refiner.py - New (correct):
  concept_weights = np.array([
      concept_states.get(cid, {}).get('w', 1.0 / self.K)  # Correct key!
      ...
  ])
  ```

**Result:**
- Round 1 proposals now use learned concept weights from tag interactions
- Proposals will have meaningful diversity based on what user liked/disliked
- PBO cold start proposals include: Learned Baseline, Top-Heavy, Diversified, and Smoothed variations

## Files Modified

1. `backend/server.py` - Fixed tracking stage reference (Issue #1) AND fixed concept weight key (Issue #4)
2. `backend/stage_refiner.py` - Fixed concept weight key lookup (Issue #4)
3. `backend/sdxl_integration.py` - Changed top_k default to 9999 (Issue #2)
4. `backend/sdxl_runner.py` - Changed top_k default to 9999 (Issue #2)
5. `backend/sdxl_embed_fuser.py` - Changed max_positives default to 9999 (Issue #2)
6. `frontend/src/components/RefinementIterationControls.jsx` - Fixed tracking.json path (Issue #3) AND fixed legacy round detection (Issue #5)

## Migration Notes

**For existing sessions with corrupted tracking:**
- Sessions created before this fix may have Round 2+ data in `/impression/tracking.json` instead of `/impression_refinement/tracking.json`
- The frontend will only show Round 1 history for these sessions
- New refinements from these sessions will work correctly going forward
- To fix old sessions: manually copy round data from `impression/tracking.json` to `impression_refinement/tracking.json`

