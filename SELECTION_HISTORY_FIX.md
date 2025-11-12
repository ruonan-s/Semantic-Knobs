# Selection History Display Fix

## Problem

The selection history section was showing as empty (not displaying previous selected images).

## Root Cause

Three issues were found:

### 1. Wrong Tracking File Path
**Problem**: Frontend was looking for tracking data in the wrong location.
- **Incorrect**: `/sessions/${sessionId}/${stage}_refinement/tracking.json`
- **Correct**: `/sessions/${sessionId}/${stage}/tracking.json`

**Reason**: The tracking system saves tracking.json to the base stage folder (e.g., `impression/`), not the refinement folder (e.g., `impression_refinement/`). This is because the tracker is initialized with the base stage name.

### 2. Wrong Reference Image Structure
**Problem**: Code was looking for `trackingData.reference_image` at the top level.

**Actual Structure**: The `reference_image` is stored per round, not at the top level:
```json
{
  "rounds": [
    {
      "round_number": 1,
      "reference_image": "impression_0_0.png",  // ← stored here
      "proposals": [...],
      "user_selection": {...}
    }
  ]
}
```

**Fix**: Extract reference image from `trackingData.rounds[0].reference_image`.

### 3. Wrong Round Field Name
**Problem**: Code was accessing `roundData.round` instead of `roundData.round_number`.

**Correct Field**: The tracking structure uses `round_number`, not `round`.

## Changes Made

### `frontend/src/components/RefinementIterationControls.jsx`

1. **Fixed tracking path** (line 42):
   ```javascript
   // Before:
   const response = await fetch(`/sessions/${sessionId}/${stage}_refinement/tracking.json`);
   
   // After:
   const response = await fetch(`/sessions/${sessionId}/${stage}/tracking.json`);
   ```

2. **Fixed reference image parsing** (lines 53-69):
   ```javascript
   // Before:
   if (trackingData.reference_image) {
     // ... tried to access top-level field
   }
   
   // After:
   if (trackingData.rounds && trackingData.rounds.length > 0) {
     const firstRound = trackingData.rounds[0];
     if (firstRound.reference_image) {
       // reference_image is just the filename (e.g., "impression_0_0.png")
       const refImagePath = firstRound.reference_image.replace('.png', '');
       history.push({
         type: 'reference',
         round: 0,
         imageId: refImagePath,
         imageUrl: `/sessions/${sessionId}/${stage}/${firstRound.reference_image}`,
         label: 'Reference',
         weights: null
       });
     }
   }
   ```

3. **Fixed round field name** (line 77):
   ```javascript
   // Before:
   fetch(`/sessions/${sessionId}/${stage}_refinement/round_${roundData.round}/weights.json`)
   
   // After:
   fetch(`/sessions/${sessionId}/${stage}_refinement/round_${roundData.round_number}/weights.json`)
   ```

4. **Added comprehensive logging**:
   - Log when tracking data is loaded
   - Log when reference image is added
   - Log when weights are loaded for each round
   - Log final history count

5. **Fixed React hooks**:
   - Wrapped `loadSelectionHistory` in `useCallback` to prevent unnecessary re-renders
   - Added proper dependency arrays to `useCallback` and `useEffect`

## Testing

To verify the fix works:

1. **Start a refinement session**:
   - Complete exploration stages (impression, spatial, etc.)
   - Select an image to refine
   - Complete a few PBO refinement rounds

2. **Check browser console**:
   - Should see: `[History] Loaded tracking data: {...}`
   - Should see: `[History] Added reference: impression_0_0.png`
   - Should see: `[History] Loaded weights for round 1, image 2` (for each round)
   - Should see: `[History] Loaded N selections` (where N > 0)

3. **Check UI**:
   - Selection History section should show images in a 3×3 grid
   - First image should be labeled "Reference" with gold border
   - Subsequent images should be labeled "Round 1", "Round 2", etc.
   - Clicking a round selection should generate new images

## Debug Commands

If the history is still empty, check these in browser console:

```javascript
// Check if tracking file exists
fetch('/sessions/YOUR_SESSION_ID/impression/tracking.json')
  .then(r => r.json())
  .then(data => console.log('Tracking data:', data));

// Check rounds structure
fetch('/sessions/YOUR_SESSION_ID/impression/tracking.json')
  .then(r => r.json())
  .then(data => console.log('Rounds:', data.rounds));

// Check if weights files exist
fetch('/sessions/YOUR_SESSION_ID/impression_refinement/round_1/weights.json')
  .then(r => r.json())
  .then(data => console.log('Round 1 weights:', data));
```

## File Structure Reference

Correct file locations:
```
sessions/
  [session_id]/
    impression/
      tracking.json              ← Tracking data for impression stage
      impression_0_0.png         ← Reference image
    impression_refinement/
      round_1/
        weights.json             ← Weights for round 1 proposals
        image_0.png
        image_1.png
        image_2.png
        image_3.png
      round_2/
        weights.json             ← Weights for round 2 proposals
        ...
```

## Status

✅ Fixed and ready for testing

**Files Modified**:
- `frontend/src/components/RefinementIterationControls.jsx`: Fixed tracking path, reference image parsing, round field names, and React hooks

