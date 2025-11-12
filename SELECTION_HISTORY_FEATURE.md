# Selection History Feature

## Overview

Added a feature to display and reuse previous selections in the PBO refinement stage. Users can now see all their past selected images (including the reference) in a 3x3 grid and click on any of them to generate new variations based on those historical weights.

## What Was Added

### Frontend Changes (`frontend/src/components/RefinementIterationControls.jsx`)

1. **New State Variables**:
   - `selectionHistory`: Array of historical selections with their weights
   - `loadingHistory`: Boolean indicating history loading state

2. **History Loading Logic**:
   - `loadSelectionHistory()`: Fetches tracking data and weights for all previous rounds
   - Automatically reloads after each new round
   - Constructs full image URLs and loads weight vectors from `weights.json` files

3. **Historical Selection Handler**:
   - `handleHistoricalSelection()`: Allows users to select a previous image to refine from
   - Calls the new backend endpoint with the historical weights
   - Updates the UI with newly generated images

4. **UI Components**:
   - **Selection History Section**: Displays below the refinement controls
   - **3x3 Grid Layout**: Shows up to 9 most recent selections
   - **Reference Image**: Highlighted with a gold border (no weights, display-only)
   - **Round Selections**: Each shows "Round N" label, clickable to generate variants
   - **Hover Effects**: Visual feedback when hovering over clickable items
   - **Error Handling**: Gracefully handles missing images with placeholder text

### Backend Changes (`backend/server.py`)

1. **New Request/Response Models**:
   ```python
   class RefineFromWeightsRequest(BaseModel):
       session_id: str
       stage: str
       weights: list[float]  # Historical weight vector
       round_number: int

   class RefineFromWeightsResponse(BaseModel):
       success: bool
       image_paths: list[str]
       round_number: int
       message: str
   ```

2. **New Endpoint**: `/api/pbo/refine-from-weights`
   - Accepts historical weight vectors from previous rounds
   - Adds the historical weights to the PBO as a candidate
   - Generates 4 new local proposals around the historical weights using `local_around()`
   - Uses varied `alpha_scale` parameters (30, 40, 50, 60) for diversity
   - Creates new round with tracking
   - Generates images using the standard SDXL pipeline
   - Saves weights and images to new round folder

## How It Works

### Data Flow

1. **On Component Mount / Round Change**:
   ```
   UI -> Fetch tracking.json -> Parse rounds and selections
   UI -> Fetch weights.json for each round -> Extract selected image weights
   UI -> Display history grid with images and labels
   ```

2. **On Historical Selection**:
   ```
   User clicks historical image
   -> UI sends weights to /api/pbo/refine-from-weights
   -> Backend adds weights as PBO candidate
   -> Backend generates 4 local variations
   -> Backend creates new round with images
   -> UI updates with new images
   -> UI reloads history to include latest round
   ```

### Weight Preservation

- Each round's `weights.json` contains:
  - `proposals`: Array of weight vectors for the 4 generated images
  - `concept_labels`: Labels for interpretation
  - `reference_image`: Original exploration stage selection
  - `selected_concept_index`: Which of the 4 was selected (if any)

- When user selects a historical image:
  - Frontend loads the corresponding `weights.json`
  - Extracts the weight vector for the selected image
  - Sends it to the backend
  - Backend uses it as a seed for local exploration

### Local Exploration Strategy

The backend generates 4 proposals around historical weights using `local_around()`:
- Each proposal uses a Dirichlet distribution centered on the historical weights
- Different `alpha_scale` values (30, 40, 50, 60) provide varying degrees of exploration
- Lower alpha = more exploration, higher alpha = closer to original
- All proposals are projected to SDXL format (top-10 tags)

## UI Features

### Visual Design

- **Container**: Purple gradient background matching refinement controls
- **Grid**: 3 columns × 3 rows (responsive, equal aspect ratios)
- **Images**: 
  - Rounded corners
  - Border indicates type (gold for reference, white for rounds)
  - Hover effect (subtle scale/shadow)
  - Opacity reduced for disabled items (reference)
- **Labels**: 
  - "Reference" for initial exploration stage selection
  - "Round N" for PBO refinement round selections
  - Positioned at bottom of each image
  - Dark semi-transparent background for readability

### User Interaction

1. **Hover**: Image scales slightly, shows brighter border
2. **Click**: 
   - Reference image: No action (no weights available)
   - Round selections: Generates new round from those weights
3. **Loading**: Shows "Loading history..." while fetching data
4. **Empty State**: Shows "No selections yet" if no history available
5. **Status Updates**: Shows generation progress ("Using weights from Round N...")

## Technical Details

### History Data Structure

```javascript
{
  type: 'reference' | 'selection',
  round: number,  // 0 for reference
  imageId: string,  // e.g., "impression_2_0" or "round_3_image_2"
  imageUrl: string,  // Full URL path
  label: string,  // Display name
  weights: number[] | null  // Weight vector (null for reference)
}
```

### File Locations

```
sessions/
  [session_id]/
    impression_refinement/
      tracking.json              # Full history of rounds and selections
      round_1/
        weights.json             # Proposals for round 1
        image_0.png
        image_1.png
        image_2.png
        image_3.png
      round_2/
        weights.json             # Proposals for round 2
        ...
```

### API Contracts

**Load History** (GET):
- `tracking.json`: Contains reference image and per-round selection indices
- `weights.json`: Contains proposals array and selected indices

**Generate from History** (POST):
```json
{
  "session_id": "string",
  "stage": "impression",
  "weights": [0.1, 0.2, ...],  // Weight vector
  "round_number": 3
}
```

Response:
```json
{
  "success": true,
  "image_paths": ["/sessions/.../round_4/image_0.png", ...],
  "round_number": 4,
  "message": "Generated round 4 from historical weights"
}
```

## Benefits

1. **Non-Linear Exploration**: Users can revisit promising directions from earlier rounds
2. **Fault Tolerance**: If a later round diverges, user can "go back" to a better state
3. **Comparison**: Visual history helps users track their refinement journey
4. **Efficiency**: Reusing good weights avoids starting refinement from scratch
5. **Learning**: Users can understand which concepts/weights led to preferred images

## Future Enhancements

Potential improvements:
1. **Annotation**: Allow users to add notes to historical selections
2. **Favorites**: Star/bookmark specific rounds for quick access
3. **Comparison View**: Side-by-side comparison of multiple historical selections
4. **Weight Visualization**: Show concept weights as a bar chart for each selection
5. **Branching**: Create alternate refinement paths from any historical point
6. **Infinite Scroll**: Support >9 historical selections with pagination
7. **Search/Filter**: Find selections by round number, date, or concept weights

## Testing Recommendations

1. **Basic Flow**:
   - Complete a few PBO refinement rounds
   - Verify history grid appears with correct images
   - Click on a historical selection
   - Confirm new round generates successfully

2. **Edge Cases**:
   - Test with only reference image (no rounds yet)
   - Test with >9 selections (verify truncation to 9)
   - Test with missing image files (verify placeholder appears)
   - Test with missing weights.json (verify graceful handling)

3. **UI/UX**:
   - Verify hover effects work
   - Verify reference image is not clickable
   - Verify status messages appear during generation
   - Verify history updates after new round

## Status

✅ Implemented and ready for testing

**Files Modified**:
- `frontend/src/components/RefinementIterationControls.jsx`: Added history UI and logic
- `backend/server.py`: Added `/api/pbo/refine-from-weights` endpoint

**No Breaking Changes**: Fully backward compatible with existing refinement flow.

