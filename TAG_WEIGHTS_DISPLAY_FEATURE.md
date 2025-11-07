# Feature: Tag Weights Display for Refinement Stages

## Overview

Repurposed the "JSON Script" button during refinement stages to show **Tag Weights** instead. This displays the concept weights used to generate each refinement image, sorted from highest to lowest.

## Problem Solved

During refinement stages:
- The "JSON Script" button showed empty data (no JSON concepts available)
- Users couldn't see which concepts/tags were used to generate each refinement image
- No way to understand why images looked the way they did

## Solution

The "JSON Script" button now:
1. **Shows "Tag Weights"** label during refinement stages
2. **Displays concept weights** sorted from highest to lowest
3. **Only shows concepts with weight > 0** (filters out unused concepts)
4. **Formats cleanly** with percentages and readable layout

## Implementation

### Files Modified

1. **`/frontend/src/App.jsx`** (Lines 90-187)
   - Added `loadImageWeights()` function
   - Modified `loadImageJson()` to route to weights for refinement stages
   - Updated button labels to show "Tag Weights" vs "JSON Script"

2. **`/frontend/src/components/JsonPanel.jsx`** (Lines 3-126)
   - Enhanced to detect and display tag weights data
   - Added special formatting for weights display
   - Collapsible raw data section

### How It Works

#### 1. Button Label Changes Dynamically

```javascript
<button onClick={() => loadImageJson(image.id)}>
  {isRefinementStage ? 'Tag Weights' : 'JSON Script'}
</button>
```

#### 2. Loading Weights from Session Files

The `loadImageWeights()` function:

1. Parses image ID to get round number and image index
2. Loads `weights.json` from the appropriate round folder
3. Extracts the weights array for that specific image
4. Filters to only concepts with weight > 0
5. Sorts by weight (descending)
6. Formats as readable list with percentages

```javascript
// Example weights.json structure:
{
  "round": 1,
  "proposals": [
    [1.0, 0.0, 0.0, ...],  // Image 0 weights
    [0.0, 1.0, 0.0, ...],  // Image 1 weights
    [0.0, 0.0, 1.0, ...],  // Image 2 weights
    [0.052, 0.052, ...]    // Image 3 weights
  ],
  "concept_labels": [
    "relaxed living space ambiance",
    "coastal retreat location",
    ...
  ]
}
```

#### 3. Display Format

**Panel Header:**
```
Tag Weights - image_0

Round: 1
Concepts with weight > 0: 3
```

**Main Display (sorted highest to lowest):**
```
1. coastal retreat location: 45.23%
2. minimalist elegance: 32.10%
3. artwork enhances character: 22.67%
```

**Collapsible Section:**
```
▶ Show Raw Data
  (Click to expand and see full JSON)
```

### Image ID Parsing

Handles multiple formats:
- `image_0` → Round from current state, index 0
- `round_2_image_1` → Round 2, index 1

```javascript
const parts = imageId.split('_');
let roundNum = refinementRound; // Default
let imageIdx = 0;

if (parts[0] === 'round') {
  roundNum = parseInt(parts[1]);
  imageIdx = parseInt(parts[3]);
} else if (parts[0] === 'image') {
  imageIdx = parseInt(parts[1]);
}
```

## User Interface

### Before (Non-Refinement Stages)
```
[Visual Tags] [JSON Script]  ← Shows scene generation JSON
```

### After (Refinement Stages)
```
[Tag Weights]  ← Shows concept weights used for this image
```

### Panel Display

**Non-Refinement:**
```
┌─────────────────────────────┐
│ JSON Script - impression_0  │
├─────────────────────────────┤
│ {                          │
│   "scene": "...",          │
│   "prompt": "...",         │
│   ...                      │
│ }                          │
└─────────────────────────────┘
```

**Refinement:**
```
┌──────────────────────────────┐
│ Tag Weights - image_0        │
├──────────────────────────────┤
│ Round: 1                     │
│ Concepts with weight > 0: 5  │
├──────────────────────────────┤
│ 1. coastal retreat: 45.23%   │
│ 2. minimalist: 32.10%        │
│ 3. artwork: 22.67%           │
│ 4. rustic: 15.45%            │
│ 5. wood paneling: 8.33%      │
├──────────────────────────────┤
│ ▶ Show Raw Data              │
└──────────────────────────────┘
```

## Benefits

1. **Transparency**: Users can see exactly what concepts were used
2. **Understanding**: Explains why refinement images look the way they do
3. **Debugging**: Helps identify if PBO is learning correctly
4. **No Wasted UI**: Repurposes button that was showing empty data

## Testing

### How to Test

1. **Start a session** and generate initial concepts
2. **Enter refinement stage** (e.g., impression_refinement)
3. **Check button label**: Should say "Tag Weights" not "JSON Script"
4. **Click "Tag Weights"** on any image
5. **Verify display**:
   - Shows round number
   - Shows concept count
   - Lists concepts sorted highest to lowest
   - Only shows concepts with weight > 0
   - Percentages add up correctly

### Expected Behavior

**Round 1 (Cold Start)**
```
First 3 images: One-hot weights
- Image 0: 100% on concept_0, 0% on others
- Image 1: 100% on concept_1, 0% on others  
- Image 2: 100% on concept_2, 0% on others

Last image: Uniform weights
- Image 3: ~5.26% on all concepts (1/19)
```

**Later Rounds (After Learning)**
```
Mixed weights based on PBO:
- Image 0: 35% concept_A, 25% concept_B, 20% concept_C, ...
- Image 1: 40% concept_D, 30% concept_A, 15% concept_E, ...
- etc.
```

## Edge Cases Handled

### 1. All Zeros (Should Not Happen)
```
No concepts with weight > 0
```

### 2. Single Concept
```
1. coastal retreat location: 100.00%
```

### 3. Many Concepts (18+)
```
Scrollable list with all concepts > 0
```

### 4. File Not Found
```
Error message: "Failed to load tag weights: Failed to load weights"
Status message shown to user
```

## Data Flow

```
User clicks "Tag Weights"
    ↓
loadImageJson(imageId)
    ↓
isRefinementStage? → YES
    ↓
loadImageWeights(imageId)
    ↓
Parse image ID → roundNum, imageIdx
    ↓
Fetch /sessions/{sessionId}/{stage}/round_{roundNum}/weights.json
    ↓
Extract proposals[imageIdx] + concept_labels
    ↓
Filter weight > 0
    ↓
Sort descending by weight
    ↓
Format as text list
    ↓
setCurrentImageJson({ tag_weights, round, total_concepts, raw_data })
    ↓
setShowJsonPanel(true)
    ↓
JsonPanel detects tag_weights field
    ↓
Renders special tag weights layout
```

## Future Enhancements (Optional)

### 1. Visual Bar Chart
Instead of just numbers, show bars:
```
1. coastal retreat location: ████████████ 45.23%
2. minimalist elegance:      ████████ 32.10%
3. artwork enhances:         ██████ 22.67%
```

### 2. Color-Coded Weights
```
High (>30%):   Green text
Medium (10-30%): Yellow text
Low (<10%):    Gray text
```

### 3. Comparison View
Show weights for all 4 images side-by-side:
```
Concept              | Img 0  | Img 1  | Img 2  | Img 3
---------------------|--------|--------|--------|--------
coastal retreat      | 45.2%  | 12.3%  | 8.7%   | 5.2%
minimalist elegance  | 32.1%  | 5.4%   | 67.8%  | 5.2%
```

### 4. Export Functionality
Button to download weights as CSV or JSON

### 5. Highlight Changed Weights
Show which weights increased/decreased from previous round

## Related Files

- **Frontend**: 
  - `/frontend/src/App.jsx` - Main logic
  - `/frontend/src/components/JsonPanel.jsx` - Display component

- **Backend** (read-only):
  - `/sessions/{sessionId}/{stage}_refinement/round_{N}/weights.json` - Data source

## Troubleshooting

### Issue: "Failed to load weights"
**Causes:**
- weights.json doesn't exist for that round
- Session folder not found
- Network error

**Solution:**
- Check console for actual error
- Verify session folder exists
- Verify round folder exists

### Issue: Wrong weights shown
**Causes:**
- Image ID parsing incorrect
- Round number mismatch

**Solution:**
- Check console log: `[TAG WEIGHTS] Loaded for image:`
- Verify roundNum and imageIdx are correct

### Issue: Empty list
**Causes:**
- All weights are actually 0 (shouldn't happen in normal operation)
- Wrong proposal index

**Solution:**
- Check "Show Raw Data" to see actual weights array
- Verify imageIdx matches the image position

## Code Snippets

### Format Weight Percentage
```javascript
const formatted = (weight * 100).toFixed(2) + '%';
// Example: 0.4523 → "45.23%"
```

### Filter and Sort
```javascript
const weightsList = conceptLabels
  .map((label, idx) => ({ label, weight: imageWeights[idx] }))
  .filter(item => item.weight > 0)
  .sort((a, b) => b.weight - a.weight);
```

### Create Display Text
```javascript
const formattedWeights = weightsList
  .map((item, idx) => 
    `${idx + 1}. ${item.label}: ${(item.weight * 100).toFixed(2)}%`
  )
  .join('\n');
```

## Success Criteria

✅ Button label changes to "Tag Weights" in refinement stages  
✅ Clicking button loads and displays concept weights  
✅ Weights sorted from highest to lowest  
✅ Only shows concepts with weight > 0  
✅ Percentages formatted to 2 decimal places  
✅ Panel shows round number and concept count  
✅ Raw data available in collapsible section  
✅ No console errors  
✅ Works across all refinement rounds  

