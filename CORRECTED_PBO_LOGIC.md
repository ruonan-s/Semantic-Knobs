# Corrected PBO Refinement Logic

## Overview

This document describes the **correct** implementation of Preferential Bayesian Optimization (PBO) for iterative refinement in the World Stylizer system.

## Key Principles

### 1. Fixed Reference Image
- **Reference image NEVER changes** during refinement
- Always use the originally selected image from the exploration stage (e.g., impression)
- All refinement rounds use the SAME reference image
- Only the fused embeddings (prompt_embeds) change between rounds

### 2. Weight Vectors = Images
- Each generated refinement image represents a specific **weight vector** over tag concepts
- When user selects an image, they are selecting a weight vector
- This selection provides a preference signal to PBO
- PBO learns which weight mixtures user prefers

### 3. Fused Embeddings
- Weights learned by PBO are multiplied with concept embeddings
- Formula: `fused_embedding = normalize(w @ MU)` where:
  - `w` = weight vector (learned by PBO)
  - `MU` = matrix of concept embeddings (from tag clustering)
- Each round generates different fused embeddings based on learned weights
- SDXL uses: reference_image + fused_embedding → new image

### 4. Iterative Learning
```
Round 1:
  Reference: impression_2_0.png (FIXED)
  PBO proposes: w1, w2, w3, w4 (initial proposals)
  SDXL generates: 4 images using reference + fused_embed(w1..4)
  
User selects: image_2 (which represents w3)
  
Round 2:
  Reference: STILL impression_2_0.png (UNCHANGED)
  PBO learns: w3 was preferred
  PBO proposes: w'1, w'2, w'3, w'4 (informed by previous selection)
  SDXL generates: 4 NEW images using SAME reference + fused_embed(w'1..4)
  
User selects: image_1 (which represents w'1)
  
Round 3:
  Reference: STILL impression_2_0.png (UNCHANGED)
  PBO learns: w'1 was preferred (in addition to w3 from round 1)
  ...and so on
```

## Implementation Details

### Backend Structure

#### File Organization
```
refinement_stage/
  round_1/
    image_0.png
    image_1.png
    image_2.png
    image_3.png
    weights.json  # Contains weight vectors for each image
  round_2/
    image_0.png
    ...
    weights.json
  ...
```

#### API Endpoints

**`/api/pbo/refine-next-round`** - Single unified endpoint
- **Input**: 
  - `session_id`: Session identifier
  - `stage`: Base stage (e.g., "impression")
  - `selected_image_id`: User's selected image
  - `all_image_ids`: All images from current round
  - `round_number`: Current round number
  
- **Process**:
  1. Record selection as preference (duels: selected vs others)
  2. PBO learns and proposes 4 new weight vectors
  3. Load ORIGINAL reference image from exploration stage
  4. Generate 4 images using SDXL img2img (reference + fused embeddings)
  5. Save images to `round_{N+1}` folder
  6. Save weight vectors to `weights.json`
  
- **Output**:
  - `image_paths`: Paths to 4 new images
  - `round_number`: New round number
  - `message`: Status message

#### Key Backend Functions

**`get_or_create_pbo_refiner(session_id, stage)`**
- Initializes `StageRefiner` with tag cluster concepts
- Creates MU matrix (concept embeddings)
- Sets up PBO with Gaussian Process

**`StageRefiner.on_favorite(favorite_id, all_ids)`**
- Records preference as duels (favorite vs others)
- Updates PBO model

**`StageRefiner.propose_next_4()`**
- Fits GP on recorded duels
- Proposes 4 new weight vectors using acquisition strategies
- Returns weight vectors as numpy arrays

**`StageRefiner.generate_images_from_proposals(proposals, sdxl_runner, init_image)`**
- Converts weight vectors to fused embeddings
- Calls SDXL img2img with reference image
- Returns PIL images

### Frontend Structure

#### Component: `RefinementIterationControls.jsx`

**Purpose**: Provide simple UI for iterative refinement

**UI Elements**:
1. **Continue to Next Stage** button
   - Proceeds to next stage (e.g., spatial) with current selection
   - Standard workflow progression
   
2. **Refine More** button
   - Automatically records selection and generates next round
   - Single-click operation (no separate "mark favorite" step)
   - Disabled until user selects an image

**State**:
- `round`: Current round number
- `isGenerating`: Loading state
- `status`: User feedback messages
- `error`: Error messages

**Callback**:
- `onRefinementComplete(newImages, round)`: Called when new round is generated
  - Parent updates image display
  - Resets selection
  - Updates round counter

### Data Flow

```
User Flow:
  1. Enter refinement stage → Round 1 generated (4 images)
  2. User selects an image
  3. User clicks "Refine More"
     → Backend records selection
     → PBO learns from preference
     → Proposes 4 new weight vectors
     → Generates 4 new images (using ORIGINAL reference)
     → Returns Round 2 images
  4. User selects from Round 2
  5. User clicks "Refine More" again
     → Round 3 generated
  6. ...continue until satisfied
  7. User clicks "Continue to Next Stage"
     → Proceeds to next stage with final selection
```

### Weight Vector Storage

**`weights.json` format**:
```json
{
  "round": 2,
  "proposals": [
    [0.2, 0.5, 0.1, 0.2],  // weight vector for image_0
    [0.3, 0.3, 0.2, 0.2],  // weight vector for image_1
    [0.1, 0.6, 0.2, 0.1],  // weight vector for image_2
    [0.4, 0.2, 0.3, 0.1]   // weight vector for image_3
  ],
  "concept_labels": ["urban", "natural", "abstract", "detailed"],
  "reference_image": "impression_2_0"
}
```

## Comparison: Old vs New Logic

### ❌ Old (Incorrect) Logic
- Each round used the selected image from previous round as reference
- Reference image changed every round
- No clear connection between weights and images
- Complex UI with separate "mark favorite" step

### ✅ New (Correct) Logic
- Reference image is FIXED (always original exploration selection)
- Only fused embeddings change (based on learned weights)
- Each image explicitly represents a weight vector
- Simple UI: select → "Refine More" (single action)

## Testing

### Manual Test Flow

1. **Start Session** → Generate impression stage images
2. **Select** impression_2_0.png
3. **Continue** → Generates Round 1 of impression_refinement (4 images)
4. **Verify**: All 4 images use impression_2_0.png as reference
5. **Select** refinement image (e.g., image_1)
6. **Refine More** → Generates Round 2 (4 NEW images)
7. **Verify**: 
   - Round 2 images STILL use impression_2_0.png as reference
   - Images look different (different fused embeddings)
   - `weights.json` saved in `round_2/` folder
8. **Select** from Round 2
9. **Refine More** → Round 3
10. **Verify**: Still using impression_2_0.png reference
11. **Continue to Next Stage** → Proceeds to spatial stage

### Verification Points

- [ ] Round 1 images saved to `round_1/` folder
- [ ] `weights.json` contains 4 weight vectors
- [ ] Reference image ID saved in `weights.json`
- [ ] All rounds use SAME reference image (check logs)
- [ ] Each round's images look visually different
- [ ] PBO learning improves convergence (user prefers later rounds)
- [ ] Can iterate 5+ rounds without errors
- [ ] "Continue to Next Stage" works after any round

## Benefits of Correct Implementation

1. **Stable Reference**: User's original choice guides all refinements
2. **Clear Semantics**: Each image = specific weight vector
3. **Effective Learning**: PBO learns in a consistent embedding space
4. **Convergence**: Multiple rounds lead to user's ideal design
5. **Simple UX**: One-click refinement, no manual preference recording
6. **Debugging**: Weight vectors saved for analysis
7. **Reproducibility**: Can regenerate any image from saved weights

## Technical Notes

### Why Fixed Reference?
- img2img uses reference as structural guide
- Fused embeddings control semantic content
- Changing reference would confuse PBO's learning (structural variations vs semantic)
- Fixed reference isolates semantic learning

### Why Store Weight Vectors?
- Debugging: Understand what each image represents
- Analysis: Identify preferred concept mixtures
- Reproducibility: Regenerate images
- Transfer: Apply learned weights to other stages

### PBO Acquisition Strategies
The system uses 4 strategies to propose diverse candidates:
1. **Thompson Sampling**: Sample from posterior
2. **Expected Improvement**: Maximize improvement probability
3. **Variance**: Explore uncertain regions
4. **Diverse**: Ensure proposal diversity

This ensures each round explores different promising regions.

## Summary

The corrected PBO refinement logic ensures:
- ✅ Fixed reference image (from exploration stage)
- ✅ Weight vectors explicitly linked to images
- ✅ Fused embeddings drive semantic variations
- ✅ Simple, intuitive UI
- ✅ Effective iterative learning
- ✅ Convergence to user preferences

This implementation enables true preferential Bayesian optimization for image refinement.


