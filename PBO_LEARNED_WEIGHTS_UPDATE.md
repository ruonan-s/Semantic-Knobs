# PBO Learned Weights Integration

## Summary

Updated PBO first-round proposal generation to use **learned concept weights from exploration** instead of generic one-hot corners. This ensures the refinement stage starts with informed proposals that respect user preferences learned during exploration.

## Changes Made

### 1. `backend/pbo.py` - Cold Start Strategy

**Location**: Lines 463-521 in `propose_batch()` method

**Previous Behavior**:
- Generated 3 one-hot corner proposals: `[1,0,0,...]`, `[0,1,0,...]`, `[0,0,1,...]`
- Generated 1 uniform/smart center proposal
- Ignored learned preferences from exploration

**New Behavior**:
Generates 4 strategically perturbed proposals based on learned weights:

#### **Proposal 1: Learned Baseline**
- Uses learned weights directly from exploration (`ema_w`)
- Represents user's expressed preferences
- Example: `[0.25, 0.20, 0.15, 0.12, 0.08, ...]`

#### **Proposal 2: Top-Heavy**
- Amplifies top 3 concepts (×1.5), dampens others (×0.5)
- Tests if user wants stronger emphasis on favorites
- Example: `[0.36, 0.29, 0.21, 0.05, 0.03, ...]`

#### **Proposal 3: Diversified**
- Boosts mid-tier concepts rank 4-7 (×1.8)
- Reduces top 3 dominance (×0.7)
- Explores promising but less dominant concepts
- Example: `[0.17, 0.14, 0.10, 0.22, 0.18, 0.12, ...]`

#### **Proposal 4: Smoothed**
- Blends learned weights with uniform: `0.7 × learned + 0.3 × uniform`
- Tests if user wants more balanced mixture
- Example: `[0.20, 0.17, 0.14, 0.12, 0.09, ...]`

### 2. `backend/test_pbo_weight_updates.py` - Updated Tests

**Changes**:
- Added learned weight initialization in test setup
- Updated Round 1 expectations to check for distributed weights (not one-hot)
- Changed success criteria to verify proposals differ between rounds
- Added entropy analysis for weight distribution

## Benefits

### ✅ **Respects Learned Preferences**
All proposals use `ema_w` from exploration as the foundation, ensuring continuity.

### ✅ **Meaningful Variations**
Each proposal tests a different hypothesis about user preference intensity and distribution.

### ✅ **Token-Efficient for SDXL**
All candidates share similar top-K concepts (just different gains), staying within the 77-token limit.

### ✅ **Faster GP Convergence**
Round 1 provides informative signal to the Gaussian Process, enabling better Round 2+ proposals.

### ✅ **No Extreme Single-Concept Images**
Avoids degenerate one-hot prompts that produce narrow, single-concept images.

## Integration Points

### Warm Start Flow

1. **Exploration Stage** (`impression`, `spatial`, etc.)
   - User interacts with images
   - System learns concept weights (`ema_w`) via preference tracking

2. **Concept Refinement Initialization** (`backend/stage_refiner.py`)
   ```python
   concept_weights = np.array([
       concept_states.get(cid, {}).get('ema_w', 1.0 / K)
       for cid in concept_ids
   ])
   ```

3. **PBO Initialization** (`backend/pbo.py`)
   ```python
   pbo = PBO(
       MU=concept_centroids,
       concept_ids=concept_ids,
       concept_weights=concept_weights  # Pass learned weights
   )
   ```

4. **First Round Proposals**
   - Cold start generates 4 perturbations of `concept_weights`
   - Each proposal tests different emphasis patterns

5. **Subsequent Rounds**
   - PBO GP uses Round 1 feedback to optimize proposals
   - Acquisition strategies (exploit, diverse, thompson, EI)

## SDXL Prompt Generation

Each weight vector → SDXL prompt via `sdxl_integration.py`:

1. **Normalize weights** to simplex (sum = 1)
2. **Compute gains** via z-score mapping: `gain = 1.0 + 0.4 × z_score`, clipped to [0.7, 1.5]
3. **Select top-10** concepts as positive phrases with gains
4. **Select 3 negatives** from concepts below deficit threshold
5. **Fuse embeddings** using `SDXLEmbedFuser`

**Example for Proposal 1 (Learned Baseline)**:
- Top-10: `cozy (1.4)`, `warm (1.2)`, `comfortable (1.0)`, `soft (0.9)`, ...
- Negatives: `industrial`, `minimalist`, `stark`

**Example for Proposal 2 (Top-Heavy)**:
- Top-10: `cozy (1.5)`, `warm (1.4)`, `comfortable (1.3)`, `soft (0.8)`, ...
- Negatives: Similar, but lower-ranked concepts more suppressed

## Testing

Run the updated test:
```bash
cd backend
conda activate apl
python test_pbo_weight_updates.py
```

Expected output:
- ✅ Round 1: 3-4 distributed proposals (≥5 non-zero concepts each)
- ✅ Round 2: Proposals differ from Round 1 (GP active)
- ✅ GP fitted after user selection

## Backward Compatibility

- If `concept_weights=None` (cold start without learned weights):
  - Falls back to uniform weights: `np.ones(K) / K`
  - Perturbations still work, but start from uniform baseline

## Related Files

- `backend/pbo.py` - Core PBO logic
- `backend/stage_refiner.py` - Passes learned weights to PBO
- `backend/concept_refinement.py` - Tracks `ema_w` during exploration
- `backend/sdxl_integration.py` - Converts weights to SDXL prompts
- `backend/test_pbo_weight_updates.py` - Verification tests

---

**Date**: November 10, 2025
**Status**: ✅ Implemented and tested

