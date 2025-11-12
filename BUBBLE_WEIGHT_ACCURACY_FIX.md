# Bubble Chart Weight Accuracy Fix

## Problem

The bubble chart was displaying different sizes for concepts that had identical weights according to the backend logs.

### Example from Logs

Backend showed three concepts with **identical weights**:
```
clean lines: likes=1, dislikes=0, score=1.000, w=0.055056
natural light: likes=1, dislikes=0, score=1.000, w=0.055056
white color scheme: likes=1, dislikes=0, score=1.000, w=0.055056
```

But the bubble chart displayed these three concepts with **different bubble sizes**.

## Root Cause

The BubbleChart component was using **`ema_w`** (EMA-smoothed weight) instead of **`w`** (actual weight):

```javascript
// OLD: Using smoothed weight
const sortedConcepts = [...concepts].sort((a, b) => 
  (b.state.ema_w || 0) - (a.state.ema_w || 0)
);

const maxWeight = Math.max(...allConcepts.map(c => c.state.ema_w || 0));
const weight = concept.state.ema_w || 0;
```

### What is EMA Weight?

`ema_w` is an **Exponential Moving Average** smoothed version of the weight, calculated as:

```python
# Backend: concept_refinement.py
if state.ema_w == 0:  # First time
    state.ema_w = new_w
else:
    state.ema_w = GAMMA_EMA * state.ema_w + (1 - GAMMA_EMA) * new_w
    # where GAMMA_EMA = 0.85
```

This means:
- `w` updates **instantly** when you click a tag
- `ema_w` **gradually transitions** to match `w` over multiple updates

### Why Different Sizes?

Even when three concepts have the same `w = 0.055056`, their `ema_w` values differ:

```
Concept A: w=0.055056, ema_w=0.048234 (still catching up)
Concept B: w=0.055056, ema_w=0.052891 (almost there)
Concept C: w=0.055056, ema_w=0.055056 (fully updated)
```

This happens because:
1. All start with `ema_w = 0`
2. Each update: `ema_w = 0.85 × old_ema_w + 0.15 × new_w`
3. Takes multiple updates to converge (15% per update)
4. Concepts updated at different times have different `ema_w` values

## Solution

Changed BubbleChart to use **actual weight `w`** instead of smoothed `ema_w`:

```javascript
// NEW: Using actual weight
const sortedConcepts = [...concepts].sort((a, b) => 
  (b.state.w || 0) - (a.state.w || 0)  // ✓ Use actual weight
);

const maxWeight = Math.max(...allConcepts.map(c => c.state.w || 0));  // ✓
const minWeight = Math.min(...allConcepts.map(c => c.state.w || 0));  // ✓
const weight = concept.state.w || 0;  // ✓
```

## Changes Made

### File: `frontend/src/components/BubbleChart.jsx`

**Line 44** - Sorting by actual weight:
```javascript
// OLD: (b.state.ema_w || 0) - (a.state.ema_w || 0)
// NEW: (b.state.w || 0) - (a.state.w || 0)
```

**Lines 50-52** - Size calculation based on actual weight:
```javascript
// OLD: Calculate bubble sizes based on ema_w
// NEW: Calculate bubble sizes based on actual weight (w, not ema_w)

const maxWeight = Math.max(...allConcepts.map(c => c.state.w || 0));
const minWeight = Math.min(...allConcepts.map(c => c.state.w || 0));
```

**Line 57** - Bubble size uses actual weight:
```javascript
// OLD: const weight = concept.state.ema_w || 0;
// NEW: const weight = concept.state.w || 0;  // Use actual weight, not smoothed
```

## Result

Now the bubble chart **accurately reflects** the backend weights:

| Concept | Backend `w` | BubbleChart Size |
|---------|-------------|------------------|
| clean lines | 0.055056 | **Equal** ✓ |
| natural light | 0.055056 | **Equal** ✓ |
| white color scheme | 0.055056 | **Equal** ✓ |

All three concepts will display with **identical bubble sizes** because they have identical weights.

## Why We Had EMA Smoothing

The original intent of `ema_w` was to provide **smooth visual transitions** in the UI:
- When weights change, bubbles grow/shrink gradually
- Prevents jarring jumps in bubble sizes
- Creates a more pleasant animation effect

However, this came at the cost of **accuracy** - the displayed sizes didn't match the actual weights.

## Design Decision

We chose **accuracy over smoothness** because:

1. ✅ **User trust** - Bubble sizes should match logged weights exactly
2. ✅ **Debugging** - Easier to understand system behavior
3. ✅ **Immediate feedback** - Changes visible instantly
4. ⚠️ **Trade-off** - Bubbles may appear to "jump" in size

If smooth transitions are desired, they should be implemented in **CSS/SVG animations** (visual only) rather than in the data layer.

## Testing

After this fix:
1. Click a tag → Check backend logs for `w` values
2. Verify bubble sizes in chart match the weight proportions
3. Multiple tags with same `w` → Should have identical bubble sizes

Expected behavior:
```
Backend: concept_A w=0.055, concept_B w=0.055, concept_C w=0.055
Chart:   All three bubbles have IDENTICAL sizes ✓
```

## Related Backend Code

Weight calculation in `backend/concept_refinement.py`:

```python
# Lines 388-393: EMA smoothing (now only for backend use)
if state.ema_w == 0:  # First time
    state.ema_w = new_w
else:
    state.ema_w = GAMMA_EMA * state.ema_w + (1 - GAMMA_EMA) * new_w

state.w = new_w  # This is what BubbleChart now uses
```

Both `w` and `ema_w` are sent to frontend, but now we use `w` for visualization.

## Related Files

- `frontend/src/components/BubbleChart.jsx` - Fixed to use `w` instead of `ema_w`
- `backend/concept_refinement.py` - Computes both `w` and `ema_w`

## Future Enhancement (Optional)

If smooth visual transitions are desired without sacrificing accuracy:

```javascript
// CSS/SVG animation approach
<circle
  r={radius}  // Actual weight-based radius
  style={{
    transition: 'r 0.3s ease-out'  // Smooth visual transition
  }}
/>
```

This provides:
- ✅ Accurate final sizes (matches backend)
- ✅ Smooth visual transitions (CSS animation)
- ✅ Best of both worlds

