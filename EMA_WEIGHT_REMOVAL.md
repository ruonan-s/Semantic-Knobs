# EMA Weight Removal - Complete Simplification

## Summary

Removed all `ema_w` (EMA-smoothed weight) logic from the system and replaced it with direct use of actual weight `w` everywhere. The system now uses only the real-time computed weights without exponential smoothing.

## Motivation

The EMA smoothing (`ema_w`) was originally intended for:
1. **UI stabilization for PBO** - Providing a smoothed signal for Preferential Bayesian Optimization
2. **Visual smoothing** - Gradual transitions in bubble chart sizes

However, the user decided that:
- ✅ **Immediate feedback is better** - Real weights should be displayed instantly
- ✅ **Accuracy over smoothness** - UI should match backend logs exactly
- ✅ **Simpler system** - No need for dual weight tracking
- ✅ **No UI stabilization needed** - Direct weight changes are acceptable

## Changes Made

### 1. **backend/concept_refinement.py**

#### Removed GAMMA_EMA Parameter
```python
# DELETED:
# GAMMA_EMA = 0.8         # EMA smoothing for UI
```

#### Updated ConceptState Dataclass
```python
@dataclass
class ConceptState:
    """Learned state for a concept"""
    like_count: int = 0
    dislike_count: int = 0
    rank_bonus: float = 0.0
    rank_penalty: float = 0.0
    score: float = 0.0
    w: float = 0.0          # normalized weight
    # DELETED: ema_w: float = 0.0      # UI-smoothed weight
    liked_tags: set = None
    disliked_tags: set = None
```

#### Removed EMA Smoothing Logic in `compute_weights()`
```python
# OLD - Lines 385-391:
# # EMA smoothing for UI
# if state.ema_w == 0:  # First time
#     state.ema_w = new_w
# else:
#     state.ema_w = GAMMA_EMA * state.ema_w + (1 - GAMMA_EMA) * new_w

# NEW - Just set w directly:
state.w = new_w
```

#### Updated Initialization
```python
# OLD:
ConceptState(w=initial_weight, ema_w=initial_weight)

# NEW:
ConceptState(w=initial_weight)
```

#### Updated Logging
```python
# OLD:
delta_ema = after_state['ema_w'] - before_state['ema_w']
print(f"Δema_w={delta_ema:+.4f}")

# NEW:
delta_w = after_state['w'] - before_state['w']
print(f"Δw={delta_w:+.4f}")
```

#### Updated `get_current_weights_for_pbo()`
```python
# OLD:
weights[i] = state.ema_w

# NEW:
weights[i] = state.w
```

#### Updated State Serialization
```python
def _serialize_concept_state(self, state: ConceptState) -> Dict:
    return {
        'like_count': state.like_count,
        'dislike_count': state.dislike_count,
        'rank_bonus': state.rank_bonus,
        'rank_penalty': state.rank_penalty,
        'score': state.score,
        'w': state.w,
        # DELETED: 'ema_w': state.ema_w,
        'liked_tags': list(state.liked_tags) if state.liked_tags else [],
        'disliked_tags': list(state.disliked_tags) if state.disliked_tags else []
    }
```

#### Updated `save_concept_weights()`
```python
# OLD:
'weight': state.w,
'ema_weight': state.ema_w,
...
weights_data['concept_weights'].sort(key=lambda x: x['ema_weight'], reverse=True)
...
sample_weights = [f"{c['label']}: {c['ema_weight']:.3f}" ...]

# NEW:
'weight': state.w,
...
weights_data['concept_weights'].sort(key=lambda x: x['weight'], reverse=True)
...
sample_weights = [f"{c['label']}: {c['weight']:.3f}" ...]
```

#### Updated `load_concept_weights_from_base_stage()`
```python
# OLD:
state.w = prev_weight_data.get('weight', state.w)
state.ema_w = prev_weight_data.get('ema_weight', state.ema_w)
...
print(f"w={state.w:.3f}, ema_w={state.ema_w:.3f}")

# NEW:
state.w = prev_weight_data.get('weight', state.w)
...
print(f"w={state.w:.3f}")
```

#### Updated Before/After State Tracking
```python
# DELETED from both before_state and after_state:
'ema_w': state.ema_w,
```

---

### 2. **backend/test_e2e_pbo_simple.py**

#### Updated Mock Concept States
```python
# OLD:
concept_states[concept_id] = {
    'w': 1.0 / K,
    'ema_w': 1.0 / K,
    ...
}

# NEW:
concept_states[concept_id] = {
    'w': 1.0 / K,
    ...
}
```

#### Updated Weight Computation
```python
# OLD:
state['ema_w'] = (state['like_count'] + 0.1) / (total_likes + len(concepts) * 0.1)
total_w = sum(s['ema_w'] for s in concept_states.values())
state['ema_w'] /= total_w

# NEW:
state['w'] = (state['like_count'] + 0.1) / (total_likes + len(concepts) * 0.1)
total_w = sum(s['w'] for s in concept_states.values())
state['w'] /= total_w
```

#### Updated Weight Extraction for PBO
```python
# OLD:
w_ui = np.array([concept_states[c['id']]['ema_w'] for c in concepts])

# NEW:
w_ui = np.array([concept_states[c['id']]['w'] for c in concepts])
```

---

### 3. **frontend/src/components/BubbleChart.jsx** (Previously Fixed)

Already updated to use `state.w` instead of `state.ema_w` for bubble sizing and sorting.

```javascript
// Already using:
const sortedConcepts = [...concepts].sort((a, b) => (b.state.w || 0) - (a.state.w || 0));
const weight = concept.state.w || 0;
```

---

## Files Checked (No Changes Needed)

These files had no `ema_w` references:
- ✅ `backend/server.py`
- ✅ `backend/pbo.py`
- ✅ `backend/stage_refiner.py`
- ✅ `backend/test_stage_refiner.py`
- ✅ `backend/test_concept_system.py`
- ✅ `backend/debug_concepts.py`

---

## Impact

### Before (with EMA)

```python
# Click tag
like_count += 1
compute_weights()
  → w = 0.055 (instant)
  → ema_w = 0.8 × 0.033 + 0.2 × 0.055 = 0.037 (smoothed)

# UI displays ema_w
bubble_size ∝ 0.037

# Multiple updates needed to converge
Update 2: ema_w = 0.041
Update 3: ema_w = 0.044
...
Eventually: ema_w → 0.055
```

**Issues:**
- ❌ UI lags behind actual weights
- ❌ Same weights show different bubble sizes
- ❌ Confusing for debugging
- ❌ Extra complexity

### After (without EMA)

```python
# Click tag
like_count += 1
compute_weights()
  → w = 0.055 (instant)

# UI displays w
bubble_size ∝ 0.055

# Done! Immediate feedback
```

**Benefits:**
- ✅ UI matches backend logs exactly
- ✅ Same weights → identical bubble sizes
- ✅ Easy to debug and understand
- ✅ Simpler codebase
- ✅ Immediate visual feedback

---

## Testing

After these changes:

1. **Test tag interactions:**
   ```bash
   cd backend
   python test_concept_system.py
   ```

2. **Test PBO workflow:**
   ```bash
   python test_e2e_pbo_simple.py
   ```

3. **Test in UI:**
   - Click a tag
   - Check console: `Δw=+0.0220` (was `Δema_w`)
   - Verify bubble size changes immediately
   - Check backend logs match bubble sizes

4. **Test weight persistence:**
   - Save concept weights
   - Load in refinement stage
   - Verify only `weight` field is used (no `ema_weight`)

---

## Migration Notes

### Saved Weight Files

Old weight files may still contain `ema_weight` fields:
```json
{
  "concept_id": "concept_0",
  "label": "warm lighting",
  "weight": 0.055,
  "ema_weight": 0.048  // Will be ignored now
}
```

The `load_concept_weights_from_base_stage()` function now only loads `weight`:
```python
state.w = prev_weight_data.get('weight', state.w)
# No longer loads: prev_weight_data.get('ema_weight', ...)
```

This is **backward compatible** - old files with `ema_weight` will simply have that field ignored.

---

## Performance Impact

**No negative impact:**
- Removed computation: ~10-20 calculations per weight update
- Simpler state: -1 float per concept
- Cleaner code: -50 lines

**Positive impacts:**
- Faster weight updates (no EMA calculation)
- Less memory per concept
- Simpler debugging

---

## Conclusion

The system is now simplified to use only actual weights (`w`) throughout. All EMA smoothing logic has been removed, providing:

1. ✅ **Immediate accuracy** - UI shows exact computed weights
2. ✅ **Visual consistency** - Equal weights display identically
3. ✅ **Debugging clarity** - Logs match UI exactly
4. ✅ **Code simplicity** - Single weight value to track
5. ✅ **Performance** - Fewer calculations

No UI stabilization mechanism is needed - the system updates weights in real-time and displays them immediately.

