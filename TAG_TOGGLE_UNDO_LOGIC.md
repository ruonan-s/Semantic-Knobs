# Tag Toggle and Undo Logic

## How It Works

### Backend Toggle Logic (concept_refinement.py:731-774)

The system properly handles **4 scenarios** when a user clicks a tag button:

#### 1. ✔️ ADD_LIKE (Neutral → Liked)
- User clicks 👍 on a neutral tag
- Backend: Adds to `liked_tags`, increments `like_count`
- Weight increases

#### 2. ✖️ UNDO_LIKE (Liked → Neutral)
- User clicks 👍 again on an already-liked tag (toggle off)
- Backend: Removes from `liked_tags`, decrements `like_count`
- Weight decreases **back toward neutral**

#### 3. ✔️ ADD_DISLIKE (Neutral → Disliked)
- User clicks 👎 on a neutral tag
- Backend: Adds to `disliked_tags`, increments `dislike_count`
- Weight decreases

#### 4. ✖️ UNDO_DISLIKE (Disliked → Neutral)
- User clicks 👎 again on an already-disliked tag (toggle off)
- Backend: Removes from `disliked_tags`, decrements `dislike_count`
- Weight increases **back toward neutral**

#### 5. 🔄 REVERSE_TO_LIKE (Disliked → Liked)
- User clicks 👍 on a currently-disliked tag
- Backend: 
  - Removes from `disliked_tags`, decrements `dislike_count`
  - Adds to `liked_tags`, increments `like_count`
- Weight increases (reverses the dislike effect)

#### 6. 🔄 REVERSE_TO_DISLIKE (Liked → Disliked)
- User clicks 👎 on a currently-liked tag
- Backend:
  - Removes from `liked_tags`, decrements `like_count`
  - Adds to `disliked_tags`, increments `dislike_count`
- Weight decreases (reverses the like effect)

## Key Implementation Details

### 1. Per-Tag Tracking

Each concept maintains sets of which specific tags were liked/disliked:

```python
class ConceptState:
    liked_tags: set[str]      # Set of tag IDs liked by user
    disliked_tags: set[str]   # Set of tag IDs disliked by user
    like_count: int           # Total number of likes
    dislike_count: int        # Total number of dislikes
```

This allows the backend to know if clicking 👍 is:
- Adding a new like (tag not in `liked_tags`)
- Toggling off an existing like (tag already in `liked_tags`)

### 2. Weight Recalculation from Scratch

After **every interaction**, weights are recalculated from scratch using current counts:

```python
# After updating like_count and dislike_count:
self.concept_states = compute_weights(
    self.concepts,
    self.concept_states
)
```

**This ensures proper undo behavior** because:
- No accumulation of deltas
- Weight is always computed from current `like_count` and `dislike_count`
- If you undo a like, `like_count` decreases, so weight decreases

### 3. Weight Computation Formula

```python
# Step 1: Compute score
score = a * like_count - b * dislike_count + rank_bonus - rank_penalty

# Step 2: Softmax normalization (ensures all weights sum to 1.0)
weights = softmax(scores / tau)

# Step 3: EMA smoothing for UI
ema_w = γ * old_ema_w + (1-γ) * new_w
```

Where:
- `a = 1.0` (like coefficient)
- `b = 1.5` (dislike coefficient, slightly stronger)
- `tau = 1.2` (softmax temperature)
- `γ = 0.7` (EMA smoothing factor)

## Important: EMA Smoothing Behavior

### What is EMA?

**Exponential Moving Average (EMA)** smooths weight transitions to prevent jarring UI changes.

### Example Timeline

Let's say a concept has initial weight 1.0% (w_base):

1. **Click 👍 (like):**
   - `like_count`: 0 → 1
   - Raw weight (`w`): 1.0% → 2.5%
   - EMA weight (`ema_w`): 1.0% → 0.7×1.0% + 0.3×2.5% = **1.45%**

2. **Click 👍 again (undo like):**
   - `like_count`: 1 → 0
   - Raw weight (`w`): 2.5% → 1.0%
   - EMA weight (`ema_w`): 1.45% → 0.7×1.45% + 0.3×1.0% = **1.32%**

3. **After more interactions:**
   - EMA gradually converges back to 1.0%

### Why EMA Doesn't Instantly Revert

**This is intentional, not a bug!** 

EMA provides:
- ✅ Smooth visual transitions (bubbles don't jump)
- ✅ Reduced sensitivity to rapid clicking
- ✅ More stable UI experience

If instant reversion is desired, set `GAMMA_EMA = 0` in `concept_refinement.py`.

## Testing the Undo Logic

### Test Case 1: Toggle Off
```
1. Start: Concept has 0 likes, 0 dislikes, weight ≈ 1/K
2. Click 👍: Concept has 1 like, 0 dislikes, weight increases
3. Click 👍 again: Concept has 0 likes, 0 dislikes, weight decreases
✅ Passes if weight trends back toward 1/K
```

### Test Case 2: Reverse Preference
```
1. Start: 0 likes, 0 dislikes
2. Click 👍: 1 like, 0 dislikes, weight increases
3. Click 👎: 0 likes, 1 dislike, weight decreases (below start)
✅ Passes if weight is now less than initial weight
```

### Test Case 3: Multiple Tags in Same Concept
```
1. Concept has 3 tags: tag_a, tag_b, tag_c
2. Click 👍 on tag_a: like_count = 1
3. Click 👍 on tag_b: like_count = 2
4. Click 👍 again on tag_a (undo): like_count = 1
5. tag_b is still liked, tag_a is neutral
✅ Passes if backend correctly tracks which specific tags are liked
```

## Enhanced Logging Output

When you interact with tags, you'll see detailed logs:

```bash
# Adding a like
  ✔️ Added like for tag tag_impression_0_1
[TAG CLICK] ADD_LIKE → concept_tropical_aesthetic: likes=0→1, dislikes=0→0, Δema_w=+0.0043

# Undoing a like
  ✖️ Toggled OFF like for tag tag_impression_0_1 (undo)
[TAG CLICK] UNDO_LIKE → concept_tropical_aesthetic: likes=1→0, dislikes=0→0, Δema_w=-0.0032

# Reversing from like to dislike
  🔄 Reversed from like to dislike for tag tag_impression_0_1
[TAG CLICK] REVERSE_TO_DISLIKE → concept_tropical_aesthetic: likes=1→0, dislikes=0→1, Δema_w=-0.0089
```

## Code References

- **Backend Logic:** `backend/concept_refinement.py:709-808` (`handle_tag_click()`)
- **Frontend Toggle:** `frontend/src/components/ConceptRefinementPanel.jsx:99-104`
- **Weight Computation:** `backend/concept_refinement.py:354-420` (`compute_weights()`)
- **EMA Constant:** `backend/concept_refinement.py:31` (`GAMMA_EMA = 0.7`)

## Configuration

To adjust behavior, modify these constants in `backend/concept_refinement.py`:

```python
# Line 26-31
A = 1.0                 # Like coefficient
B = 1.5                 # Dislike coefficient (stronger than likes)
TAU = 1.2               # Softmax temperature (higher = more uniform)
GAMMA_EMA = 0.7         # EMA smoothing (0=instant, 1=no change)
```

## Summary

✅ **Undo works correctly** - Clicking the same button toggles off  
✅ **Reverse works correctly** - Clicking opposite button reverses preference  
✅ **Weights recalculated from scratch** - No accumulation errors  
✅ **Per-tag tracking** - System knows which specific tags are liked/disliked  
⚠️ **EMA smoothing** - Changes are gradual, not instant (by design)  

The system properly "undoes" the calculation by decrementing counts and recalculating weights fresh each time.

