# Auto-Categorization System Explained

## What Is Auto-Categorization?

Auto-categorization is an **intelligent system that automatically groups concepts into three categories** based on their weights and user interactions:

- **🟢 POSITIVE** - Concepts the user likes (high weights)
- **⚪ NEUTRAL** - Concepts with medium/baseline weights  
- **🔴 NEGATIVE** - Concepts the user dislikes (low weights)

This happens **automatically after every tag interaction** without requiring manual sorting.

## When Is It Used?

You see this log message when the system determines concept categories:

```bash
[GET_CATEGORIZED] Using auto-categorization
```

This happens:
1. **After tag clicks** - When you like/dislike tags
2. **After image selection** - When you select an image
3. **On initialization** - When concepts are first created
4. **In API responses** - To show category counts to frontend

## How Does It Work?

### Step 1: Calculate Statistical Thresholds

Instead of fixed thresholds, the system adapts to your current weight distribution:

```python
# Get weight statistics across all concepts
w_median = median of all concept weights
w_std = standard deviation of weights

# Adaptive thresholds (0.5 standard deviations from median)
positive_threshold = w_median + 0.5 × w_std
negative_threshold = w_median - 0.5 × w_std
```

**Why adaptive?** 
- Early on: All weights near baseline (1/K), thresholds are tight
- After interactions: Some concepts boosted/reduced, thresholds spread out
- Ensures meaningful categorization at any stage

### Step 2: Categorize Each Concept

For each concept, check its weight and interaction history:

```python
if concept has net dislikes (dislikes > likes):
    → NEGATIVE (user explicitly dislikes this)
    
elif weight >= positive_threshold:
    → POSITIVE (weight is significantly above median)
    
elif weight <= negative_threshold:
    → NEGATIVE (weight is significantly below median)
    
else:
    → NEUTRAL (weight is near median)
```

### Step 3: Sort Within Categories

Within each category, concepts are sorted by weight:
- **Positive:** High weight → Low weight (most liked first)
- **Neutral:** High weight → Low weight (best neutrals first)
- **Negative:** Low weight → High weight (worst first)

## Example Scenario

### Initial State (50 concepts, no interactions)
```
All concepts have weight ≈ 2.0% (1/50)
w_median = 0.020
w_std = 0.0001 (very small, all equal)

Thresholds:
  positive_threshold = 0.020 + 0.5×0.0001 = 0.02005
  negative_threshold = 0.020 - 0.5×0.0001 = 0.01995

Result: All 50 concepts → NEUTRAL ⚪
```

### After Liking 3 Tags
```
3 concepts boosted to ~3.5% weight
47 concepts remain at ~1.8% weight

w_median = 0.018
w_std = 0.008

Thresholds:
  positive_threshold = 0.018 + 0.5×0.008 = 0.022
  negative_threshold = 0.018 - 0.5×0.008 = 0.014

Result:
  🟢 POSITIVE: 3 concepts (weight > 0.022)
  ⚪ NEUTRAL: 44 concepts (0.014 < weight < 0.022)
  🔴 NEGATIVE: 3 concepts (weight < 0.014)
```

### After Many Interactions
```
User has liked 10 concepts, disliked 5, ignored rest

w_median = 0.015
w_std = 0.012

Thresholds:
  positive_threshold = 0.015 + 0.5×0.012 = 0.021
  negative_threshold = 0.015 - 0.5×0.012 = 0.009

Result:
  🟢 POSITIVE: 12 concepts (liked + some benefited from softmax)
  ⚪ NEUTRAL: 28 concepts (medium weights)
  🔴 NEGATIVE: 10 concepts (disliked + suppressed by softmax)
```

## Special Rules

### 1. Dislike Priority

If a concept has **more dislikes than likes**, it's **always negative**, regardless of weight:

```python
if dislike_count > like_count:
    category = NEGATIVE  # User explicitly doesn't want this
```

**Why?** User preference overrides statistical thresholds.

### 2. Minimum Gap

Thresholds must have at least 1.5% gap between them:

```python
if positive_threshold - negative_threshold < 0.03:
    positive_threshold = w_median + 0.015
    negative_threshold = w_median - 0.015
```

**Why?** Prevents all concepts from being categorized the same.

### 3. Score Consideration

The system also logs score statistics (used for debugging):

```python
score = a × likes - b × dislikes + rank_bonus - rank_penalty
```

Scores help validate that categorization makes sense.

## Auto vs. Explicit Categorization

The system supports **two modes**:

### Auto-Categorization (Default)
- **Used:** After tag clicks, image selection
- **How:** Automatically determined by weights/interactions
- **Log:** `[GET_CATEGORIZED] Using auto-categorization`

### Explicit Categorization
- **Used:** When user manually drags concepts into lists
- **How:** User's manual placement is stored
- **Log:** `[GET_CATEGORIZED] Using explicit categorization`

After any tag click, the system **clears explicit categorization** and returns to auto mode:

```python
# In handle_tag_click():
self._explicit_categorization = None  # Clear manual sorting
categorized = self.get_categorized_concepts()  # Use auto
```

**Why?** Weights have changed, so manual sorting may be outdated.

## Console Output Example

```bash
[CATEGORIZATION] K=45, w_mean=0.022222, w_median=0.020000, w_std=0.008500
  Score stats: s_mean=0.150, s_median=0.000
  Thresholds: positive >= 0.024250, negative <= 0.015750

  📊 Concepts with interactions:
    tropical aesthetic: likes=2, dislikes=0, score=2.000, w=0.035000
    minimalist design: likes=1, dislikes=0, score=1.000, w=0.028000
    cluttered layout: likes=0, dislikes=1, score=-1.500, w=0.012000

  tropical aesthetic: w=0.035000, score=2.000, likes=2, dislikes=0 -> POSITIVE
  minimalist design: w=0.028000, score=1.000, likes=1, dislikes=0 -> POSITIVE
  modern style: w=0.023000, score=0.000, likes=0, dislikes=0 -> NEUTRAL
  vintage look: w=0.019000, score=0.000, likes=0, dislikes=0 -> NEUTRAL
  cluttered layout: w=0.012000, score=-1.500, likes=0, dislikes=1 -> NEGATIVE
  ...
  
  Result: 8 positive, 30 neutral, 7 negative
```

## Where Categories Are Used

### 1. API Responses
Every concept API response includes categorized lists:

```json
{
  "success": true,
  "concepts": [...],
  "categorized": {
    "positive": ["concept_id_1", "concept_id_2", ...],
    "neutral": ["concept_id_20", ...],
    "negative": ["concept_id_45", ...]
  }
}
```

### 2. Frontend Display
- Could be used to show color-coded concept lists
- Could filter bubble chart by category
- Currently logged but not heavily used in UI

### 3. PBO Refinement
Negative concepts can be **excluded from prompt** during image generation:

```python
negative_ids = get_negative_concept_ids()
# Filter these out when building prompt
```

### 4. Debug Logging
Helps developers understand how weights translate to preferences:

```bash
[TAG CLICK] ADD_LIKE → concept_tropical: likes=0→1, dislikes=0→0, Δema_w=+0.0043
[CATEGORIZATION] ...
  tropical aesthetic: w=0.028000 -> POSITIVE
```

## Configuration

Adjust categorization behavior in `backend/concept_refinement.py`:

```python
# Line 489
threshold_factor = 0.5  # Standard deviations from median
                        # Higher = fewer concepts in positive/negative
                        # Lower = more concepts in positive/negative

# Line 495
min_gap = 0.015  # Minimum threshold separation (1.5%)
                 # Ensures meaningful categorization
```

## Benefits

1. **Automatic** - No manual sorting required
2. **Adaptive** - Adjusts to your interaction pattern
3. **Balanced** - Ensures reasonable distribution
4. **Preference-aware** - Respects explicit likes/dislikes
5. **Fast** - Simple numpy operations, no ML required

## Summary

**Auto-categorization** is the system's way of understanding your preferences by:
- Analyzing weight distribution
- Detecting outliers (liked/disliked concepts)
- Grouping concepts into meaningful categories
- Adapting thresholds as you interact more

It's like having an assistant who watches your tag clicks and says: "Based on your interactions, these concepts seem positive, these seem negative, and these are neutral."

The `[GET_CATEGORIZED] Using auto-categorization` message just means the system is using this intelligent automatic grouping rather than manual sorting you might have done earlier.

