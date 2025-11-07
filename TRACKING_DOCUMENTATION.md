# SDXL Generation Tracking System

## Overview

The tracking system logs every detail of the SDXL generation pipeline to help debug, analyze, and understand the refinement process. A `tracking.json` file is created in each session folder containing complete information about concepts, weights, transformations, and user selections.

## What Gets Tracked

### Session Level
- Session ID, stage, descriptor
- All concepts with labels and metadata
- Timestamp of tracking start

### Round Level
- Round number
- Reference image used
- All proposals generated (4 weight vectors)
- User selection (which image was chosen)
- PBO updates (duels added)

### Proposal Level (Per Image)
- **Weight transformations:**
  - Raw weights
  - Normalized weights (simplex)
  - Z-scores
  - Gains (before and after clipping)
  - Statistical summary (mean, std, min, max)

- **Concept breakdown:**
  - Each concept's weight, gain, and rank
  - Whether included in positive/negative phrases
  - Complete transformation pipeline per concept

- **Prompt composition:**
  - Descriptor (if used) with gain
  - Each positive phrase with original and normalized gain
  - All negative phrases
  - Which phrases are concepts vs descriptor

- **Generation parameters:**
  - Strength, steps, guidance scale
  - Image dimensions
  - Mode (txt2img or img2img)
  - Seed used

## File Structure

### Two Files Generated

The tracking system automatically creates **two files** in each session folder:

1. **`tracking.json`** - Complete machine-readable data
   - Full nested JSON structure
   - All transformations and statistics
   - Programmatic access
   - ~40KB per round

2. **`tracking_readable.txt`** - Human-readable summary
   - Formatted plain text
   - Easy to scan and read
   - Concept evolution view
   - Round summaries
   - ~20KB per round

### Location
```
backend/sessions/[session_name]/
  ├── tracking.json          (machine-readable)
  └── tracking_readable.txt  (human-readable)
```

### Schema

```json
{
  "session_id": "session_123",
  "descriptor": "A comfortable space for reading",
  "stage": "impression",
  "created_at": "2025-10-29T15:46:36Z",
  
  "concepts": [
    {
      "id": "concept_0",
      "label": "warm lighting",
      "centroid_shape": [384],
      "source": "tag_cluster"
    }
  ],
  
  "rounds": [
    {
      "round_number": 1,
      "reference_image": "impression_2_0.png",
      "started_at": "2025-10-29T15:47:00Z",
      
      "proposals": [
        {
          "proposal_index": 0,
          "seed": 42,
          "generated_image": "impression_refinement/round_1/image_0.png",
          "generated_at": "2025-10-29T15:47:15Z",
          
          "weight_statistics": {
            "raw_weights": [0.8, 0.5, 0.3, 0.1, 0.0],
            "normalized_weights": [0.5, 0.3125, 0.1875, 0.0625, 0.0],
            "mean": 0.2,
            "std": 0.191,
            "min": 0.0,
            "max": 0.5
          },
          
          "concept_breakdown": [
            {
              "concept_id": "concept_0",
              "label": "warm lighting",
              "weight_raw": 0.8,
              "weight_normalized": 0.5,
              "z_score": 1.571,
              "gain_before_clip": 1.628,
              "gain_after_clip": 1.5,
              "rank": 1,
              "included_positive": true,
              "included_negative": false
            },
            {
              "concept_id": "concept_1",
              "label": "cozy atmosphere",
              "weight_raw": 0.5,
              "weight_normalized": 0.3125,
              "z_score": 0.589,
              "gain_before_clip": 1.236,
              "gain_after_clip": 1.236,
              "rank": 2,
              "included_positive": true,
              "included_negative": false
            }
          ],
          
          "prompt_composition": {
            "positive_phrases": [
              {
                "text": "A comfortable space for reading",
                "gain_original": 1.5,
                "gain_normalized": 0.289,
                "is_descriptor": true
              },
              {
                "text": "warm lighting",
                "gain_original": 1.5,
                "gain_normalized": 0.289,
                "is_descriptor": false
              },
              {
                "text": "cozy atmosphere",
                "gain_original": 1.236,
                "gain_normalized": 0.238,
                "is_descriptor": false
              }
            ],
            "negative_phrases": ["harsh lighting", "industrial"]
          },
          
          "generation_params": {
            "strength": 0.75,
            "steps": 30,
            "guidance_scale": 7.5,
            "height": 1024,
            "width": 1024,
            "top_k": 10,
            "num_negatives": 3,
            "mode": "img2img"
          }
        }
      ],
      
      "user_selection": {
        "selected_index": 2,
        "selected_image": "impression_refinement/round_1/image_2.png",
        "selection_timestamp": "2025-10-29T15:50:12Z"
      },
      
      "pbo_update": {
        "duels_added": [
          {
            "winner_index": 2,
            "loser_index": 0,
            "strength": 1.0,
            "type": "strong_duel"
          },
          {
            "winner_index": 2,
            "loser_index": 1,
            "strength": 1.0,
            "type": "strong_duel"
          },
          {
            "winner_index": 2,
            "loser_index": 3,
            "strength": 1.0,
            "type": "strong_duel"
          }
        ],
        "num_duels": 3,
        "gp_fitted": true
      }
    }
  ]
}
```

## Usage Examples

### Reading Tracking Data

```python
import json
from pathlib import Path

# Load tracking data
session_path = Path("backend/sessions/[fast]_A_comfortable_space_for_reading_2025-10-29_15-46-36")
tracking_file = session_path / "tracking.json"

with open(tracking_file, 'r') as f:
    data = json.load(f)

# Get descriptor
print(f"Descriptor: {data['descriptor']}")

# Get all rounds
for round_data in data['rounds']:
    print(f"\nRound {round_data['round_number']}")
    print(f"  Reference: {round_data['reference_image']}")
    print(f"  Proposals: {len(round_data['proposals'])}")
    
    # Get selected image
    if 'user_selection' in round_data:
        sel = round_data['user_selection']
        print(f"  Selected: Index {sel['selected_index']}")
```

### Analyzing Weight Evolution

```python
# Compare weights across rounds
for round_data in data['rounds']:
    print(f"\nRound {round_data['round_number']} weight distributions:")
    
    for proposal in round_data['proposals']:
        stats = proposal['weight_statistics']
        print(f"  Proposal {proposal['proposal_index']}:")
        print(f"    Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
        print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
```

### Finding Which Concepts Were Used

```python
# See which concepts contributed to a specific image
round_1 = data['rounds'][0]
proposal_0 = round_1['proposals'][0]

print("Concepts in positive phrases:")
for concept in proposal_0['concept_breakdown']:
    if concept['included_positive']:
        print(f"  {concept['label']}: weight={concept['weight_normalized']:.3f}, gain={concept['gain_after_clip']:.3f}")

print("\nConcepts in negative phrases:")
for concept in proposal_0['concept_breakdown']:
    if concept['included_negative']:
        print(f"  {concept['label']}")
```

### Debugging Prompt Composition

```python
# See exactly what went into the prompt
proposal = round_1['proposals'][0]
comp = proposal['prompt_composition']

print("Final prompt embedding composition:")
print("\nPositive phrases (with normalized gains):")
for phrase in comp['positive_phrases']:
    descriptor_marker = " [DESCRIPTOR]" if phrase['is_descriptor'] else ""
    print(f"  {phrase['text']}: {phrase['gain_normalized']:.3f}{descriptor_marker}")

print(f"\nNegative phrases:")
for phrase in comp['negative_phrases']:
    print(f"  {phrase}")
```

## Integration Flow

### 1. Session Start
```python
from backend.tracking import create_tracker

tracker = create_tracker(
    session_path=Path(session_folder),
    session_id=session_id,
    stage="impression",
    descriptor="A comfortable space for reading"
)
tracker.set_concepts(concepts)
```

### 2. Round Start
```python
tracker.start_round(
    round_number=1,
    reference_image="impression_2_0.png"
)
```

### 3. Generation (Automatic)
When `sdxl_runner.generate_from_mixture()` is called with `tracker`, it automatically records all proposal details.

```python
images = refiner.generate_images_from_proposals(
    proposals=proposals,
    sdxl_runner=sdxl_runner,
    tracker=tracker,  # Enables tracking
    generated_image_paths=image_paths,
    descriptor=descriptor,
    ...
)
```

### 4. User Selection
```python
tracker.record_selection(
    selected_index=2,
    all_indices=[0, 1, 2, 3]
)
```

## Debugging Use Cases

### 1. "Why did this image look different?"
Check the `concept_breakdown` to see exact weights and gains for each concept.

### 2. "Which concepts are dominating?"
Look at `weight_statistics` and `rank` in `concept_breakdown`.

### 3. "Is the descriptor being used?"
Check `prompt_composition` → `positive_phrases` → find where `is_descriptor: true`.

### 4. "How are weights changing over rounds?"
Compare `normalized_weights` across rounds for the same concepts.

### 5. "Why is PBO not learning?"
Check `pbo_update` → `duels_added` to see if selections are being recorded.

### 6. "What's the actual prompt sent to SDXL?"
See `prompt_composition` with `gain_normalized` values (these are the actual embedding weights).

## Best Practices

1. **Always enable tracking in production** - Minimal overhead, invaluable for debugging
2. **Archive tracking files** - Keep them with generated images for future reference
3. **Compare successful vs unsuccessful sessions** - Look for patterns in weight distributions
4. **Check concept coverage** - Ensure diverse concepts are being selected
5. **Monitor gain normalization** - Verify descriptor and concepts get appropriate weights

## Using the Human-Readable Format

### Quick Review

Simply open `tracking_readable.txt` in any text editor:

```bash
# View the readable summary
cat backend/sessions/[session]/tracking_readable.txt

# Or open in editor
code backend/sessions/[session]/tracking_readable.txt
```

### Finding Information Quickly

**To see concept evolution:**
- Scroll to "CONCEPT WEIGHT EVOLUTION" section
- Each concept shows all rounds and proposals
- See which proposal was selected for each round

**To see what went into each image:**
- Scroll to "ROUND SUMMARIES" section
- Each image shows:
  - Prompt composition (descriptor + concepts)
  - Top concepts by weight
  - Which concepts were used/not used
- Selected image marked with ★

### Example Workflow

1. **Generate images** → Both files auto-created
2. **Quick check** → Open `tracking_readable.txt`
3. **See evolution** → Check concept weights across proposals
4. **Verify selection** → Confirm which image was chosen
5. **Deep dive** → If needed, analyze `tracking.json` with scripts

### Sharing with Team

The readable format is perfect for:
- Email updates ("Here's what the model learned...")
- Documentation ("Image 2 focused on X and Y...")
- Bug reports ("Concept A had weight 0.8 but wasn't included...")
- Design reviews ("The selected image emphasized...")

## Performance Impact

- **Memory:** ~10KB per proposal, ~40KB per round (JSON), ~20KB per round (readable)
- **CPU:** Negligible (simple arithmetic + text formatting)
- **I/O:** Two file writes per save (~2ms total)
- **Total overhead:** <1% of generation time

## Future Enhancements

Potential additions to tracking:
- CLIP similarity scores between proposals
- Embedding space visualization data
- User interaction timing
- Concept co-occurrence patterns
- Weight convergence metrics

