# SDXL Generation Suite

Standalone SDXL image generation with concept-based semantic control.

## Quick Start

**For semantic slider experiments (recommended):**
```bash
conda activate apl
cd SDXL
python cozy_sweep.py
```

**For low-level embedding experiments (educational):**
```bash
python directional_interpolation.py
```

## What's What

### 🎚️ `cozy_sweep.py` - **USE THIS FOR CHI STUDY**

Semantic slider in **concept weight space** (`w`).

**Why this one?**
- ✅ Works at right abstraction level (concept space, not embeddings)
- ✅ Integrates naturally with PBO
- ✅ Uses existing `SDXLRunner` pipeline (no embedding hacks)
- ✅ Interpretable and debuggable

**Use cases:**
- Within-context validation: "Does higher alpha → more cozy?"
- Cross-context transfer: "Does bedroom 'cozy' work in living rooms?"
- PBO integration: `w_alpha = w_neutral + alpha * (w_cozy - w_neutral)`

**Read:** [`README_COZY_SWEEP.md`](README_COZY_SWEEP.md)

---

### 🔬 `directional_interpolation.py` - Educational Only

Direct manipulation of SDXL embeddings.

**Why not for production?**
- ⚠️ Works at too-low level (embedding space)
- ⚠️ Bypasses phrase → embedding logic
- ⚠️ Duplicates existing functionality

**Use cases:**
- Understanding how SDXL uses embeddings
- Debugging CLIP space behavior
- Learning exercise

**Read:** [`README_DIRECTIONAL.md`](README_DIRECTIONAL.md)

---

### 🏗️ Core Pipeline

- **`sdxl_runner.py`** - High-level API: `generate_from_mixture(w, concepts, descriptor)`
- **`sdxl_integration.py`** - Concept weights → phrases with gains
- **`sdxl_embed_fuser.py`** - Phrases → SDXL embeddings (proper token structure)
- **`diffusion_runner.py`** - Low-level SDXL pipeline wrapper
- **`sdxl_config.py`** - Stage-specific generation parameters

**Read:** [`ARCHITECTURE.md`](ARCHITECTURE.md) for full details

---

## Documentation

| File | Purpose |
|------|---------|
| `README_COZY_SWEEP.md` | ⭐ Main guide for semantic sliders |
| `ARCHITECTURE.md` | Technical overview of the stack |
| `README_DIRECTIONAL.md` | Guide for embedding-level experiments |
| `CHANGES.md` | Changelog for directional interpolation refactor |

## Decision Tree

**"I want to do semantic slider experiments for my CHI paper"**
→ Use `cozy_sweep.py` ([guide](README_COZY_SWEEP.md))

**"I have PBO-learned weights and want to test them"**
→ Use `cozy_sweep.py` with `generate_cozy_sweep()` ([guide](README_COZY_SWEEP.md))

**"I want to test cross-context transfer"**
→ Use `cozy_sweep.py` with `generate_cross_context_sweep()` ([guide](README_COZY_SWEEP.md))

**"I want to understand how SDXL embeddings work"**
→ Read `directional_interpolation.py` code ([guide](README_DIRECTIONAL.md))

**"I need low-level control over CLIP embeddings"**
→ Use `directional_interpolation.py` ([guide](README_DIRECTIONAL.md))

**"I'm building new features on top of SDXL generation"**
→ Extend `SDXLRunner` or `SDXLEmbedFuser` ([architecture](ARCHITECTURE.md))

## Key Concepts

### Concept Weight Space (`w`)

A K-dimensional vector representing mixture weights over concepts:
```python
w = [0.20, 0.18, 0.15, 0.12, ...]  # K concepts, sum to 1.0
```

**Operations:**
- Interpolation: `w_alpha = w_neutral + alpha * (w_cozy - w_neutral)`
- Normalization: `w = w / w.sum()`
- Top-K selection: Keep only highest-weight concepts

### Semantic Direction

The difference between two weight vectors:
```python
w_neutral = [0.11, 0.11, 0.11, ...]  # Uniform
w_cozy = [0.20, 0.18, 0.15, ...]     # Learned

direction = w_cozy - w_neutral        # Direction in concept space
```

### Slider (Alpha)

Interpolation parameter:
```python
alpha = 0.0  # Neutral baseline
alpha = 0.5  # Halfway to learned
alpha = 1.0  # Full learned
alpha < 0    # Opposite direction
```

### Descriptor

Text that anchors the scene type:
```python
descriptor = "a cozy bedroom interior, wide angle, soft lighting, photo"
#            ^^^^^^^^^^^^^^^^^^^^^^ fixed scene type
#            w controls: warm, plants, textiles, wood, ... (variable attributes)
```

## Pipeline Flow

```
User Inputs:
  w_neutral, w_cozy, concepts, descriptor, alpha
    ↓
cozy_sweep.py:
  w_alpha = w_neutral + alpha * (w_cozy - w_neutral)
    ↓
SDXLRunner.generate_from_mixture(w_alpha, concepts, descriptor):
  ↓
  concepts_to_sdxl_phrases(w_alpha, concepts):
    → [(phrase, gain), ...], [neg_phrase, ...]
  ↓
  SDXLEmbedFuser.fuse_weighted_phrases(phrases):
    → prompt_embeds [1,77,2048], pooled [1,1280]
  ↓
  DiffusionRunner.generate_embeds(embeddings):
    → PIL Image
```

**Clean separation of concerns:**
1. **Semantic:** Concept weights (interpretable)
2. **Syntactic:** Phrases (human-readable)
3. **Numeric:** Embeddings (SDXL-native)

## Examples

### Example 1: Within-Context Validation

```python
from cozy_sweep import generate_cozy_sweep
from sdxl_runner import SDXLRunner
import numpy as np

# Setup
runner = SDXLRunner()
concepts = [
    {"id": "c0", "label": "warm, soft lighting"},
    {"id": "c1", "label": "many potted plants"},
    # ... more concepts
]

# Define baseline and learned weights
K = len(concepts)
w_neutral = np.ones(K) / K
w_cozy = np.array([0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.04])

# Generate sweep
sweep = generate_cozy_sweep(
    runner=runner,
    w_neutral=w_neutral,
    w_cozy=w_cozy,
    concepts=concepts,
    descriptor="a cozy bedroom interior, wide angle, soft lighting",
    alphas=[-0.2, 0.0, 0.2, 0.4, 0.6],
    output_dir="outputs/within_context"
)

# Show to user in random order
# Ask: "Rank from least to most cozy"
# Expected: Ranking correlates with alpha
```

### Example 2: Cross-Context Transfer

```python
from cozy_sweep import generate_cross_context_sweep

# Learn "cozy" in bedroom context (via PBO or manual)
w_cozy_bedroom = learned_cozy_weights

# Test transfer to other contexts
results = generate_cross_context_sweep(
    runner=runner,
    w_baseline=w_neutral,
    w_learned=w_cozy_bedroom,
    concepts=concepts,
    descriptors=[
        "a bedroom interior, wide angle, natural lighting",
        "a living room interior, wide angle, natural lighting",
        "a coffee shop interior, wide angle, natural lighting"
    ],
    alphas=[0.0, 0.5, 1.0],
    output_dir="outputs/cross_context"
)

# Question: Does bedroom "cozy" transfer to living room / cafe?
# Compare alpha=1 (learned) vs alpha=0 (baseline)
```

## Requirements

```bash
conda activate apl
# Should already have:
# - torch
# - diffusers
# - transformers
# - PIL (Pillow)
# - numpy
```

## Troubleshooting

**"Which file should I use?"**
→ Use `cozy_sweep.py` for user studies and PBO integration.

**"Images don't vary across alphas"**
→ Check that `w_neutral` and `w_cozy` are different. Increase alpha range.

**"Images drift away from scene type"**
→ Make descriptor more explicit. Use cozy_sweep.py (not directional_interpolation.py).

**"Cross-context transfer doesn't work"**
→ This is an empirical question! That's what you're testing. Compare to baseline.

**"SDXL pipeline fails to load"**
→ Check CUDA/PyTorch. Try `device="cpu"`. Update diffusers: `pip install -U diffusers`

## Citation

If you use this code for research, please cite your CHI paper once published! 😊

## Summary

- 🎚️ **For CHI study:** Use `cozy_sweep.py`
- 🔬 **For learning:** Read `directional_interpolation.py`
- 📚 **For details:** See `ARCHITECTURE.md`
- ✅ **Start here:** `README_COZY_SWEEP.md`

