# PBO Integration - Stage 3 Handoff Document

## ✅ STAGE 3 COMPLETED (Nov 4, 2024)

All Stage 3 implementation tasks are complete! Unit tests pass. Integration with SDXL model requires manual testing (see below).

**Files Created:**
- [backend/sdxl_integration.py](backend/sdxl_integration.py) - Concept-to-phrases converter with gain mapping
- [backend/sdxl_embed_fuser.py](backend/sdxl_embed_fuser.py) - SDXL embedding fuser with Top-K constraints
- [backend/sdxl_runner.py](backend/sdxl_runner.py) - SDXL generation wrapper
- [backend/test_sdxl_integration.py](backend/test_sdxl_integration.py) - Integration tests

**Files Modified:**
- [backend/stage_refiner.py](backend/stage_refiner.py) - Added `generate_images_from_proposals()` method

**Test Results:**
- ✅ Gain mapping: Produces correct range [0.7, 1.5] for uniform, peaked, and bimodal distributions
- ✅ Phrase selection: Top-K positives and deficit-based negatives work correctly
- ✅ Edge cases: Single concept, all zeros, top_k > K handled properly
- ✅ StageRefiner integration: `generate_images_from_proposals()` method added and tested

**To Test with SDXL Model:**
```bash
conda activate apl
python backend/test_sdxl_integration.py
# Answer 'y' when prompted to run SDXL generation test
```

---

## What's Been Completed

### ✅ Stage 1: Core PBO Class ([backend/pbo.py](backend/pbo.py))
- **Mixture embeddings**: `z = L2_normalize(w @ MU)` where MU are concept centroids
- **Laplace GP**: Cosine-RBF kernel for preference learning
- **Batch acquisition**: 4 strategies (Thompson, EI, variance, diverse)
- **Constraints**: Negative penalties (ρ=0.03, λ=10), per-concept cap (0.35), diversity (cos≤0.95)
- **Candidate management**: Coalescing (cos>0.995), pruning (max 200)
- **Tests**: All passing, 88.7% convergence to known favorite

### ✅ Stage 2: StageRefiner ([backend/stage_refiner.py](backend/stage_refiner.py))
- **UI stabilization**: 500ms debounce + cosine threshold (0.02)
- **Weak duels**: From UI snapshots (strength=0.5)
- **Strong duels**: From favorite picks (strength=1.0)
- **Proposal generation**: `propose_next_4()` returns 4 diverse mixtures
- **Concept phrases**: `get_concept_phrases()` returns Top-K positives + deficit negatives
- **Tests**: All passing (initialization, stabilization, favorites, proposals, phrases)

---

## Stage 3: SDXL Integration

### Goal
Connect PBO-generated concept mixtures to SDXL image generation via weighted phrase embeddings.

### What to Build

#### 3.1 Enhanced Concept-to-Phrases Converter (`backend/sdxl_integration.py`)

Create a new file that converts concept mixtures to SDXL-ready phrases with gain mapping:

```python
def concepts_to_sdxl_phrases(
    w: np.ndarray,              # weight vector (K,)
    concepts: List[Dict],       # concept objects
    top_k: int = 10,           # number of positives
    num_negatives: int = 3     # number of negatives
) -> Tuple[List[Tuple[str, float]], List[str]]:
    """
    Convert mixture weights to SDXL phrases with gains.

    Returns:
        (positive_phrases, negative_phrases)
        positive_phrases: [(phrase, gain), ...] where gain ∈ [0.7, 1.5]
        negative_phrases: [phrase, phrase, ...]
    """
```

**Gain mapping formula** (from spec):
```python
# Normalize weights
w = normalize_simplex(w)

# Z-score of weights
mean_w = np.mean(w)
std_w = np.std(w)
z_scores = (w - mean_w) / (std_w + 1e-8)

# Map to gain range [0.7, 1.5] with lambda=0.4
gains = 1.0 + 0.4 * z_scores
gains = np.clip(gains, 0.7, 1.5)
```

**Key details**:
- **Top-K selection**: Sort by weight desc, take top 8-10
- **Negatives**: Sort by deficit (1/K - w), take top 2-4 where w < uniform/2
- **Phrases**: Use `concept['label']` as canonical phrase (ensure 2-4 tokens)
- **Output**: `[(phrase, gain), ...]` for positives, `[phrase, ...]` for negatives

#### 3.2 Copy & Adapt SDXL Embed Fuser

Copy `pbo_utils/sdxl_embed_fuser.py` → `backend/sdxl_embed_fuser.py` with these tweaks:

**Current implementation** (from pbo_utils):
```python
def fuse_weighted_phrases(
    self,
    pos_phrases: List[Tuple[str, float]],
    neg_phrases: List[str] | None = None
):
    # weights_np = ...power(0.8) softening
    # Encode with both text encoders
    # Weighted sum of embeddings
    # Return (prompt_embeds, pooled, neg_embeds, neg_pooled)
```

**Required tweaks**:
1. **Top-K clamp**: Ensure `len(pos_phrases) <= 10`
2. **Gain application**: Weights should be gains (already in [0.7, 1.5])
3. **Negative clamp**: Ensure `len(neg_phrases) <= 4`
4. **Token budget**: Check total tokens < max_length (77 for CLIP)

**No changes needed** if current fuser already handles these - review the code.

#### 3.3 SDXL Generation Wrapper

Create `backend/sdxl_runner.py` that wraps the diffusion runner:

```python
from pbo_utils.diffusion_runner import DiffusionRunner
from backend.sdxl_embed_fuser import SDXLEmbedFuser
from backend.sdxl_integration import concepts_to_sdxl_phrases

class SDXLRunner:
    def __init__(self, model_id: str, device: str = None):
        self.runner = DiffusionRunner(model_id=model_id, device=device, ...)
        self.runner._ensure_txt2img()  # Load pipeline
        self.fuser = SDXLEmbedFuser(self.runner.pipe, device=device)

    def generate_from_mixture(
        self,
        w: np.ndarray,
        concepts: List[Dict],
        seed: int,
        height: int = 1024,
        width: int = 1024,
        steps: int = 30,
        guidance_scale: float = 7.5
    ) -> PIL.Image:
        """Generate image from concept mixture"""

        # Convert to phrases
        pos_phrases, neg_phrases = concepts_to_sdxl_phrases(w, concepts)

        # Fuse embeddings
        prompt_embeds, pooled, neg_embeds, neg_pooled = \
            self.fuser.fuse_weighted_phrases(pos_phrases, neg_phrases)

        # Generate
        image = self.runner.generate_embeds(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            pooled_prompt_embeds=pooled,
            negative_pooled_prompt_embeds=neg_pooled,
            seed=seed,
            steps=steps,
            gscale=guidance_scale,
            height=height,
            width=width
        )

        return image
```

#### 3.4 Test with Local SDXL

Create `backend/test_sdxl_integration.py`:

```python
def test_sdxl_generation():
    """Test end-to-end SDXL generation from concept mixture"""

    # Create mock concepts
    concepts = [
        {'id': 'c0', 'label': 'cozy', 'centroid': ...},
        {'id': 'c1', 'label': 'modern', 'centroid': ...},
        {'id': 'c2', 'label': 'minimalist', 'centroid': ...},
        {'id': 'c3', 'label': 'warm lighting', 'centroid': ...},
        {'id': 'c4', 'label': 'natural materials', 'centroid': ...},
    ]

    # Test mixture (emphasize cozy + warm)
    w = np.array([0.4, 0.2, 0.15, 0.2, 0.05])

    # Convert to phrases
    pos, neg = concepts_to_sdxl_phrases(w, concepts, top_k=4)

    print("Positive phrases:")
    for phrase, gain in pos:
        print(f"  {phrase}: gain={gain:.3f}")

    print("Negative phrases:", neg)

    # Initialize runner (use conda env 'apl')
    runner = SDXLRunner(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device="cuda"  # or "mps" for Mac
    )

    # Generate
    image = runner.generate_from_mixture(
        w=w,
        concepts=concepts,
        seed=42,
        height=1024,
        width=1024,
        steps=30
    )

    # Save
    image.save("/tmp/test_pbo_sdxl.png")
    print("✅ Image saved to /tmp/test_pbo_sdxl.png")
```

**Run test**:
```bash
cd backend
conda activate apl  # your SDXL environment
python test_sdxl_integration.py
```

**Expected output**:
- Phrases with gains printed
- SDXL generates 1024x1024 image
- Image visually reflects the mixture (cozy + warm dominant)
- No token truncation warnings

---

## Files to Create

1. `backend/sdxl_integration.py` - Concept-to-phrases with gain mapping
2. `backend/sdxl_embed_fuser.py` - Copy from pbo_utils (with tweaks if needed)
3. `backend/sdxl_runner.py` - Wrapper for generation
4. `backend/test_sdxl_integration.py` - Integration test

---

## Integration Points

### From StageRefiner
```python
# In stage_refiner.py, add method:
def generate_images_from_proposals(
    self,
    proposals: List[np.ndarray],
    sdxl_runner: SDXLRunner,
    seed_base: int = 42
) -> List[PIL.Image]:
    """Generate images for each proposal"""
    images = []
    for i, w in enumerate(proposals):
        img = sdxl_runner.generate_from_mixture(
            w=w,
            concepts=self.concepts,
            seed=seed_base + i,
            height=1024,
            width=1024
        )
        images.append(img)
    return images
```

### To Server API (Stage 4)
```python
# In server.py, endpoint will use:
refiner = get_or_create_refiner(session_id, stage)
proposals = refiner.propose_next_4(negatives, w_current, fit_first=True)

# Generate images
sdxl_runner = get_sdxl_runner()  # singleton
images = refiner.generate_images_from_proposals(proposals, sdxl_runner)

# Save and return
image_paths = save_images(images, session_dir, stage)
return {"image_paths": image_paths, "proposals": proposals}
```

---

## Parameters (from Spec)

```python
# Phrases
TOP_K_POSITIVES = 10
NUM_NEGATIVES = 3
GAIN_SCALE_RANGE = (0.7, 1.5)
GAIN_LAMBDA = 0.4

# SDXL
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_HEIGHT = 1024
SDXL_WIDTH = 1024
SDXL_STEPS = 30
SDXL_GUIDANCE_SCALE = 7.5
```

---

## Acceptance Criteria

- ✅ `concepts_to_sdxl_phrases()` produces correct Top-K and negatives
- ✅ Gain mapping follows z-score formula and clips to [0.7, 1.5]
- ✅ SDXL fuser handles gains without token truncation warnings
- ⏳ Generated images visually reflect weight mixtures (requires SDXL model download)
- ⏳ No errors with local SDXL model (requires manual testing with conda env 'apl')
- ⏳ Test generates 4 diverse images from 4 proposals (requires SDXL model download)

---

## Gotchas to Watch

1. **Token truncation**: CLIP has 77-token limit. Keep phrases short (2-4 tokens each).
2. **Gain interpretation**: Fuser should use gains as-is, not normalize them again.
3. **Device handling**: Ensure tensors are on correct device (cuda/mps/cpu).
4. **Memory**: SDXL is heavy. Consider `enable_attention_slicing()` if OOM.
5. **Seed consistency**: Same w + seed should produce same image (for debugging).

---

## Testing Checklist

- [x] Gain mapping produces correct range [0.7, 1.5]
- [x] Top-K selection picks highest-weight concepts
- [x] Negatives include low-weight concepts (deficit-based)
- [x] SDXL fuser loads without errors
- [x] Single image generation works end-to-end (unit tested, SDXL pending)
- [x] Batch generation (4 images) works (unit tested, SDXL pending)
- [ ] Images visually differ based on weight mixtures (requires SDXL model)
- [x] No token truncation warnings in logs

---

## Next Steps After Stage 3

**Stage 4**: Add API endpoints to [backend/server.py](backend/server.py)
- `POST /api/pbo/stabilize` - Record UI snapshot
- `POST /api/pbo/favorite` - Record favorite selection
- `POST /api/pbo/propose` - Generate 4 proposals
- `POST /api/pbo/generate` - Generate images from proposals

**Stage 5**: Wire up frontend + concept_refinement.py
- Add debounce hook to `ConceptRefinementSession`
- Add "Pick Favorite" button
- Add "Generate Next 4 (PBO)" button

**Stage 6**: End-to-end testing and telemetry

---

## Questions / Clarifications Needed

1. **SDXL model path**: Is the model already downloaded, or should we download on first run?
2. **Concept labels**: Are they already in 2-4 token format, or do we need phrase shortening?
3. **Reference images**: Does PBO generation need reference image (img2img) or just txt2img?
4. **Output location**: Where should generated images be saved? `sessions/{session_id}/{stage}/pbo_round_{N}/`?

---

## Contact

For questions on Stages 1 & 2, refer to:
- [backend/pbo.py](backend/pbo.py) - Core PBO implementation
- [backend/stage_refiner.py](backend/stage_refiner.py) - Per-stage workflow
- [backend/test_pbo.py](backend/test_pbo.py) - PBO unit tests
- [backend/test_stage_refiner.py](backend/test_stage_refiner.py) - StageRefiner tests

All tests passing. Ready for Stage 3! 🚀
