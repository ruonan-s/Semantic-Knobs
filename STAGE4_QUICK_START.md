# Stage 4 Quick Start: Server API Integration

Stage 3 is complete! Here's how to proceed with Stage 4: adding API endpoints.

## Quick Reference

### Files to Modify
- `backend/server.py` - Add new PBO endpoints

### Endpoints to Add

#### 1. POST /api/pbo/stabilize
Record UI snapshot (debounced weight changes).

```python
@app.post("/api/pbo/stabilize")
async def stabilize_ui(request: StabilizeRequest):
    """
    Record stabilized UI weights as weak duel.

    Body:
        session_id: str
        stage: str
        w_ui: List[float]  # current UI weights

    Returns:
        {
            "snapshot_recorded": bool,
            "candidate_id": str | None
        }
    """
    refiner = get_or_create_refiner(
        session_id=request.session_id,
        stage=request.stage
    )

    w_ui = np.array(request.w_ui)
    recorded = refiner.on_ui_stabilize(w_ui)

    return {
        "snapshot_recorded": recorded,
        "candidate_id": refiner.last_snapshot_cid if recorded else None
    }
```

#### 2. POST /api/pbo/favorite
Record favorite image selection (strong duels).

```python
@app.post("/api/pbo/favorite")
async def record_favorite(request: FavoriteRequest):
    """
    Record user's favorite image pick.

    Body:
        session_id: str
        stage: str
        favorite_image_id: str
        all_image_ids: List[str]

    Returns:
        {
            "duels_added": int,
            "favorite_candidate_id": str
        }
    """
    refiner = get_or_create_refiner(
        session_id=request.session_id,
        stage=request.stage
    )

    refiner.on_favorite(
        favorite_image_id=request.favorite_image_id,
        all_image_ids=request.all_image_ids
    )

    return {
        "duels_added": len(request.all_image_ids) - 1,
        "favorite_candidate_id": refiner.image_to_candidate[request.favorite_image_id]
    }
```

#### 3. POST /api/pbo/propose
Generate 4 new proposals using PBO.

```python
@app.post("/api/pbo/propose")
async def propose_candidates(request: ProposeRequest):
    """
    Generate 4 new concept mixtures.

    Body:
        session_id: str
        stage: str
        negatives: List[str] | None  # concept IDs to avoid
        w_current: List[float] | None  # current UI weights for seeding

    Returns:
        {
            "proposals": List[List[float]],  # 4 weight vectors
            "proposal_ids": List[str]
        }
    """
    refiner = get_or_create_refiner(
        session_id=request.session_id,
        stage=request.stage
    )

    negatives = set(request.negatives) if request.negatives else None
    w_current = np.array(request.w_current) if request.w_current else None

    proposals = refiner.propose_next_4(
        negatives=negatives,
        w_current=w_current,
        fit_first=True
    )

    return {
        "proposals": [w.tolist() for w in proposals],
        "proposal_ids": [f"prop_{i}" for i in range(len(proposals))]
    }
```

#### 4. POST /api/pbo/generate
Generate images from proposals using SDXL.

```python
@app.post("/api/pbo/generate")
async def generate_images(request: GenerateRequest):
    """
    Generate images from concept mixtures.

    Body:
        session_id: str
        stage: str
        proposals: List[List[float]]  # from /api/pbo/propose
        seed_base: int = 42

    Returns:
        {
            "image_paths": List[str],
            "proposals": List[List[float]]
        }
    """
    refiner = get_or_create_refiner(
        session_id=request.session_id,
        stage=request.stage
    )

    # Get or create SDXL runner (singleton)
    sdxl_runner = get_sdxl_runner()

    # Generate images
    proposals_np = [np.array(w) for w in request.proposals]
    images = refiner.generate_images_from_proposals(
        proposals=proposals_np,
        sdxl_runner=sdxl_runner,
        seed_base=request.seed_base
    )

    # Save images
    session_dir = Path(f"sessions/{request.session_id}/{request.stage}")
    session_dir.mkdir(parents=True, exist_ok=True)

    pbo_round = len(list(session_dir.glob("pbo_round_*")))
    round_dir = session_dir / f"pbo_round_{pbo_round}"
    round_dir.mkdir(exist_ok=True)

    image_paths = []
    for i, img in enumerate(images):
        path = round_dir / f"image_{i}.png"
        img.save(path)
        image_paths.append(str(path))

    return {
        "image_paths": image_paths,
        "proposals": request.proposals
    }
```

### Helper Functions

```python
from backend.sdxl_runner import SDXLRunner
from backend.stage_refiner import StageRefiner

# Global singleton for SDXL runner
_sdxl_runner = None

def get_sdxl_runner() -> SDXLRunner:
    """Get or create global SDXL runner."""
    global _sdxl_runner
    if _sdxl_runner is None:
        _sdxl_runner = SDXLRunner(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            device=None,  # Auto-detect
            height=1024,
            width=1024,
            steps=30,
            guidance_scale=7.5
        )
    return _sdxl_runner

# Global cache for StageRefiners
_refiners = {}

def get_or_create_refiner(session_id: str, stage: str) -> StageRefiner:
    """Get or create StageRefiner for session/stage."""
    key = f"{session_id}:{stage}"

    if key not in _refiners:
        # Load session data
        session_dir = Path(f"sessions/{session_id}/{stage}")

        # Load concepts, image_ids, incidence_matrix from session
        # (This depends on your existing session storage format)
        concepts = load_concepts(session_id, stage)
        concept_states = load_concept_states(session_id, stage)
        image_ids = load_image_ids(session_id, stage)
        incidence_matrix = load_incidence_matrix(session_id, stage)

        _refiners[key] = StageRefiner(
            session_id=session_id,
            stage=stage,
            concepts=concepts,
            concept_states=concept_states,
            image_ids=image_ids,
            incidence_matrix=incidence_matrix,
            session_dir=session_dir
        )

    return _refiners[key]
```

## Testing

Test endpoints with curl:

```bash
# 1. Stabilize
curl -X POST http://localhost:8000/api/pbo/stabilize \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "stage": "stage1", "w_ui": [0.3, 0.3, 0.2, 0.1, 0.1]}'

# 2. Propose
curl -X POST http://localhost:8000/api/pbo/propose \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "stage": "stage1", "negatives": null, "w_current": null}'

# 3. Generate (requires SDXL model)
curl -X POST http://localhost:8000/api/pbo/generate \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "stage": "stage1",
    "proposals": [[0.3, 0.3, 0.2, 0.1, 0.1], [0.2, 0.4, 0.2, 0.1, 0.1]],
    "seed_base": 42
  }'

# 4. Favorite
curl -X POST http://localhost:8000/api/pbo/favorite \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "stage": "stage1",
    "favorite_image_id": "img_5",
    "all_image_ids": ["img_3", "img_4", "img_5", "img_6"]
  }'
```

## Next Steps (Stage 5)

After Stage 4 is complete:

1. **Frontend Integration**
   - Add debounce hook to `ConceptRefinementSession` → calls `/api/pbo/stabilize`
   - Add "Pick Favorite" button → calls `/api/pbo/favorite`
   - Add "Generate Next 4 (PBO)" button → calls `/api/pbo/propose` + `/api/pbo/generate`

2. **UI Flow**
   ```
   User adjusts sliders
     ↓ (debounced 500ms)
   POST /api/pbo/stabilize → weak duel added

   User clicks "Generate Next 4 (PBO)"
     ↓
   POST /api/pbo/propose → get 4 proposals
     ↓
   POST /api/pbo/generate → generate 4 images
     ↓
   Display 4 images in grid

   User picks favorite
     ↓
   POST /api/pbo/favorite → strong duels added
     ↓
   Repeat
   ```

3. **Telemetry**
   - Log PBO convergence metrics
   - Track user interactions (stabilize, favorite, generate)
   - Monitor SDXL generation time

## Questions?

Refer to:
- [STAGE3_HANDOFF.md](STAGE3_HANDOFF.md) - Stage 3 details
- [backend/stage_refiner.py](backend/stage_refiner.py) - StageRefiner API
- [backend/sdxl_runner.py](backend/sdxl_runner.py) - SDXL generation API
- [backend/test_sdxl_integration.py](backend/test_sdxl_integration.py) - Integration examples
