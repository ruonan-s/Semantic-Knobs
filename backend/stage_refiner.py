"""
StageRefiner - Manages PBO refinement for a single stage

Handles:
- UI stabilization → weak duels (debounced snapshots)
- Favorite selection → strong duels (image comparisons)
- Proposal generation → 4 new mixtures for SDXL
- Image generation coordination (delegates to SDXL pipeline)
"""

from __future__ import annotations
import numpy as np
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

try:
    from backend.pbo import PBO, normalize_simplex, cosine_similarity
except ImportError:
    from pbo import PBO, normalize_simplex, cosine_similarity


# ============================================================================
# Parameters
# ============================================================================
STABILIZE_DEBOUNCE_MS = 500
STABILIZE_COSINE_THRESHOLD = 0.02


# ============================================================================
# StageRefiner Class
# ============================================================================
class StageRefiner:
    """
    Manages PBO-driven refinement for a single stage.

    Workflow:
    1. Initialize with concepts (from concept_refinement.py)
    2. User interactions → w_ui updates → on_ui_stabilize() (debounced)
    3. User picks favorite → on_favorite() (strong duels)
    4. User clicks "Next 4" → propose_next_4() → gen_images()
    5. Repeat
    """

    def __init__(
        self,
        session_id: str,
        stage: str,
        concepts: List[Dict],  # from concept_refinement.py
        concept_states: Dict[str, Dict],  # from concept_refinement.py
        image_ids: List[str],
        incidence_matrix: Dict[str, Dict[str, int]],  # image_id -> concept_id -> count
        session_dir: Path,
        random_state: int = 42
    ):
        self.session_id = session_id
        self.stage = stage
        self.concepts = concepts  # list of Concept dicts
        self.concept_states = concept_states
        self.image_ids = image_ids
        self.incidence_matrix = incidence_matrix
        self.session_dir = Path(session_dir)
        self.rng = np.random.RandomState(random_state)

        # Build concept centroids matrix MU (K, d)
        self.K = len(concepts)
        self.concept_ids = [c['id'] for c in concepts]

        # Extract centroids
        MU = []
        for concept in concepts:
            centroid = np.array(concept['centroid'], dtype=np.float32)
            # Ensure L2 normalized
            norm = np.linalg.norm(centroid)
            if norm > 1e-8:
                centroid = centroid / norm
            else:
                # Degenerate - use random
                centroid = self.rng.randn(len(centroid)).astype(np.float32)
                centroid = centroid / np.linalg.norm(centroid)
            MU.append(centroid)

        self.MU = np.array(MU, dtype=np.float32)  # (K, d)
        self.d = self.MU.shape[1]

        # Extract concept weights (w) for warm start
        concept_weights = np.array([
            concept_states.get(cid, {}).get('w', 1.0 / self.K)
            for cid in self.concept_ids
        ], dtype=np.float32)
        
        # Normalize weights to sum to 1 (should already be, but ensure)
        if concept_weights.sum() > 0:
            concept_weights = concept_weights / concept_weights.sum()
        else:
            concept_weights = np.ones(self.K, dtype=np.float32) / self.K
        
        print(f"[StageRefiner] Concept weights for PBO initialization:")
        top_3_indices = np.argsort(-concept_weights)[:3]
        for idx in top_3_indices:
            print(f"  {concepts[idx]['label']}: {concept_weights[idx]:.4f}")

        # Initialize PBO with concept weights for warm start
        self.pbo = PBO(
            MU=self.MU,
            concept_ids=self.concept_ids,
            random_state=random_state,
            concept_weights=concept_weights  # Pass weights for warm start
        )

        # Snapshot tracking for stabilization
        self.last_snapshot_time = 0
        self.last_snapshot_w: Optional[np.ndarray] = None
        self.last_snapshot_cid: Optional[str] = None

        # Image candidate mapping (for duels from favorite picks)
        self.image_to_candidate: Dict[str, str] = {}  # image_id -> candidate_id

        print(f"[StageRefiner] Initialized for {session_id}/{stage}")
        print(f"  K={self.K} concepts, d={self.d} embedding dim")
        print(f"  {len(image_ids)} images")

    def on_ui_stabilize(self, w_ui: np.ndarray) -> bool:
        """
        Called when UI weights stabilize (debounced).

        Checks:
        1. Debounce: at least STABILIZE_DEBOUNCE_MS since last snapshot
        2. Significant change: cosine(w_prev, w_now) < 1 - STABILIZE_COSINE_THRESHOLD

        If both pass:
        - Add candidate for w_ui
        - If prev snapshot exists, add weak duel (now > prev)
        - Update snapshot tracking

        Args:
            w_ui: current UI weights (K,)

        Returns:
            True if snapshot was recorded, False if debounce/threshold not met
        """
        current_time = time.time() * 1000  # ms

        # Check debounce
        if current_time - self.last_snapshot_time < STABILIZE_DEBOUNCE_MS:
            return False

        # Check significant change
        w_ui = normalize_simplex(w_ui)

        if self.last_snapshot_w is not None:
            cos_sim = cosine_similarity(w_ui, self.last_snapshot_w)
            if cos_sim > (1.0 - STABILIZE_COSINE_THRESHOLD):
                # Too similar - skip
                return False

        # Record snapshot
        cid = self.pbo.add_candidate(w_ui)

        # Add weak duel if previous exists
        if self.last_snapshot_cid is not None and self.last_snapshot_cid != cid:
            self.pbo.add_preference(cid, self.last_snapshot_cid, strength=0.5)
            print(f"[StageRefiner] Weak duel: {cid} ≻ {self.last_snapshot_cid} (stabilize)")

        # Update tracking
        self.last_snapshot_time = current_time
        self.last_snapshot_w = w_ui.copy()
        self.last_snapshot_cid = cid

        print(f"[StageRefiner] Snapshot recorded: {cid}")
        return True

    def on_favorite(
        self,
        favorite_image_id: str,
        all_image_ids: List[str],
        incidence_matrix: Optional[Dict[str, Dict[str, int]]] = None
    ) -> None:
        """
        Called when user picks a favorite image among candidates.

        Builds proxy weights for each image: w^(j) ∝ A[j, :]
        Registers each as a PBO candidate, then adds strong duels (fav > others).

        Args:
            favorite_image_id: ID of favorite image
            all_image_ids: List of all candidate image IDs in this round
            incidence_matrix: Optional updated incidence matrix
                              (defaults to self.incidence_matrix)
        """
        if incidence_matrix is None:
            incidence_matrix = self.incidence_matrix

        print(f"\n[StageRefiner] Favorite selected: {favorite_image_id}")
        print(f"  Comparing against {len(all_image_ids)} images")

        # Build proxy weights for each image
        image_weights = {}
        image_cands = {}

        for img_id in all_image_ids:
            # Get concept incidences for this image
            concept_counts = incidence_matrix.get(img_id, {})

            # Build weight vector (proportional to incidence)
            w_img = np.zeros(self.K, dtype=np.float32)
            for i, cid in enumerate(self.concept_ids):
                w_img[i] = concept_counts.get(cid, 0)

            # Normalize
            w_img = normalize_simplex(w_img)
            image_weights[img_id] = w_img

            # Add as PBO candidate
            cand_id = self.pbo.add_candidate(w_img, candidate_id=f"img_{img_id}")
            image_cands[img_id] = cand_id
            self.image_to_candidate[img_id] = cand_id

            print(f"  Image {img_id}: w_max={w_img.max():.3f}, cand={cand_id}")

        # Add strong duels: favorite > others
        fav_cid = image_cands[favorite_image_id]

        for img_id in all_image_ids:
            if img_id != favorite_image_id:
                other_cid = image_cands[img_id]
                self.pbo.add_preference(fav_cid, other_cid, strength=1.0)
                print(f"  Strong duel: {fav_cid} ≻ {other_cid}")

        print(f"[StageRefiner] Favorite processed: {len(all_image_ids) - 1} strong duels added")

    def propose_next_4(
        self,
        negatives: Optional[set] = None,
        w_current: Optional[np.ndarray] = None,
        fit_first: bool = True
    ) -> List[np.ndarray]:
        """
        Propose 4 new concept mixtures using PBO.

        Args:
            negatives: set of negative concept IDs (user dislikes)
            w_current: current UI weights (for Dirichlet seeding)
            fit_first: whether to fit GP before proposing (default True)

        Returns:
            List of 4 weight vectors (each K,)
        """
        print(f"\n[StageRefiner] Proposing next 4 candidates")

        # Fit GP if requested
        if fit_first:
            self.pbo.fit()

        # Propose batch
        proposals = self.pbo.propose_batch(
            q=4,
            negatives=negatives,
            w_current=w_current
        )

        print(f"[StageRefiner] Proposed {len(proposals)} candidates")
        return proposals

    def get_concept_phrases(
        self,
        w: np.ndarray,
        top_k: int = 10
    ) -> Tuple[List[str], List[str]]:
        """
        Convert weight mixture to concept phrases for SDXL.

        Args:
            w: weight vector (K,)
            top_k: number of top concepts to include as positives

        Returns:
            (positive_phrases, negative_phrases)
            positive_phrases: list of (phrase, weight) tuples
            negative_phrases: list of phrase strings
        """
        w = normalize_simplex(w)

        # Get Top-K positives by weight
        top_indices = np.argsort(-w)[:top_k]
        positive_phrases = []

        for idx in top_indices:
            concept = self.concepts[idx]
            phrase = concept['label']  # use concept label as phrase
            weight = float(w[idx])
            positive_phrases.append((phrase, weight))

        # Get negatives (bottom concepts by deficit: 1/K - w)
        uniform = 1.0 / self.K
        deficit = uniform - w
        neg_indices = np.argsort(-deficit)[:3]  # top 3 by deficit

        negative_phrases = []
        for idx in neg_indices:
            if w[idx] < uniform * 0.5:  # only if significantly below uniform
                concept = self.concepts[idx]
                phrase = concept['label']
                negative_phrases.append(phrase)

        return positive_phrases, negative_phrases

    def generate_images_from_proposals(
        self,
        proposals: List[np.ndarray],
        sdxl_runner,  # SDXLRunner instance
        seed_base: int = 42,
        tracker: Optional[Any] = None,
        generated_image_paths: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Generate images for each proposal using SDXL.

        This method bridges StageRefiner with SDXLRunner for Stage 3+ integration.

        Args:
            proposals: List of weight vectors (from propose_next_4())
            sdxl_runner: SDXLRunner instance (from backend.sdxl_runner)
            seed_base: Base seed for generation (each proposal gets seed_base + i)
            tracker: GenerationTracker instance for logging (optional)
            generated_image_paths: List of paths where images will be saved (optional)
            **kwargs: Additional arguments for SDXLRunner.generate_from_mixture()
                     Common kwargs: init_image, descriptor, strength, verbose

        Returns:
            List of PIL Images

        Example:
            >>> from backend.sdxl_runner import SDXLRunner
            >>> from backend.tracking import create_tracker
            >>> runner = SDXLRunner(model_id="stabilityai/stable-diffusion-xl-base-1.0")
            >>> refiner = StageRefiner(...)
            >>> tracker = create_tracker(session_path, session_id, stage, descriptor)
            >>> proposals = refiner.propose_next_4()
            >>> images = refiner.generate_images_from_proposals(
            ...     proposals, runner, 
            ...     descriptor="A comfortable space for reading",
            ...     tracker=tracker
            ... )
        """
        print(f"\n[StageRefiner] Generating {len(proposals)} images from proposals...")

        images = []
        for i, w in enumerate(proposals):
            print(f"\n  Proposal {i+1}/{len(proposals)}:")
            
            # Get image path for this proposal if provided
            image_path = generated_image_paths[i] if generated_image_paths and i < len(generated_image_paths) else None
            
            img = sdxl_runner.generate_from_mixture(
                w=w,
                concepts=self.concepts,
                seed=seed_base + i,
                stage=self.stage,  # Pass stage for loading strength from config
                tracker=tracker,  # Pass tracker for logging
                proposal_index=i,  # Pass index for tracking
                generated_image_path=image_path,  # Pass path for tracking
                **kwargs
            )
            images.append(img)

        print(f"\n[StageRefiner] ✅ Generated {len(images)} images")
        return images

    def to_dict(self) -> Dict:
        """Serialize state"""
        return {
            'session_id': self.session_id,
            'stage': self.stage,
            'K': self.K,
            'd': self.d,
            'num_images': len(self.image_ids),
            'pbo_state': self.pbo.to_dict(),
            'last_snapshot_cid': self.last_snapshot_cid,
            'num_image_candidates': len(self.image_to_candidate)
        }
