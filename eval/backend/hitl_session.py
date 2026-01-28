"""
HITL Refinement Session - Continuous Aesthetic Discovery with PCA

Orchestrates the Human-in-the-Loop preference learning loop using
PCA-reduced embedding space for efficient GP learning.

Flow:
1. Load tags → compute 768-dim CLIP embeddings
2. Fit PCA to reduce to ~32 dimensions
3. GP operates in reduced space → faster convergence
4. Sample in reduced space → reconstruct to 768 for SDXL
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from exploration_GP import PreferenceLearner
from dim_reducer import EmbeddingReducer
from hitl_sampler import ContinuousEmbeddingSampler, ContinuousComposition
from hitl_fuser import DirectEmbeddingInjector, generate_with_direct_injection
from repelling_optimizer import ContinuousSpaceOptimizer


class HITLRefinementSession:
    """
    Orchestrates continuous aesthetic discovery in reduced CLIP space.
    
    Uses PCA to reduce 768-dim CLIP embeddings to ~32 dimensions where
    the GP can effectively learn and converge.
    """
    
    def __init__(
        self,
        session_id: str,
        session_folder: str,
        sdxl_runner: Any = None,
        pipe: Any = None,
        base_prompt: str = "",
        negative_phrases: Optional[List[str]] = None,
        convergence_threshold: float = 0.15,  # Lower in reduced space
        max_rounds: int = 10,
        reduced_dim: int = 32  # Target dimensionality
    ):
        """
        Initialize HITL refinement session.
        
        Args:
            reduced_dim: Target dimensionality for PCA (default 32)
        """
        self.session_id = session_id
        self.session_folder = session_folder
        self.sdxl_runner = sdxl_runner
        self.pipe = pipe
        self.base_prompt = base_prompt
        self.negative_phrases = negative_phrases or []
        self.convergence_threshold = convergence_threshold
        self.max_rounds = max_rounds
        self.reduced_dim = reduced_dim
        
        # Components
        self.reducer: Optional[EmbeddingReducer] = None
        self.gp: Optional[PreferenceLearner] = None
        self.sampler: Optional[ContinuousEmbeddingSampler] = None
        self.optimizer: Optional[ContinuousSpaceOptimizer] = None
        self.injector: Optional[DirectEmbeddingInjector] = None
        
        # Store ORIGINAL 768-dim embeddings
        self.positive_embeddings_768: Optional[np.ndarray] = None
        self.negative_embeddings_768: Optional[np.ndarray] = None
        self.neutral_embeddings_768: Optional[np.ndarray] = None
        
        # State
        self.round_count = 0
        self.compositions_history: List[List[ContinuousComposition]] = []
        self.rankings_history: List[List[int]] = []
        self.is_initialized = False
        self.is_converged = False
        
        # Persistence
        self.state_file = os.path.join(session_folder, "hitl_state.json")
        self.hitl_folder = os.path.join(session_folder, "hitl")
        
        # CLIP model
        self._clip_model = None
        self._clip_device = None
    
    def _load_clip(self):
        """Lazy-load CLIP model."""
        if self._clip_model is None:
            import clip
            self._clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model, _ = clip.load("ViT-L/14", device=self._clip_device)
            print(f"[HITLSession] Loaded CLIP ViT-L/14 on {self._clip_device}")
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get 768-dim CLIP embeddings for texts."""
        if not texts:
            return np.zeros((0, 768))
        
        self._load_clip()
        import clip
        
        with torch.no_grad():
            tokens = clip.tokenize(texts, truncate=True).to(self._clip_device)
            features = self._clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().astype(np.float32)
    
    def _load_json(self, filename: str) -> Dict:
        """Load JSON file from session folder."""
        impression_path = os.path.join(self.session_folder, "impression", filename)
        if os.path.exists(impression_path):
            with open(impression_path) as f:
                return json.load(f)
        
        direct_path = os.path.join(self.session_folder, filename)
        if os.path.exists(direct_path):
            with open(direct_path) as f:
                return json.load(f)
        
        raise FileNotFoundError(f"Cannot find {filename} in {self.session_folder}")
    
    def initialize_from_exploration(self):
        """
        Initialize from exploration stage outputs with PCA reduction.
        
        1. Load tags and compute 768-dim CLIP embeddings
        2. Fit PCA to reduce to reduced_dim dimensions
        3. Initialize GP, sampler, optimizer in reduced space
        """
        print(f"[HITLSession] Initializing with PCA reduction to {self.reduced_dim} dims")
        
        os.makedirs(self.hitl_folder, exist_ok=True)
        
        # Load exploration outputs
        tag_prefs = self._load_json("tag_preferences.json")
        
        positive_tags = tag_prefs.get("positive", [])
        negative_tags = tag_prefs.get("negative", [])
        neutral_tags = tag_prefs.get("neutral", [])
        
        print(f"[HITLSession] Tags: {len(positive_tags)} pos, {len(negative_tags)} neg, {len(neutral_tags)} neutral")
        
        # Compute 768-dim CLIP embeddings
        self.positive_embeddings_768 = self._get_embeddings(positive_tags)
        self.negative_embeddings_768 = self._get_embeddings(negative_tags)
        self.neutral_embeddings_768 = self._get_embeddings(neutral_tags)
        
        # Fit PCA on ALL embeddings
        all_embeddings = []
        if len(self.positive_embeddings_768) > 0:
            all_embeddings.append(self.positive_embeddings_768)
        if len(self.negative_embeddings_768) > 0:
            all_embeddings.append(self.negative_embeddings_768)
        if len(self.neutral_embeddings_768) > 0:
            all_embeddings.append(self.neutral_embeddings_768)
        
        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
        else:
            all_embeddings = np.random.randn(10, 768)  # Fallback
        
        # Fit PCA reducer
        actual_dim = min(self.reduced_dim, len(all_embeddings) - 1, 768)
        self.reducer = EmbeddingReducer(n_components=actual_dim)
        self.reducer.fit(all_embeddings)
        
        # Initialize GP in REDUCED space
        self.gp = PreferenceLearner(
            embedding_dim=self.reducer.dim,  # Reduced dimension!
            n_inducing=50,
            device=self._clip_device or "cuda"
        )
        
        # Initialize optimizer in reduced space
        self.optimizer = ContinuousSpaceOptimizer(
            gp=self.gp,
            reducer=self.reducer,
            positive_embeddings_768=self.positive_embeddings_768,
            negative_embeddings_768=self.negative_embeddings_768,
            neutral_embeddings_768=self.neutral_embeddings_768,
            convergence_threshold=self.convergence_threshold
        )
        self.optimizer.seed_gp()
        
        # Initialize sampler in reduced space
        # Pass optimizer as GP - it exposes predict_utility() interface
        self.sampler = ContinuousEmbeddingSampler(
            gp=self.optimizer,  # Optimizer has predict_utility() method
            reducer=self.reducer,
            positive_embeddings_768=self.positive_embeddings_768,
            negative_embeddings_768=self.negative_embeddings_768,
            neutral_embeddings_768=self.neutral_embeddings_768
        )
        
        # Initialize injector (still works with 768-dim for SDXL)
        if self.pipe is not None:
            self.injector = DirectEmbeddingInjector(
                pipe=self.pipe,
                device=getattr(self.pipe, 'device', 'cuda')
            )
        
        self.is_initialized = True
        self._save_state()
        
        n_obs = len(self.optimizer.all_observations)
        print(f"[HITLSession] Initialized in {self.reducer.dim}D space, "
              f"GP seeded with {n_obs} observations, "
              f"{self.reducer.explained_variance*100:.1f}% variance explained")
    
    def generate_round(self, q: int = 4) -> Tuple[List[ContinuousComposition], List[str]]:
        """Generate q compositions and images for ranking."""
        if not self.is_initialized:
            raise RuntimeError("Session not initialized")
        
        self.sampler.invalidate_cache()
        
        # Sample compositions (in reduced space, reconstructed to 768)
        compositions = self.sampler.sample_batch(batch_size=q, n_points=10)
        
        # Generate images
        image_paths = self._generate_images(compositions)
        
        self.compositions_history.append(compositions)
        self._save_state()
        
        print(f"[HITLSession] Generated round {self.round_count + 1} with {len(compositions)} compositions")
        
        return compositions, image_paths
    
    def _generate_images(self, compositions: List[ContinuousComposition]) -> List[str]:
        """Generate images using direct embedding injection."""
        image_paths = []
        
        neg_composition = self.sampler.sample_negative_composition(n_points=10)
        neg_prompt_text = ", ".join(self.negative_phrases) if self.negative_phrases else ""
        
        for i, comp in enumerate(compositions):
            img_filename = f"round_{self.round_count}_img_{i}.png"
            img_path = os.path.join(self.hitl_folder, img_filename)
            
            if self.injector is not None and self.pipe is not None:
                try:
                    # Direct embedding injection (uses 768-dim points)
                    prompt_embeds, pooled, neg_embeds, neg_pooled, attn_ctrl = \
                        self.injector.inject_composition(
                            composition=comp,
                            base_prompt=self.base_prompt,
                            negative_composition=neg_composition,
                            neg_prompt=neg_prompt_text
                        )
                    
                    image = generate_with_direct_injection(
                        self.pipe,
                        prompt_embeds, pooled,
                        neg_embeds, neg_pooled,
                        attn_ctrl,
                        seed=42 + i + self.round_count * 100,
                        use_attention_hooks=True
                    )
                    image.save(img_path)
                    
                    avg_utility = np.mean(comp.utilities)
                    avg_sigma = np.mean(comp.uncertainties)
                    print(f"[HITLSession] Generated {img_filename} - utility: {avg_utility:.3f}, σ: {avg_sigma:.3f}")
                    
                except Exception as e:
                    print(f"[HITLSession] Generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    self._create_placeholder_image(img_path, comp)
            else:
                self._create_placeholder_image(img_path, comp)
            
            image_paths.append(img_path)
        
        return image_paths
    
    def _create_placeholder_image(self, path: str, comp: ContinuousComposition):
        """Create a placeholder image with debug info."""
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', (512, 512), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        
        draw.text((20, 20), f"Placeholder - Round {self.round_count}", fill='black')
        draw.text((20, 50), f"Points: {len(comp.points)}", fill='black')
        draw.text((20, 80), f"Reduced dim: {comp.points_reduced.shape[1]}", fill='black')
        draw.text((20, 110), f"Strategies: {comp.sampling_strategies[:3]}", fill='black')
        draw.text((20, 140), f"Mean utility: {np.mean(comp.utilities):.3f}", fill='black')
        draw.text((20, 170), f"Mean σ: {np.mean(comp.uncertainties):.3f}", fill='black')
        
        img.save(path)
    
    def record_ranking(self, ranking: List[int]) -> Dict:
        """Record user's ordinal ranking and update GP in reduced space."""
        if not self.compositions_history:
            raise RuntimeError("No compositions to rank")
        
        current_compositions = self.compositions_history[-1]
        
        if len(ranking) != len(current_compositions):
            raise ValueError(f"Ranking length {len(ranking)} != compositions {len(current_compositions)}")
        
        self.rankings_history.append(ranking)
        
        # Update optimizer (in reduced space)
        result = self.optimizer.update_from_ranking(current_compositions, ranking)
        
        # Update sampler's distribution (pass 768-dim for reconstruction)
        best_idx = ranking[0]
        preferred_points = current_compositions[best_idx].points  # 768-dim
        self.sampler.update_distribution(preferred_points, learning_rate=0.3)
        
        self.round_count += 1
        
        gp_variance = result.get("gp_variance", 1.0)
        self.is_converged = gp_variance < self.convergence_threshold or self.round_count >= self.max_rounds
        
        self._save_state()
        
        metrics = {
            "round": self.round_count,
            "gp_variance": gp_variance,
            "is_converged": self.is_converged,
            "total_pairs": result.get("total_pairs", 0),
            "centroid_utility": result.get("centroid_utility", 0.0),
            "reduced_dim": self.reducer.dim,
        }
        
        print(f"[HITLSession] Round {self.round_count} - σ: {gp_variance:.4f} "
              f"(threshold: {self.convergence_threshold}), Converged: {self.is_converged}")
        
        return metrics
    
    def finalize(self) -> str:
        """Export refined preferences to final_selection.json."""
        # Get current centroid in 768-dim
        centroid_768 = self.optimizer.get_current_centroid_768()
        
        # Get top regions in 768-dim
        top_regions = self.optimizer.get_top_regions(n_regions=5)
        
        metrics = self.optimizer._compute_metrics()
        
        final_data = {
            "session_id": self.session_id,
            "base_prompt": self.base_prompt,
            "rounds_completed": self.round_count,
            "is_converged": self.is_converged,
            "final_variance": metrics["gp_variance"],
            "centroid_utility": metrics["centroid_utility"],
            "reduced_dim": self.reducer.dim,
            "explained_variance": self.reducer.explained_variance,
            "preference_centroid": centroid_768.tolist(),
            "top_regions": [r.tolist() for r in top_regions],
            "timestamp": datetime.now().isoformat()
        }
        
        output_path = os.path.join(self.session_folder, "final_selection.json")
        with open(output_path, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        print(f"[HITLSession] Finalized preferences to {output_path}")
        
        return output_path
    
    def _save_state(self):
        """Save session state to disk."""
        state = {
            "session_id": self.session_id,
            "round_count": self.round_count,
            "is_initialized": self.is_initialized,
            "is_converged": self.is_converged,
            "rankings_history": self.rankings_history,
            "base_prompt": self.base_prompt,
            "negative_phrases": self.negative_phrases,
            "reduced_dim": self.reducer.dim if self.reducer else self.reduced_dim,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load session state from disk."""
        if not os.path.exists(self.state_file):
            return False
        
        with open(self.state_file) as f:
            state = json.load(f)
        
        self.round_count = state.get("round_count", 0)
        self.is_initialized = state.get("is_initialized", False)
        self.is_converged = state.get("is_converged", False)
        self.rankings_history = state.get("rankings_history", [])
        
        return True
    
    def get_top_concepts(self, k: int = 5) -> List[Dict]:
        """Get top utility concepts for display."""
        if not self.is_initialized or self.optimizer is None:
            return []
        
        top_regions = self.optimizer.get_top_regions(n_regions=k)
        
        concepts = []
        for i, region in enumerate(top_regions):
            # Reduce to get utility prediction
            region_reduced = self.reducer.reduce(region)
            mu, sigma = self.gp.predict_utility(region_reduced.reshape(1, -1))
            concepts.append({
                "rank": i + 1,
                "utility": float(mu[0]),
                "uncertainty": float(sigma[0])
            })
        
        return concepts
    
    @classmethod
    def load_or_create(
        cls,
        session_id: str,
        session_folder: str,
        sdxl_runner: Any = None,
        pipe: Any = None,
        base_prompt: str = "",
        negative_phrases: Optional[List[str]] = None
    ) -> "HITLRefinementSession":
        """Idempotent initialization: load existing or create new session."""
        session = cls(
            session_id=session_id,
            session_folder=session_folder,
            sdxl_runner=sdxl_runner,
            pipe=pipe,
            base_prompt=base_prompt,
            negative_phrases=negative_phrases
        )
        
        if session._load_state() and session.is_initialized:
            print(f"[HITLSession] Loaded existing state (round {session.round_count})")
            session.initialize_from_exploration()
        else:
            print(f"[HITLSession] Initializing fresh...")
            session.initialize_from_exploration()
        
        return session
