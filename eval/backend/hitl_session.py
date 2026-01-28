"""
HITL Refinement Session - Session Orchestration

Manages the full HITL refinement loop:
1. Initialize from exploration outputs (tag_preferences.json, concept_weights.json)
2. Seed GP with exploration utilities for warm start
3. Generate rounds of compositions for ranking
4. Update GP from ordinal rankings
5. Track variance-based convergence
6. Persist state for browser refresh recovery (idempotency)
"""

import os
import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

from exploration_GP import PreferenceLearner, PreferencePair
from hitl_sampler import HITLSampler, CompositionSample
from repelling_optimizer import RepellingOptimizer, RankingToPairConverter, create_synthetic_pairs_from_utilities
from hitl_fuser import HITLCompositionFuser, generate_with_hooks, create_composition_prompt


class HITLRefinementSession:
    """
    Manages HITL refinement loop for a single eval session.
    
    Key features:
    - GP seeded with exploration utilities (warm start)
    - Variance-based convergence (not mu shift)
    - Persistent state for browser refresh recovery
    """
    
    def __init__(
        self,
        session_id: str,
        session_folder: str,
        sdxl_runner: Any = None,
        pipe: Any = None,
        base_prompt: str = "",
        negative_phrases: Optional[List[str]] = None,
        convergence_threshold: float = 0.05,
        max_rounds: int = 10
    ):
        self.session_id = session_id
        self.session_folder = session_folder
        self.sdxl_runner = sdxl_runner
        self.pipe = pipe
        self.base_prompt = base_prompt
        self.negative_phrases = negative_phrases or []
        self.convergence_threshold = convergence_threshold
        self.max_rounds = max_rounds
        
        # Components
        self.gp: Optional[PreferenceLearner] = None
        self.sampler: Optional[HITLSampler] = None
        self.optimizer: Optional[RepellingOptimizer] = None
        self.fuser: Optional[HITLCompositionFuser] = None
        
        # Tag data
        self.all_tags: List[str] = []
        self.all_embeddings: Optional[np.ndarray] = None
        self.positive_tags: List[str] = []
        self.negative_tags: List[str] = []
        self.neutral_tags: List[str] = []
        
        # State
        self.round_count = 0
        self.compositions_history: List[List[CompositionSample]] = []
        self.rankings_history: List[List[int]] = []
        self.is_initialized = False
        self.is_converged = False
        
        # Persistence file
        self.state_file = os.path.join(session_folder, "hitl_state.json")
        self.hitl_folder = os.path.join(session_folder, "hitl")
        
        # CLIP model for image embeddings
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_device = None
        
        # Image embeddings for variance tracking
        self.image_embeddings_history: List[np.ndarray] = []  # Per-round image embeddings
    
    def _load_clip(self):
        """Lazy-load CLIP model."""
        if self._clip_model is None:
            import clip
            self._clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model, _ = clip.load("ViT-L/14", device=self._clip_device)
            print(f"[HITLSession] Loaded CLIP ViT-L/14 on {self._clip_device}")
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get CLIP embeddings for texts."""
        self._load_clip()
        import clip
        
        with torch.no_grad():
            tokens = clip.tokenize(texts, truncate=True).to(self._clip_device)
            features = self._clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().astype(np.float32)
    
    def _load_json(self, filename: str) -> Dict:
        """Load JSON file from session folder."""
        # Check impression subfolder first
        impression_path = os.path.join(self.session_folder, "impression", filename)
        if os.path.exists(impression_path):
            with open(impression_path) as f:
                return json.load(f)
        
        # Then check session folder directly
        direct_path = os.path.join(self.session_folder, filename)
        if os.path.exists(direct_path):
            with open(direct_path) as f:
                return json.load(f)
        
        raise FileNotFoundError(f"Cannot find {filename} in {self.session_folder}")
    
    def initialize_from_exploration(self):
        """
        Seed GP with exploration data for warm start.
        
        1. Load tag_preferences.json (positive/negative/neutral tag lists)
        2. Load concept_weights.json (GP utilities from exploration)
        3. SEED THE GP with utilities as initial observations
        4. Initialize (positive, negative) pairs in optimizer for repulsion
        """
        print(f"[HITLSession] Initializing from exploration for session {self.session_id}")
        
        # Create HITL folder
        os.makedirs(self.hitl_folder, exist_ok=True)
        
        # Load exploration outputs
        tag_prefs = self._load_json("tag_preferences.json")
        
        try:
            concept_weights = self._load_json("concept_weights.json")
        except FileNotFoundError:
            concept_weights = {"concept_weights": []}
        
        # Extract tag lists
        self.positive_tags = tag_prefs.get("positive", [])
        self.negative_tags = tag_prefs.get("negative", [])
        self.neutral_tags = tag_prefs.get("neutral", [])
        
        # Combine all tags
        self.all_tags = self.positive_tags + self.negative_tags + self.neutral_tags
        
        if not self.all_tags:
            raise ValueError("No tags found in exploration outputs")
        
        print(f"[HITLSession] Loaded {len(self.positive_tags)} positive, {len(self.negative_tags)} negative, {len(self.neutral_tags)} neutral tags")
        
        # Get embeddings for all tags
        self.all_embeddings = self._get_embeddings(self.all_tags)
        
        # Separate embeddings by category
        n_pos = len(self.positive_tags)
        n_neg = len(self.negative_tags)
        positive_embeddings = self.all_embeddings[:n_pos]
        negative_embeddings = self.all_embeddings[n_pos:n_pos + n_neg]
        
        # Create GP
        self.gp = PreferenceLearner(
            embedding_dim=768,
            n_inducing=min(64, len(self.all_tags)),
            device=self._clip_device
        )
        
        # Seed GP from concept weights if available
        self._seed_gp_from_concept_weights(concept_weights)
        
        # Initialize optimizer with GP and negative embeddings
        self.optimizer = RepellingOptimizer(
            preference_gp=self.gp,
            negative_embeddings=list(negative_embeddings),
            convergence_threshold=self.convergence_threshold
        )
        
        # Initialize with (positive, negative) pairs for repulsion
        if len(positive_embeddings) > 0 and len(negative_embeddings) > 0:
            self.optimizer.initialize_with_negatives(list(positive_embeddings))
        
        # Initialize sampler
        self.sampler = HITLSampler(
            preference_gp=self.gp,
            all_tag_embeddings=self.all_embeddings,
            all_tag_labels=self.all_tags
        )
        
        # Initialize fuser if pipe is available
        if self.pipe is not None:
            self.fuser = HITLCompositionFuser(
                pipe=self.pipe,
                device=getattr(self.pipe, 'device', 'cuda')
            )
        
        self.is_initialized = True
        self._save_state()
        
        print(f"[HITLSession] Initialization complete. GP seeded with {len(self.gp.preference_pairs)} pairs")
    
    def _seed_gp_from_concept_weights(self, concept_weights: Dict):
        """
        Seed GP with exploration utilities as initial observations.
        
        Creates synthetic pairwise comparisons based on utility ordering.
        """
        weights_list = concept_weights.get("concept_weights", [])
        
        if not weights_list:
            print("[HITLSession] No concept weights found, skipping GP seeding")
            return
        
        # Build utility lookup
        utility_by_label = {}
        for cw in weights_list:
            label = cw.get("label", "")
            utility = cw.get("utility", cw.get("score", cw.get("weight", 0.5)))
            utility_by_label[label] = utility
        
        # Match tags to utilities
        tag_utilities = []
        for i, tag in enumerate(self.all_tags):
            utility = utility_by_label.get(tag, 0.5)
            tag_utilities.append(utility)
        
        utilities = np.array(tag_utilities)
        
        # Create synthetic pairs from utility ordering
        pairs = create_synthetic_pairs_from_utilities(
            self.all_embeddings,
            utilities,
            self.all_tags,
            n_pairs_per_tag=3
        )
        
        if pairs:
            self.gp.add_preferences(pairs)
            self.gp.fit(n_epochs=50, verbose=False)
            print(f"[HITLSession] Seeded GP with {len(pairs)} synthetic pairs from exploration")
    
    def generate_round(self, q: int = 4) -> Tuple[List[CompositionSample], List[str]]:
        """
        Generate q compositions using UCB acquisition.
        
        Returns:
            (compositions, image_paths)
        """
        if not self.is_initialized:
            raise RuntimeError("Session not initialized. Call initialize_from_exploration first.")
        
        # Invalidate sampler cache after GP update
        self.sampler.invalidate_cache()
        
        # Sample compositions
        compositions = self.sampler.sample_batch(batch_size=q, n_points=10)
        
        # Generate images
        image_paths = self._generate_images(compositions)
        
        # Store in history
        self.compositions_history.append(compositions)
        
        self._save_state()
        
        print(f"[HITLSession] Generated round {self.round_count + 1} with {len(compositions)} compositions")
        
        return compositions, image_paths
    
    def _generate_images(self, compositions: List[CompositionSample]) -> List[str]:
        """Generate images for compositions."""
        image_paths = []
        
        for i, comp in enumerate(compositions):
            # Create image path
            img_filename = f"round_{self.round_count}_img_{i}.png"
            img_path = os.path.join(self.hitl_folder, img_filename)
            
            if self.fuser is not None and self.pipe is not None:
                # Full generation with cross-attention weighting
                try:
                    prompt_embeds, pooled, neg_embeds, neg_pooled, attn_controller = self.fuser.fuse_composition(
                        comp,
                        base_prompt=self.base_prompt,
                        neg_phrases=self.negative_phrases
                    )
                    
                    image = generate_with_hooks(
                        self.pipe,
                        prompt_embeds, pooled,
                        neg_embeds, neg_pooled,
                        attn_controller,  # Cross-attention scaling controller
                        seed=42 + i + self.round_count * 100
                    )
                    image.save(img_path)
                except Exception as e:
                    print(f"[HITLSession] Image generation failed: {e}")
                    # Create placeholder
                    self._create_placeholder_image(img_path, comp)
            elif self.sdxl_runner is not None:
                # Use sdxl_runner with text prompt
                try:
                    prompt = create_composition_prompt(comp, self.base_prompt)
                    image = self.sdxl_runner.generate(prompt=prompt)
                    image.save(img_path)
                except Exception as e:
                    print(f"[HITLSession] SDXL runner generation failed: {e}")
                    self._create_placeholder_image(img_path, comp)
            else:
                # No generation available, create placeholder
                self._create_placeholder_image(img_path, comp)
            
            image_paths.append(img_path)
        
        return image_paths
    
    def _create_placeholder_image(self, path: str, comp: CompositionSample):
        """Create a placeholder image with tag info."""
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', (512, 512), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        
        # Draw tag labels
        y = 20
        for i, (label, weight) in enumerate(zip(comp.tag_labels[:5], comp.weights[:5])):
            text = f"{label}: {weight:.2f}"
            draw.text((20, y), text, fill=(0, 0, 0))
            y += 30
        
        img.save(path)
    
    def record_ranking(self, ranking: List[int]) -> Dict:
        """
        Record ordinal ranking and update GP.
        
        Args:
            ranking: List of image indices in order of preference
                    e.g., [0, 2, 1, 3] means image 0 is best, then 2, then 1, then 3
        
        Returns:
            Dict with convergence metrics
        """
        if not self.compositions_history:
            raise RuntimeError("No compositions to rank. Call generate_round first.")
        
        current_compositions = self.compositions_history[-1]
        
        # Validate ranking
        if len(ranking) != len(current_compositions):
            raise ValueError(f"Ranking length {len(ranking)} != compositions {len(current_compositions)}")
        
        # Store ranking
        self.rankings_history.append(ranking)
        
        # Update optimizer with ranking
        result = self.optimizer.update_from_ranking(current_compositions, ranking)
        
        self.round_count += 1
        
        # Compute IMAGE variance (not GP variance)
        # This measures how similar the generated images are
        round_image_paths = [
            os.path.join(self.hitl_folder, f"round_{self.round_count - 1}_img_{i}.png")
            for i in range(len(current_compositions))
        ]
        image_variance = self.compute_round_image_variance(round_image_paths)
        
        # Store GP variance for reference
        gp_variance = result.get("gp_variance", 1.0)
        
        # Let user decide when to finalize - no automatic convergence
        # Only mark converged if max rounds reached
        self.is_converged = self.round_count >= self.max_rounds
        
        self._save_state()
        
        metrics = {
            "round": self.round_count,
            "image_variance": image_variance,  # New: image-based variance
            "gp_variance": gp_variance,        # Keep for reference
            "is_converged": self.is_converged,
            "total_pairs": result.get("total_pairs", 0),
            "mean_utility": result.get("mean_utility", 0.0),
        }
        
        print(f"[HITLSession] Recorded ranking for round {self.round_count}. Image variance: {image_variance:.4f}, GP variance: {gp_variance:.4f}, Converged: {self.is_converged}")
        
        return metrics
    
    def finalize(self) -> str:
        """
        Export refined preferences: the best 10 tags with attention weights.
        
        This outputs the FINAL curated tag set that represents user preference,
        with proper attention weights for cross-attention in image generation.
        
        Returns:
            Path to saved file
        """
        # Use sampler to get best 10 diverse tags (same logic as composition sampling)
        # This ensures we don't have redundant/similar tags
        if self.sampler is not None:
            composition = self.sampler.sample_composition(n_points=10)
            
            # Build output from the optimized composition
            concept_weights = []
            for i in range(len(composition.tag_labels)):
                concept_weights.append({
                    "concept_id": f"refined_{i}",
                    "label": composition.tag_labels[i],
                    # Cross-attention map weight: scales attn_scores for this tag's tokens
                    "attn_map_weight": float(composition.weights[i]),
                    "utility": float(composition.point_ucb_scores[i]),
                    "tag_index": int(composition.tag_indices[i]),
                })
            
            # Also compute total weight sum for verification (should be ~1.0)
            total_weight = sum(c["attn_map_weight"] for c in concept_weights)
            
        else:
            # Fallback: simple top-10 by utility
            mu, sigma = self.gp.predict_utility(self.all_embeddings)
            top_indices = np.argsort(mu)[-10:][::-1]
            
            # Compute softmax attention weights
            top_utilities = mu[top_indices]
            exp_u = np.exp(top_utilities - np.max(top_utilities))
            attention_weights = exp_u / (exp_u.sum() + 1e-8)
            
            concept_weights = []
            for i, idx in enumerate(top_indices):
                concept_weights.append({
                    "concept_id": f"refined_{i}",
                    "label": self.all_tags[idx],
                    "attn_map_weight": float(attention_weights[i]),
                    "utility": float(mu[idx]),
                    "uncertainty": float(sigma[idx]),
                })
            total_weight = 1.0
        
        output = {
            "stage": "hitl_refinement",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "rounds_completed": self.round_count,
            "total_pairs": len(self.optimizer.all_pairs) if self.optimizer else 0,
            "is_converged": self.is_converged,
            # The curated 10 tags with cross-attention map weights
            # These weights scale the attention scores for each tag's tokens
            # during the diffusion process: attn_scores[:, token_idx] *= weight
            "final_tags": concept_weights,
            "usage": "cross_attention_map_scaling",
            "description": "Each tag's weight scales its tokens' attention map contribution during image generation",
        }
        
        output_path = os.path.join(self.session_folder, "refined_preferences.json")
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Log the final selection
        print(f"[HITLSession] Finalized refinement after {self.round_count} rounds")
        print(f"[HITLSession] Final 10 tags with cross-attention map weights:")
        for c in concept_weights[:5]:
            print(f"  - {c['label']}: {c['attn_map_weight']:.3f}")
        print(f"  ... and {len(concept_weights) - 5} more")
        print(f"[HITLSession] Saved to {output_path}")
        
        return output_path
    
    def get_top_concepts(self, k: int = 5) -> List[Dict]:
        """Get top-K concepts by current GP utility."""
        if self.gp is None or self.all_embeddings is None:
            return []
        
        mu, sigma = self.gp.predict_utility(self.all_embeddings)
        top_indices = np.argsort(mu)[-k:][::-1]
        
        return [
            {
                "label": self.all_tags[idx],
                "utility": float(mu[idx]),
                "uncertainty": float(sigma[idx]),
            }
            for idx in top_indices
        ]
    
    # ============== Image Variance Tracking ==============
    
    def _load_clip_for_images(self):
        """Load CLIP model for computing image embeddings."""
        if self._clip_model is not None and self._clip_preprocess is not None:
            return
        
        try:
            import clip
            self._clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-L/14", device=self._clip_device)
            model.eval()
            self._clip_model = model
            self._clip_preprocess = preprocess
            print(f"[HITLSession] Loaded CLIP ViT-L/14 for image embeddings on {self._clip_device}")
        except Exception as e:
            print(f"[HITLSession] Error loading CLIP: {e}")
            raise
    
    def _compute_image_embedding(self, image_path: str) -> np.ndarray:
        """Compute CLIP embedding for a single image."""
        from PIL import Image
        
        self._load_clip_for_images()
        
        image = Image.open(image_path).convert('RGB')
        image_input = self._clip_preprocess(image).unsqueeze(0).to(self._clip_device)
        
        with torch.no_grad():
            features = self._clip_model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu().numpy()[0].astype(np.float32)
    
    def compute_round_image_variance(self, image_paths: List[str]) -> float:
        """
        Compute variance of image embeddings for a round.
        
        Lower variance = images are more similar = preferences converging
        """
        embeddings = []
        for path in image_paths:
            if os.path.exists(path):
                emb = self._compute_image_embedding(path)
                embeddings.append(emb)
        
        if len(embeddings) < 2:
            return 1.0  # High variance if not enough images
        
        embeddings = np.array(embeddings)
        
        # Compute pairwise cosine distances
        from scipy.spatial.distance import pdist
        distances = pdist(embeddings, metric='cosine')
        
        # Return mean cosine distance as variance measure
        # Low distance = similar images = converging
        return float(np.mean(distances))
    
    # ============== Persistence for Idempotency ==============
    
    def _save_state(self):
        """Save session state for browser refresh recovery."""
        # Serialize compositions history (10 tags + weights per image per round)
        compositions_serialized = []
        for round_compositions in self.compositions_history:
            round_data = []
            for comp in round_compositions:
                round_data.append({
                    "tag_labels": comp.tag_labels,
                    "weights": comp.weights.tolist(),
                    "tag_indices": comp.tag_indices,
                    "ucb_scores": comp.point_ucb_scores.tolist() if comp.point_ucb_scores is not None else []
                })
            compositions_serialized.append(round_data)
        
        state = {
            "session_id": self.session_id,
            "round_count": self.round_count,
            "is_initialized": self.is_initialized,
            "is_converged": self.is_converged,
            "rankings_history": self.rankings_history,
            "compositions_history": compositions_serialized,  # 10 tags + weights per image
            "base_prompt": self.base_prompt,
            "negative_phrases": self.negative_phrases,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _restore_from_state(self, state_file: str):
        """Restore session state from file."""
        with open(state_file) as f:
            state = json.load(f)
        
        self.round_count = state.get("round_count", 0)
        self.rankings_history = state.get("rankings_history", [])
        self.is_converged = state.get("is_converged", False)
        self.base_prompt = state.get("base_prompt", self.base_prompt)
        self.negative_phrases = state.get("negative_phrases", self.negative_phrases)
        
        # Re-initialize from exploration
        self.initialize_from_exploration()
        
        # Replay rankings to rebuild GP state
        for i, ranking in enumerate(self.rankings_history):
            if i < len(self.compositions_history):
                compositions = self.compositions_history[i]
            else:
                compositions = self.sampler.sample_batch(batch_size=4, n_points=10)
                self.compositions_history.append(compositions)
            
            self.optimizer.update_from_ranking(compositions, ranking, refit=False)
        
        # Final GP fit
        if self.optimizer.all_pairs:
            self.optimizer._fit_gp()
        
        print(f"[HITLSession] Restored from state. Round: {self.round_count}, Rankings: {len(self.rankings_history)}")
    
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
        """
        Idempotent initialization: reload existing state if available.
        
        If hitl_state.json exists, restore session state.
        Otherwise, initialize fresh from exploration.
        """
        session = cls(
            session_id=session_id,
            session_folder=session_folder,
            sdxl_runner=sdxl_runner,
            pipe=pipe,
            base_prompt=base_prompt,
            negative_phrases=negative_phrases
        )
        
        state_file = os.path.join(session_folder, "hitl_state.json")
        
        if os.path.exists(state_file):
            print(f"[HITLSession] Found existing state, restoring...")
            session._restore_from_state(state_file)
        else:
            print(f"[HITLSession] No existing state, initializing fresh...")
            session.initialize_from_exploration()
        
        return session
    
    def get_status(self) -> Dict:
        """Get current session status."""
        return {
            "session_id": self.session_id,
            "is_initialized": self.is_initialized,
            "is_converged": self.is_converged,
            "round_count": self.round_count,
            "max_rounds": self.max_rounds,
            "total_tags": len(self.all_tags),
            "positive_tags": len(self.positive_tags),
            "negative_tags": len(self.negative_tags),
            "total_pairs": len(self.optimizer.all_pairs) if self.optimizer else 0,
            "rankings_recorded": len(self.rankings_history),
        }
