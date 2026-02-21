"""
HITL Refinement Session V2 - Tag-Level GP System

Uses the new TagGPRefiner for interpretable tag-level preference learning.

Key differences from V1:
- Each tag has individual μ (utility) and σ (uncertainty)
- 4 diverse option strategies per round (exploit, explore, UCB, challenger)
- Direct tag-level updates from pairwise comparisons
- Built-in logging for debugging
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

# Optional torch import (for CLIP image variance)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

# Import the new tag-level GP refiner
from tag_gp_refiner import TagGPRefiner, GPRefinerConfig, RefinementOption, TagCategory

# Image generation - use same approach as V1
try:
    from hitl_fuser import HITLCompositionFuser, generate_with_hooks
    from hitl_sampler import CompositionSample
    HAS_FUSER = True
except ImportError:
    HAS_FUSER = False
    CompositionSample = None
    print("[HITLSessionV2] Warning: hitl_fuser not available")


@dataclass
class CompositionV2:
    """Composition data for a single option."""
    option_id: int
    strategy: str
    tag_labels: List[str]
    tag_ids: List[str]
    weights: List[float]
    mus: List[float]
    sigmas: List[float]
    image_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "option_id": self.option_id,
            "strategy": self.strategy,
            "tag_labels": self.tag_labels,
            "weights": [round(w, 4) for w in self.weights],
            "mus": [round(m, 4) for m in self.mus],
            "sigmas": [round(s, 4) for s in self.sigmas],
            "image_path": self.image_path,
        }


class HITLRefinementSessionV2:
    """
    HITL refinement using tag-level GP.
    
    Each tag maintains:
    - μ (mu): Expected utility
    - σ (sigma): Uncertainty
    
    Per round:
    1. Generate 4 options with different strategies
    2. Generate images using cross-attention weighting
    3. User ranks options 1st to 4th
    4. Update tag utilities based on rankings
    """
    
    def __init__(
        self,
        session_id: str,
        session_folder: str,
        pipe: Any = None,
        base_prompt: str = "",
        negative_prompt: str = "",
        max_rounds: int = 999,
        tags_per_option: int = 10,
        image_size: Tuple[int, int] = (1024, 1024),
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ):
        self.session_id = session_id
        self.session_folder = Path(session_folder)
        self.pipe = pipe
        self.base_prompt = base_prompt
        self.negative_prompt = negative_prompt
        self.max_rounds = max_rounds
        self.tags_per_option = tags_per_option
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        # Create folders
        self.hitl_folder = self.session_folder / "hitl"
        self.hitl_folder.mkdir(parents=True, exist_ok=True)
        
        # State file
        self.state_file = self.session_folder / "hitl_state_v2.json"
        self.round_diagnostics_file = self.session_folder / "gp_round_diagnostics_v2.jsonl"
        self.generation_counter = 0
        
        # Tag-level GP refiner
        self.refiner: Optional[TagGPRefiner] = None
        
        # Fuser for image generation
        self.fuser: Optional[HITLCompositionFuser] = None
        
        # Tag data from exploration
        self.positive_tags: List[str] = []
        self.neutral_tags: List[str] = []
        self.negative_tags: List[str] = []
        self.selected_image_tags: Set[str] = set()
        
        # State
        self.round_count = 0
        self.is_initialized = False
        self.is_converged = False
        self.compositions_history: List[List[CompositionV2]] = []
        self.rankings_history: List[List[int]] = []
        
        # Per-round tag state snapshots for rollback
        # key: round_number (after ranking), value: dict of tag_id -> {mu, sigma, ...}
        self.tag_states_snapshots: Dict[int, Dict] = {}
        
        # Best pick (1st ranked image) per round for gallery
        # key: round_number, value: {url, image_path, option_id, tags, weights}
        self.best_picks: Dict[int, Dict] = {}
        
        # CLIP for image variance (lazy loaded)
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_device = None
    
    def _load_json(self, filename: str) -> Dict:
        """Load JSON from session folder or impression subfolder."""
        impression_path = self.session_folder / "impression" / filename
        if impression_path.exists():
            with open(impression_path) as f:
                return json.load(f)
        
        direct_path = self.session_folder / filename
        if direct_path.exists():
            with open(direct_path) as f:
                return json.load(f)
        
        raise FileNotFoundError(f"Cannot find {filename} in {self.session_folder}")
    
    def initialize_from_exploration(self) -> Dict:
        """
        Initialize tag-level GP from exploration outputs.
        
        Loads:
        - tag_preferences.json: positive/neutral/negative tags
        - visual_tags.json (optional): tags from selected images
        """
        print(f"[HITLSessionV2] Initializing from exploration: {self.session_id}")
        
        # Load tag preferences
        tag_prefs = self._load_json("tag_preferences.json")
        
        self.positive_tags = tag_prefs.get("positive", [])
        self.neutral_tags = tag_prefs.get("neutral", [])
        self.negative_tags = tag_prefs.get("negative", [])
        
        # Deduplicate (case-insensitive)
        self.positive_tags = self._deduplicate_tags(self.positive_tags)
        self.neutral_tags = self._deduplicate_tags(self.neutral_tags)
        
        # Remove any overlap between positive and neutral
        positive_set = set(t.lower() for t in self.positive_tags)
        self.neutral_tags = [t for t in self.neutral_tags if t.lower() not in positive_set]
        
        # Try to load selected image tags
        try:
            visual_tags = self._load_json("visual_tags.json")
            # Tags from images user selected
            for img_data in visual_tags.get("selected_images", []):
                self.selected_image_tags.update(img_data.get("tags", []))
        except FileNotFoundError:
            # No visual tags, use positive tags as selected
            self.selected_image_tags = set(self.positive_tags)
        
        print(f"[HITLSessionV2] Tags: {len(self.positive_tags)} positive, "
              f"{len(self.neutral_tags)} neutral, {len(self.negative_tags)} negative")
        print(f"[HITLSessionV2] Selected image tags: {len(self.selected_image_tags)}")
        
        # Create config
        config = GPRefinerConfig(
            max_rounds=self.max_rounds,
            tags_per_option=self.tags_per_option,
        )
        
        # Create refiner
        self.refiner = TagGPRefiner(config)
        
        # Set up logger
        self.refiner.set_logger(self.session_id, self.session_folder)
        
        # Initialize from exploration
        result = self.refiner.initialize_from_exploration(
            positive_tags=self.positive_tags,
            neutral_tags=self.neutral_tags,
            selected_image_tags=self.selected_image_tags,
        )
        
        # Initialize fuser if pipe available
        if self.pipe is not None and HAS_FUSER:
            self.fuser = HITLCompositionFuser(
                pipe=self.pipe,
                device=getattr(self.pipe, 'device', 'cuda')
            )
            print("[HITLSessionV2] Fuser initialized for cross-attention weighting")
        
        self.is_initialized = True
        self._save_state()
        
        return {
            "status": "initialized",
            "positive_tags": len(self.positive_tags),
            "neutral_tags": len(self.neutral_tags),
            "total_tags": result["total_tags"],
            "categories": result["categories"],
        }
    
    def _deduplicate_tags(self, tags: List[str]) -> List[str]:
        """Remove duplicate tags (case-insensitive)."""
        seen = set()
        result = []
        for tag in tags:
            key = tag.lower().strip()
            if key not in seen:
                seen.add(key)
                result.append(tag)
        return result
    
    def generate_round(self) -> Tuple[List[CompositionV2], List[str]]:
        """
        Generate 4 options for the current round.
        
        Returns:
            (compositions, image_paths)
        """
        if not self.is_initialized:
            raise RuntimeError("Session not initialized")
        
        # No hard round limit - user decides when to stop via "Finish" button
        
        # Generate options using tag-level GP
        options = self.refiner.generate_round_options()
        self.generation_counter += 1
        
        # Convert to CompositionV2 and compute weights
        compositions = []
        for opt in options:
            tag_states = [self.refiner.tags[tid] for tid in opt.tag_ids]
            
            # Compute attention weights from tag utilities
            weights = self._compute_attention_weights(tag_states)
            
            comp = CompositionV2(
                option_id=opt.option_id,
                strategy=opt.strategy.value,
                tag_labels=opt.tags,
                tag_ids=opt.tag_ids,
                weights=weights,
                mus=[t.mu for t in tag_states],
                sigmas=[t.sigma for t in tag_states],
            )
            compositions.append(comp)
        
        # Generate images
        image_paths = self._generate_images(compositions)
        
        # Update compositions with image paths
        for comp, path in zip(compositions, image_paths):
            comp.image_path = path
        
        # Store history
        self.compositions_history.append(compositions)
        self._save_state()
        
        print(f"[HITLSessionV2] Generated round {self.refiner.current_round}")
        for comp in compositions:
            print(f"  Option {comp.option_id} [{comp.strategy:10s}]: "
                  f"{comp.tag_labels[:3]}...")
        
        return compositions, image_paths
    
    def _compute_attention_weights(self, tag_states) -> List[float]:
        """
        Compute cross-attention weights from tag utilities.
        
        Uses softmax over μ values, scaled to [0.5, 1.5] range.
        """
        mus = np.array([t.mu for t in tag_states])
        
        # Softmax
        exp_mu = np.exp(mus - np.max(mus))
        softmax = exp_mu / (exp_mu.sum() + 1e-8)
        
        # Scale to [0.5, 1.5] range for cross-attention
        # Higher μ = higher weight = more influence on generation
        weights = 0.5 + softmax * len(mus)  # Roughly centers around 1.0
        
        # Normalize to sum to len(tags) (average weight = 1.0)
        weights = weights * (len(mus) / weights.sum())
        
        return weights.tolist()
    
    def _generate_images(self, compositions: List[CompositionV2]) -> List[str]:
        """Generate images for compositions using cross-attention weighting.
        
        Uses the same approach as V1 HITLRefinementSession:
        1. Convert CompositionV2 to CompositionSample
        2. Use fuser.fuse_composition() to get prompt embeddings + attention controller
        3. Use generate_with_hooks() to run generation with attention scaling
        """
        image_paths = []
        round_num = self.refiner.current_round
        neg_phrases = self.negative_prompt.split(", ") if self.negative_prompt else []
        
        for i, comp in enumerate(compositions):
            img_filename = f"round_{round_num - 1}_img_{i}.png"
            img_path = self.hitl_folder / img_filename
            
            if self.fuser is not None and self.pipe is not None and HAS_FUSER:
                try:
                    # Convert CompositionV2 to CompositionSample (same format as V1)
                    comp_sample = CompositionSample(
                        points=np.zeros((len(comp.tag_labels), 768)),  # Placeholder - not used in generation
                        weights=np.array(comp.weights),
                        tag_labels=comp.tag_labels,
                        tag_indices=list(range(len(comp.tag_labels))),
                        point_ucb_scores=np.array(comp.mus) if comp.mus else None,
                    )
                    
                    # Use same approach as V1: fuse_composition + generate_with_hooks
                    prompt_embeds, pooled, neg_embeds, neg_pooled, attn_controller = self.fuser.fuse_composition(
                        comp_sample,
                        base_prompt=self.base_prompt,
                        neg_phrases=neg_phrases
                    )
                    
                    seed = 42 + i + round_num * 100 + self.generation_counter * 10000
                    image = generate_with_hooks(
                        self.pipe,
                        prompt_embeds, pooled,
                        neg_embeds, neg_pooled,
                        attn_controller,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        height=self.image_size[0],
                        width=self.image_size[1],
                        seed=seed
                    )
                    image.save(img_path)
                except Exception as e:
                    print(f"[HITLSessionV2] Image generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    self._create_placeholder_image(img_path, comp)
            else:
                self._create_placeholder_image(img_path, comp)
            
            image_paths.append(str(img_path))
        
        return image_paths
    
    def _create_placeholder_image(self, path: Path, comp: CompositionV2):
        """Create placeholder image with tag info."""
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('RGB', (512, 512), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Title
        draw.text((20, 10), f"Option {comp.option_id} ({comp.strategy})", fill=(0, 0, 0))
        
        # Draw tags with weights
        y = 50
        for i, (tag, weight, mu) in enumerate(zip(comp.tag_labels[:8], comp.weights[:8], comp.mus[:8])):
            text = f"{tag[:25]:25s} w={weight:.2f} μ={mu:+.2f}"
            draw.text((20, y), text, fill=(50, 50, 50))
            y += 25
        
        if len(comp.tag_labels) > 8:
            draw.text((20, y), f"... and {len(comp.tag_labels) - 8} more", fill=(100, 100, 100))
        
        img.save(path)
    
    def record_ranking(self, ranking: List[int]) -> Dict:
        """
        Record user ranking and update tag utilities.
        
        Args:
            ranking: [best_option_id, 2nd_best, 3rd, worst]
                    e.g., [2, 0, 3, 1] means option 2 is best
        
        Returns:
            Metrics dictionary
        """
        if not self.compositions_history:
            raise RuntimeError("No compositions to rank")
        
        # Submit ranking to refiner
        result = self.refiner.submit_ranking(ranking)
        
        # Store ranking
        self.rankings_history.append(ranking)
        self.round_count = self.refiner.current_round
        
        # Save tag state snapshot for rollback
        self._save_tag_snapshot(self.round_count)
        
        # Save best pick (1st ranked image)
        self._save_best_pick(self.round_count, ranking)
        
        # Compute image variance
        current_compositions = self.compositions_history[-1]
        image_paths = [comp.image_path for comp in current_compositions if comp.image_path]
        image_variance = self._compute_image_variance(image_paths) if image_paths else 1.0
        
        # User decides when to stop - no auto-convergence
        self.is_converged = False
        
        self._save_state()
        
        # Build metrics
        metrics = {
            "round": self.round_count,
            "beta": round(self.refiner.beta, 3),
            "learning_rate": round(self.refiner.learning_rate, 4),
            "image_variance": round(image_variance, 4),
            "is_converged": self.is_converged,
            "comparisons_made": len(result.pairwise_comparisons),
            "tags_updated": len(result.tag_updates),
            "pairwise_accuracy_before_update": round(result.diagnostics.get("pairwise_accuracy_before_update", 0.0), 4),
            "ranking_log_likelihood_before_update": round(result.diagnostics.get("ranking_log_likelihood_before_update", 0.0), 4),
            "mean_pair_margin_before_update": round(result.diagnostics.get("mean_pair_margin_before_update", 0.0), 4),
            "predicted_top_option_before_update": int(result.diagnostics.get("predicted_top_option_before_update", -1)),
            "actual_top_option": int(result.diagnostics.get("actual_top_option", -1)),
            "spearman_rank_corr_before_update": round(result.diagnostics.get("spearman_rank_corr_before_update", 0.0), 4),
            "kendall_pair_agreement_before_update": round(result.diagnostics.get("kendall_pair_agreement_before_update", 0.0), 4),
            "predicted_ranking_before_update": [
                int(x) for x in result.diagnostics.get("predicted_ranking_before_update", [])
            ],
            "top_5_tags": [
                {"tag": t.text, "mu": round(t.mu, 3), "sigma": round(t.sigma, 3)}
                for t in sorted(self.refiner.tags.values(), key=lambda x: x.mu, reverse=True)[:5]
            ],
        }

        # Persist round diagnostics for post-hoc analysis across sessions.
        self._append_round_diagnostics(metrics, ranking, result)
        
        print(f"[HITLSessionV2] Recorded ranking for round {self.round_count}")
        print(f"  β={metrics['beta']}, tags_updated={metrics['tags_updated']}")
        
        return metrics

    def _append_round_diagnostics(self, metrics: Dict, ranking: List[int], result: Any) -> None:
        """Append one JSONL record per round with learning diagnostics."""
        try:
            option_snapshots = []
            for opt in self.refiner.current_options:
                tag_states = [self.refiner.tags[tid] for tid in opt.tag_ids if tid in self.refiner.tags]
                mu_mean = float(np.mean([t.mu for t in tag_states])) if tag_states else 0.0
                sigma_mean = float(np.mean([t.sigma for t in tag_states])) if tag_states else 0.0
                option_snapshots.append({
                    "option_id": int(opt.option_id),
                    "strategy": str(opt.strategy.value),
                    "mu_mean": round(mu_mean, 6),
                    "sigma_mean": round(sigma_mean, 6),
                    "n_tags": len(opt.tag_ids),
                })

            payload = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "round": int(self.round_count),
                "ranking": [int(r) for r in ranking],
                "pairwise_comparisons": [[int(a), int(b)] for a, b in result.pairwise_comparisons],
                "metrics": metrics,
                "diagnostics": result.diagnostics,
                "option_snapshots": option_snapshots,
                "top_10_tags_after_update": [
                    {"tag": t.text, "mu": round(float(t.mu), 6), "sigma": round(float(t.sigma), 6)}
                    for t in sorted(self.refiner.tags.values(), key=lambda x: x.mu, reverse=True)[:10]
                ],
            }

            with open(self.round_diagnostics_file, "a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            print(f"[HITLSessionV2] Warning: failed to append round diagnostics: {e}")
    
    def _save_tag_snapshot(self, round_num: int) -> None:
        """Save a snapshot of all tag states for potential rollback."""
        if not self.refiner:
            return
        
        snapshot = {}
        for tag_id, tag in self.refiner.tags.items():
            snapshot[tag_id] = {
                "text": tag.text,
                "category": tag.category.value,
                "mu": tag.mu,
                "sigma": tag.sigma,
                "times_shown": tag.times_shown,
                "times_in_winner": tag.times_in_winner,
                "times_in_loser": tag.times_in_loser,
            }
        self.tag_states_snapshots[round_num] = snapshot
    
    def _save_best_pick(self, round_num: int, ranking: List[int]) -> None:
        """Save the 1st-ranked image info for the best-of gallery."""
        if not self.compositions_history:
            return
        
        current_compositions = self.compositions_history[-1]
        best_option_id = ranking[0]
        
        # Find the composition for the winning option
        best_comp = None
        for comp in current_compositions:
            if comp.option_id == best_option_id:
                best_comp = comp
                break
        
        if best_comp is None and best_option_id < len(current_compositions):
            best_comp = current_compositions[best_option_id]
        
        if best_comp:
            self.best_picks[round_num] = {
                "round": round_num,
                "option_id": int(best_comp.option_id),
                "image_path": best_comp.image_path,
                "tags": list(best_comp.tag_labels),
                "weights": [float(w) for w in best_comp.weights],
                "strategy": str(best_comp.strategy),
            }
    
    def _compute_image_variance(self, image_paths: List[str]) -> float:
        """Compute variance of images using CLIP embeddings."""
        if len(image_paths) < 2:
            return 1.0
        
        if not HAS_TORCH:
            # Cannot compute without torch
            return 1.0
        
        try:
            self._load_clip()
            
            from PIL import Image
            embeddings = []
            
            for path in image_paths:
                if os.path.exists(path):
                    image = Image.open(path).convert('RGB')
                    image_input = self._clip_preprocess(image).unsqueeze(0).to(self._clip_device)
                    
                    with torch.no_grad():
                        features = self._clip_model.encode_image(image_input)
                        features = features / features.norm(dim=-1, keepdim=True)
                    
                    embeddings.append(features.cpu().numpy()[0])
            
            if len(embeddings) < 2:
                return 1.0
            
            embeddings = np.array(embeddings)
            
            # Compute pairwise cosine distances
            from scipy.spatial.distance import pdist
            distances = pdist(embeddings, metric='cosine')
            
            return float(np.mean(distances))
            
        except Exception as e:
            print(f"[HITLSessionV2] Image variance computation failed: {e}")
            return 1.0
    
    def _load_clip(self):
        """Lazy load CLIP model."""
        if not HAS_TORCH:
            raise RuntimeError("torch not available for CLIP")
        
        if self._clip_model is None:
            import clip
            self._clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model, self._clip_preprocess = clip.load("ViT-L/14", device=self._clip_device)
            self._clip_model.eval()
    
    def finalize(self) -> Dict:
        """
        Finalize refinement and export best tags with weights.
        
        Returns:
            Final selection dictionary
        """
        if self.refiner is None:
            raise RuntimeError("Session not initialized")
        
        # Get final tags and weights from refiner
        final_result = self.refiner.get_final_result()
        
        # Build output
        final_tags = final_result["final_tags"]
        weights = final_result["weights"]
        
        output = {
            "stage": "hitl_refinement_v2",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "rounds_completed": self.round_count,
            "is_converged": self.is_converged,
            "tags": final_tags,
            "weights": weights,
            "tag_details": [
                {
                    "rank": i + 1,
                    "tag": tag,
                    "weight": weights.get(tag, 0),
                }
                for i, tag in enumerate(final_tags)
            ],
            "all_tag_details": final_result.get("all_tag_details", []),
        }
        
        # Save V2 format
        output_path = self.session_folder / "refined_preferences_v2.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Also save compatibility format (refined_preferences.json)
        # with label/attn_map_weight keys matching the old GP refinement format
        compat_output = {
            "stage": "hitl_refinement_v2",
            "final_tags": [
                {
                    "label": tag,
                    "attn_map_weight": weights.get(tag, 0),
                }
                for tag in final_tags
            ],
        }
        compat_path = self.session_folder / "refined_preferences.json"
        with open(compat_path, 'w') as f:
            json.dump(compat_output, f, indent=2)
        
        print(f"[HITLSessionV2] Finalized after {self.round_count} rounds")
        print(f"[HITLSessionV2] Top 5 tags:")
        for i, tag in enumerate(final_tags[:5]):
            print(f"  {i+1}. {tag}: {weights.get(tag, 0):.3f}")
        print(f"[HITLSessionV2] Saved refined_preferences_v2.json and refined_preferences.json")
        
        return output
    
    def rollback_to_round(self, target_round: int) -> Dict:
        """
        Roll back to a previous round's tag states.
        
        The user preferred round X's winner over everything since.
        
        Strategy:
        1. Restore tag states from round X's snapshot
        2. Inject bonus: tags in round X's winner get a μ boost,
           tags in current round's best get a μ penalty
        3. Resume round counter from target_round
        4. Truncate history to target_round
        
        Args:
            target_round: The round number to roll back to
            
        Returns:
            Rollback summary
        """
        if not self.refiner:
            raise RuntimeError("Session not initialized")
        
        if target_round not in self.tag_states_snapshots:
            raise ValueError(f"No snapshot for round {target_round}. Available: {list(self.tag_states_snapshots.keys())}")
        
        print(f"[HITLSessionV2] Rolling back from round {self.round_count} to round {target_round}")
        
        # Capture currently displayed options before state restore/truncation.
        # These are treated as less preferred than the chosen previous best.
        current_displayed_options = list(self.refiner.current_options) if self.refiner.current_options else []

        # Get the winning tags from the target round
        target_best = self.best_picks.get(target_round, {})
        target_best_tags = set(target_best.get("tags", []))
        
        # Get the winning tags from the current/latest round
        current_best_tags = set()
        if self.best_picks:
            latest_round = max(self.best_picks.keys())
            current_best = self.best_picks.get(latest_round, {})
            current_best_tags = set(current_best.get("tags", []))
        
        # 1. Restore tag states from snapshot
        snapshot = self.tag_states_snapshots[target_round]
        for tag_id, tag_data in snapshot.items():
            if tag_id in self.refiner.tags:
                tag = self.refiner.tags[tag_id]
                tag.mu = tag_data["mu"]
                tag.sigma = tag_data["sigma"]
                tag.times_shown = tag_data.get("times_shown", 0)
                tag.times_in_winner = tag_data.get("times_in_winner", 0)
                tag.times_in_loser = tag_data.get("times_in_loser", 0)
        
        # 2. Inject constraints from rollback intent:
        # chosen previous winner > each currently displayed option
        lr = self.refiner.learning_rate
        tags_boosted = []
        tags_penalized = []
        applied_pair_constraints = 0

        # Fast text -> tag_id lookup
        text_to_tag_id = {}
        for tag_id, tag in self.refiner.tags.items():
            if tag.text not in text_to_tag_id:
                text_to_tag_id[tag.text] = tag_id

        target_best_ids = {text_to_tag_id[t] for t in target_best_tags if t in text_to_tag_id}

        if target_best_ids and current_displayed_options:
            constraint_strength = 0.8
            sigma_shrink = 0.92

            for opt in current_displayed_options:
                worse_ids = set(opt.tag_ids)
                only_in_best = target_best_ids - worse_ids
                only_in_worse = worse_ids - target_best_ids

                if not only_in_best and not only_in_worse:
                    continue

                applied_pair_constraints += 1

                for tag_id in only_in_best:
                    tag = self.refiner.tags.get(tag_id)
                    if not tag:
                        continue
                    tag.mu += lr * constraint_strength
                    tag.sigma = max(self.refiner.config.min_sigma, tag.sigma * sigma_shrink)
                    tags_boosted.append(tag.text)

                for tag_id in only_in_worse:
                    tag = self.refiner.tags.get(tag_id)
                    if not tag:
                        continue
                    tag.mu -= lr * constraint_strength
                    tags_penalized.append(tag.text)
                    # Keep uncertainty slightly high for "rejected now" tags so they can recover if needed.
                    tag.sigma = max(self.refiner.config.min_sigma, tag.sigma * 0.98)
        
        # Keep a weaker fallback comparison (target winner > latest winner)
        # for cases where we cannot map/construct pair constraints.
        if applied_pair_constraints == 0:
            for tag_id, tag in self.refiner.tags.items():
                in_target_winner = tag.text in target_best_tags
                in_current_winner = tag.text in current_best_tags
                
                if in_target_winner and not in_current_winner:
                    tag.mu += lr * 1.0
                    tag.sigma *= 0.9
                    tag.sigma = max(tag.sigma, self.refiner.config.min_sigma)
                    tags_boosted.append(tag.text)
                elif in_current_winner and not in_target_winner:
                    tag.mu -= lr * 0.5
                    tags_penalized.append(tag.text)
        
        # 3. Resume round counter from target round
        old_round = self.round_count
        self.refiner.current_round = target_round
        self.round_count = target_round
        
        # 4. Truncate history to target_round
        self.compositions_history = self.compositions_history[:target_round]
        self.rankings_history = self.rankings_history[:target_round]
        
        # Remove snapshots and best picks after target round
        self.tag_states_snapshots = {
            k: v for k, v in self.tag_states_snapshots.items()
            if k <= target_round
        }
        self.best_picks = {
            k: v for k, v in self.best_picks.items()
            if k <= target_round
        }
        
        self.is_converged = False
        self._save_state()
        
        result = {
            "status": "rolled_back",
            "from_round": old_round,
            "to_round": target_round,
            "applied_pair_constraints": applied_pair_constraints,
            "tags_boosted": tags_boosted,
            "tags_penalized": tags_penalized,
            "remaining_best_picks": list(self.best_picks.keys()),
        }
        
        print(f"[HITLSessionV2] Rollback complete: {old_round} -> {target_round}")
        print(
            f"  Applied {applied_pair_constraints} pair constraints, "
            f"boosted {len(tags_boosted)} tags, penalized {len(tags_penalized)} tags"
        )
        
        # Log rollback event
        if self.refiner and self.refiner.logger:
            self.refiner.logger._add_warning("rollback",
                f"Rolled back from round {old_round} to round {target_round}. "
                f"Boosted: {tags_boosted[:3]}, Penalized: {tags_penalized[:3]}")
        
        return result
    
    def get_best_picks_list(self) -> List[Dict]:
        """Get best picks for the frontend gallery."""
        picks = []
        for round_num in sorted(self.best_picks.keys()):
            pick = self.best_picks[round_num]
            picks.append({
                "round": round_num,
                "image_path": pick.get("image_path"),
                "tags": pick.get("tags", [])[:5],
                "strategy": pick.get("strategy", ""),
            })
        return picks
    
    def get_status(self) -> Dict:
        """Get current session status."""
        status = {
            "session_id": self.session_id,
            "is_initialized": self.is_initialized,
            "is_converged": self.is_converged,
            "round_count": self.round_count,
            "max_rounds": self.max_rounds,
            "rankings_recorded": len(self.rankings_history),
        }
        
        if self.refiner:
            status["beta"] = round(self.refiner.beta, 3)
            status["total_tags"] = len(self.refiner.tags)
            status["top_5_tags"] = [
                {"tag": t.text, "mu": round(t.mu, 3)}
                for t in sorted(self.refiner.tags.values(), key=lambda x: x.mu, reverse=True)[:5]
            ]
        
        return status
    
    # ============== Persistence ==============
    
    def _save_state(self):
        """Save session state for recovery."""
        compositions_serialized = []
        for round_comps in self.compositions_history:
            round_data = [comp.to_dict() for comp in round_comps]
            compositions_serialized.append(round_data)
        
        # Save tag states
        tag_states = {}
        if self.refiner:
            for tag_id, tag in self.refiner.tags.items():
                tag_states[tag_id] = {
                    "text": tag.text,
                    "category": tag.category.value,
                    "mu": round(tag.mu, 6),
                    "sigma": round(tag.sigma, 6),
                    "times_shown": tag.times_shown,
                    "times_in_winner": tag.times_in_winner,
                    "times_in_loser": tag.times_in_loser,
                }
        
        # Serialize snapshots and best_picks (round keys as strings for JSON)
        snapshots_serialized = {
            str(k): v for k, v in self.tag_states_snapshots.items()
        }
        best_picks_serialized = {
            str(k): v for k, v in self.best_picks.items()
        }
        
        state = {
            "version": "2.0",
            "session_id": self.session_id,
            "round_count": self.round_count,
            "generation_counter": self.generation_counter,
            "is_initialized": self.is_initialized,
            "is_converged": self.is_converged,
            "rankings_history": self.rankings_history,
            "compositions_history": compositions_serialized,
            "tag_states": tag_states,
            "tag_states_snapshots": snapshots_serialized,
            "best_picks": best_picks_serialized,
            "positive_tags": self.positive_tags,
            "neutral_tags": self.neutral_tags,
            "selected_image_tags": list(self.selected_image_tags),
            "base_prompt": self.base_prompt,
            "negative_prompt": self.negative_prompt,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _restore_from_state(self):
        """Restore session from saved state."""
        if not self.state_file.exists():
            return False
        
        with open(self.state_file) as f:
            state = json.load(f)
        
        # Check version
        if state.get("version") != "2.0":
            print("[HITLSessionV2] State file is from V1, cannot restore")
            return False
        
        # Restore basic state
        self.round_count = state.get("round_count", 0)
        self.generation_counter = state.get("generation_counter", 0)
        self.is_initialized = state.get("is_initialized", False)
        self.is_converged = state.get("is_converged", False)
        self.rankings_history = state.get("rankings_history", [])
        self.positive_tags = state.get("positive_tags", [])
        self.neutral_tags = state.get("neutral_tags", [])
        self.selected_image_tags = set(state.get("selected_image_tags", []))
        self.base_prompt = state.get("base_prompt", self.base_prompt)
        self.negative_prompt = state.get("negative_prompt", self.negative_prompt)
        
        # Restore compositions history
        self.compositions_history = []
        for round_data in state.get("compositions_history", []):
            round_comps = []
            for comp_data in round_data:
                comp = CompositionV2(
                    option_id=comp_data["option_id"],
                    strategy=comp_data["strategy"],
                    tag_labels=comp_data["tag_labels"],
                    tag_ids=comp_data.get("tag_ids", []),
                    weights=comp_data["weights"],
                    mus=comp_data.get("mus", []),
                    sigmas=comp_data.get("sigmas", []),
                    image_path=comp_data.get("image_path"),
                )
                round_comps.append(comp)
            self.compositions_history.append(round_comps)
        
        # Restore refiner with tag states
        if state.get("tag_states"):
            config = GPRefinerConfig(
                max_rounds=self.max_rounds,
                tags_per_option=self.tags_per_option,
            )
            self.refiner = TagGPRefiner(config)
            self.refiner.set_logger(self.session_id, self.session_folder)
            
            # Initialize and then override with saved states
            self.refiner.initialize_from_exploration(
                positive_tags=self.positive_tags,
                neutral_tags=self.neutral_tags,
                selected_image_tags=self.selected_image_tags,
            )
            
            # Override tag states
            for tag_id, tag_data in state["tag_states"].items():
                if tag_id in self.refiner.tags:
                    tag = self.refiner.tags[tag_id]
                    tag.mu = tag_data["mu"]
                    tag.sigma = tag_data["sigma"]
                    tag.times_shown = tag_data.get("times_shown", 0)
                    tag.times_in_winner = tag_data.get("times_in_winner", 0)
                    tag.times_in_loser = tag_data.get("times_in_loser", 0)
            
            # Restore round count
            self.refiner.current_round = self.round_count
        
        # Restore snapshots and best picks
        self.tag_states_snapshots = {
            int(k): v for k, v in state.get("tag_states_snapshots", {}).items()
        }
        self.best_picks = state.get("best_picks", {})
        # Ensure keys are ints
        if self.best_picks:
            self.best_picks = {int(k) if isinstance(k, str) else k: v for k, v in self.best_picks.items()}
        
        print(f"[HITLSessionV2] Restored from state: round={self.round_count}, snapshots={len(self.tag_states_snapshots)}")
        return True
    
    @classmethod
    def load_or_create(
        cls,
        session_id: str,
        session_folder: str,
        pipe: Any = None,
        base_prompt: str = "",
        negative_prompt: str = "",
        **kwargs
    ) -> "HITLRefinementSessionV2":
        """
        Load existing session or create new one.
        """
        session = cls(
            session_id=session_id,
            session_folder=session_folder,
            pipe=pipe,
            base_prompt=base_prompt,
            negative_prompt=negative_prompt,
            **kwargs
        )
        
        # Try to restore
        if session._restore_from_state():
            # Initialize fuser if pipe available
            if pipe is not None and HAS_FUSER:
                session.fuser = HITLCompositionFuser(
                    pipe=pipe,
                    device=getattr(pipe, 'device', 'cuda')
                )
            return session
        
        # No saved state, initialize fresh
        session.initialize_from_exploration()
        return session
