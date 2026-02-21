"""
Tag-Level GP Refinement System

A simplified, principled approach to refining tag selections through
user preference feedback over 6 rounds.

Key features:
- Direct tag-level utilities (μ, σ) instead of weight mixtures
- Prior initialization from exploration (positive/neutral × selected/other)
- 4 strategic options per round: Exploit, Explore, UCB, Challenger
- User RANKS all 4 options → 6 pairwise comparisons per round
- Decaying exploration coefficient β over rounds
- Diversity constraints that relax as refinement progresses
- Cross-attention weighted image generation via hitl_fuser
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any
from enum import Enum
import json
from pathlib import Path
from datetime import datetime
from PIL import Image

# Import logger
try:
    from gp_refinement_logger import GPRefinementLogger, create_logger
    HAS_LOGGER = True
except ImportError:
    try:
        # Try relative import for when running from project root
        from eval.backend.gp_refinement_logger import GPRefinementLogger, create_logger
        HAS_LOGGER = True
    except ImportError:
        HAS_LOGGER = False
        GPRefinementLogger = None
        print("[GPRefiner] Warning: Logger not available")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GPRefinerConfig:
    """Configuration for the GP refinement process."""
    
    # Round settings (no hard limit - user decides when to stop)
    max_rounds: int = 999
    tags_per_option: int = 10
    
    # Tag composition per option
    top_positive_count: int = 3      # From top-tier positive tags
    remaining_positive_count: int = 4  # From remaining positive tags
    neutral_count: int = 3            # From neutral tags
    
    # Initial prior values
    prior_positive_selected_mu: float = 1.0
    prior_positive_selected_sigma: float = 0.5
    prior_positive_other_mu: float = 0.5
    prior_positive_other_sigma: float = 0.7
    prior_neutral_selected_mu: float = 0.0
    prior_neutral_selected_sigma: float = 0.5
    prior_neutral_other_mu: float = -0.3
    prior_neutral_other_sigma: float = 0.7
    
    # Learning parameters
    base_learning_rate: float = 0.12
    learning_rate_decay: float = 0.98  # Per round
    sigma_shrink_factor: float = 0.92  # How much to reduce σ on feedback
    min_sigma: float = 0.1  # Floor for uncertainty
    
    # Preference strength by rank distance
    rank_distance_weights: Dict[int, float] = field(default_factory=lambda: {
        3: 1.0,   # 1st vs 4th
        2: 0.6,   # 1st vs 3rd, 2nd vs 4th
        1: 0.3,   # Adjacent ranks
    })
    
    # Diversity constraints (max overlap as fraction)
    diversity_schedule: Dict[int, float] = field(default_factory=lambda: {
        1: 0.40,  # Round 1-3: keep options clearly distinct
        2: 0.40,
        3: 0.45,
        4: 0.55,  # Round 4-6: still diverse, start mild convergence
        5: 0.55,
        6: 0.60,
        7: 0.70,  # Later rounds can converge more
        8: 0.75,
    })


# ============================================================================
# Data Structures
# ============================================================================

class TagCategory(Enum):
    """Category from exploration stage."""
    POSITIVE_SELECTED = "positive_selected"
    POSITIVE_OTHER = "positive_other"
    NEUTRAL_SELECTED = "neutral_selected"
    NEUTRAL_OTHER = "neutral_other"


@dataclass
class TagState:
    """State for a single tag."""
    tag_id: str
    text: str
    category: TagCategory
    mu: float  # Mean utility
    sigma: float  # Uncertainty (std dev)
    
    # Tracking
    times_shown: int = 0
    times_in_winner: int = 0
    times_in_loser: int = 0
    
    def score(self, beta: float) -> float:
        """Compute UCB-style selection score."""
        return self.mu + beta * self.sigma
    
    def to_dict(self) -> Dict:
        return {
            "tag_id": self.tag_id,
            "text": self.text,
            "category": self.category.value,
            "mu": round(self.mu, 4),
            "sigma": round(self.sigma, 4),
            "times_shown": self.times_shown,
            "times_in_winner": self.times_in_winner,
            "times_in_loser": self.times_in_loser,
        }


class OptionStrategy(Enum):
    """Strategy used to generate an option."""
    EXPLOIT = "exploit"        # Highest mean μ
    EXPLORE = "explore"        # Highest uncertainty σ
    UCB_BALANCED = "ucb"       # μ + β × σ
    CHALLENGER = "challenger"  # Top tags with swaps


@dataclass
class RefinementOption:
    """A single option presented to the user."""
    option_id: int
    strategy: OptionStrategy
    tags: List[str]  # Tag texts
    tag_ids: List[str]  # Tag IDs for tracking
    weights: Optional[List[float]] = None  # Attention weights for each tag
    image_path: Optional[str] = None  # Path to generated image
    
    def to_dict(self) -> Dict:
        return {
            "option_id": self.option_id,
            "strategy": self.strategy.value,
            "tags": self.tags,
            "tag_ids": self.tag_ids,
            "weights": self.weights,
            "image_path": self.image_path,
        }


@dataclass
class RoundResult:
    """Result of a single refinement round."""
    round_num: int
    options: List[RefinementOption]
    ranking: List[int]  # User ranking [best_option_id, ..., worst_option_id]
    pairwise_comparisons: List[Tuple[int, int]]  # (better_id, worse_id) pairs
    tag_updates: Dict[str, Dict[str, float]]  # tag_id -> {mu_delta, sigma_factor}
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "round_num": self.round_num,
            "options": [o.to_dict() for o in self.options],
            "ranking": self.ranking,
            "pairwise_comparisons": self.pairwise_comparisons,
            "tag_updates": self.tag_updates,
            "diagnostics": self.diagnostics,
        }


# ============================================================================
# Main GP Refiner Class
# ============================================================================

class TagGPRefiner:
    """
    Tag-level GP refinement through ranked preference feedback.
    
    Usage:
        refiner = TagGPRefiner(config)
        refiner.set_sdxl_pipeline(pipe, base_prompt)  # For image generation
        refiner.initialize_from_exploration(positive_tags, neutral_tags, selected_image_tags)
        
        for round_num in range(1, 7):
            options = refiner.generate_round_options()
            images = refiner.generate_round_images(output_dir)  # Optional: generate images
            # ... show images to user, get ranking ...
            refiner.submit_ranking(ranking)  # e.g., [2, 0, 3, 1]
        
        final_tags, weights = refiner.get_final_selection()
    """
    
    def __init__(self, config: Optional[GPRefinerConfig] = None):
        self.config = config or GPRefinerConfig()
        
        # Tag states
        self.tags: Dict[str, TagState] = {}
        self.positive_tag_ids: List[str] = []
        self.neutral_tag_ids: List[str] = []
        
        # Round tracking
        self.current_round: int = 0
        self.round_history: List[RoundResult] = []
        self.current_options: List[RefinementOption] = []
        
        # Random state for reproducibility
        self.rng = np.random.RandomState(42)
        
        # Logger
        self.logger: Optional[GPRefinementLogger] = None
        
        # SDXL integration
        self.pipe: Optional[Any] = None
        self.fuser: Optional[Any] = None
        self.base_prompt: str = ""
        self.negative_prompt: str = "illustration, painted, drawing, cartoon, anime, 3D render, CGI"
        self.image_height: int = 512
        self.image_width: int = 512
        self.num_inference_steps: int = 30
        self.guidance_scale: float = 7.5
    
    @property
    def beta(self) -> float:
        """Current exploration coefficient (decays over rounds, floors at 0.2)."""
        if self.current_round == 0:
            return 2.0
        # Decay over first 6 rounds, then floor at 0.2 for exploitation-heavy later rounds
        decay_rounds = 6
        decay = 2.0 * (1.0 - min((self.current_round - 1) / decay_rounds, 1.0))
        return max(decay, 0.2)
    
    @property
    def learning_rate(self) -> float:
        """Current learning rate (decays slightly over rounds)."""
        return self.config.base_learning_rate * (
            self.config.learning_rate_decay ** (self.current_round - 1)
        )
    
    @property
    def max_overlap(self) -> float:
        """Maximum allowed overlap between options for current round."""
        if self.current_round in self.config.diversity_schedule:
            return self.config.diversity_schedule[self.current_round]
        # For rounds beyond the schedule, use the highest defined value
        max_scheduled_round = max(self.config.diversity_schedule.keys())
        return self.config.diversity_schedule[max_scheduled_round]
    
    @property
    def is_complete(self) -> bool:
        """Whether refinement is complete."""
        return self.current_round >= self.config.max_rounds
    
    # -------------------------------------------------------------------------
    # Logging Setup
    # -------------------------------------------------------------------------
    
    def set_logger(self, session_id: str, session_folder: Path) -> None:
        """Set up logging for this refinement session."""
        if HAS_LOGGER:
            self.logger = create_logger(session_id, session_folder)
            print(f"[GPRefiner] Logger initialized for session {session_id}")
        else:
            print("[GPRefiner] Logger not available")
    
    # -------------------------------------------------------------------------
    # SDXL Integration
    # -------------------------------------------------------------------------
    
    def set_sdxl_pipeline(
        self,
        pipe: Any,
        base_prompt: str,
        negative_prompt: Optional[str] = None,
        image_height: int = 512,
        image_width: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> None:
        """
        Configure SDXL pipeline for image generation.
        
        Args:
            pipe: SDXL pipeline (e.g., from diffusers)
            base_prompt: Base prompt for all generations (e.g., "Cozy Living Room")
            negative_prompt: Negative prompt (optional)
            image_height: Image height
            image_width: Image width
            num_inference_steps: Diffusion steps
            guidance_scale: CFG scale
        """
        self.pipe = pipe
        self.base_prompt = base_prompt
        self.image_height = image_height
        self.image_width = image_width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        if negative_prompt:
            self.negative_prompt = negative_prompt
        
        # Initialize the fuser for cross-attention weighting
        try:
            from hitl_fuser import HITLCompositionFuser
            self.fuser = HITLCompositionFuser(pipe)
            print(f"[GPRefiner] SDXL fuser initialized with base_prompt: '{base_prompt}'")
        except ImportError as e:
            print(f"[GPRefiner] Warning: Could not import hitl_fuser: {e}")
            self.fuser = None
    
    def _compute_option_weights(self, option: RefinementOption) -> List[float]:
        """
        Compute attention weights for an option's tags.
        
        Uses the current μ values, normalized via softmax for attention scaling.
        Higher μ = higher attention weight.
        """
        tag_states = [self.tags[tid] for tid in option.tag_ids]
        mus = np.array([t.mu for t in tag_states])
        
        # Softmax normalization with temperature
        # Lower temperature = sharper weights (more contrast)
        temperature = 1.0
        exp_mus = np.exp(mus / temperature)
        weights = exp_mus / exp_mus.sum()
        
        # Scale to attention range [0.5, 1.5]
        # This ensures all tags contribute but with varying emphasis
        weights_scaled = 0.5 + weights * (1.5 - 0.5) / weights.max()
        
        return weights_scaled.tolist()
    
    def generate_round_images(
        self,
        output_dir: Path,
        seed_base: int = 42,
    ) -> List[str]:
        """
        Generate images for all 4 current options using cross-attention weighting.
        
        Args:
            output_dir: Directory to save images
            seed_base: Base seed for reproducibility
        
        Returns:
            List of image paths
        """
        if not self.current_options:
            raise ValueError("No options generated. Call generate_round_options() first.")
        
        if not self.pipe or not self.fuser:
            print("[GPRefiner] Warning: No SDXL pipeline configured, skipping image generation")
            return []
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        
        # Import generation function
        try:
            from hitl_fuser import generate_with_attention_control
        except ImportError:
            print("[GPRefiner] Could not import generation function")
            return []
        
        for i, option in enumerate(self.current_options):
            # Compute attention weights for this option's tags
            weights = self._compute_option_weights(option)
            option.weights = weights
            
            print(f"\n[GPRefiner] Generating image for Option {option.option_id} ({option.strategy.value})")
            print(f"  Tags: {option.tags[:3]}... ({len(option.tags)} total)")
            print(f"  Weights: {[f'{w:.2f}' for w in weights[:3]]}...")
            
            # Create embeddings with attention control
            prompt_embeds, pooled, neg_embeds, neg_pooled, attn_controller = \
                self.fuser.fuse_tags_with_weights(
                    tag_labels=option.tags,
                    tag_weights=weights,
                    base_prompt=self.base_prompt,
                    neg_phrases=[self.negative_prompt] if self.negative_prompt else None,
                )
            
            # Generate image
            seed = seed_base + self.current_round * 10 + i
            image = generate_with_attention_control(
                pipe=self.pipe,
                prompt_embeds=prompt_embeds,
                pooled=pooled,
                neg_embeds=neg_embeds,
                neg_pooled=neg_pooled,
                attn_controller=attn_controller,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                seed=seed,
                height=self.image_height,
                width=self.image_width,
            )
            
            # Save image
            image_path = output_dir / f"option_{option.option_id}.png"
            image.save(str(image_path))
            option.image_path = str(image_path)
            image_paths.append(str(image_path))
            
            print(f"  Saved: {image_path}")
        
        return image_paths
    
    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    
    def initialize_from_exploration(
        self,
        positive_tags: List[str],
        neutral_tags: List[str],
        selected_image_tags: Set[str],
        tag_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict:
        """
        Initialize tag utilities from exploration results.
        
        Args:
            positive_tags: Tags marked as positive/liked
            neutral_tags: Tags that were neutral
            selected_image_tags: Tags that appeared in user-selected images
            tag_embeddings: Optional CLIP embeddings for tags (for future use)
        
        Returns:
            Initialization summary
        """
        self.tags.clear()
        self.positive_tag_ids.clear()
        self.neutral_tag_ids.clear()
        
        # Initialize positive tags
        for i, tag in enumerate(positive_tags):
            tag_id = f"pos_{i}"
            in_selected = tag in selected_image_tags
            
            if in_selected:
                category = TagCategory.POSITIVE_SELECTED
                mu = self.config.prior_positive_selected_mu
                sigma = self.config.prior_positive_selected_sigma
            else:
                category = TagCategory.POSITIVE_OTHER
                mu = self.config.prior_positive_other_mu
                sigma = self.config.prior_positive_other_sigma
            
            self.tags[tag_id] = TagState(
                tag_id=tag_id,
                text=tag,
                category=category,
                mu=mu,
                sigma=sigma,
            )
            self.positive_tag_ids.append(tag_id)
        
        # Initialize neutral tags
        for i, tag in enumerate(neutral_tags):
            tag_id = f"neu_{i}"
            in_selected = tag in selected_image_tags
            
            if in_selected:
                category = TagCategory.NEUTRAL_SELECTED
                mu = self.config.prior_neutral_selected_mu
                sigma = self.config.prior_neutral_selected_sigma
            else:
                category = TagCategory.NEUTRAL_OTHER
                mu = self.config.prior_neutral_other_mu
                sigma = self.config.prior_neutral_other_sigma
            
            self.tags[tag_id] = TagState(
                tag_id=tag_id,
                text=tag,
                category=category,
                mu=mu,
                sigma=sigma,
            )
            self.neutral_tag_ids.append(tag_id)
        
        self.current_round = 0
        
        result = {
            "status": "initialized",
            "positive_tags": len(self.positive_tag_ids),
            "neutral_tags": len(self.neutral_tag_ids),
            "total_tags": len(self.tags),
            "categories": {
                "positive_selected": sum(1 for t in self.tags.values() 
                                        if t.category == TagCategory.POSITIVE_SELECTED),
                "positive_other": sum(1 for t in self.tags.values() 
                                     if t.category == TagCategory.POSITIVE_OTHER),
                "neutral_selected": sum(1 for t in self.tags.values() 
                                       if t.category == TagCategory.NEUTRAL_SELECTED),
                "neutral_other": sum(1 for t in self.tags.values() 
                                    if t.category == TagCategory.NEUTRAL_OTHER),
            }
        }
        
        # Log initialization
        if self.logger:
            config_dict = {
                "max_rounds": self.config.max_rounds,
                "tags_per_option": self.config.tags_per_option,
                "base_learning_rate": self.config.base_learning_rate,
                "sigma_shrink_factor": self.config.sigma_shrink_factor,
                "priors": {
                    "positive_selected": {"mu": self.config.prior_positive_selected_mu, 
                                         "sigma": self.config.prior_positive_selected_sigma},
                    "positive_other": {"mu": self.config.prior_positive_other_mu,
                                      "sigma": self.config.prior_positive_other_sigma},
                    "neutral_selected": {"mu": self.config.prior_neutral_selected_mu,
                                        "sigma": self.config.prior_neutral_selected_sigma},
                    "neutral_other": {"mu": self.config.prior_neutral_other_mu,
                                     "sigma": self.config.prior_neutral_other_sigma},
                },
            }
            self.logger.log_initialization(
                positive_tags=positive_tags,
                neutral_tags=neutral_tags,
                selected_image_tags=selected_image_tags,
                tag_states=self.tags,
                config=config_dict,
            )
        
        return result
    
    # -------------------------------------------------------------------------
    # Option Generation
    # -------------------------------------------------------------------------
    
    def generate_round_options(self) -> List[RefinementOption]:
        """
        Generate 4 diverse options for the current round.
        
        Returns:
            List of 4 RefinementOption objects
        """
        self.current_round += 1
        beta = self.beta
        
        print(f"\n[GPRefiner] Round {self.current_round} | β={beta:.2f} | max_overlap={self.max_overlap:.0%}")
        
        options = []
        
        # Option 1: Exploitation (highest mean)
        option1 = self._generate_exploit_option()
        options.append(option1)
        
        # Option 2: Exploration (highest uncertainty)
        option2 = self._generate_explore_option(existing_options=options)
        options.append(option2)
        
        # Option 3: UCB Balanced
        option3 = self._generate_ucb_option(beta, existing_options=options)
        options.append(option3)
        
        # Option 4: Challenger (swap alternatives)
        option4 = self._generate_challenger_option(option1, beta, existing_options=options)
        options.append(option4)
        
        self.current_options = options
        
        # Log option summaries
        for opt in options:
            tag_states = [self.tags[tid] for tid in opt.tag_ids]
            avg_mu = np.mean([t.mu for t in tag_states])
            avg_sigma = np.mean([t.sigma for t in tag_states])
            print(f"  Option {opt.option_id} [{opt.strategy.value:10s}]: "
                  f"avg_μ={avg_mu:.2f}, avg_σ={avg_sigma:.2f}")
        
        # === LOGGING ===
        if self.logger:
            # Start round logging
            self.logger.start_round(
                round_num=self.current_round,
                beta=beta,
                max_overlap=self.max_overlap,
                learning_rate=self.learning_rate,
            )
            
            # Log each option
            for opt in options:
                tag_states_list = [self.tags[tid] for tid in opt.tag_ids]
                self.logger.log_option(
                    option_id=opt.option_id,
                    strategy=opt.strategy.value,
                    tag_ids=opt.tag_ids,
                    tag_texts=opt.tags,
                    tag_mus=[t.mu for t in tag_states_list],
                    tag_sigmas=[t.sigma for t in tag_states_list],
                    weights=opt.weights,
                    image_path=opt.image_path,
                )
            
            # Log diversity checks
            for i, opt_a in enumerate(options):
                for opt_b in options[i+1:]:
                    shared = set(opt_a.tags) & set(opt_b.tags)
                    self.logger.log_diversity_check(
                        option_a=opt_a.option_id,
                        option_b=opt_b.option_id,
                        shared_tags=list(shared),
                        max_allowed=self.max_overlap,
                    )
        
        return options
    
    def _select_tags_by_score(
        self,
        score_fn,
        exclude_ids: Optional[Set[str]] = None,
        category_filter: Optional[List[TagCategory]] = None,
    ) -> List[str]:
        """
        Select tags for an option based on a scoring function.
        
        Returns tag_ids maintaining the required composition:
        - top_positive_count from top positive tags
        - remaining_positive_count from remaining positive tags
        - neutral_count from neutral tags
        """
        exclude_ids = exclude_ids or set()
        
        # Score all tags
        positive_scores = []
        neutral_scores = []
        
        for tag_id, tag in self.tags.items():
            if tag_id in exclude_ids:
                continue
            
            score = score_fn(tag)
            
            if tag_id in self.positive_tag_ids:
                positive_scores.append((tag_id, score))
            else:
                neutral_scores.append((tag_id, score))
        
        # Sort by score descending
        positive_scores.sort(key=lambda x: x[1], reverse=True)
        neutral_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        
        # Select top positive tags
        top_pos_count = min(self.config.top_positive_count, len(positive_scores))
        selected.extend([tid for tid, _ in positive_scores[:top_pos_count]])
        
        # Select remaining positive tags
        remaining_pos = positive_scores[top_pos_count:]
        rem_pos_count = min(self.config.remaining_positive_count, len(remaining_pos))
        selected.extend([tid for tid, _ in remaining_pos[:rem_pos_count]])
        
        # Select neutral tags
        neu_count = min(self.config.neutral_count, len(neutral_scores))
        selected.extend([tid for tid, _ in neutral_scores[:neu_count]])
        
        # If we don't have enough tags, fill with remaining
        total_needed = self.config.tags_per_option
        if len(selected) < total_needed:
            all_remaining = [
                (tid, score_fn(self.tags[tid])) 
                for tid in self.tags 
                if tid not in selected and tid not in exclude_ids
            ]
            all_remaining.sort(key=lambda x: x[1], reverse=True)
            for tid, _ in all_remaining:
                if len(selected) >= total_needed:
                    break
                selected.append(tid)
        
        return selected[:total_needed]
    
    def _generate_exploit_option(self) -> RefinementOption:
        """Generate option using pure exploitation (highest μ)."""
        tag_ids = self._select_tags_by_score(lambda t: t.mu)
        
        return RefinementOption(
            option_id=0,
            strategy=OptionStrategy.EXPLOIT,
            tags=[self.tags[tid].text for tid in tag_ids],
            tag_ids=tag_ids,
        )
    
    def _generate_explore_option(
        self, 
        existing_options: List[RefinementOption]
    ) -> RefinementOption:
        """Generate option using pure exploration (highest σ)."""
        # Ensure diversity from existing options
        tag_ids = self._select_diverse_tags(
            score_fn=lambda t: t.sigma,
            existing_options=existing_options,
        )
        
        return RefinementOption(
            option_id=1,
            strategy=OptionStrategy.EXPLORE,
            tags=[self.tags[tid].text for tid in tag_ids],
            tag_ids=tag_ids,
        )
    
    def _generate_ucb_option(
        self, 
        beta: float,
        existing_options: List[RefinementOption]
    ) -> RefinementOption:
        """Generate option using UCB (μ + β × σ)."""
        tag_ids = self._select_diverse_tags(
            score_fn=lambda t: t.score(beta),
            existing_options=existing_options,
        )
        
        return RefinementOption(
            option_id=2,
            strategy=OptionStrategy.UCB_BALANCED,
            tags=[self.tags[tid].text for tid in tag_ids],
            tag_ids=tag_ids,
        )
    
    def _generate_challenger_option(
        self,
        exploit_option: RefinementOption,
        beta: float,
        existing_options: List[RefinementOption]
    ) -> RefinementOption:
        """
        Generate challenger option: start with exploit, swap 2-3 tags with 
        promising alternatives (not in top 10 by mean, but high UCB).
        """
        # Start with exploit tags
        base_tags = set(exploit_option.tag_ids)
        
        # Find promising alternatives
        top_by_mean = set(self._select_tags_by_score(lambda t: t.mu))
        
        alternatives = []
        for tag_id, tag in self.tags.items():
            if tag_id not in top_by_mean:
                alternatives.append((tag_id, tag.score(beta)))
        
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        # Swap 2-3 tags
        num_swaps = min(3, len(alternatives))
        if self.current_round >= 8:
            num_swaps = 2  # Fewer swaps in later rounds
        
        # Remove lowest-scoring tags from base
        base_list = list(base_tags)
        base_scores = [(tid, self.tags[tid].score(beta)) for tid in base_list]
        base_scores.sort(key=lambda x: x[1])  # Ascending (lowest first)
        
        tags_to_remove = [tid for tid, _ in base_scores[:num_swaps]]
        tags_to_add = [tid for tid, _ in alternatives[:num_swaps]]
        
        new_tags = [tid for tid in base_list if tid not in tags_to_remove] + tags_to_add
        
        # Ensure diversity
        new_tags = self._ensure_diversity(new_tags, existing_options)
        
        return RefinementOption(
            option_id=3,
            strategy=OptionStrategy.CHALLENGER,
            tags=[self.tags[tid].text for tid in new_tags],
            tag_ids=new_tags,
        )
    
    def _select_diverse_tags(
        self,
        score_fn,
        existing_options: List[RefinementOption],
    ) -> List[str]:
        """Select tags ensuring diversity from existing options."""
        tag_ids = self._select_tags_by_score(score_fn)
        return self._ensure_diversity(tag_ids, existing_options)
    
    def _ensure_diversity(
        self,
        tag_ids: List[str],
        existing_options: List[RefinementOption],
    ) -> List[str]:
        """
        Ensure the tag set has sufficient diversity from existing options.
        If too similar, swap some tags.
        
        Note: When total tags < 2 * tags_per_option, perfect diversity is impossible.
        We do our best given available tags.
        """
        total_tags = len(self.tags)
        tags_per_option = self.config.tags_per_option
        
        # Calculate minimum possible overlap given total tags
        # With N total and K per option, min overlap = max(0, 2K - N)
        min_possible_overlap = max(0, 2 * tags_per_option - total_tags)
        min_overlap_fraction = min_possible_overlap / tags_per_option
        
        # Adjust max_overlap_count if our constraint is impossible
        target_max_overlap = self.max_overlap
        if min_overlap_fraction > target_max_overlap:
            # Constraint is impossible - use minimum possible + small margin
            adjusted_overlap = min_overlap_fraction + 0.1
            max_overlap_count = int(adjusted_overlap * tags_per_option)
            # Log this once per round
            if not hasattr(self, '_diversity_warning_shown'):
                self._diversity_warning_shown = set()
            warn_key = f"r{self.current_round}"
            if warn_key not in self._diversity_warning_shown:
                print(f"  [Diversity] With {total_tags} tags, min overlap is {min_overlap_fraction:.0%}. "
                      f"Relaxing constraint from {target_max_overlap:.0%} to {adjusted_overlap:.0%}")
                self._diversity_warning_shown.add(warn_key)
        else:
            max_overlap_count = int(target_max_overlap * tags_per_option)
        
        tag_set = set(tag_ids)
        
        for existing in existing_options:
            existing_set = set(existing.tag_ids)
            overlap = len(tag_set & existing_set)
            
            if overlap > max_overlap_count:
                # Too similar - swap some overlapping tags
                overlapping = list(tag_set & existing_set)
                self.rng.shuffle(overlapping)
                
                num_to_swap = overlap - max_overlap_count
                tags_to_remove = overlapping[:num_to_swap]
                
                # Find replacement tags not in any existing option or current set
                all_used = tag_set.copy()
                for opt in existing_options:
                    all_used.update(opt.tag_ids)
                
                available = [
                    tid for tid in self.tags.keys()
                    if tid not in all_used
                ]
                
                # If no available tags, we can't improve diversity
                if not available:
                    continue
                
                # Score and select replacements
                # Prefer uncertain-but-promising replacements to keep exploration alive.
                available_scored = [
                    (tid, self.tags[tid].sigma + 0.35 * self.tags[tid].mu)
                    for tid in available
                ]
                available_scored.sort(key=lambda x: x[1], reverse=True)
                
                replacements = [tid for tid, _ in available_scored[:num_to_swap]]
                
                # Swap
                for old_tid, new_tid in zip(tags_to_remove, replacements):
                    if old_tid in tag_set and new_tid:
                        tag_set.remove(old_tid)
                        tag_set.add(new_tid)
        
        return list(tag_set)[:tags_per_option]
    
    # -------------------------------------------------------------------------
    # Feedback Processing
    # -------------------------------------------------------------------------
    
    def submit_ranking(self, ranking: List[int]) -> RoundResult:
        """
        Submit user ranking for the current round's options.
        
        Args:
            ranking: List of option_ids from best to worst.
                     e.g., [2, 0, 3, 1] means Option 2 is best, Option 1 is worst.
        
        Returns:
            RoundResult with update details
        """
        if len(ranking) != len(self.current_options):
            raise ValueError(f"Ranking must have {len(self.current_options)} elements")
        if len(set(ranking)) != len(ranking):
            raise ValueError("Ranking contains duplicate option IDs")
        valid_option_ids = {opt.option_id for opt in self.current_options}
        if set(ranking) != valid_option_ids:
            raise ValueError(
                f"Ranking IDs {ranking} do not match current option IDs {sorted(valid_option_ids)}"
            )
        
        # Generate all pairwise comparisons
        pairwise = []
        for i, better_id in enumerate(ranking):
            for worse_id in ranking[i+1:]:
                pairwise.append((better_id, worse_id))
        
        print(f"\n[GPRefiner] Processing ranking: {ranking}")
        print(f"  Generated {len(pairwise)} pairwise comparisons")

        # Pre-update diagnostics (how well model predicted user's ranking)
        option_id_to_option = {opt.option_id: opt for opt in self.current_options}
        option_scores_before, option_unc_before = self._compute_option_scores()
        ranking_diagnostics = self._compute_ranking_diagnostics(
            ranking=ranking,
            pairwise=pairwise,
            option_scores=option_scores_before,
            option_uncertainty=option_unc_before
        )
        
        # Process each comparison
        tag_updates: Dict[str, Dict[str, float]] = {}
        sigma_update_counts: Dict[str, int] = {}
        
        for better_id, worse_id in pairwise:
            rank_better = ranking.index(better_id)
            rank_worse = ranking.index(worse_id)
            rank_distance = rank_worse - rank_better

            preference_strength = self._compute_pairwise_strength(
                better_id=better_id,
                worse_id=worse_id,
                rank_distance=rank_distance,
                option_scores=option_scores_before,
                option_uncertainty=option_unc_before
            )
            
            updates = self._process_pairwise_comparison(
                better_id, worse_id, preference_strength
            )
            
            # Merge updates
            for tag_id, update in updates.items():
                if tag_id not in tag_updates:
                    tag_updates[tag_id] = {"mu_delta": 0.0, "sigma_factor": 1.0}
                    sigma_update_counts[tag_id] = 0
                tag_updates[tag_id]["mu_delta"] += update["mu_delta"]
                tag_updates[tag_id]["sigma_factor"] *= update["sigma_factor"]
                sigma_update_counts[tag_id] += 1
        
        # Apply accumulated updates
        for tag_id, update in tag_updates.items():
            tag = self.tags[tag_id]
            tag.mu += update["mu_delta"]
            # Use geometric mean of shrink factors so σ does not collapse too fast
            # when the same tag appears in many pairwise updates in one round.
            sigma_count = max(1, sigma_update_counts.get(tag_id, 1))
            sigma_factor = update["sigma_factor"] ** (1.0 / sigma_count)
            tag.sigma = max(
                self.config.min_sigma,
                tag.sigma * sigma_factor
            )
        
        # Update tracking
        for opt in self.current_options:
            for tag_id in opt.tag_ids:
                self.tags[tag_id].times_shown += 1
        
        best_option = option_id_to_option[ranking[0]]
        worst_option = option_id_to_option[ranking[-1]]
        
        for tag_id in best_option.tag_ids:
            self.tags[tag_id].times_in_winner += 1
        for tag_id in worst_option.tag_ids:
            self.tags[tag_id].times_in_loser += 1
        
        # Create round result
        result = RoundResult(
            round_num=self.current_round,
            options=self.current_options,
            ranking=ranking,
            pairwise_comparisons=pairwise,
            tag_updates=tag_updates,
            diagnostics=ranking_diagnostics,
        )
        
        self.round_history.append(result)
        
        # Log top tags after update
        self._log_top_tags()
        
        # === LOGGING ===
        if self.logger:
            # Log ranking
            self.logger.log_ranking(ranking)
            
            # Log each comparison with details
            for better_id, worse_id in pairwise:
                better_opt = option_id_to_option[better_id]
                worse_opt = option_id_to_option[worse_id]
                
                self.logger.log_comparison(
                    better_option_id=better_id,
                    worse_option_id=worse_id,
                    ranking=ranking,
                    better_tags=set(better_opt.tags),
                    worse_tags=set(worse_opt.tags),
                )
            
            # Log tag updates (with before/after values)
            # Note: We need to reconstruct before values from the deltas
            for tag_id, update in tag_updates.items():
                tag = self.tags[tag_id]
                mu_after = tag.mu
                sigma_after = tag.sigma
                mu_before = mu_after - update["mu_delta"]
                # sigma_before is harder to reconstruct due to multiplicative factor
                # Approximate: sigma_before = sigma_after / update["sigma_factor"]
                sigma_before = sigma_after / update["sigma_factor"] if update["sigma_factor"] > 0 else sigma_after
                
                # Determine update reason
                reason = "in_better" if update["mu_delta"] > 0 else "in_worse"
                
                self.logger.log_tag_update(
                    tag_id=tag_id,
                    text=tag.text,
                    mu_before=mu_before,
                    mu_after=mu_after,
                    sigma_before=sigma_before,
                    sigma_after=sigma_after,
                    update_reason=reason,
                    from_comparison="multiple",  # Accumulated from multiple comparisons
                )
            
            # End round logging with diagnostics
            self.logger.end_round(
                tag_states=self.tags,
                diagnostics={
                    "ranking_received": ranking,
                    "best_option_strategy": option_id_to_option[ranking[0]].strategy.value,
                    "worst_option_strategy": option_id_to_option[ranking[-1]].strategy.value,
                    "pairwise_accuracy_before_update": ranking_diagnostics.get("pairwise_accuracy_before_update"),
                    "ranking_log_likelihood_before_update": ranking_diagnostics.get("ranking_log_likelihood_before_update"),
                }
            )
        
        return result

    def _compute_option_scores(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Compute per-option utility/uncertainty summaries before updates.
        """
        option_scores: Dict[int, float] = {}
        option_unc: Dict[int, float] = {}
        for opt in self.current_options:
            tag_states = [self.tags[tid] for tid in opt.tag_ids if tid in self.tags]
            if not tag_states:
                option_scores[opt.option_id] = 0.0
                option_unc[opt.option_id] = 1.0
                continue
            mus = np.array([t.mu for t in tag_states], dtype=float)
            sigmas = np.array([t.sigma for t in tag_states], dtype=float)
            # Align prediction score with image generation more closely:
            # weighted utility over tags instead of plain mean μ.
            # Weights mimic attention emphasis from relative μ within option.
            temp = 0.8
            exp_m = np.exp((mus - mus.max()) / max(1e-6, temp))
            w = exp_m / (exp_m.sum() + 1e-8)
            utility = mus + 0.20 * sigmas
            option_scores[opt.option_id] = float(np.dot(w, utility))
            option_unc[opt.option_id] = float(np.dot(w, sigmas))
        return option_scores, option_unc

    def _compute_pairwise_strength(
        self,
        better_id: int,
        worse_id: int,
        rank_distance: int,
        option_scores: Dict[int, float],
        option_uncertainty: Dict[int, float],
    ) -> float:
        """
        Calibrated pairwise preference strength:
        - Base from rank distance schedule
        - Stronger when rank distance is larger
        - Slightly stronger when compared options are uncertain
        """
        base = self.config.rank_distance_weights.get(rank_distance, 0.5)

        # Smooth distance factor for robustness when #options changes
        max_dist = max(1, len(self.current_options) - 1)
        dist_norm = max(0.0, min(1.0, rank_distance / max_dist))
        distance_factor = 0.8 + 0.4 * dist_norm  # [0.8, 1.2]

        unc = (option_uncertainty.get(better_id, 0.5) + option_uncertainty.get(worse_id, 0.5)) / 2.0
        unc_factor = 1.0 + min(0.2, 0.1 * unc)  # modest boost in uncertain regions

        # Slightly de-emphasize very obvious pairs (large current score gap),
        # prioritize close calls where ranking feedback is most informative.
        score_gap = abs(option_scores.get(better_id, 0.0) - option_scores.get(worse_id, 0.0))
        gap_factor = max(0.85, 1.1 - min(0.25, 0.15 * score_gap))

        calibrated = base * distance_factor * unc_factor * gap_factor
        return float(max(0.05, min(1.5, calibrated)))

    def _compute_ranking_diagnostics(
        self,
        ranking: List[int],
        pairwise: List[Tuple[int, int]],
        option_scores: Dict[int, float],
        option_uncertainty: Dict[int, float],
    ) -> Dict[str, Any]:
        """
        Pre-update diagnostics for analyzing model learning quality.
        """
        if not pairwise:
            return {
                "pairwise_accuracy_before_update": 0.0,
                "ranking_log_likelihood_before_update": 0.0,
                "mean_pair_margin_before_update": 0.0,
                "predicted_top_option_before_update": float(ranking[0]) if ranking else -1.0,
            }

        correct = 0.0
        log_lik = 0.0
        margins = []

        for better_id, worse_id in pairwise:
            s_b = option_scores.get(better_id, 0.0)
            s_w = option_scores.get(worse_id, 0.0)
            margin = s_b - s_w
            margins.append(margin)

            if margin > 0:
                correct += 1.0
            elif margin == 0:
                correct += 0.5

            # Bradley-Terry-style pair probability with uncertainty-scaled temperature
            tau = max(0.15, (option_uncertainty.get(better_id, 0.5) + option_uncertainty.get(worse_id, 0.5)) / 2.0)
            z = margin / tau
            p = 1.0 / (1.0 + np.exp(-z))
            p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
            log_lik += float(np.log(p))

        pairwise_acc = correct / len(pairwise)
        mean_margin = float(np.mean(margins)) if margins else 0.0
        predicted_top = max(option_scores.items(), key=lambda x: x[1])[0] if option_scores else -1
        predicted_ranking = [oid for oid, _ in sorted(option_scores.items(), key=lambda x: x[1], reverse=True)]

        # Full-ranking agreement diagnostics
        rank_pos_true = {oid: i for i, oid in enumerate(ranking)}
        rank_pos_pred = {oid: i for i, oid in enumerate(predicted_ranking)}
        common_ids = [oid for oid in predicted_ranking if oid in rank_pos_true]

        if len(common_ids) >= 2:
            true_vals = np.array([rank_pos_true[oid] for oid in common_ids], dtype=float)
            pred_vals = np.array([rank_pos_pred[oid] for oid in common_ids], dtype=float)
            # Spearman over rank positions (manual, no scipy dependency)
            true_centered = true_vals - true_vals.mean()
            pred_centered = pred_vals - pred_vals.mean()
            denom = np.sqrt((true_centered ** 2).sum() * (pred_centered ** 2).sum()) + 1e-8
            spearman = float((true_centered * pred_centered).sum() / denom)
        else:
            spearman = 0.0

        # Kendall-like pair agreement between predicted and actual full ranking
        concordant = 0.0
        total = 0.0
        for i, a in enumerate(common_ids):
            for b in common_ids[i + 1:]:
                total += 1.0
                actual_pref = rank_pos_true[a] < rank_pos_true[b]
                pred_pref = rank_pos_pred[a] < rank_pos_pred[b]
                if actual_pref == pred_pref:
                    concordant += 1.0
        kendall_pair_agreement = float(concordant / total) if total > 0 else 0.0

        return {
            "pairwise_accuracy_before_update": float(pairwise_acc),
            "ranking_log_likelihood_before_update": float(log_lik),
            "mean_pair_margin_before_update": float(mean_margin),
            "predicted_top_option_before_update": float(predicted_top),
            "actual_top_option": float(ranking[0]) if ranking else -1.0,
            "predicted_ranking_before_update": [int(x) for x in predicted_ranking],
            "actual_ranking": [int(x) for x in ranking],
            "spearman_rank_corr_before_update": float(spearman),
            "kendall_pair_agreement_before_update": float(kendall_pair_agreement),
        }
    
    def _process_pairwise_comparison(
        self,
        better_id: int,
        worse_id: int,
        preference_strength: float,
    ) -> Dict[str, Dict[str, float]]:
        """
        Process a single pairwise comparison and compute tag updates.
        
        Only updates tags that differ between the two options.
        """
        option_map = {opt.option_id: opt for opt in self.current_options}
        better_opt = option_map[better_id]
        worse_opt = option_map[worse_id]
        
        better_tags = set(better_opt.tag_ids)
        worse_tags = set(worse_opt.tag_ids)
        
        # Only update differentiating tags
        tags_only_in_better = better_tags - worse_tags
        tags_only_in_worse = worse_tags - better_tags
        
        updates = {}
        lr = self.learning_rate
        shrink = self.config.sigma_shrink_factor
        
        for tag_id in tags_only_in_better:
            updates[tag_id] = {
                "mu_delta": lr * preference_strength,
                "sigma_factor": shrink,
            }
        
        for tag_id in tags_only_in_worse:
            updates[tag_id] = {
                "mu_delta": -lr * preference_strength,
                "sigma_factor": shrink,
            }
        
        return updates
    
    def _log_top_tags(self, n: int = 10):
        """Log current top tags by mean utility."""
        sorted_tags = sorted(
            self.tags.values(),
            key=lambda t: t.mu,
            reverse=True
        )[:n]
        
        print(f"\n  Top {n} tags by μ:")
        for i, tag in enumerate(sorted_tags):
            print(f"    {i+1}. {tag.text[:30]:30s} μ={tag.mu:+.3f} σ={tag.sigma:.3f}")
    
    # -------------------------------------------------------------------------
    # Final Selection
    # -------------------------------------------------------------------------
    
    def get_final_selection(
        self,
        n_tags: int = 10,
        use_softmax: bool = True,
        temperature: float = 1.0,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Get final tag selection with weights after refinement.
        
        Args:
            n_tags: Number of tags to select
            use_softmax: Use softmax for weights (smoother distribution)
            temperature: Softmax temperature (lower = sharper)
        
        Returns:
            (tag_texts, weights_dict) where weights_dict maps tag_text -> weight
        """
        # Select top n tags by posterior mean
        sorted_tags = sorted(
            self.tags.values(),
            key=lambda t: t.mu,
            reverse=True
        )[:n_tags]
        
        tag_texts = [t.text for t in sorted_tags]
        mus = np.array([t.mu for t in sorted_tags])
        
        if use_softmax:
            # Softmax normalization
            exp_mus = np.exp(mus / temperature)
            weights = exp_mus / exp_mus.sum()
        else:
            # Simple normalization (shift to positive first)
            shifted = mus - mus.min() + 0.1
            weights = shifted / shifted.sum()
        
        weights_dict = {
            text: float(w) for text, w in zip(tag_texts, weights)
        }
        
        return tag_texts, weights_dict
    
    def get_final_result(self) -> Dict:
        """Get comprehensive final result."""
        tag_texts, weights = self.get_final_selection()
        
        # Build detailed tag info
        tag_details = []
        for tag_id, tag in sorted(
            self.tags.items(),
            key=lambda x: x[1].mu,
            reverse=True
        ):
            tag_details.append({
                "text": tag.text,
                "final_mu": round(tag.mu, 4),
                "final_sigma": round(tag.sigma, 4),
                "category": tag.category.value,
                "times_shown": tag.times_shown,
                "win_rate": (
                    tag.times_in_winner / tag.times_shown 
                    if tag.times_shown > 0 else 0
                ),
            })
        
        result = {
            "final_tags": tag_texts,
            "weights": weights,
            "all_tag_details": tag_details,
            "rounds_completed": self.current_round,
            "total_comparisons": sum(
                len(r.pairwise_comparisons) for r in self.round_history
            ),
            "round_history": [r.to_dict() for r in self.round_history],
        }
        
        # === LOGGING ===
        if self.logger:
            self.logger.log_final_selection(
                final_tags=tag_texts,
                weights=weights,
                tag_states=self.tags,
            )
            # Save the log
            self.logger.save()
        
        return result
    
    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    
    def save_state(self, path: Path) -> None:
        """Save refiner state to disk."""
        state = {
            "config": {
                "max_rounds": self.config.max_rounds,
                "tags_per_option": self.config.tags_per_option,
                "base_learning_rate": self.config.base_learning_rate,
            },
            "current_round": self.current_round,
            "tags": {tid: tag.to_dict() for tid, tag in self.tags.items()},
            "positive_tag_ids": self.positive_tag_ids,
            "neutral_tag_ids": self.neutral_tag_ids,
            "round_history": [r.to_dict() for r in self.round_history],
            "saved_at": datetime.now().isoformat(),
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"[GPRefiner] State saved to {path}")
    
    def load_state(self, path: Path) -> None:
        """Load refiner state from disk."""
        with open(path) as f:
            state = json.load(f)
        
        self.current_round = state["current_round"]
        self.positive_tag_ids = state["positive_tag_ids"]
        self.neutral_tag_ids = state["neutral_tag_ids"]
        
        # Restore tags
        self.tags.clear()
        for tag_id, tag_data in state["tags"].items():
            self.tags[tag_id] = TagState(
                tag_id=tag_data["tag_id"],
                text=tag_data["text"],
                category=TagCategory(tag_data["category"]),
                mu=tag_data["mu"],
                sigma=tag_data["sigma"],
                times_shown=tag_data.get("times_shown", 0),
                times_in_winner=tag_data.get("times_in_winner", 0),
                times_in_loser=tag_data.get("times_in_loser", 0),
            )
        
        print(f"[GPRefiner] State loaded from {path}")
        print(f"  Round: {self.current_round}, Tags: {len(self.tags)}")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_refiner_from_exploration(
    exploration_result_path: str,
    selected_image_indices: List[int] = None,
) -> TagGPRefiner:
    """
    Create a TagGPRefiner from exploration stage output.
    
    Args:
        exploration_result_path: Path to concept_weights.json or tag_preferences.json
        selected_image_indices: Which images the user selected during exploration
    
    Returns:
        Initialized TagGPRefiner
    """
    with open(exploration_result_path) as f:
        data = json.load(f)
    
    # Extract positive and neutral tags
    if "concept_weights" in data:
        # concept_weights.json format
        positive_tags = []
        neutral_tags = []
        
        for cw in data["concept_weights"]:
            if cw.get("category") == "positive" or cw.get("score", 0) > 0.5:
                positive_tags.append(cw["label"])
            elif cw.get("category") == "neutral" or -0.5 <= cw.get("score", 0) <= 0.5:
                neutral_tags.append(cw["label"])
    
    elif "positive" in data:
        # tag_preferences.json format
        positive_tags = data["positive"]
        neutral_tags = data.get("neutral", [])
    
    else:
        raise ValueError("Unrecognized file format")
    
    # For now, assume half of positive tags were in selected images
    # (In real usage, this should come from actual selection tracking)
    selected_image_tags = set(positive_tags[:len(positive_tags)//2])
    
    refiner = TagGPRefiner()
    refiner.initialize_from_exploration(
        positive_tags=positive_tags,
        neutral_tags=neutral_tags,
        selected_image_tags=selected_image_tags,
    )
    
    return refiner


# ============================================================================
# Demo / Test
# ============================================================================

def demo():
    """Demonstrate the GP refinement process."""
    print("=" * 60)
    print("Tag GP Refinement Demo")
    print("=" * 60)
    
    # Simulate exploration output
    positive_tags = [
        "warm lighting", "cozy atmosphere", "natural wood", "soft textures",
        "earth tones", "ambient glow", "comfortable furniture", "plants",
        "minimalist decor", "open space", "large windows", "neutral colors",
    ]
    
    neutral_tags = [
        "modern style", "industrial elements", "bold accents", "geometric patterns",
        "metallic surfaces", "dark corners", "statement art", "layered rugs",
    ]
    
    # Assume first 6 positive tags were in selected images
    selected_tags = set(positive_tags[:6])
    
    # Initialize refiner
    refiner = TagGPRefiner()
    init_result = refiner.initialize_from_exploration(
        positive_tags=positive_tags,
        neutral_tags=neutral_tags,
        selected_image_tags=selected_tags,
    )
    
    print(f"\nInitialization: {init_result}")
    
    # Simulate 6 rounds
    for round_num in range(1, 7):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}")
        print("=" * 60)
        
        # Generate options
        options = refiner.generate_round_options()
        
        print("\nOptions presented:")
        for opt in options:
            print(f"\n  Option {opt.option_id} ({opt.strategy.value}):")
            print(f"    Tags: {opt.tags}")
        
        # Simulate user ranking (prefer options with "warm" and "cozy" tags)
        def score_option(opt):
            score = 0
            for tag in opt.tags:
                if "warm" in tag.lower() or "cozy" in tag.lower():
                    score += 2
                if "natural" in tag.lower() or "soft" in tag.lower():
                    score += 1
                if "industrial" in tag.lower() or "bold" in tag.lower():
                    score -= 1
            return score + np.random.normal(0, 0.5)  # Add noise
        
        scored = [(opt.option_id, score_option(opt)) for opt in options]
        scored.sort(key=lambda x: x[1], reverse=True)
        ranking = [opt_id for opt_id, _ in scored]
        
        print(f"\nSimulated user ranking: {ranking}")
        
        # Submit ranking
        result = refiner.submit_ranking(ranking)
    
    # Get final result
    print(f"\n{'='*60}")
    print("FINAL RESULT")
    print("=" * 60)
    
    final = refiner.get_final_result()
    print(f"\nFinal tags: {final['final_tags']}")
    print(f"\nWeights:")
    for tag, weight in final['weights'].items():
        print(f"  {tag}: {weight:.3f}")
    
    print(f"\nTotal comparisons: {final['total_comparisons']}")


if __name__ == "__main__":
    demo()
