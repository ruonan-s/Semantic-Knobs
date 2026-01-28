"""
Stage 4: Attention Gradient Weight Refinement
Fine-tune relative weights of winning tags from elimination phase.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np


@dataclass
class WeightState:
    """Current state of tag weights."""
    weights: dict[str, float]
    history: list[dict[str, float]] = field(default_factory=list)
    
    def normalize(self, target_sum: float = None):
        """Normalize weights."""
        if target_sum is None:
            target_sum = len(self.weights) * 0.8  # Average of 0.8 per tag
        
        current_sum = sum(self.weights.values())
        if current_sum > 0:
            factor = target_sum / current_sum
            self.weights = {k: v * factor for k, v in self.weights.items()}
    
    def get_sorted_weights(self) -> list[tuple[str, float]]:
        """Get weights sorted by value descending."""
        return sorted(self.weights.items(), key=lambda x: x[1], reverse=True)


@dataclass
class RefinementRoundConfig:
    """Configuration for a refinement round."""
    round_num: int
    images: list[dict]  # Each has 'weights' and 'prompt' and 'strategy'


@dataclass
class RefinementFeedback:
    """User feedback from a refinement round."""
    selected_image_idx: int
    attention_maps: list[dict[str, float]]


class AttentionGradientRefiner:
    """
    Refines tag weights using attention gradients from user preferences.
    """
    
    def __init__(
        self,
        base_prompt: str,
        tags: list[str],
        initial_weights: Optional[dict[str, float]] = None,
        learning_rate: float = 0.12,
        momentum: float = 0.25,
        convergence_threshold: float = 0.04,
        max_rounds: int = 4
    ):
        self.base_prompt = base_prompt
        self.tags = tags
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.convergence_threshold = convergence_threshold
        self.max_rounds = max_rounds
        
        # Initialize weights
        if initial_weights:
            self.state = WeightState(weights=initial_weights.copy())
        else:
            self.state = WeightState(weights={tag: 1.0 for tag in tags})
        
        self.state.normalize()
        
        # For momentum
        self.prev_gradients: dict[str, float] = {tag: 0.0 for tag in tags}
        
        # Track rounds
        self.round_history: list[dict] = []
        self.current_round = 0
    
    def generate_round(self) -> RefinementRoundConfig:
        """
        Generate 4 image configurations with weight variations.
        
        Strategy:
        - Image A: Current best weights (exploitation)
        - Image B: Exaggerate differences (amplify high, dampen low)
        - Image C: Exploration (boost low-weighted tags)
        - Image D: Random perturbation
        """
        self.current_round += 1
        
        base_weights = self.state.weights.copy()
        mean_weight = np.mean(list(base_weights.values()))
        sorted_tags = sorted(self.tags, key=lambda t: base_weights[t])
        
        low_tags = set(sorted_tags[:len(self.tags)//2])
        high_tags = set(sorted_tags[len(self.tags)//2:])
        
        images = []
        
        # Image A: Current weights
        images.append({
            "image_id": 0,
            "strategy": "current_best",
            "weights": base_weights.copy(),
            "prompt": self._build_weighted_prompt(base_weights)
        })
        
        # Image B: Exaggerated
        exaggerated = {}
        for tag, w in base_weights.items():
            new_w = mean_weight + 1.5 * (w - mean_weight)
            exaggerated[tag] = np.clip(new_w, 0.2, 1.5)
        images.append({
            "image_id": 1,
            "strategy": "exaggerated",
            "weights": exaggerated,
            "prompt": self._build_weighted_prompt(exaggerated)
        })
        
        # Image C: Exploration (boost low)
        exploration = {}
        for tag, w in base_weights.items():
            if tag in low_tags:
                exploration[tag] = min(w + 0.3, 1.4)
            else:
                exploration[tag] = max(w - 0.15, 0.3)
        images.append({
            "image_id": 2,
            "strategy": "exploration",
            "weights": exploration,
            "prompt": self._build_weighted_prompt(exploration)
        })
        
        # Image D: Perturbed
        perturbed = {}
        for tag, w in base_weights.items():
            noise = np.random.normal(0, 0.15)
            perturbed[tag] = np.clip(w + noise, 0.2, 1.5)
        images.append({
            "image_id": 3,
            "strategy": "perturbed",
            "weights": perturbed,
            "prompt": self._build_weighted_prompt(perturbed)
        })
        
        return RefinementRoundConfig(
            round_num=self.current_round,
            images=images
        )
    
    def _build_weighted_prompt(self, weights: dict[str, float]) -> str:
        """Build prompt with weight emphasis notation."""
        sorted_tags = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        parts = []
        for tag, weight in sorted_tags:
            if weight >= 1.1:
                parts.append(f"({tag}:{weight:.2f})")
            elif weight >= 0.7:
                parts.append(tag)
            else:
                parts.append(f"[{tag}:{weight:.2f}]")
        
        return f"{self.base_prompt}, {', '.join(parts)}"
    
    def process_feedback(
        self,
        round_config: RefinementRoundConfig,
        feedback: RefinementFeedback
    ) -> dict:
        """
        Process feedback and update weights using attention gradient.
        
        Returns dict with update info and convergence status.
        """
        selected_idx = feedback.selected_image_idx
        selected_image = round_config.images[selected_idx]
        selected_attention = feedback.attention_maps[selected_idx]
        
        # Compute attention gradient
        gradients = {}
        for tag in self.tags:
            # Attention in selected image
            selected_attn = selected_attention.get(tag, 0.0)
            
            # Mean attention in rejected images
            rejected_attns = [
                feedback.attention_maps[i].get(tag, 0.0)
                for i in range(4) if i != selected_idx
            ]
            rejected_mean = np.mean(rejected_attns) if rejected_attns else 0.0
            
            # Gradient = how much more attention in selected vs rejected
            raw_gradient = selected_attn - rejected_mean
            
            # Scale by current weight (prevent runaway)
            current_weight = self.state.weights[tag]
            scaled_gradient = raw_gradient * (0.5 + 0.5 * current_weight)
            
            gradients[tag] = scaled_gradient
        
        # Apply momentum
        momentum_gradients = {}
        for tag in self.tags:
            momentum_gradients[tag] = (
                (1 - self.momentum) * gradients[tag] +
                self.momentum * self.prev_gradients.get(tag, 0.0)
            )
        
        self.prev_gradients = momentum_gradients.copy()
        
        # Save current weights for history
        self.state.history.append(self.state.weights.copy())
        
        # Update weights
        weight_updates = {}
        for tag in self.tags:
            old_weight = self.state.weights[tag]
            new_weight = old_weight + self.learning_rate * momentum_gradients[tag]
            new_weight = np.clip(new_weight, 0.2, 1.5)
            
            weight_updates[tag] = new_weight - old_weight
            self.state.weights[tag] = new_weight
        
        # Normalize
        self.state.normalize()
        
        # Check convergence
        max_change = max(abs(v) for v in weight_updates.values())
        is_converged = max_change < self.convergence_threshold
        
        # Record round
        self.round_history.append({
            "round_num": self.current_round,
            "selected_idx": selected_idx,
            "selected_strategy": selected_image["strategy"],
            "gradients": gradients,
            "weight_updates": weight_updates,
            "max_change": max_change,
            "is_converged": is_converged
        })
        
        return {
            "weight_updates": weight_updates,
            "max_change": max_change,
            "is_converged": is_converged,
            "current_weights": self.state.weights.copy()
        }
    
    def check_convergence(self) -> bool:
        """Check if weights have converged."""
        if len(self.state.history) < 2:
            return False
        
        prev_weights = self.state.history[-1]
        max_change = max(
            abs(self.state.weights[tag] - prev_weights.get(tag, 0))
            for tag in self.tags
        )
        
        return max_change < self.convergence_threshold
    
    def get_final_result(self) -> dict:
        """Get final optimized weights."""
        sorted_weights = self.state.get_sorted_weights()
        
        return {
            "base_prompt": self.base_prompt,
            "optimized_tags": dict(sorted_weights),
            "final_prompt": self._build_weighted_prompt(self.state.weights),
            "rounds_used": len(self.round_history),
            "convergence_history": [
                {"round": r["round_num"], "max_change": r["max_change"]}
                for r in self.round_history
            ]
        }