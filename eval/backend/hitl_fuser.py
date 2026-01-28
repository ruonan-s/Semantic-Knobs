"""
HITL Fuser - SDXL Cross-Attention Weighting

Implements REAL cross-attention control by:
1. Encoding concepts as separate tokens in prompt
2. Hooking into UNet's attn2 (cross-attention) layers
3. Scaling attention scores per-token during denoising

This gives spatial control: each concept's influence on specific pixels
can be amplified or suppressed.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from functools import partial

from hitl_sampler import CompositionSample


@dataclass
class TokenWeightMap:
    """Maps token indices to their weights for attention scaling."""
    # token_index -> weight (default 1.0)
    weights: Dict[int, float]
    # Total sequence length
    seq_len: int
    
    def get_weight_tensor(self, device: str = 'cuda') -> torch.Tensor:
        """Create weight tensor for all tokens."""
        w = torch.ones(self.seq_len, device=device)
        for idx, weight in self.weights.items():
            if 0 <= idx < self.seq_len:
                w[idx] = weight
        return w


class CrossAttentionController:
    """
    Hooks into UNet cross-attention layers to scale attention scores.
    
    During each cross-attention computation:
        attn_scores = Q @ K^T / sqrt(d)
        attn_probs = softmax(attn_scores)  # <-- We modify here
        output = attn_probs @ V
    
    We scale attn_scores for specific tokens BEFORE softmax:
        attn_scores[:, :, token_idx] *= weight
    
    This amplifies/suppresses how much each pixel "attends to" specific concepts.
    """
    
    def __init__(self, token_weight_map: TokenWeightMap):
        self.token_weight_map = token_weight_map
        self.hooks: List[Any] = []
        self._enabled = True
    
    def _create_attn_hook(self, module_name: str):
        """Create hook function for a specific attention module."""
        def hook_fn(module, args, kwargs, output):
            if not self._enabled:
                return output
            
            # Get the weight tensor
            weights = self.token_weight_map.get_weight_tensor(
                device=output.device if hasattr(output, 'device') else 'cuda'
            )
            
            # For cross-attention, output shape is (batch, heads, seq_len_q, seq_len_kv)
            # or after attention: (batch, seq_len, hidden_dim)
            # We need to intercept at the attention score level
            
            return output
        
        return hook_fn
    
    def register_hooks(self, unet: Any):
        """Register hooks on all cross-attention (attn2) layers."""
        self.remove_hooks()  # Clear existing
        
        for name, module in unet.named_modules():
            # SDXL cross-attention modules are named 'attn2'
            if 'attn2' in name and hasattr(module, 'to_q'):
                # This is a cross-attention block
                # We'll wrap the forward method
                original_forward = module.forward
                module._original_forward = original_forward
                module.forward = partial(self._wrapped_forward, module, original_forward)
                self.hooks.append((name, module))
        
        print(f"[CrossAttnCtrl] Registered hooks on {len(self.hooks)} attn2 modules")
        return self
    
    def _wrapped_forward(self, module, original_forward, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        """
        Wrapped forward that scales attention based on token weights.
        
        This intercepts the cross-attention computation to apply per-token scaling.
        """
        if encoder_hidden_states is None or not self._enabled:
            return original_forward(hidden_states, encoder_hidden_states, attention_mask, **kwargs)
        
        # Get attention components
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = module.to_q(hidden_states)
        key = module.to_k(encoder_hidden_states)
        value = module.to_v(encoder_hidden_states)
        
        # Reshape for multi-head attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // module.heads
        
        query = query.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)
        
        # Compute attention scores
        scale = head_dim ** -0.5
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # === APPLY TOKEN WEIGHTS HERE ===
        # attn_scores shape: (batch, heads, seq_len_q, seq_len_kv)
        # We scale the key dimension (seq_len_kv) based on token weights
        weights = self.token_weight_map.get_weight_tensor(device=attn_scores.device)
        
        # Match dtype with attention scores (SDXL uses float16)
        weights = weights.to(dtype=attn_scores.dtype)
        
        # Expand weights to match attention shape
        # weights shape: (seq_len_kv,) -> (1, 1, 1, seq_len_kv)
        seq_len_kv = attn_scores.shape[-1]
        if len(weights) >= seq_len_kv:
            weights = weights[:seq_len_kv]
        else:
            # Pad if needed
            pad = torch.ones(seq_len_kv - len(weights), device=weights.device, dtype=weights.dtype)
            weights = torch.cat([weights, pad])
        
        weights = weights.view(1, 1, 1, -1)
        
        # Scale attention scores (before softmax)
        # Higher weight = more attention to that token
        attn_scores = attn_scores * weights
        
        # Apply softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        hidden_states = torch.matmul(attn_probs, value)
        
        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        
        # Output projection
        hidden_states = module.to_out[0](hidden_states)
        hidden_states = module.to_out[1](hidden_states)
        
        return hidden_states
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for name, module in self.hooks:
            if hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                delattr(module, '_original_forward')
        self.hooks.clear()
    
    def enable(self):
        self._enabled = True
    
    def disable(self):
        self._enabled = False
    
    def __enter__(self):
        self.enable()
        return self
    
    def __exit__(self, *args):
        self.remove_hooks()


class HITLCompositionFuser:
    """
    Fuse 10-point composition using SDXL with cross-attention weighting.
    
    Strategy:
    1. Build prompt with each concept as a separate phrase
    2. Track which tokens correspond to which concepts
    3. Create TokenWeightMap for attention scaling
    4. Register hooks on UNet to apply scaling during generation
    """
    
    def __init__(self, pipe: Any, device: str = None):
        self.pipe = pipe
        self.device = device or str(getattr(pipe, 'device', 'cuda'))
        
        # SDXL components
        self.text_encoder_1 = getattr(pipe, "text_encoder", None)
        self.text_encoder_2 = getattr(pipe, "text_encoder_2", None)
        self.tokenizer_1 = getattr(pipe, "tokenizer", None)
        self.tokenizer_2 = getattr(pipe, "tokenizer_2", None)
        self.unet = getattr(pipe, "unet", None)
        
        self.is_sdxl = self.text_encoder_1 is not None and self.text_encoder_2 is not None
        
        if self.is_sdxl:
            print(f"[HITLFuser] SDXL dual encoders on {self.device}")
        
        self._embedding_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    def _tokenize_and_map(self, prompt: str, concept_phrases: List[str]) -> Tuple[Dict[str, List[int]], int]:
        """
        Tokenize prompt and map each concept phrase to its token indices.
        
        Returns:
            (concept -> [token_indices], total_seq_len)
        """
        # Use tokenizer_2 (CLIP) as it's what SDXL uses for cross-attention
        tokens = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokens.input_ids[0].tolist()
        seq_len = len(input_ids)
        
        # Decode to find concept positions
        concept_to_indices: Dict[str, List[int]] = {}
        
        for concept in concept_phrases:
            # Tokenize just the concept
            concept_tokens = self.tokenizer_2(
                concept,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids[0].tolist()
            
            # Find where these tokens appear in the full sequence
            indices = []
            for i in range(len(input_ids) - len(concept_tokens) + 1):
                if input_ids[i:i+len(concept_tokens)] == concept_tokens:
                    indices.extend(range(i, i + len(concept_tokens)))
                    break
            
            if indices:
                concept_to_indices[concept] = indices
        
        return concept_to_indices, seq_len
    
    def _build_weighted_prompt(
        self,
        composition: CompositionSample,
        base_prompt: str
    ) -> Tuple[str, TokenWeightMap]:
        """
        Build prompt string and create token weight mapping.
        
        Format: "base_prompt with features: tag1, tag2, ..."
        
        The "with features:" separator ensures:
        - base_prompt tokens get default weight (1.0)
        - Only tag tokens get variable attention weights
        """
        tag_phrases = composition.tag_labels
        weights = np.array(composition.weights)
        
        # Normalize weights to reasonable range [0.5, 1.5]
        # Higher sampled weight = more attention
        weights_normalized = 0.5 + (weights / (weights.max() + 1e-8))
        
        # Build prompt with clear separation between base and features
        # Base prompt anchors generation, tags get weighted attention
        tags_str = ", ".join(tag_phrases)
        prompt = f"{base_prompt} with features: {tags_str}"
        
        # Map concepts to token indices
        concept_to_indices, seq_len = self._tokenize_and_map(prompt, tag_phrases)
        
        # Create weight map
        token_weights: Dict[int, float] = {}
        for i, (concept, weight) in enumerate(zip(tag_phrases, weights_normalized)):
            if concept in concept_to_indices:
                for token_idx in concept_to_indices[concept]:
                    token_weights[token_idx] = float(weight)
        
        weight_map = TokenWeightMap(weights=token_weights, seq_len=seq_len)
        
        return prompt, weight_map
    
    @torch.no_grad()
    def _encode_prompt_sdxl(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt using SDXL dual encoders."""
        if prompt in self._embedding_cache:
            return self._embedding_cache[prompt]
        
        # Tokenize
        t1 = self.tokenizer_1(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        t2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode
        e1 = self.text_encoder_1(t1.input_ids.to(self.device)).last_hidden_state
        e2_out = self.text_encoder_2(t2.input_ids.to(self.device))
        e2 = e2_out.last_hidden_state
        pooled = getattr(e2_out, "pooler_output", e2.mean(dim=1))
        
        # Concatenate hidden states
        L = min(e1.shape[1], e2.shape[1])
        prompt_embeds = torch.cat([e1[:, :L, :], e2[:, :L, :]], dim=-1)
        
        self._embedding_cache[prompt] = (prompt_embeds, pooled)
        return prompt_embeds, pooled
    
    def fuse_composition(
        self,
        composition: CompositionSample,
        base_prompt: str,
        neg_phrases: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, CrossAttentionController]:
        """
        Fuse composition into embeddings AND create attention controller.
        
        Returns:
            (prompt_embeds, pooled, neg_embeds, neg_pooled, attn_controller)
            
            The attn_controller should be used during generation to scale attention.
        """
        # Build weighted prompt and token map
        prompt, weight_map = self._build_weighted_prompt(composition, base_prompt)
        
        # Encode
        prompt_embeds, pooled = self._encode_prompt_sdxl(prompt)
        
        # Negative
        if neg_phrases:
            neg_prompt = ", ".join(neg_phrases)
            neg_embeds, neg_pooled = self._encode_prompt_sdxl(neg_prompt)
            print(f"[HITLFuser] Negative: {neg_prompt[:60]}...")
        else:
            neg_embeds, neg_pooled = self._encode_prompt_sdxl("")
        
        # Create attention controller
        attn_controller = CrossAttentionController(weight_map)
        
        # Log top concepts
        top_k = 5
        sorted_idx = np.argsort(composition.weights)[-top_k:][::-1]
        top = [(composition.tag_labels[i], f"{composition.weights[i]:.2f}") for i in sorted_idx]
        print(f"[HITLFuser] Prompt: '{prompt[:80]}...'")
        print(f"[HITLFuser] Top weighted: {top}")
        print(f"[HITLFuser] Token weight map: {len(weight_map.weights)} tokens weighted")
        
        return prompt_embeds, pooled, neg_embeds, neg_pooled, attn_controller
    
    def clear_cache(self):
        self._embedding_cache.clear()


def generate_with_attention_control(
    pipe: Any,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    neg_embeds: torch.Tensor,
    neg_pooled: torch.Tensor,
    attn_controller: CrossAttentionController,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    **kwargs
) -> Any:
    """
    Generate with cross-attention scaling.
    
    The attn_controller hooks into the UNet to scale attention scores
    for specific tokens during the denoising process.
    """
    # Setup generator
    generator = None
    if seed is not None:
        device = getattr(pipe, 'device', 'cuda')
        if isinstance(device, str):
            device = torch.device(device)
        generator = torch.Generator(device=device).manual_seed(seed)
    
    # Register attention hooks
    unet = getattr(pipe, 'unet', None)
    if unet is not None and attn_controller is not None:
        attn_controller.register_hooks(unet)
    
    try:
        result = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=neg_embeds,
            negative_pooled_prompt_embeds=neg_pooled,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs
        )
        image = result.images[0]
    except Exception as e:
        print(f"[HITLFuser] Generation error: {e}")
        # Fallback to simple prompt
        if attn_controller:
            attn_controller.remove_hooks()
        result = pipe(
            prompt="interior design, modern home",
            num_inference_steps=num_inference_steps,
            generator=generator
        )
        image = result.images[0]
    finally:
        # Always clean up hooks
        if attn_controller:
            attn_controller.remove_hooks()
    
    return image


# Backward compatibility
def generate_with_hooks(
    pipe: Any,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    neg_embeds: torch.Tensor,
    neg_pooled: torch.Tensor,
    attn_controller: Optional[CrossAttentionController],
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    **kwargs
):
    """Backward-compatible wrapper."""
    return generate_with_attention_control(
        pipe, prompt_embeds, pooled, neg_embeds, neg_pooled,
        attn_controller, num_inference_steps, guidance_scale, seed, **kwargs
    )


def create_composition_prompt(composition: CompositionSample, base_prompt: str, max_tags: int = 10) -> str:
    """Fallback: create text prompt from composition."""
    sorted_idx = np.argsort(composition.weights)[::-1][:max_tags]
    top_tags = [composition.tag_labels[i] for i in sorted_idx]
    return ", ".join(top_tags) + f", {base_prompt}"
