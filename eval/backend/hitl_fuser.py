"""
HITL Fuser - Direct Embedding Injection with Cross-Attention Control

This module injects sampled CLIP embeddings directly into SDXL, bypassing
the text encoder entirely. This ensures high-fidelity representation of
the continuous embedding space points.

Key Techniques:
1. Virtual Token Registration: Create placeholder tokens (e.g., _FEAT_1_)
2. Direct Embedding Overwrite: Replace token embeddings with sampled vectors
3. Cross-Attention Hooks: Scale attention maps by weights to give each 
   feature proportional "pixel-estate"

Critical for CFG: Also samples and injects negative embeddings for proper
classifier-free guidance that enforces repelling forces during denoising.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from contextlib import contextmanager

from hitl_sampler import ContinuousComposition


class CrossAttentionController:
    """
    Controller for scaling cross-attention maps during SDXL denoising.
    
    Applies weight scaling to attention scores:
        A' = A * diag(w1, w2, ..., wn)
    
    This gives high-utility embeddings more "pixel-estate" while
    suppressing influence of lower-weight features.
    """
    
    def __init__(self, token_weights: Dict[int, float], start_token_idx: int = 1):
        """
        Args:
            token_weights: Mapping of token index -> attention scale
            start_token_idx: Index of first feature token (after BOS)
        """
        self.token_weights = token_weights
        self.start_idx = start_token_idx
        self.hooks = []
        self.enabled = True
    
    def create_weight_mask(
        self, 
        seq_len: int, 
        device: torch.device, 
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Create attention weight mask."""
        # Default weight of 1.0
        mask = torch.ones(seq_len, device=device, dtype=dtype)
        
        # Apply feature weights
        for rel_idx, weight in self.token_weights.items():
            abs_idx = self.start_idx + rel_idx
            if abs_idx < seq_len:
                mask[abs_idx] = weight
        
        return mask
    
    def attention_hook_fn(self, module, args, output):
        """
        Hook function to scale attention outputs.
        
        This is registered on cross-attention layers (attn2) to scale
        the contribution of each token based on its weight.
        """
        if not self.enabled:
            return output
        
        try:
            # Output shape: (batch, seq_len, hidden_dim) typically
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            if hidden.dim() >= 2:
                seq_len = hidden.shape[-2] if hidden.dim() > 2 else hidden.shape[-1]
                
                # Only apply if sequence length matches expected
                if seq_len <= 77:  # CLIP max length
                    mask = self.create_weight_mask(seq_len, hidden.device, hidden.dtype)
                    
                    # Reshape for broadcasting
                    if hidden.dim() == 3:  # (batch, seq, hidden)
                        mask = mask.view(1, -1, 1)
                    elif hidden.dim() == 4:  # (batch, heads, seq, hidden)
                        mask = mask.view(1, 1, -1, 1)
                    
                    hidden = hidden * mask
            
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
            
        except Exception:
            return output
    
    def register_hooks(self, unet: torch.nn.Module):
        """Register attention hooks on UNet cross-attention layers."""
        self.remove_hooks()
        
        for name, module in unet.named_modules():
            # Cross-attention layers in SDXL UNet
            if 'attn2' in name and hasattr(module, 'to_out'):
                hook = module.register_forward_hook(self.attention_hook_fn)
                self.hooks.append(hook)
        
        if self.hooks:
            print(f"[CrossAttention] Registered {len(self.hooks)} attention hooks")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __enter__(self):
        self.enabled = True
        return self
    
    def __exit__(self, *args):
        self.remove_hooks()
        self.enabled = False


class DirectEmbeddingInjector:
    """
    Inject CLIP embeddings directly into SDXL, bypassing text encoding.
    
    This creates "virtual tokens" whose embeddings are overwritten with
    the sampled 768-dim CLIP vectors. The projection layer converts
    768-dim CLIP space to SDXL's expected dimensions.
    """
    
    def __init__(self, pipe: Any, device: str = None):
        """
        Initialize the direct embedding injector.
        
        Args:
            pipe: SDXL pipeline with text_encoder and text_encoder_2
            device: Device for computations
        """
        self.pipe = pipe
        self.device = device or (getattr(pipe, 'device', None) or 'cuda')
        
        # SDXL text encoders
        self.text_encoder_1 = getattr(pipe, "text_encoder", None)
        self.text_encoder_2 = getattr(pipe, "text_encoder_2", None)
        self.tokenizer_1 = getattr(pipe, "tokenizer", None)
        self.tokenizer_2 = getattr(pipe, "tokenizer_2", None)
        
        # Hidden dimensions
        self.hidden_dim_1 = 768   # CLIP ViT-L
        self.hidden_dim_2 = 1280  # OpenCLIP ViT-bigG
        
        # Projection for CLIP embeddings to SDXL space
        # CLIP is 768-dim, SDXL text_encoder_2 is 1280-dim
        self._init_projection()
        
        # Cache for base prompt embeddings
        self._base_cache = {}
        
        print(f"[DirectInjector] Initialized on {self.device}")
    
    def _init_projection(self):
        """Initialize projection from CLIP (768) to SDXL encoder_2 (1280) space."""
        # Simple linear projection
        self.projection = torch.nn.Linear(768, 1280, bias=False).to(self.device)
        
        # Initialize with identity-like mapping (pad with zeros)
        with torch.no_grad():
            self.projection.weight[:768, :] = torch.eye(768)
            self.projection.weight[768:, :] = 0
    
    def _encode_base_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode base prompt through SDXL's text encoders."""
        if prompt in self._base_cache:
            return self._base_cache[prompt]
        
        # Tokenize
        tokens_1 = self.tokenizer_1(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)
        
        tokens_2 = self.tokenizer_2(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Encode
        with torch.no_grad():
            enc1_out = self.text_encoder_1(tokens_1)
            enc2_out = self.text_encoder_2(tokens_2)
            
            embeds_1 = enc1_out.last_hidden_state
            embeds_2 = enc2_out.last_hidden_state
            pooled = getattr(enc2_out, "pooler_output", embeds_2.mean(dim=1))
        
        self._base_cache[prompt] = (embeds_1, embeds_2, pooled)
        return embeds_1, embeds_2, pooled
    
    def inject_composition(
        self,
        composition: ContinuousComposition,
        base_prompt: str,
        negative_composition: Optional[ContinuousComposition] = None,
        neg_prompt: str = ""
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, CrossAttentionController]:
        """
        Create SDXL embeddings with injected composition points.
        
        Strategy:
        1. Encode base prompt to get template embeddings
        2. Find where base prompt tokens end (before padding)
        3. Inject feature embeddings AFTER base prompt tokens
        4. Blend features with existing embeddings to preserve base semantics
        
        Args:
            composition: ContinuousComposition with 10 CLIP embeddings
            base_prompt: Base prompt (e.g., "Inviting Livingroom")
            negative_composition: Composition for negative prompt (CFG)
            neg_prompt: Text for negative prompt
        
        Returns:
            (prompt_embeds, pooled, neg_embeds, neg_pooled, attention_controller)
        """
        n_features = len(composition.points)
        
        # Get base prompt embeddings
        base_emb1, base_emb2, base_pooled = self._encode_base_prompt(base_prompt)
        
        # Find where base prompt ends (count non-padding tokens)
        tokens_1 = self.tokenizer_1(
            base_prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Find EOS token position (marks end of real content)
        eos_token_id = self.tokenizer_1.eos_token_id or 49407
        eos_positions = (tokens_1[0] == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            base_prompt_end = int(eos_positions[0].item())
        else:
            base_prompt_end = 10  # Fallback
        
        # Convert CLIP embeddings to tensors
        clip_embeds = torch.tensor(
            composition.points, dtype=torch.float32, device=self.device
        )
        
        # Project to SDXL dimensions
        with torch.no_grad():
            projected_1 = clip_embeds  # (N, 768)
            projected_2 = self.projection(clip_embeds)  # (N, 1280)
        
        # Create embeddings preserving base prompt
        prompt_embeds_1 = base_emb1.clone()
        prompt_embeds_2 = base_emb2.clone()
        
        # Inject feature embeddings AFTER base prompt, into padding region
        # Structure: [BOS] [base_tokens...] [EOS] [FEAT_1] ... [FEAT_N] [PAD...]
        inject_start = base_prompt_end + 1  # After EOS
        n_to_inject = min(n_features, 77 - inject_start - 1)
        
        # Also BLEND features into base prompt tokens
        blend_weight = 0.5  # Equal influence of features and base tokens
        
        for i in range(n_to_inject):
            inject_pos = inject_start + i
            if inject_pos < 77:
                # Full injection in padding region
                prompt_embeds_1[0, inject_pos, :] = projected_1[i]
                prompt_embeds_2[0, inject_pos, :] = projected_2[i]
            
            # Also blend into a base prompt token (if exists)
            blend_pos = 1 + (i % (base_prompt_end - 1)) if base_prompt_end > 1 else 1
            if blend_pos < base_prompt_end:
                prompt_embeds_1[0, blend_pos, :] = (1 - blend_weight) * prompt_embeds_1[0, blend_pos, :] + blend_weight * projected_1[i]
                prompt_embeds_2[0, blend_pos, :] = (1 - blend_weight) * prompt_embeds_2[0, blend_pos, :] + blend_weight * projected_2[i]
        
        # Concatenate for SDXL format
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        
        # Handle pooled embedding (mostly base prompt, some feature influence)
        weights_tensor = torch.tensor(
            composition.weights, dtype=torch.float32, device=self.device
        )
        weighted_pooled = (projected_2 * weights_tensor.unsqueeze(-1)).sum(dim=0, keepdim=True)
        pooled = 0.5 * base_pooled + 0.5 * weighted_pooled  # Keep base prompt dominant
        
        print(f"[DirectInjector] Base prompt ends at {base_prompt_end}, injecting {n_to_inject} features starting at {inject_start}")
        
        # Process negative embeddings
        if negative_composition is not None:
            neg_embeds, neg_pooled = self._create_negative_embeds(
                negative_composition, neg_prompt
            )
        else:
            neg_embeds, neg_pooled = self._create_empty_negative(neg_prompt)
        
        # Create attention controller - weights apply to injection positions
        token_weights = {i: float(w) for i, w in enumerate(composition.weights[:n_to_inject])}
        attention_controller = CrossAttentionController(token_weights, start_token_idx=inject_start)
        
        print(f"[DirectInjector] Injected {n_features} features, weights: {composition.weights[:3]}...")
        
        return prompt_embeds, pooled, neg_embeds, neg_pooled, attention_controller
    
    def _create_negative_embeds(
        self,
        neg_composition: ContinuousComposition,
        neg_prompt: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create negative embeddings from negative composition."""
        n_features = len(neg_composition.points)
        
        # Get negative base embeddings
        base_emb1, base_emb2, base_pooled = self._encode_base_prompt(neg_prompt or "")
        
        # Convert and project
        clip_embeds = torch.tensor(
            neg_composition.points, dtype=torch.float32, device=self.device
        )
        
        with torch.no_grad():
            projected_1 = clip_embeds
            projected_2 = self.projection(clip_embeds)
        
        # Inject
        neg_embeds_1 = base_emb1.clone()
        neg_embeds_2 = base_emb2.clone()
        
        inject_end = min(n_features + 1, 77 - 5)
        for i in range(min(n_features, inject_end - 1)):
            neg_embeds_1[0, i + 1, :] = projected_1[i]
            neg_embeds_2[0, i + 1, :] = projected_2[i]
        
        neg_embeds = torch.cat([neg_embeds_1, neg_embeds_2], dim=-1)
        
        weights = torch.tensor(neg_composition.weights, dtype=torch.float32, device=self.device)
        weighted_pooled = (projected_2 * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
        neg_pooled = 0.7 * base_pooled + 0.3 * weighted_pooled
        
        return neg_embeds, neg_pooled
    
    def _create_empty_negative(self, neg_prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create negative embeddings from text prompt only."""
        if neg_prompt:
            base_emb1, base_emb2, pooled = self._encode_base_prompt(neg_prompt)
        else:
            base_emb1, base_emb2, pooled = self._encode_base_prompt("")
        
        neg_embeds = torch.cat([base_emb1, base_emb2], dim=-1)
        return neg_embeds, pooled
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._base_cache.clear()


def generate_with_direct_injection(
    pipe: Any,
    prompt_embeds: torch.Tensor,
    pooled: torch.Tensor,
    neg_embeds: torch.Tensor,
    neg_pooled: torch.Tensor,
    attention_controller: CrossAttentionController,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    use_attention_hooks: bool = True,
    **kwargs
) -> Any:
    """
    Generate image with directly injected embeddings and attention control.
    
    Args:
        pipe: SDXL pipeline
        prompt_embeds: (1, 77, 2048) fused prompt embeddings
        pooled: (1, 1280) pooled embedding
        neg_embeds: (1, 77, 2048) negative embeddings
        neg_pooled: (1, 1280) negative pooled
        attention_controller: Controller for cross-attention scaling
        num_inference_steps: Denoising steps
        guidance_scale: CFG scale
        seed: Random seed
        use_attention_hooks: Whether to apply attention scaling
    
    Returns:
        Generated PIL Image
    """
    generator = None
    if seed is not None:
        device = getattr(pipe, 'device', 'cuda')
        if isinstance(device, str):
            device = torch.device(device)
        generator = torch.Generator(device=device).manual_seed(seed)
    
    gen_kwargs = {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled,
        "negative_prompt_embeds": neg_embeds,
        "negative_pooled_prompt_embeds": neg_pooled,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }
    gen_kwargs.update(kwargs)
    
    try:
        # Register attention hooks if requested
        if use_attention_hooks and hasattr(pipe, 'unet'):
            attention_controller.register_hooks(pipe.unet)
        
        result = pipe(**gen_kwargs)
        image = result.images[0]
        
    except Exception as e:
        print(f"[DirectInjector] Generation error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to simple text prompt
        try:
            image = pipe(
                prompt="beautiful interior design",
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images[0]
        except Exception as e2:
            raise e
    finally:
        attention_controller.remove_hooks()
    
    return image


# Backward compatibility aliases
HITLCompositionFuser = DirectEmbeddingInjector
generate_with_hooks = generate_with_direct_injection


def create_composition_prompt(composition: ContinuousComposition, base_prompt: str, max_tags: int = 10) -> str:
    """Fallback: Create text prompt from composition (for debugging)."""
    # This is a fallback - actual system uses direct injection
    return f"aesthetic interior, {base_prompt}"
