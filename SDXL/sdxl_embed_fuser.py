# backend/sdxl_embed_fuser.py
# Build SDXL prompt_embeds / pooled_prompt_embeds from weighted phrases.
# Adapted from pbo_utils/sdxl_embed_fuser.py with constraints for PBO integration.

from __future__ import annotations
from typing import List, Tuple
import torch
import numpy as np


class SDXLEmbedFuser:
    """
    SDXL embedding fuser with Top-K and token budget constraints.

    Features:
        - Accepts phrases and weights directly (no gain computation)
        - Clamps positive phrases to max 10
        - Clamps negative phrases to max 5
        - Warns if token budget exceeded
        - Caches phrase encodings for efficiency
    """

    def __init__(self, pipe, device: str | None = None):
        self.pipe = pipe
        self.device = device or (pipe._execution_device if hasattr(pipe, "_execution_device") else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer_1 = getattr(pipe, "tokenizer", None)
        self.tokenizer_2 = getattr(pipe, "tokenizer_2", None)
        self.text_encoder_1 = getattr(pipe, "text_encoder", None)
        self.text_encoder_2 = getattr(pipe, "text_encoder_2", None)
        assert self.tokenizer_1 is not None and self.tokenizer_2 is not None, "SDXL pipeline must have tokenizer and tokenizer_2"
        assert self.text_encoder_1 is not None and self.text_encoder_2 is not None, "SDXL pipeline must have text_encoder and text_encoder_2"
        # cache for phrase -> (prompt_embeds, pooled)
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def _encode_phrase(self, phrase: str):
        """Encode a single phrase with both CLIP text encoders."""
        # Check cache
        if phrase in self._cache:
            pe, pp = self._cache[phrase]
            return pe, pp

        # Encode with both encoders, pad/truncate to each tokenizer's max_length
        t1 = self.tokenizer_1(
            phrase, return_tensors="pt", padding="max_length", truncation=True,
            max_length=self.tokenizer_1.model_max_length
        ).to(self.device)
        t2 = self.tokenizer_2(
            phrase, return_tensors="pt", padding="max_length", truncation=True,
            max_length=self.tokenizer_2.model_max_length
        ).to(self.device)

        e1 = self.text_encoder_1(**t1).last_hidden_state    # (1, L1, H1)
        e2_out = self.text_encoder_2(**t2)
        e2 = e2_out.last_hidden_state                       # (1, L2, H2)

        # pooled: prefer pooler_output if present; else mean-pool
        pooled = getattr(e2_out, "pooler_output", None)
        if pooled is None:
            pooled = e2.mean(dim=1)

        # pad/truncate to equal seq len before concat
        L = max(e1.shape[1], e2.shape[1])

        def pad_to(x, L):
            if x.shape[1] == L:
                return x
            if x.shape[1] > L:
                return x[:, :L, :]
            pad = (0, 0, 0, L - x.shape[1])
            return torch.nn.functional.pad(x, pad, mode="constant", value=0.0)

        e1p = pad_to(e1, L)
        e2p = pad_to(e2, L)

        prompt_embeds = torch.cat([e1p, e2p], dim=-1)  # (1, L, H1+H2) ≈ (1,77,2048)

        # Cache
        self._cache[phrase] = (prompt_embeds, pooled)
        return prompt_embeds, pooled  # (1,L,D), (1,H_pool)

    def _check_token_budget(self, phrases: List[str], max_tokens: int = 77, label: str = "phrases"):
        """
        Check if phrases exceed token budget and warn if necessary.

        Args:
            phrases: List of phrases to check
            max_tokens: Maximum tokens allowed (default: 77 for CLIP)
            label: Label for warning message
        """
        # Concatenate all phrases
        combined = " ".join(phrases)

        # Tokenize to check length
        tokens = self.tokenizer_1(combined, return_tensors="pt", truncation=False)
        token_count = tokens['input_ids'].shape[1]

        if token_count > max_tokens:
            print(f"⚠️  Warning: {label} token count ({token_count}) exceeds max ({max_tokens}). "
                  f"This may cause truncation. Consider shorter phrases or fewer concepts.")

    @torch.no_grad()
    def fuse_weighted_phrases(
        self,
        phrases: List[str],
        weights: np.ndarray,
        neg_phrases: List[str] | None = None,
        max_positives: int = 10,
        max_negatives: int = 5
    ):
        """
        Fuse weighted phrases into SDXL embeddings.

        Args:
            phrases: List of phrase strings
            weights: Weight for each phrase (same length as phrases)
            neg_phrases: List of negative phrases (optional)
            max_positives: Maximum number of positive phrases (default: 10)
            max_negatives: Maximum number of negative phrases (default: 5)

        Returns:
            (fused_prompt, fused_pooled, neg_prompt, neg_pooled)
        """
        # Clamp positive phrases to Top-K (only if exceeds limit)
        if len(phrases) > max_positives:
            print(f"⚠️  Clamping {len(phrases)} positive phrases to top {max_positives}")
            phrases = phrases[:max_positives]
            weights = weights[:max_positives]

        # Clamp negative phrases
        if neg_phrases and len(neg_phrases) > max_negatives:
            print(f"⚠️  Clamping {len(neg_phrases)} negative phrases to top {max_negatives}")
            neg_phrases = neg_phrases[:max_negatives]

        if len(phrases) == 0:
            raise ValueError("No positive phrases given.")

        # Check token budget
        self._check_token_budget(phrases, label="positive")
        if neg_phrases:
            self._check_token_budget(neg_phrases, label="negative")

        # Convert weights to torch tensor and normalize for fusion
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        weights_tensor = weights_tensor / (weights_tensor.sum() + 1e-8)

        # Encode and fuse positives
        p_embeds = []
        p_pooled = []
        for phrase in phrases:
            pe, pp = self._encode_phrase(phrase)
            p_embeds.append(pe)
            p_pooled.append(pp)

        # stack and weighted sum
        P = torch.stack(p_embeds, dim=0)           # (N, 1, L, D)
        Pp = torch.stack(p_pooled, dim=0)          # (N, 1, H_pool)
        w = weights_tensor.view(-1, 1, 1, 1)       # (N,1,1,1)
        wp = weights_tensor.view(-1, 1, 1)         # (N,1,1)
        fused_prompt = (w * P).sum(dim=0)          # (1, L, D)
        fused_pooled = (wp * Pp).sum(dim=0)        # (1, H_pool)

        # Encode negatives (simple average)
        if neg_phrases and len(neg_phrases) > 0:
            n_embeds = []
            n_pooled = []
            for phrase in neg_phrases:
                ne, npool = self._encode_phrase(phrase)
                n_embeds.append(ne)
                n_pooled.append(npool)
            N = torch.stack(n_embeds, dim=0)
            Np = torch.stack(n_pooled, dim=0)
            neg_prompt = N.mean(dim=0)             # (1, L, D)
            neg_pooled = Np.mean(dim=0)            # (1, H_pool)
        else:
            # better fallback: unconditional by encoding empty string once
            empty_pos, empty_pooled = self._encode_phrase("")
            neg_prompt = empty_pos
            neg_pooled = empty_pooled

        return fused_prompt, fused_pooled, neg_prompt, neg_pooled
    
    @torch.no_grad()
    def fuse_with_alpha(
        self,
        descriptor: str | None,
        tag_phrases: List[str],
        tag_weights: np.ndarray,
        alpha: float,
        neg_phrases: List[str] | None = None,
        max_tags: int = 10,
        max_negatives: int = 8
    ):
        """
        Fuse descriptor and weighted tags with alpha interpolation.
        
        Formula: e_total = (1-alpha) * e_desc + alpha * e_tags
        
        Where:
            e_desc = embedding of descriptor
            e_tags = weighted sum of tag embeddings: Σ(w_i * embed(tag_i))
            alpha = interpolation factor (0 = pure descriptor, 1 = pure tags)
        
        Args:
            descriptor: Base scene description (e.g., "a bedroom interior")
            tag_phrases: List of concept tag phrases
            tag_weights: Weight for each tag (normalized, should sum to ~1.0)
            alpha: Strength factor for tags (0.0 to 1.0+)
            neg_phrases: List of negative phrases (optional)
            max_tags: Maximum number of tag phrases (default: 10)
            max_negatives: Maximum number of negative phrases (default: 5)
        
        Returns:
            (fused_prompt, fused_pooled, neg_prompt, neg_pooled)
        """
        # Clamp tags
        if len(tag_phrases) > max_tags:
            print(f"⚠️  Clamping {len(tag_phrases)} tag phrases to top {max_tags}")
            tag_phrases = tag_phrases[:max_tags]
            tag_weights = tag_weights[:max_tags]
        
        # Clamp negatives
        if neg_phrases and len(neg_phrases) > max_negatives:
            print(f"⚠️  Clamping {len(neg_phrases)} negative phrases to top {max_negatives}")
            neg_phrases = neg_phrases[:max_negatives]
        
        # Step 1: Encode descriptor
        if descriptor:
            desc_prompt_embeds, desc_pooled = self._encode_phrase(descriptor)
        else:
            # If no descriptor, use zero embeddings
            desc_prompt_embeds = torch.zeros(1, 77, 2048, device=self.device)
            desc_pooled = torch.zeros(1, 1280, device=self.device)
        
        # Step 2: Encode and fuse weighted tags: e_tags = Σ(w_i * embed(tag_i))
        if len(tag_phrases) > 0:
            # Convert weights to torch tensor and normalize
            weights_tensor = torch.tensor(tag_weights, dtype=torch.float32, device=self.device)
            weights_tensor = weights_tensor / (weights_tensor.sum() + 1e-8)
            
            # Encode all tags
            tag_embeds = []
            tag_pooled_list = []
            for phrase in tag_phrases:
                pe, pp = self._encode_phrase(phrase)
                tag_embeds.append(pe)
                tag_pooled_list.append(pp)
            
            # Stack and weighted sum
            T = torch.stack(tag_embeds, dim=0)           # (N, 1, L, D)
            Tp = torch.stack(tag_pooled_list, dim=0)     # (N, 1, H_pool)
            w = weights_tensor.view(-1, 1, 1, 1)         # (N,1,1,1)
            wp = weights_tensor.view(-1, 1, 1)           # (N,1,1)
            
            tags_prompt_embeds = (w * T).sum(dim=0)      # (1, L, D)
            tags_pooled = (wp * Tp).sum(dim=0)           # (1, H_pool)
        else:
            # No tags: use zero embeddings
            tags_prompt_embeds = torch.zeros_like(desc_prompt_embeds)
            tags_pooled = torch.zeros_like(desc_pooled)
        
        # Step 3: Combine with alpha: e_total = (1-alpha)*e_desc + alpha*e_tags
        fused_prompt = (1 - alpha) * desc_prompt_embeds + alpha * tags_prompt_embeds
        fused_pooled = (1 - alpha) * desc_pooled + alpha * tags_pooled
        
        # Step 4: Encode negatives (simple average)
        if neg_phrases and len(neg_phrases) > 0:
            n_embeds = []
            n_pooled = []
            for phrase in neg_phrases:
                ne, npool = self._encode_phrase(phrase)
                n_embeds.append(ne)
                n_pooled.append(npool)
            N = torch.stack(n_embeds, dim=0)
            Np = torch.stack(n_pooled, dim=0)
            neg_prompt = N.mean(dim=0)             # (1, L, D)
            neg_pooled = Np.mean(dim=0)            # (1, H_pool)
        else:
            # Fallback: unconditional by encoding empty string
            empty_pos, empty_pooled = self._encode_phrase("")
            neg_prompt = empty_pos
            neg_pooled = empty_pooled
        
        return fused_prompt, fused_pooled, neg_prompt, neg_pooled
