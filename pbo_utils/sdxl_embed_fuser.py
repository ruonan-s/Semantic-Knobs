# pbo_min/embeddings/sdxl_embed_fuser.py
# Build SDXL prompt_embeds / pooled_prompt_embeds from weighted phrases.
# We rely on the pipeline’s tokenizers & text encoders to ensure shapes match.

from __future__ import annotations
from typing import List, Tuple
import torch
import numpy as np

class SDXLEmbedFuser:
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
            if x.shape[1] == L: return x
            if x.shape[1] > L:  return x[:, :L, :]
            pad = (0,0,0, L - x.shape[1])
            return torch.nn.functional.pad(x, pad, mode="constant", value=0.0)
        e1p = pad_to(e1, L)
        e2p = pad_to(e2, L)

        prompt_embeds = torch.cat([e1p, e2p], dim=-1)  # (1, L, H1+H2) ≈ (1,77,2048)
        # Cache
        self._cache[phrase] = (prompt_embeds, pooled)
        return prompt_embeds, pooled  # (1,L,D), (1,H_pool)

    @torch.no_grad()
    def fuse_weighted_phrases(self, pos_phrases: List[Tuple[str, float]], neg_phrases: List[str] | None = None):
        # Normalize weights
        if len(pos_phrases) == 0:
            raise ValueError("No positive phrases given.")
        weights_np = np.array([max(0.0, float(w)) for _, w in pos_phrases], dtype=np.float32)
        # soften extremes with exponent < 1 for stability
        weights_np = np.power(weights_np, 0.8)
        weights_np = weights_np / (weights_np.sum() + 1e-8)
        weights = torch.tensor(weights_np, device=self.device, dtype=torch.float32)

        # Encode and fuse positives
        p_embeds = []
        p_pooled = []
        for phrase, _ in pos_phrases:
            pe, pp = self._encode_phrase(phrase)
            p_embeds.append(pe)
            p_pooled.append(pp)
        # stack and weighted sum
        P = torch.stack(p_embeds, dim=0)           # (N, 1, L, D)
        Pp = torch.stack(p_pooled, dim=0)          # (N, 1, H_pool)
        w = weights.view(-1, 1, 1, 1)              # (N,1,1,1)
        wp = weights.view(-1, 1, 1)                # (N,1,1)
        fused_prompt = (w * P).sum(dim=0)          # (1, L, D)
        fused_pooled = (wp * Pp).sum(dim=0)        # (1, H_pool)

        # Encode negatives (combined string)
        if neg_phrases and len(neg_phrases) > 0:
            combined_neg = ", ".join(neg_phrases)
            neg_prompt, neg_pooled = self._encode_phrase(combined_neg)
        else:
            # better fallback: unconditional by encoding empty string once
            empty_pos, empty_pooled = self._encode_phrase("")
            neg_prompt = empty_pos
            neg_pooled = empty_pooled

        return fused_prompt, fused_pooled, neg_prompt, neg_pooled
