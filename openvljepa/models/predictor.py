"""Predictor: last 8 transformer layers of Llama-3.2-1B (pretrained), non-causal.

Matches Meta VL-JEPA paper Sec 3.1 spec:
  Predictor = last 8 layers of Llama-3.2-1B (~490M trainable).
  Text tokenizer + token embedding from Llama-3.2-1B.
  Max 512 query tokens with [PAD] for short queries.
  Causal attention disabled so vision + query embeddings jointly attended.
  Linear projection to shared 1536-dim embedding.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class Predictor(nn.Module):
    """V-JEPA2 visual embeddings + query tokens → predicted target embedding.

    Architecture:
      visual tokens (B, Nv, vis_dim) ─┐
                                       ├─→ concat → [last-8 Llama layers, non-causal] → mean pool → projector
      query token ids (B, Lq) ────────┘
    """

    def __init__(self, llama_name: str = "meta-llama/Llama-3.2-1B",
                 n_keep_layers: int = 8, vis_dim: int = 1024,
                 proj_dim: int = 1536, disable_causal_mask: bool = True,
                 torch_dtype=None):
        super().__init__()
        full = AutoModelForCausalLM.from_pretrained(llama_name, torch_dtype=torch_dtype)
        base = full.model  # LlamaModel
        self.hidden_size = base.config.hidden_size  # 2048 for Llama-3.2-1B

        # Extract submodules we keep (PyTorch will register them as children of self)
        self.embed_tokens = base.embed_tokens
        # Freeze Llama embedding table — paper's 490M trainable budget covers only
        # last 8 transformer layers + projections, not the 262M-param vocab embedding.
        # Also avoids destabilizing the Llama prior at small batch sizes.
        self.embed_tokens.requires_grad_(False)
        self.layers = nn.ModuleList(list(base.layers[-n_keep_layers:]))
        self.norm = base.norm
        self.rotary_emb = getattr(base, "rotary_emb", None)  # transformers >=4.45 attaches RoPE on LlamaModel

        # New linear projections (always created fresh, in default fp32 → cast later via autocast)
        self.vis_proj = nn.Linear(vis_dim, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, proj_dim)

        self.disable_causal_mask = disable_causal_mask

        # Drop unused references so GC frees first (N-8) layers, lm_head, etc.
        base.layers = nn.ModuleList()
        base.embed_tokens = None
        base.norm = None
        if hasattr(base, "rotary_emb"):
            base.rotary_emb = None
        del full

    def _build_bidirectional_mask(self, attention_mask: torch.Tensor,
                                  dtype: torch.dtype) -> torch.Tensor:
        """Build a 4D non-causal additive attention mask from a 2D pad mask.

        Args:
            attention_mask: (B, L) — 1 for valid, 0 for pad
            dtype: target dtype matching hidden states
        Returns:
            (B, 1, L, L) contiguous additive mask — 0 for attend, -inf for pad column
        """
        B, L = attention_mask.shape
        pad = (1.0 - attention_mask.to(torch.float32))  # 1 where pad
        neg_inf = torch.finfo(dtype).min
        mask = (pad * neg_inf).to(dtype)              # (B, L) — 0 valid, -inf pad
        return mask[:, None, None, :].expand(B, 1, L, L).contiguous()

    def forward(self, vis_embeds: torch.Tensor,
                query_ids: torch.Tensor,
                query_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vis_embeds: (B, Nv, vis_dim) raw V-JEPA2 visual embeddings
            query_ids:  (B, Lq) Llama-3.2 token ids
            query_mask: (B, Lq) — 1 for real, 0 for [PAD]. REQUIRED — pass real attention mask
                        from the tokenizer so pad positions are excluded from pool and attention.
        Returns:
            (B, proj_dim) predicted target embedding in shared space
        """
        assert query_mask is not None, (
            "query_mask is required: Llama tokenizer uses eos as pad, "
            "without the mask pad tokens contaminate attention and pooling."
        )
        B, Nv, _ = vis_embeds.shape

        # Embed: visual lifted to Llama hidden, query through Llama embed_tokens
        v = self.vis_proj(vis_embeds)                  # (B, Nv, H)
        q = self.embed_tokens(query_ids)               # (B, Lq, H)
        h = torch.cat([v, q], dim=1)                   # (B, L, H), L = Nv+Lq
        seq_len = h.shape[1]

        # Full pad mask: visual tokens always valid
        vis_mask = torch.ones(B, Nv, device=h.device, dtype=torch.float32)
        full_pad_mask = torch.cat([vis_mask, query_mask.to(torch.float32)], dim=1)  # (B, L)

        # 4D attention mask (always bidirectional in this implementation)
        # If causal mask is desired, build a causal-and-pad mask here.
        if self.disable_causal_mask:
            attn_mask_4d = self._build_bidirectional_mask(full_pad_mask, h.dtype)
        else:
            # Build causal-and-pad: lower-triangular ones, with pad columns set to -inf
            neg_inf = torch.finfo(h.dtype).min
            causal = torch.triu(
                torch.full((seq_len, seq_len), neg_inf, dtype=h.dtype, device=h.device),
                diagonal=1,
            )  # (L, L) upper-tri -inf, lower 0
            pad = (1.0 - full_pad_mask.to(torch.float32)).to(h.dtype) * neg_inf  # (B, L)
            attn_mask_4d = (causal[None, None, :, :] + pad[:, None, None, :]).contiguous()

        # Position ids for RoPE
        position_ids = torch.arange(seq_len, device=h.device).unsqueeze(0).expand(B, -1)

        # RoPE position embeddings (transformers >=4.45 expects this from caller)
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(h, position_ids)
        else:
            position_embeddings = None

        # Run through kept Llama decoder layers (transformers v5 signature)
        for layer in self.layers:
            layer_out = layer(
                h,
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            h = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        h = self.norm(h)  # (B, L, H)

        # Masked mean pool over both visual + query positions
        mask = full_pad_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return self.out_proj(pooled)  # (B, proj_dim)
