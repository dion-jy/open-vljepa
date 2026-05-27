"""Y-Encoder: EmbeddingGemma-300M pretrained → projection to shared 1536-dim space.

Matches Meta VL-JEPA paper Sec 3.1 spec:
  Y-Encoder = EmbeddingGemma-300M (pretrained), trained with LR x0.05.
  Average pooling on non-[PAD] tokens, then linear projection.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class YEncoder(nn.Module):
    """Pretrained EmbeddingGemma + masked mean pool + linear projection."""

    def __init__(self, model_name: str = "google/embeddinggemma-300m",
                 proj_dim: int = 1536, torch_dtype=None):
        super().__init__()
        # Load the base transformer; EmbeddingGemma's HF repo ships a SentenceTransformer
        # wrapper, but AutoModel pulls the underlying Gemma3 text model whose
        # `last_hidden_state` is what we want (pre-pooling token-level reps).
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.hidden_size = self.model.config.hidden_size
        self.projector = nn.Linear(self.hidden_size, proj_dim)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L)
            attention_mask: (B, L) — 1 for real tokens, 0 for [PAD]
        Returns:
            (B, proj_dim) target embedding in shared space
        """
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # (B, L, hidden_size)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = h.mean(dim=1)

        return self.projector(pooled)
