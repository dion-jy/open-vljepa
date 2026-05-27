"""OpenVL-JEPA: V-JEPA2 encoder + Llama-3.2-1B predictor + EmbeddingGemma Y-Encoder.

Faithful to Meta VL-JEPA paper (arxiv:2512.10942) Sec 3.1:
  X-Encoder: V-JEPA2 ViT-L (frozen)
  Predictor: last 8 layers of Llama-3.2-1B (pretrained), non-causal
  Y-Encoder: EmbeddingGemma-300M (pretrained, slow LR x0.05)
  Shared embedding dim: 1536
  Loss: bi-directional InfoNCE
"""

import torch
import torch.nn as nn

from .encoder import VJEPA2Encoder
from .y_encoder import YEncoder
from .predictor import Predictor
from ..losses.infonce import bidirectional_infonce


class OpenVLJEPA(nn.Module):
    def __init__(self, encoder_cfg: dict, y_encoder_cfg: dict, predictor_cfg: dict,
                 torch_dtype=None):
        super().__init__()

        # V-JEPA2 X-Encoder (frozen). Output raw 1024-dim; predictor lifts to 2048.
        self.x_encoder = VJEPA2Encoder(
            model_name=encoder_cfg["model_name"],
            output_dim=encoder_cfg.get("output_dim", None),
            frozen=encoder_cfg.get("frozen", True),
        )
        vis_dim = self.x_encoder.output_dim

        # Y-Encoder (EmbeddingGemma)
        self.y_encoder = YEncoder(
            model_name=y_encoder_cfg["model_name"],
            proj_dim=y_encoder_cfg["proj_dim"],
            torch_dtype=torch_dtype,
        )

        # Predictor (Llama-3.2-1B last 8 layers, non-causal)
        self.predictor = Predictor(
            llama_name=predictor_cfg["llama_name"],
            n_keep_layers=predictor_cfg.get("n_keep_layers", 8),
            vis_dim=vis_dim,
            proj_dim=predictor_cfg["proj_dim"],
            disable_causal_mask=predictor_cfg.get("disable_causal_mask", True),
            torch_dtype=torch_dtype,
        )

        assert y_encoder_cfg["proj_dim"] == predictor_cfg["proj_dim"], (
            "Y-Encoder and Predictor must project to the same shared dim"
        )

    def forward(self, pixel_values: torch.Tensor,
                query_ids: torch.Tensor,
                query_mask: torch.Tensor = None,
                target_ids: torch.Tensor = None,
                target_mask: torch.Tensor = None,
                temperature: float = 0.07):
        """Forward pass. Returns loss if target_ids given, else predicted embedding.

        DDP only hooks forward(), so loss computation happens here for
        proper gradient synchronization across GPUs.
        """
        vis_embeds = self.x_encoder(pixel_values)               # (B, Nv, vis_dim)
        pred = self.predictor(vis_embeds, query_ids, query_mask)  # (B, proj_dim)

        if target_ids is not None:
            target = self.y_encoder(target_ids, target_mask)    # (B, proj_dim)
            return bidirectional_infonce(pred, target, temperature)

        return pred
