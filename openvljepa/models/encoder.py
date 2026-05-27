"""V-JEPA2 encoder wrapper — frozen visual backbone."""

import torch
import torch.nn as nn
from transformers import AutoModel


class VJEPA2Encoder(nn.Module):
    """Wraps HuggingFace V-JEPA2 encoder.

    Loads only the encoder (skip_predictor=True), projects to output_dim.
    Input: (B, T, C, H, W) video tensors.
    Output: (B, N, output_dim) where N=2048 for ViT-L/16, 256x256, 16 frames.
    """

    def __init__(self, model_name: str, output_dim: int = None, frozen: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = self.model.config.hidden_size  # 1024 for ViT-L

        # output_dim=None → pass through raw hidden (predictor lifts to its own hidden)
        if output_dim is None or output_dim == self.hidden_size:
            self.projector = nn.Identity()
            self.output_dim = self.hidden_size
        else:
            self.projector = nn.Linear(self.hidden_size, output_dim)
            self.output_dim = output_dim

        if frozen:
            self.model.requires_grad_(False)
            self.model.eval()

        self.frozen = frozen

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values_videos: (B, T, C, H, W)
        Returns:
            (B, N, output_dim) visual embeddings
        """
        if self.frozen:
            with torch.no_grad():
                outputs = self.model(pixel_values_videos, skip_predictor=True)
                hidden = outputs.last_hidden_state.detach()  # (B, N, 1024)
        else:
            outputs = self.model(pixel_values_videos, skip_predictor=True)
            hidden = outputs.last_hidden_state

        return self.projector(hidden)
