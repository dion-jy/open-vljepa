"""Verify all bug fixes: forward+backward pass + y_encoder gradient flow."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

# Test without V-JEPA2 (avoid downloading large model for verification)
# We mock the encoder to test the rest of the pipeline.

from openvljepa.models.y_encoder import YEncoder
from openvljepa.models.predictor import Predictor
from openvljepa.losses.infonce import bidirectional_infonce

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

B, N_vis, D = 4, 16, 1024
seq_q, seq_c = 32, 64
vocab_size = 30522

# Create modules
y_encoder = YEncoder(vocab_size=vocab_size, hidden_size=D, num_layers=2, num_heads=8, num_kv_heads=1).to(device)
predictor = Predictor(vocab_size=vocab_size, hidden_size=D, num_layers=2, num_heads=8, num_kv_heads=1, dora=True).to(device)

# Dummy data
vis_embeds = torch.randn(B, N_vis, D, device=device)
query_ids = torch.randint(0, vocab_size, (B, seq_q), device=device)
query_mask = torch.ones(B, seq_q, dtype=torch.long, device=device)
query_mask[:, -8:] = 0  # simulate PAD tokens
caption_ids = torch.randint(0, vocab_size, (B, seq_c), device=device)
caption_mask = torch.ones(B, seq_c, dtype=torch.long, device=device)
caption_mask[:, -16:] = 0  # simulate PAD tokens

print("\n=== Bug 1: Y-Encoder gradient flow ===")
pred = predictor(vis_embeds, query_ids, query_mask)
target = y_encoder(caption_ids, caption_mask)
loss = bidirectional_infonce(pred, target, temperature=0.07)
loss.backward()

y_grad_count = sum(1 for p in y_encoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
y_total = sum(1 for p in y_encoder.parameters())
print(f"  Y-Encoder params with gradient: {y_grad_count}/{y_total}")
assert y_grad_count > 0, "FAIL: Y-Encoder has no gradients!"
print("  PASS: Y-Encoder gradients are flowing")

pred_grad_count = sum(1 for p in predictor.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
pred_total = sum(1 for p in predictor.parameters() if p.requires_grad)
print(f"  Predictor trainable params with gradient: {pred_grad_count}/{pred_total}")

print("\n=== Bug 3: Masked mean pooling ===")
# Test that different masks produce different outputs
y_encoder.zero_grad()
out_full = y_encoder(caption_ids, torch.ones_like(caption_mask))
mask_half = caption_mask.clone()
mask_half[:, caption_mask.shape[1]//2:] = 0
out_masked = y_encoder(caption_ids, mask_half)
diff = (out_full - out_masked).abs().mean().item()
print(f"  Embedding diff (full vs masked): {diff:.6f}")
assert diff > 1e-4, "FAIL: Masks have no effect!"
print("  PASS: Masked mean pooling works")

print("\n=== Bug 4: GradScaler API ===")
scaler = torch.amp.GradScaler("cuda", enabled=True)
print("  PASS: torch.amp.GradScaler('cuda') works")

print("\n=== Forward+Backward summary ===")
print(f"  pred shape: {pred.shape}")
print(f"  target shape: {target.shape}")
print(f"  loss: {loss.item():.4f}")
print("\nAll checks passed!")
