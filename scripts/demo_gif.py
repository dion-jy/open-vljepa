"""Generate animated GIF demo: text query → top-K video retrieval, videos playing.

For each query, retrieves top-K MSRVTT videos and renders them as a stacked-row
animated GIF (8 frames sampled, 5fps loop).

Usage:
    python scripts/demo_gif.py --ckpt checkpoints_stage_b_after_a/best.pt --out demos/best_r23.9.gif
"""

import argparse
import io
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from openvljepa.data.msrvtt import MSRVTTDataset
from openvljepa.models.vljepa import OpenVLJEPA


DEFAULT_QUERIES = [
    "a man is playing guitar on stage",
    "a person is cooking food in the kitchen",
    "people are dancing",
    "a car driving on the road",
    "someone playing basketball",
    "a cartoon character in a video game",
    "a dog is running in the grass",
    "a woman is talking on a news show",
]


def denormalize_to_uint8(frame_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (frame_tensor * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def render_query_cell(text, width, height):
    """Render a query text as a fixed-size image cell."""
    img = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # Wrap text to lines
    words = text.split()
    lines, cur = [], ""
    max_chars_per_line = width // 11
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars_per_line:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    total_h = len(lines) * 24
    y = (height - total_h) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        draw.text(((width - line_w) // 2, y), line, fill=(20, 20, 20), font=font)
        y += 24
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints_stage_b_after_a/best.pt")
    parser.add_argument("--n_videos", type=int, default=300)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--out", default="demos/best_r23.9_animated.gif")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--cell_size", type=int, default=160)
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    print("Building model (bf16)...")
    model = OpenVLJEPA(cfg["encoder"], cfg["y_encoder"], cfg["predictor"],
                       torch_dtype=torch.bfloat16)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval().to(device)

    query_tokenizer = AutoTokenizer.from_pretrained(cfg["predictor"]["llama_name"])
    target_tokenizer = AutoTokenizer.from_pretrained(cfg["y_encoder"]["model_name"])

    print("Building MSRVTT test pool...")
    ds = MSRVTTDataset(
        data_dir=cfg["data"]["data_dir"],
        query_tokenizer=query_tokenizer,
        target_tokenizer=target_tokenizer,
        split="test",
        num_frames=cfg["data"]["num_frames"],
        image_size=cfg["data"]["image_size"],
    )

    # Collect first N unique videos
    seen = set()
    video_idx_of_first_caption = []
    for i, s in enumerate(ds.samples):
        if s["video_id"] not in seen:
            seen.add(s["video_id"])
            video_idx_of_first_caption.append(i)
            if len(video_idx_of_first_caption) >= args.n_videos:
                break
    print(f"Pool: {len(video_idx_of_first_caption)} videos")

    # Encode pool with default retrieval prompt
    default_query = "Describe the video."
    qe = query_tokenizer(default_query, max_length=cfg["data"]["max_query_len"],
                         padding="max_length", truncation=True, return_tensors="pt")
    q_ids = qe["input_ids"].to(device)
    q_mask = qe["attention_mask"].to(device)

    pred_embeds = []
    video_ids = []
    frames_cache = {}
    print("Encoding videos...")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for n, idx in enumerate(video_idx_of_first_caption):
            sample = ds[idx]
            pv = sample["pixel_values"].unsqueeze(0).to(device)
            pred = model(pv, q_ids, q_mask)
            pred_embeds.append(pred.float().cpu())
            video_ids.append(sample["video_id"])
            frames_cache[sample["video_id"]] = sample["pixel_values"]
            if (n + 1) % 50 == 0:
                print(f"  {n+1}/{len(video_idx_of_first_caption)}")
    pred_embeds = F.normalize(torch.cat(pred_embeds, 0), dim=-1).to(device)

    # Compose GIF: each row = [query text | top-k video cells]
    cell = args.cell_size
    text_cell_w = int(cell * 1.8)
    n_rows = len(DEFAULT_QUERIES)
    total_w = text_cell_w + cell * args.top_k
    total_h = n_rows * cell

    # Pre-render text cells (static)
    static_text = np.stack(
        [render_query_cell(q, text_cell_w, cell) for q in DEFAULT_QUERIES],
        axis=0,  # (n_rows, H, W, 3)
    )

    # Compute top-k for each query
    print("Ranking + computing top-k...")
    top_videos = []  # list per query of (vid_id, frames_uint8)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for qi, query_text in enumerate(DEFAULT_QUERIES):
            te = target_tokenizer(query_text, max_length=cfg["data"]["max_caption_len"],
                                  padding="max_length", truncation=True, return_tensors="pt")
            t_ids = te["input_ids"].to(device)
            t_mask = te["attention_mask"].to(device)
            target = F.normalize(model.y_encoder(t_ids, t_mask).float(), dim=-1)
            sims = (pred_embeds.float() @ target.float().T).squeeze(-1).cpu().float().numpy()
            top_idx = np.argsort(-sims)[: args.top_k]
            this = []
            for idx in top_idx:
                vid = video_ids[idx]
                frames = frames_cache[vid]  # (T, C, H, W)
                # Resize each frame to cell × cell
                resized = []
                for t in range(frames.shape[0]):
                    img = Image.fromarray(denormalize_to_uint8(frames[t]))
                    img = img.resize((cell, cell), Image.BILINEAR)
                    resized.append(np.array(img))
                this.append(np.stack(resized, axis=0))  # (T, cell, cell, 3)
            top_videos.append(this)

    num_frames = top_videos[0][0].shape[0]  # T

    # Build per-frame composites
    print(f"Building {num_frames}-frame GIF...")
    frames_for_gif = []
    for t in range(num_frames):
        canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 250
        for qi in range(n_rows):
            y0 = qi * cell
            canvas[y0:y0+cell, :text_cell_w] = static_text[qi]
            for ki in range(args.top_k):
                x0 = text_cell_w + ki * cell
                canvas[y0:y0+cell, x0:x0+cell] = top_videos[qi][ki][t]
        frames_for_gif.append(canvas)

    # Slow down: each frame held for 1/fps seconds
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames_for_gif, duration=1.0 / args.fps, loop=0)
    print(f"\nSaved: {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
