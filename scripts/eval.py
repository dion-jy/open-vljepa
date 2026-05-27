"""OpenVL-JEPA text-to-video retrieval evaluation on MSRVTT test split.

Protocol (Meta VL-JEPA paper Sec 4.1):
    - Each test video is mapped to a *predicted* embedding S_Y_hat by feeding the
      video and a default "retrieval captioning prompt" (e.g., "Describe the
      video.") through the X-Encoder + Predictor.
    - Each test caption is mapped to a *target* embedding S_Y by feeding it
      through the Y-Encoder.
    - Caption-side query is ranked against the video pool by cosine similarity;
      R@1 / R@5 / R@10 / median rank are reported.

MSRVTT test split contains 2990 unique videos with ~20 captions each.
The standard MSRVTT-1K (JSFusion) split is not available from our
annotations.json, so we evaluate against the full 2990-video pool.

Usage:
    python scripts/eval.py --ckpt checkpoints/best.pt
    python scripts/eval.py --ckpt checkpoints/best.pt --n_videos 300  # debug
    python scripts/eval.py --ckpt checkpoints/best.pt --cache_dir cache/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from openvljepa.data.msrvtt import MSRVTTDataset, _ensure_pad_token  # noqa: E402
from openvljepa.models.vljepa import OpenVLJEPA  # noqa: E402


DEFAULT_RETRIEVAL_PROMPT = "Describe the video."


# -----------------------------------------------------------------------------
# Dataset adapters: separate video-only and caption-only iteration
# -----------------------------------------------------------------------------


class VideoOnlyDataset(Dataset):
    """One sample per unique test video. Loads frames only (no caption)."""

    def __init__(self, base: MSRVTTDataset, video_ids):
        self.base = base
        self.video_ids = list(video_ids)
        # video_id -> first sample index in base (for frame loading)
        self._first_idx = {}
        for i, s in enumerate(base.samples):
            vid = s["video_id"]
            if vid not in self._first_idx:
                self._first_idx[vid] = i

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        sample = self.base[self._first_idx[vid]]
        return {
            "pixel_values": sample["pixel_values"],
            "video_id": vid,
        }


class CaptionOnlyDataset(Dataset):
    """One sample per (video_id, caption) pair. Tokenizes target only."""

    def __init__(self, base: MSRVTTDataset):
        self.base = base
        self.target_tokenizer = base.target_tokenizer
        self.max_caption_len = base.max_caption_len

    def __len__(self):
        return len(self.base.samples)

    def __getitem__(self, idx):
        s = self.base.samples[idx]
        tenc = self.target_tokenizer(
            s["caption"],
            max_length=self.max_caption_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "caption_ids": tenc["input_ids"].squeeze(0),
            "caption_mask": tenc["attention_mask"].squeeze(0),
            "video_id": s["video_id"],
            "caption_text": s["caption"],
        }


# -----------------------------------------------------------------------------
# Model / checkpoint loading
# -----------------------------------------------------------------------------


def load_model_and_cfg(ckpt_path: str, device: torch.device):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    print("Building OpenVL-JEPA (bf16 modules where possible)...")
    model = OpenVLJEPA(
        cfg["encoder"], cfg["y_encoder"], cfg["predictor"],
        torch_dtype=torch.bfloat16,
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    non_xencoder_missing = [k for k in missing if not k.startswith("x_encoder.model.")]
    if non_xencoder_missing:
        print(f"  WARNING: missing non-frozen keys: {non_xencoder_missing[:5]}...")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected[:5]}...")
    print(
        f"  Loaded {len(ckpt['model_state_dict'])} keys "
        f"(missing={len(missing)} — frozen x_encoder reloaded from HF; "
        f"epoch={ckpt.get('epoch','?')})"
    )

    model.eval().to(device)
    return model, cfg, ckpt


# -----------------------------------------------------------------------------
# Encoding loops
# -----------------------------------------------------------------------------


@torch.no_grad()
def encode_videos(model, video_loader, query_ids, query_mask, device):
    """Encode each video with the default retrieval prompt → (N_v, D)."""
    pred_embeds = []
    video_ids = []
    t0 = time.time()
    n_batches = len(video_loader)
    for bi, batch in enumerate(video_loader):
        pv = batch["pixel_values"].to(device, non_blocking=True)
        B = pv.shape[0]
        q_ids = query_ids.expand(B, -1)
        q_mask = query_mask.expand(B, -1)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(pv, q_ids, q_mask)  # (B, D)
            pred = F.normalize(pred.float(), dim=-1)

        pred_embeds.append(pred.cpu())
        video_ids.extend(batch["video_id"])

        if (bi + 1) % 10 == 0 or (bi + 1) == n_batches:
            elapsed = time.time() - t0
            rate = (bi + 1) / elapsed
            eta = (n_batches - bi - 1) / max(rate, 1e-6)
            print(
                f"  [video] batch {bi+1}/{n_batches}  "
                f"({rate:.2f} bat/s, ETA {eta:.0f}s)"
            )

    return torch.cat(pred_embeds, dim=0), video_ids


@torch.no_grad()
def encode_captions(model, caption_loader, device):
    """Encode each caption through Y-Encoder → (N_c, D)."""
    target_embeds = []
    cap_video_ids = []
    cap_texts = []
    t0 = time.time()
    n_batches = len(caption_loader)
    for bi, batch in enumerate(caption_loader):
        c_ids = batch["caption_ids"].to(device, non_blocking=True)
        c_mask = batch["caption_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            tgt = model.y_encoder(c_ids, c_mask)  # (B, D)
            tgt = F.normalize(tgt.float(), dim=-1)

        target_embeds.append(tgt.cpu())
        cap_video_ids.extend(batch["video_id"])
        cap_texts.extend(batch["caption_text"])

        if (bi + 1) % 20 == 0 or (bi + 1) == n_batches:
            elapsed = time.time() - t0
            rate = (bi + 1) / elapsed
            eta = (n_batches - bi - 1) / max(rate, 1e-6)
            print(
                f"  [caption] batch {bi+1}/{n_batches}  "
                f"({rate:.2f} bat/s, ETA {eta:.0f}s)"
            )

    return torch.cat(target_embeds, dim=0), cap_video_ids, cap_texts


# -----------------------------------------------------------------------------
# Retrieval metric
# -----------------------------------------------------------------------------


def compute_t2v_metrics(
    caption_embeds: torch.Tensor,
    cap_video_ids: list,
    video_embeds: torch.Tensor,
    video_ids: list,
    ks=(1, 5, 10),
    sim_chunk: int = 1024,
) -> dict:
    """Text-to-video retrieval: for each caption, rank videos by cosine sim
    and check rank of the ground-truth video.

    Args:
        caption_embeds: (Nc, D) normalized
        cap_video_ids:  length-Nc list of ground-truth video ids
        video_embeds:   (Nv, D) normalized
        video_ids:      length-Nv list of video ids (column order)
    """
    vid_to_col = {v: i for i, v in enumerate(video_ids)}
    gt = torch.tensor(
        [vid_to_col[v] for v in cap_video_ids], dtype=torch.long
    )

    Nc = caption_embeds.shape[0]
    Nv = video_embeds.shape[0]
    V = video_embeds.to(torch.float32)
    C = caption_embeds.to(torch.float32)

    # Chunked sim → rank-of-gt to keep memory bounded.
    gt_ranks = torch.empty(Nc, dtype=torch.long)
    for s in range(0, Nc, sim_chunk):
        e = min(s + sim_chunk, Nc)
        sim = C[s:e] @ V.T  # (chunk, Nv)
        gt_col = gt[s:e]
        gt_score = sim.gather(1, gt_col[:, None]).squeeze(1)  # (chunk,)
        # rank = number of videos with strictly higher sim
        higher = (sim > gt_score[:, None]).sum(dim=1)
        gt_ranks[s:e] = higher  # 0 = top-1

    metrics = {}
    for k in ks:
        metrics[f"R@{k}"] = (gt_ranks < k).float().mean().item() * 100.0
    metrics["MedianR"] = float(gt_ranks.float().median().item()) + 1.0
    metrics["MeanR"] = float(gt_ranks.float().mean().item()) + 1.0
    metrics["num_queries"] = Nc
    metrics["pool_size"] = Nv
    return metrics


def compute_v2t_metrics(
    video_embeds: torch.Tensor,
    video_ids: list,
    caption_embeds: torch.Tensor,
    cap_video_ids: list,
    ks=(1, 5, 10),
    sim_chunk: int = 256,
) -> dict:
    """Video-to-text retrieval: for each video, the correct items are any caption
    whose video_id matches. We compute the *best* rank among correct captions
    (standard "any-correct" V2T R@K).
    """
    # video_id -> indices of captions belonging to it
    vid_to_caps = {}
    for i, v in enumerate(cap_video_ids):
        vid_to_caps.setdefault(v, []).append(i)

    Nv = video_embeds.shape[0]
    Nc = caption_embeds.shape[0]
    V = video_embeds.to(torch.float32)
    C = caption_embeds.to(torch.float32)

    best_ranks = torch.empty(Nv, dtype=torch.long)
    for s in range(0, Nv, sim_chunk):
        e = min(s + sim_chunk, Nv)
        sim = V[s:e] @ C.T  # (chunk, Nc)
        # Argsort once per chunk row; then find min position over correct caption ids
        order = sim.argsort(dim=1, descending=True)  # (chunk, Nc)
        # rank-of-each-caption = position in `order`
        # build inverse: for each row, position[idx] = rank
        pos = torch.empty_like(order)
        pos.scatter_(1, order, torch.arange(Nc).expand(e - s, -1))
        for r, vid in enumerate(video_ids[s:e]):
            cap_idx = vid_to_caps.get(vid, [])
            if not cap_idx:
                best_ranks[s + r] = Nc  # missing → worst
                continue
            best_ranks[s + r] = pos[r, torch.tensor(cap_idx)].min()

    metrics = {}
    for k in ks:
        metrics[f"R@{k}"] = (best_ranks < k).float().mean().item() * 100.0
    metrics["MedianR"] = float(best_ranks.float().median().item()) + 1.0
    metrics["MeanR"] = float(best_ranks.float().mean().item()) + 1.0
    metrics["num_queries"] = Nv
    metrics["pool_size"] = Nc
    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/best.pt",
                        help="path to trained checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video_batch_size", type=int, default=8)
    parser.add_argument("--caption_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_videos", type=int, default=-1,
                        help="cap test-video pool size for quick debugging (-1=all)")
    parser.add_argument("--retrieval_prompt", default=DEFAULT_RETRIEVAL_PROMPT,
                        help="text prompt used to map every video to its predicted "
                             "embedding S_Y_hat")
    parser.add_argument("--cache_dir", default=None,
                        help="if set, save/load video/caption embeddings here")
    parser.add_argument("--out_json", default=None,
                        help="path to save metrics JSON (default: alongside ckpt)")
    parser.add_argument("--skip_v2t", action="store_true",
                        help="skip Video-to-Text direction (T2V only)")
    parser.add_argument("--data_dir", default=None,
                        help="override MSRVTT data_dir (use when ckpt was trained "
                             "on a different dataset like WebVid)")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 64)
    print("OpenVL-JEPA Text-to-Video Retrieval Eval — MSRVTT test")
    print("=" * 64)

    # 1) Model + config
    model, cfg, ckpt = load_model_and_cfg(args.ckpt, device)

    # 2) Tokenizers
    query_tokenizer = AutoTokenizer.from_pretrained(cfg["predictor"]["llama_name"])
    target_tokenizer = AutoTokenizer.from_pretrained(cfg["y_encoder"]["model_name"])
    query_tokenizer = _ensure_pad_token(query_tokenizer)
    target_tokenizer = _ensure_pad_token(target_tokenizer)

    # 3) Test dataset (the base one; we'll wrap it for video-only and caption-only)
    print("Building MSRVTT test split...")
    base = MSRVTTDataset(
        data_dir=args.data_dir or cfg["data"]["data_dir"],
        query_tokenizer=query_tokenizer,
        target_tokenizer=target_tokenizer,
        split="test",
        num_frames=cfg["data"].get("num_frames", 16),
        image_size=cfg["data"].get("image_size", 256),
        max_query_len=cfg["data"].get("max_query_len", 512),
        max_caption_len=cfg["data"].get("max_caption_len", 512),
    )

    # Collect unique video ids (insertion order)
    seen = []
    seen_set = set()
    for s in base.samples:
        if s["video_id"] not in seen_set:
            seen_set.add(s["video_id"])
            seen.append(s["video_id"])
    if args.n_videos > 0:
        seen = seen[: args.n_videos]
        seen_set = set(seen)
        # filter caption samples too
        base.samples = [s for s in base.samples if s["video_id"] in seen_set]

    print(f"  unique videos in pool: {len(seen)}")
    print(f"  total (video, caption) pairs: {len(base.samples)}")

    # 4) Encode pool of test videos (uses default retrieval prompt for all)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{Path(args.ckpt).stem}__n{len(seen)}"
    vid_cache = cache_dir / f"{cache_key}__video.pt" if cache_dir else None
    cap_cache = cache_dir / f"{cache_key}__caption.pt" if cache_dir else None

    if vid_cache and vid_cache.exists():
        print(f"Loading video embeddings from {vid_cache}")
        blob = torch.load(vid_cache, weights_only=False)
        video_embeds, video_ids = blob["embeds"], blob["video_ids"]
    else:
        print(f"Encoding videos with prompt: \"{args.retrieval_prompt}\"")
        qenc = query_tokenizer(
            args.retrieval_prompt,
            max_length=cfg["data"].get("max_query_len", 512),
            padding="max_length", truncation=True, return_tensors="pt",
        )
        q_ids = qenc["input_ids"].to(device)
        q_mask = qenc["attention_mask"].to(device)

        video_ds = VideoOnlyDataset(base, seen)
        video_loader = DataLoader(
            video_ds, batch_size=args.video_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
        )
        video_embeds, video_ids = encode_videos(model, video_loader, q_ids, q_mask, device)
        if vid_cache:
            torch.save({"embeds": video_embeds, "video_ids": video_ids}, vid_cache)
            print(f"  cached → {vid_cache}")

    print(f"  video embeds: {tuple(video_embeds.shape)}")

    # 5) Encode all test captions
    if cap_cache and cap_cache.exists():
        print(f"Loading caption embeddings from {cap_cache}")
        blob = torch.load(cap_cache, weights_only=False)
        caption_embeds = blob["embeds"]
        cap_video_ids = blob["cap_video_ids"]
        cap_texts = blob["cap_texts"]
    else:
        print("Encoding captions through Y-Encoder...")
        cap_ds = CaptionOnlyDataset(base)
        cap_loader = DataLoader(
            cap_ds, batch_size=args.caption_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
        )
        caption_embeds, cap_video_ids, cap_texts = encode_captions(
            model, cap_loader, device
        )
        if cap_cache:
            torch.save(
                {"embeds": caption_embeds, "cap_video_ids": cap_video_ids,
                 "cap_texts": cap_texts},
                cap_cache,
            )
            print(f"  cached → {cap_cache}")

    print(f"  caption embeds: {tuple(caption_embeds.shape)}")

    # 6) Metrics: T2V (primary), V2T (optional)
    print("\nComputing T2V retrieval metrics...")
    t2v = compute_t2v_metrics(caption_embeds, cap_video_ids, video_embeds, video_ids)
    print("\n=== Text → Video retrieval ===")
    for k in ("R@1", "R@5", "R@10"):
        print(f"  {k:<6s} = {t2v[k]:6.2f} %")
    print(f"  MedianR = {t2v['MedianR']:.1f}")
    print(f"  MeanR   = {t2v['MeanR']:.1f}")
    print(f"  queries = {t2v['num_queries']}, pool = {t2v['pool_size']}")

    results = {
        "ckpt": str(Path(args.ckpt).resolve()),
        "epoch": ckpt.get("epoch", None),
        "retrieval_prompt": args.retrieval_prompt,
        "t2v": t2v,
    }

    if not args.skip_v2t:
        print("\nComputing V2T retrieval metrics...")
        v2t = compute_v2t_metrics(video_embeds, video_ids, caption_embeds, cap_video_ids)
        print("\n=== Video → Text retrieval ===")
        for k in ("R@1", "R@5", "R@10"):
            print(f"  {k:<6s} = {v2t[k]:6.2f} %")
        print(f"  MedianR = {v2t['MedianR']:.1f}")
        print(f"  MeanR   = {v2t['MeanR']:.1f}")
        print(f"  queries = {v2t['num_queries']}, pool = {v2t['pool_size']}")
        results["v2t"] = v2t

    out_path = (
        Path(args.out_json)
        if args.out_json
        else Path(args.ckpt).with_suffix(".eval.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics → {out_path}")


if __name__ == "__main__":
    main()
