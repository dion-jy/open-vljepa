"""WebVid-2M webdataset loader for Stage B video-text pretraining.

Reads tar shards produced by video2dataset:
    {shard}.tar  containing  {id}.mp4 + {id}.txt + {id}.json

For Stage B:
- Uniformly samples num_frames frames per clip
- Center-crop/resize to image_size (only if v2d output not already at target)
- Default retrieval prompt as query, WebVid "name" caption as target
- Filters tars that lack {shard}_stats.json (safe against partial v2d writes)
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import webdataset as wds

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False


DEFAULT_QUERY = "Describe the video."


def _decode_video_caption(sample, num_frames, image_size):
    vid_bytes = sample.get("mp4")
    cap_bytes = sample.get("txt") or sample.get("caption")
    if vid_bytes is None or cap_bytes is None:
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(vid_bytes)
            tmp_path = tmp.name

        vr = VideoReader(tmp_path, ctx=cpu(0))
        total = len(vr)
        if total < num_frames:
            indices = list(range(total)) + [total - 1] * (num_frames - total)
        else:
            indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C) uint8
    except Exception:
        return None
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    if frames.shape[-1] != image_size:
        frames = T.functional.resize(frames, image_size, antialias=True)
        frames = T.functional.center_crop(frames, image_size)
    frames = T.functional.normalize(frames, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    caption = cap_bytes.decode("utf-8") if isinstance(cap_bytes, bytes) else str(cap_bytes)
    return frames, caption.strip()


def _tokenize(sample, query_tokenizer, target_tokenizer,
              query_text, max_query_len, max_caption_len, num_frames):
    if sample is None:
        return None
    frames, caption = sample

    qenc = query_tokenizer(query_text, max_length=max_query_len,
                           padding="max_length", truncation=True, return_tensors="pt")
    tenc = target_tokenizer(caption, max_length=max_caption_len,
                            padding="max_length", truncation=True, return_tensors="pt")

    return {
        "pixel_values": frames,
        "query_ids": qenc["input_ids"].squeeze(0),
        "query_mask": qenc["attention_mask"].squeeze(0),
        "caption_ids": tenc["input_ids"].squeeze(0),
        "caption_mask": tenc["attention_mask"].squeeze(0),
        "caption_text": caption,
    }


def _collate(batch):
    # Collate only keys present in every sample (combined WebVid+MSRVTT loader
    # mixes sources with different optional keys like 'video_id').
    common = set(batch[0].keys())
    for b in batch[1:]:
        common &= set(b.keys())
    out = {}
    for k in common:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out


def build_webvid_pipeline(data_cfg, query_tokenizer, target_tokenizer):
    """Unbatched WebVid pipeline — shared with combined loader."""
    data_dir = Path(data_cfg["data_dir"])
    # Only use shards whose _stats.json exists (safe against concurrent DL writes)
    completed = []
    for tar in data_dir.glob("*.tar"):
        if (data_dir / f"{tar.stem}_stats.json").exists():
            completed.append(str(tar))
    tars = sorted(completed)
    if not tars:
        raise FileNotFoundError(f"No completed WebVid tars (with stats.json) at {data_dir}")

    image_size = data_cfg.get("image_size", 256)
    num_frames = data_cfg.get("num_frames", 8)
    max_query_len = data_cfg.get("max_query_len", 64)
    max_caption_len = data_cfg.get("max_caption_len", 64)
    query_text = data_cfg.get("query_text", DEFAULT_QUERY)

    def decode(s):
        return _decode_video_caption(s, num_frames, image_size)

    def tokenize(s):
        return _tokenize(s, query_tokenizer, target_tokenizer,
                         query_text, max_query_len, max_caption_len, num_frames)

    return (
        wds.WebDataset(tars, shardshuffle=True, resampled=True, nodesplitter=wds.split_by_node)
        .shuffle(500)
        .map(decode)
        .select(lambda x: x is not None)
        .map(tokenize)
        .select(lambda x: x is not None)
    )


def build_webvid_dataloader(cfg, query_tokenizer, target_tokenizer,
                             split: str = "train", distributed: bool = False):
    """WebVid iterable dataloader (no __len__) with epoch_size attribute."""
    from torch.utils.data import DataLoader
    from .msrvtt import _ensure_pad_token

    query_tokenizer = _ensure_pad_token(query_tokenizer)
    target_tokenizer = _ensure_pad_token(target_tokenizer)

    data_cfg = cfg["data"]
    ds = build_webvid_pipeline(data_cfg, query_tokenizer, target_tokenizer)

    batch_size = data_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 4)

    total_samples = data_cfg.get("epoch_size", 1_500_000)
    world_size = 1
    if distributed:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
    epoch_size = (total_samples // max(1, batch_size)) // world_size

    ds = ds.batched(batch_size, collation_fn=_collate)
    loader = DataLoader(
        ds, batch_size=None, num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    loader.epoch_size = epoch_size
    return loader
