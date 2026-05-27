"""CC3M (Conceptual Captions 3M) webdataset loader for Stage A image-text pretraining.

Data structure (pixparse/cc3m-wds):
    cc3m-train-NNNN.tar
        000000000.jpg   ← image
        000000000.txt   ← caption
        000000000.json  ← metadata (unused)

For Stage A:
    - 1 frame per "video" input (image duplicated to satisfy V-JEPA2 tubelet_size=2)
    - Default retrieval prompt as query (since predictor expects query)
    - Caption tokenized via EmbeddingGemma tokenizer (Y-Encoder side)
"""

import io
import random
from pathlib import Path

import torch
import torchvision.transforms as T
import webdataset as wds
from PIL import Image

DEFAULT_QUERY = "Describe the image."


def _decode_image_caption(sample, transform, image_size):
    """Decode one webdataset sample: image (jpg bytes) + caption (txt bytes)."""
    img_bytes = sample.get("jpg") or sample.get("jpeg") or sample.get("png")
    cap_bytes = sample.get("txt") or sample.get("caption")
    if img_bytes is None or cap_bytes is None:
        return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None
    img = T.functional.resize(img, image_size, antialias=True)
    img = T.functional.center_crop(img, image_size)
    img = T.functional.to_tensor(img)  # (3, H, W) in [0,1]
    img = T.functional.normalize(
        img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    caption = cap_bytes.decode("utf-8") if isinstance(cap_bytes, bytes) else str(cap_bytes)
    return img, caption.strip()


def _tokenize(sample, query_tokenizer, target_tokenizer,
              query_text, max_query_len, max_caption_len, num_frames=1):
    """Tokenize and assemble the training dict for one sample."""
    if sample is None:
        return None
    img, caption = sample

    # Duplicate single frame to satisfy V-JEPA2 tubelet_size (typically 2)
    # We use num_frames=1 conceptually but stack T copies for V-JEPA's 3D conv input.
    # V-JEPA2's embedding code auto-duplicates if T < tubelet_size, so num_frames=1 is fine.
    frames = img.unsqueeze(0)  # (1, 3, H, W)
    if num_frames > 1:
        frames = frames.expand(num_frames, -1, -1, -1).contiguous()

    qenc = query_tokenizer(
        query_text, max_length=max_query_len, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    tenc = target_tokenizer(
        caption, max_length=max_caption_len, padding="max_length",
        truncation=True, return_tensors="pt"
    )

    return {
        "pixel_values": frames,
        "query_ids": qenc["input_ids"].squeeze(0),
        "query_mask": qenc["attention_mask"].squeeze(0),
        "caption_ids": tenc["input_ids"].squeeze(0),
        "caption_mask": tenc["attention_mask"].squeeze(0),
        "caption_text": caption,
    }


def build_cc3m_dataloader(cfg, query_tokenizer, target_tokenizer,
                           split: str = "train", distributed: bool = False):
    """Build CC3M webdataset dataloader for Stage A pretraining.

    Returns a torch DataLoader (iterable, not map-style — no __len__).
    """
    from torch.utils.data import DataLoader
    from .msrvtt import _ensure_pad_token
    query_tokenizer = _ensure_pad_token(query_tokenizer)
    target_tokenizer = _ensure_pad_token(target_tokenizer)
    data_cfg = cfg["data"]

    data_dir = Path(data_cfg["data_dir"])
    pattern = data_cfg.get("tar_pattern", f"cc3m-{split}-{{0000..0594}}.tar")
    tars = sorted(str(p) for p in data_dir.glob(f"cc3m-{split}-*.tar"))
    if not tars:
        raise FileNotFoundError(f"No CC3M tars found at {data_dir}")

    image_size = data_cfg.get("image_size", 256)
    num_frames = data_cfg.get("num_frames", 1)
    max_query_len = data_cfg.get("max_query_len", 64)
    max_caption_len = data_cfg.get("max_caption_len", 64)
    query_text = data_cfg.get("query_text", DEFAULT_QUERY)

    transform = None  # handled inline (image_size + crop + normalize)

    def decode(s):
        return _decode_image_caption(s, transform, image_size)

    def tokenize(s):
        return _tokenize(s, query_tokenizer, target_tokenizer,
                         query_text, max_query_len, max_caption_len, num_frames)

    # Build iterable webdataset pipeline
    ds = (
        wds.WebDataset(tars, shardshuffle=True, resampled=True, nodesplitter=wds.split_by_node)
        .shuffle(1000)
        .map(decode)
        .select(lambda x: x is not None)
        .map(tokenize)
        .select(lambda x: x is not None)
    )

    batch_size = data_cfg.get("batch_size", 200)
    num_workers = data_cfg.get("num_workers", 4)

    # Estimate dataset size for epoch length (cc3m has ~3M after filtering, conservatively 2.5M)
    # Webdataset iterators are infinite; we cap per-rank steps via epoch_size.
    total_samples = data_cfg.get("epoch_size", 2_500_000)
    world_size = 1
    if distributed:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
    epoch_size = (total_samples // max(1, batch_size)) // world_size
    ds = ds.batched(batch_size, collation_fn=_collate)

    loader = DataLoader(
        ds, batch_size=None, num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Attach a synthetic __len__ via with_epoch so train loop knows steps/epoch
    loader = loader  # webdataset loaders are infinite; we cap via epoch_size at trainer side
    loader.epoch_size = epoch_size
    return loader


def _collate(batch):
    """Collate list of sample dicts into a batched dict."""
    out = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out
