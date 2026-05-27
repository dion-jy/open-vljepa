"""Combined WebVid + MSRVTT dataloader for Stage B.

Treats both datasets as a single union pool. Per-sample Bernoulli draw
weights each source by its actual size (n_wv / (n_wv + n_mr)) — equivalent
to uniformly sampling from the union, so larger datasets dominate batches
proportionally.

Each DataLoader worker independently iterates both sources; WebVid shards
across DDP ranks via wds.split_by_node, MSRVTT is small enough (~140K) that
every rank cycling the full set with its own shuffle is acceptable for
contrastive training.
"""

import glob
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

from .webvid import build_webvid_pipeline, _collate
from .msrvtt import MSRVTTDataset, _ensure_pad_token


def _count_webvid_successes(data_dir: str) -> int:
    total = 0
    for s in glob.glob(os.path.join(data_dir, "*_stats.json")):
        try:
            with open(s) as f:
                total += int(json.load(f).get("successes", 0))
        except Exception:
            continue
    return total


class CombinedWebVidMSRVTT(IterableDataset):
    def __init__(self, webvid_pipeline, msrvtt_dataset, p_webvid=0.5, seed=0):
        super().__init__()
        self.wv = webvid_pipeline
        self.mr = msrvtt_dataset
        self.p = float(p_webvid)
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info else 0
        rng = random.Random(self.seed + wid * 7919 + os.getpid())

        wv_iter = iter(self.wv)
        n = len(self.mr)
        mr_order = list(range(n))
        rng.shuffle(mr_order)
        mr_i = 0

        while True:
            if rng.random() < self.p:
                try:
                    yield next(wv_iter)
                except StopIteration:
                    wv_iter = iter(self.wv)
                    try:
                        yield next(wv_iter)
                    except StopIteration:
                        # WebVid pipeline somehow exhausted; fall through to MSRVTT
                        pass
            else:
                if mr_i >= n:
                    rng.shuffle(mr_order)
                    mr_i = 0
                yield self.mr[mr_order[mr_i]]
                mr_i += 1


def build_combined_dataloader(cfg, query_tokenizer, target_tokenizer,
                              split: str = "train", distributed: bool = False):
    """Build combined WebVid (iterable) + MSRVTT (map-style) dataloader."""
    query_tokenizer = _ensure_pad_token(query_tokenizer)
    target_tokenizer = _ensure_pad_token(target_tokenizer)

    data_cfg = cfg["data"]

    # WebVid pipeline (uses data_cfg.data_dir for tar root)
    wv_pipeline = build_webvid_pipeline(data_cfg, query_tokenizer, target_tokenizer)

    # MSRVTT map-style dataset (uses data_cfg.msrvtt_data_dir)
    msrvtt_dir = data_cfg.get("msrvtt_data_dir") or data_cfg.get("msrvtt_dir")
    if not msrvtt_dir:
        raise ValueError("cfg.data.msrvtt_data_dir is required for combined dataset")
    msrvtt_ds = MSRVTTDataset(
        data_dir=msrvtt_dir,
        query_tokenizer=query_tokenizer,
        target_tokenizer=target_tokenizer,
        split=split,
        num_frames=data_cfg.get("num_frames", 8),
        image_size=data_cfg.get("image_size", 256),
        max_query_len=data_cfg.get("max_query_len", 64),
        max_caption_len=data_cfg.get("max_caption_len", 64),
    )

    # Auto-weight by actual dataset sizes — sample uniformly from the union
    n_wv = _count_webvid_successes(data_cfg["data_dir"])
    n_mr = len(msrvtt_ds)
    if n_wv + n_mr == 0:
        raise RuntimeError("Both WebVid and MSRVTT report 0 samples")
    p_webvid = n_wv / (n_wv + n_mr)
    print(f"[combined] WebVid={n_wv} MSRVTT={n_mr} → p_webvid={p_webvid:.3f}")
    combined = CombinedWebVidMSRVTT(
        wv_pipeline, msrvtt_ds, p_webvid=p_webvid, seed=0,
    )

    batch_size = data_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 4)

    total_samples = data_cfg.get("epoch_size", 1_500_000)
    world_size = 1
    if distributed:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
    epoch_size = (total_samples // max(1, batch_size)) // world_size

    loader = DataLoader(
        combined,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    loader.epoch_size = epoch_size
    return loader
