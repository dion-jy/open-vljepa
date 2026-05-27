"""MSRVTT dataset for VL-JEPA.

Two tokenizers per sample:
  - query_tokenizer:  Llama-3.2-1B tokenizer (for Predictor input)
  - target_tokenizer: EmbeddingGemma tokenizer (for Y-Encoder input)
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False


DEFAULT_QUERIES = [
    "Describe the video.",
    "What is happening in the video?",
    "What do you see?",
    "Summarize the video content.",
    "What is the main action?",
    "Describe what is shown.",
    "What is going on here?",
    "Explain the video.",
]


def _ensure_pad_token(tokenizer):
    """Llama-3.2 tokenizer ships without a pad token; assigning eos as pad conflates
    them (real query's EOS would be masked out). Llama-3.2 includes a reserved pad
    token at id 128004 (`<|finetune_right_pad_id|>`). Use it where available."""
    if tokenizer.pad_token is not None:
        return tokenizer
    # Try Llama-3.2's reserved pad token first
    LLAMA32_PAD = "<|finetune_right_pad_id|>"
    vocab = tokenizer.get_vocab()
    if LLAMA32_PAD in vocab:
        tokenizer.pad_token = LLAMA32_PAD
        tokenizer.pad_token_id = vocab[LLAMA32_PAD]
    else:
        # Fallback for other tokenizers (EmbeddingGemma already has pad)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


class MSRVTTDataset(Dataset):
    """Video + query + caption triplets.

    annotations.json schema:
      { "<video_id>": {"captions": [...], "split": "train"|"test"}, ... }
    """

    def __init__(self, data_dir: str, query_tokenizer, target_tokenizer,
                 split: str = "train", num_frames: int = 16, image_size: int = 256,
                 max_query_len: int = 512, max_caption_len: int = 512):
        self.data_dir = Path(data_dir)
        self.video_dir = self.data_dir / "videos"
        self.query_tokenizer = _ensure_pad_token(query_tokenizer)
        self.target_tokenizer = _ensure_pad_token(target_tokenizer)
        self.num_frames = num_frames
        self.max_query_len = max_query_len
        self.max_caption_len = max_caption_len

        ann_file = self.data_dir / "annotations.json"
        with open(ann_file) as f:
            all_annotations = json.load(f)

        self.samples = []
        for video_id, info in all_annotations.items():
            if info.get("split", "train") == split:
                for caption in info["captions"]:
                    self.samples.append({"video_id": video_id, "caption": caption})

        self.transform = T.Compose([
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def _load_video(self, video_path: Path) -> torch.Tensor:
        if not HAS_DECORD:
            raise ImportError("decord is required for video loading")
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total = len(vr)
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        return self.transform(frames)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample["video_id"]
        caption = sample["caption"]

        frames = self._load_video(self.video_dir / f"{video_id}.mp4")
        query = random.choice(DEFAULT_QUERIES)

        qenc = self.query_tokenizer(
            query, max_length=self.max_query_len,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        tenc = self.target_tokenizer(
            caption, max_length=self.max_caption_len,
            padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "pixel_values": frames,
            "query_ids": qenc["input_ids"].squeeze(0),
            "query_mask": qenc["attention_mask"].squeeze(0),
            "caption_ids": tenc["input_ids"].squeeze(0),
            "caption_mask": tenc["attention_mask"].squeeze(0),
            "video_id": video_id,
            "caption_text": caption,
        }


class DummyMSRVTTDataset(Dataset):
    """Random tensors for shape/gradient sanity checks (no HF access required)."""

    def __init__(self, num_samples: int = 200, num_frames: int = 16, image_size: int = 256,
                 query_vocab_size: int = 128256, target_vocab_size: int = 256000,
                 max_query_len: int = 64, max_caption_len: int = 64):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.image_size = image_size
        self.query_vocab_size = query_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_query_len = max_query_len
        self.max_caption_len = max_caption_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        qlen = torch.randint(3, self.max_query_len + 1, (1,)).item()
        clen = torch.randint(3, self.max_caption_len + 1, (1,)).item()
        qmask = torch.zeros(self.max_query_len, dtype=torch.long)
        qmask[:qlen] = 1
        cmask = torch.zeros(self.max_caption_len, dtype=torch.long)
        cmask[:clen] = 1
        return {
            "pixel_values": torch.randn(self.num_frames, 3, self.image_size, self.image_size),
            "query_ids": torch.randint(0, self.query_vocab_size, (self.max_query_len,)),
            "query_mask": qmask,
            "caption_ids": torch.randint(0, self.target_vocab_size, (self.max_caption_len,)),
            "caption_mask": cmask,
            "video_id": f"video{idx}",
            "caption_text": f"dummy caption {idx}",
        }


def build_dataloader(cfg: dict, query_tokenizer=None, target_tokenizer=None,
                     split: str = "train", distributed: bool = False) -> DataLoader:
    """Dispatch dataloader builder based on cfg['data']['dataset'] or 'loader'."""
    data_cfg = cfg["data"]

    # Dispatch: CC3M (Stage A) and WebVid (Stage B video-text) use webdataset builders
    if data_cfg.get("dataset") == "cc3m":
        from .cc3m import build_cc3m_dataloader
        return build_cc3m_dataloader(
            cfg, query_tokenizer=query_tokenizer,
            target_tokenizer=target_tokenizer,
            split=split, distributed=distributed,
        )
    if data_cfg.get("dataset") == "webvid":
        from .webvid import build_webvid_dataloader
        return build_webvid_dataloader(
            cfg, query_tokenizer=query_tokenizer,
            target_tokenizer=target_tokenizer,
            split=split, distributed=distributed,
        )
    if data_cfg.get("dataset") == "combined":
        from .combined import build_combined_dataloader
        return build_combined_dataloader(
            cfg, query_tokenizer=query_tokenizer,
            target_tokenizer=target_tokenizer,
            split=split, distributed=distributed,
        )

    if data_cfg.get("dummy", False):
        dataset = DummyMSRVTTDataset(
            num_samples=data_cfg.get("num_samples", 200),
            num_frames=data_cfg.get("num_frames", 16),
            image_size=data_cfg.get("image_size", 256),
            max_query_len=data_cfg.get("max_query_len", 64),
            max_caption_len=data_cfg.get("max_caption_len", 64),
        )
    else:
        dataset = MSRVTTDataset(
            data_dir=data_cfg["data_dir"],
            query_tokenizer=query_tokenizer,
            target_tokenizer=target_tokenizer,
            split=split,
            num_frames=data_cfg.get("num_frames", 16),
            image_size=data_cfg.get("image_size", 256),
            max_query_len=data_cfg.get("max_query_len", 512),
            max_caption_len=data_cfg.get("max_caption_len", 512),
        )

    sampler = None
    shuffle = (split == "train")
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 4),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=(split == "train"),
    )
