"""OpenVL-JEPA training script with DDP, bf16, grad accumulation, grad checkpointing.

Architecture: V-JEPA2 (frozen) + Llama-3.2-1B last 8 layers (predictor) + EmbeddingGemma-300M (Y-Encoder)
Loss: bi-directional InfoNCE

Usage:
    # Single GPU
    python scripts/train.py --config configs/base.yaml --dummy

    # Multi-GPU DDP
    torchrun --nproc_per_node=8 scripts/train.py --config configs/base.yaml
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from openvljepa.config import load_config
from openvljepa.models.vljepa import OpenVLJEPA
from openvljepa.data.msrvtt import MSRVTTDataset, build_dataloader

# In-training eval reuses helpers from scripts/eval.py
from scripts.eval import (  # noqa: E402
    VideoOnlyDataset, CaptionOnlyDataset,
    encode_videos, encode_captions,
    compute_t2v_metrics, compute_v2t_metrics,
    DEFAULT_RETRIEVAL_PROMPT,
)


# ---------------------------------------------------------------------------
# Distributed
# ---------------------------------------------------------------------------

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    return 0, 0, 1, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def log(msg: str, rank: int = 0):
    if is_main_process(rank):
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Optimizer / Scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, train_cfg: dict, y_encoder_lr_mult: float = 0.05):
    """Two-LR AdamW: y_encoder gets slow LR (x0.05). Predictor + projector get base LR."""
    lr = train_cfg["lr"]
    m = model.module if isinstance(model, DDP) else model

    # All predictor parameters (incl. vis_proj, out_proj)
    predictor_params = list(m.predictor.parameters())
    y_encoder_params = list(m.y_encoder.parameters())

    param_groups = [
        {"params": predictor_params, "lr": lr, "name": "predictor"},
        {"params": y_encoder_params, "lr": lr * y_encoder_lr_mult, "name": "y_encoder"},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=train_cfg.get("weight_decay", 0.01))


def build_scheduler(optimizer, train_cfg: dict, steps_per_epoch: int):
    """warmup_constant (paper-faithful) or cosine schedule."""
    warmup_steps = train_cfg.get("warmup_steps", 500)
    total_steps = train_cfg["epochs"] * steps_per_epoch
    schedule = train_cfg.get("schedule", "cosine")

    if schedule == "warmup_constant":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0  # constant after warmup (paper Sec 3.2)
    else:  # cosine
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Activation checkpointing
# ---------------------------------------------------------------------------

def enable_gradient_checkpointing(model: nn.Module):
    """Apply HF-style gradient checkpointing to predictor.layers and y_encoder.model."""
    m = model.module if isinstance(model, DDP) else model
    # EmbeddingGemma (HF model) supports gradient_checkpointing_enable()
    if hasattr(m.y_encoder.model, "gradient_checkpointing_enable"):
        m.y_encoder.model.gradient_checkpointing_enable()
    # Llama layers — apply per layer via torch.utils.checkpoint
    from torch.utils.checkpoint import checkpoint
    for layer in m.predictor.layers:
        original_forward = layer.forward
        def make_ckpt_forward(orig):
            def ckpt_forward(*args, **kwargs):
                return checkpoint(orig, *args, use_reentrant=False, **kwargs)
            return ckpt_forward
        layer.forward = make_ckpt_forward(original_forward)


# ---------------------------------------------------------------------------
# Per-epoch eval hook (DDP-distributed across all ranks)
# ---------------------------------------------------------------------------

def evaluate_during_training(model, cfg, query_tokenizer, target_tokenizer,
                              device, rank, epoch, ckpt_dir):
    """Run T2V/V2T retrieval eval at end of epoch using ALL ranks in parallel.

    Each rank encodes a shard of videos + captions via DistributedSampler, then
    embeddings are all-gathered. Rank 0 computes final metrics and writes the
    history JSON; other ranks return None.

    DistributedSampler pads to equal length across ranks, so a few samples may be
    duplicated at the tail — we de-dup by video_id (videos) and by index (captions).

    Returns (t2v_metrics, v2t_metrics) on rank 0, None elsewhere.
    """
    eval_cfg = cfg.get("eval", {})
    every_n = int(eval_cfg.get("every_n_epochs", 0) or 0)
    at_epochs = eval_cfg.get("at_epochs", None) or []
    max_seconds = eval_cfg.get("max_seconds", None)

    # Triggers: (every_n > 0 AND match) OR (explicit list AND match)
    by_every = every_n > 0 and (epoch + 1) % every_n == 0
    by_list = bool(at_epochs) and (epoch + 1) in at_epochs
    if not (by_every or by_list):
        return None  # no eval this epoch

    # Safety cap decision: rank 0 owns last_elapsed; broadcast skip flag so all ranks agree
    skip_flag = 0
    if rank == 0 and max_seconds is not None:
        last_elapsed = getattr(evaluate_during_training, "_last_elapsed_s", 0.0)
        if last_elapsed > max_seconds:
            skip_flag = 1
            log(f"  [Eval epoch {epoch+1}] skipped (last eval took "
                f"{last_elapsed:.0f}s > max_seconds={max_seconds})", rank)

    if dist.is_initialized() and dist.get_world_size() > 1:
        skip_tensor = torch.tensor([skip_flag], device=device, dtype=torch.long)
        dist.broadcast(skip_tensor, src=0)
        skip_flag = int(skip_tensor.item())

    if skip_flag:
        return None

    distributed = dist.is_initialized() and dist.get_world_size() > 1
    world_size = dist.get_world_size() if distributed else 1

    n_videos = eval_cfg.get("n_videos", None)
    eval_bs = int(eval_cfg.get("batch_size", 8))

    m = model.module if isinstance(model, DDP) else model
    was_training = m.training
    m.eval()

    t2v = None
    v2t = None
    try:
        if rank == 0:
            log(f"  [Eval epoch {epoch+1}] DDP-distributed eval on {world_size} GPUs...", rank)
        test_ds = MSRVTTDataset(
            data_dir=cfg.get("eval", {}).get("data_dir", cfg["data"]["data_dir"]),
            query_tokenizer=query_tokenizer,
            target_tokenizer=target_tokenizer,
            split="test",
            num_frames=cfg["data"]["num_frames"],
            image_size=cfg["data"]["image_size"],
            max_query_len=cfg["data"]["max_query_len"],
            max_caption_len=cfg["data"]["max_caption_len"],
        )

        # Unique videos (deterministic across ranks — sorted, then take first N)
        seen = set()
        vids = []
        for s in test_ds.samples:
            if s["video_id"] not in seen:
                seen.add(s["video_id"])
                vids.append(s["video_id"])
                if n_videos and len(vids) >= n_videos:
                    break

        cap_dataset = CaptionOnlyDataset(test_ds)
        if n_videos:
            cap_dataset.base.samples = [s for s in cap_dataset.base.samples if s["video_id"] in seen]

        vid_dataset = VideoOnlyDataset(test_ds, vids)

        # DistributedSampler shards across ranks, pads to equal length so all-gather works
        if distributed:
            from torch.utils.data.distributed import DistributedSampler
            vid_sampler = DistributedSampler(vid_dataset, shuffle=False, drop_last=False)
            cap_sampler = DistributedSampler(cap_dataset, shuffle=False, drop_last=False)
        else:
            vid_sampler = cap_sampler = None

        vid_loader = DataLoader(vid_dataset, batch_size=eval_bs, sampler=vid_sampler,
                                num_workers=2, pin_memory=True)
        cap_loader = DataLoader(cap_dataset, batch_size=eval_bs * 4, sampler=cap_sampler,
                                num_workers=2, pin_memory=True)

        qenc = query_tokenizer(
            DEFAULT_RETRIEVAL_PROMPT,
            max_length=cfg["data"]["max_query_len"],
            padding="max_length", truncation=True, return_tensors="pt",
        )
        q_ids = qenc["input_ids"].to(device)
        q_mask = qenc["attention_mask"].to(device)

        t0 = time.time()
        # Each rank encodes its shard
        local_video_embeds, local_video_ids = encode_videos(m, vid_loader, q_ids, q_mask, device)
        local_caption_embeds, local_cap_video_ids, _ = encode_captions(m, cap_loader, device)

        # All-gather across ranks
        if distributed:
            # Move tensors back to GPU for all_gather, contiguous
            lv = local_video_embeds.to(device).contiguous()
            lc = local_caption_embeds.to(device).contiguous()
            gathered_v = [torch.zeros_like(lv) for _ in range(world_size)]
            gathered_c = [torch.zeros_like(lc) for _ in range(world_size)]
            dist.all_gather(gathered_v, lv)
            dist.all_gather(gathered_c, lc)

            # IDs are strings — gather as Python objects
            gathered_vid_ids = [None] * world_size
            gathered_cap_ids = [None] * world_size
            dist.all_gather_object(gathered_vid_ids, local_video_ids)
            dist.all_gather_object(gathered_cap_ids, local_cap_video_ids)

            video_embeds = torch.cat(gathered_v, dim=0).cpu()
            caption_embeds = torch.cat(gathered_c, dim=0).cpu()
            video_ids = [vid for sub in gathered_vid_ids for vid in sub]
            cap_video_ids = [vid for sub in gathered_cap_ids for vid in sub]
        else:
            video_embeds = local_video_embeds
            caption_embeds = local_caption_embeds
            video_ids = local_video_ids
            cap_video_ids = local_cap_video_ids

        # Only rank 0 computes metrics + writes history
        if rank == 0:
            # De-duplicate (DistributedSampler may have padded with repeats)
            seen_v = {}
            for i, vid in enumerate(video_ids):
                if vid not in seen_v:
                    seen_v[vid] = i
            uniq_v_idx = list(seen_v.values())
            video_embeds = video_embeds[uniq_v_idx]
            video_ids = [video_ids[i] for i in uniq_v_idx]

            # Captions: dedup by full (cap_video_id, cap_idx_in_global) pair.
            # Since cap_dataset is deterministic, simplest is to keep first N=len(cap_dataset) entries.
            n_caps = len(cap_dataset)
            caption_embeds = caption_embeds[:n_caps]
            cap_video_ids = cap_video_ids[:n_caps]

            video_embeds = F.normalize(video_embeds.float(), dim=-1)
            caption_embeds = F.normalize(caption_embeds.float(), dim=-1)

            t2v = compute_t2v_metrics(caption_embeds, cap_video_ids, video_embeds, video_ids)
            v2t = compute_v2t_metrics(video_embeds, video_ids, caption_embeds, cap_video_ids)
            elapsed = time.time() - t0

            log(f"  [Eval epoch {epoch+1}] "
                f"T2V R@1={t2v['R@1']:.2f} R@5={t2v['R@5']:.2f} R@10={t2v['R@10']:.2f} MedR={t2v['MedianR']:.0f} | "
                f"V2T R@1={v2t['R@1']:.2f} R@5={v2t['R@5']:.2f} R@10={v2t['R@10']:.2f} | "
                f"pool={t2v['pool_size']} queries={t2v['num_queries']} time={elapsed:.1f}s", rank)

            history_path = Path(ckpt_dir) / "eval_history.json"
            history = []
            if history_path.exists():
                try:
                    with open(history_path) as f:
                        history = json.load(f)
                except json.JSONDecodeError:
                    history = []
            history.append({
                "epoch": epoch + 1,
                "t2v": t2v, "v2t": v2t,
                "elapsed_s": elapsed,
            })
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)

            evaluate_during_training._last_elapsed_s = elapsed
    except Exception as e:
        log(f"  [Eval epoch {epoch+1}] ERROR — eval failed on rank {rank}: "
            f"{type(e).__name__}: {e}", 0)  # log only on rank 0 to avoid spam, but
        # NOTE: the barrier in finally keeps DDP healthy even when one rank errors
    finally:
        # Restore train state regardless of success/failure
        if was_training:
            m.train()
            m.x_encoder.eval()  # x_encoder (model + projector) stays frozen
        # Mandatory barrier to release all ranks
        if dist.is_initialized():
            dist.barrier()

    return t2v, v2t


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, scheduler,
                    train_cfg, device, epoch, rank=0, steps_per_epoch=None,
                    ckpt_dir=None, cfg=None):
    model.train()
    m = model.module if isinstance(model, DDP) else model
    m.x_encoder.model.eval()  # keep V-JEPA2 in eval mode (frozen)

    total_loss = 0.0
    num_steps = 0
    temperature = train_cfg.get("temperature", 0.07)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    log_every = train_cfg.get("log_every", 50)
    accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    use_bf16 = train_cfg.get("bf16", True)
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    save_every_steps = int(train_cfg.get("save_every_steps", 0))  # 0 = disabled

    # webdataset iterators are infinite; cap by steps_per_epoch
    total_steps = steps_per_epoch if steps_per_epoch is not None else None

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        if total_steps is not None and step >= total_steps:
            break
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        query_ids = batch["query_ids"].to(device, non_blocking=True)
        query_mask = batch["query_mask"].to(device, non_blocking=True)
        caption_ids = batch["caption_ids"].to(device, non_blocking=True)
        caption_mask = batch["caption_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            loss = model(pixel_values, query_ids, query_mask,
                         caption_ids, caption_mask, temperature)

        (loss / accum_steps).backward()

        if (step + 1) % accum_steps == 0:
            grad_norm = nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                grad_clip,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        else:
            grad_norm = torch.tensor(0.0)

        total_loss += loss.item()
        num_steps += 1

        if (step + 1) % log_every == 0:
            avg_loss = total_loss / num_steps
            lr_pred = optimizer.param_groups[0]["lr"]
            lr_yenc = optimizer.param_groups[1]["lr"]
            denom = total_steps if total_steps is not None else "?"
            log(f"  [Epoch {epoch+1} Step {step+1}/{denom}] "
                f"loss={avg_loss:.4f} grad_norm={grad_norm:.4f} "
                f"lr_pred={lr_pred:.2e} lr_yenc={lr_yenc:.2e}", rank)

        # Step-level checkpoint for crash resilience (overwrites latest.pt)
        if (save_every_steps > 0 and ckpt_dir is not None
                and (step + 1) % save_every_steps == 0 and rank == 0):
            sd = {k: v for k, v in m.state_dict().items()
                  if not k.startswith("x_encoder.model.")}
            tmp_path = Path(ckpt_dir) / "latest.pt.tmp"
            torch.save({
                "epoch": epoch + 1,
                "step": step + 1,
                "model_state_dict": sd,
                "loss": total_loss / max(num_steps, 1),
                "config": cfg,
            }, tmp_path)
            tmp_path.replace(Path(ckpt_dir) / "latest.pt")
            log(f"  [Epoch {epoch+1} Step {step+1}] Saved latest.pt", rank)

    return total_loss / max(num_steps, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--dummy_samples", type=int, default=200)
    parser.add_argument("--init_from", type=str, default=None,
                        help="Path to a checkpoint (.pt) to initialize model weights from "
                             "(e.g., Stage A best.pt for Stage B retrain). x_encoder weights "
                             "are re-fetched from HF; only predictor + y_encoder weights load.")
    args = parser.parse_args()

    rank, local_rank, world_size, distributed = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    log("=" * 60, rank)
    log(f"OpenVL-JEPA Training  |  GPUs: {world_size}  |  DDP: {distributed}", rank)
    log("=" * 60, rank)

    cfg = load_config(args.config)
    train_cfg = cfg["training"]

    if not distributed:
        device = torch.device(train_cfg.get("device", "cuda:0"))

    # Tokenizers: Llama for query, EmbeddingGemma for target
    log("Loading tokenizers...", rank)
    query_tokenizer = AutoTokenizer.from_pretrained(cfg["predictor"]["llama_name"])
    target_tokenizer = AutoTokenizer.from_pretrained(cfg["y_encoder"]["model_name"])

    # Build model (load pretrained weights in bf16 to save memory)
    use_bf16 = train_cfg.get("bf16", True)
    model_dtype = torch.bfloat16 if use_bf16 else None
    log(f"Building model (dtype={'bf16' if use_bf16 else 'fp32'})...", rank)
    model = OpenVLJEPA(cfg["encoder"], cfg["y_encoder"], cfg["predictor"],
                       torch_dtype=model_dtype).to(device)

    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log(f"Parameters: {total_p:.1f}M total, {train_p:.1f}M trainable", rank)

    # Optional: initialize predictor + y_encoder weights from a prior stage's checkpoint
    init_from = args.init_from or train_cfg.get("init_from")
    if init_from:
        log(f"Loading init weights from: {init_from}", rank)
        init_ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
        init_state = init_ckpt["model_state_dict"]
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        non_xenc_missing = [k for k in missing if not k.startswith("x_encoder.model.")]
        log(f"  Loaded {len(init_state)} keys "
            f"(missing={len(missing)} all x_encoder.* expected; "
            f"non-xenc missing={len(non_xenc_missing)}, unexpected={len(unexpected)})", rank)
        if non_xenc_missing:
            log(f"  WARN: non-frozen missing keys: {non_xenc_missing[:5]}", rank)
        del init_ckpt, init_state

    # Gradient checkpointing
    if train_cfg.get("gradient_checkpointing", False):
        enable_gradient_checkpointing(model)
        log("Enabled gradient checkpointing on predictor + y_encoder", rank)

    # DDP wrap
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        log(f"Wrapped model with DDP on {world_size} GPUs", rank)

    # Dataloader
    if args.dummy:
        cfg["data"]["dummy"] = True
        cfg["data"]["num_samples"] = args.dummy_samples
    dataloader = build_dataloader(
        cfg, query_tokenizer=query_tokenizer, target_tokenizer=target_tokenizer,
        split="train", distributed=distributed,
    )
    # webdataset is iterable (no __len__) — use epoch_size attribute if present
    if hasattr(dataloader, "epoch_size"):
        steps_per_epoch = int(dataloader.epoch_size)
        log(f"Dataset: webdataset, steps_per_epoch={steps_per_epoch} (epoch_size)", rank)
    else:
        steps_per_epoch = len(dataloader)
        log(f"Dataset: {len(dataloader.dataset)} samples, {steps_per_epoch} batches/gpu", rank)

    # Optimizer + scheduler
    y_lr_mult = cfg["y_encoder"].get("lr_multiplier", 0.05)
    optimizer = build_optimizer(model, train_cfg, y_encoder_lr_mult=y_lr_mult)
    scheduler = build_scheduler(optimizer, train_cfg, steps_per_epoch)

    # Checkpoint dir
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    if is_main_process(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Loop
    log(f"\nStarting training for {train_cfg['epochs']} epochs...", rank)
    best_loss = float("inf")

    for epoch in range(train_cfg["epochs"]):
        if distributed and hasattr(dataloader, "sampler") and \
                dataloader.sampler is not None and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        t0 = time.time()
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, scheduler,
            train_cfg, device, epoch, rank,
            steps_per_epoch=steps_per_epoch,
            ckpt_dir=ckpt_dir, cfg=cfg,
        )
        elapsed = time.time() - t0

        if distributed:
            loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        log(f"Epoch {epoch+1}/{train_cfg['epochs']}: "
            f"loss={avg_loss:.4f} time={elapsed:.1f}s "
            f"VRAM={torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB", rank)

        if is_main_process(rank):
            save_every = int(train_cfg.get("save_every", 5))
            keep_last_n = train_cfg.get("keep_last_n", None)  # None = keep all
            save_opt = bool(train_cfg.get("save_optimizer_state", False))

            m = model.module if isinstance(model, DDP) else model
            state_dict = {
                k: v for k, v in m.state_dict().items()
                if not k.startswith("x_encoder.model.")
            }

            # epoch_N.pt: ONLY at save_every cadence (decoupled from best update)
            if (epoch + 1) % save_every == 0:
                payload = {
                    "epoch": epoch + 1,
                    "model_state_dict": state_dict,
                    "loss": avg_loss,
                    "config": cfg,
                }
                if save_opt:
                    payload["optimizer_state_dict"] = optimizer.state_dict()
                ckpt_path = ckpt_dir / f"epoch_{epoch+1}.pt"
                torch.save(payload, ckpt_path)
                log(f"  Saved checkpoint: {ckpt_path} "
                    f"({'with' if save_opt else 'no'} optimizer)", rank)

                # Rolling cleanup: keep only the most recent `keep_last_n` epoch_*.pt
                if keep_last_n is not None and int(keep_last_n) > 0:
                    existing = sorted(
                        ckpt_dir.glob("epoch_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]),
                    )
                    for old in existing[: -int(keep_last_n)]:
                        try:
                            old.unlink()
                            log(f"  Pruned old checkpoint: {old.name}", rank)
                        except OSError as e:
                            log(f"  Failed to prune {old.name}: {e}", rank)

            # best.pt: ALWAYS save on improvement (lightweight, no optimizer)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": state_dict,
                    "loss": avg_loss,
                    "config": cfg,
                }, ckpt_dir / "best.pt")

        if distributed:
            dist.barrier()

        # Per-epoch eval (gated by cfg.eval.every_n_epochs; rank 0 only, others wait)
        evaluate_during_training(
            model, cfg, query_tokenizer, target_tokenizer,
            device, rank, epoch, ckpt_dir,
        )

    log(f"\nTraining complete. Best loss: {best_loss:.4f}", rank)
    cleanup_distributed()


if __name__ == "__main__":
    main()
