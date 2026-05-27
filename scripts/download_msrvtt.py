"""Download and prepare MSRVTT dataset.

Downloads video metadata + captions from HuggingFace,
then downloads videos from URLs.

Usage:
    python scripts/download_msrvtt.py --output_dir data/msrvtt
"""

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/msrvtt")
    parser.add_argument("--skip_videos", action="store_true",
                        help="Only download annotations, skip video files")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Max number of videos to download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MSRVTT annotations from HuggingFace...")
    ds = load_dataset("AlexZigma/msr-vtt")

    # Build annotations dict: {video_id: {"captions": [...], "split": "train"/"val"}}
    annotations = defaultdict(lambda: {"captions": [], "split": "train", "url": ""})

    for split_name in ds:
        for item in ds[split_name]:
            vid = item["video_id"]
            annotations[vid]["captions"].append(item["caption"])
            annotations[vid]["split"] = item["split"]
            if item.get("url"):
                annotations[vid]["url"] = item["url"]

    # Save annotations
    ann_path = output_dir / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(dict(annotations), f, indent=2)

    num_videos = len(annotations)
    num_captions = sum(len(v["captions"]) for v in annotations.values())
    print(f"Saved {num_videos} videos, {num_captions} captions to {ann_path}")

    if args.skip_videos:
        print("Skipping video download (--skip_videos)")
        return

    # Download videos
    print(f"\nDownloading videos to {video_dir}...")
    downloaded = 0
    failed = 0
    video_ids = sorted(annotations.keys())
    if args.max_videos:
        video_ids = video_ids[:args.max_videos]

    for vid in video_ids:
        url = annotations[vid]["url"]
        if not url:
            continue

        out_path = video_dir / f"{vid}.mp4"
        if out_path.exists():
            downloaded += 1
            continue

        try:
            subprocess.run(
                ["wget", "-q", "-O", str(out_path), url],
                timeout=60, check=True
            )
            downloaded += 1
            if downloaded % 100 == 0:
                print(f"  Downloaded {downloaded}/{len(video_ids)} videos")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            failed += 1
            out_path.unlink(missing_ok=True)

    print(f"\nDone: {downloaded} downloaded, {failed} failed")


if __name__ == "__main__":
    main()
