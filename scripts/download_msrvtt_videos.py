"""Download MSRVTT videos from YouTube using yt-dlp."""
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_video(vid, url, video_dir):
    out_path = video_dir / f"{vid}.mp4"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return vid, True, "exists"
    try:
        subprocess.run(
            ["yt-dlp", "-q", "--no-warnings",
             "-f", "worst[ext=mp4]/worst",  # smallest format to save space
             "--max-filesize", "50M",
             "-o", str(out_path),
             url],
            timeout=120, check=True,
            capture_output=True
        )
        return vid, True, "downloaded"
    except Exception as e:
        out_path.unlink(missing_ok=True)
        return vid, False, str(e)[:80]

def main():
    ann_path = Path("/home/junyeob/openvl-jepa/data/msrvtt/annotations.json")
    video_dir = Path("/home/junyeob/openvl-jepa/data/msrvtt/videos")
    video_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_path) as f:
        annotations = json.load(f)

    # Filter train split only first
    train_vids = {k: v for k, v in annotations.items()
                  if v.get("split") == "train" and v.get("url")}
    print(f"Total train videos to download: {len(train_vids)}")

    downloaded = 0
    failed = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(download_video, vid, info["url"], video_dir): vid
            for vid, info in train_vids.items()
        }
        for future in as_completed(futures):
            vid, success, msg = future.result()
            if success:
                if msg == "exists":
                    skipped += 1
                else:
                    downloaded += 1
            else:
                failed += 1
            total = downloaded + failed + skipped
            if total % 50 == 0:
                print(f"Progress: {total}/{len(train_vids)} "
                      f"(dl={downloaded} skip={skipped} fail={failed})")

    print(f"\nDone: {downloaded} downloaded, {skipped} existing, {failed} failed")

if __name__ == "__main__":
    main()
