#!/bin/bash
# Re-download MSRVTT videos in parallel from URLs in annotations.json.
# Usage: bash scripts/redownload_msrvtt_videos.sh [test_only]
#   test_only: pass "test" to limit to test split (smaller set ~3K)

set -u

OUT_DIR=/data/junyeob/openvl-jepa-images/msrvtt/videos
ANN=/home/junyeob/openvl-jepa/data/msrvtt/annotations.json
SPLIT=${1:-all}
PARALLEL=${PARALLEL:-12}

mkdir -p "$OUT_DIR"

# Extract (id, url) pairs, optionally filter by split
python3 -c "
import json
ann = json.load(open('$ANN'))
target = '$SPLIT'
for vid, info in ann.items():
    if target != 'all' and info.get('split','train') != target: continue
    url = info.get('url','')
    if url: print(vid + ' ' + url)
" > /tmp/msrvtt_urls.txt
N=$(wc -l < /tmp/msrvtt_urls.txt)
echo "Total to download: $N (split=$SPLIT)"

# Skip already-downloaded
> /tmp/msrvtt_todo.txt
while IFS=' ' read -r vid url; do
    if [ ! -s "$OUT_DIR/${vid}.mp4" ]; then
        echo "$vid $url" >> /tmp/msrvtt_todo.txt
    fi
done < /tmp/msrvtt_urls.txt
TODO=$(wc -l < /tmp/msrvtt_todo.txt)
echo "Already have: $((N - TODO)) / Remaining: $TODO"

# Parallel wget (with timeout)
cat /tmp/msrvtt_todo.txt | xargs -P "$PARALLEL" -L 1 -I {} bash -c '
    set -- {}
    vid=$1; url=$2
    wget -q --timeout=30 --tries=2 -O "'"$OUT_DIR"'/${vid}.mp4" "$url" 2>/dev/null \
        || rm -f "'"$OUT_DIR"'/${vid}.mp4"
'

DONE=$(ls "$OUT_DIR"/*.mp4 2>/dev/null | wc -l)
echo "Final count: $DONE videos in $OUT_DIR"
