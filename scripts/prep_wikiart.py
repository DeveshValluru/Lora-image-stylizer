"""Stream-filter huggan/wikiart into Ukiyo-e and Van Gogh imagefolders, single pass.

Output:
  .cache/datasets/ukiyoe/    {img_*.png, metadata.jsonl}
  .cache/datasets/van_gogh/  {img_*.png, metadata.jsonl}

Streaming mode avoids the ~25 GB HF cache; only filtered images hit disk.
"""
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

OUT = {
    "ukiyoe":   Path(".cache/datasets/ukiyoe"),
    "van_gogh": Path(".cache/datasets/van_gogh"),
}
TRIGGER = {
    "ukiyoe":   "ukyoe_style",
    "van_gogh": "vngogh_style",
}
TARGET_STYLE_IDX  = 26   # 'Ukiyo_e'
TARGET_ARTIST_IDX = 22   # 'vincent-van-gogh'
MAX_PER_STYLE     = 2000  # safety cap; unlikely to be hit for either

for d in OUT.values():
    d.mkdir(parents=True, exist_ok=True)

print("Streaming huggan/wikiart (no full cache)...")
ds = load_dataset("huggan/wikiart", split="train", streaming=True)

features    = ds.features
artist_names = features["artist"].names
genre_names  = features["genre"].names if hasattr(features["genre"], "names") else None

counts   = {"ukiyoe": 0, "van_gogh": 0}
metadata = {"ukiyoe": [], "van_gogh": []}

pbar = tqdm(desc="wikiart", unit="rows")
for ex in ds:
    pbar.update(1)

    if ex["style"] == TARGET_STYLE_IDX and counts["ukiyoe"] < MAX_PER_STYLE:
        target = "ukiyoe"
    elif ex["artist"] == TARGET_ARTIST_IDX and counts["van_gogh"] < MAX_PER_STYLE:
        target = "van_gogh"
    else:
        continue

    # Build a simple caption from available metadata
    genre = genre_names[ex["genre"]] if (genre_names and ex.get("genre", -1) >= 0) else None
    if target == "ukiyoe":
        artist = artist_names[ex["artist"]].replace("-", " ") if ex["artist"] >= 0 else None
        caption = f"a {genre or 'painting'}{' by ' + artist if artist else ''} in ukiyo-e style"
    else:
        caption = f"a {genre or 'painting'} by Vincent van Gogh"

    idx = counts[target]
    file_name = f"img_{idx:05d}.png"
    ex["image"].convert("RGB").save(OUT[target] / file_name)
    metadata[target].append({
        "file_name": file_name,
        "text": f"{caption}, {TRIGGER[target]}",
    })
    counts[target] += 1
    pbar.set_postfix(counts)

pbar.close()

print()
for style, rows in metadata.items():
    out = OUT[style] / "metadata.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {style}: {len(rows):>4d} examples → {OUT[style]}")
    if rows:
        print(f"     sample: {rows[0]['text']}")