import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = Path(".cache/datasets/pixel_art")
TRIGGER = "pxlart_style"
N_SAMPLES = 2000
SEED = 42


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Dataset(cached after first run)")

ds = load_dataset("m1guelpf/nouns", split="train")
print(f"  full size: {len(ds)}")

ds = ds.shuffle(seed=SEED).select(range(N_SAMPLES))
print(f"subsampled to {len(ds)}")

print(f"Writing images + metadata to {OUTPUT_DIR}/ ...")

rows = []

for i, ex in enumerate(tqdm(ds)):
    file_name = f"img_{i:05d}.png"
    ex["image"].save(OUTPUT_DIR / file_name)
    rows.append({"file_name": file_name, "text": f"{ex['text']}, {TRIGGER}"})

with (OUTPUT_DIR / "metadata.jsonl").open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"\nDone — {N_SAMPLES} images written to {OUTPUT_DIR}/")
print(f"Sample caption: {rows[0]['text']}")