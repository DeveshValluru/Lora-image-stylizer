"""Inspect huggan/wikiart label vocabulary without downloading any images.

WikiArt is ~25 GB if fully downloaded; we only want to confirm label spellings
before deciding what to actually pull.
"""
from datasets import load_dataset

print("Streaming wikiart metadata only (no image download)...")
ds = load_dataset("huggan/wikiart", split="train", streaming=True)
features = ds.features

print("\nColumns:", list(features.keys()))

# --- Styles ---
styles = features["style"].names if hasattr(features["style"], "names") else None
if styles:
    print(f"\nAll {len(styles)} styles:")
    for i, s in enumerate(styles):
        print(f"  {i:3d}: {s}")

    print("\nUkiyo-e matches:")
    for i, s in enumerate(styles):
        if "ukiyo" in s.lower():
            print(f"  {i:3d}: {s!r}")

# --- Artists ---
artists = features["artist"].names if hasattr(features["artist"], "names") else None
if artists:
    print(f"\nTotal artists: {len(artists)}")
    print("\nVan Gogh matches:")
    for i, a in enumerate(artists):
        if "gogh" in a.lower():
            print(f"  {i:3d}: {a!r}")