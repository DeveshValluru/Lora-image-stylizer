from datasets import load_dataset

ds = load_dataset("m1guelpf/nouns")

print("DatasetDict:", ds)
print("\nSplits:", list(ds.keys()))
print("\nFirst example keys:", list(ds["train"][0].keys()))
print("\nFirst example (preview):")

ex = ds["train"][0]
for k, v in ex.items():
    if hasattr(v, "size") and hasattr(v, "mode"):       # PIL image
        print(f"  {k}: PIL image, size={v.size}, mode={v.mode}")
    else:
        s = str(v)
        print(f"  {k}: {s[:200]}{'...' if len(s) > 200 else ''}")

print(f"\nTotal train examples: {len(ds['train'])}")