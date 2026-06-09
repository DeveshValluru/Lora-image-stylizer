"""Verify the loratraining venv has everything needed."""
import importlib

REQUIRED = [
    # (display name, import name)
    ("torch",        "torch"),
    ("torchvision",  "torchvision"),
    ("diffusers",    "diffusers"),
    ("transformers", "transformers"),
    ("accelerate",   "accelerate"),
    ("peft",         "peft"),
    ("datasets",     "datasets"),
    ("bitsandbytes", "bitsandbytes"),
    ("safetensors",  "safetensors"),
    ("fastapi",      "fastapi"),
    ("uvicorn",      "uvicorn"),
    ("gradio",       "gradio"),
    ("pyyaml",       "yaml"),
    ("pillow",       "PIL"),
    ("tqdm",         "tqdm"),
]

ok, missing = [], []
for name, import_name in REQUIRED:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        ok.append((name, ver))
        print(f"  [OK]      {name:14s} {ver}")
    except ImportError:
        missing.append(name)
        print(f"  [MISSING] {name}")

print()
import torch
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print()
if missing:
    print(f"MISSING {len(missing)} packages: {missing}")
    print("Install with: pip install " + " ".join(missing))
else:
    print(f"All {len(ok)} required packages installed.")