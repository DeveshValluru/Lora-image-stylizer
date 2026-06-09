# `loras/`

Trained LoRA adapters live here, one folder per style:

```
loras/
├── pixel_art/pytorch_lora_weights.safetensors
├── ukiyoe/pytorch_lora_weights.safetensors
└── van_gogh/pytorch_lora_weights.safetensors
```

These files are **gitignored** — they're ~30 MB each and not the kind of binary asset that belongs in a git history.

To regenerate them, follow the [Reproduction section in the root README](../README.md#reproduction-local-3-5-hours-end-to-end).

`serve.py` and `scripts/test_lora.py` both expect this exact layout (`loras/{style}/pytorch_lora_weights.safetensors`). If you train with a different `--output_dir`, you'll need to move the resulting `pytorch_lora_weights.safetensors` into the matching slot here.
