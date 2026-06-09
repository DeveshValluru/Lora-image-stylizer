# Three-Style LoRA Image Stylizer

Local image-to-image style transfer for an **RTX 4060 (8 GB)** or similar. Three LoRA adapters trained on top of Stable Diffusion 1.5, hot-swappable behind a single FastAPI + Gradio service. No paid APIs, no cloud, no GPU rental — the whole thing runs on a laptop.

| Style | Trigger | Trained on |
|---|---|---|
| **Pixel Art** | `pxlart_style` | [`m1guelpf/nouns`](https://huggingface.co/datasets/m1guelpf/nouns) (subsampled to 2,000) |
| **Ukiyo-e** | `ukyoe_style` | [`huggan/wikiart`](https://huggingface.co/datasets/huggan/wikiart) filtered to `style == "Ukiyo_e"` (1,167 imgs) |
| **Van Gogh** | `vngogh_style` | [`huggan/wikiart`](https://huggingface.co/datasets/huggan/wikiart) filtered to `artist == "vincent-van-gogh"` (1,889 imgs) |

See [`SPEC.md`](SPEC.md) for the full design doc, hyperparameter table, and risk log.

---

## Architecture

```
                       ┌─────────────────────────────────────────┐
                       │           SD 1.5 base UNet              │
                       │      (loaded once, ~3 GB VRAM)          │
                       └────────────────┬────────────────────────┘
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            ↓                           ↓                           ↓
     pixel_art LoRA              ukiyoe LoRA                van_gogh LoRA
     (~30 MB adapter)            (~30 MB adapter)           (~30 MB adapter)

  ┌───────────────────────────────────────────────────────────────────┐
  │      FastAPI service  +  Gradio UI  (uvicorn, port 8000)          │
  │   pipe.set_adapters([style])  hot-swap per request                │
  └───────────────────────────────────────────────────────────────────┘
```

- **One SD 1.5 base model** loaded into VRAM at startup.
- **All three LoRAs** loaded as named adapters via `pipeline.load_lora_weights(..., adapter_name=...)`.
- Each request activates one with `pipeline.set_adapters([style])` — zero extra VRAM cost per adapter beyond ~30 MB each.
- Single process: REST endpoints + Gradio UI live on the same port.

---

## Reproduction (local, ~5 hours end to end)

Requires Python 3.12, an NVIDIA GPU with **≥8 GB VRAM**, and a recent driver (CUDA 12.6+ recommended). Tested on RTX 4060 Laptop.

### 1. Clone + create venv

```powershell
# Windows / PowerShell
python -m venv loratraining
.\loratraining\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python -m venv loratraining
source loratraining/bin/activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

Sanity check:

```
python scripts/check_env.py
```

Should report all 15 packages installed, `CUDA available: True`, and your GPU + VRAM.

### 3. Hugging Face login

```
hf auth login
```

(Or set the `HF_TOKEN` environment variable.)

### 4. Configure accelerate

```
accelerate config default --mixed_precision bf16
```

### 5. Pre-cache SD 1.5 weights (~5 GB, one-time)

```
python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5')"
```

### 6. Prepare datasets

```
python scripts/prep_pixel_art.py        # ~5 min  | ~100 MB on disk
python scripts/prep_wikiart.py          # ~75 min | ~25 GB streamed, ~100 MB on disk
```

The WikiArt prep streams the entire dataset (no full cache) and pulls out both the Ukiyo-e and Van Gogh subsets in one pass.

### 7. Train the three LoRAs

**Pixel Art** (~30 min on RTX 4060):

```powershell
accelerate launch scripts/train_text_to_image_lora.py `
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" `
  --train_data_dir=".cache/datasets/pixel_art" `
  --image_column="image" --caption_column="text" `
  --resolution=512 --center_crop --random_flip `
  --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing `
  --max_train_steps=1200 --learning_rate=1e-4 `
  --lr_scheduler="constant" --lr_warmup_steps=100 `
  --rank=16 --mixed_precision="bf16" --use_8bit_adam `
  --seed=42 --output_dir="loras/pixel_art" `
  --validation_prompt="a pixel art character with a wizard hat, pxlart_style" `
  --validation_epochs=1 --checkpointing_steps=300 --report_to="tensorboard"
```

**Ukiyo-e** (~2 hr — bigger images, slower per-step):

```powershell
accelerate launch scripts/train_text_to_image_lora.py `
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" `
  --train_data_dir=".cache/datasets/ukiyoe" `
  --image_column="image" --caption_column="text" `
  --resolution=512 --center_crop --random_flip `
  --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing `
  --max_train_steps=1200 --learning_rate=1e-4 `
  --lr_scheduler="constant" --lr_warmup_steps=100 `
  --rank=16 --mixed_precision="bf16" --use_8bit_adam `
  --seed=42 --output_dir="loras/ukiyoe" `
  --validation_prompt="a landscape of mount fuji with cherry blossoms, ukyoe_style" `
  --validation_epochs=1 --checkpointing_steps=300 --report_to="tensorboard"
```

**Van Gogh** (~2.5 hr):

```powershell
accelerate launch scripts/train_text_to_image_lora.py `
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" `
  --train_data_dir=".cache/datasets/van_gogh" `
  --image_column="image" --caption_column="text" `
  --resolution=512 --center_crop --random_flip `
  --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing `
  --max_train_steps=1500 --learning_rate=1e-4 `
  --lr_scheduler="constant" --lr_warmup_steps=100 `
  --rank=16 --mixed_precision="bf16" --use_8bit_adam `
  --seed=42 --output_dir="loras/van_gogh" `
  --validation_prompt="a vase of sunflowers in a yellow room, vngogh_style" `
  --validation_epochs=1 --checkpointing_steps=300 --report_to="tensorboard"
```

If any training run is interrupted (laptop sleeps, power loss, Ctrl+C), append `--resume_from_checkpoint="latest"` to the same command — it picks up from the most recent `checkpoint-*` folder.

To monitor: in a second terminal, `tensorboard --logdir loras/<style>` → open http://localhost:6006.

### 8. Sanity-check each LoRA

```
python scripts/test_lora.py pixel_art
python scripts/test_lora.py ukiyoe
python scripts/test_lora.py van_gogh
```

Each writes `test_{style}.png` — a 2×2 grid of style-appropriate validation prompts.

### 9. Serve

```
python serve.py
```

Wait for `Application startup complete.`, then open:

- **http://localhost:8000/** — service status (JSON)
- **http://localhost:8000/ui** — Gradio interface (upload + style picker + slider)
- **http://localhost:8000/docs** — FastAPI auto-generated Swagger UI

REST endpoints:

```bash
# List styles
curl http://localhost:8000/styles

# Stylize an image
curl -X POST http://localhost:8000/stylize \
  -F "image=@photo.jpg" \
  -F "style=van_gogh" \
  -F "strength=0.65" \
  -F "prompt=a sunlit field of wheat" \
  -o stylized.png
```

---

## Reproduction (Docker)

The Docker image is **serving-only**. Train on the host (steps 1–8 above), then mount `loras/` into the container.

Requires Docker Desktop with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (Linux native, or Windows with WSL2 backend).

```bash
# Build (takes ~10 min the first time — pip installs torch + diffusers stack)
docker compose build

# Run (mounts loras/ read-only; uses GPU)
docker compose up
```

Check it's alive:

```bash
curl http://localhost:8000/health
```

The compose file:
- Mounts `./loras` read-only into the container.
- Persists the HF model cache as a named volume (so SD 1.5 isn't re-downloaded on every `docker compose up`).
- Passes `HF_TOKEN` from the host shell if set.
- Reserves all NVIDIA GPUs (set `count: 1` if you have multiple and want to limit).

---

## Project structure

```
QLORATRAINING/
├── README.md                       # this file
├── SPEC.md                         # design doc / spec
├── requirements.txt                # pip dependencies
├── Dockerfile                      # serving image (CUDA 12.4 + py3.12)
├── docker-compose.yml              # GPU passthrough + volume mounts
├── .gitignore
├── .dockerignore
├── serve.py                        # FastAPI + Gradio entry point
├── configs/
│   └── styles.yaml                 # per-style dataset / trigger / overrides
├── scripts/
│   ├── check_env.py                # verify the venv has everything
│   ├── explore_pixel_art.py        # peek at Nouns dataset structure
│   ├── explore_wikiart.py          # peek at WikiArt label vocabulary
│   ├── prep_pixel_art.py           # subsample Nouns + append trigger
│   ├── prep_wikiart.py             # stream-filter WikiArt → Ukiyo-e + Van Gogh
│   ├── train_text_to_image_lora.py # diffusers' official LoRA script (downloaded)
│   └── test_lora.py                # 4-sample test of a trained LoRA
└── loras/                          # trained adapters (gitignored)
    ├── README.md
    ├── pixel_art/pytorch_lora_weights.safetensors
    ├── ukiyoe/pytorch_lora_weights.safetensors
    └── van_gogh/pytorch_lora_weights.safetensors
```

> All `scripts/*.py` use paths relative to the **current working directory**, not their own location. Run them from the project root (the directory containing `serve.py`), not from inside `scripts/`.

---

## Adding a fourth style

1. Edit [`configs/styles.yaml`](configs/styles.yaml) — add a block with `dataset`, optional `filter`, `trigger`, `caption_template`, and any `train_overrides`.
2. Write `scripts/prep_<name>.py` modeled on [`scripts/prep_pixel_art.py`](scripts/prep_pixel_art.py): subsample → append trigger to captions → save as imagefolder under `.cache/datasets/<name>/`.
3. Train: copy one of the `accelerate launch` commands above, swap `--train_data_dir`, `--output_dir`, `--validation_prompt`.
4. Register it in the `STYLES` dict at the top of [`serve.py`](serve.py).
5. Add it to the `STYLES` dict in [`scripts/test_lora.py`](scripts/test_lora.py) so the sanity-check script works.
6. Restart `serve.py`.

---

## Tech stack

- **Base model:** [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- **LoRA training:** [`diffusers`](https://github.com/huggingface/diffusers) + [`peft`](https://github.com/huggingface/peft) + [`accelerate`](https://github.com/huggingface/accelerate) + [`bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes) (8-bit AdamW)
- **Serving:** [FastAPI](https://fastapi.tiangolo.com/) + [Gradio](https://www.gradio.app/) + [uvicorn](https://www.uvicorn.org/)
- **PyTorch:** 2.7+ with CUDA 12.8 wheels (`--index-url https://download.pytorch.org/whl/cu128`)

---

## License

Code: MIT (or your preferred license — adjust here).

Data and trained weights inherit upstream terms:
- Nouns dataset — CC0 (public domain).
- WikiArt — non-commercial research use only.
- SD 1.5 weights — [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license).
