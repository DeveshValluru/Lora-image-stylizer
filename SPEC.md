# Project Spec — Three-Style LoRA Image Service

## Goal
Local image-to-image style transfer with three swappable styles, served via FastAPI + Gradio on an RTX 4060 Laptop (8 GB). Zero API costs, all weights/datasets pulled from Hugging Face.

## Naming note
Folder is `QLORATRAINING/`, but we landed on **plain LoRA** (no quantization). SD 1.5 fits comfortably in bf16; "QLoRA" would just slow training for no VRAM benefit. We may revisit QLoRA in a future v2 with SDXL.

---

## Scope

### In
- Train 3 LoRA adapters on SD 1.5 from public HF datasets.
- Serve them via one FastAPI process; hot-swap LoRAs per request.
- Gradio UI mounted on the same app.
- Local-only inference (no cloud, no API keys beyond a one-time HF login).

### Out (deferred)
- SDXL / SD3.5 / FLUX base models (future v2).
- Text-to-image-only flows (text2img will work as a side effect but img2img is the headline).
- ControlNet structure preservation (try plain img2img first; add only if needed).
- Multi-user auth, rate limiting, cloud deploy.

---

## Architecture

### Training (offline, run 3×)
- One `train_lora.py` script, `--style {pixel_art|ukiyoe|van_gogh}` selects the run.
- Reads `configs/styles.yaml` for dataset + trigger + per-style overrides.
- Output: `loras/{style}.safetensors` (~30–50 MB each).

### Inference (online, FastAPI)
- Load SD 1.5 pipeline once at startup.
- Load all three LoRAs via `pipeline.load_lora_weights(..., adapter_name=...)`.
- Per request: `pipeline.set_adapters([style])` → img2img → return PNG.
- Gradio interface mounted at `/`.

---

## Models & datasets

| | Source | Notes |
|---|---|---|
| **Base** | `stable-diffusion-v1-5/stable-diffusion-v1-5` | HF mirror of the original Runway SD 1.5 weights |
| **Style 1: Pixel Art (Nouns)** | `m1guelpf/nouns` | 49,859 pixel-art characters with auto-generated captions (CC0). Subsample to ~2,000 for training. |
| **Style 2: Ukiyo-e** | `huggan/wikiart`, filtered `style == "Ukiyo_e"` | Verify exact label spelling in data prep |
| **Style 3: Van Gogh** | `huggan/wikiart`, filtered `artist == "vincent-van-gogh"` | Largest of the three (~800+ images) |

Triggers (rare tokens, unlikely to collide with anything in the base vocab):
- `pxlart_style`
- `ukyoe_style`
- `vngogh_style`

Captions: Nouns ships captions in the `text` column (zero prep — auto-generated from the characters' on-chain attributes). For the WikiArt subsets, try `fusing/wikiart_captions` first; fallback to BLIP-2 auto-caption for any rows missing captions. Captions get the trigger appended via template in `styles.yaml`.

---

## Training config (defaults per LoRA)

| Param | Value |
|---|---|
| Base model | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Resolution | 512 × 512 |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| LoRA target modules | UNet attention: `to_q`, `to_k`, `to_v`, `to_out.0` |
| Optimizer | 8-bit AdamW (bitsandbytes) |
| Learning rate | 1.0e-4 |
| LR scheduler | constant w/ 100-step warmup |
| Per-device batch size | 2 |
| Gradient accumulation | 4 (effective batch = 8) |
| Mixed precision | bf16 |
| Gradient checkpointing | on |
| Max train steps | 1500 (overridable per style) |
| Random horizontal flip | yes |
| Caption dropout | 0.05 (for classifier-free guidance) |
| Validation | 4 fixed prompts every 250 steps, sample images saved |
| Final save | `loras/{style}.safetensors` |

Per-style overrides live in `configs/styles.yaml`.

---

## Inference API

### `GET /styles`
```json
{ "styles": ["pixel_art", "ukiyoe", "van_gogh"] }
```

### `POST /stylize` — multipart form
| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `image` | file (png/jpg) | yes | — | input photo |
| `style` | string | yes | — | one of the three style names |
| `strength` | float 0–1 | no | 0.6 | img2img denoising strength |
| `prompt` | string | no | `""` | extra text appended after the trigger |
| `negative_prompt` | string | no | sensible default | |
| `seed` | int | no | random | reproducibility |
| `steps` | int | no | 25 | diffusion steps |
| `guidance` | float | no | 7.5 | CFG scale |

Returns: `image/png`.

### `GET /health`
```json
{ "status": "ok", "base_loaded": true, "adapters": ["pixel_art","ukiyoe","van_gogh"] }
```

### `GET /`
Gradio UI: file upload → style dropdown → strength slider → generate.

---

## Hardware budget

| Phase | VRAM peak | Time on 4060 |
|---|---|---|
| Train per LoRA | ~5–6 GB | 30–60 min |
| Train all three | sequential | ~2 hr total |
| Inference (one request) | ~3 GB | 3–5 sec @ 512², 25 steps |

---

## Deliverables

```
QLORATRAINING/
├── SPEC.md                                  # this file
├── requirements.txt
├── configs/
│   └── styles.yaml                          # per-style dataset/trigger/overrides
├── data_prep.py                             # load + filter + caption HF datasets, cache locally
├── train_lora.py --style {name}             # trains one LoRA from styles.yaml
├── serve.py                                 # FastAPI + Gradio service
└── loras/
    ├── pixel_art.safetensors
    ├── ukiyoe.safetensors
    └── van_gogh.safetensors
```

---

## Acceptance criteria

1. All three LoRAs train end-to-end on 4060 8GB without OOM.
2. `GET /styles` returns the three style names.
3. `POST /stylize` produces a styled image for each of the three styles.
4. Same input image through all three styles yields visibly distinct outputs.
5. Gradio UI at `http://localhost:8000/` works for upload → generate.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| `huggan/wikiart` label naming may differ from assumption (`Ukiyo_e` vs `Ukiyo-e` etc.) | Verify in `data_prep.py` before training; fail loudly with available label list if mismatched. |
| BLIP-2 captions too generic → LoRA learns subject not style | Inspect 10 captions per dataset before training; tweak template if needed. |
| Nouns 50K imgs is way more than needed; full-dataset training is wasteful | Subsample to ~2,000 (configurable in `styles.yaml`). Validate every 250; pick best-step adapter. |
| WikiArt subset for Ukiyo-e may be small (~100 imgs) | If <80 images, fall back to broader filter (e.g., `genre == "ukiyo-e"` or add Hokusai/Hiroshige artist filter). |
| First-run dataset download + BLIP-2 captioning is slow | One-time cost; cache to `./.cache/`. |

---

## Open questions (defer until after first run)

- Add `POST /stylize_text2img` for pure text-to-image with style? (~5 extra lines)
- Use ControlNet (Canny/Depth) for stronger structure preservation when style strength is high?
- Expose multi-LoRA blending (`set_adapters([a,b], [w_a, w_b])`) as an endpoint?
