"""FastAPI + Gradio image-to-image styling service.

Loads SD 1.5 once. Hot-swaps between three trained LoRAs per request.

Endpoints:
  GET  /styles            list available style names
  GET  /health            liveness check
  POST /stylize           multipart form: image + style + (optional knobs) -> PNG
  GET  /                  Gradio UI

Run:
  uvicorn serve:app --host 0.0.0.0 --port 8000
  # or
  python serve.py
"""
from __future__ import annotations

import io
import random
from contextlib import asynccontextmanager
from threading import Lock
from typing import Optional

import gradio as gr
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

STYLES: dict[str, dict[str, str]] = {
    "pixel_art": {
        "lora_dir": "loras/pixel_art",
        "trigger":  "pxlart_style",
        # Auto-attached when the user leaves Extra Prompt blank, so casual users
        # get a strong style activation without having to know what to write.
        "default_prompt": "a pixel art character, low resolution, blocky outlines, limited palette, retro game sprite",
    },
    "ukiyoe": {
        "lora_dir": "loras/ukiyoe",
        "trigger":  "ukyoe_style",
        "default_prompt": "an ukiyo-e woodblock print, flat color zones, traditional japanese art, ink outlines, muted palette",
    },
    "van_gogh": {
        "lora_dir": "loras/van_gogh",
        "trigger":  "vngogh_style",
        "default_prompt": "an oil painting in the style of vincent van gogh, thick visible brushstrokes, impasto, post-impressionism, swirling textures",
    },
}

DEFAULT_NEGATIVE = "blurry, low quality, distorted, watermark, text"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# --------------------------------------------------------------------------- #
# Pipeline (loaded once at startup, shared across requests)
# --------------------------------------------------------------------------- #

PIPE: Optional[StableDiffusionImg2ImgPipeline] = None
PIPE_LOCK = Lock()   # serialize set_adapters + generate; CUDA is single-stream anyway


def load_pipeline() -> None:
    global PIPE
    print(f"[serve] Loading SD 1.5 img2img pipeline on {DEVICE}/{DTYPE} ...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        safety_checker=None,             # disable false-positive blocker
        requires_safety_checker=False,
    ).to(DEVICE)

    for name, cfg in STYLES.items():
        print(f"[serve]   loading LoRA: {name}  <-  {cfg['lora_dir']}")
        pipe.load_lora_weights(cfg["lora_dir"], adapter_name=name)

    PIPE = pipe
    print(f"[serve] Ready. Adapters: {list(STYLES)}")


def stylize(
    image: Image.Image,
    style: str,
    strength: float = 0.6,
    prompt: str = "",
    negative_prompt: str = DEFAULT_NEGATIVE,
    seed: Optional[int] = None,
    steps: int = 25,
    guidance: float = 7.5,
    lora_scale: float = 1.0,
) -> Image.Image:
    """Run img2img with the selected LoRA active. Thread-safe via PIPE_LOCK."""
    if PIPE is None:
        raise RuntimeError("Pipeline not loaded.")
    if style not in STYLES:
        raise ValueError(f"Unknown style {style!r}. Options: {list(STYLES)}")

    trigger        = STYLES[style]["trigger"]
    default_prompt = STYLES[style]["default_prompt"]

    # Always attach the style's default prompt so the LoRA has rich semantic
    # context to anchor to. User's Extra Prompt (subject / mood) goes in front.
    if prompt:
        full_prompt = f"{prompt}, {default_prompt}, {trigger}"
    else:
        full_prompt = f"{default_prompt}, {trigger}"

    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((512, 512))

    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(DEVICE).manual_seed(int(seed))

    with PIPE_LOCK:
        # adapter_weights controls how strongly the LoRA influences the UNet.
        # 1.0 = trained default; 1.5-2.0 helps when img2img inputs are very
        # photo-real and need extra push to actually take on the style.
        PIPE.set_adapters([style], adapter_weights=[lora_scale])
        out = PIPE(
            prompt=full_prompt,
            image=image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            negative_prompt=negative_prompt,
            generator=generator,
        ).images[0]

    return out


# --------------------------------------------------------------------------- #
# FastAPI
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_pipeline()
    yield
    # nothing to clean up: PIPE lives until process exit


app = FastAPI(title="LoRA Image Stylizer", lifespan=lifespan)


@app.get("/styles")
def get_styles():
    return {"styles": list(STYLES)}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "base_loaded": PIPE is not None,
        "adapters": list(STYLES) if PIPE is not None else [],
        "device": DEVICE,
    }


@app.post("/stylize")
async def post_stylize(
    image: UploadFile = File(...),
    style: str = Form(...),
    strength: float = Form(0.6),
    prompt: str = Form(""),
    negative_prompt: str = Form(DEFAULT_NEGATIVE),
    seed: Optional[int] = Form(None),
    steps: int = Form(25),
    guidance: float = Form(7.5),
    lora_scale: float = Form(1.0),
):
    if style not in STYLES:
        raise HTTPException(400, f"Unknown style {style!r}. Options: {list(STYLES)}")
    if not 0.0 < strength <= 1.0:
        raise HTTPException(400, "strength must be in (0, 1]")
    if not 0.0 <= lora_scale <= 3.0:
        raise HTTPException(400, "lora_scale must be in [0, 3]")

    img_bytes = await image.read()
    try:
        pil = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        raise HTTPException(400, f"Couldn't decode image: {e}")

    out = stylize(pil, style, strength, prompt, negative_prompt,
                  seed, steps, guidance, lora_scale)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# --------------------------------------------------------------------------- #
# Gradio UI (mounted on the same FastAPI app)
# --------------------------------------------------------------------------- #

def gradio_fn(image, style, strength, lora_scale, prompt, seed_str):
    if image is None:
        raise gr.Error("Please upload an image first.")
    seed_val = int(seed_str) if seed_str and seed_str.strip() else None
    return stylize(
        image, style, float(strength), prompt or "",
        seed=seed_val, lora_scale=float(lora_scale),
    )


with gr.Blocks(title="LoRA Image Stylizer") as ui:
    gr.Markdown(
        "# Three-Style LoRA Image Stylizer\n"
        "Upload a photo, pick a style, get a stylized version. "
        "Three trained LoRAs hot-swap on the same SD 1.5 base.\n\n"
        "Each style auto-attaches its own style-cue prompt under the hood — "
        "you only need to describe **what's in the image** (the subject), "
        "not the artistic style. Leave Extra Prompt blank for a quick try.\n\n"
        "**Knobs:**\n"
        "- **Strength** 0.7–0.85 for dramatic transfer; 0.4–0.6 to keep the input recognizable.\n"
        "- **LoRA weight** 1.5 = balanced default. Push to 2.0+ for stubborn photo inputs; drop to 1.0 if features warp.\n"
        "- **Extra prompt** just describes the subject (e.g. `the eiffel tower at sunset`); the style words are added automatically."
    )
    with gr.Row():
        with gr.Column():
            inp_img    = gr.Image(type="pil", label="Input image")
            style_dd   = gr.Dropdown(choices=list(STYLES), value="van_gogh", label="Style")
            strength_s = gr.Slider(0.10, 0.95, value=0.75, step=0.05,
                                   label="Style strength (img2img denoising)")
            lora_s     = gr.Slider(0.0, 2.5, value=1.5, step=0.1,
                                   label="LoRA weight (how strongly the style takes over)")
            prompt_tb  = gr.Textbox(label="Extra prompt (optional — describes the subject)",
                                    placeholder="e.g. eiffel tower at sunset / a red sports car / a portrait of a woman")
            seed_tb    = gr.Textbox(label="Seed (optional integer)",
                                    placeholder="leave blank for random")
            btn        = gr.Button("Stylize", variant="primary")
        with gr.Column():
            out_img    = gr.Image(type="pil", label="Stylized output")

    btn.click(
        gradio_fn,
        inputs=[inp_img, style_dd, strength_s, lora_s, prompt_tb, seed_tb],
        outputs=out_img,
    )

@app.get("/")
def root():
    return {
        "service": "LoRA Image Stylizer",
        "ui":      "/ui",
        "docs":    "/docs",
        "endpoints": {
            "GET /styles":   "list available styles",
            "GET /health":   "service liveness",
            "POST /stylize": "image + style + knobs -> PNG",
        },
    }


app = gr.mount_gradio_app(app, ui, path="/ui")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
