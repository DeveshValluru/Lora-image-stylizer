"""Generate 4 sample images from a trained LoRA, save as a 2x2 grid.

Usage:
    python test_lora.py pixel_art
    python test_lora.py ukiyoe
    python test_lora.py van_gogh
"""
import sys

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

STYLES = {
    "pixel_art": {
        "trigger": "pxlart_style",
        "prompts": [
            "a pixel art character with a wizard hat, {trigger}",
            "a pixel art character with a crown, {trigger}",
            "a pixel art character with headphones and sunglasses, {trigger}",
            "a pixel art character with a chef hat, {trigger}",
        ],
    },
    "ukiyoe": {
        "trigger": "ukyoe_style",
        "prompts": [
            "a landscape of mount fuji with cherry blossoms, {trigger}",
            "a great wave at sea with boats, {trigger}",
            "a kabuki actor in a dramatic pose, {trigger}",
            "a geisha walking through a bamboo forest, {trigger}",
        ],
    },
    "van_gogh": {
        "trigger": "vngogh_style",
        "prompts": [
            "a vase of sunflowers, {trigger}",
            "a starry night sky over a small village, {trigger}",
            "a wheat field with crows, {trigger}",
            "a self portrait of a man with a red beard, {trigger}",
        ],
    },
}

style = sys.argv[1] if len(sys.argv) > 1 else "pixel_art"
if style not in STYLES:
    raise SystemExit(f"Unknown style {style!r}. Pick one of: {list(STYLES)}")

cfg     = STYLES[style]
prompts = [p.format(trigger=cfg["trigger"]) for p in cfg["prompts"]]
LORA_DIR = f"loras/{style}"
OUT      = f"test_{style}.png"

print("Loading SD 1.5 pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,                    # disable false-positive blocker
    requires_safety_checker=False,
).to("cuda")

print(f"Loading LoRA from {LORA_DIR} ...")
pipe.load_lora_weights(LORA_DIR)

print(f"Generating 4 images for style '{style}' (seed=42)...")
images = pipe(
    prompts,
    num_inference_steps=25,
    guidance_scale=7.5,
    generator=torch.Generator("cuda").manual_seed(42),
).images

# 2x2 grid
w, h = images[0].size
grid = Image.new("RGB", (2 * w, 2 * h))
grid.paste(images[0], (0, 0))
grid.paste(images[1], (w, 0))
grid.paste(images[2], (0, h))
grid.paste(images[3], (w, h))
grid.save(OUT)
print(f"Saved → {OUT}  ({grid.size[0]}x{grid.size[1]})")
