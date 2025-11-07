
# pbo_min/config.py
# =================
# Edit these variables directly; no CLI parsing needed.

from pathlib import Path

# === PATHS (EDIT THESE) ===
SESSION_DIR     = Path("/Users/ruonansun/Desktop/Refinement/sessions/[seq]_A_cozy_and_retro_coffeeshop_2025-08-14_10-50-07")
# Use one reference image from the session as the structural anchor
REF_IMAGE_PATH  = SESSION_DIR / "spatial" / "spatial_1_0.png"
TAGS_JSON       = SESSION_DIR / "tags.json"
CLUSTERS_JSON   = SESSION_DIR / "clusters.json"
OUT_DIR         = SESSION_DIR / "outputs"

# === DIFFUSION SETTINGS ===
MODEL_ID        = "stabilityai/stable-diffusion-xl-base-1.0"
# Use prompt_embeds (SDXL) instead of long strings to avoid truncation.
# Requires an SDXL pipeline (has tokenizer_1/2 and text_encoder_1/2).
USE_PROMPT_EMBEDS = True

GUIDANCE_SCALE  = 8
STEPS_DRAFT     = 20       # faster drafts
STEPS_FINAL     = 38
HEIGHT          = 768
WIDTH           = 768

# === IMG2IMG (OPTIONAL, NO CONTROLNET) ===
# Use the reference image as init image without ControlNet
USE_IMG2IMG            = True
IMG2IMG_STRENGTH_DRAFT = 0.45   # 0.25–0.35 recommended for drafts
IMG2IMG_STRENGTH_FINAL = 0.45   # slightly lower for finals to keep structure
INIT_RESIZE_MODE       = "fit_center_crop"  # ["fit_center_crop", "stretch", "letterbox"]

# === CONTROLNET (OPTIONAL) ===
# If diffusers and ControlNet weights are available, enable to anchor structure/style from reference
USE_CONTROLNET        = False
CONTROLNET_MODEL_ID   = "diffusers/controlnet-depth-sdxl-1.0"  # change to a valid SDXL ControlNet depth model id you have
CONTROL_METHOD        = "depth"  # "canny" | "edges" | "depth"
CONTROLNET_SCALE      = 0.35      # lower to avoid outline hugging
CONTROL_START         = 0.0       # apply from the start
CONTROL_END           = 0.5       # stop halfway to let model relax

# === PBO LOOP SETTINGS ===
SEED_PER_ROUND  = 1234     # same seed inside a round
K_PER_ROUND     = 4        # 4 candidates per round
ROUNDS_A        = 2        # reduced for quick test
ROUNDS_B        = 2        # reduced for quick test
TOP_GROUPS_B    = 2        # how many groups to refine in Stage B

# === PROMPT BUILDER SETTINGS ===
MAX_PHRASES_PER_GROUP = 3
EMPHASIS_RANGE        = (0.8, 1.6)   # token strength mapping
NEGATIVE_FRACTION     = 0.3
NEUTRAL_INJECT_PROB   = 0.2

# === UI / TESTING ===
NON_INTERACTIVE       = False         # auto-pick a candidate index each round
NON_INTERACTIVE_CHOICE= 0            # pick first candidate by default

# Ensure output dir exists
OUT_DIR.mkdir(parents=True, exist_ok=True)



