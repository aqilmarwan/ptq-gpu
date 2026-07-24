#!/usr/bin/env bash
#
# Grab a ready-made SDXL LoRA into place as inference/loras/neon-atlas.safetensors,
# so the LoRA variants can be built WITHOUT training (use --skip-train). The
# serving path fuses the LoRA into the engine, so a pre-trained one works as-is.
#
#   ./pipelines/fetch_lora.sh                       # default cyberpunk LoRA
#   ./pipelines/fetch_lora.sh <hf-repo> <filename>  # any diffusers LoRA repo
set -euo pipefail

REPO="${1:-issaccyj/lora-sdxl-cyberpunk}"
FILE="${2:-pytorch_lora_weights.safetensors}"
DEST="inference/loras/neon-atlas.safetensors"

echo ">> $REPO/$FILE -> $DEST"
curl -fL "https://huggingface.co/${REPO}/resolve/main/${FILE}" -o "$DEST"
echo ">> done ($(du -h "$DEST" | cut -f1)). Build the LoRA variants with --skip-train."
