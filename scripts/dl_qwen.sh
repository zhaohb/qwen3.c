#!/usr/bin/env bash
set -e
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
DPY=/c/Users/Win11/.local/share/mamba/envs/deskmate/python
echo "[dl] snapshot_download Qwen/Qwen3.6-35B-A3B ..."
SNAP=$("$DPY" - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3.6-35B-A3B", allow_patterns=["*.safetensors","*.json","*.txt","tokenizer*"], max_workers=8)
print(p)
PY
)
echo "[dl] snapshot at: $SNAP"
echo "[cv] converting -> qwen35_i4 (int4 experts) ..."
"$DPY" c/tools/convert_qwen35_moe.py --model "$SNAP" --out "$(pwd)/qwen35_i4" --expert-bits 4
echo "[done] container at $(pwd)/qwen35_i4"
