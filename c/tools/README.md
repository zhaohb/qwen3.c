# Tools

| Script | Purpose |
|--------|---------|
| `convert_qwen35_moe.py` | HF Qwen3.5/3.6-MoE → container (RTN int4/int8, or GPTQ→fmt=6 lossless repack) |
| `make_tiny_qwen.py` | Tiny synthetic MoE for logits oracle vs HuggingFace |

```bash
pip install safetensors numpy
# optional for make_tiny_qwen.py / HF download:
# pip install torch transformers huggingface-hub

python convert_qwen35_moe.py --model /path/to/Qwen3.5-35B-A3B-GPTQ-Int4 --out /path/to/SNAP
```

See the repo root `README.md` / `README.zh-CN.md` for download and convert steps.
