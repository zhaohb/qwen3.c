# qwen3.c

Qwen3.5 / 3.6-**35B-A3B MoE** inference on consumer **Intel Arc / UMA iGPUs**
(Vulkan): engine, compute shaders, and weight converters.

## What you get

| Area | Contents |
|------|----------|
| Engine | `c/qwen.c` — GatedDeltaNet + gated GQA + 256×top-8 MoE |
| GPU | `c/backend_vulkan.c` — expert tier, fused blocks, decode stream, indexed MoE |
| Shaders | `c/shaders/*.comp` (compile to `.spv` locally) |
| Convert | HF RTN int4 / **GPTQ Int4 → fmt=6** (`c/tools/`) |

Validated models: **Qwen3.6-35B-A3B** (HF→RTN), **Qwen3.5-35B-A3B-GPTQ-Int4** (fmt=6).

Measured decode (Intel Arc B390, GPTQ + stream + `--moe-ix`): **~32 tok/s** (decode-only).

## Convert Qwen3.5-35B-A3B-GPTQ-Int4 (recommended)

Routed experts in the GPTQ checkpoint are already int4. The converter detects
`quantization_config.quant_method=gptq` and **losslessly repacks** them to engine
**fmt=6** (no dequant→requant). Attn / shared_expert / embed / lm_head stay float
in the container.

Deps: `pip install safetensors numpy huggingface_hub`

### 1) Download the HF checkpoint

```bash
# optional mirror
set HF_ENDPOINT=https://hf-mirror.com

python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-35B-A3B-GPTQ-Int4', allow_patterns=['*.safetensors','*.json','*.txt','tokenizer*'], max_workers=8))"
```

Use the printed path as `<HF_GPTQ_DIR>` (`config.json`, safetensors shards, tokenizer).

### 2) Convert to a qwen3.c container

From the repo root:

```bash
python c/tools/convert_qwen35_moe.py --model <HF_GPTQ_DIR> --out <SNAP>
```

You should see something like:

```text
GPTQ int4 checkpoint: group_size=128, ... -> repacking experts verbatim into fmt=6
```

`<SNAP>` must contain `config.json`, `out-dense.safetensors`, `out-layer-*.safetensors`,
and the tokenizer files. Do **not** pass `--expert-bits 4` on a GPTQ tree — that flag is
for plain HF RTN; GPTQ is auto-detected.

Container size is roughly **25–27 GB** (local SSD recommended). The first generate run
seeds `.coli_usage`; the second run fills the expert VRAM tier.

Plain HF (non-GPTQ) weights: same script with `--expert-bits 4`.

## Quick start (Windows + MinGW + Vulkan)

```bat
:: 1) Build
scripts\build_win.bat

:: 2) Convert GPTQ checkpoint (recommended)
python c\tools\convert_qwen35_moe.py --model <HF_GPTQ_DIR> --out <SNAP>

:: 3) Run (CLI flags; no env required)
cd c
qwen_vk.exe --snap <SNAP> --shaders shaders\qmatmul.spv --stream --moe-ix --prompt "what is openvino" --ngen 128 8
```

PowerShell build: `.\scripts\build_win.ps1`.  
Linux / MSYS: `cd c && make qwen VK=1`.  
Help: `qwen_vk.exe --help`.

## Recommended config (iGPU perf path)

Build `c\qwen_vk.exe` and `c\shaders\*.spv` first, then from `c`:

```bat
cd /d <qwen3.c>\c
qwen_vk.exe --snap <path-to-your-GPTQ-container> ^
  --shaders shaders\qmatmul.spv --stream --moe-ix --experts 7284 ^
  --prompt "你好" --ngen 33 --chat 8
```

| Flag | Why |
|------|-----|
| `--moe-ix` | Descriptor-indexed MoE + dual-CB async (largest decode gain) |
| `--stream` | Decode residual stream + device top-k (default on; set explicitly) |
| `--experts 7284` | Pin ~all hot experts so decode stays tier-served (~100%) |
| `--ngen 33` | Short decode window for tok/s measurement (32 decode steps) |
| trailing `8` | Per-layer CPU LRU depth (independent of VRAM pin count) |

Expect log lines: `moe_ix ping-pong CBs ENABLED`, `moe_ix fused into route submit`,
and a summary like `decode 32 tok in …s (~32 tok/s)`.

If device budget is tight, lower `--experts` (e.g. `1024`) or raise
`--reserve-gb` — rate drops when experts miss to CPU.

## CLI cheat sheet

| Flag | Meaning |
|------|---------|
| `--snap DIR` | Model container (required) |
| `--prompt TEXT` | User prompt |
| `--shaders PATH` | SPIR-V path |
| `--stream` / `--moe-ix` | Decode stream / indexed MoE |
| `--experts N` | Max hot experts pinned in VRAM |
| `--reserve-gb F` | Keep this much device budget free (default 3) |
| `--ngen N` | Max new tokens |

All knobs are CLI / defaults only (no env-var fallback).
`qwen_vk` enables Vulkan by default (`--no-vulkan` to disable).

## Layout

```text
qwen3.c/
├── README.md / README.zh-CN.md
├── LICENSE                 # Apache-2.0
├── NOTICE                  # copyright + third-party attribution
├── c/
│   ├── qwen.c              # engine
│   ├── qwen_opts.*         # CLI / runtime options (no env)
│   ├── backend_vulkan.*    # Vulkan backend
│   ├── st.h / tok.h / …
│   ├── shaders/*.comp
│   ├── tools/              # convert + tiny oracle
│   └── Makefile
└── scripts/                # build_win.*, dl_qwen.sh
```

## License

This project is licensed under the **Apache License, Version 2.0**.
See [LICENSE](LICENSE) and [NOTICE](NOTICE).

**Model weights are not licensed under Apache-2.0.** Qwen / GPTQ checkpoints
(and containers converted from them) remain under their publishers' terms;
this repository does not redistribute those weights.
