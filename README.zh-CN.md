# qwen3.c

在消费级 **Intel Arc / UMA 核显**（Vulkan）上跑 Qwen3.5 / 3.6-**35B-A3B MoE**：
引擎、compute shader、权重转换。

## 包含什么

| 模块 | 路径 |
|------|------|
| 引擎 | `c/qwen.c` — GDN + 门控 GQA + 256×top-8 MoE |
| GPU | `c/backend_vulkan.c` — 专家 pin、块融合、decode stream、索引 MoE |
| Shader | `c/shaders/*.comp`（本机编成 `.spv`） |
| 转换 | HF RTN int4 / **GPTQ Int4 → fmt=6**（`c/tools/`） |

验证过的模型：**Qwen3.6-35B-A3B**（HF→RTN）、**Qwen3.5-35B-A3B-GPTQ-Int4**（fmt=6）。

## 转换 Qwen3.5-35B-A3B-GPTQ-Int4（推荐）

GPTQ 检查点里**路由专家已是 int4**；脚本会检测到 `quantization_config.quant_method=gptq`，
把专家**无损重打包**为引擎的 **fmt=6**（不做 dequant→requant）。attn / shared_expert /
embed / lm_head 等仍按 float 写入容器。

依赖：`pip install safetensors numpy huggingface_hub`

### 1) 下载 HF 检查点

```bash
# 可选国内镜像
set HF_ENDPOINT=https://hf-mirror.com

python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-35B-A3B-GPTQ-Int4', allow_patterns=['*.safetensors','*.json','*.txt','tokenizer*'], max_workers=8))"
```

记下打印出的本地目录，下面记为 `<HF_GPTQ_DIR>`（内含 `config.json`、分片 safetensors、tokenizer）。

### 2) 转成 qwen3.c 容器

在仓库根目录执行：

```bash
python c/tools/convert_qwen35_moe.py ^
    --model <HF_GPTQ_DIR> ^
    --out   <SNAP>
```

成功时会有类似：

```text
GPTQ int4 checkpoint: group_size=128, ... -> repacking experts verbatim into fmt=6
```

`<SNAP>` 里应有：`config.json`、`out-dense.safetensors`、`out-layer-*.safetensors`、
以及 tokenizer（`tokenizer.json` 等）。**不要**对 GPTQ 目录再加 `--expert-bits 4`——
那是普通 HF 的 RTN 路径；GPTQ 由脚本自动识别。

磁盘大约 **25–27 GB**；建议放在本地 SSD。首次推理前没有 `.coli_usage`，第二跑才会填满专家 tier。

普通 HF（非 GPTQ）权重仍可用同一脚本加 `--expert-bits 4` 做 RTN int4。

## 快速开始（Windows + MinGW + Vulkan）

```bat
:: 1) 编译
scripts\build_win.bat

:: 2) 转换 GPTQ 权重（推荐）
python c\tools\convert_qwen35_moe.py --model <HF_GPTQ_DIR> --out <SNAP>

:: 3) 运行（命令行参数，不必再 set 环境变量）
cd c
qwen_vk.exe --snap <SNAP> --shaders shaders\qmatmul.spv --stream --moe-ix --prompt "what is openvino" --ngen 128 8
```

PowerShell：`.\scripts\build_win.ps1`  
Linux / MSYS：`cd c && make qwen VK=1`  
帮助：`qwen_vk.exe --help`

## 推荐配置（iGPU 高性能路径）

Intel Arc 系 UMA 核显、GPTQ-35B 容器、**已有** `.coli_usage`（第二次及以后，专家
tier 已填满）时，建议用下面配置。

先编译出 `c\qwen_vk.exe` 与 `c\shaders\*.spv`，再在 `c` 目录下执行：

```bat
cd /d <qwen3.c>\c
qwen_vk.exe --snap <你的GPTQ容器目录> ^
  --shaders shaders\qmatmul.spv --stream --moe-ix --experts 7284 ^
  --dense-bits 4 --lmhead-bits 4 ^
  --prompt "你好" --ngen 33 --chat 8
```

| 参数 | 作用 |
|------|------|
| `--moe-ix` | 描述符索引 MoE + 双 CB 异步（decode 增益最大） |
| `--stream` | decode residual stream + device top-k（默认开，建议显式写出） |
| `--experts 7284` | 尽量 pin 满热专家，decode 接近 100% tier-served |
| `--dense-bits 4` | 注意力 / GDN 投影按 int4 分组量化（默认 int8） |
| `--lmhead-bits 4` | lm_head 按 int4 分组量化（默认 int8） |
| `--ngen 33` | 测速用短 decode 窗口（32 步） |
| 末尾 `8` | 每层 CPU LRU 深度（与 VRAM pin 数量无关） |

### dense 量化位宽

decode 时 dense 权重是单 token 读取量最大的一块：40 层投影约 1.4 GB，lm_head
（2048×248320）自己就有 0.5 GB，合计比 top-8 专家还多。改成分组非对称 int4
（fmt=6，默认 group size 128）几乎把这一路的带宽减半，VRAM 也从 1.88 GB 降到
0.98 GB，腾出的空间可以多 pin 专家。

Intel Arc B390 + Qwen3.5-35B-GPTQ 实测（`--experts 7284`，其余参数同上）：

| 配置 | tok/s | dense VRAM |
|------|-------|-----------|
| 默认（都是 int8） | 33.1 | 1.88 GB |
| `--lmhead-bits 4` | 35.2 | 1.62 GB |
| `--dense-bits 4` | 37.4 | 1.24 GB |
| 两个都开 | **40.5** | 0.98 GB |

输出质量没有观察到退化（同一 prompt 下贪心解码结果与 int8 基本逐 token 一致，
偶尔因浮点结合律不同而分叉）。默认仍是 int8，需要显式开启。

日志应出现：`moe_ix ping-pong CBs ENABLED`、`moe_ix fused into route submit`，
以及类似 `decode 32 tok in …s (~32 tok/s)` 的汇总。

设备预算不够时下调 `--experts`（如 `1024`）或提高 `--reserve-gb`——
专家回落到 CPU 后速率会明显下降。

## 命令行速查

| 参数 | 含义 |
|------|------|
| `--snap DIR` | 模型容器（必填） |
| `--prompt TEXT` | 用户提示 |
| `--shaders PATH` | SPIR-V 路径 |
| `--stream` / `--moe-ix` | decode stream / 索引 MoE |
| `--experts N` | VRAM 常驻热专家上限 |
| `--reserve-gb F` | 设备侧预留 GB（默认 3） |
| `--dense-bits N` | 投影权重 8 或 4 bit（默认 8） |
| `--lmhead-bits N` | lm_head 权重 8 或 4 bit（默认 8） |
| `--dense-gs N` | int4 分组大小（默认 128） |
| `--ngen N` | 最大生成 token 数 |

所有开关仅通过命令行 / 内置默认值配置（不再读环境变量）。
`qwen_vk` 默认开 Vulkan（`--no-vulkan` 关闭）。

## 目录结构

```text
qwen3.c/
├── README.md / README.zh-CN.md
├── LICENSE                 # Apache-2.0
├── NOTICE                  # 版权与第三方致谢
├── c/
│   ├── qwen.c
│   ├── qwen_opts.*         # CLI / 运行时选项（不读 env）
│   ├── backend_vulkan.*
│   ├── st.h / tok.h / …
│   ├── shaders/*.comp
│   ├── tools/
│   └── Makefile
└── scripts/
```

## 优化手段摘要

1. 热度 pin + 预算封顶的专家三层存储  
2. Dense 上 GPU + GDN/GQA / RMSNorm+router 块融合  
3. GPTQ → fmt=6 无损重打包  
4. Decode residual stream + device top-k + eg→route semaphore  
5. 描述符索引 MoE + 双 CB ping-pong（`--moe-ix`）

## 许可证

本项目采用 **Apache License 2.0**，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

**模型权重不适用本仓库的 Apache-2.0。** Qwen / GPTQ 检查点及其转换产物
遵循各发行方协议；本仓库不附带这些权重。
