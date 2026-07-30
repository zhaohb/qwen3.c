#!/usr/bin/env python3
# Copyright 2026 qwen3.c contributors
# SPDX-License-Identifier: Apache-2.0
# See the top-level NOTICE file for third-party attribution.
"""Convert a HF Qwen3.5/3.6-MoE checkpoint (model_type qwen3_5_moe, e.g.
Qwen/Qwen3.6-35B-A3B) into the qwen3.c engine container.

Layout written to --out (consumed by c/qwen.c via st.h):
  config.json                   flattened text_config the C engine reads
  out-dense.safetensors         embedding + final norm (float32)
  out-layer-NN.safetensors      per-layer dense weights (float32) + routed
                                experts quantized to int4 (fmt 2):
      model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight      U8 nibble-packed
      model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight.qs   F32 per-row scales

The int4 nibble packing matches colibri's fmt=2 decode (backend_vulkan.c /
qmatmul.comp): element j of a row lives in nibble (j&7) of uint32 word j>>3,
value = nibble - 8, weight ~= value * qs[row]. So the SAME tensors feed the
CPU dequant path and coli_vk_tensor_ensure(fmt=2) for the Vulkan expert tier.

Ignored checkpoint keys: mtp.* (multi-token prediction head), model.visual.*
(vision tower) — text-only decode.

Usage:
  python convert_qwen35_moe.py --model /path/to/hf_snapshot --out /path/to/container
  python convert_qwen35_moe.py --model ... --out ... --expert-bits 8   # int8 experts (fmt 1)
"""
import argparse, json, os, sys, glob
import numpy as np

try:
    from safetensors import safe_open
    from safetensors.numpy import save_file
except ImportError:
    sys.exit("pip install safetensors")


def bf16_to_f32(a):
    """safetensors returns bfloat16 as ml_dtypes/torch types numpy can't always take;
    view-cast through uint16 -> uint32 works for any provider."""
    if a.dtype == np.float32:
        return a
    if a.dtype == np.float16 or a.dtype == np.float64:
        return a.astype(np.float32)
    # bfloat16: upcast by left-shifting into the high half of a float32
    b = a.view(np.uint16).astype(np.uint32) << 16
    return b.view(np.float32)


class Shards:
    """Reads tensors across the HF shard set without loading whole files."""

    def __init__(self, model_dir):
        self.dir = model_dir
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        self.map = None
        if os.path.exists(idx):
            with open(idx, "r", encoding="utf-8") as f:
                self.map = json.load(f)["weight_map"]
        self.files = {}   # path -> handle kept open

    def _handle(self, fname):
        if fname not in self.files:
            self.files[fname] = safe_open(os.path.join(self.dir, fname), framework="np")
        return self.files[fname]

    def _file_of(self, name):
        if self.map is not None:
            f = self.map.get(name)
            if f:
                return f
            return None
        # single-file checkpoints
        for p in glob.glob(os.path.join(self.dir, "*.safetensors")):
            h = self._handle(os.path.basename(p))
            if name in h.keys():
                return os.path.basename(p)
        return None

    def get_raw(self, name, required=True):
        """Tensor with its on-disk dtype (int32 GPTQ packing must not be float-cast)."""
        f = self._file_of(name)
        if f is None:
            if required:
                sys.exit(f"missing tensor: {name}")
            return None
        return np.asarray(self._handle(f).get_tensor(name))

    def get(self, name, required=True):
        f = self._file_of(name)
        if f is None:
            if required:
                sys.exit(f"missing tensor: {name}")
            return None
        try:
            t = self._handle(f).get_tensor(name)
            return bf16_to_f32(np.asarray(t))
        except TypeError:
            # some safetensors builds refuse bfloat16 under framework="np":
            # fall back to torch for this tensor only
            import torch
            from safetensors.torch import safe_open as topen
            with topen(os.path.join(self.dir, f), framework="pt") as h:
                return h.get_tensor(name).to(torch.float32).numpy()


def pack_int4(w):
    """Symmetric per-row int4 (fmt 2): q in [-7,7], stored nibble = q+8."""
    amax = np.abs(w).max(axis=1)
    s = np.maximum(amax / 7.0, 1e-8).astype(np.float32)
    q = np.clip(np.rint(w / s[:, None]), -7, 7).astype(np.int8) + 8
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed, s


def quant_int8(w):
    """Symmetric per-row int8 (fmt 1)."""
    amax = np.abs(w).max(axis=1)
    s = np.maximum(amax / 127.0, 1e-8).astype(np.float32)
    q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)
    return q, s


def pack_int4_grouped_asym(w, gs):
    """Grouped ASYMMETRIC int4 (colibri fmt=6, GPTQ-style storage): one scale+zero per
    gs-input group, q in [0,15], nibble packing identical to fmt=2 (element j -> low/high
    nibble of byte j>>1). Decode: w ~= (nibble - zero_g) * scale_g. Returns
    (packed U8 [O,(I+1)//2], sz F32 [O, ng*2] interleaved [scale_g, zero_g])."""
    O, I = w.shape
    ng = (I + gs - 1) // gs
    q = np.zeros((O, I), dtype=np.int32)
    sz = np.zeros((O, ng, 2), dtype=np.float32)
    for g in range(ng):
        i0, i1 = g * gs, min((g + 1) * gs, I)
        blk = w[:, i0:i1]
        wmin = blk.min(axis=1); wmax = blk.max(axis=1)
        scale = np.maximum((wmax - wmin) / 15.0, 1e-8).astype(np.float32)
        zero = (-wmin / scale).astype(np.float32)
        q[:, i0:i1] = np.clip(np.rint(blk / scale[:, None] + zero[:, None]), 0, 15).astype(np.int32)
        sz[:, g, 0] = scale; sz[:, g, 1] = zero
    if I % 2:                                   # pad to even for nibble packing
        q = np.concatenate([q, np.zeros((O, 1), dtype=np.int32)], axis=1)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed, sz.reshape(O, ng * 2)


def repack_gptq_int4(qweight, qzeros, scales):
    """GPTQ 4-bit (desc_act=False) -> colibri fmt=6, with no dequant/requant round trip:
    both formats store gs-grouped nibbles in [0,15] plus a per-group (scale, zero) and
    decode as (nibble - zero) * scale, so the nibbles transfer verbatim and only the
    packing axes change. Inputs are the HF GPTQ tensors:
      qweight [I/8, O] int32 — element i of column o at nibble (i%8) of word i//8
      qzeros  [ng, O/8] int32 — zero of (group g, output o) at nibble (o%8)
      scales  [ng, O] f16
    Returns (packed U8 [O, I/2], sz F32 [O, ng*2] interleaved [scale_g, zero_g])."""
    Iw, O = qweight.shape
    I, ng = Iw * 8, scales.shape[0]
    qw = qweight.view(np.uint32) if qweight.dtype == np.int32 else qweight.astype(np.uint32)
    nib = np.stack([(qw >> (4 * j)) & 0xF for j in range(8)], axis=1).reshape(I, O)
    q = np.ascontiguousarray(nib.T).astype(np.uint8)             # [O, I], row-major over inputs
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    zt = qzeros.view(np.uint32) if qzeros.dtype == np.int32 else qzeros.astype(np.uint32)
    zn = np.stack([(zt >> (4 * j)) & 0xF for j in range(8)], axis=2).reshape(ng, O)
    sz = np.stack([scales.astype(np.float32).T, zn.T.astype(np.float32)], axis=-1)
    return packed, np.ascontiguousarray(sz.reshape(O, ng * 2), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF snapshot dir (config.json + safetensors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expert-bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--group-size", type=int, default=0,
                    help="int4 group size for GROUPED-ASYMMETRIC experts (fmt=6, GPTQ-style); "
                         "must be a multiple of 8. 0 (default) = per-row RTN (fmt=2/1).")
    args = ap.parse_args()
    if args.group_size and args.group_size % 8:
        sys.exit("--group-size must be a multiple of 8")

    with open(os.path.join(args.model, "config.json"), "r", encoding="utf-8") as f:
        hf = json.load(f)
    tc = hf.get("text_config", hf)   # ForConditionalGeneration nests it; ForCausalLM doesn't

    # Already-GPTQ checkpoints (e.g. Qwen3.5-35B-A3B-GPTQ-Int4) carry int4 experts in
    # exactly colibri's fmt=6 shape, so adopt their group size and repack rather than
    # dequantizing to float and re-rounding.
    qc = hf.get("quantization_config") or {}
    gptq = qc.get("quant_method") == "gptq"
    if gptq:
        if qc.get("bits") != 4:
            sys.exit(f"only 4-bit GPTQ is supported (checkpoint has bits={qc.get('bits')})")
        if qc.get("desc_act"):
            sys.exit("GPTQ desc_act=True reorders inputs per row; unsupported")
        args.group_size = qc["group_size"]
        print(f"GPTQ int4 checkpoint: group_size={args.group_size}, sym={qc.get('sym')} "
              f"-> repacking experts verbatim into fmt=6")
    if tc.get("model_type") not in ("qwen3_5_moe_text", "qwen3_5_moe", "qwen3_next"):
        print(f"warning: model_type={tc.get('model_type')} — expected qwen3_5_moe_text", file=sys.stderr)

    nl = tc["num_hidden_layers"]
    layer_types = tc.get("layer_types")
    if layer_types is None:
        fai = tc.get("full_attention_interval", 4)
        layer_types = ["full_attention" if (i + 1) % fai == 0 else "linear_attention" for i in range(nl)]
    rope = tc.get("rope_parameters") or {}
    cfg = {
        "model_type": "qwen35_moe_colibri",
        "hidden_size": tc["hidden_size"],
        "num_hidden_layers": nl,
        "layer_types": layer_types,
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc["head_dim"],
        "partial_rotary_factor": rope.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25)),
        "rope_theta": rope.get("rope_theta", tc.get("rope_theta", 1e7)),
        "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
        "linear_conv_kernel_dim": tc["linear_conv_kernel_dim"],
        "linear_num_key_heads": tc["linear_num_key_heads"],
        "linear_key_head_dim": tc["linear_key_head_dim"],
        "linear_num_value_heads": tc["linear_num_value_heads"],
        "linear_value_head_dim": tc["linear_value_head_dim"],
        "num_experts": tc["num_experts"],
        "num_experts_per_tok": tc["num_experts_per_tok"],
        "moe_intermediate_size": tc["moe_intermediate_size"],
        "shared_expert_intermediate_size": tc["shared_expert_intermediate_size"],
        "vocab_size": tc["vocab_size"],
        "expert_bits": 4 if args.group_size else args.expert_bits,
        "expert_fmt": 6 if args.group_size else (2 if args.expert_bits == 4 else 1),
        "group_size": args.group_size,
        "bos_token_id": tc.get("bos_token_id"),
        "eos_token_id": tc.get("eos_token_id"),
    }

    S = Shards(args.model)
    # ForConditionalGeneration prefix vs plain ForCausalLM prefix
    prefix = "model.language_model." if S._file_of("model.language_model.embed_tokens.weight") else "model."
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)

    # Tokenizer files the C engine (tok.h) needs for text prompts. Prefer the
    # HF snapshot's tokenizer.json (byte-level BPE, cl100k family — same as GLM).
    import shutil
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "special_tokens_map.json", "vocab.json", "merges.txt"):
        src = os.path.join(args.model, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, name))
            print(f"copied {name}")
    if not os.path.isfile(os.path.join(args.out, "tokenizer.json")):
        print("warning: no tokenizer.json in snapshot — text PROMPT mode will need one",
              file=sys.stderr)

    quant = (lambda w: pack_int4_grouped_asym(w, args.group_size)) if args.group_size \
            else (pack_int4 if args.expert_bits == 4 else quant_int8)

    # dense file: embedding + final norm (+ lm_head only if untied)
    dense = {
        "model.embed_tokens.weight": S.get(prefix + "embed_tokens.weight"),
        "model.norm.weight": S.get(prefix + "norm.weight"),
    }
    lm = S.get("lm_head.weight", required=False)
    if lm is not None:
        dense["lm_head.weight"] = lm    # untied checkpoint; engine falls back to embed otherwise
    save_file(dense, os.path.join(args.out, "out-dense.safetensors"))
    print(f"dense: embed {dense['model.embed_tokens.weight'].shape}, lm_head {'tied' if lm is None else 'separate'}")

    E = cfg["num_experts"]
    for n in range(nl):
        hfp = f"{prefix}layers.{n}."
        op = f"model.layers.{n}."
        t = {}

        def put(dst, src, required=True, reshape=None):
            a = S.get(hfp + src, required=required)
            if a is None:
                return
            if reshape is not None:
                a = a.reshape(reshape)
            t[op + dst] = np.ascontiguousarray(a, dtype=np.float32)

        put("input_layernorm.weight", "input_layernorm.weight")
        put("post_attention_layernorm.weight", "post_attention_layernorm.weight")
        if layer_types[n] == "full_attention":
            for w in ("q_proj", "k_proj", "v_proj", "o_proj"):
                put(f"self_attn.{w}.weight", f"self_attn.{w}.weight")
            put("self_attn.q_norm.weight", "self_attn.q_norm.weight")
            put("self_attn.k_norm.weight", "self_attn.k_norm.weight")
        else:
            for w in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                put(f"linear_attn.{w}.weight", f"linear_attn.{w}.weight")
            conv = S.get(hfp + "linear_attn.conv1d.weight")           # [conv_dim, 1, K]
            t[op + "linear_attn.conv1d.weight"] = np.ascontiguousarray(
                conv.reshape(conv.shape[0], conv.shape[-1]), dtype=np.float32)
            put("linear_attn.dt_bias", "linear_attn.dt_bias")
            put("linear_attn.A_log", "linear_attn.A_log")
            put("linear_attn.norm.weight", "linear_attn.norm.weight")

        put("mlp.gate.weight", "mlp.gate.weight")
        for w in ("gate_proj", "up_proj", "down_proj"):
            put(f"mlp.shared_expert.{w}.weight", f"mlp.shared_expert.{w}.weight")
        put("mlp.shared_expert_gate.weight", "mlp.shared_expert_gate.weight")

        # routed experts. Hub checkpoints pack them 3D (gate_up [E, 2*I, D] with
        # gate rows first, down [E, D, I]); save_pretrained may instead emit
        # per-expert Linear keys (experts.E.gate_proj.weight). Support both.
        I = cfg["moe_intermediate_size"]
        if gptq:
            # GPTQ stores each projection pre-quantized; repack the nibbles as-is.
            def expert_q(e):
                for w in ("gate_proj", "up_proj", "down_proj"):
                    p = f"{hfp}mlp.experts.{e}.{w}."
                    yield w, repack_gptq_int4(S.get_raw(p + "qweight"), S.get_raw(p + "qzeros"),
                                              S.get_raw(p + "scales"))
        else:
            gu = S.get(hfp + "mlp.experts.gate_up_proj", required=False)
            if gu is not None:
                dn = S.get(hfp + "mlp.experts.down_proj")
                assert gu.shape[1] == 2 * I, f"gate_up dim {gu.shape} vs I={I}"
                expert = lambda e: (("gate_proj", gu[e, :I, :]), ("up_proj", gu[e, I:, :]), ("down_proj", dn[e]))
            else:
                expert = lambda e: tuple(
                    (w, S.get(f"{hfp}mlp.experts.{e}.{w}.weight")) for w in ("gate_proj", "up_proj", "down_proj"))

            def expert_q(e):
                for name, w in expert(e):
                    yield name, quant(np.ascontiguousarray(w, dtype=np.float32))
        for e in range(E):
            for name, (q, s) in expert_q(e):
                t[f"{op}mlp.experts.{e}.{name}.weight"] = q
                t[f"{op}mlp.experts.{e}.{name}.weight.qs"] = s

        save_file(t, os.path.join(args.out, f"out-layer-{n:02d}.safetensors"))
        print(f"layer {n:2d} [{layer_types[n]:16s}] written ({len(t)} tensors)")

    print("done:", args.out)


if __name__ == "__main__":
    main()
