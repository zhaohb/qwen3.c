#!/usr/bin/env python3
"""Build a tiny random Qwen3.5-MoE text model + greedy reference, for validating
c/qwen.c against HF transformers (needs transformers >= 5.x with qwen3_5_moe).

Writes:
  <out>/hf/           HF checkpoint (safetensors f32 + config.json)
  <out>/ref.json      {"prompt_ids": [...], "full_ids": [...]}
  <out>/ref_logits.bin  [n_new, vocab] f32 — step logits along the greedy path
Then convert with tools/convert_qwen35_moe.py --model <out>/hf --out <out>/container
and run: SNAP=<out>/container LOGITS_OUT=got.bin qwen <cap> <out>/ref.json
"""
import argparse, json, os, sys
import numpy as np
import torch

from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--np", type=int, default=12, help="prompt length")
    ap.add_argument("--new", type=int, default=8, help="greedy tokens to generate")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = Qwen3_5MoeTextConfig(
        hidden_size=64,
        num_hidden_layers=4,          # linear,linear,linear,full (full_attention_interval=4)
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,                  # rot = 32*0.25 = 8
        intermediate_size=96,         # unused (every layer is MoE) but required
        num_experts=8,
        num_experts_per_tok=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=48,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_num_value_heads=4,
        linear_value_head_dim=16,
        vocab_size=512,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    model = Qwen3_5MoeForCausalLM(cfg).eval()
    with torch.no_grad():           # default init leaves norms at exactly 0/1; randomize everything
        for name, p in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.uniform_(-0.2, 0.2)
            elif name.endswith(("A_log", "dt_bias")):
                p.uniform_(-1.0, 1.0)
            else:
                p.normal_(0.0, 0.08)
    model = model.float()

    os.makedirs(args.out, exist_ok=True)
    hfdir = os.path.join(args.out, "hf")
    model.save_pretrained(hfdir, safe_serialization=True)
    print("saved tiny HF model to", hfdir)

    rng = np.random.default_rng(args.seed)
    prompt = rng.integers(0, cfg.vocab_size, size=args.np).tolist()
    ids = list(prompt)
    logits_steps = []
    with torch.no_grad():
        for _ in range(args.new):   # re-forward the whole sequence each step (tiny model)
            out = model(input_ids=torch.tensor([ids]), use_cache=False)
            lg = out.logits[0, -1].float().numpy()
            logits_steps.append(lg.copy())
            ids.append(int(lg.argmax()))
    with open(os.path.join(args.out, "ref.json"), "w") as f:
        json.dump({"prompt_ids": prompt, "full_ids": ids}, f)
    np.stack(logits_steps).astype(np.float32).tofile(os.path.join(args.out, "ref_logits.bin"))
    print("prompt:", prompt)
    print("greedy:", ids[len(prompt):])
    print("wrote ref.json + ref_logits.bin", np.stack(logits_steps).shape)


if __name__ == "__main__":
    main()
