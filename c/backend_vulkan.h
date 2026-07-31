/*
 * Copyright 2026 qwen3.c contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * See the top-level NOTICE file for third-party attribution.
 */

#ifndef COLIBRI_BACKEND_VULKAN_H
#define COLIBRI_BACKEND_VULKAN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque persistent device copy of one resident quantized tensor,
 * mirroring backend_cuda.h. On Strix Halo the "upload" writes into
 * HOST_VISIBLE|DEVICE_LOCAL memory — same physical RAM the iGPU reads,
 * so there is no PCIe copy, unlike the discrete-CUDA path. */
typedef struct ColiVkTensor ColiVkTensor;

/* Bring up instance/device/queue/pipeline. Returns 1 on success.
 * spv_path points at the compiled qmatmul.spv. */
int  coli_vk_init(const char *spv_path);
void coli_vk_shutdown(void);
int  coli_vk_available(void);
void coli_vk_mem_info(size_t *used_bytes, size_t *tensor_count);

/* VRAM pressure-proofing (both no-ops when the extension is absent):
 * alloc_priority sets the eviction-priority class of SUBSEQUENT weight uploads
 * (VK_EXT_memory_priority; scratches and the KV mirror pin themselves at 1.0) —
 * the engine brackets the bulk expert-tier fill at 0.4 so an oversubscribed heap
 * evicts cold experts, never the per-token attention working set.
 * mem_budget reports device-local usage/budget in GB (VK_EXT_memory_budget);
 * returns 0 if unavailable. */
void coli_vk_alloc_priority(float p);
int  coli_vk_mem_budget(double *used_gb, double *budget_gb);

/* y[S,O] = (x[S,I] @ dequant(W[O,I])^T) * scale[O].
 * fmt matches QT in glm.c: 1=int8, 2=int4. (0=f32,3=int2 fall back to CPU.)
 * First call uploads W+scales; later calls reuse the resident copy.
 * Returns 1 on success, 0 if unavailable / unsupported fmt. */
int  coli_vk_matmul(ColiVkTensor **tensor,
                    float *y, const float *x,
                    const void *weights, const float *scales,
                    int fmt, int S, int I, int O, int gs);

/* Same projection for a single row (S=1), reduced to its greedy winner ON THE DEVICE:
 * the lm_head logits stay in GPU memory and only (index, value) is read back, so a
 * vocab-sized vector never crosses the bus and the host never scans it. Ties resolve to
 * the lowest index, matching an ascending host scan. val is NULLable.
 * Returns 0 (argmax.spv missing / device gone) -> caller falls back to coli_vk_matmul. */
int  coli_vk_matmul_argmax(ColiVkTensor **tensor,
                           const void *weights, const float *scales,
                           int fmt, int I, int O, int gs,
                           const float *x, int *idx, float *val);

/* Fused first half of the expert MLP in ONE dispatch (VK equivalent of
 * grouped_hidden_w4_dual): hidden[s,o] = silu(gate(x)) * up(x), reading x once for both
 * projections. D = input (hidden) dim, I = moe_inter. gate/up upload on first call.
 * Returns 0 if unavailable (no gate_up shader) / unsupported fmt so the caller falls back. */
int  coli_vk_gate_up(ColiVkTensor **gate, ColiVkTensor **up,
                     float *hidden, const float *x,
                     const void *gw, const float *gs,
                     const void *uw, const float *us,
                     int fmt, int S, int D, int I, int grp);

/* Full batched expert MLP for `count` experts in ONE submit, hidden staying on-device:
 * for each c, y_c = down_c(silu(gate_c(x_c)) * up_c(x_c)). x/y packed [sum(rows)*D];
 * experts are resident (gate/up: D->I, down: I->D). Mirrors coli_cuda_expert_group.
 * Returns 0 -> caller falls back to CPU. */
int  coli_vk_expert_group(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                          ColiVkTensor *const *downs, const int *rows, int count,
                          float *y, const float *x);
/* Async form: _issue submits the group and returns immediately (one in flight max);
 * the caller computes its CPU share, then _take joins and reads back the packed y.
 * Both return 0 on failure (caller computes those experts on the CPU instead). */
int  coli_vk_expert_group_issue(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                                ColiVkTensor *const *downs, const int *rows, int count,
                                const float *x);
int  coli_vk_expert_group_take(float *y);

/* Upload a resident tensor without computing (expert tier: gate/up/down uploaded once,
 * then driven by coli_vk_expert_group). Returns 0 on failure/unsupported fmt. */
int  coli_vk_tensor_ensure(ColiVkTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int grp);

/* MLA absorb attention core (decode). The KV latent/rope caches live in persistent
 * per-layer device buffers: _ensure allocates a layer's cache at max_rows (once; resize
 * via _reset), _row mirrors one host row (absolute position), _reset drops all layers.
 * The caller keeps a valid-watermark and re-mirrors rows after any invalidation.
 * absorb runs S causal query rows over cache rows [st0, T) in one submit:
 * q [S,H*(Q+R)] roped, kv_b [H*(Q+V), K] uploads once (fmt 1=int8/2=int4),
 * ctx out [S,H*V]. Returns 0 -> caller falls back to CPU. */
int  coli_vk_kv_ensure(int layer, int max_rows, int K, int Rd);
int  coli_vk_kv_row(int layer, int pos, const float *L, const float *R);
void coli_vk_kv_reset(void);
int  coli_vk_attention_absorb(ColiVkTensor **kvb, const void *w, const float *sc, int fmt, int grp,
                              float *ctx, const float *q, int layer, int S, int H,
                              int Q, int R, int V, int K, int st0, int T, float scale);
/* Two resident matmuls sharing one input x in ONE submit (q_a + kv_a prologue pair).
 * Returns 0 -> caller falls back to single-matmul calls. */
/* q-prep chain: [q_a+kv_a pair] -> rmsnorm(q latent) -> q_b in ONE submit (needs
 * rmsnorm.spv next to the main shader; returns 0 without it -> 3-submit path).
 * lnw = the q-latent RMS-norm weights [Oqa], resident per layer after first call. */
int  coli_vk_attn_qprep(int layer,
                        ColiVkTensor **qa,  const void *wqa,  const float *sqa,  int Oqa,
                        ColiVkTensor **kva, const void *wkva, const float *skva, int Okva,
                        ColiVkTensor **qb,  const void *wqb,  const float *sqb,  int Oqb,
                        int fmt, int grp, const float *lnw, float eps,
                        const float *x, int S, int I, float *q_out, float *kv_out,
                        float *lat_out /* normed q latent [S,Oqa], NULLable — DSA indexer input */);
int  coli_vk_matmul_pair(ColiVkTensor **t1p, float *y1, const void *w1, const float *s1, int O1,
                         ColiVkTensor **t2p, float *y2, const void *w2, const float *s2, int O2,
                         int fmt, const float *x, int S, int I, int grp);

/* Fused variant: absorb + resident o-projection ([Dout, H*V]) in one submit; ctx stays
 * on-device, only out [S,Dout] is read back. Falls back like absorb (returns 0). */
int  coli_vk_attention_absorb_project(ColiVkTensor **kvb, const void *w, const float *sc, int fmt, int grp,
                              ColiVkTensor **ot, const void *ow, const float *osc, int ofmt, int ogrp,
                              float *out, const float *q, int layer, int S, int H,
                              int Q, int R, int V, int K, int st0, int T, float scale, int Dout);

void   coli_vk_tensor_free(ColiVkTensor *t);
size_t coli_vk_tensor_bytes(const ColiVkTensor *t);

/* ---- GatedDeltaNet delta rule (Qwen hybrid linear-attention layers) ----
 * coli_vk_gdn_state_ensure allocates+zeros the device-resident per-layer recurrent
 * state (once); coli_vk_gdn_reset zeros all states in place at each new sequence;
 * coli_vk_gdn runs one decode step (recurrence + readout + gated RMSNorm) on the GPU.
 * All return 0 -> caller falls back to the CPU path. */
int  coli_vk_gdn_available(void);
int  coli_vk_gdn_state_ensure(int layer, int VH, int KD, int VD);
void coli_vk_gdn_reset(void);
int  coli_vk_gdn(int layer, const float *qn, const float *kn, const float *v, const float *z,
                 const float *decay, const float *beta, const float *gnorm, float *y,
                 int KH, int KD, int VH, int VD, float eps);

/* ---- Standard GQA decode attention (Qwen full-attention layers) ----
 * Reuses the device KV mirror (coli_vk_kv_ensure(layer, KH*max_t, hd, hd) then
 * coli_vk_kv_row(layer, kh*max_t+pos, k_row, v_row)). q is [S,H,2*hd] (q|gate per
 * head); ctx [S,H,hd] read back pre o-projection. Returns 0 -> caller falls back. */
int  coli_vk_gqa_available(void);
int  coli_vk_gqa(int layer, const float *q, float *ctx, int S, int H, int KH, int hd,
                 int max_t, int st0, int T, float scale);
/* Fused GQA + o-projection (ctx stays on device); `ot` is the resident o_proj tensor
 * (H*hd -> Dout). out[S,Dout] read back. Returns 0 -> caller falls back. */
int  coli_vk_gqa_project(int layer, const float *q, ColiVkTensor *ot, float *out,
                         int S, int H, int KH, int hd, int max_t, int st0, int T,
                         float scale, int Dout);
/* Whole GQA block for S rows in ONE submit: q/k/v matmul -> q/k-norm+rope+KV-write ->
 * attention -> o-proj. The device KV mirror is written on-device (becomes canonical);
 * needs coli_vk_kv_ensure + coli_vk_gqa_norm_weight first. out[S,Dout] read back. */
int  coli_vk_gqa_full_available(void);
int  coli_vk_gqa_norm_weight(int layer, const float *qnw, const float *knw, int hd);
int  coli_vk_gqa_full(int layer, const float *x, int D, ColiVkTensor *gq_t, ColiVkTensor *gk_t,
                      ColiVkTensor *gv_t, ColiVkTensor *out_t, float *out, int S, int H, int KH,
                      int hd, int rot, int pos_base, int max_t, float eps, float theta, int Dout);

/* ---- MoE prep fused into the attn/GDN full submit (Qwen decode) ----
 * coli_vk_layer_norm_weight uploads per-model-layer in/post RMSNorm weights once.
 * *_full_route runs in_ln (GQA) or uses a host-provided post-in_ln x (GDN) + the
 * existing full block + residual+=delta + post_ln + router matmul in ONE submit.
 * Read back: resid_out [S,D], nrm_out [S,D] (expert input), pr_out [S,E] (router logits).
 * Returns 0 -> caller keeps the staged CPU rmsnorm/router path. */
int  coli_vk_moe_route_available(void);
int  coli_vk_layer_norm_weight(int layer, const float *in_ln, const float *post_ln, int D);
int  coli_vk_gqa_full_route(int layer, const float *resid, int D,
                            ColiVkTensor *gq_t, ColiVkTensor *gk_t, ColiVkTensor *gv_t, ColiVkTensor *out_t,
                            ColiVkTensor *router, int ln_layer,
                            float *resid_out, float *nrm_out, float *pr_out,
                            int S, int H, int KH, int hd, int rot, int pos_base, int max_t,
                            float eps, float theta, int Dout, int E);
int  coli_vk_gdn_full_route(int layer, const float *x_in, const float *resid, int D,
                            ColiVkTensor *gqkv_t, ColiVkTensor *gz_t, const float *params,
                            ColiVkTensor *out_t, ColiVkTensor *router, int ln_layer,
                            float *resid_out, float *nrm_out, float *pr_out,
                            int KH, int KD, int VH, int VD, int conv_dim, int conv_k,
                            float eps, int Dout, int E);

/* ---- Decode residual stream (P0/P1/P2) ----
 * Resid + post-norm stay on device across layers. Route runs softmax+top-k on
 * device and only reads back idx[K]/val[K] (not pr[E]); expert_group replicates
 * device nrm, accumulates into the stream, and signals a GPU semaphore so the
 * next layer's route can wait without a host eg fence. GDN path also folds
 * in_ln + b/a + decay/beta pack onto the GPU. */
int  coli_vk_stream_available(void);
int  coli_vk_stream_begin(const float *x, int D);
/* Drain in-flight work. If x!=NULL copy resid back; if x==NULL keep resid on device
 * for coli_vk_stream_norm_argmax (Phase 1: skip host final-norm round-trip). */
int  coli_vk_stream_end(float *x, int D);
int  coli_vk_stream_add(const float *y, int D);          /* host CPU-expert delta into stream */
const float *coli_vk_stream_nrm(void);                   /* last route's nrm (host-cached) */
int  coli_vk_final_norm_weight(const float *w, int D);
/* stream_end(NULL) then this: final RMSNorm + lm_head + argmax, only idx read back. */
int  coli_vk_stream_norm_argmax(ColiVkTensor **tensor, const void *weights, const float *scales,
                                int fmt, int I, int O, int gs, float eps, int *idx, float *val);
/* Copy post-stream_end resid from the device-mapped stream buffer (fallback path). */
int  coli_vk_stream_copy_resid(float *x, int D);
void coli_vk_route_cache_stats(void);
int  coli_vk_gdn_ba_weight(int layer, const float *alog, const float *dtb,
                           const float *gnorm, int VH, int VD);
int  coli_vk_gqa_full_route_pipe(int layer, int D,
                                 ColiVkTensor *gq_t, ColiVkTensor *gk_t, ColiVkTensor *gv_t, ColiVkTensor *out_t,
                                 ColiVkTensor *router, int ln_layer,
                                 int *idx_out, float *val_out, int topk,
                                 int H, int KH, int hd, int rot, int pos_base, int max_t,
                                 float eps, float theta, int Dout, int E);
int  coli_vk_gdn_full_route_pipe(int layer, int D,
                                 ColiVkTensor *gqkv_t, ColiVkTensor *gz_t,
                                 ColiVkTensor *gba_t, ColiVkTensor *gbb_t,
                                 ColiVkTensor *out_t, ColiVkTensor *router, int ln_layer,
                                 int *idx_out, float *val_out, int topk,
                                 int KH, int KD, int VH, int VD, int conv_dim, int conv_k,
                                 float eps, int Dout, int E);
int  coli_vk_expert_group_issue_pipe(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                                     ColiVkTensor *const *downs, const int *rows,
                                     const float *weights, int count, int D);
int  coli_vk_expert_group_take_pipe(void);
/* Device-indexed MoE (COLI_VK_MOE_IX=1): tier experts in UPDATE_AFTER_BIND
 * descriptor arrays; optional fuse into route_pipe (fmt=6 + shared table slot). */
int  coli_vk_moe_ix_available(void);
int  coli_vk_eg_table_init(int n_layers, int E, int D, int I, int gs);
int  coli_vk_eg_table_set(int layer, int eid, ColiVkTensor *g, ColiVkTensor *u, ColiVkTensor *d);
int  coli_vk_eg_table_set_shared(int layer, ColiVkTensor *g, ColiVkTensor *u, ColiVkTensor *d,
                                 const float *sh_gate, int D);
int  coli_vk_stream_moe_fused(void);
/* Async moe_ix ping-pong (COLI_VK_MOE_IX_PP, default on with MOE_IX=1): top-k idx
 * is drained into hist at stream_end; apply eusage from coli_vk_stream_ix_hist. */
int  coli_vk_stream_ix_pending(void);
int  coli_vk_stream_ix_hist(int layer, int *idx_out, int K);
/* COLI_VK_EG_CACHE=0 disables eg cmd/descriptor resubmit cache (default on).
 * COLI_VK_EG_STATS=1 prints hit/miss once at first stream_end. */

/* Fused GatedDeltaNet + out-projection (y stays on device); `ot` is the resident
 * out_proj tensor (VH*VD -> Dout). out[Dout] read back. Returns 0 -> caller falls back. */
int  coli_vk_gdn_project(int layer, const float *qn, const float *kn, const float *v, const float *z,
                         const float *decay, const float *beta, const float *gnorm,
                         ColiVkTensor *ot, float *out, int KH, int KD, int VH, int VD,
                         float eps, int Dout);
/* Whole GDN block for one token in ONE submit: gqkv/gz matmul -> conv1d(+ring) ->
 * qknorm+delta+gatednorm -> out-proj, all on device. params packs [decay(VH)|beta(VH)|
 * gnorm(VD)]. Needs coli_vk_gdn_state_ensure + coli_vk_gdn_conv_ensure/_weight first. */
int  coli_vk_gdn_full_available(void);
int  coli_vk_gdn_conv_ensure(int layer, int conv_k, int conv_dim);
int  coli_vk_gdn_conv_weight(int layer, const float *cw, int conv_k, int conv_dim);
int  coli_vk_gdn_full(int layer, const float *x, int D, ColiVkTensor *gqkv_t, ColiVkTensor *gz_t,
                      const float *params, ColiVkTensor *out_t, float *out,
                      int KH, int KD, int VH, int VD, int conv_dim, int conv_k, float eps, int Dout);

#ifdef __cplusplus
}
#endif

#endif
