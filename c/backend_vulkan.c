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

// Vulkan compute backend for quantized matmul on UMA iGPUs (e.g. Intel Arc).
// Strix Halo iGPU (RADV gfx1151). Mirrors backend_cuda.c's contract but
// exploits unified memory: weight "uploads" write into HOST_VISIBLE|
// DEVICE_LOCAL memory — the same physical RAM the iGPU reads — so there is
// no PCIe copy. That is what makes offloading *streamed experts* profitable
// here, which the discrete-CUDA path deliberately avoids.
//
// M2 scope: correctness + a standalone GPU-vs-CPU test harness. Synchronous
// submit/wait per call; async queues and zero-copy import come in M4.
#include "backend_vulkan.h"
#include "qwen_opts.h"
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
static double vk_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1000.0 + t.tv_nsec/1e6; }

#define VKCHECK(x, what) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "[VK] %s failed: %d\n", what, _r); return 0; } } while (0)

/* --vk-prof timestamps. A route CB carries DP_TS_N of them (entry, then one per
 * stage) so the ~0.5 ms per layer splits into attention / route tail / MoE. Two
 * groups ping-pong with the CBs, one covers the sync route CB, and a final pair
 * brackets the lm_head + argmax submit. */
#define DP_TS_N 4
#define DP_TSQ_AM (3 * DP_TS_N)

struct ColiVkTensor {
    VkBuffer wbuf, sbuf;
    VkDeviceMemory wmem, smem;
    size_t wbytes;
    int fmt, I, O, rowWords, gs;
};

typedef struct {
    VkBuffer buf; VkDeviceMemory mem; void *ptr; size_t cap;
} Scratch;

/* Persistent device-side KV latent/rope cache for one layer (MLA attention).
 * Host appends rows as tokens decode (absolute-position indexing); the absorb
 * kernel reads them in place. Allocated once at max_t rows, like the CUDA
 * kv_dev shadow. */
#define VK_KV_LAYERS 160
typedef struct {
    VkBuffer bl, br; VkDeviceMemory ml, mr; void *pl, *pr;
    int rows, K, R;
} VkKvLayer;

static struct {
    int ready;
    VkInstance inst;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfam;
    uint32_t memtype;            // HOST_VISIBLE|HOST_COHERENT (prefer DEVICE_LOCAL) — for inputs/weights
    uint32_t memtype_cached;     // HOST_CACHED — for buffers the CPU reads back (outputs)
    VkDescriptorSetLayout dsl;
    VkPipelineLayout plyt;
    VkPipeline pipe;
    VkShaderModule shader;
    VkDescriptorPool dpool;
    VkDescriptorSet dset;
    /* fused dual gate+up+silu pipeline (6 bindings): x, Wg, gscale, Wu, uscale, hidden */
    VkShaderModule shader_gu; VkDescriptorSetLayout dsl_gu; VkPipelineLayout plyt_gu;
    VkPipeline pipe_gu; VkDescriptorPool dpool_gu; VkDescriptorSet dset_gu;
    /* DP4A expert path (fmt=6). quant_rows int8-izes an activation per row — needed twice:
     * for the incoming x, and again for the hidden, whose amax spans outputs that the
     * gate_up phase splits across workgroups. gate_up then takes 7 bindings (the extra
     * per-row scale keeps it off plyt_gu) and down takes 5. */
    VkShaderModule shader_qr; VkDescriptorSetLayout dsl_qr; VkPipelineLayout plyt_qr;
    VkPipeline pipe_qr; VkDescriptorPool dpool_qr; VkDescriptorSet dset_qr_h, dset_qr_x;
    VkShaderModule shader_gu_dp4a; VkDescriptorSetLayout dsl_gu4; VkPipelineLayout plyt_gu4;
    VkPipeline pipe_gu_dp4a; VkDescriptorPool dpool_gu4; VkDescriptorSet dset_gu4;
    VkShaderModule shader_dn_dp4a; VkDescriptorSetLayout dsl_dn4; VkPipelineLayout plyt_dn4;
    VkPipeline pipe_dn_dp4a; VkDescriptorPool dpool_dn4; VkDescriptorSet dset_dn4;
    VkDescriptorSet eg_gu4[64], eg_dn4[64];
    Scratch eg_xq, eg_xs, eg_hq, eg_hs;   /* int8 activation/hidden + their per-row scales */
    /* MLA absorb attention core (7 bindings): q, W, scales, Lcache, Rcache, scores, ctx */
    VkShaderModule shader_att; VkDescriptorSetLayout dsl_att; VkPipelineLayout plyt_att;
    VkPipeline pipe_att; VkDescriptorPool dpool_att; VkDescriptorSet dset_att;
    /* FlashMLA-style chunked absorb (8 bindings, 3 push-constant modes): q, W,
     * scales, Lcache, Rcache, qabs scratch, chunk partials, ctx. Optional like
     * the other extra shaders — absent file keeps the per-head kernel. */
    VkShaderModule shader_attf; VkDescriptorSetLayout dsl_attf; VkPipelineLayout plyt_attf;
    VkPipeline pipe_attf; VkDescriptorPool dpool_attf; VkDescriptorSet dset_attf;
    Scratch att_qabs;            /* flash: absorbed queries [S,H,K] (GPU-only) */
    Scratch att_part;            /* flash: chunk partials [S,H,nc,K+2] (GPU-only) */
    /* GatedDeltaNet delta rule (3 bindings: packed in | state | y). The per-head
     * recurrent state S[VH*KD*VD] is device-resident per GDN layer across tokens;
     * gdn_in/gdn_y are the per-call packed-input / read-back-y scratches. */
    VkShaderModule shader_gdn; VkDescriptorSetLayout dsl_gdn; VkPipelineLayout plyt_gdn;
    VkPipeline pipe_gdn; VkDescriptorPool dpool_gdn; VkDescriptorSet dset_gdn;
    VkBuffer gdn_st[VK_KV_LAYERS]; VkDeviceMemory gdn_stm[VK_KV_LAYERS]; void *gdn_stp[VK_KV_LAYERS];
    size_t gdn_bytes[VK_KV_LAYERS];
    Scratch gdn_in, gdn_y;
    /* Standard GQA decode attention (Qwen full-attention layers; 4 bindings:
     * q | K-mirror | V-mirror | ctx). Reuses the G.kv[] device KV mirror (K->bl,
     * V->br) and the G.x / G.y (cached) scratches like the absorb path. */
    VkShaderModule shader_gqa; VkDescriptorSetLayout dsl_gqa; VkPipelineLayout plyt_gqa;
    VkPipeline pipe_gqa; VkDescriptorPool dpool_gqa; VkDescriptorSet dset_gqa;
    /* Fused GDN input block (gqkv/gz matmul -> conv1d -> qknorm+delta -> out-proj in ONE
     * submit). conv1d + delta-cv pipelines, device-resident per-layer conv ring + weights,
     * 3 extra qmatmul descriptor sets (gqkv/gz/out bound concurrently), device-only scratches. */
    VkShaderModule shader_gdnconv; VkDescriptorSetLayout dsl_gdnconv; VkPipelineLayout plyt_gdnconv;
    VkPipeline pipe_gdnconv; VkDescriptorPool dpool_gdnconv; VkDescriptorSet dset_gdnconv;
    VkShaderModule shader_gdncv; VkDescriptorSetLayout dsl_gdncv; VkPipelineLayout plyt_gdncv;
    VkPipeline pipe_gdncv; VkDescriptorPool dpool_gdncv; VkDescriptorSet dset_gdncv;
    VkBuffer gdn_ring[VK_KV_LAYERS]; VkDeviceMemory gdn_ringm[VK_KV_LAYERS]; void *gdn_ringp[VK_KV_LAYERS];
    size_t gdn_ring_bytes[VK_KV_LAYERS];
    VkBuffer gdn_cw[VK_KV_LAYERS]; VkDeviceMemory gdn_cwm[VK_KV_LAYERS]; void *gdn_cwp[VK_KV_LAYERS];
    VkDescriptorPool gdnf_pool; VkDescriptorSet gdnf_gqkv, gdnf_gz, gdnf_out;
    Scratch gdnf_qkv, gdnf_z, gdnf_cv, gdnf_pr;
    /* Fused GQA input block (q/k/v matmul -> q/k-norm+rope+KV-write -> attention ->
     * o-proj in one submit). q/k-norm+rope pipeline, device-resident per-layer q/k norm
     * weights, 4 extra qmatmul sets (gq/gk/gv/out), device-only q/k/v/ctx scratch. */
    VkShaderModule shader_qkv; VkDescriptorSetLayout dsl_qkv; VkPipelineLayout plyt_qkv;
    VkPipeline pipe_qkv; VkDescriptorPool dpool_qkv; VkDescriptorSet dset_qkv;
    VkBuffer gqa_qnw[VK_KV_LAYERS]; VkDeviceMemory gqa_qnwm[VK_KV_LAYERS]; void *gqa_qnwp[VK_KV_LAYERS];
    VkBuffer gqa_knw[VK_KV_LAYERS]; VkDeviceMemory gqa_knwm[VK_KV_LAYERS]; void *gqa_knwp[VK_KV_LAYERS];
    VkDescriptorPool gqaf_pool; VkDescriptorSet gqaf_q, gqaf_k, gqaf_v, gqaf_out;
    Scratch gqaf_qg, gqaf_kb, gqaf_vb;
    VkCommandPool cpool;
    VkCommandBuffer cmd;
    VkCommandBuffer cmd_rec; /* recording target: cmd or cmd_pp[slot] */
    VkFence fence;
    Scratch x, y, h;   /* h = fused gate+up hidden output */
    /* full expert-group scratch: activations/hidden/output for K experts + per-expert
     * descriptor sets (gate_up: dsl_gu, down: dsl), so gate_up->down runs on-device in
     * one submit with hidden never leaving the GPU. */
    Scratch eg_x, eg_h, eg_y;
    VkDescriptorPool eg_pool; VkDescriptorSet eg_gu[64], eg_dn[64]; int eg_nsets;
    /* expert-group ASYNC state: its own command buffer + fence so an in-flight group
     * never collides with the main cmd/fence (dense matmuls, absorb) — issue() returns
     * immediately, the CPU computes its share, take() joins. */
    VkCommandBuffer eg_cmd; VkFence eg_fence; int eg_inflight; size_t eg_pending_yb;
    double eg_t0, eg_t1, eg_t2, eg_t3; int eg_prof;
    /* q-prep chain (pair -> rmsnorm -> q_b in ONE submit): norm pipeline (3 bindings),
     * a 3rd matmul set + norm set, GPU-only latent intermediates, per-layer resident
     * norm-weight buffers (tiny, uploaded once like the KV mirror). */
    VkShaderModule shader_nrm; VkDescriptorSetLayout dsl_nrm; VkPipelineLayout plyt_nrm;
    VkPipeline pipe_nrm; VkDescriptorPool qprep_pool; VkDescriptorSet dset_qp3, dset_nrm;
    Scratch qp1, qp2;
    VkBuffer lnbuf[VK_KV_LAYERS]; VkDeviceMemory lnmem[VK_KV_LAYERS]; int lnlen[VK_KV_LAYERS];
    Scratch att_sc;              /* attention score scratch (GPU-only) */
    Scratch att_ctx;             /* fused absorb+o: ctx stays on device (GPU-only) */
    Scratch y2;                  /* second output of the fused matmul pair (readback) */
    VkDescriptorPool pair_pool; VkDescriptorSet dset_pair;   /* 4-binding set for the pair's 2nd matmul */
    /* device-side argmax tail (lm_head): logits stay in am_y, only the winning
     * (index, value) pair comes back — am_pi/am_pv hold the partial winners. */
    VkShaderModule shader_am; VkDescriptorSetLayout dsl_am; VkPipelineLayout plyt_am;
    VkPipeline pipe_am; VkDescriptorPool dpool_am; VkDescriptorSet dset_am;
    Scratch am_y, am_pi, am_pv;
    /* MoE prep fused into the attn/GDN full submit: residual+=delta + rmsnorm_zc +
     * router matmul. in/post LN weights are resident per model layer; route_* are
     * the per-call scratches (resid upload, updated resid, post-norm, attn delta). */
    VkShaderModule shader_nrmz; VkPipeline pipe_nrmz;          /* shares dsl/plyt with rmsnorm */
    VkShaderModule shader_rrnz; VkDescriptorSetLayout dsl_rrnz; VkPipelineLayout plyt_rrnz;
    VkPipeline pipe_rrnz; VkDescriptorPool dpool_rrnz; VkDescriptorSet dset_rrnz;
    VkDescriptorPool route_pool; VkDescriptorSet dset_route_rt; /* router matmul set */
    VkBuffer qln_in[VK_KV_LAYERS]; VkDeviceMemory qln_inm[VK_KV_LAYERS]; void *qln_inp[VK_KV_LAYERS];
    VkBuffer qln_post[VK_KV_LAYERS]; VkDeviceMemory qln_postm[VK_KV_LAYERS]; void *qln_postp[VK_KV_LAYERS];
    int qln_D[VK_KV_LAYERS];
    Scratch route_x, route_xo, route_nrm, att_delta;
    /* Decode residual stream + eg→route GPU chain (P0/P1/P2). stream holds the
     * live residual; route_nrm feeds expert_group without a host memcpy; eg_sem
     * lets the next layer's route wait on the GPU instead of a host eg fence. */
    Scratch stream, moe_w, gdn_ba, gdn_bb, route_idx, route_val;
    VkSemaphore eg_sem; int eg_sem_armed, stream_D, stream_live;
    /* eg pipe resubmit cache (A1/A2): reuse recorded eg_cmd when shape+scratch
     * buffers match; only re-wr_desc weight bindings that changed. Safe because
     * issue_pipe runs only when the previous eg is idle (eg_inflight cleared). */
    int eg_pipe_cmd_ready, eg_pipe_n, eg_pipe_D, eg_pipe_I, eg_pipe_fmt, eg_pipe_dfmt;
    int eg_pipe_dp4a, eg_pipe_dp4a_dn, eg_pipe_grw, eg_pipe_ggs, eg_pipe_drw, eg_pipe_dgs;
    int eg_pipe_cache_on, eg_pipe_hits, eg_pipe_misses;
    /* GDN route CB cache (Phase 1): GDN pipe push-constants are shape-stable across
     * tokens (no pos/T), so after the first record per (pp_slot, ln_layer) we
     * resubmit the same CB. Expert weights stay live via UPDATE_AFTER_BIND. */
    int route_gdn_ready[2][VK_KV_LAYERS];
    int route_gdn_fused[2][VK_KV_LAYERS];
    VkBuffer route_gdn_key_stream, route_gdn_key_x, route_gdn_key_idx;
    int route_cache_hits, route_cache_misses;
    /* Device-resident final RMSNorm weight for stream→norm→lm_head argmax fuse. */
    VkBuffer final_nw; VkDeviceMemory final_nwm; void *final_nwp; int final_nD;
    /* --vk-prof: timestamps bracketing each route CB. Two queries per ping-pong
     * slot plus two for the sync CB; ts_base_rec tracks the pair being recorded. */
    VkQueryPool tsq; double ts_period_ms; int ts_base_rec, pp_tsq[2];
    VkBuffer eg_pipe_xb, eg_pipe_hb, eg_pipe_yb, eg_pipe_wb, eg_pipe_sb, eg_pipe_nb;
    VkBuffer eg_pipe_xqb, eg_pipe_xsb, eg_pipe_hqb, eg_pipe_hsb;
    VkBuffer eg_cache_gw[64], eg_cache_gs[64], eg_cache_uw[64], eg_cache_us[64];
    VkBuffer eg_cache_dw[64], eg_cache_ds[64];
    VkShaderModule shader_rep; VkDescriptorSetLayout dsl_rep; VkPipelineLayout plyt_rep;
    VkPipeline pipe_rep; VkDescriptorPool dpool_rep; VkDescriptorSet dset_rep;
    VkShaderModule shader_macc; VkDescriptorSetLayout dsl_macc; VkPipelineLayout plyt_macc;
    VkPipeline pipe_macc; VkDescriptorPool dpool_macc; VkDescriptorSet dset_macc;
    VkShaderModule shader_gdp; VkDescriptorSetLayout dsl_gdp; VkPipelineLayout plyt_gdp;
    VkPipeline pipe_gdp; VkDescriptorPool dpool_gdp; VkDescriptorSet dset_gdp;
    VkShaderModule shader_topk; VkDescriptorSetLayout dsl_topk; VkPipelineLayout plyt_topk;
    VkPipeline pipe_topk; VkDescriptorPool dpool_topk; VkDescriptorSet dset_topk;
    /* COLI_VK_MOE_IX: device-indexed experts (descriptor arrays + moe_indexed.comp). */
    int moe_ix_hw, moe_ix_user, moe_ix_fused, moe_ix_fused_last;
    int eg_ix_nlayers, eg_ix_E, eg_ix_D, eg_ix_I, eg_ix_gs, eg_ix_nslots;
    int eg_ix_grw, eg_ix_drw, eg_ix_sh_ok[VK_KV_LAYERS];
    VkBuffer eg_ix_valid, eg_ix_shgate, eg_ix_dummy_w, eg_ix_dummy_s;
    VkDeviceMemory eg_ix_validm, eg_ix_shgatem, eg_ix_dummym, eg_ix_dummy_sm;
    void *eg_ix_validp, *eg_ix_shgatep;
    uint32_t *eg_ix_valid_cpu;
    VkShaderModule shader_moe_ix, shader_moe_pack;
    VkDescriptorSetLayout dsl_moe_ix, dsl_moe_pack;
    VkPipelineLayout plyt_moe_ix, plyt_moe_pack;
    VkPipeline pipe_moe_ix, pipe_moe_pack;
    VkDescriptorPool dpool_moe_ix, dpool_moe_pack;
    VkDescriptorSet dset_moe_ix, dset_moe_pack;
    /* Ping-pong route CBs for moe_ix async: CPU records slot s while GPU runs 1-s.
     * Dual descriptor sets (per-layer weight binds) + dual idx/val (hist reclaim).
     * COLI_VK_MOE_IX_PP=0 disables (default on when MOE_IX=1). */
    int moe_ix_pp, pp_rec, pp_slot, pp_inflight[2], pp_layer[2], pp_topk[2];
    int ix_hist_pending; /* async fused: eusage applied from hist after drain */
    int moe_ix_rt_bound;
    VkBuffer moe_ix_rt_bufs[7];
    VkCommandBuffer cmd_pp[2]; VkFence fence_pp[2];
    VkDescriptorPool pp_pool;
    VkDescriptorSet pp_nrm[2], pp_gq[2], pp_gk[2], pp_gv[2], pp_go[2];
    VkDescriptorSet pp_qkv[2], pp_gqa[2], pp_rrnz[2], pp_route[2], pp_topk_ds[2];
    VkDescriptorSet pp_macc[2], pp_pack[2];
    VkDescriptorSet pp_gbb[2], pp_gba[2], pp_gdp[2];
    VkDescriptorSet pp_gqkv[2], pp_gz[2], pp_gdnconv[2], pp_gdncv[2], pp_gdnout[2];
    Scratch route_idx_pp[2], route_val_pp[2];
    int ix_hist[VK_KV_LAYERS][64];
    int ix_hist_ready[VK_KV_LAYERS], ix_hist_k[VK_KV_LAYERS];
    VkDescriptorPool pipe_mm_pool; VkDescriptorSet dset_gba, dset_gbb; /* b/a matmul sets */
    VkBuffer gdn_alog[VK_KV_LAYERS], gdn_dtb[VK_KV_LAYERS], gdn_gn[VK_KV_LAYERS];
    VkDeviceMemory gdn_alogm[VK_KV_LAYERS], gdn_dtbm[VK_KV_LAYERS], gdn_gnm[VK_KV_LAYERS];
    void *gdn_alogp[VK_KV_LAYERS], *gdn_dtbp[VK_KV_LAYERS], *gdn_gnp[VK_KV_LAYERS];
    int gdn_ba_VH[VK_KV_LAYERS], gdn_ba_VD[VK_KV_LAYERS];
    VkKvLayer kv[VK_KV_LAYERS];  /* per-layer resident KV latent/rope cache */
    /* resubmit cache: skip vkUpdateDescriptorSets + command re-record when the bound
     * tensor / shape / scratch buffers are unchanged from the previous call (the hot-
     * expert-called-repeatedly pattern). The synchronous fence wait each call means no
     * submission is ever in flight, so rebinding/re-recording only when something
     * actually changed is safe. */
    ColiVkTensor *bound_tensor; int bound_S, bound_I, bound_O, cmd_ready;
    VkBuffer bound_xbuf, bound_ybuf;
    size_t used_bytes, tensor_count;
    /* VRAM pressure-proofing: with VK_EXT_memory_priority the attention working set
     * (KV mirror, scratches) outranks bulk expert weights, so an oversubscribed heap
     * evicts cold tier experts instead of thrashing the per-token attention submits
     * (measured: decode attention 7.8s at 7.6 GB resident -> 17.8s at 15.2 GB).
     * VK_EXT_memory_budget lets the tier fill stop at a reserve instead of guessing. */
    int has_prio, has_budget;
    float prio;                  /* priority applied to the NEXT allocations (class knob) */
} G;

struct PC { int fmt, S, I, O, rowWords, gs; };
/* Push constants of the row-quantize and the two DP4A expert kernels. sbase indexes the
 * shared int8 activation + per-row scale buffers, which are bound whole (a 4B-granular
 * scale slice cannot meet minStorageBufferOffsetAlignment); only the outputs are sliced. */
struct PCQr { int S, I; };
struct PCDp4a { int S, I, O, rowWords, gs, sbase; };
struct PCN { int S, D; float eps; };
/* Push constants of the device argmax reduction (must match argmax.comp). */
struct PCAm { int N, stage, ngrp; };
#define AM_GRP 64            /* stage-0 workgroups; stage 1 folds their partial winners */
/* Push constants of the absorb attention kernel (must match attention_absorb.comp). */
struct PCAttn { int fmt, S, H, Q, R, V, K, st0, T, rowWords, cap; float scale; int gs; };
/* Push constants of the flash absorb kernel (must match attention_absorb_flash.comp). */
struct PCAttnF { int mode, fmt, S, H, Q, R, V, K, st0, T, rowWords, gs, nc, ch; float scale; };
/* Push constants of the GatedDeltaNet delta-rule kernel (must match gdn_delta.comp). */
struct PCGdn { uint32_t KH, KD, VH, VD; float eps; };
/* Push constants of the GQA decode attention kernel (must match gqa_attn.comp). */
struct PCGqa { int S, H, KH, hd, max_t, st0, T; float scale; };
/* Push constants of the fused GDN input-block kernels. */
struct PCConv { int CD, K; };                                  /* gdn_conv.comp */
struct PCGdnCv { uint32_t KH, KD, VH, VD, key_dim; float eps; };  /* gdn_delta_cv.comp */
/* Push constants of the GQA q/k-norm+rope+KV-write kernel (must match gqa_qkv.comp). */
struct PCQkv { int S, H, KH, hd, rot, pos_base, max_t; float eps, theta; };
struct PCRep { int count, D; };
struct PCGdp { int VH, VD; };
/* softmax_topk.comp: do_pack writes moe_w[0..K) (+ shared gate at K when D>0). */
struct PCTopk { int E, K, base, D, do_pack; };
struct PCMoeIx { int mode, K, D, I, E, base, rowWords_g, rowWords_d, gs, do_shared, ibase; };
struct PCMoePack { int K, D, do_shared, base; };

static int pick_memtype(void) {
    VkPhysicalDeviceMemoryProperties m;
    vkGetPhysicalDeviceMemoryProperties(G.phys, &m);
    int best = -1;
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) return (int)i; // ideal on APU
            if (best < 0) best = (int)i;
        }
    }
    return best;
}

/* Cached+coherent host-visible type for buffers the CPU READS BACK. pick_memtype prefers
 * DEVICE_LOCAL host-visible = write-combined VRAM over ReBAR, which the CPU writes fast but
 * reads catastrophically slowly (~40 MB/s). Outputs must be HOST_CACHED for cheap readback. */
static int pick_memtype_cached(void) {
    VkPhysicalDeviceMemoryProperties m;
    vkGetPhysicalDeviceMemoryProperties(G.phys, &m);
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) return (int)i;
    }
    return pick_memtype();   /* no cached type -> fall back (no worse than before) */
}

static int alloc_hostvis_mt(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem, void **ptr, uint32_t memtype) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(G.dev, &bi, NULL, buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G.dev, *buf, &req);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = memtype};
#ifdef VK_EXT_memory_priority
    VkMemoryPriorityAllocateInfoEXT pri = {.sType = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT,
        .priority = G.prio};
    if (G.has_prio) ai.pNext = &pri;
#endif
    VKCHECK(vkAllocateMemory(G.dev, &ai, NULL, mem), "vkAllocateMemory");
    VKCHECK(vkBindBufferMemory(G.dev, *buf, *mem, 0), "vkBindBufferMemory");
    if (ptr) VKCHECK(vkMapMemory(G.dev, *mem, 0, bytes, 0, ptr), "vkMapMemory");
    return 1;
}
/* Priority class of subsequent allocations (VK_EXT_memory_priority; no-op without it).
 * Scratches/KV force 1.0 internally; weight uploads take whatever is current — the
 * engine sets 0.4 around the bulk expert-tier fill, dense stays at the 0.75 default. */
void coli_vk_alloc_priority(float p) { G.prio = p < 0 ? 0 : p > 1 ? 1 : p; }

/* Device-local heap usage/budget in GB (VK_EXT_memory_budget). Returns 0 when the
 * extension is absent — callers then keep their count-based caps unchanged. */
int coli_vk_mem_budget(double *used_gb, double *budget_gb) {
#ifdef VK_EXT_memory_budget
    if (!G.has_budget || !G.phys) return 0;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bud = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &bud};
    vkGetPhysicalDeviceMemoryProperties2(G.phys, &mp2);
    double u = 0, b = 0;
    for (uint32_t i = 0; i < mp2.memoryProperties.memoryHeapCount; i++)
        if (mp2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            u += (double)bud.heapUsage[i]; b += (double)bud.heapBudget[i];
        }
    if (used_gb) *used_gb = u / 1e9;
    if (budget_gb) *budget_gb = b / 1e9;
    return b > 0;
#else
    (void)used_gb; (void)budget_gb; return 0;
#endif
}
static int alloc_hostvis(size_t bytes, VkBuffer *buf, VkDeviceMemory *mem, void **ptr) {
    return alloc_hostvis_mt(bytes, buf, mem, ptr, G.memtype);
}

static int scratch_reserve_mt(Scratch *s, size_t bytes, uint32_t memtype) {
    if (s->cap >= bytes) return 1;
    if (s->buf) { vkDestroyBuffer(G.dev, s->buf, NULL); vkFreeMemory(G.dev, s->mem, NULL); }
    s->buf = VK_NULL_HANDLE; s->cap = 0; s->ptr = NULL;
    float p0 = G.prio; G.prio = 1.0f;            /* scratches ride every submit: never evict */
    int ok = alloc_hostvis_mt(bytes, &s->buf, &s->mem, &s->ptr, memtype);
    G.prio = p0;
    if (!ok) return 0;
    s->cap = bytes;
    return 1;
}
static int scratch_reserve(Scratch *s, size_t bytes) { return scratch_reserve_mt(s, bytes, G.memtype); }

static int rowwords(int fmt, int I) {
    size_t rb = fmt == 1 ? (size_t)I                         // bytes/row on CPU side
              : fmt == 5 ? ((size_t)I + 63) / 64 * 24        // int3-g64: 24B per 64-group
              : (size_t)(I + 1) / 2;                         // int4 variants (2,4,6): nibble-packed
    return (int)((rb + 3) / 4);                              // padded to uint32 (24|4: exact)
}
/* Scale floats per tensor: per-row formats carry O, int3-g64 carries O*ceil(I/64)
 * (one f32 per 64-input group), grouped int4 carries O*ng, grouped ASYMMETRIC int4
 * (fmt=6) carries O*ng*2 (interleaved [scale,zero] per group). upload_tensor and
 * tensor_free must agree on this. */
static size_t scale_floats(int fmt, int I, int O, int gs) {
    if (fmt == 5) return (size_t)O * (((size_t)I + 63) / 64);
    if (fmt == 4) return (size_t)O * (((size_t)I + gs - 1) / gs);       // per-group [O,ng]
    if (fmt == 6) return (size_t)O * (((size_t)I + gs - 1) / gs) * 2;   // per-group [O,ng,2] scale+zero
    return (size_t)O;
}

static VkShaderModule load_spv(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[VK] cannot open %s\n", path); return VK_NULL_HANDLE; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0 || n % 4 != 0) {   // SPIR-V is a stream of uint32; empty/non-seekable/odd size is invalid
        fprintf(stderr, "[VK] bad SPIR-V size %ld in %s\n", n, path); fclose(f); return VK_NULL_HANDLE; }
    uint32_t *code = malloc((size_t)n);
    if (!code) { fclose(f); return VK_NULL_HANDLE; }
    if (fread(code, 1, n, f) != (size_t)n) { fclose(f); free(code); return VK_NULL_HANDLE; }
    fclose(f);
    VkShaderModuleCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = n, .pCode = code};
    VkShaderModule m;
    VkResult r = vkCreateShaderModule(G.dev, &si, NULL, &m);
    free(code);
    return r == VK_SUCCESS ? m : VK_NULL_HANDLE;
}

/* Build a compute pipeline + descriptor pool/set for nbind storage buffers with a
 * pc_size-byte push constant. Used by the 4-binding matmul, 6-binding gate_up and
 * 7-binding absorb attention pipelines. */
static int build_pipeline(int nbind, size_t pc_size, VkShaderModule shader,
                          VkDescriptorSetLayout *dsl, VkPipelineLayout *plyt, VkPipeline *pipe,
                          VkDescriptorPool *dpool, VkDescriptorSet *dset) {
    VkDescriptorSetLayoutBinding b[8];
    for (int i = 0; i < nbind; i++) b[i] = (VkDescriptorSetLayoutBinding){
        .binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dsli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = (uint32_t)nbind, .pBindings = b};
    VKCHECK(vkCreateDescriptorSetLayout(G.dev, &dsli, NULL, dsl), "descSetLayout");
    VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = (uint32_t)pc_size};
    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
    VKCHECK(vkCreatePipelineLayout(G.dev, &pli, NULL, plyt), "pipelineLayout");
    VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main"},
        .layout = *plyt};
    VKCHECK(vkCreateComputePipelines(G.dev, VK_NULL_HANDLE, 1, &cpi, NULL, pipe), "pipeline");
    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = (uint32_t)nbind};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, dpool), "descPool");
    VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = *dpool, .descriptorSetCount = 1, .pSetLayouts = dsl};
    VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa, dset), "allocDescSet");
    return 1;
}

static int moe_ix_features_ok(void) {
    VkPhysicalDeviceDescriptorIndexingFeatures dix = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &dix};
    vkGetPhysicalDeviceFeatures2(G.phys, &f2);
    return dix.runtimeDescriptorArray && dix.shaderStorageBufferArrayNonUniformIndexing &&
           dix.descriptorBindingPartiallyBound && dix.descriptorBindingStorageBufferUpdateAfterBind;
}

static int build_moe_ix_pipeline(int nslots, VkShaderModule shader,
                                 VkDescriptorSetLayout *dsl, VkPipelineLayout *plyt, VkPipeline *pipe,
                                 VkDescriptorPool *dpool, VkDescriptorSet *dset) {
    VkDescriptorBindingFlags arf = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                   VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    VkDescriptorBindingFlags abf[13];
    for (int i = 0; i < 13; i++) abf[i] = (i >= 7) ? arf : 0;
    VkDescriptorSetLayoutBindingFlagsCreateInfo fl = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = 13, .pBindingFlags = abf};
    VkDescriptorSetLayoutBinding b[13];
    for (int i = 0; i < 13; i++) {
        b[i] = (VkDescriptorSetLayoutBinding){
            .binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = (uint32_t)(i >= 7 ? nslots : 1),
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    }
    VkDescriptorSetLayoutCreateInfo dsli = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = 13, .pBindings = b, .pNext = &fl};
    VKCHECK(vkCreateDescriptorSetLayout(G.dev, &dsli, NULL, dsl), "moe_ix dsl");
    VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0,
        .size = (uint32_t)sizeof(struct PCMoeIx)};
    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
    VKCHECK(vkCreatePipelineLayout(G.dev, &pli, NULL, plyt), "moe_ix plyt");
    VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main"},
        .layout = *plyt};
    VKCHECK(vkCreateComputePipelines(G.dev, VK_NULL_HANDLE, 1, &cpi, NULL, pipe), "moe_ix pipe");
    /* bindings 0..6 scalar + bindings 7..12 × nslots arrays */
    uint32_t nd = (uint32_t)(7 + 6 * nslots);
    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = nd};
    VkDescriptorPoolCreateInfo dpi = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, dpool), "moe_ix pool");
    VkDescriptorSetAllocateInfo dsa = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = *dpool, .descriptorSetCount = 1, .pSetLayouts = dsl};
    VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa, dset), "moe_ix set");
    return 1;
}

static void eg_ix_bind_array_slot(int di, ColiVkTensor *g, ColiVkTensor *u, ColiVkTensor *d) {
    VkDescriptorBufferInfo bi[6] = {
        {g->wbuf, 0, VK_WHOLE_SIZE}, {g->sbuf, 0, VK_WHOLE_SIZE},
        {u->wbuf, 0, VK_WHOLE_SIZE}, {u->sbuf, 0, VK_WHOLE_SIZE},
        {d->wbuf, 0, VK_WHOLE_SIZE}, {d->sbuf, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[6];
    for (int i = 0; i < 6; i++) {
        wr[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = G.dset_moe_ix, .dstBinding = (uint32_t)(7 + i), .dstArrayElement = (uint32_t)di,
            .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bi[i]};
    }
    vkUpdateDescriptorSets(G.dev, 6, wr, 0, NULL);
}

static void eg_ix_bind_dummy_slot(int di) {
    VkDescriptorBufferInfo bi[6] = {
        {G.eg_ix_dummy_w, 0, VK_WHOLE_SIZE}, {G.eg_ix_dummy_s, 0, VK_WHOLE_SIZE},
        {G.eg_ix_dummy_w, 0, VK_WHOLE_SIZE}, {G.eg_ix_dummy_s, 0, VK_WHOLE_SIZE},
        {G.eg_ix_dummy_w, 0, VK_WHOLE_SIZE}, {G.eg_ix_dummy_s, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[6];
    for (int i = 0; i < 6; i++) {
        wr[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = G.dset_moe_ix, .dstBinding = (uint32_t)(7 + i), .dstArrayElement = (uint32_t)di,
            .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bi[i]};
    }
    vkUpdateDescriptorSets(G.dev, 6, wr, 0, NULL);
}

/* "…/qmatmul.spv" -> "…/qmatmul<suffix>" (sibling of the main shader). */
static void derive_sibling(const char *spv, const char *suffix, char *out, size_t n) {
    const char *dot = strstr(spv, ".spv");
    if (dot && (size_t)(dot - spv) + strlen(suffix) + 1 < n) {
        size_t pre = (size_t)(dot - spv);
        memcpy(out, spv, pre); strcpy(out + pre, suffix);
    } else snprintf(out, n, "%s", spv);
}
/* "…/qmatmul.spv" -> "…/attention_absorb.spv" (same directory). */
static void derive_dir_file(const char *spv, const char *fname, char *out, size_t n) {
    /* Accept both POSIX and Windows separators (MinGW argv often uses '\'). */
    const char *sl = strrchr(spv, '/');
    const char *bs = strrchr(spv, '\\');
    if (bs && (!sl || bs > sl)) sl = bs;
    size_t pre = sl ? (size_t)(sl - spv) + 1 : 0;
    if (pre + strlen(fname) + 1 < n) { memcpy(out, spv, pre); strcpy(out + pre, fname); }
    else snprintf(out, n, "%s", fname);
}

int coli_vk_init(const char *spv_path) {
    if (G.ready) return 1;
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app};
    VKCHECK(vkCreateInstance(&ici, NULL, &G.inst), "vkCreateInstance");

    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(G.inst, &nd, NULL);
    if (!nd) { fprintf(stderr, "[VK] no devices\n"); return 0; }
    VkPhysicalDevice devs[8]; if (nd > 8) nd = 8;
    vkEnumeratePhysicalDevices(G.inst, &nd, devs);
    // Prefer a real GPU over a CPU/software device (llvmpipe) on multi-adapter hosts:
    // discrete > integrated > virtual > other/cpu. Falls back to devs[0] if all equal.
    G.phys = devs[0];
    int bestrank = -1;
    for (uint32_t i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(devs[i], &p);
        int rank = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? 4 :
                   p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3 :
                   p.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? 2 :
                   p.deviceType == VK_PHYSICAL_DEVICE_TYPE_OTHER          ? 1 : 0; // CPU last
        if (rank > bestrank) { bestrank = rank; G.phys = devs[i]; }
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(G.phys, &nq, NULL);
    VkQueueFamilyProperties qf[16]; if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(G.phys, &nq, qf);
    G.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { G.qfam = i; break; }
    if (G.qfam == UINT32_MAX) { fprintf(stderr, "[VK] no compute queue\n"); return 0; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = G.qfam, .queueCount = 1, .pQueuePriorities = &prio};
    /* Pressure-proofing extensions (both optional, detected at runtime):
     * memory_priority ranks allocations for the kernel's eviction order,
     * memory_budget exposes how much VRAM a new allocation can still take. */
    const char *dext[2]; uint32_t ndext = 0;
    {
        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(G.phys, NULL, &ne, NULL);
        VkExtensionProperties *ep = ne ? malloc(ne * sizeof(*ep)) : NULL;
        if (ep) {
            vkEnumerateDeviceExtensionProperties(G.phys, NULL, &ne, ep);
            for (uint32_t i = 0; i < ne; i++) {
#ifdef VK_EXT_memory_priority
                if (!strcmp(ep[i].extensionName, VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME)) G.has_prio = 1;
#endif
#ifdef VK_EXT_memory_budget
                if (!strcmp(ep[i].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) G.has_budget = 1;
#endif
            }
            free(ep);
        }
    }
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
#ifdef VK_EXT_memory_priority
    VkPhysicalDeviceMemoryPriorityFeaturesEXT prif = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT,
        .memoryPriority = VK_TRUE};
    if (G.has_prio) { dext[ndext++] = VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME; prif.pNext = (void *)di.pNext; di.pNext = &prif; }
#endif
#ifdef VK_EXT_memory_budget
    if (G.has_budget) dext[ndext++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
#endif
    G.moe_ix_hw = 0;
    G.moe_ix_user = g_qwen_opts.moe_ix != 0;
    VkPhysicalDeviceDescriptorIndexingFeatures dix_en = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    if (G.moe_ix_user && moe_ix_features_ok()) {
        G.moe_ix_hw = 1;
        dix_en.runtimeDescriptorArray = VK_TRUE;
        dix_en.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
        dix_en.descriptorBindingPartiallyBound = VK_TRUE;
        dix_en.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
        dix_en.pNext = (void *)di.pNext;
        di.pNext = &dix_en;
    } else if (G.moe_ix_user) {
        fprintf(stderr, "[VK] --moe-ix but descriptor indexing unavailable — disabled\n");
        G.moe_ix_user = 0;
    }
    di.enabledExtensionCount = ndext; di.ppEnabledExtensionNames = ndext ? dext : NULL;
    G.prio = 0.75f;                              /* default class: dense/resident weights */
    VKCHECK(vkCreateDevice(G.phys, &di, NULL, &G.dev), "vkCreateDevice");
    vkGetDeviceQueue(G.dev, G.qfam, 0, &G.queue);
    if (G.has_prio || G.has_budget)
        fprintf(stderr, "[VK] VRAM pressure-proofing: memory_priority %s, memory_budget %s\n",
                G.has_prio ? "on" : "absent", G.has_budget ? "on" : "absent");

    int mt = pick_memtype();
    if (mt < 0) { fprintf(stderr, "[VK] no host-visible memory\n"); return 0; }
    G.memtype = (uint32_t)mt;
    G.memtype_cached = (uint32_t)pick_memtype_cached();

    G.shader = load_spv(spv_path);
    if (!G.shader) return 0;
    if (!build_pipeline(4, sizeof(struct PC), G.shader, &G.dsl, &G.plyt, &G.pipe, &G.dpool, &G.dset)) return 0;

    /* Optional fused gate+up pipeline: skip gracefully if its shader isn't present
     * (single-matmul path keeps working). */
    char gu_path[512]; derive_sibling(spv_path, "_gate_up.spv", gu_path, sizeof(gu_path));
    G.shader_gu = load_spv(gu_path);
    if (G.shader_gu && !build_pipeline(6, sizeof(struct PC), G.shader_gu, &G.dsl_gu, &G.plyt_gu, &G.pipe_gu, &G.dpool_gu, &G.dset_gu))
        return 0;
    /* Optional DP4A expert path: all three kernels or none, since both matmuls consume
     * int8 activations that only quant_rows produces. */
    {
        char qr_path[512], gud_path[512], dn_path[512];
        derive_dir_file(spv_path, "quant_rows.spv", qr_path, sizeof(qr_path));
        derive_dir_file(spv_path, "qmatmul_gate_up_dp4a.spv", gud_path, sizeof(gud_path));
        derive_dir_file(spv_path, "qmatmul_down_dp4a.spv", dn_path, sizeof(dn_path));
        G.shader_qr = load_spv(qr_path);
        G.shader_gu_dp4a = load_spv(gud_path);
        G.shader_dn_dp4a = load_spv(dn_path);
        if (G.shader_qr && G.shader_gu_dp4a && G.shader_dn_dp4a) {
            if (!build_pipeline(3, sizeof(struct PCQr), G.shader_qr, &G.dsl_qr, &G.plyt_qr,
                                &G.pipe_qr, &G.dpool_qr, &G.dset_qr_h) ||
                !build_pipeline(7, sizeof(struct PCDp4a), G.shader_gu_dp4a, &G.dsl_gu4, &G.plyt_gu4,
                                &G.pipe_gu_dp4a, &G.dpool_gu4, &G.dset_gu4) ||
                !build_pipeline(5, sizeof(struct PCDp4a), G.shader_dn_dp4a, &G.dsl_dn4, &G.plyt_dn4,
                                &G.pipe_dn_dp4a, &G.dpool_dn4, &G.dset_dn4))
                return 0;
            fprintf(stderr, "[VK] DP4A expert kernels loaded (int4 gate/up/down)\n");
        } else {
            G.pipe_qr = VK_NULL_HANDLE; G.pipe_gu_dp4a = VK_NULL_HANDLE; G.pipe_dn_dp4a = VK_NULL_HANDLE;
        }
    }

    /* Optional device-argmax pipeline: lets the lm_head projection reduce to a top-1 on
     * the GPU (coli_vk_matmul_argmax); absent -> callers read the full logit vector. */
    char am_path[512]; derive_dir_file(spv_path, "argmax.spv", am_path, sizeof(am_path));
    G.shader_am = load_spv(am_path);
    if (G.shader_am && !build_pipeline(3, sizeof(struct PCAm), G.shader_am, &G.dsl_am, &G.plyt_am,
                                       &G.pipe_am, &G.dpool_am, &G.dset_am))
        return 0;

    /* Optional MLA absorb attention pipeline (same directory as the main shader). */
    /* Optional rmsnorm pipeline: enables the pair->norm->q_b single-submit chain
     * (coli_vk_attn_qprep); absent -> callers keep the 3-submit path. */
    char nrm_path[512]; derive_dir_file(spv_path, "rmsnorm.spv", nrm_path, sizeof(nrm_path));
    G.shader_nrm = load_spv(nrm_path);
    if (G.shader_nrm) {
        VkDescriptorPool np; VkDescriptorSet ns;
        if (!build_pipeline(3, sizeof(struct PCN), G.shader_nrm, &G.dsl_nrm, &G.plyt_nrm, &G.pipe_nrm, &np, &ns))
            return 0;
        G.dset_nrm = ns;
        /* one extra 4-binding matmul set for the chain's 3rd matmul (dset+dset_pair serve 1+2) */
        VkDescriptorPoolSize ps3 = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4};
        VkDescriptorPoolCreateInfo dpi3 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps3};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi3, NULL, &G.qprep_pool), "qprep descPool");
        VkDescriptorSetAllocateInfo dsa3 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.qprep_pool, .descriptorSetCount = 1, .pSetLayouts = &G.dsl};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa3, &G.dset_qp3), "qprep descSet");
    }
    /* Qwen zero-centered RMSNorm: same 3-buffer / PCN layout as plain rmsnorm, so it
     * reuses dsl_nrm/plyt_nrm and only needs its own pipeline object. */
    char nrmz_path[512]; derive_dir_file(spv_path, "rmsnorm_zc.spv", nrmz_path, sizeof(nrmz_path));
    G.shader_nrmz = load_spv(nrmz_path);
    if (G.shader_nrmz && G.plyt_nrm) {
        VkComputePipelineCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                      .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = G.shader_nrmz, .pName = "main"},
            .layout = G.plyt_nrm};
        VKCHECK(vkCreateComputePipelines(G.dev, VK_NULL_HANDLE, 1, &cpi, NULL, &G.pipe_nrmz), "nrmz pipeline");
    }
    /* residual+=delta + rmsnorm_zc — MoE prep tail riding the attn/GDN full submit. */
    char rrnz_path[512]; derive_dir_file(spv_path, "residual_rmsnorm_zc.spv", rrnz_path, sizeof(rrnz_path));
    G.shader_rrnz = load_spv(rrnz_path);
    if (G.shader_rrnz && !build_pipeline(5, sizeof(struct PCN), G.shader_rrnz, &G.dsl_rrnz, &G.plyt_rrnz,
                                         &G.pipe_rrnz, &G.dpool_rrnz, &G.dset_rrnz))
        return 0;
    if (G.pipe_rrnz) {
        VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.route_pool), "route descPool");
        VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.route_pool, .descriptorSetCount = 1, .pSetLayouts = &G.dsl};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa, &G.dset_route_rt), "route router set");
        fprintf(stderr, "[VK] MoE prep fusion loaded (residual+rmsnorm_zc+router)\n");
    }
    /* Optional stream pipeline: replicate nrm → eg_x, moe accumulate into resid,
     * GDN b/a pack, device softmax+topk. Absent any of these keeps the staged path. */
    {
        char rep_path[512], macc_path[512], gdp_path[512], topk_path[512];
        derive_dir_file(spv_path, "replicate_rows.spv", rep_path, sizeof(rep_path));
        derive_dir_file(spv_path, "moe_accumulate.spv", macc_path, sizeof(macc_path));
        derive_dir_file(spv_path, "gdn_pack_params.spv", gdp_path, sizeof(gdp_path));
        derive_dir_file(spv_path, "softmax_topk.spv", topk_path, sizeof(topk_path));
        G.shader_rep = load_spv(rep_path);
        G.shader_macc = load_spv(macc_path);
        G.shader_gdp = load_spv(gdp_path);
        G.shader_topk = load_spv(topk_path);
        if (G.shader_rep && G.shader_macc && G.shader_gdp && G.shader_topk &&
            G.pipe_rrnz && G.shader_gu) {
            if (!build_pipeline(2, sizeof(struct PCRep), G.shader_rep, &G.dsl_rep, &G.plyt_rep,
                                &G.pipe_rep, &G.dpool_rep, &G.dset_rep) ||
                !build_pipeline(3, sizeof(struct PCRep), G.shader_macc, &G.dsl_macc, &G.plyt_macc,
                                &G.pipe_macc, &G.dpool_macc, &G.dset_macc) ||
                !build_pipeline(6, sizeof(struct PCGdp), G.shader_gdp, &G.dsl_gdp, &G.plyt_gdp,
                                &G.pipe_gdp, &G.dpool_gdp, &G.dset_gdp) ||
                !build_pipeline(6, sizeof(struct PCTopk), G.shader_topk, &G.dsl_topk, &G.plyt_topk,
                                &G.pipe_topk, &G.dpool_topk, &G.dset_topk))
                return 0;
            VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 8};
            VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &ps};
            VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.pipe_mm_pool), "pipe mm pool");
            VkDescriptorSetLayout lg[2] = {G.dsl, G.dsl};
            VkDescriptorSet sets[2];
            VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = G.pipe_mm_pool, .descriptorSetCount = 2, .pSetLayouts = lg};
            VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa, sets), "pipe mm sets");
            G.dset_gba = sets[0]; G.dset_gbb = sets[1];
            VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            VKCHECK(vkCreateSemaphore(G.dev, &sci, NULL, &G.eg_sem), "eg semaphore");
            fprintf(stderr, "[VK] decode stream pipeline loaded (device nrm→eg, topk, resid accumulate, eg→route sem)\n");
        } else {
            G.pipe_rep = VK_NULL_HANDLE; G.pipe_macc = VK_NULL_HANDLE;
            G.pipe_gdp = VK_NULL_HANDLE; G.pipe_topk = VK_NULL_HANDLE;
        }
    }
    if (G.moe_ix_hw && G.pipe_macc && G.pipe_topk) {
        char mix_path[512], mpk_path[512];
        derive_dir_file(spv_path, "moe_indexed.spv", mix_path, sizeof(mix_path));
        derive_dir_file(spv_path, "moe_pack_w.spv", mpk_path, sizeof(mpk_path));
        G.shader_moe_ix = load_spv(mix_path);
        G.shader_moe_pack = load_spv(mpk_path);
        if (G.shader_moe_ix && G.shader_moe_pack &&
            build_pipeline(4, sizeof(struct PCMoePack), G.shader_moe_pack, &G.dsl_moe_pack, &G.plyt_moe_pack,
                           &G.pipe_moe_pack, &G.dpool_moe_pack, &G.dset_moe_pack))
            fprintf(stderr, "[VK] moe_pack_w loaded (indexed MoE weight pack)\n");
        else
            G.pipe_moe_pack = VK_NULL_HANDLE;
    }
    char att_path[512]; derive_dir_file(spv_path, "attention_absorb.spv", att_path, sizeof(att_path));
    G.shader_att = load_spv(att_path);
    if (G.shader_att && !build_pipeline(7, sizeof(struct PCAttn), G.shader_att, &G.dsl_att, &G.plyt_att, &G.pipe_att, &G.dpool_att, &G.dset_att))
        return 0;
    /* Optional FlashMLA-style chunked absorb (head-batched, for long contexts). */
    char attf_path[512]; derive_dir_file(spv_path, "attention_absorb_flash.spv", attf_path, sizeof(attf_path));
    G.shader_attf = load_spv(attf_path);
    if (G.shader_attf && !build_pipeline(8, sizeof(struct PCAttnF), G.shader_attf, &G.dsl_attf, &G.plyt_attf, &G.pipe_attf, &G.dpool_attf, &G.dset_attf))
        return 0;
    /* Optional GatedDeltaNet delta-rule kernel (Qwen hybrid linear-attention layers). */
    char gdn_path[512]; derive_dir_file(spv_path, "gdn_delta.spv", gdn_path, sizeof(gdn_path));
    G.shader_gdn = load_spv(gdn_path);
    if (G.shader_gdn && !build_pipeline(3, sizeof(struct PCGdn), G.shader_gdn, &G.dsl_gdn, &G.plyt_gdn, &G.pipe_gdn, &G.dpool_gdn, &G.dset_gdn))
        return 0;
    /* Optional standard GQA decode attention (Qwen full-attention layers). */
    char gqa_path[512]; derive_dir_file(spv_path, "gqa_attn.spv", gqa_path, sizeof(gqa_path));
    G.shader_gqa = load_spv(gqa_path);
    if (G.shader_gqa && !build_pipeline(4, sizeof(struct PCGqa), G.shader_gqa, &G.dsl_gqa, &G.plyt_gqa, &G.pipe_gqa, &G.dpool_gqa, &G.dset_gqa))
        return 0;
    /* Optional fused GDN input block (conv1d + delta-cv). Needs the base qmatmul pipeline
     * (G.dsl) for the gqkv/gz/out matmuls, so 3 extra sets are allocated from gdnf_pool. */
    char gcv_path[512]; derive_dir_file(spv_path, "gdn_conv.spv", gcv_path, sizeof(gcv_path));
    G.shader_gdnconv = load_spv(gcv_path);
    if (G.shader_gdnconv && !build_pipeline(4, sizeof(struct PCConv), G.shader_gdnconv, &G.dsl_gdnconv, &G.plyt_gdnconv, &G.pipe_gdnconv, &G.dpool_gdnconv, &G.dset_gdnconv))
        return 0;
    char gdc_path[512]; derive_dir_file(spv_path, "gdn_delta_cv.spv", gdc_path, sizeof(gdc_path));
    G.shader_gdncv = load_spv(gdc_path);
    if (G.shader_gdncv && !build_pipeline(5, sizeof(struct PCGdnCv), G.shader_gdncv, &G.dsl_gdncv, &G.plyt_gdncv, &G.pipe_gdncv, &G.dpool_gdncv, &G.dset_gdncv))
        return 0;
    if (G.shader_gdnconv && G.shader_gdncv) {        /* 3 extra 4-binding qmatmul sets (gqkv|gz|out) */
        VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3 * 4};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 3, .poolSizeCount = 1, .pPoolSizes = &ps};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.gdnf_pool), "gdnf descPool");
        VkDescriptorSetLayout lg[3] = {G.dsl, G.dsl, G.dsl};
        VkDescriptorSet sets[3];
        VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.gdnf_pool, .descriptorSetCount = 3, .pSetLayouts = lg};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa, sets), "gdnf sets");
        G.gdnf_gqkv = sets[0]; G.gdnf_gz = sets[1]; G.gdnf_out = sets[2];
    }
    /* Optional fused GQA input block (q/k-norm+rope+KV-write). Needs the base qmatmul
     * pipeline (G.dsl) for gq/gk/gv/out, so 4 extra sets come from gqaf_pool. */
    char qkv_path[512]; derive_dir_file(spv_path, "gqa_qkv.spv", qkv_path, sizeof(qkv_path));
    G.shader_qkv = load_spv(qkv_path);
    if (G.shader_qkv && !build_pipeline(7, sizeof(struct PCQkv), G.shader_qkv, &G.dsl_qkv, &G.plyt_qkv, &G.pipe_qkv, &G.dpool_qkv, &G.dset_qkv))
        return 0;
    if (G.shader_qkv && G.pipe_gqa) {                /* 4 extra 4-binding qmatmul sets (gq|gk|gv|out) */
        VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4 * 4};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 4, .poolSizeCount = 1, .pPoolSizes = &ps};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.gqaf_pool), "gqaf descPool");
        VkDescriptorSetLayout lg[4] = {G.dsl, G.dsl, G.dsl, G.dsl};
        VkDescriptorSet sets[4];
        VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.gqaf_pool, .descriptorSetCount = 4, .pSetLayouts = lg};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa, sets), "gqaf sets");
        G.gqaf_q = sets[0]; G.gqaf_k = sets[1]; G.gqaf_v = sets[2]; G.gqaf_out = sets[3];
    }

    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = G.qfam};
    VKCHECK(vkCreateCommandPool(G.dev, &cpci, NULL, &G.cpool), "cmdPool");
    VkCommandBufferAllocateInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = G.cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VKCHECK(vkAllocateCommandBuffers(G.dev, &cbi, &G.cmd), "cmdBuf");
    VKCHECK(vkAllocateCommandBuffers(G.dev, &cbi, &G.eg_cmd), "eg cmdBuf");
    { VkCommandBufferAllocateInfo cbi2 = cbi; cbi2.commandBufferCount = 2;
      VKCHECK(vkAllocateCommandBuffers(G.dev, &cbi2, G.cmd_pp), "cmdBuf pp"); }
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKCHECK(vkCreateFence(G.dev, &fi, NULL, &G.fence), "fence");
    VKCHECK(vkCreateFence(G.dev, &fi, NULL, &G.eg_fence), "eg fence");
    VKCHECK(vkCreateFence(G.dev, &fi, NULL, &G.fence_pp[0]), "fence pp0");
    VKCHECK(vkCreateFence(G.dev, &fi, NULL, &G.fence_pp[1]), "fence pp1");
    G.pp_inflight[0] = G.pp_inflight[1] = 0;
    G.pp_layer[0] = G.pp_layer[1] = -1;
    G.pp_rec = 0; G.pp_slot = 0; G.moe_ix_pp = 0; G.pp_pool = VK_NULL_HANDLE;
    G.cmd_rec = G.cmd;
    G.ix_hist_pending = 0; G.moe_ix_rt_bound = 0;
    G.eg_sem_armed = 0; G.stream_live = 0; G.stream_D = 0;
    G.eg_pipe_cmd_ready = 0; G.eg_pipe_hits = 0; G.eg_pipe_misses = 0;
    G.eg_pipe_cache_on = g_qwen_opts.eg_cache != 0;
    memset(G.route_gdn_ready, 0, sizeof(G.route_gdn_ready));
    memset(G.route_gdn_fused, 0, sizeof(G.route_gdn_fused));
    G.route_gdn_key_stream = VK_NULL_HANDLE;
    G.route_gdn_key_x = VK_NULL_HANDLE;
    G.route_gdn_key_idx = VK_NULL_HANDLE;
    G.route_cache_hits = 0; G.route_cache_misses = 0;
    G.final_nw = VK_NULL_HANDLE; G.final_nwm = VK_NULL_HANDLE;
    G.final_nwp = NULL; G.final_nD = 0;

    G.ready = 1;
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(G.phys, &p);
    /* --vk-prof timestamp pairs: 2 per ping-pong slot + 2 for the sync route CB. */
    if (g_qwen_opts.vk_prof && p.limits.timestampPeriod > 0.0f) {
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = DP_TSQ_AM + DP_TS_N};
        if (vkCreateQueryPool(G.dev, &qi, NULL, &G.tsq) != VK_SUCCESS) G.tsq = VK_NULL_HANDLE;
        else G.ts_period_ms = (double)p.limits.timestampPeriod / 1e6;
    }
    G.ts_base_rec = -1; G.pp_tsq[0] = G.pp_tsq[1] = -1;
    fprintf(stderr, "[VK] ready: %s, compute qfam %u, memtype %u%s%s%s\n", p.deviceName, G.qfam, G.memtype,
            G.shader_gu ? ", fused gate+up" : "", G.shader_att ? ", absorb attention" : "",
            G.shader_attf ? ", flash absorb" : "");
    if (G.shader_gdn) fprintf(stderr, "[VK] GatedDeltaNet delta-rule kernel loaded\n");
    if (G.shader_gqa) fprintf(stderr, "[VK] GQA decode attention kernel loaded\n");
    if (G.shader_gdnconv && G.shader_gdncv) fprintf(stderr, "[VK] fused GDN input-block kernels loaded\n");
    if (G.shader_qkv) fprintf(stderr, "[VK] fused GQA input-block kernel loaded\n");
    return 1;
}

int coli_vk_available(void) { return G.ready; }

void coli_vk_mem_info(size_t *used, size_t *count) {
    if (used) *used = G.used_bytes;
    if (count) *count = G.tensor_count;
}

/* Weight-tensor suballocator: many tensors share a few big VkDeviceMemory blocks.
 * WHY: per-submit driver cost measures LINEAR in the number of distinct device-memory
 * objects the queue actively references (~0.35 ms/submit extra at a 950-expert tier's
 * ~5.7k allocations: decode attention 7.9s @420 -> 17.6s @950, flat once the GPU paths
 * moved to CPU; IDLE allocations cost nothing until first referenced — harness ballast
 * probe). Packing tier+dense uploads into 256 MB arenas keeps the referenced-BO count
 * in the dozens. Arena slices are never reclaimed per-tensor (registry/dense uploads
 * live for the process; the rare fill-failure free leaks its slice, bounded) — a
 * tensor's mem handle stays VK_NULL_HANDLE, which coli_vk_tensor_free's vkFreeMemory
 * treats as the documented no-op. */
typedef struct VkWArena { VkDeviceMemory mem; uint8_t *base; size_t cap, off; struct VkWArena *next; } VkWArena;
static VkWArena *g_warena;
#define VK_WARENA_BLOCK ((size_t)256 << 20)
static int arena_suballoc(size_t bytes, VkBuffer *buf, void **ptr) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(G.dev, &bi, NULL, buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(G.dev, *buf, &req);
    if (!(req.memoryTypeBits & (1u << G.memtype))) { vkDestroyBuffer(G.dev, *buf, NULL); *buf = VK_NULL_HANDLE; return 0; }
    size_t align = req.alignment ? req.alignment : 256, off = 0;
    VkWArena *a = g_warena;
    for (; a; a = a->next) {
        off = (a->off + align - 1) & ~(align - 1);
        if (off + req.size <= a->cap) break;
    }
    if (!a) {
        size_t cap = req.size > VK_WARENA_BLOCK ? (req.size + 4095) & ~(size_t)4095 : VK_WARENA_BLOCK;
        a = calloc(1, sizeof(*a));
        if (!a) { vkDestroyBuffer(G.dev, *buf, NULL); *buf = VK_NULL_HANDLE; return 0; }
        VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = cap, .memoryTypeIndex = G.memtype};
#ifdef VK_EXT_memory_priority
        VkMemoryPriorityAllocateInfoEXT pri = {.sType = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT,
            .priority = G.prio};
        if (G.has_prio) ai.pNext = &pri;
#endif
        if (vkAllocateMemory(G.dev, &ai, NULL, &a->mem) != VK_SUCCESS ||
            vkMapMemory(G.dev, a->mem, 0, cap, 0, (void **)&a->base) != VK_SUCCESS) {
            if (a->mem) vkFreeMemory(G.dev, a->mem, NULL);
            free(a); vkDestroyBuffer(G.dev, *buf, NULL); *buf = VK_NULL_HANDLE; return 0;
        }
        a->cap = cap; a->next = g_warena; g_warena = a;
        off = 0;
    }
    VKCHECK(vkBindBufferMemory(G.dev, *buf, a->mem, off), "vkBindBufferMemory");
    if (ptr) *ptr = a->base + off;
    a->off = off + req.size;
    return 1;
}

static int upload_tensor(ColiVkTensor **out, const void *weights, const float *scales,
                         int fmt, int I, int O, int gs) {
    if (*out) return (*out)->fmt == fmt && (*out)->I == I && (*out)->O == O;
    if (fmt != 1 && fmt != 2 && fmt != 5 &&
        !((fmt == 4 || fmt == 6) && gs >= 8 && gs % 8 == 0)) return 0;   /* fmt=4/6: word-aligned groups only */
    ColiVkTensor *t = calloc(1, sizeof(*t));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->rowWords = rowwords(fmt, I); t->gs = (fmt == 4 || fmt == 6) ? gs : 0;
    size_t stride = (size_t)t->rowWords * 4;         // padded row bytes
    size_t cpu_rb = fmt == 1 ? (size_t)I
                  : fmt == 5 ? ((size_t)I + 63) / 64 * 24 : (size_t)(I + 1) / 2;
    size_t sfl = scale_floats(fmt, I, O, gs);            // fmt=5: O*ceil(I/64) group scales
    t->wbytes = stride * (size_t)O;
    void *wptr;
    if (!arena_suballoc(t->wbytes, &t->wbuf, &wptr)) { free(t); return 0; }
    memset(wptr, 0, t->wbytes);
    for (int o = 0; o < O; o++)                        // copy row-by-row into padded layout
        memcpy((uint8_t *)wptr + (size_t)o * stride,
               (const uint8_t *)weights + (size_t)o * cpu_rb, cpu_rb);
    void *sptr;
    if (!arena_suballoc(sfl * sizeof(float), &t->sbuf, &sptr)) {
        vkDestroyBuffer(G.dev, t->wbuf, NULL); free(t); return 0;
    }
    memcpy(sptr, scales, sfl * sizeof(float));
    // Counters are touched concurrently: frees run from expert_load under
    // `#pragma omp parallel`, so RMW them atomically (torn counts otherwise).
    __atomic_add_fetch(&G.used_bytes, t->wbytes + sfl * sizeof(float), __ATOMIC_RELAXED);
    __atomic_add_fetch(&G.tensor_count, 1, __ATOMIC_RELAXED);
    *out = t;
    return 1;
}

/* Upload a resident tensor without computing (for the expert tier: gate/up/down are
 * uploaded once, then driven by coli_vk_expert_group). Returns 0 on failure. */
int coli_vk_tensor_ensure(ColiVkTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int grp) {
    if (!G.ready) return 0;
    return upload_tensor(tensor, weights, scales, fmt, I, O, grp);
}

/* Sync-path fence wait. A blocked vkWaitForFences pays a scheduler wake on
 * signal (~50-150 us) — and the engine fences ~2 sync submits per layer per
 * token, so the wakes alone cost seconds per run. Spin on vkGetFenceStatus for
 * a short budget first (the common decode dispatch completes in 0.5-2 ms),
 * then fall back to the blocking wait. The spinning thread is stalled on the
 * GPU result anyway. COLI_VK_SPIN_US=0 restores the pure blocking wait. */
static long g_vk_spin_us = -1;
static VkResult vk_fence_wait(VkFence f) {
    if (g_vk_spin_us < 0) {
        g_vk_spin_us = g_qwen_opts.spin_us;
        if (g_vk_spin_us < 0) g_vk_spin_us = 0;
    }
    if (g_vk_spin_us > 0) {
        double t0 = vk_now();
        do {
            VkResult r = vkGetFenceStatus(G.dev, f);
            if (r != VK_NOT_READY) return r;   /* VK_SUCCESS or a real error */
        } while ((vk_now() - t0) * 1000.0 < (double)g_vk_spin_us);
    }
    return vkWaitForFences(G.dev, 1, &f, VK_TRUE, 10000000000ULL);
}

/* Loud variant for the sync paths that previously dropped the error on the floor:
 * a failed wait disables the GPU for the rest of the run, so say WHY (the -4
 * device-lost vs 2 timeout distinction is the whole diagnosis). */
static VkResult vk_fence_wait_loud(VkFence f, const char *who) {
    VkResult r = vk_fence_wait(f);
    if (r != VK_SUCCESS)
        fprintf(stderr, "[VK] %s: fence wait failed: %d — disabling GPU offload, staying on CPU\n", who, r);
    return r;
}

/* Decode-path host accounting (--vk-prof). The ping-pong path keeps the CPU a
 * layer ahead of the GPU, so what matters is the split between "blocked on a
 * fence" (device-bound: the remaining headroom is bandwidth) and "recording +
 * writing descriptors + submitting" (host-bound: worth attacking on the CPU). */
static double g_dp_wait_ms, g_dp_sub_ms; static long g_dp_sub_n, g_dp_wait_n;
static double g_dp_gpu_ms, g_dp_am_ms; static long g_dp_gpu_n, g_dp_am_n;
static double dp_t0(void) { return g_qwen_opts.vk_prof ? vk_now() : 0.0; }

static double g_seg_ms[DP_TS_N - 1];
static void dp_ts_begin(VkCommandBuffer cb, int base) {
    G.ts_base_rec = -1;
    if (!G.tsq || !g_qwen_opts.vk_prof || base < 0) return;
    G.ts_base_rec = base;
    vkCmdResetQueryPool(cb, G.tsq, (uint32_t)base, DP_TS_N);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, G.tsq, (uint32_t)base);
}
/* Stage boundary k (1..DP_TS_N-1). Unwritten marks just read back as unavailable. */
static void dp_ts_mark(VkCommandBuffer cb, int k) {
    if (G.ts_base_rec < 0) return;
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, G.tsq, (uint32_t)(G.ts_base_rec + k));
}
static void dp_ts_end(VkCommandBuffer cb) { dp_ts_mark(cb, DP_TS_N - 1); }
/* Call only after the CB's fence has signalled, so the queries are available. */
static double dp_ts_read(int base) {
    /* Argmax CB only writes timestamps at [base] and [base+DP_TS_N-1]. */
    if (!G.tsq || base < 0 || !g_qwen_opts.vk_prof) return 0.0;
    uint64_t t[DP_TS_N];
    if (vkGetQueryPoolResults(G.dev, G.tsq, (uint32_t)base, DP_TS_N, sizeof(t), t, sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return 0.0;
    if (t[DP_TS_N - 1] <= t[0]) return 0.0;
    return (double)(t[DP_TS_N - 1] - t[0]) * G.ts_period_ms;
}
static void dp_ts_collect(int base) {
    if (!G.tsq || base < 0 || !g_qwen_opts.vk_prof) return;
    uint64_t t[DP_TS_N];
    if (vkGetQueryPoolResults(G.dev, G.tsq, (uint32_t)base, DP_TS_N, sizeof(t), t, sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return;
    if (t[DP_TS_N - 1] <= t[0]) return;
    g_dp_gpu_ms += (double)(t[DP_TS_N - 1] - t[0]) * G.ts_period_ms;
    g_dp_gpu_n++;
    for (int k = 1; k < DP_TS_N; k++)
        if (t[k] > t[k - 1]) g_seg_ms[k - 1] += (double)(t[k] - t[k - 1]) * G.ts_period_ms;
}
static void dp_wait_add(double t0) {
    if (g_qwen_opts.vk_prof) { g_dp_wait_ms += vk_now() - t0; g_dp_wait_n++; }
}
static void dp_sub_add(double t0) {
    if (g_qwen_opts.vk_prof) { g_dp_sub_ms += vk_now() - t0; g_dp_sub_n++; }
}

void coli_vk_decode_prof(double decode_ms, int tokens) {
    if (!g_qwen_opts.vk_prof || !g_dp_sub_n) return;
    double other = decode_ms - g_dp_wait_ms - g_dp_sub_ms;
    fprintf(stderr,
        "[VK_PROF] decode %.0f ms / %d tok | fence wait %.0f ms (%.1f%%, n=%ld) | "
        "submit %.0f ms (%.1f%%, n=%ld) | record+desc+host %.0f ms (%.1f%%)\n",
        decode_ms, tokens, g_dp_wait_ms, 100.0 * g_dp_wait_ms / decode_ms, g_dp_wait_n,
        g_dp_sub_ms, 100.0 * g_dp_sub_ms / decode_ms, g_dp_sub_n,
        other, 100.0 * other / decode_ms);
    if (tokens > 0)
        fprintf(stderr, "[VK_PROF] per token: %.1f submits, wait %.2f ms, submit %.2f ms, host %.2f ms\n",
                (double)g_dp_sub_n / tokens, g_dp_wait_ms / tokens,
                g_dp_sub_ms / tokens, other / tokens);
    /* GPU busy vs wall: the gap is bubbles between submits (device idle while the
     * queue is empty), which is what merging CBs into fewer submits would buy. */
    if (g_dp_gpu_n) {
        double gpu = g_dp_gpu_ms + g_dp_am_ms;
        fprintf(stderr, "[VK_PROF] on device: route %.0f ms over %ld CBs (%.3f ms/CB) + lm_head/argmax "
                        "%.0f ms over %ld (%.3f ms) = %.1f%% of decode — bubble %.0f ms (%.1f%%)\n",
                g_dp_gpu_ms, g_dp_gpu_n, g_dp_gpu_ms / g_dp_gpu_n,
                g_dp_am_ms, g_dp_am_n, g_dp_am_n ? g_dp_am_ms / g_dp_am_n : 0.0,
                100.0 * gpu / decode_ms, decode_ms - gpu, 100.0 * (decode_ms - gpu) / decode_ms);
        fprintf(stderr, "[VK_PROF] route CB stages: attn %.0f ms (%.1f%%) | tail rmsnorm+router+topk %.0f ms (%.1f%%) | "
                        "moe_ix %.0f ms (%.1f%%)\n",
                g_seg_ms[0], 100.0 * g_seg_ms[0] / g_dp_gpu_ms,
                g_seg_ms[1], 100.0 * g_seg_ms[1] / g_dp_gpu_ms,
                g_seg_ms[2], 100.0 * g_seg_ms[2] / g_dp_gpu_ms);
    }
}

/* Global submit/wait totals across EVERY synchronous GPU path (VK_PROF=1) — the
 * per-path counters miss traffic that flows through the fused pair/absorb/group
 * entries, so the tier-size-linear per-submit tax is localized here instead. */
static double g_vsub_ms, g_vwait_ms; static long g_vsub_n;
static void vkprof_tick(void) {
    if ((++g_vsub_n & 2047) == 0)
        fprintf(stderr, "[VK_PROF sub] n=%ld | submit %.0f | wait %.0f ms\n", g_vsub_n, g_vsub_ms, g_vwait_ms);
}

int coli_vk_matmul(ColiVkTensor **tensor, float *y, const float *x,
                   const void *weights, const float *scales,
                   int fmt, int S, int I, int O, int gs) {
    if (!G.ready || S < 1 || !upload_tensor(tensor, weights, scales, fmt, I, O, gs)) return 0;
    ColiVkTensor *t = *tensor;
    /* VK_PROF=1: phase split of the dense per-call cost, printed every 8192 calls —
     * separates our code (memcpy/desc/record) from the driver (submit) and the GPU
     * (fence wait) to localize the tier-size-linear tax. */
    static double p_x, p_desc, p_rec, p_sub, p_wait, p_y; static long p_n;
    double t0 = G.eg_prof ? vk_now() : 0, tA;
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    VkBuffer old_x = G.x.buf, old_y = G.y.buf;
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve_mt(&G.y, yb, G.memtype_cached)) return 0;  /* y read back */
    memcpy(G.x.ptr, x, xb);
    if (G.eg_prof) { tA = vk_now(); p_x += tA - t0; t0 = tA; }

    /* Rebind descriptors only when the tensor or a scratch buffer changed (a realloc
     * makes the old VkBuffer handle stale); otherwise the previous binding is still valid. */
    int rebind = G.bound_tensor != t || G.x.buf != old_x || G.y.buf != old_y
              || G.bound_xbuf != G.x.buf || G.bound_ybuf != G.y.buf;
    if (rebind) {
        VkDescriptorBufferInfo bi[4] = {
            {.buffer = G.x.buf, .range = VK_WHOLE_SIZE},
            {.buffer = t->wbuf, .range = VK_WHOLE_SIZE},
            {.buffer = t->sbuf, .range = VK_WHOLE_SIZE},
            {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
        VkWriteDescriptorSet w[4];
        for (int i = 0; i < 4; i++) w[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = G.dset,
            .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
        vkUpdateDescriptorSets(G.dev, 4, w, 0, NULL);
        G.bound_tensor = t; G.bound_xbuf = G.x.buf; G.bound_ybuf = G.y.buf;
    }
    if (G.eg_prof) { tA = vk_now(); p_desc += tA - t0; t0 = tA; }

    /* Re-record the command buffer only when the binding or the dispatch shape changed.
     * Recorded WITHOUT one-time-submit so the same buffer can be resubmitted verbatim —
     * for repeated calls to the same expert this drops setup to a bare submit+wait. */
    if (rebind || !G.cmd_ready || G.bound_S != S || G.bound_I != I || G.bound_O != O) {
        VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
        VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
        vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
        vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
        struct PC pc = {fmt, S, I, O, t->rowWords, t->gs};
        vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        /* Grid-stride shader: one subgroup per output row (~8 rows/workgroup at wave32).
         * Launch ~O/8 workgroups for occupancy; the shader loops to cover any O / wave width. */
        vkCmdDispatch(G.cmd, (uint32_t)((O + 7) / 8), (uint32_t)S, 1);
        VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");
        G.cmd_ready = 1; G.bound_S = S; G.bound_I = I; G.bound_O = O;
    }
    if (G.eg_prof) { tA = vk_now(); p_rec += tA - t0; t0 = tA; }

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { tA = vk_now(); p_sub += tA - t0; g_vsub_ms += tA - t0; t0 = tA; }
    // Bounded wait: a GPU hang/TDR must never wedge the process. 10s is orders of
    // magnitude over a single-GEMV dispatch; on timeout/device-loss disable VK for
    // the rest of the run and fall back to CPU (the caller degrades on our 0 return).
    VkResult wr = vk_fence_wait(G.fence);
    if (wr != VK_SUCCESS) {
        fprintf(stderr, "[VK] fence wait failed: %d — disabling GPU offload, staying on CPU\n", wr);
        G.ready = 0;
        return 0;
    }
    if (G.eg_prof) { tA = vk_now(); p_wait += tA - t0; g_vwait_ms += tA - t0; t0 = tA; vkprof_tick(); }
    memcpy(y, G.y.ptr, yb);
    if (G.eg_prof) {
        p_y += vk_now() - t0;
        if ((++p_n & 8191) == 0)
            fprintf(stderr, "[VK_PROF dense] n=%ld | memcpy_x %.0f | desc %.0f | record %.0f | submit %.0f | wait %.0f | memcpy_y %.0f ms\n",
                    p_n, p_x, p_desc, p_rec, p_sub, p_wait, p_y);
    }
    return 1;
}

/* Fused first half of the expert MLP: hidden = silu(gate(x)) * up(x), computed in ONE
 * dispatch that reads x once for both projections. gate/up are resident (uploaded on
 * first call). D = input (hidden) dim, I = moe_inter. Returns 0 -> caller falls back. */
int coli_vk_gate_up(ColiVkTensor **gate, ColiVkTensor **up, float *hidden, const float *x,
                    const void *gw, const float *gs, const void *uw, const float *us,
                    int fmt, int S, int D, int I, int grp) {
    if (!G.ready || !G.shader_gu || S < 1 || D > 6144) return 0;   /* shader stages x in xsh[6144] */
    if (!upload_tensor(gate, gw, gs, fmt, D, I, grp) || !upload_tensor(up, uw, us, fmt, D, I, grp)) return 0;
    ColiVkTensor *tg = *gate, *tu = *up;
    size_t xb = (size_t)S * D * sizeof(float), hb = (size_t)S * I * sizeof(float);
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve_mt(&G.h, hb, G.memtype_cached)) return 0;  /* hidden read back */
    memcpy(G.x.ptr, x, xb);

    VkDescriptorBufferInfo bi[6] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = tg->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tg->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = tu->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tu->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.h.buf, .range = VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[6];
    for (int i = 0; i < 6; i++) w[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = G.dset_gu,
        .dstBinding = (uint32_t)i, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    vkUpdateDescriptorSets(G.dev, 6, w, 0, NULL);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gu);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &G.dset_gu, 0, NULL);
    struct PC pc = {fmt, S, D, I, tg->rowWords, tg->gs};   // PC.I = input D, PC.O = moe_inter I
    vkCmdPushConstants(G.cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)((I + 7) / 8), (uint32_t)S, 1);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double vp0 = G.eg_prof ? vk_now() : 0;
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { double vp1 = vk_now(); g_vsub_ms += vp1 - vp0; vp0 = vp1; }
    if (vk_fence_wait_loud(G.fence, "gate_up") != VK_SUCCESS) { G.ready = 0; return 0; }
    if (G.eg_prof) { g_vwait_ms += vk_now() - vp0; vkprof_tick(); }
    memcpy(hidden, G.h.ptr, hb);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}

static void wr_desc(VkDescriptorSet set, int n, const VkDescriptorBufferInfo *bi) {
    VkWriteDescriptorSet w[8];
    for (int i = 0; i < n; i++) w[i] = (VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = (uint32_t)i,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    vkUpdateDescriptorSets(G.dev, (uint32_t)n, w, 0, NULL);
}

/* lm_head + greedy pick in ONE submit, with the logit vector never leaving the device:
 * matmul into the GPU-only am_y, then two argmax dispatches fold it to a single winner
 * that costs 8 bytes to read back instead of vocab*4 (and saves the host the scan).
 * Returns 0 -> caller falls back to coli_vk_matmul + its own scan. */
int coli_vk_matmul_argmax(ColiVkTensor **tensor, const void *weights, const float *scales,
                         int fmt, int I, int O, int gs, const float *x, int *idx, float *val) {
    if (!G.ready || !G.pipe_am || O < 1 ||
        !upload_tensor(tensor, weights, scales, fmt, I, O, gs)) return 0;
    ColiVkTensor *t = *tensor;
    size_t pb = AM_GRP * 4 < 256 ? 256 : AM_GRP * 4;
    if (!scratch_reserve(&G.x, (size_t)I * 4) ||
        !scratch_reserve(&G.am_y, (size_t)O * 4) ||                       /* GPU-only logits */
        !scratch_reserve_mt(&G.am_pi, pb, G.memtype_cached) ||            /* winner reads back */
        !scratch_reserve_mt(&G.am_pv, pb, G.memtype_cached)) return 0;
    memcpy(G.x.ptr, x, (size_t)I * 4);

    VkDescriptorBufferInfo mi[4] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE},    {.buffer = t->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = t->sbuf, .range = VK_WHOLE_SIZE},    {.buffer = G.am_y.buf, .range = VK_WHOLE_SIZE}};
    VkDescriptorBufferInfo ai[3] = {
        {.buffer = G.am_y.buf, .range = VK_WHOLE_SIZE}, {.buffer = G.am_pi.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.am_pv.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset, 4, mi);
    wr_desc(G.dset_am, 3, ai);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    struct PC pc = {fmt, 1, I, O, t->rowWords, t->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)((O + 7) / 8), 1, 1);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_am);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_am, 0, 1, &G.dset_am, 0, NULL);
    for (int stage = 0; stage < 2; stage++) {
        vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        struct PCAm am = {O, stage, AM_GRP};
        vkCmdPushConstants(G.cmd, G.plyt_am, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(am), &am);
        vkCmdDispatch(G.cmd, stage == 0 ? (uint32_t)AM_GRP : 1u, 1, 1);
    }
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    if (vk_fence_wait_loud(G.fence, "argmax") != VK_SUCCESS) { G.ready = 0; return 0; }
    *idx = (int)((const uint32_t *)G.am_pi.ptr)[0];
    if (val) *val = ((const float *)G.am_pv.ptr)[0];
    return 1;
}

/* Full batched expert MLP for `count` experts, hidden staying on-device:
 * for each c, hidden_c = silu(gate_c(x_c))*up_c(x_c) (fused), then y_c = down_c(hidden_c).
 * x/y are packed [sum(rows)*D]; experts are resident VkTensors (gate/up: D->I, down: I->D).
 * Mirrors coli_cuda_expert_group. Split into prepare+submit / take so the caller can
 * overlap the GPU batch with its own CPU share (issue -> CPU rows -> take); the group
 * runs on its OWN command buffer + fence, so in-flight work never collides with the
 * main pipeline (dense matmuls, absorb attention). Returns 0 -> caller falls back. */
static int eg_prepare_submit(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                             ColiVkTensor *const *downs, const int *rows, int count,
                             const float *x) {
    int dbg = g_qwen_opts.eg_dbg != 0;
    if (!G.ready || !G.shader_gu || count < 1 || count > 64) { if (dbg) fprintf(stderr,"[eg] reject: ready=%d shader_gu=%p count=%d\n",G.ready,(void*)G.shader_gu,count); return 0; }
    ColiVkTensor *g0 = gates[0]; if (!g0) { if (dbg) fprintf(stderr,"[eg] reject: g0 NULL\n"); return 0; }
    int D = g0->I, I = g0->O, fmt = g0->fmt, total = 0, off[64];
    if (D > 6144) { if (dbg) fprintf(stderr,"[eg] reject: D=%d>6144\n",D); return 0; }   /* gate_up shader stages x in xsh[6144] */
    int dfmt = downs[0]->fmt;   /* down may be a different quant than gate/up (per-projection
                                 * containers, e.g. --up-bits 3); gate/up must MATCH — the
                                 * fused gate_up shader decodes both with one fmt. */
    for (int c = 0; c < count; c++) {
        off[c] = total; total += rows[c];
        if (rows[c] < 1 || gates[c]->I != D || gates[c]->O != I || gates[c]->fmt != fmt ||
            ups[c]->I != D || ups[c]->O != I || ups[c]->fmt != fmt ||
            downs[c]->I != I || downs[c]->O != D || downs[c]->fmt != dfmt) {
            if (dbg) fprintf(stderr,"[eg] reject c=%d/%d: rows=%d g(I=%d O=%d fmt=%d) u(I=%d O=%d fmt=%d) d(I=%d O=%d fmt=%d) | want D=%d I=%d fmt=%d dfmt=%d\n",
                c,count,rows[c],gates[c]->I,gates[c]->O,gates[c]->fmt,ups[c]->I,ups[c]->O,ups[c]->fmt,downs[c]->I,downs[c]->O,downs[c]->fmt,D,I,fmt,dfmt);
            return 0;
        }
    }
    size_t xb = (size_t)total*D*4, hb = (size_t)total*I*4, yb = (size_t)total*D*4;
    if (!scratch_reserve(&G.eg_x, xb) || !scratch_reserve(&G.eg_h, hb) ||
        !scratch_reserve_mt(&G.eg_y, yb, G.memtype_cached)) return 0;   /* eg_y is read back -> cached */
    G.eg_prof = g_qwen_opts.vk_prof != 0;
    if (G.eg_prof) G.eg_t0 = vk_now();
    memcpy(G.eg_x.ptr, x, xb);
    if (G.eg_prof) G.eg_t1 = vk_now();

    /* DP4A: quantize each activation row to int8 on-device, then let the integer-dot
     * kernels replace the scalar ones. gate_up and down are gated separately because a
     * container may quantize the two projections differently.
     *
     * OPT-IN (COLI_VK_DP4A=1), because it does not pay off as the decode path stands:
     * every expert here gets rows=1 (moe_token runs per token), so both matmuls are
     * mat-VECs reading ~13 MB of int4 weights per group — already at ~40 of the ~48 GB/s
     * this iGPU sustains. DP4A removes ALU work, not bytes, so measured on Qwen3.5-35B-A3B
     * it moved the group from 0.327 to 0.317 ms (3%) and the end-to-end rate from 11.06 to
     * 11.15 tok/s, while int8 activations cost ~2.7% perplexity. It becomes worthwhile only
     * once several tokens' rows share one expert (rows>1 amortizes the weight stream and
     * turns these kernels ALU-bound), which is what batching prefill per expert would do. */
    int dp4a_on = G.pipe_qr && g_qwen_opts.dp4a != 0;
    int dp4a = dp4a_on && fmt == 6 && D <= 6144;         /* gate_up stages x in xsh[1536] */
    int dp4a_dn = dp4a_on && dfmt == 6 && I <= 8192;     /* down stages hidden in xsh[2048] */
    size_t sb_min = (size_t)total * 4 < 256 ? 256 : (size_t)total * 4;
    if (dp4a && !(scratch_reserve(&G.eg_xq, xb / 4) && scratch_reserve(&G.eg_xs, sb_min))) dp4a = 0;
    if (dp4a_dn && !(scratch_reserve(&G.eg_hq, hb / 4) && scratch_reserve(&G.eg_hs, sb_min))) dp4a_dn = 0;

    if (!G.eg_pool) {   /* one-time: 64 scalar gate_up (6) + 64 scalar down (4), plus for the
                         * DP4A path 64 gate_up (7) + 64 down (5) + the x-quantize set (3) */
        int n4 = G.pipe_qr ? 64 : 0, nq = G.pipe_qr ? 1 : 0;
        VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = (uint32_t)(64*6 + 64*4 + n4*7 + n4*5 + nq*3)};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = (uint32_t)(128 + 2*n4 + nq), .poolSizeCount = 1, .pPoolSizes = &ps};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.eg_pool), "eg descPool");
        VkDescriptorSetLayout lg[64], ld[64], lg4[64], ld4[64];
        for (int c = 0; c < 64; c++) { lg[c] = G.dsl_gu; ld[c] = G.dsl; lg4[c] = G.dsl_gu4; ld4[c] = G.dsl_dn4; }
        VkDescriptorSetAllocateInfo ag = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = G.eg_pool, .descriptorSetCount = 64, .pSetLayouts = lg};
        VkDescriptorSetAllocateInfo ad = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = G.eg_pool, .descriptorSetCount = 64, .pSetLayouts = ld};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &ag, G.eg_gu), "eg gu sets");
        VKCHECK(vkAllocateDescriptorSets(G.dev, &ad, G.eg_dn), "eg dn sets");
        if (n4) {
            VkDescriptorSetAllocateInfo ag4 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = G.eg_pool, .descriptorSetCount = 64, .pSetLayouts = lg4};
            VkDescriptorSetAllocateInfo ad4 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = G.eg_pool, .descriptorSetCount = 64, .pSetLayouts = ld4};
            VkDescriptorSetAllocateInfo aq = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = G.eg_pool, .descriptorSetCount = 1, .pSetLayouts = &G.dsl_qr};
            VKCHECK(vkAllocateDescriptorSets(G.dev, &ag4, G.eg_gu4), "eg gu4 sets");
            VKCHECK(vkAllocateDescriptorSets(G.dev, &ad4, G.eg_dn4), "eg dn4 sets");
            VKCHECK(vkAllocateDescriptorSets(G.dev, &aq, &G.dset_qr_x), "eg qr-x set");
        }
        G.eg_nsets = 64;
    }
    for (int c = 0; c < count; c++) {
        VkDeviceSize xo = (VkDeviceSize)off[c]*D*4, ho = (VkDeviceSize)off[c]*I*4, yo = (VkDeviceSize)off[c]*D*4;
        if (dp4a) {
            VkDescriptorBufferInfo g4[7] = {
                {G.eg_xq.buf, 0, VK_WHOLE_SIZE}, {G.eg_xs.buf, 0, VK_WHOLE_SIZE},
                {gates[c]->wbuf, 0, VK_WHOLE_SIZE}, {gates[c]->sbuf, 0, VK_WHOLE_SIZE},
                {ups[c]->wbuf, 0, VK_WHOLE_SIZE}, {ups[c]->sbuf, 0, VK_WHOLE_SIZE},
                {G.eg_h.buf, ho, (VkDeviceSize)rows[c]*I*4}};
            wr_desc(G.eg_gu4[c], 7, g4);
        } else {
            VkDescriptorBufferInfo gi[6] = {
                {G.eg_x.buf, xo, (VkDeviceSize)rows[c]*D*4}, {gates[c]->wbuf, 0, VK_WHOLE_SIZE},
                {gates[c]->sbuf, 0, VK_WHOLE_SIZE}, {ups[c]->wbuf, 0, VK_WHOLE_SIZE},
                {ups[c]->sbuf, 0, VK_WHOLE_SIZE}, {G.eg_h.buf, ho, (VkDeviceSize)rows[c]*I*4}};
            wr_desc(G.eg_gu[c], 6, gi);
        }
        if (dp4a_dn) {
            VkDescriptorBufferInfo d4[5] = {
                {G.eg_hq.buf, 0, VK_WHOLE_SIZE}, {G.eg_hs.buf, 0, VK_WHOLE_SIZE},
                {downs[c]->wbuf, 0, VK_WHOLE_SIZE}, {downs[c]->sbuf, 0, VK_WHOLE_SIZE},
                {G.eg_y.buf, yo, (VkDeviceSize)rows[c]*D*4}};
            wr_desc(G.eg_dn4[c], 5, d4);
            continue;
        }
        VkDescriptorBufferInfo di[4] = {
            {G.eg_h.buf, ho, (VkDeviceSize)rows[c]*I*4}, {downs[c]->wbuf, 0, VK_WHOLE_SIZE},
            {downs[c]->sbuf, 0, VK_WHOLE_SIZE}, {G.eg_y.buf, yo, (VkDeviceSize)rows[c]*D*4}};
        wr_desc(G.eg_dn[c], 4, di);
    }
    /* whole-buffer sets for the row-quantize passes (one dispatch each, all rows at once) */
    if (dp4a) {
        VkDescriptorBufferInfo qx[3] = {{G.eg_x.buf, 0, (VkDeviceSize)xb},
                                        {G.eg_xq.buf, 0, (VkDeviceSize)(xb / 4)},
                                        {G.eg_xs.buf, 0, VK_WHOLE_SIZE}};
        wr_desc(G.dset_qr_x, 3, qx);
    }
    if (dp4a_dn) {
        VkDescriptorBufferInfo qh[3] = {{G.eg_h.buf, 0, (VkDeviceSize)hb},
                                        {G.eg_hq.buf, 0, (VkDeviceSize)(hb / 4)},
                                        {G.eg_hs.buf, 0, VK_WHOLE_SIZE}};
        wr_desc(G.dset_qr_h, 3, qh);
    }
    if (G.eg_prof) G.eg_t2 = vk_now();

    VKCHECK(vkResetCommandBuffer(G.eg_cmd, 0), "eg resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.eg_cmd, &begin), "eg beginCmd");
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    /* phase 0: x -> int8 + per-row scale (all rows in one dispatch) */
    if (dp4a) {
        struct PCQr qr = {total, D};
        vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_qr);
        vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_qr, 0, 1, &G.dset_qr_x, 0, NULL);
        vkCmdPushConstants(G.eg_cmd, G.plyt_qr, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(qr), &qr);
        vkCmdDispatch(G.eg_cmd, (uint32_t)total, 1, 1);
        vkCmdPipelineBarrier(G.eg_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    /* phase 1: fused gate+up+silu -> hidden (per expert, bound to its x/hidden slices) */
    vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dp4a ? G.pipe_gu_dp4a : G.pipe_gu);
    for (int c = 0; c < count; c++) {
        if (dp4a) {
            struct PCDp4a pc = {rows[c], D, I, gates[c]->rowWords, gates[c]->gs, off[c]};
            vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu4, 0, 1, &G.eg_gu4[c], 0, NULL);
            vkCmdPushConstants(G.eg_cmd, G.plyt_gu4, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        } else {
            struct PC pc = {fmt, rows[c], D, I, gates[c]->rowWords, gates[c]->gs};
            vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &G.eg_gu[c], 0, NULL);
            vkCmdPushConstants(G.eg_cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        }
        vkCmdDispatch(G.eg_cmd, (uint32_t)((I + 7) / 8), (uint32_t)rows[c], 1);
    }
    vkCmdPipelineBarrier(G.eg_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    /* phase 1.5: hidden -> int8 + per-row scale, feeding the DP4A down kernel */
    if (dp4a_dn) {
        struct PCQr qr = {total, I};
        vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_qr);
        vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_qr, 0, 1, &G.dset_qr_h, 0, NULL);
        vkCmdPushConstants(G.eg_cmd, G.plyt_qr, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(qr), &qr);
        vkCmdDispatch(G.eg_cmd, (uint32_t)total, 1, 1);
        vkCmdPipelineBarrier(G.eg_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    /* phase 2: down projection hidden -> y */
    vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dp4a_dn ? G.pipe_dn_dp4a : G.pipe);
    for (int c = 0; c < count; c++) {
        if (dp4a_dn) {
            struct PCDp4a pc = {rows[c], I, D, downs[c]->rowWords, downs[c]->gs, off[c]};
            vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_dn4, 0, 1, &G.eg_dn4[c], 0, NULL);
            vkCmdPushConstants(G.eg_cmd, G.plyt_dn4, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        } else {
            struct PC pc = {dfmt, rows[c], I, D, downs[c]->rowWords, downs[c]->gs};
            vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.eg_dn[c], 0, NULL);
            vkCmdPushConstants(G.eg_cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        }
        vkCmdDispatch(G.eg_cmd, (uint32_t)((D + 7) / 8), (uint32_t)rows[c], 1);
    }
    VKCHECK(vkEndCommandBuffer(G.eg_cmd), "eg endCmd");
    if (G.eg_prof) G.eg_t3 = vk_now();

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.eg_cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.eg_fence), "eg resetFence");
    { double vp0 = G.eg_prof ? vk_now() : 0;
      VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.eg_fence), "eg queueSubmit");
      if (G.eg_prof) g_vsub_ms += vk_now() - vp0; }
    G.eg_pending_yb = yb; G.eg_inflight = 1;
    return 1;
}

/* Issue a group asynchronously: submit and return WITHOUT waiting, so the caller
 * computes its CPU share concurrently. Exactly one group may be in flight. */
int coli_vk_expert_group_issue(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                               ColiVkTensor *const *downs, const int *rows, int count,
                               const float *x) {
    if (G.eg_inflight) return 0;
    return eg_prepare_submit(gates, ups, downs, rows, count, x);
}

/* Join the in-flight group and read back the packed outputs. */
int coli_vk_expert_group_take(float *y) {
    if (!G.eg_inflight) return 0;
    G.eg_inflight = 0;
    if (vk_fence_wait(G.eg_fence) != VK_SUCCESS) {
        fprintf(stderr, "[VK] expert-group fence wait failed — disabling GPU offload\n");
        G.ready = 0; return 0;
    }
    double t4 = G.eg_prof ? vk_now() : 0;
    memcpy(y, G.eg_y.ptr, G.eg_pending_yb);
    if (G.eg_prof) {
        double t5 = vk_now();
        fprintf(stderr, "[VK_PROF] memcpy_x %.3f | desc %.3f | record %.3f | issue->take %.3f | memcpy_y %.3f ms\n",
                G.eg_t1-G.eg_t0, G.eg_t2-G.eg_t1, G.eg_t3-G.eg_t2, t4-G.eg_t3, t5-t4);
    }
    return 1;
}

/* Synchronous form (shared expert, harness): issue + take in one call. */
int coli_vk_expert_group(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                         ColiVkTensor *const *downs, const int *rows, int count,
                         float *y, const float *x) {
    if (G.eg_inflight) return 0;
    if (!eg_prepare_submit(gates, ups, downs, rows, count, x)) return 0;
    return coli_vk_expert_group_take(y);
}

/* ---- MLA absorb attention core -------------------------------------------------
 * The KV latent (L, [rows,K]) and rope (R, [rows,Rd]) caches live in persistent
 * per-layer device buffers, appended row-by-row as tokens decode (the host stays
 * canonical; glm.c tracks a valid-watermark and re-appends after invalidation).
 * Rows are indexed by ABSOLUTE position, so kv_start windows just skip rows. */

int coli_vk_kv_ensure(int layer, int max_rows, int K, int Rd) {
    if (!G.ready || layer < 0 || layer >= VK_KV_LAYERS || max_rows < 1 || K < 1 || Rd < 1) return 0;
    VkKvLayer *v = &G.kv[layer];
    if (v->bl) return v->rows >= max_rows && v->K == K && v->R == Rd;  /* resize goes through coli_vk_kv_reset */
    float p0 = G.prio; G.prio = 1.0f;            /* KV mirror rides every attention submit */
    int ok1 = alloc_hostvis((size_t)max_rows * K * 4, &v->bl, &v->ml, &v->pl);
    int ok = ok1 && alloc_hostvis((size_t)max_rows * Rd * 4, &v->br, &v->mr, &v->pr);
    G.prio = p0;
    if (!ok) {
        if (ok1) { vkDestroyBuffer(G.dev, v->bl, NULL); vkFreeMemory(G.dev, v->ml, NULL); }
        memset(v, 0, sizeof(*v)); return 0;
    }
    v->rows = max_rows; v->K = K; v->R = Rd;
    return 1;
}

/* Mirror one host cache row into the device copy (write-combined memory: the CPU
 * only ever WRITES these buffers, the GPU reads them). */
int coli_vk_kv_row(int layer, int pos, const float *L, const float *R) {
    if (layer < 0 || layer >= VK_KV_LAYERS) return 0;
    VkKvLayer *v = &G.kv[layer];
    if (!v->pl || pos < 0 || pos >= v->rows) return 0;
    memcpy((float *)v->pl + (size_t)pos * v->K, L, (size_t)v->K * 4);
    memcpy((float *)v->pr + (size_t)pos * v->R, R, (size_t)v->R * 4);
    return 1;
}

/* Drop all per-layer KV device caches (cache resize in kv_alloc). */
void coli_vk_kv_reset(void) {
    for (int i = 0; i < VK_KV_LAYERS; i++) {
        VkKvLayer *v = &G.kv[i];
        if (!v->bl) continue;
        if (G.ready) {   /* dead device: leak GPU handles like coli_vk_tensor_free */
            vkDestroyBuffer(G.dev, v->bl, NULL); vkFreeMemory(G.dev, v->ml, NULL);
            vkDestroyBuffer(G.dev, v->br, NULL); vkFreeMemory(G.dev, v->mr, NULL);
        }
        memset(v, 0, sizeof(*v));
    }
}

/* ---- GatedDeltaNet delta-rule (Qwen hybrid layers) ------------------------
 * The per-head recurrent state S[VH*KD*VD] is device-resident per GDN layer and
 * persists across decode steps (analogous to the KV mirror, but read-write and
 * never read back). coli_vk_gdn_state_ensure allocates+zeros it once; the state
 * is zeroed again at each new sequence via coli_vk_gdn_reset. */
int coli_vk_gdn_available(void) { return G.ready && G.pipe_gdn ? 1 : 0; }

int coli_vk_gdn_state_ensure(int layer, int VH, int KD, int VD) {
    if (!G.ready || !G.pipe_gdn || layer < 0 || layer >= VK_KV_LAYERS ||
        VH < 1 || KD < 1 || VD < 1) return 0;
    size_t bytes = (size_t)VH * KD * VD * 4;
    if (G.gdn_st[layer]) return G.gdn_bytes[layer] == bytes;   /* resize goes through reset+ensure */
    float p0 = G.prio; G.prio = 1.0f;             /* state rides every GDN submit: never evict */
    int ok = alloc_hostvis(bytes, &G.gdn_st[layer], &G.gdn_stm[layer], &G.gdn_stp[layer]);
    G.prio = p0;
    if (!ok) { G.gdn_st[layer] = VK_NULL_HANDLE; return 0; }
    memset(G.gdn_stp[layer], 0, bytes);           /* fresh recurrent state = 0 */
    G.gdn_bytes[layer] = bytes;
    return 1;
}

/* Zero every GDN layer's recurrent state in place (start of a new sequence). Keeps
 * the buffers allocated — no realloc churn between prompts. */
void coli_vk_gdn_reset(void) {
    for (int i = 0; i < VK_KV_LAYERS; i++) {
        if (G.gdn_stp[i]) memset(G.gdn_stp[i], 0, G.gdn_bytes[i]);
        if (G.gdn_ringp[i]) memset(G.gdn_ringp[i], 0, G.gdn_ring_bytes[i]);   /* fresh conv ring */
    }
}

static void coli_vk_gdn_free(void) {   /* full teardown (shutdown only) */
    for (int i = 0; i < VK_KV_LAYERS; i++) {
        if (G.gdn_st[i]) {
            if (G.ready) { vkDestroyBuffer(G.dev, G.gdn_st[i], NULL); vkFreeMemory(G.dev, G.gdn_stm[i], NULL); }
            G.gdn_st[i] = VK_NULL_HANDLE; G.gdn_stm[i] = VK_NULL_HANDLE; G.gdn_stp[i] = NULL; G.gdn_bytes[i] = 0;
        }
        if (G.gdn_ring[i]) {
            if (G.ready) { vkDestroyBuffer(G.dev, G.gdn_ring[i], NULL); vkFreeMemory(G.dev, G.gdn_ringm[i], NULL); }
            G.gdn_ring[i] = VK_NULL_HANDLE; G.gdn_ringm[i] = VK_NULL_HANDLE; G.gdn_ringp[i] = NULL; G.gdn_ring_bytes[i] = 0;
        }
        if (G.gdn_cw[i]) {
            if (G.ready) { vkDestroyBuffer(G.dev, G.gdn_cw[i], NULL); vkFreeMemory(G.dev, G.gdn_cwm[i], NULL); }
            G.gdn_cw[i] = VK_NULL_HANDLE; G.gdn_cwm[i] = VK_NULL_HANDLE; G.gdn_cwp[i] = NULL;
        }
    }
}

/* One decode step of the GatedDeltaNet delta rule + gated RMSNorm for one GDN
 * layer (S=1). Inputs are host arrays (computed CPU-side: conv, q/k L2-norm,
 * decay, beta); the recurrence + readout + gated norm run on the GPU against the
 * resident state, and y[VH*VD] is read back. Returns 0 -> caller falls back to CPU.
 *   qn/kn: [KH*KD]  v/z: [VH*VD]  decay/beta: [VH]  gnorm: [VD]  y: [VH*VD] */
int coli_vk_gdn(int layer, const float *qn, const float *kn, const float *v, const float *z,
                const float *decay, const float *beta, const float *gnorm, float *y,
                int KH, int KD, int VH, int VD, float eps) {
    if (!G.ready || !G.pipe_gdn || layer < 0 || layer >= VK_KV_LAYERS) return 0;
    if (KH < 1 || KD < 1 || VH < 1 || VD < 1 || VD > 512 || (VH % KH) != 0) return 0;
    if (!G.gdn_st[layer] || G.gdn_bytes[layer] != (size_t)VH * KD * VD * 4) return 0;
    size_t QK = (size_t)KH * KD, HV = (size_t)VH * VD;
    size_t in_n = 2 * QK + 2 * HV + 2 * (size_t)VH + (size_t)VD;
    if (!scratch_reserve(&G.gdn_in, in_n * 4) ||
        !scratch_reserve_mt(&G.gdn_y, HV * 4, G.memtype_cached)) return 0;   /* y read back -> cached */

    float *ip = (float *)G.gdn_in.ptr;            /* pack: qn|kn|v|z|decay|beta|gnorm */
    memcpy(ip,                    qn,    QK * 4);
    memcpy(ip + QK,               kn,    QK * 4);
    memcpy(ip + 2 * QK,           v,     HV * 4);
    memcpy(ip + 2 * QK + HV,      z,     HV * 4);
    memcpy(ip + 2 * QK + 2 * HV,          decay, (size_t)VH * 4);
    memcpy(ip + 2 * QK + 2 * HV + VH,     beta,  (size_t)VH * 4);
    memcpy(ip + 2 * QK + 2 * HV + 2 * VH, gnorm, (size_t)VD * 4);

    VkDescriptorBufferInfo bi[3] = {
        {.buffer = G.gdn_in.buf,     .range = VK_WHOLE_SIZE},
        {.buffer = G.gdn_st[layer],  .range = VK_WHOLE_SIZE},
        {.buffer = G.gdn_y.buf,      .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_gdn, 3, bi);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gdn);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gdn, 0, 1, &G.dset_gdn, 0, NULL);
    struct PCGdn pc = {(uint32_t)KH, (uint32_t)KD, (uint32_t)VH, (uint32_t)VD, eps};
    vkCmdPushConstants(G.cmd, G.plyt_gdn, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)VH, 1, 1);     /* one workgroup per value-head */
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (vk_fence_wait_loud(G.fence, "gdn") != VK_SUCCESS) { G.ready = 0; return 0; }
    memcpy(y, G.gdn_y.ptr, HV * 4);
    G.cmd_ready = 0; G.bound_tensor = NULL;       /* shared cmd/binding clobbered */
    return 1;
}

/* Fused GatedDeltaNet + output projection in ONE submit: the delta-rule kernel
 * writes y[VH*VD] to G.gdn_y, a barrier, then the resident out-projection matmul
 * (qmatmul) reads it and writes out[Dout] — y never round-trips to the host.
 * `ot` is the already-resident out_proj tensor (value_dim=VH*VD -> Dout). */
int coli_vk_gdn_project(int layer, const float *qn, const float *kn, const float *v, const float *z,
                        const float *decay, const float *beta, const float *gnorm,
                        ColiVkTensor *ot, float *out, int KH, int KD, int VH, int VD,
                        float eps, int Dout) {
    if (!G.ready || !G.pipe_gdn || !ot || layer < 0 || layer >= VK_KV_LAYERS) return 0;
    if (KH < 1 || KD < 1 || VH < 1 || VD < 1 || VD > 512 || (VH % KH) != 0 || Dout < 1) return 0;
    if (!G.gdn_st[layer] || G.gdn_bytes[layer] != (size_t)VH * KD * VD * 4) return 0;
    if (ot->I != VH * VD || ot->O != Dout) return 0;
    size_t QK = (size_t)KH * KD, HV = (size_t)VH * VD;
    size_t in_n = 2 * QK + 2 * HV + 2 * (size_t)VH + (size_t)VD, ob = (size_t)Dout * 4;
    if (!scratch_reserve(&G.gdn_in, in_n * 4) || !scratch_reserve(&G.gdn_y, HV * 4) ||
        !scratch_reserve_mt(&G.y, ob, G.memtype_cached)) return 0;   /* gdn_y GPU-only here; out read back */

    float *ip = (float *)G.gdn_in.ptr;            /* pack: qn|kn|v|z|decay|beta|gnorm */
    memcpy(ip,                    qn,    QK * 4);
    memcpy(ip + QK,               kn,    QK * 4);
    memcpy(ip + 2 * QK,           v,     HV * 4);
    memcpy(ip + 2 * QK + HV,      z,     HV * 4);
    memcpy(ip + 2 * QK + 2 * HV,          decay, (size_t)VH * 4);
    memcpy(ip + 2 * QK + 2 * HV + VH,     beta,  (size_t)VH * 4);
    memcpy(ip + 2 * QK + 2 * HV + 2 * VH, gnorm, (size_t)VD * 4);

    VkDescriptorBufferInfo bi[3] = {
        {.buffer = G.gdn_in.buf, .range = VK_WHOLE_SIZE}, {.buffer = G.gdn_st[layer], .range = VK_WHOLE_SIZE},
        {.buffer = G.gdn_y.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_gdn, 3, bi);
    VkDescriptorBufferInfo oi[4] = {                 /* out-proj: x=gdn_y | W | scales | out */
        {.buffer = G.gdn_y.buf, .range = VK_WHOLE_SIZE}, {.buffer = ot->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = ot->sbuf, .range = VK_WHOLE_SIZE},    {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset, 4, oi);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gdn);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gdn, 0, 1, &G.dset_gdn, 0, NULL);
    struct PCGdn pc = {(uint32_t)KH, (uint32_t)KD, (uint32_t)VH, (uint32_t)VD, eps};
    vkCmdPushConstants(G.cmd, G.plyt_gdn, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)VH, 1, 1);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    struct PC opc = {ot->fmt, 1, VH * VD, Dout, ot->rowWords, ot->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(opc), &opc);
    vkCmdDispatch(G.cmd, (uint32_t)((Dout + 7) / 8), 1, 1);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (vk_fence_wait_loud(G.fence, "gdn_project") != VK_SUCCESS) { G.ready = 0; return 0; }
    memcpy(out, G.y.ptr, ob);
    G.cmd_ready = 0; G.bound_tensor = NULL;
    return 1;
}

int coli_vk_gdn_full_available(void) {
    return G.ready && G.pipe_gdnconv && G.pipe_gdncv && G.pipe_gdn && G.pipe ? 1 : 0;
}

/* Device-resident conv1d ring (per GDN layer), zeroed = fresh sequence. */
int coli_vk_gdn_conv_ensure(int layer, int conv_k, int conv_dim) {
    if (!G.ready || layer < 0 || layer >= VK_KV_LAYERS || conv_k < 1 || conv_k > 16 || conv_dim < 1) return 0;
    size_t bytes = (size_t)conv_k * conv_dim * 4;
    if (G.gdn_ring[layer]) return G.gdn_ring_bytes[layer] == bytes;
    float p0 = G.prio; G.prio = 1.0f;
    int ok = alloc_hostvis(bytes, &G.gdn_ring[layer], &G.gdn_ringm[layer], &G.gdn_ringp[layer]);
    G.prio = p0;
    if (!ok) { G.gdn_ring[layer] = VK_NULL_HANDLE; return 0; }
    memset(G.gdn_ringp[layer], 0, bytes);
    G.gdn_ring_bytes[layer] = bytes;
    return 1;
}

/* Upload one GDN layer's conv weights [conv_dim, conv_k] resident (once). */
int coli_vk_gdn_conv_weight(int layer, const float *cw, int conv_k, int conv_dim) {
    if (!G.ready || layer < 0 || layer >= VK_KV_LAYERS) return 0;
    size_t bytes = (size_t)conv_dim * conv_k * 4;
    if (!G.gdn_cw[layer]) {
        float p0 = G.prio; G.prio = 1.0f;
        int ok = alloc_hostvis(bytes, &G.gdn_cw[layer], &G.gdn_cwm[layer], &G.gdn_cwp[layer]);
        G.prio = p0;
        if (!ok) { G.gdn_cw[layer] = VK_NULL_HANDLE; return 0; }
    }
    memcpy(G.gdn_cwp[layer], cw, bytes);
    return 1;
}

/* Whole GatedDeltaNet block for one token in ONE submit — no host round-trip between ops:
 *   gqkv/gz matmul (x -> qkv,z) -> conv1d+ring -> qknorm+delta+gatednorm -> out-proj.
 * Keeps qkv/z/cv/y in GPU-only scratch. decay/beta computed CPU-side (from x, cheap) and
 * gnorm are packed into `params` [decay(VH)|beta(VH)|gnorm(VD)]. gqkv_t/gz_t/out_t are the
 * resident dense tensors. out[Dout] read back. Returns 0 -> caller falls back. */
int coli_vk_gdn_full(int layer, const float *x, int D, ColiVkTensor *gqkv_t, ColiVkTensor *gz_t,
                     const float *params, ColiVkTensor *out_t, float *out,
                     int KH, int KD, int VH, int VD, int conv_dim, int conv_k, float eps, int Dout) {
    if (!coli_vk_gdn_full_available() || !gqkv_t || !gz_t || !out_t) return 0;
    if (layer < 0 || layer >= VK_KV_LAYERS || VD > 512 || KD > 256 || (VH % KH) != 0) return 0;
    if (!G.gdn_st[layer] || !G.gdn_ring[layer] || !G.gdn_cw[layer]) return 0;
    int key_dim = KH * KD, value_dim = VH * VD;
    if (gqkv_t->I != D || gqkv_t->O != conv_dim || gz_t->I != D || gz_t->O != value_dim ||
        out_t->I != value_dim || out_t->O != Dout || conv_dim != 2 * key_dim + value_dim) return 0;
    size_t xb = (size_t)D * 4, prb = (size_t)(2 * VH + VD) * 4;
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve(&G.gdnf_qkv, (size_t)conv_dim * 4) ||
        !scratch_reserve(&G.gdnf_z, (size_t)value_dim * 4) || !scratch_reserve(&G.gdnf_cv, (size_t)conv_dim * 4) ||
        !scratch_reserve(&G.gdn_y, (size_t)value_dim * 4) || !scratch_reserve(&G.gdnf_pr, prb) ||
        !scratch_reserve_mt(&G.y, (size_t)Dout * 4, G.memtype_cached)) return 0;
    memcpy(G.x.ptr, x, xb);
    memcpy(G.gdnf_pr.ptr, params, prb);

    VkDescriptorBufferInfo bqkv[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gqkv_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gqkv_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gdnf_qkv.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gdnf_gqkv, 4, bqkv);
    VkDescriptorBufferInfo bz[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gz_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gz_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gdnf_z.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gdnf_gz, 4, bz);
    VkDescriptorBufferInfo bc[4] = {{.buffer=G.gdnf_qkv.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gdn_cw[layer],.range=VK_WHOLE_SIZE},{.buffer=G.gdn_ring[layer],.range=VK_WHOLE_SIZE},{.buffer=G.gdnf_cv.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.dset_gdnconv, 4, bc);
    VkDescriptorBufferInfo bd[5] = {{.buffer=G.gdnf_cv.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gdnf_z.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gdnf_pr.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gdn_st[layer],.range=VK_WHOLE_SIZE},{.buffer=G.gdn_y.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.dset_gdncv, 5, bd);
    VkDescriptorBufferInfo bo[4] = {{.buffer=G.gdn_y.buf,.range=VK_WHOLE_SIZE},{.buffer=out_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=out_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.y.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gdnf_out, 4, bo);

    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    /* 1) gqkv + gz matmuls (x -> qkv, z) */
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gdnf_gqkv, 0, NULL);
    struct PC pq = {gqkv_t->fmt, 1, D, conv_dim, gqkv_t->rowWords, gqkv_t->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pq), &pq);
    vkCmdDispatch(G.cmd, (uint32_t)((conv_dim + 7) / 8), 1, 1);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gdnf_gz, 0, NULL);
    struct PC pz = {gz_t->fmt, 1, D, value_dim, gz_t->rowWords, gz_t->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pz), &pz);
    vkCmdDispatch(G.cmd, (uint32_t)((value_dim + 7) / 8), 1, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    /* 2) conv1d + ring update -> cv */
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gdnconv);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gdnconv, 0, 1, &G.dset_gdnconv, 0, NULL);
    struct PCConv pcv = {conv_dim, conv_k};
    vkCmdPushConstants(G.cmd, G.plyt_gdnconv, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcv), &pcv);
    vkCmdDispatch(G.cmd, (uint32_t)((conv_dim + 255) / 256), 1, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    /* 3) qknorm + delta + gated RMSNorm -> y */
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gdncv);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gdncv, 0, 1, &G.dset_gdncv, 0, NULL);
    struct PCGdnCv pd = {(uint32_t)KH, (uint32_t)KD, (uint32_t)VH, (uint32_t)VD, (uint32_t)key_dim, eps};
    vkCmdPushConstants(G.cmd, G.plyt_gdncv, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pd), &pd);
    vkCmdDispatch(G.cmd, (uint32_t)VH, 1, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    /* 4) out projection (y -> out) */
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gdnf_out, 0, NULL);
    struct PC po = {out_t->fmt, 1, value_dim, Dout, out_t->rowWords, out_t->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(po), &po);
    vkCmdDispatch(G.cmd, (uint32_t)((Dout + 7) / 8), 1, 1);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (vk_fence_wait_loud(G.fence, "gdn_full") != VK_SUCCESS) { G.ready = 0; return 0; }
    memcpy(out, G.y.ptr, (size_t)Dout * 4);
    G.cmd_ready = 0; G.bound_tensor = NULL;
    return 1;
}

/* Prefill GDN: pack up to GDN_SEQ_CHUNK tokens into one CB. Recurrent state stays
 * sequential via barriers; fences fall from one-per-token to one-per-chunk.
 * Descriptors stay fixed on single-token work buffers; each token is staged in via
 * vkCmdCopyBuffer from the host-uploaded batch (updating a bound set mid-CB would
 * make every dispatch see the last token's offsets). */
#define GDN_SEQ_CHUNK 64

int coli_vk_gdn_full_seq(int layer, const float *x, int S, int D,
                         ColiVkTensor *gqkv_t, ColiVkTensor *gz_t, const float *params,
                         ColiVkTensor *out_t, float *out,
                         int KH, int KD, int VH, int VD, int conv_dim, int conv_k,
                         float eps, int Dout) {
    if (S == 1)
        return coli_vk_gdn_full(layer, x, D, gqkv_t, gz_t, params, out_t, out,
                                KH, KD, VH, VD, conv_dim, conv_k, eps, Dout);
    if (!coli_vk_gdn_full_available() || !gqkv_t || !gz_t || !out_t || S < 1) return 0;
    if (layer < 0 || layer >= VK_KV_LAYERS || VD > 512 || KD > 256 || (VH % KH) != 0) return 0;
    if (!G.gdn_st[layer] || !G.gdn_ring[layer] || !G.gdn_cw[layer]) return 0;
    int key_dim = KH * KD, value_dim = VH * VD;
    int pr_stride = 2 * VH + VD;
    if (gqkv_t->I != D || gqkv_t->O != conv_dim || gz_t->I != D || gz_t->O != value_dim ||
        out_t->I != value_dim || out_t->O != Dout || conv_dim != 2 * key_dim + value_dim) return 0;

    VkMemoryBarrier mb_c = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    VkMemoryBarrier mb_t2c = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    VkMemoryBarrier mb_c2t = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT};

    for (int s0 = 0; s0 < S; s0 += GDN_SEQ_CHUNK) {
        int n = S - s0; if (n > GDN_SEQ_CHUNK) n = GDN_SEQ_CHUNK;
        size_t xb = (size_t)n * D * 4, prb = (size_t)n * (size_t)pr_stride * 4;
        size_t yb = (size_t)n * Dout * 4;
        size_t x1 = (size_t)D * 4, pr1 = (size_t)pr_stride * 4, y1 = (size_t)Dout * 4;
        /* batch uploads: route_x / y2 / gdnf_pr; single-token work: x / gdn_ba / y */
        if (!scratch_reserve(&G.route_x, xb) || !scratch_reserve(&G.gdnf_pr, prb) ||
            !scratch_reserve_mt(&G.y2, yb, G.memtype_cached) ||
            !scratch_reserve(&G.x, x1) || !scratch_reserve(&G.gdn_ba, pr1) ||
            !scratch_reserve_mt(&G.y, y1, G.memtype_cached) ||
            !scratch_reserve(&G.gdnf_qkv, (size_t)conv_dim * 4) ||
            !scratch_reserve(&G.gdnf_z, (size_t)value_dim * 4) ||
            !scratch_reserve(&G.gdnf_cv, (size_t)conv_dim * 4) ||
            !scratch_reserve(&G.gdn_y, (size_t)value_dim * 4))
            return 0;
        memcpy(G.route_x.ptr, x + (int64_t)s0 * D, xb);
        memcpy(G.gdnf_pr.ptr, params + (int64_t)s0 * pr_stride, prb);

        /* Bind once to fixed single-token work buffers. */
        VkDescriptorBufferInfo bqkv[4] = {
            {.buffer=G.x.buf,.range=VK_WHOLE_SIZE}, {.buffer=gqkv_t->wbuf,.range=VK_WHOLE_SIZE},
            {.buffer=gqkv_t->sbuf,.range=VK_WHOLE_SIZE}, {.buffer=G.gdnf_qkv.buf,.range=VK_WHOLE_SIZE}};
        wr_desc(G.gdnf_gqkv, 4, bqkv);
        VkDescriptorBufferInfo bz[4] = {
            {.buffer=G.x.buf,.range=VK_WHOLE_SIZE}, {.buffer=gz_t->wbuf,.range=VK_WHOLE_SIZE},
            {.buffer=gz_t->sbuf,.range=VK_WHOLE_SIZE}, {.buffer=G.gdnf_z.buf,.range=VK_WHOLE_SIZE}};
        wr_desc(G.gdnf_gz, 4, bz);
        VkDescriptorBufferInfo bc[4] = {
            {.buffer=G.gdnf_qkv.buf,.range=VK_WHOLE_SIZE}, {.buffer=G.gdn_cw[layer],.range=VK_WHOLE_SIZE},
            {.buffer=G.gdn_ring[layer],.range=VK_WHOLE_SIZE}, {.buffer=G.gdnf_cv.buf,.range=VK_WHOLE_SIZE}};
        wr_desc(G.dset_gdnconv, 4, bc);
        VkDescriptorBufferInfo bd[5] = {
            {.buffer=G.gdnf_cv.buf,.range=VK_WHOLE_SIZE}, {.buffer=G.gdnf_z.buf,.range=VK_WHOLE_SIZE},
            {.buffer=G.gdn_ba.buf,.range=VK_WHOLE_SIZE}, {.buffer=G.gdn_st[layer],.range=VK_WHOLE_SIZE},
            {.buffer=G.gdn_y.buf,.range=VK_WHOLE_SIZE}};
        wr_desc(G.dset_gdncv, 5, bd);
        VkDescriptorBufferInfo bo[4] = {
            {.buffer=G.gdn_y.buf,.range=VK_WHOLE_SIZE}, {.buffer=out_t->wbuf,.range=VK_WHOLE_SIZE},
            {.buffer=out_t->sbuf,.range=VK_WHOLE_SIZE}, {.buffer=G.y.buf,.range=VK_WHOLE_SIZE}};
        wr_desc(G.gdnf_out, 4, bo);

        VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
        VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");

        for (int i = 0; i < n; i++) {
            VkBufferCopy cx = {.srcOffset = (VkDeviceSize)i * x1, .dstOffset = 0, .size = x1};
            VkBufferCopy cp = {.srcOffset = (VkDeviceSize)i * pr1, .dstOffset = 0, .size = pr1};
            vkCmdCopyBuffer(G.cmd, G.route_x.buf, G.x.buf, 1, &cx);
            vkCmdCopyBuffer(G.cmd, G.gdnf_pr.buf, G.gdn_ba.buf, 1, &cp);
            vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb_t2c, 0, NULL, 0, NULL);

            vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
            vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gdnf_gqkv, 0, NULL);
            struct PC pq = {gqkv_t->fmt, 1, D, conv_dim, gqkv_t->rowWords, gqkv_t->gs};
            vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pq), &pq);
            vkCmdDispatch(G.cmd, (uint32_t)((conv_dim + 7) / 8), 1, 1);
            vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gdnf_gz, 0, NULL);
            struct PC pz = {gz_t->fmt, 1, D, value_dim, gz_t->rowWords, gz_t->gs};
            vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pz), &pz);
            vkCmdDispatch(G.cmd, (uint32_t)((value_dim + 7) / 8), 1, 1);
            vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb_c, 0, NULL, 0, NULL);

            vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gdnconv);
            vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gdnconv, 0, 1, &G.dset_gdnconv, 0, NULL);
            struct PCConv pcv = {conv_dim, conv_k};
            vkCmdPushConstants(G.cmd, G.plyt_gdnconv, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcv), &pcv);
            vkCmdDispatch(G.cmd, (uint32_t)((conv_dim + 255) / 256), 1, 1);
            vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb_c, 0, NULL, 0, NULL);

            vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gdncv);
            vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gdncv, 0, 1, &G.dset_gdncv, 0, NULL);
            struct PCGdnCv pd = {(uint32_t)KH, (uint32_t)KD, (uint32_t)VH, (uint32_t)VD, (uint32_t)key_dim, eps};
            vkCmdPushConstants(G.cmd, G.plyt_gdncv, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pd), &pd);
            vkCmdDispatch(G.cmd, (uint32_t)VH, 1, 1);
            vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb_c, 0, NULL, 0, NULL);

            vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
            vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gdnf_out, 0, NULL);
            struct PC po = {out_t->fmt, 1, value_dim, Dout, out_t->rowWords, out_t->gs};
            vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(po), &po);
            vkCmdDispatch(G.cmd, (uint32_t)((Dout + 7) / 8), 1, 1);

            vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb_c2t, 0, NULL, 0, NULL);
            VkBufferCopy cy = {.srcOffset = 0, .dstOffset = (VkDeviceSize)i * y1, .size = y1};
            vkCmdCopyBuffer(G.cmd, G.y.buf, G.y2.buf, 1, &cy);
            if (i + 1 < n)
                vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb_t2c, 0, NULL, 0, NULL);
        }
        VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
        VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
        VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
        if (vk_fence_wait_loud(G.fence, "gdn_full_seq") != VK_SUCCESS) { G.ready = 0; return 0; }
        memcpy(out + (int64_t)s0 * Dout, G.y2.ptr, yb);
        G.cmd_ready = 0; G.bound_tensor = NULL;
    }
    { static int once;
      if (!once) { fprintf(stderr, "[VK] GDN prefill seq ENABLED (chunk=%d tokens/submit)\n", GDN_SEQ_CHUNK); once = 1; } }
    return 1;
}

int coli_vk_gqa_available(void) { return G.ready && G.pipe_gqa ? 1 : 0; }

/* Standard GQA decode attention for S causal query rows of one full-attention
 * layer, one submit. The device KV mirror (G.kv[layer]) must already hold the
 * normed+roped K rows in ->bl and the raw V rows in ->br (row = kh*max_t + pos),
 * appended via coli_vk_kv_row against an ensure of (KH*max_t, hd, hd). q is the
 * per-head [q(hd)|gate(hd)] block [S,H,2*hd]; ctx [S,H,hd] is read back (pre
 * o-projection). Returns 0 -> caller falls back to CPU. */
int coli_vk_gqa(int layer, const float *q, float *ctx, int S, int H, int KH, int hd,
                int max_t, int st0, int T, float scale) {
    if (!G.ready || !G.pipe_gqa || S < 1 || H < 1 || KH < 1 || layer < 0 || layer >= VK_KV_LAYERS) return 0;
    if (hd > 256 || (H % KH) != 0 || st0 < 0 || T - S - st0 < 0) return 0;
    VkKvLayer *kv = &G.kv[layer];
    if (!kv->bl || kv->rows < KH * max_t || kv->K != hd || kv->R != hd) return 0;
    size_t qb = (size_t)S * H * 2 * hd * 4, cb = (size_t)S * H * hd * 4;
    if (!scratch_reserve(&G.x, qb) || !scratch_reserve_mt(&G.y, cb, G.memtype_cached)) return 0;
    memcpy(G.x.ptr, q, qb);

    VkDescriptorBufferInfo bi[4] = {
        {.buffer = G.x.buf,  .range = VK_WHOLE_SIZE}, {.buffer = kv->bl, .range = VK_WHOLE_SIZE},
        {.buffer = kv->br,   .range = VK_WHOLE_SIZE}, {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_gqa, 4, bi);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gqa);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gqa, 0, 1, &G.dset_gqa, 0, NULL);
    struct PCGqa pc = {S, H, KH, hd, max_t, st0, T, scale};
    vkCmdPushConstants(G.cmd, G.plyt_gqa, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)H, (uint32_t)S, 1);   /* one workgroup per (head, query row) */
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (vk_fence_wait_loud(G.fence, "gqa") != VK_SUCCESS) { G.ready = 0; return 0; }
    memcpy(ctx, G.y.ptr, cb);
    G.cmd_ready = 0; G.bound_tensor = NULL;
    return 1;
}

/* Fused GQA attention + output projection in ONE submit: the attention kernel
 * writes ctx to a GPU-only scratch (G.att_ctx), a barrier, then the resident
 * o-projection matmul (qmatmul pipeline) reads that scratch and writes out[S,Dout]
 * — ctx never round-trips to the host. `ot` is the already-resident o_proj tensor
 * (H*hd -> Dout). Returns 0 -> caller falls back to gqa + separate mm_dense. */
int coli_vk_gqa_project(int layer, const float *q, ColiVkTensor *ot, float *out,
                        int S, int H, int KH, int hd, int max_t, int st0, int T,
                        float scale, int Dout) {
    if (!G.ready || !G.pipe_gqa || !ot || S < 1 || H < 1 || layer < 0 || layer >= VK_KV_LAYERS) return 0;
    if (hd > 256 || (H % KH) != 0 || st0 < 0 || T - S - st0 < 0 || Dout < 1) return 0;
    if (ot->I != H * hd || ot->O != Dout) return 0;
    VkKvLayer *kv = &G.kv[layer];
    if (!kv->bl || kv->rows < KH * max_t || kv->K != hd || kv->R != hd) return 0;
    size_t qb = (size_t)S * H * 2 * hd * 4, cb = (size_t)S * H * hd * 4, ob = (size_t)S * Dout * 4;
    if (!scratch_reserve(&G.x, qb) || !scratch_reserve(&G.att_ctx, cb) ||
        !scratch_reserve_mt(&G.y, ob, G.memtype_cached)) return 0;   /* att_ctx GPU-only; y read back */
    memcpy(G.x.ptr, q, qb);

    VkDescriptorBufferInfo bi[4] = {                 /* gqa: q | K | V | ctx(scratch) */
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = kv->bl, .range = VK_WHOLE_SIZE},
        {.buffer = kv->br, .range = VK_WHOLE_SIZE},  {.buffer = G.att_ctx.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_gqa, 4, bi);
    VkDescriptorBufferInfo oi[4] = {                 /* o-proj: x=ctx | W | scales | out */
        {.buffer = G.att_ctx.buf, .range = VK_WHOLE_SIZE}, {.buffer = ot->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = ot->sbuf, .range = VK_WHOLE_SIZE},      {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset, 4, oi);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gqa);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gqa, 0, 1, &G.dset_gqa, 0, NULL);
    struct PCGqa pc = {S, H, KH, hd, max_t, st0, T, scale};
    vkCmdPushConstants(G.cmd, G.plyt_gqa, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)H, (uint32_t)S, 1);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    struct PC opc = {ot->fmt, S, H * hd, Dout, ot->rowWords, ot->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(opc), &opc);
    vkCmdDispatch(G.cmd, (uint32_t)((Dout + 7) / 8), (uint32_t)S, 1);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (vk_fence_wait_loud(G.fence, "gqa_project") != VK_SUCCESS) { G.ready = 0; return 0; }
    memcpy(out, G.y.ptr, ob);
    G.cmd_ready = 0; G.bound_tensor = NULL;
    return 1;
}

int coli_vk_gqa_full_available(void) {
    return G.ready && G.pipe_qkv && G.pipe_gqa && G.pipe ? 1 : 0;
}

/* Upload one full-attention layer's q/k RMSNorm weights [hd] resident (once). */
int coli_vk_gqa_norm_weight(int layer, const float *qnw, const float *knw, int hd) {
    if (!G.ready || layer < 0 || layer >= VK_KV_LAYERS || hd < 1) return 0;
    size_t bytes = (size_t)hd * 4;
    if (!G.gqa_qnw[layer]) {
        float p0 = G.prio; G.prio = 1.0f;
        int ok = alloc_hostvis(bytes, &G.gqa_qnw[layer], &G.gqa_qnwm[layer], &G.gqa_qnwp[layer])
              && alloc_hostvis(bytes, &G.gqa_knw[layer], &G.gqa_knwm[layer], &G.gqa_knwp[layer]);
        G.prio = p0;
        if (!ok) { G.gqa_qnw[layer] = VK_NULL_HANDLE; return 0; }
    }
    memcpy(G.gqa_qnwp[layer], qnw, bytes);
    memcpy(G.gqa_knwp[layer], knw, bytes);
    return 1;
}

/* Whole GQA block for S query rows of one full-attention layer in ONE submit:
 *   q/k/v matmul (x -> qg,k,v) -> q/k-norm+rope + KV-mirror write -> attention -> o-proj.
 * qg/k/v/ctx stay in GPU-only scratch; the device KV mirror (G.kv[layer]) is written on
 * device (becomes canonical). qg is [S,H,2*hd] (q|gate). out[S,Dout] read back. */
int coli_vk_gqa_full(int layer, const float *x, int D, ColiVkTensor *gq_t, ColiVkTensor *gk_t,
                     ColiVkTensor *gv_t, ColiVkTensor *out_t, float *out, int S, int H, int KH,
                     int hd, int rot, int pos_base, int max_t, float eps, float theta, int Dout) {
    if (!coli_vk_gqa_full_available() || !gq_t || !gk_t || !gv_t || !out_t) return 0;
    if (S < 1 || hd > 256 || (H % KH) != 0 || layer < 0 || layer >= VK_KV_LAYERS || Dout < 1) return 0;
    if (!G.gqa_qnw[layer]) return 0;
    VkKvLayer *kv = &G.kv[layer];
    if (!kv->bl || kv->rows < KH * max_t || kv->K != hd || kv->R != hd) return 0;
    int qo = H * 2 * hd, ko = KH * hd, T = pos_base + S;
    if (gq_t->I != D || gq_t->O != qo || gk_t->I != D || gk_t->O != ko ||
        gv_t->I != D || gv_t->O != ko || out_t->I != H * hd || out_t->O != Dout) return 0;
    size_t xb = (size_t)S * D * 4, cb = (size_t)S * H * hd * 4, ob = (size_t)S * Dout * 4;
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve(&G.gqaf_qg, (size_t)S*qo*4) ||
        !scratch_reserve(&G.gqaf_kb, (size_t)S*ko*4) || !scratch_reserve(&G.gqaf_vb, (size_t)S*ko*4) ||
        !scratch_reserve(&G.att_ctx, cb) || !scratch_reserve_mt(&G.y, ob, G.memtype_cached)) return 0;
    memcpy(G.x.ptr, x, xb);

    VkDescriptorBufferInfo bq[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gq_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gq_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gqaf_qg.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gqaf_q, 4, bq);
    VkDescriptorBufferInfo bk[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gk_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gk_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gqaf_kb.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gqaf_k, 4, bk);
    VkDescriptorBufferInfo bv[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gv_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gv_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gqaf_vb.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gqaf_v, 4, bv);
    VkDescriptorBufferInfo bn[7] = {{.buffer=G.gqaf_qg.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gqaf_kb.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gqaf_vb.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gqa_qnw[layer],.range=VK_WHOLE_SIZE},{.buffer=G.gqa_knw[layer],.range=VK_WHOLE_SIZE},{.buffer=kv->bl,.range=VK_WHOLE_SIZE},{.buffer=kv->br,.range=VK_WHOLE_SIZE}};
    wr_desc(G.dset_qkv, 7, bn);
    VkDescriptorBufferInfo ba[4] = {{.buffer=G.gqaf_qg.buf,.range=VK_WHOLE_SIZE},{.buffer=kv->bl,.range=VK_WHOLE_SIZE},{.buffer=kv->br,.range=VK_WHOLE_SIZE},{.buffer=G.att_ctx.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.dset_gqa, 4, ba);
    VkDescriptorBufferInfo bo[4] = {{.buffer=G.att_ctx.buf,.range=VK_WHOLE_SIZE},{.buffer=out_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=out_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.y.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gqaf_out, 4, bo);

    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    /* 1) q/k/v matmuls (x -> qg, k, v) */
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    struct PC pq = {gq_t->fmt, S, D, qo, gq_t->rowWords, gq_t->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gqaf_q, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pq), &pq);
    vkCmdDispatch(G.cmd, (uint32_t)((qo + 7) / 8), (uint32_t)S, 1);
    struct PC pk = {gk_t->fmt, S, D, ko, gk_t->rowWords, gk_t->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gqaf_k, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pk), &pk);
    vkCmdDispatch(G.cmd, (uint32_t)((ko + 7) / 8), (uint32_t)S, 1);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gqaf_v, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pk), &pk);
    vkCmdDispatch(G.cmd, (uint32_t)((ko + 7) / 8), (uint32_t)S, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    /* 2) q/k-norm + rope + KV-mirror write */
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_qkv);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_qkv, 0, 1, &G.dset_qkv, 0, NULL);
    struct PCQkv pn = {S, H, KH, hd, rot, pos_base, max_t, eps, theta};
    vkCmdPushConstants(G.cmd, G.plyt_qkv, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pn), &pn);
    vkCmdDispatch(G.cmd, (uint32_t)H, (uint32_t)S, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    /* 3) attention -> ctx */
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gqa);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gqa, 0, 1, &G.dset_gqa, 0, NULL);
    struct PCGqa pg = {S, H, KH, hd, max_t, 0, T, 1.0f / sqrtf((float)hd)};
    vkCmdPushConstants(G.cmd, G.plyt_gqa, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pg), &pg);
    vkCmdDispatch(G.cmd, (uint32_t)H, (uint32_t)S, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    /* 4) o projection (ctx -> out) */
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gqaf_out, 0, NULL);
    struct PC po = {out_t->fmt, S, H * hd, Dout, out_t->rowWords, out_t->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(po), &po);
    vkCmdDispatch(G.cmd, (uint32_t)((Dout + 7) / 8), (uint32_t)S, 1);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si2 = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si2, G.fence), "queueSubmit");
    if (vk_fence_wait_loud(G.fence, "gqa_full") != VK_SUCCESS) { G.ready = 0; return 0; }
    memcpy(out, G.y.ptr, ob);
    G.cmd_ready = 0; G.bound_tensor = NULL;
    return 1;
}

int coli_vk_moe_route_available(void) {
    return G.ready && G.pipe_nrmz && G.pipe_rrnz && G.pipe && G.dset_route_rt ? 1 : 0;
}

/* Upload one model layer's in/post RMSNorm weights (f32, size D) into resident buffers. */
int coli_vk_layer_norm_weight(int layer, const float *in_ln, const float *post_ln, int D) {
    if (!G.ready || layer < 0 || layer >= VK_KV_LAYERS || D < 1 || !in_ln || !post_ln) return 0;
    size_t bytes = (size_t)D * 4;
    if (!G.qln_in[layer] || G.qln_D[layer] != D) {
        if (G.qln_in[layer]) {
            vkDestroyBuffer(G.dev, G.qln_in[layer], NULL); vkFreeMemory(G.dev, G.qln_inm[layer], NULL);
            vkDestroyBuffer(G.dev, G.qln_post[layer], NULL); vkFreeMemory(G.dev, G.qln_postm[layer], NULL);
            G.qln_in[layer] = VK_NULL_HANDLE;
        }
        float p0 = G.prio; G.prio = 1.0f;
        int ok = alloc_hostvis(bytes, &G.qln_in[layer], &G.qln_inm[layer], &G.qln_inp[layer])
              && alloc_hostvis(bytes, &G.qln_post[layer], &G.qln_postm[layer], &G.qln_postp[layer]);
        G.prio = p0;
        if (!ok) { G.qln_in[layer] = VK_NULL_HANDLE; return 0; }
        G.qln_D[layer] = D;
    }
    memcpy(G.qln_inp[layer], in_ln, bytes);
    memcpy(G.qln_postp[layer], post_ln, bytes);
    return 1;
}

/* Append residual+=delta + post_ln + router onto the already-begun main cmd buffer.
 * Expects: resid in route_x, attn/GDN delta in att_delta, post_ln at qln_post[ln_layer].
 * Writes resid_out -> route_xo, nrm -> route_nrm, pr -> y2 (all host-cached). */
static void route_tail_record(ColiVkTensor *router, int ln_layer, int S, int D, int E, float eps) {
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    VkDescriptorBufferInfo br[5] = {
        {.buffer = G.route_x.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.att_delta.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.qln_post[ln_layer], .range = VK_WHOLE_SIZE},
        {.buffer = G.route_xo.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.route_nrm.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_rrnz, 5, br);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_rrnz);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_rrnz, 0, 1, &G.dset_rrnz, 0, NULL);
    struct PCN pn = {S, D, eps};
    vkCmdPushConstants(G.cmd, G.plyt_rrnz, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pn), &pn);
    vkCmdDispatch(G.cmd, (uint32_t)S, 1, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    VkDescriptorBufferInfo bt[4] = {
        {.buffer = G.route_nrm.buf, .range = VK_WHOLE_SIZE}, {.buffer = router->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = router->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y2.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_route_rt, 4, bt);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset_route_rt, 0, NULL);
    struct PC pc = {router->fmt, S, D, E, router->rowWords, router->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)((E + 7) / 8), (uint32_t)S, 1);
}

/* GQA full block + in_ln + residual + post_ln + router in ONE submit. */
int coli_vk_gqa_full_route(int layer, const float *resid, int D,
                           ColiVkTensor *gq_t, ColiVkTensor *gk_t, ColiVkTensor *gv_t, ColiVkTensor *out_t,
                           ColiVkTensor *router, int ln_layer,
                           float *resid_out, float *nrm_out, float *pr_out,
                           int S, int H, int KH, int hd, int rot, int pos_base, int max_t,
                           float eps, float theta, int Dout, int E) {
    if (!coli_vk_moe_route_available() || !coli_vk_gqa_full_available()) return 0;
    if (!gq_t || !gk_t || !gv_t || !out_t || !router || !resid || !resid_out || !nrm_out || !pr_out) return 0;
    if (S < 1 || hd > 256 || (H % KH) != 0 || layer < 0 || layer >= VK_KV_LAYERS || Dout < 1 || E < 1) return 0;
    if (ln_layer < 0 || ln_layer >= VK_KV_LAYERS || !G.qln_in[ln_layer] || G.qln_D[ln_layer] != D) return 0;
    if (!G.gqa_qnw[layer]) return 0;
    VkKvLayer *kv = &G.kv[layer];
    if (!kv->bl || kv->rows < KH * max_t || kv->K != hd || kv->R != hd) return 0;
    int qo = H * 2 * hd, ko = KH * hd, T = pos_base + S;
    if (gq_t->I != D || gq_t->O != qo || gk_t->I != D || gk_t->O != ko ||
        gv_t->I != D || gv_t->O != ko || out_t->I != H * hd || out_t->O != Dout ||
        router->I != D || router->O != E) return 0;
    size_t xb = (size_t)S * D * 4, cb = (size_t)S * H * hd * 4, pb = (size_t)S * E * 4;
    if (!scratch_reserve_mt(&G.route_x, xb, G.memtype) ||
        !scratch_reserve(&G.x, xb) ||
        !scratch_reserve(&G.gqaf_qg, (size_t)S*qo*4) || !scratch_reserve(&G.gqaf_kb, (size_t)S*ko*4) ||
        !scratch_reserve(&G.gqaf_vb, (size_t)S*ko*4) || !scratch_reserve(&G.att_ctx, cb) ||
        !scratch_reserve(&G.att_delta, xb) ||
        !scratch_reserve_mt(&G.route_xo, xb, G.memtype_cached) ||
        !scratch_reserve_mt(&G.route_nrm, xb, G.memtype_cached) ||
        !scratch_reserve_mt(&G.y2, pb, G.memtype_cached)) return 0;
    memcpy(G.route_x.ptr, resid, xb);

    /* in_ln: resid -> G.x (attn input) */
    VkDescriptorBufferInfo bin[3] = {
        {.buffer = G.route_x.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.qln_in[ln_layer], .range = VK_WHOLE_SIZE},
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_nrm, 3, bin);

    VkDescriptorBufferInfo bq[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gq_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gq_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gqaf_qg.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gqaf_q, 4, bq);
    VkDescriptorBufferInfo bk[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gk_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gk_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gqaf_kb.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gqaf_k, 4, bk);
    VkDescriptorBufferInfo bv[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gv_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gv_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gqaf_vb.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gqaf_v, 4, bv);
    VkDescriptorBufferInfo bn[7] = {{.buffer=G.gqaf_qg.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gqaf_kb.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gqaf_vb.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gqa_qnw[layer],.range=VK_WHOLE_SIZE},{.buffer=G.gqa_knw[layer],.range=VK_WHOLE_SIZE},{.buffer=kv->bl,.range=VK_WHOLE_SIZE},{.buffer=kv->br,.range=VK_WHOLE_SIZE}};
    wr_desc(G.dset_qkv, 7, bn);
    VkDescriptorBufferInfo ba[4] = {{.buffer=G.gqaf_qg.buf,.range=VK_WHOLE_SIZE},{.buffer=kv->bl,.range=VK_WHOLE_SIZE},{.buffer=kv->br,.range=VK_WHOLE_SIZE},{.buffer=G.att_ctx.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.dset_gqa, 4, ba);
    VkDescriptorBufferInfo bo[4] = {{.buffer=G.att_ctx.buf,.range=VK_WHOLE_SIZE},{.buffer=out_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=out_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.att_delta.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gqaf_out, 4, bo);

    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    /* 0) in_ln */
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_nrmz);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_nrm, 0, 1, &G.dset_nrm, 0, NULL);
    struct PCN pin = {S, D, eps};
    vkCmdPushConstants(G.cmd, G.plyt_nrm, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pin), &pin);
    vkCmdDispatch(G.cmd, (uint32_t)S, 1, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    /* 1-4) same as gqa_full, but o-proj lands in att_delta (stays on device) */
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    struct PC pq = {gq_t->fmt, S, D, qo, gq_t->rowWords, gq_t->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gqaf_q, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pq), &pq);
    vkCmdDispatch(G.cmd, (uint32_t)((qo + 7) / 8), (uint32_t)S, 1);
    struct PC pk = {gk_t->fmt, S, D, ko, gk_t->rowWords, gk_t->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gqaf_k, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pk), &pk);
    vkCmdDispatch(G.cmd, (uint32_t)((ko + 7) / 8), (uint32_t)S, 1);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gqaf_v, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pk), &pk);
    vkCmdDispatch(G.cmd, (uint32_t)((ko + 7) / 8), (uint32_t)S, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_qkv);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_qkv, 0, 1, &G.dset_qkv, 0, NULL);
    struct PCQkv pn = {S, H, KH, hd, rot, pos_base, max_t, eps, theta};
    vkCmdPushConstants(G.cmd, G.plyt_qkv, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pn), &pn);
    vkCmdDispatch(G.cmd, (uint32_t)H, (uint32_t)S, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gqa);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gqa, 0, 1, &G.dset_gqa, 0, NULL);
    struct PCGqa pg = {S, H, KH, hd, max_t, 0, T, 1.0f / sqrtf((float)hd)};
    vkCmdPushConstants(G.cmd, G.plyt_gqa, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pg), &pg);
    vkCmdDispatch(G.cmd, (uint32_t)H, (uint32_t)S, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gqaf_out, 0, NULL);
    struct PC po = {out_t->fmt, S, H * hd, Dout, out_t->rowWords, out_t->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(po), &po);
    vkCmdDispatch(G.cmd, (uint32_t)((Dout + 7) / 8), (uint32_t)S, 1);
    /* 5) residual + post_ln + router */
    route_tail_record(router, ln_layer, S, D, E, eps);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (vk_fence_wait_loud(G.fence, "gqa_full_route") != VK_SUCCESS) { G.ready = 0; return 0; }
    memcpy(resid_out, G.route_xo.ptr, xb);
    memcpy(nrm_out, G.route_nrm.ptr, xb);
    memcpy(pr_out, G.y2.ptr, pb);
    G.cmd_ready = 0; G.bound_tensor = NULL;
    return 1;
}

/* GDN full block + residual + post_ln + router. x_in is already post-in_ln (b/a
 * still computed on the host from it); resid is the pre-attn residual stream. */
int coli_vk_gdn_full_route(int layer, const float *x_in, const float *resid, int D,
                           ColiVkTensor *gqkv_t, ColiVkTensor *gz_t, const float *params,
                           ColiVkTensor *out_t, ColiVkTensor *router, int ln_layer,
                           float *resid_out, float *nrm_out, float *pr_out,
                           int KH, int KD, int VH, int VD, int conv_dim, int conv_k,
                           float eps, int Dout, int E) {
    if (!coli_vk_moe_route_available() || !coli_vk_gdn_full_available()) return 0;
    if (!gqkv_t || !gz_t || !out_t || !router || !x_in || !resid || !resid_out || !nrm_out || !pr_out) return 0;
    if (layer < 0 || layer >= VK_KV_LAYERS || VD > 512 || KD > 256 || (VH % KH) != 0 || E < 1) return 0;
    if (ln_layer < 0 || ln_layer >= VK_KV_LAYERS || !G.qln_post[ln_layer] || G.qln_D[ln_layer] != D) return 0;
    if (!G.gdn_st[layer] || !G.gdn_ring[layer] || !G.gdn_cw[layer]) return 0;
    int key_dim = KH * KD, value_dim = VH * VD;
    if (gqkv_t->I != D || gqkv_t->O != conv_dim || gz_t->I != D || gz_t->O != value_dim ||
        out_t->I != value_dim || out_t->O != Dout || conv_dim != 2 * key_dim + value_dim ||
        router->I != D || router->O != E) return 0;
    size_t xb = (size_t)D * 4, prb = (size_t)(2 * VH + VD) * 4, pb = (size_t)E * 4;
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve_mt(&G.route_x, xb, G.memtype) ||
        !scratch_reserve(&G.gdnf_qkv, (size_t)conv_dim * 4) || !scratch_reserve(&G.gdnf_z, (size_t)value_dim * 4) ||
        !scratch_reserve(&G.gdnf_cv, (size_t)conv_dim * 4) || !scratch_reserve(&G.gdn_y, (size_t)value_dim * 4) ||
        !scratch_reserve(&G.gdnf_pr, prb) || !scratch_reserve(&G.att_delta, xb) ||
        !scratch_reserve_mt(&G.route_xo, xb, G.memtype_cached) ||
        !scratch_reserve_mt(&G.route_nrm, xb, G.memtype_cached) ||
        !scratch_reserve_mt(&G.y2, pb, G.memtype_cached)) return 0;
    memcpy(G.x.ptr, x_in, xb);
    memcpy(G.route_x.ptr, resid, xb);
    memcpy(G.gdnf_pr.ptr, params, prb);

    VkDescriptorBufferInfo bqkv[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gqkv_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gqkv_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gdnf_qkv.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gdnf_gqkv, 4, bqkv);
    VkDescriptorBufferInfo bz[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gz_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gz_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gdnf_z.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gdnf_gz, 4, bz);
    VkDescriptorBufferInfo bc[4] = {{.buffer=G.gdnf_qkv.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gdn_cw[layer],.range=VK_WHOLE_SIZE},{.buffer=G.gdn_ring[layer],.range=VK_WHOLE_SIZE},{.buffer=G.gdnf_cv.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.dset_gdnconv, 4, bc);
    VkDescriptorBufferInfo bd[5] = {{.buffer=G.gdnf_cv.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gdnf_z.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gdnf_pr.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gdn_st[layer],.range=VK_WHOLE_SIZE},{.buffer=G.gdn_y.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.dset_gdncv, 5, bd);
    VkDescriptorBufferInfo bo[4] = {{.buffer=G.gdn_y.buf,.range=VK_WHOLE_SIZE},{.buffer=out_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=out_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.att_delta.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gdnf_out, 4, bo);

    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gdnf_gqkv, 0, NULL);
    struct PC pq = {gqkv_t->fmt, 1, D, conv_dim, gqkv_t->rowWords, gqkv_t->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pq), &pq);
    vkCmdDispatch(G.cmd, (uint32_t)((conv_dim + 7) / 8), 1, 1);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gdnf_gz, 0, NULL);
    struct PC pz = {gz_t->fmt, 1, D, value_dim, gz_t->rowWords, gz_t->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pz), &pz);
    vkCmdDispatch(G.cmd, (uint32_t)((value_dim + 7) / 8), 1, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gdnconv);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gdnconv, 0, 1, &G.dset_gdnconv, 0, NULL);
    struct PCConv pcv = {conv_dim, conv_k};
    vkCmdPushConstants(G.cmd, G.plyt_gdnconv, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcv), &pcv);
    vkCmdDispatch(G.cmd, (uint32_t)((conv_dim + 255) / 256), 1, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gdncv);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gdncv, 0, 1, &G.dset_gdncv, 0, NULL);
    struct PCGdnCv pd = {(uint32_t)KH, (uint32_t)KD, (uint32_t)VH, (uint32_t)VD, (uint32_t)key_dim, eps};
    vkCmdPushConstants(G.cmd, G.plyt_gdncv, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pd), &pd);
    vkCmdDispatch(G.cmd, (uint32_t)VH, 1, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gdnf_out, 0, NULL);
    struct PC po = {out_t->fmt, 1, value_dim, Dout, out_t->rowWords, out_t->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(po), &po);
    vkCmdDispatch(G.cmd, (uint32_t)((Dout + 7) / 8), 1, 1);
    route_tail_record(router, ln_layer, 1, D, E, eps);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (vk_fence_wait_loud(G.fence, "gdn_full_route") != VK_SUCCESS) { G.ready = 0; return 0; }
    memcpy(resid_out, G.route_xo.ptr, xb);
    memcpy(nrm_out, G.route_nrm.ptr, xb);
    memcpy(pr_out, G.y2.ptr, pb);
    G.cmd_ready = 0; G.bound_tensor = NULL;
    return 1;
}

/* ---- Decode residual stream (device nrm → eg, resid accumulate, eg→route sem) ---- */

int coli_vk_stream_available(void) {
    int want_topk = g_qwen_opts.topk != 0;
    return G.ready && G.pipe_rep && G.pipe_macc && G.pipe_gdp && G.pipe_topk &&
           G.pipe_rrnz && G.shader_gu && G.eg_sem && want_topk ? 1 : 0;
}

static int stream_wait_eg(void) {
    if (!G.eg_inflight) return 1;
    G.eg_inflight = 0;
    if (vk_fence_wait(G.eg_fence) != VK_SUCCESS) {
        fprintf(stderr, "[VK] stream eg fence wait failed — disabling GPU offload\n");
        G.ready = 0; return 0;
    }
    return 1;
}

static int pp_drain_all(void); /* defined with moe_ix ping-pong helpers below */

/* Binary semaphores stay signaled until waited; stream_end joins the eg fence but
 * does not consume eg_sem. Drain with an empty submit so the next issue can signal. */
static int stream_drain_sem(void) {
    if (!G.eg_sem_armed) return 1;
    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1, .pWaitSemaphores = &G.eg_sem, .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (vk_fence_wait_loud(G.fence, "drain_eg_sem") != VK_SUCCESS) { G.ready = 0; return 0; }
    G.eg_sem_armed = 0; G.cmd_ready = 0;
    return 1;
}

int coli_vk_stream_begin(const float *x, int D) {
    if (!coli_vk_stream_available() || !x || D < 1) return 0;
    if (!pp_drain_all()) return 0;
    if (!stream_wait_eg()) return 0;
    if (!stream_drain_sem()) return 0;
    size_t xb = (size_t)D * 4;
    if (!scratch_reserve_mt(&G.stream, xb, G.memtype_cached)) return 0;
    memcpy(G.stream.ptr, x, xb);
    G.stream_D = D; G.stream_live = 1;
    G.pp_slot = 0; G.pp_rec = 0; G.cmd_rec = G.cmd;
    G.ix_hist_pending = 0;
    memset(G.ix_hist_ready, 0, sizeof(G.ix_hist_ready));
    memset(G.ix_hist_k, 0, sizeof(G.ix_hist_k));
    return 1;
}

int coli_vk_stream_end(float *x, int D) {
    if (!G.stream_live || D != G.stream_D) return 0;
    if (!pp_drain_all()) return 0;
    if (!stream_wait_eg()) return 0;
    if (!stream_drain_sem()) return 0;
    /* x==NULL: drain only and keep resid on device for stream_norm_argmax. */
    if (x) memcpy(x, G.stream.ptr, (size_t)D * 4);
    G.stream_live = 0;
    G.cmd_rec = G.cmd; G.pp_rec = 0;
    if (G.eg_pipe_cache_on && (G.eg_pipe_hits + G.eg_pipe_misses) > 0) {
        static int logged;
        if (!logged && g_qwen_opts.eg_stats) {
            fprintf(stderr, "[VK] eg_pipe cache: hits=%d misses=%d (%.1f%% hit)\n",
                    G.eg_pipe_hits, G.eg_pipe_misses,
                    100.0 * G.eg_pipe_hits / (G.eg_pipe_hits + G.eg_pipe_misses));
            logged = 1;
        }
    }
    return 1;
}

/* Print once from the engine after a generate run (--eg-stats). */
void coli_vk_route_cache_stats(void) {
    int tot = G.route_cache_hits + G.route_cache_misses;
    if (!tot) return;
    fprintf(stderr, "[VK] route_gdn CB cache: hits=%d misses=%d (%.1f%% hit)\n",
            G.route_cache_hits, G.route_cache_misses,
            100.0 * G.route_cache_hits / tot);
}

int coli_vk_stream_add(const float *y, int D) {
    if (!G.stream_live || !y || D != G.stream_D) return 0;
    if (!stream_wait_eg()) return 0;   /* resid must be idle before host touches it */
    float *r = (float *)G.stream.ptr;
    for (int d = 0; d < D; d++) r[d] += y[d];
    return 1;
}

int coli_vk_stream_copy_resid(float *x, int D) {
    if (!x || !G.stream.ptr || D != G.stream_D || D < 1) return 0;
    memcpy(x, G.stream.ptr, (size_t)D * 4);
    return 1;
}

/* Upload model.norm.weight once for the stream→norm→argmax fuse. */
int coli_vk_final_norm_weight(const float *w, int D) {
    if (!G.ready || !w || D < 1 || !G.pipe_nrmz) return 0;
    size_t bytes = (size_t)D * 4;
    if (!G.final_nw || G.final_nD != D) {
        if (G.final_nw) {
            vkDestroyBuffer(G.dev, G.final_nw, NULL);
            vkFreeMemory(G.dev, G.final_nwm, NULL);
            G.final_nw = VK_NULL_HANDLE; G.final_nwp = NULL;
        }
        float p0 = G.prio; G.prio = 1.0f;
        int ok = alloc_hostvis(bytes, &G.final_nw, &G.final_nwm, &G.final_nwp);
        G.prio = p0;
        if (!ok) return 0;
        G.final_nD = D;
    }
    memcpy(G.final_nwp, w, bytes);
    return 1;
}

/* After coli_vk_stream_end(NULL, D): resid stays in G.stream. Run final RMSNorm +
 * lm_head + device argmax in ONE submit; only the winning token id is read back. */
int coli_vk_stream_norm_argmax(ColiVkTensor **tensor, const void *weights, const float *scales,
                               int fmt, int I, int O, int gs, float eps, int *idx, float *val) {
    if (!G.ready || !G.pipe_am || !G.pipe_nrmz || !G.final_nw || G.final_nD != I ||
        !G.stream.buf || G.stream_D != I || O < 1 || !idx) return 0;
    if (!upload_tensor(tensor, weights, scales, fmt, I, O, gs)) return 0;
    ColiVkTensor *t = *tensor;
    size_t pb = AM_GRP * 4 < 256 ? 256 : AM_GRP * 4;
    if (!scratch_reserve(&G.x, (size_t)I * 4) ||
        !scratch_reserve(&G.am_y, (size_t)O * 4) ||
        !scratch_reserve_mt(&G.am_pi, pb, G.memtype_cached) ||
        !scratch_reserve_mt(&G.am_pv, pb, G.memtype_cached)) return 0;

    VkDescriptorBufferInfo bn[3] = {
        {.buffer = G.stream.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.final_nw, .range = VK_WHOLE_SIZE},
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_nrm, 3, bn);
    VkDescriptorBufferInfo mi[4] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = t->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = t->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.am_y.buf, .range = VK_WHOLE_SIZE}};
    VkDescriptorBufferInfo ai[3] = {
        {.buffer = G.am_y.buf, .range = VK_WHOLE_SIZE}, {.buffer = G.am_pi.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.am_pv.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset, 4, mi);
    wr_desc(G.dset_am, 3, ai);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    dp_ts_begin(G.cmd, DP_TSQ_AM);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_nrmz);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_nrm, 0, 1, &G.dset_nrm, 0, NULL);
    struct PCN pin = {1, I, eps};
    vkCmdPushConstants(G.cmd, G.plyt_nrm, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pin), &pin);
    vkCmdDispatch(G.cmd, 1, 1, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    struct PC pc = {fmt, 1, I, O, t->rowWords, t->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)((O + 7) / 8), 1, 1);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_am);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_am, 0, 1, &G.dset_am, 0, NULL);
    for (int stage = 0; stage < 2; stage++) {
        vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        struct PCAm am = {O, stage, AM_GRP};
        vkCmdPushConstants(G.cmd, G.plyt_am, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(am), &am);
        vkCmdDispatch(G.cmd, stage == 0 ? (uint32_t)AM_GRP : 1u, 1, 1);
    }
    dp_ts_end(G.cmd);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    double ts = dp_t0();
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    dp_sub_add(ts);
    G.cmd_ready = 0; G.bound_tensor = NULL;
    double tw = dp_t0();
    if (vk_fence_wait_loud(G.fence, "stream_norm_argmax") != VK_SUCCESS) { G.ready = 0; return 0; }
    dp_wait_add(tw);
    { double ms = dp_ts_read(DP_TSQ_AM); if (ms > 0.0) { g_dp_am_ms += ms; g_dp_am_n++; } }
    *idx = (int)((const uint32_t *)G.am_pi.ptr)[0];
    if (val) *val = ((const float *)G.am_pv.ptr)[0];
    return 1;
}

const float *coli_vk_stream_nrm(void) {
    return G.stream_live && G.route_nrm.ptr ? (const float *)G.route_nrm.ptr : NULL;
}

int coli_vk_moe_ix_available(void) {
    return G.ready && G.moe_ix_hw && G.moe_ix_user && G.pipe_moe_ix && G.pipe_moe_pack &&
           G.dset_moe_ix != VK_NULL_HANDLE ? 1 : 0;
}

int coli_vk_stream_moe_fused(void) {
    return G.moe_ix_fused_last ? 1 : 0;
}

/* Allocate the dual descriptor sets used by async moe_ix route recording. */
static int moe_ix_pp_ensure_sets(void) {
    if (G.pp_pool && G.pp_gdnout[1]) return 1; /* fully allocated */
    if (G.pp_pool) { /* partial failure leftover */
        vkDestroyDescriptorPool(G.dev, G.pp_pool, NULL);
        G.pp_pool = VK_NULL_HANDLE;
        memset(G.pp_nrm, 0, sizeof(G.pp_nrm));
        memset(G.pp_gq, 0, sizeof(G.pp_gq));
        memset(G.pp_gdnout, 0, sizeof(G.pp_gdnout));
    }
    if (!G.dsl || !G.dsl_nrm || !G.dsl_qkv || !G.dsl_gqa || !G.dsl_rrnz ||
        !G.dsl_moe_pack || !G.dsl_macc || !G.dsl_topk || !G.dsl_gdp ||
        !G.dsl_gdnconv || !G.dsl_gdncv)
        return 0;
    /* 2 slots × (4×4 qmatmul + nrm3+qkv7+gqa4+rrnz5+route4+topk3+pack4+macc3
     * + gbb4+gba4+gdp6+gqkv4+gz4+gdnconv4+gdncv5+gdnout4) = 2×88 = 176 */
    const int nset = 48;       /* headroom beyond 2×19 */
    const uint32_t ndesc = 512; /* 2×88 bindings + driver slack */
    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = ndesc};
    VkDescriptorPoolCreateInfo dpi = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = (uint32_t)nset, .poolSizeCount = 1, .pPoolSizes = &ps};
    if (vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.pp_pool) != VK_SUCCESS) {
        fprintf(stderr, "[VK] pp pool create failed\n"); G.pp_pool = VK_NULL_HANDLE; return 0;
    }
    for (int s = 0; s < 2; s++) {
        VkDescriptorSetLayout L4[4] = {G.dsl, G.dsl, G.dsl, G.dsl};
        VkDescriptorSet sq[4];
        VkDescriptorSetAllocateInfo a4 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.pp_pool, .descriptorSetCount = 4, .pSetLayouts = L4};
        if (vkAllocateDescriptorSets(G.dev, &a4, sq) != VK_SUCCESS) goto pp_fail;
        G.pp_gq[s] = sq[0]; G.pp_gk[s] = sq[1]; G.pp_gv[s] = sq[2]; G.pp_go[s] = sq[3];
        VkDescriptorSetLayout lay;
        VkDescriptorSetAllocateInfo a1 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.pp_pool, .descriptorSetCount = 1, .pSetLayouts = &lay};
#define PP_ONE(dst, dslayout, tag) do { \
            lay = (dslayout); \
            if (vkAllocateDescriptorSets(G.dev, &a1, &(dst)) != VK_SUCCESS) { \
                fprintf(stderr, "[VK] pp %s failed\n", tag); goto pp_fail; } \
        } while (0)
        PP_ONE(G.pp_nrm[s], G.dsl_nrm, "nrm");
        PP_ONE(G.pp_qkv[s], G.dsl_qkv, "qkv");
        PP_ONE(G.pp_gqa[s], G.dsl_gqa, "gqa");
        PP_ONE(G.pp_rrnz[s], G.dsl_rrnz, "rrnz");
        PP_ONE(G.pp_route[s], G.dsl, "route");
        PP_ONE(G.pp_topk_ds[s], G.dsl_topk, "topk");
        PP_ONE(G.pp_pack[s], G.dsl_moe_pack, "pack");
        PP_ONE(G.pp_macc[s], G.dsl_macc, "macc");
        PP_ONE(G.pp_gbb[s], G.dsl, "gbb");
        PP_ONE(G.pp_gba[s], G.dsl, "gba");
        PP_ONE(G.pp_gdp[s], G.dsl_gdp, "gdp");
        PP_ONE(G.pp_gqkv[s], G.dsl, "gqkv");
        PP_ONE(G.pp_gz[s], G.dsl, "gz");
        PP_ONE(G.pp_gdnconv[s], G.dsl_gdnconv, "gdnconv");
        PP_ONE(G.pp_gdncv[s], G.dsl_gdncv, "gdncv");
        PP_ONE(G.pp_gdnout[s], G.dsl, "gdnout");
#undef PP_ONE
    }
    G.moe_ix_pp = g_qwen_opts.moe_ix_pp != 0;
    fprintf(stderr, "[VK] moe_ix ping-pong CBs %s (dual desc sets)\n",
            G.moe_ix_pp ? "ENABLED" : "disabled");
    return 1;
pp_fail:
    vkDestroyDescriptorPool(G.dev, G.pp_pool, NULL);
    G.pp_pool = VK_NULL_HANDLE;
    memset(G.pp_nrm, 0, sizeof(G.pp_nrm));
    memset(G.pp_gq, 0, sizeof(G.pp_gq));
    memset(G.pp_gdnout, 0, sizeof(G.pp_gdnout));
    G.moe_ix_pp = 0;
    return 0;
}

static int pp_reclaim_slot(int slot) {
    if (!G.pp_inflight[slot]) return 1;
    double tw = dp_t0();
    if (vk_fence_wait_loud(G.fence_pp[slot], "pp_reclaim") != VK_SUCCESS) {
        G.ready = 0; return 0;
    }
    dp_wait_add(tw);
    dp_ts_collect(G.pp_tsq[slot]);
    G.pp_tsq[slot] = -1;
    G.pp_inflight[slot] = 0;
    int L = G.pp_layer[slot], K = G.pp_topk[slot];
    int ibase = slot * 64;
    if (L >= 0 && L < VK_KV_LAYERS && K > 0 && G.route_idx.ptr &&
        G.route_idx.cap >= (size_t)(ibase + K) * 4) {
        memcpy(G.ix_hist[L], (int *)G.route_idx.ptr + ibase, (size_t)K * sizeof(int));
        G.ix_hist_ready[L] = 1;
        G.ix_hist_k[L] = K;
    }
    G.pp_layer[slot] = -1;
    return 1;
}

/* Acquire a ping-pong slot for recording layer `ln`. Returns slot or -1. */
static int pp_acquire(int ln, int topk) {
    if (!G.moe_ix_pp || !G.pp_pool) return -1;
    int slot = G.pp_slot;
    if (!pp_reclaim_slot(slot)) return -1;
    /* idx/val live in shared route_idx/val with base=slot*64 (GPU-serialized via eg_sem). */
    size_t kb = 128 * 4;
    if (kb < 256) kb = 256;
    if (!scratch_reserve_mt(&G.route_idx, kb, G.memtype_cached) ||
        !scratch_reserve_mt(&G.route_val, kb, G.memtype_cached))
        return -1;
    G.pp_rec = 1;
    G.pp_slot = slot;
    G.pp_layer[slot] = ln;
    G.pp_topk[slot] = topk;
    return slot;
}

static void pp_advance(void) {
    G.pp_slot ^= 1;
}

/* Swap in dual descriptor sets for slot s; save priors for pp_pop_sets. */
static VkDescriptorSet pp_save_nrm, pp_save_gq, pp_save_gk, pp_save_gv, pp_save_go;
static VkDescriptorSet pp_save_qkv, pp_save_gqa, pp_save_rrnz, pp_save_route, pp_save_topk;
static VkDescriptorSet pp_save_pack, pp_save_macc;
static VkDescriptorSet pp_save_gbb, pp_save_gba, pp_save_gdp;
static VkDescriptorSet pp_save_gqkv, pp_save_gz, pp_save_gdnconv, pp_save_gdncv, pp_save_gdnout;
static int pp_sets_pushed;

static void pp_push_sets(int s) {
    if (!G.pp_pool) return;
    pp_save_nrm = G.dset_nrm; G.dset_nrm = G.pp_nrm[s];
    pp_save_gq = G.gqaf_q; G.gqaf_q = G.pp_gq[s];
    pp_save_gk = G.gqaf_k; G.gqaf_k = G.pp_gk[s];
    pp_save_gv = G.gqaf_v; G.gqaf_v = G.pp_gv[s];
    pp_save_go = G.gqaf_out; G.gqaf_out = G.pp_go[s];
    pp_save_qkv = G.dset_qkv; G.dset_qkv = G.pp_qkv[s];
    pp_save_gqa = G.dset_gqa; G.dset_gqa = G.pp_gqa[s];
    pp_save_rrnz = G.dset_rrnz; G.dset_rrnz = G.pp_rrnz[s];
    pp_save_route = G.dset_route_rt; G.dset_route_rt = G.pp_route[s];
    pp_save_topk = G.dset_topk; G.dset_topk = G.pp_topk_ds[s];
    pp_save_pack = G.dset_moe_pack; G.dset_moe_pack = G.pp_pack[s];
    pp_save_macc = G.dset_macc; G.dset_macc = G.pp_macc[s];
    pp_save_gbb = G.dset_gbb; G.dset_gbb = G.pp_gbb[s];
    pp_save_gba = G.dset_gba; G.dset_gba = G.pp_gba[s];
    pp_save_gdp = G.dset_gdp; G.dset_gdp = G.pp_gdp[s];
    pp_save_gqkv = G.gdnf_gqkv; G.gdnf_gqkv = G.pp_gqkv[s];
    pp_save_gz = G.gdnf_gz; G.gdnf_gz = G.pp_gz[s];
    pp_save_gdnconv = G.dset_gdnconv; G.dset_gdnconv = G.pp_gdnconv[s];
    pp_save_gdncv = G.dset_gdncv; G.dset_gdncv = G.pp_gdncv[s];
    pp_save_gdnout = G.gdnf_out; G.gdnf_out = G.pp_gdnout[s];
    G.cmd_rec = G.cmd_pp[s];
    pp_sets_pushed = 1;
}

static void pp_pop_sets(void) {
    if (!pp_sets_pushed) return;
    G.dset_nrm = pp_save_nrm;
    G.gqaf_q = pp_save_gq; G.gqaf_k = pp_save_gk; G.gqaf_v = pp_save_gv; G.gqaf_out = pp_save_go;
    G.dset_qkv = pp_save_qkv; G.dset_gqa = pp_save_gqa;
    G.dset_rrnz = pp_save_rrnz; G.dset_route_rt = pp_save_route; G.dset_topk = pp_save_topk;
    G.dset_moe_pack = pp_save_pack; G.dset_macc = pp_save_macc;
    G.dset_gbb = pp_save_gbb; G.dset_gba = pp_save_gba; G.dset_gdp = pp_save_gdp;
    G.gdnf_gqkv = pp_save_gqkv; G.gdnf_gz = pp_save_gz;
    G.dset_gdnconv = pp_save_gdnconv; G.dset_gdncv = pp_save_gdncv; G.gdnf_out = pp_save_gdnout;
    G.cmd_rec = G.cmd;
    pp_sets_pushed = 0;
}

static int pp_drain_all(void) {
    for (int s = 0; s < 2; s++)
        if (!pp_reclaim_slot(s)) return 0;
    G.pp_rec = 0;
    return 1;
}

/* Bind moe_ix runtime buffers once; non-UAB bindings must not change while a CB uses them. */
static void moe_ix_bind_runtime(void) {
    if (!G.dset_moe_ix || !G.route_nrm.buf || !G.route_idx.buf || !G.route_val.buf ||
        !G.eg_ix_shgate || !G.eg_ix_valid || !G.eg_h.buf || !G.eg_y.buf)
        return;
    VkBuffer want[7] = {G.route_nrm.buf, G.route_idx.buf, G.route_val.buf, G.eg_ix_shgate,
                        G.eg_ix_valid, G.eg_h.buf, G.eg_y.buf};
    if (G.moe_ix_rt_bound && memcmp(G.moe_ix_rt_bufs, want, sizeof(want)) == 0) return;
    if (G.pp_inflight[0] || G.pp_inflight[1]) return; /* keep prior binding */
    VkDescriptorBufferInfo bix[7] = {
        {.buffer = want[0], .range = VK_WHOLE_SIZE},
        {.buffer = want[1], .range = VK_WHOLE_SIZE},
        {.buffer = want[2], .range = VK_WHOLE_SIZE},
        {.buffer = want[3], .range = VK_WHOLE_SIZE},
        {.buffer = want[4], .range = VK_WHOLE_SIZE},
        {.buffer = want[5], .range = VK_WHOLE_SIZE},
        {.buffer = want[6], .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_moe_ix, 7, bix);
    memcpy(G.moe_ix_rt_bufs, want, sizeof(want));
    G.moe_ix_rt_bound = 1;
}

/* Start ping-pong recording for a fused route layer. 1 = using cmd_pp[slot]. */
static int route_pipe_begin_pp(int ln, int topk) {
    G.pp_rec = 0;
    G.cmd_rec = G.cmd;
    if (!coli_vk_moe_ix_available()) return 0;
    if (!G.pp_pool && !moe_ix_pp_ensure_sets()) return 0;
    if (!G.moe_ix_pp || !G.pp_pool) return 0;
    int slot = pp_acquire(ln, topk);
    if (slot < 0) return 0;
    pp_push_sets(slot);
    return 1;
}

int coli_vk_stream_ix_hist(int layer, int *idx_out, int K) {
    if (layer < 0 || layer >= VK_KV_LAYERS || !idx_out || K < 1) return 0;
    if (!G.ix_hist_ready[layer]) return 0;
    int n = G.ix_hist_k[layer];
    if (n < 1) n = K;
    if (n > K) n = K;
    if (n > 64) n = 64;
    memcpy(idx_out, G.ix_hist[layer], (size_t)n * sizeof(int));
    return n;
}

int coli_vk_stream_ix_pending(void) {
    return G.ix_hist_pending ? 1 : 0;
}

int coli_vk_eg_table_init(int n_layers, int E, int D, int I, int gs) {
    if (!G.ready || !G.moe_ix_hw || !G.shader_moe_ix || n_layers < 1 || n_layers > VK_KV_LAYERS ||
        E < 1 || D < 1 || I < 1 || gs < 1)
        return 0;
    int nslots = n_layers * (E + 1);
    if (!G.dset_moe_ix) {
        if (!build_moe_ix_pipeline(nslots, G.shader_moe_ix, &G.dsl_moe_ix, &G.plyt_moe_ix,
                                   &G.pipe_moe_ix, &G.dpool_moe_ix, &G.dset_moe_ix))
            return 0;
    } else if (G.eg_ix_nslots != nslots)
        return 0;
    G.eg_ix_nlayers = n_layers; G.eg_ix_E = E; G.eg_ix_D = D; G.eg_ix_I = I; G.eg_ix_gs = gs;
    G.eg_ix_nslots = nslots;
    for (int i = 0; i < VK_KV_LAYERS; i++) G.eg_ix_sh_ok[i] = 0;
    size_t vb = (size_t)nslots * sizeof(uint32_t), sb = (size_t)n_layers * (size_t)D * 4;
    if (!G.eg_ix_valid) {
        if (!alloc_hostvis(vb, &G.eg_ix_valid, &G.eg_ix_validm, &G.eg_ix_validp) ||
            !alloc_hostvis(sb, &G.eg_ix_shgate, &G.eg_ix_shgatem, &G.eg_ix_shgatep) ||
            !alloc_hostvis(16, &G.eg_ix_dummy_w, &G.eg_ix_dummym, NULL) ||
            !alloc_hostvis(16, &G.eg_ix_dummy_s, &G.eg_ix_dummy_sm, NULL)) {
            return 0;
        }
        G.eg_ix_valid_cpu = (uint32_t *)G.eg_ix_validp;
        memset(G.eg_ix_validp, 0, vb);
        memset(G.eg_ix_shgatep, 0, sb);
        for (int di = 0; di < nslots; di++) eg_ix_bind_dummy_slot(di);
        fprintf(stderr, "[VK] eg_table: %d slots (%d layers × %d+1 shared)\n", nslots, n_layers, E);
    } else {
        memset(G.eg_ix_validp, 0, vb);
        for (int i = 0; i < VK_KV_LAYERS; i++) G.eg_ix_sh_ok[i] = 0;
        for (int di = 0; di < nslots; di++) eg_ix_bind_dummy_slot(di);
    }
    /* Pre-size route/eg scratches so ping-pong never reallocs mid-token (would invalidate dset_moe_ix). */
    {
        size_t xb = (size_t)D * 4, kb = 128 * 4 < 256 ? 256 : 128 * 4;
        size_t nslot = 65; /* topk<=64 + shared */
        size_t hb = nslot * (size_t)I * 4, yb = nslot * (size_t)D * 4, wb = nslot * 4;
        if (wb < 256) wb = 256;
        (void)scratch_reserve_mt(&G.route_nrm, xb, G.memtype_cached);
        (void)scratch_reserve_mt(&G.route_idx, kb, G.memtype_cached);
        (void)scratch_reserve_mt(&G.route_val, kb, G.memtype_cached);
        (void)scratch_reserve(&G.eg_h, hb);
        (void)scratch_reserve(&G.eg_y, yb);
        (void)scratch_reserve(&G.moe_w, wb);
        moe_ix_bind_runtime();
    }
    (void)moe_ix_pp_ensure_sets();
    return 1;
}

int coli_vk_eg_table_set(int layer, int eid, ColiVkTensor *g, ColiVkTensor *u, ColiVkTensor *d) {
    if (!G.dset_moe_ix || layer < 0 || layer >= G.eg_ix_nlayers || eid < 0 || eid >= G.eg_ix_E)
        return 0;
    if (!g || !u || !d || g->fmt != 6 || u->fmt != 6 || d->fmt != 6) return 0;
    int di = layer * (G.eg_ix_E + 1) + eid;
    eg_ix_bind_array_slot(di, g, u, d);
    G.eg_ix_valid_cpu[di] = 1;
    return 1;
}

int coli_vk_eg_table_set_shared(int layer, ColiVkTensor *g, ColiVkTensor *u, ColiVkTensor *d,
                                const float *sh_gate, int D) {
    if (!G.dset_moe_ix || layer < 0 || layer >= G.eg_ix_nlayers || D != G.eg_ix_D)
        return 0;
    if (!g || !u || !d || !sh_gate || g->fmt != 6) return 0;
    int di = layer * (G.eg_ix_E + 1) + G.eg_ix_E;
    eg_ix_bind_array_slot(di, g, u, d);
    G.eg_ix_grw = g->rowWords; G.eg_ix_drw = d->rowWords;
    memcpy((float *)G.eg_ix_shgatep + (size_t)layer * (size_t)D, sh_gate, (size_t)D * 4);
    G.eg_ix_valid_cpu[di] = 1;
    G.eg_ix_sh_ok[layer] = 1;
    return 1;
}

int coli_vk_gdn_ba_weight(int layer, const float *alog, const float *dtb,
                          const float *gnorm, int VH, int VD) {
    if (!G.ready || layer < 0 || layer >= VK_KV_LAYERS || VH < 1 || VD < 1 ||
        !alog || !dtb || !gnorm) return 0;
    size_t vb = (size_t)VH * 4, db = (size_t)VD * 4;
    if (!G.gdn_alog[layer] || G.gdn_ba_VH[layer] != VH || G.gdn_ba_VD[layer] != VD) {
        if (G.gdn_alog[layer]) {
            vkDestroyBuffer(G.dev, G.gdn_alog[layer], NULL); vkFreeMemory(G.dev, G.gdn_alogm[layer], NULL);
            vkDestroyBuffer(G.dev, G.gdn_dtb[layer], NULL); vkFreeMemory(G.dev, G.gdn_dtbm[layer], NULL);
            vkDestroyBuffer(G.dev, G.gdn_gn[layer], NULL); vkFreeMemory(G.dev, G.gdn_gnm[layer], NULL);
            G.gdn_alog[layer] = VK_NULL_HANDLE;
        }
        float p0 = G.prio; G.prio = 1.0f;
        int ok = alloc_hostvis(vb, &G.gdn_alog[layer], &G.gdn_alogm[layer], &G.gdn_alogp[layer])
              && alloc_hostvis(vb, &G.gdn_dtb[layer], &G.gdn_dtbm[layer], &G.gdn_dtbp[layer])
              && alloc_hostvis(db, &G.gdn_gn[layer], &G.gdn_gnm[layer], &G.gdn_gnp[layer]);
        G.prio = p0;
        if (!ok) { G.gdn_alog[layer] = VK_NULL_HANDLE; return 0; }
        G.gdn_ba_VH[layer] = VH; G.gdn_ba_VD[layer] = VD;
    }
    memcpy(G.gdn_alogp[layer], alog, vb);
    memcpy(G.gdn_dtbp[layer], dtb, vb);
    memcpy(G.gdn_gnp[layer], gnorm, db);
    return 1;
}

/* residual+post_ln+router+softmax_topk; resid in-place on stream (binding 0 == 3). */
static void route_tail_record_stream(ColiVkTensor *router, int ln_layer, int D, int E, int K, float eps) {
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(G.cmd_rec, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    VkDescriptorBufferInfo br[5] = {
        {.buffer = G.stream.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.att_delta.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.qln_post[ln_layer], .range = VK_WHOLE_SIZE},
        {.buffer = G.stream.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.route_nrm.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_rrnz, 5, br);
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_rrnz);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_rrnz, 0, 1, &G.dset_rrnz, 0, NULL);
    struct PCN pn = {1, D, eps};
    vkCmdPushConstants(G.cmd_rec, G.plyt_rrnz, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pn), &pn);
    vkCmdDispatch(G.cmd_rec, 1, 1, 1);
    vkCmdPipelineBarrier(G.cmd_rec, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    VkDescriptorBufferInfo bt[4] = {
        {.buffer = G.route_nrm.buf, .range = VK_WHOLE_SIZE}, {.buffer = router->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = router->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y2.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_route_rt, 4, bt);
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset_route_rt, 0, NULL);
    struct PC pc = {router->fmt, 1, D, E, router->rowWords, router->gs};
    vkCmdPushConstants(G.cmd_rec, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd_rec, (uint32_t)((E + 7) / 8), 1, 1);
    vkCmdPipelineBarrier(G.cmd_rec, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    /* Pack mix weights into moe_w here (single WG) so moe_ix can skip moe_pack_w. */
    int do_pack = 0;
    if (coli_vk_moe_ix_available() && ln_layer >= 0 && ln_layer < G.eg_ix_nlayers &&
        G.eg_ix_sh_ok[ln_layer] && G.eg_ix_shgate) {
        size_t wb = (size_t)(K + 1) * 4;
        if (scratch_reserve(&G.moe_w, wb < 256 ? 256 : wb)) do_pack = 1;
    }
    VkBuffer dummy = G.route_nrm.buf;
    VkDescriptorBufferInfo btk[6] = {
        {.buffer = G.y2.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.route_idx.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.route_val.buf, .range = VK_WHOLE_SIZE},
        {.buffer = do_pack ? G.route_nrm.buf : dummy, .range = VK_WHOLE_SIZE},
        {.buffer = do_pack ? G.eg_ix_shgate : dummy,
         .offset = do_pack ? (VkDeviceSize)(size_t)ln_layer * (size_t)D * 4 : 0,
         .range = do_pack ? (VkDeviceSize)(size_t)D * 4 : VK_WHOLE_SIZE},
        {.buffer = do_pack ? G.moe_w.buf : dummy, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_topk, 6, btk);
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_topk);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_topk, 0, 1, &G.dset_topk, 0, NULL);
    int ibase = G.pp_rec ? G.pp_slot * 64 : 0;
    struct PCTopk ptk = {E, K, ibase, do_pack ? D : 0, do_pack};
    vkCmdPushConstants(G.cmd_rec, G.plyt_topk, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ptk), &ptk);
    vkCmdDispatch(G.cmd_rec, 1, 1, 1);
}

static int route_try_moe_ix_fused(int ln_layer, int D, int E, int topk) {
    if (!coli_vk_moe_ix_available() || !G.dset_moe_ix || G.eg_ix_nslots < 1) return 0;
    if (ln_layer < 0 || ln_layer >= G.eg_ix_nlayers || !G.eg_ix_sh_ok[ln_layer]) return 0;
    /* Async pp: prior layer is ordered by eg_sem; skip host eg fence. */
    if (!G.pp_rec && !stream_wait_eg()) return 0;
    int K = topk, do_sh = 1, nslot = K + do_sh;
    int ibase = G.pp_rec ? G.pp_slot * 64 : 0;
    if (G.eg_ix_I < 1 || G.eg_ix_grw < 1 || G.eg_ix_drw < 1) return 0;
    size_t hb = (size_t)nslot * (size_t)G.eg_ix_I * 4, yb = (size_t)nslot * (size_t)D * 4;
    size_t wb = (size_t)nslot * 4;
    if (!scratch_reserve(&G.eg_h, hb) || !scratch_reserve(&G.eg_y, yb) ||
        !scratch_reserve(&G.moe_w, wb < 256 ? 256 : wb)) return 0;

    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(G.cmd_rec, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);

    moe_ix_bind_runtime();
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_moe_ix);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_moe_ix, 0, 1, &G.dset_moe_ix, 0, NULL);
    int base = ln_layer * (E + 1);
    struct PCMoeIx pc0 = {0, K, D, G.eg_ix_I, E, base, G.eg_ix_grw, G.eg_ix_drw, G.eg_ix_gs, do_sh, ibase};
    vkCmdPushConstants(G.cmd_rec, G.plyt_moe_ix, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc0), &pc0);
    vkCmdDispatch(G.cmd_rec, (uint32_t)((G.eg_ix_I + 7) / 8), (uint32_t)nslot, 1);
    vkCmdPipelineBarrier(G.cmd_rec, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    struct PCMoeIx pc1 = {1, K, D, G.eg_ix_I, E, base, G.eg_ix_grw, G.eg_ix_drw, G.eg_ix_gs, do_sh, ibase};
    vkCmdPushConstants(G.cmd_rec, G.plyt_moe_ix, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), &pc1);
    vkCmdDispatch(G.cmd_rec, (uint32_t)((D + 7) / 8), (uint32_t)nslot, 1);
    vkCmdPipelineBarrier(G.cmd_rec, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);

    /* Mix weights already packed by softmax_topk into moe_w. */
    VkDescriptorBufferInfo bacc[3] = {
        {.buffer = G.eg_y.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.moe_w.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.stream.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_macc, 3, bacc);
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_macc);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_macc, 0, 1, &G.dset_macc, 0, NULL);
    struct PCRep pa = {nslot, D};
    vkCmdPushConstants(G.cmd_rec, G.plyt_macc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pa), &pa);
    vkCmdDispatch(G.cmd_rec, (uint32_t)((D + 255) / 256), 1, 1);
    G.moe_ix_fused = 1;
    { static int once;
      if (!once) { fprintf(stderr, "[VK] moe_ix fused into route submit (descriptor-indexed MoE)\n"); once = 1; } }
    return 1;
}

static int route_submit_wait_topk(int *idx_out, float *val_out, int K, const char *tag) {
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    int fused_sig = G.moe_ix_fused;
    int use_pp = G.pp_rec && G.moe_ix_pp;
    int async = fused_sig && use_pp;
    int slot = G.pp_slot;
    int ibase = use_pp ? slot * 64 : 0;
    VkCommandBuffer cbuf = use_pp ? G.cmd_pp[slot] : G.cmd_rec;
    VkFence fence = use_pp ? G.fence_pp[slot] : G.fence;
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &cbuf};
    int waited_eg = G.eg_sem_armed;
    if (G.eg_sem_armed) {
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &G.eg_sem;
        si.pWaitDstStageMask = &wait_stage;
        G.eg_sem_armed = 0;
    }
    if (fused_sig) {
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &G.eg_sem;
    }
    G.moe_ix_fused_last = fused_sig;
    G.moe_ix_fused = 0;
    double ts = dp_t0();
    VKCHECK(vkResetFences(G.dev, 1, &fence), "resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, fence), "queueSubmit");
    dp_sub_add(ts);
    if (async) {
        G.pp_tsq[slot] = G.ts_base_rec;   /* collected once this slot's fence signals */
        G.pp_inflight[slot] = 1;
        G.ix_hist_pending = 1;
        if (fused_sig) G.eg_sem_armed = 1;
        pp_pop_sets();
        G.pp_rec = 0;
        pp_advance();
        memset(idx_out, 0, (size_t)K * sizeof(int));
        memset(val_out, 0, (size_t)K * sizeof(float));
        G.cmd_ready = 0; G.bound_tensor = NULL;
        return 1;
    }
    double tw = dp_t0();
    if (vk_fence_wait_loud(fence, tag) != VK_SUCCESS) { G.ready = 0; return 0; }
    dp_wait_add(tw);
    dp_ts_collect(G.ts_base_rec);
    if (fused_sig) G.eg_sem_armed = 1;
    if (waited_eg && G.eg_inflight) {
        if (vk_fence_wait(G.eg_fence) != VK_SUCCESS) { G.ready = 0; return 0; }
        G.eg_inflight = 0;
    }
    memcpy(idx_out, (int *)G.route_idx.ptr + ibase, (size_t)K * sizeof(int));
    memcpy(val_out, (float *)G.route_val.ptr + ibase, (size_t)K * sizeof(float));
    if (use_pp) {
        G.pp_inflight[slot] = 0;
        G.pp_layer[slot] = -1;
        pp_pop_sets();
        G.pp_rec = 0;
        pp_advance();
    }
    G.cmd_ready = 0; G.bound_tensor = NULL;
    return 1;
}

int coli_vk_gqa_full_route_pipe(int layer, int D,
                                ColiVkTensor *gq_t, ColiVkTensor *gk_t, ColiVkTensor *gv_t, ColiVkTensor *out_t,
                                ColiVkTensor *router, int ln_layer,
                                int *idx_out, float *val_out, int topk,
                                int H, int KH, int hd, int rot, int pos_base, int max_t,
                                float eps, float theta, int Dout, int E) {
    if (!coli_vk_stream_available() || !coli_vk_gqa_full_available() || !G.stream_live) return 0;
    if (!gq_t || !gk_t || !gv_t || !out_t || !router || !idx_out || !val_out) return 0;
    if (hd > 256 || (H % KH) != 0 || layer < 0 || layer >= VK_KV_LAYERS || Dout < 1 || E < 1) return 0;
    if (topk < 1 || topk > 64 || topk > E || E > 256) return 0;  /* softmax_topk.comp WG=256 */
    if (D != G.stream_D) return 0;
    if (ln_layer < 0 || ln_layer >= VK_KV_LAYERS || !G.qln_in[ln_layer] || G.qln_D[ln_layer] != D) return 0;
    if (!G.gqa_qnw[layer]) return 0;
    VkKvLayer *kv = &G.kv[layer];
    if (!kv->bl || kv->rows < KH * max_t || kv->K != hd || kv->R != hd) return 0;
    int qo = H * 2 * hd, ko = KH * hd, T = pos_base + 1, S = 1;
    if (gq_t->I != D || gq_t->O != qo || gk_t->I != D || gk_t->O != ko ||
        gv_t->I != D || gv_t->O != ko || out_t->I != H * hd || out_t->O != Dout ||
        router->I != D || router->O != E) return 0;
    (void)route_pipe_begin_pp(ln_layer, topk);
    size_t xb = (size_t)D * 4, cb = (size_t)H * hd * 4, pb = (size_t)E * 4;
    size_t kb = G.pp_rec ? (size_t)128 * 4 : (size_t)topk * 4;
    if (kb < 256) kb = 256;
    if (!scratch_reserve(&G.x, xb) ||
        !scratch_reserve(&G.gqaf_qg, (size_t)qo*4) || !scratch_reserve(&G.gqaf_kb, (size_t)ko*4) ||
        !scratch_reserve(&G.gqaf_vb, (size_t)ko*4) || !scratch_reserve(&G.att_ctx, cb) ||
        !scratch_reserve(&G.att_delta, xb) ||
        !scratch_reserve_mt(&G.route_nrm, xb, G.memtype_cached) ||
        !scratch_reserve(&G.y2, pb) ||   /* logits stay on device; only idx/val come back */
        !scratch_reserve_mt(&G.route_idx, kb, G.memtype_cached) ||
        !scratch_reserve_mt(&G.route_val, kb, G.memtype_cached)) {
        if (G.pp_rec) { pp_pop_sets(); G.pp_rec = 0; G.pp_layer[G.pp_slot] = -1; }
        return 0;
    }

    VkDescriptorBufferInfo bin[3] = {
        {.buffer = G.stream.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.qln_in[ln_layer], .range = VK_WHOLE_SIZE},
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_nrm, 3, bin);
    VkDescriptorBufferInfo bq[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gq_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gq_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gqaf_qg.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gqaf_q, 4, bq);
    VkDescriptorBufferInfo bk[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gk_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gk_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gqaf_kb.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gqaf_k, 4, bk);
    VkDescriptorBufferInfo bv[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gv_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gv_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gqaf_vb.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gqaf_v, 4, bv);
    VkDescriptorBufferInfo bn[7] = {{.buffer=G.gqaf_qg.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gqaf_kb.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gqaf_vb.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gqa_qnw[layer],.range=VK_WHOLE_SIZE},{.buffer=G.gqa_knw[layer],.range=VK_WHOLE_SIZE},{.buffer=kv->bl,.range=VK_WHOLE_SIZE},{.buffer=kv->br,.range=VK_WHOLE_SIZE}};
    wr_desc(G.dset_qkv, 7, bn);
    VkDescriptorBufferInfo ba[4] = {{.buffer=G.gqaf_qg.buf,.range=VK_WHOLE_SIZE},{.buffer=kv->bl,.range=VK_WHOLE_SIZE},{.buffer=kv->br,.range=VK_WHOLE_SIZE},{.buffer=G.att_ctx.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.dset_gqa, 4, ba);
    VkDescriptorBufferInfo bo[4] = {{.buffer=G.att_ctx.buf,.range=VK_WHOLE_SIZE},{.buffer=out_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=out_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.att_delta.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gqaf_out, 4, bo);

    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    VKCHECK(vkResetCommandBuffer(G.cmd_rec, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd_rec, &begin), "beginCmd");
    dp_ts_begin(G.cmd_rec, G.pp_rec ? G.pp_slot * DP_TS_N : 2 * DP_TS_N);
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_nrmz);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_nrm, 0, 1, &G.dset_nrm, 0, NULL);
    struct PCN pin = {S, D, eps};
    vkCmdPushConstants(G.cmd_rec, G.plyt_nrm, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pin), &pin);
    vkCmdDispatch(G.cmd_rec, (uint32_t)S, 1, 1);
    vkCmdPipelineBarrier(G.cmd_rec, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    struct PC pq = {gq_t->fmt, S, D, qo, gq_t->rowWords, gq_t->gs};
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gqaf_q, 0, NULL);
    vkCmdPushConstants(G.cmd_rec, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pq), &pq);
    vkCmdDispatch(G.cmd_rec, (uint32_t)((qo + 7) / 8), (uint32_t)S, 1);
    struct PC pk = {gk_t->fmt, S, D, ko, gk_t->rowWords, gk_t->gs};
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gqaf_k, 0, NULL);
    vkCmdPushConstants(G.cmd_rec, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pk), &pk);
    vkCmdDispatch(G.cmd_rec, (uint32_t)((ko + 7) / 8), (uint32_t)S, 1);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gqaf_v, 0, NULL);
    vkCmdPushConstants(G.cmd_rec, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pk), &pk);
    vkCmdDispatch(G.cmd_rec, (uint32_t)((ko + 7) / 8), (uint32_t)S, 1);
    vkCmdPipelineBarrier(G.cmd_rec, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_qkv);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_qkv, 0, 1, &G.dset_qkv, 0, NULL);
    struct PCQkv pn = {S, H, KH, hd, rot, pos_base, max_t, eps, theta};
    vkCmdPushConstants(G.cmd_rec, G.plyt_qkv, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pn), &pn);
    vkCmdDispatch(G.cmd_rec, (uint32_t)H, (uint32_t)S, 1);
    vkCmdPipelineBarrier(G.cmd_rec, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gqa);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gqa, 0, 1, &G.dset_gqa, 0, NULL);
    struct PCGqa pg = {S, H, KH, hd, max_t, 0, T, 1.0f / sqrtf((float)hd)};
    vkCmdPushConstants(G.cmd_rec, G.plyt_gqa, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pg), &pg);
    vkCmdDispatch(G.cmd_rec, (uint32_t)H, (uint32_t)S, 1);
    vkCmdPipelineBarrier(G.cmd_rec, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gqaf_out, 0, NULL);
    struct PC po = {out_t->fmt, S, H * hd, Dout, out_t->rowWords, out_t->gs};
    vkCmdPushConstants(G.cmd_rec, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(po), &po);
    vkCmdDispatch(G.cmd_rec, (uint32_t)((Dout + 7) / 8), (uint32_t)S, 1);
    dp_ts_mark(G.cmd_rec, 1);
    route_tail_record_stream(router, ln_layer, D, E, topk, eps);
    dp_ts_mark(G.cmd_rec, 2);
    G.moe_ix_fused = 0;
    /* Expert fmt lives in the eg_table (fmt=6), not on dense attn tensors. */
    (void)route_try_moe_ix_fused(ln_layer, D, E, topk);
    dp_ts_end(G.cmd_rec);
    VKCHECK(vkEndCommandBuffer(G.cmd_rec), "endCmd");
    return route_submit_wait_topk(idx_out, val_out, topk, "gqa_full_route_pipe");
}

int coli_vk_gdn_full_route_pipe(int layer, int D,
                                ColiVkTensor *gqkv_t, ColiVkTensor *gz_t,
                                ColiVkTensor *gba_t, ColiVkTensor *gbb_t,
                                ColiVkTensor *out_t, ColiVkTensor *router, int ln_layer,
                                int *idx_out, float *val_out, int topk,
                                int KH, int KD, int VH, int VD, int conv_dim, int conv_k,
                                float eps, int Dout, int E) {
    if (!coli_vk_stream_available() || !coli_vk_gdn_full_available() || !G.stream_live) return 0;
    if (!gqkv_t || !gz_t || !gba_t || !gbb_t || !out_t || !router || !idx_out || !val_out) return 0;
    if (layer < 0 || layer >= VK_KV_LAYERS || VD > 512 || KD > 256 || (VH % KH) != 0 || E < 1) return 0;
    if (topk < 1 || topk > 64 || topk > E || E > 256) return 0;
    if (D != G.stream_D) return 0;
    if (ln_layer < 0 || ln_layer >= VK_KV_LAYERS || !G.qln_in[ln_layer] || G.qln_D[ln_layer] != D) return 0;
    if (!G.gdn_st[layer] || !G.gdn_ring[layer] || !G.gdn_cw[layer]) return 0;
    if (!G.gdn_alog[layer] || G.gdn_ba_VH[layer] != VH || G.gdn_ba_VD[layer] != VD) return 0;
    int key_dim = KH * KD, value_dim = VH * VD;
    if (gqkv_t->I != D || gqkv_t->O != conv_dim || gz_t->I != D || gz_t->O != value_dim ||
        gba_t->I != D || gba_t->O != VH || gbb_t->I != D || gbb_t->O != VH ||
        out_t->I != value_dim || out_t->O != Dout || conv_dim != 2 * key_dim + value_dim ||
        router->I != D || router->O != E) return 0;
    (void)route_pipe_begin_pp(ln_layer, topk);
    size_t xb = (size_t)D * 4, prb = (size_t)(2 * VH + VD) * 4, erb = (size_t)E * 4;
    size_t bab = (size_t)VH * 4;
    size_t kb = G.pp_rec ? (size_t)128 * 4 : (size_t)topk * 4;
    if (kb < 256) kb = 256;
    if (!scratch_reserve(&G.x, xb) ||
        !scratch_reserve(&G.gdnf_qkv, (size_t)conv_dim * 4) || !scratch_reserve(&G.gdnf_z, (size_t)value_dim * 4) ||
        !scratch_reserve(&G.gdnf_cv, (size_t)conv_dim * 4) || !scratch_reserve(&G.gdn_y, (size_t)value_dim * 4) ||
        !scratch_reserve(&G.gdnf_pr, prb) || !scratch_reserve(&G.att_delta, xb) ||
        !scratch_reserve(&G.gdn_ba, bab) || !scratch_reserve(&G.gdn_bb, bab) ||
        !scratch_reserve_mt(&G.route_nrm, xb, G.memtype_cached) ||
        !scratch_reserve(&G.y2, erb) ||
        !scratch_reserve_mt(&G.route_idx, kb, G.memtype_cached) ||
        !scratch_reserve_mt(&G.route_val, kb, G.memtype_cached)) {
        if (G.pp_rec) { pp_pop_sets(); G.pp_rec = 0; G.pp_layer[G.pp_slot] = -1; }
        return 0;
    }

    /* NOTE: do NOT cache/resubmit GDN route CBs across layers. With moe_ix ping-pong
     * only cmd_pp[0|1] exist; even layers share slot 0 and overwrite each other's
     * recorded CB. A (slot,layer) ready bit would then resubmit the wrong layer's
     * work (garbled decode). Re-record every call. */

    /* in_ln(stream) -> G.x; b/a from G.x; pack params; then GDN full + route tail */
    VkDescriptorBufferInfo bin[3] = {
        {.buffer = G.stream.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.qln_in[ln_layer], .range = VK_WHOLE_SIZE},
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_nrm, 3, bin);
    VkDescriptorBufferInfo bb[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gbb_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gbb_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gdn_bb.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.dset_gbb, 4, bb);
    VkDescriptorBufferInfo ba[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gba_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gba_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gdn_ba.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.dset_gba, 4, ba);
    VkDescriptorBufferInfo bp[6] = {
        {.buffer=G.gdn_ba.buf,.range=VK_WHOLE_SIZE}, {.buffer=G.gdn_bb.buf,.range=VK_WHOLE_SIZE},
        {.buffer=G.gdn_alog[layer],.range=VK_WHOLE_SIZE}, {.buffer=G.gdn_dtb[layer],.range=VK_WHOLE_SIZE},
        {.buffer=G.gdn_gn[layer],.range=VK_WHOLE_SIZE}, {.buffer=G.gdnf_pr.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.dset_gdp, 6, bp);
    VkDescriptorBufferInfo bqkv[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gqkv_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gqkv_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gdnf_qkv.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gdnf_gqkv, 4, bqkv);
    VkDescriptorBufferInfo bz[4] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=gz_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=gz_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.gdnf_z.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gdnf_gz, 4, bz);
    VkDescriptorBufferInfo bc[4] = {{.buffer=G.gdnf_qkv.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gdn_cw[layer],.range=VK_WHOLE_SIZE},{.buffer=G.gdn_ring[layer],.range=VK_WHOLE_SIZE},{.buffer=G.gdnf_cv.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.dset_gdnconv, 4, bc);
    VkDescriptorBufferInfo bd[5] = {{.buffer=G.gdnf_cv.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gdnf_z.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gdnf_pr.buf,.range=VK_WHOLE_SIZE},{.buffer=G.gdn_st[layer],.range=VK_WHOLE_SIZE},{.buffer=G.gdn_y.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.dset_gdncv, 5, bd);
    VkDescriptorBufferInfo bo[4] = {{.buffer=G.gdn_y.buf,.range=VK_WHOLE_SIZE},{.buffer=out_t->wbuf,.range=VK_WHOLE_SIZE},{.buffer=out_t->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.att_delta.buf,.range=VK_WHOLE_SIZE}};
    wr_desc(G.gdnf_out, 4, bo);

    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    VKCHECK(vkResetCommandBuffer(G.cmd_rec, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd_rec, &begin), "beginCmd");
    dp_ts_begin(G.cmd_rec, G.pp_rec ? G.pp_slot * DP_TS_N : 2 * DP_TS_N);
    /* 0) in_ln */
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_nrmz);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_nrm, 0, 1, &G.dset_nrm, 0, NULL);
    struct PCN pin = {1, D, eps};
    vkCmdPushConstants(G.cmd_rec, G.plyt_nrm, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pin), &pin);
    vkCmdDispatch(G.cmd_rec, 1, 1, 1);
    vkCmdPipelineBarrier(G.cmd_rec, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    /* 1) b/a projections + pack decay|beta|gnorm */
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset_gbb, 0, NULL);
    struct PC pcb = {gbb_t->fmt, 1, D, VH, gbb_t->rowWords, gbb_t->gs};
    vkCmdPushConstants(G.cmd_rec, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcb), &pcb);
    vkCmdDispatch(G.cmd_rec, (uint32_t)((VH + 7) / 8), 1, 1);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset_gba, 0, NULL);
    struct PC pca = {gba_t->fmt, 1, D, VH, gba_t->rowWords, gba_t->gs};
    vkCmdPushConstants(G.cmd_rec, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pca), &pca);
    vkCmdDispatch(G.cmd_rec, (uint32_t)((VH + 7) / 8), 1, 1);
    vkCmdPipelineBarrier(G.cmd_rec, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gdp);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gdp, 0, 1, &G.dset_gdp, 0, NULL);
    struct PCGdp pgdp = {VH, VD};
    vkCmdPushConstants(G.cmd_rec, G.plyt_gdp, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pgdp), &pgdp);
    vkCmdDispatch(G.cmd_rec, (uint32_t)(((VH > VD ? VH : VD) + 255) / 256), 1, 1);
    vkCmdPipelineBarrier(G.cmd_rec, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    /* 2) GDN full (same as gdn_full_route body) */
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gdnf_gqkv, 0, NULL);
    struct PC pq = {gqkv_t->fmt, 1, D, conv_dim, gqkv_t->rowWords, gqkv_t->gs};
    vkCmdPushConstants(G.cmd_rec, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pq), &pq);
    vkCmdDispatch(G.cmd_rec, (uint32_t)((conv_dim + 7) / 8), 1, 1);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gdnf_gz, 0, NULL);
    struct PC pz = {gz_t->fmt, 1, D, value_dim, gz_t->rowWords, gz_t->gs};
    vkCmdPushConstants(G.cmd_rec, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pz), &pz);
    vkCmdDispatch(G.cmd_rec, (uint32_t)((value_dim + 7) / 8), 1, 1);
    vkCmdPipelineBarrier(G.cmd_rec, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gdnconv);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gdnconv, 0, 1, &G.dset_gdnconv, 0, NULL);
    struct PCConv pcv = {conv_dim, conv_k};
    vkCmdPushConstants(G.cmd_rec, G.plyt_gdnconv, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcv), &pcv);
    vkCmdDispatch(G.cmd_rec, (uint32_t)((conv_dim + 255) / 256), 1, 1);
    vkCmdPipelineBarrier(G.cmd_rec, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gdncv);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gdncv, 0, 1, &G.dset_gdncv, 0, NULL);
    struct PCGdnCv pd = {(uint32_t)KH, (uint32_t)KD, (uint32_t)VH, (uint32_t)VD, (uint32_t)key_dim, eps};
    vkCmdPushConstants(G.cmd_rec, G.plyt_gdncv, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pd), &pd);
    vkCmdDispatch(G.cmd_rec, (uint32_t)VH, 1, 1);
    vkCmdPipelineBarrier(G.cmd_rec, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd_rec, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.gdnf_out, 0, NULL);
    struct PC po = {out_t->fmt, 1, value_dim, Dout, out_t->rowWords, out_t->gs};
    vkCmdPushConstants(G.cmd_rec, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(po), &po);
    vkCmdDispatch(G.cmd_rec, (uint32_t)((Dout + 7) / 8), 1, 1);
    dp_ts_mark(G.cmd_rec, 1);
    route_tail_record_stream(router, ln_layer, D, E, topk, eps);
    dp_ts_mark(G.cmd_rec, 2);
    G.moe_ix_fused = 0;
    (void)route_try_moe_ix_fused(ln_layer, D, E, topk);
    dp_ts_end(G.cmd_rec);
    VKCHECK(vkEndCommandBuffer(G.cmd_rec), "endCmd");
    return route_submit_wait_topk(idx_out, val_out, topk, "gdn_full_route_pipe");
}

/* Ensure the shared eg descriptor pool/sets exist (once). */
static int eg_pool_ensure(void) {
    if (G.eg_pool) return 1;
    int n4 = G.pipe_qr ? 64 : 0, nq = G.pipe_qr ? 1 : 0;
    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = (uint32_t)(64*6 + 64*4 + n4*7 + n4*5 + nq*3)};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = (uint32_t)(128 + 2*n4 + nq), .poolSizeCount = 1, .pPoolSizes = &ps};
    VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.eg_pool), "eg descPool");
    VkDescriptorSetLayout lg[64], ld[64], lg4[64], ld4[64];
    for (int c = 0; c < 64; c++) { lg[c] = G.dsl_gu; ld[c] = G.dsl; lg4[c] = G.dsl_gu4; ld4[c] = G.dsl_dn4; }
    VkDescriptorSetAllocateInfo ag = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = G.eg_pool, .descriptorSetCount = 64, .pSetLayouts = lg};
    VkDescriptorSetAllocateInfo ad = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = G.eg_pool, .descriptorSetCount = 64, .pSetLayouts = ld};
    VKCHECK(vkAllocateDescriptorSets(G.dev, &ag, G.eg_gu), "eg gu sets");
    VKCHECK(vkAllocateDescriptorSets(G.dev, &ad, G.eg_dn), "eg dn sets");
    if (n4) {
        VkDescriptorSetAllocateInfo ag4 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.eg_pool, .descriptorSetCount = 64, .pSetLayouts = lg4};
        VkDescriptorSetAllocateInfo ad4 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.eg_pool, .descriptorSetCount = 64, .pSetLayouts = ld4};
        VkDescriptorSetAllocateInfo aq = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.eg_pool, .descriptorSetCount = 1, .pSetLayouts = &G.dsl_qr};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &ag4, G.eg_gu4), "eg gu4 sets");
        VKCHECK(vkAllocateDescriptorSets(G.dev, &ad4, G.eg_dn4), "eg dn4 sets");
        VKCHECK(vkAllocateDescriptorSets(G.dev, &aq, &G.dset_qr_x), "eg qr-x set");
    }
    G.eg_nsets = 64;
    return 1;
}

/* Bind one expert's weight buffers into slot c; skip UpdateDescriptorSets when unchanged. */
static void eg_pipe_bind_slot(int c, int dp4a, int dp4a_dn, int D, int I,
                              ColiVkTensor *g, ColiVkTensor *u, ColiVkTensor *dn, int force) {
    int same = !force &&
        G.eg_cache_gw[c] == g->wbuf && G.eg_cache_gs[c] == g->sbuf &&
        G.eg_cache_uw[c] == u->wbuf && G.eg_cache_us[c] == u->sbuf &&
        G.eg_cache_dw[c] == dn->wbuf && G.eg_cache_ds[c] == dn->sbuf;
    if (same) return;
    VkDeviceSize xo = (VkDeviceSize)c * D * 4, ho = (VkDeviceSize)c * I * 4, yo = (VkDeviceSize)c * D * 4;
    if (dp4a) {
        VkDescriptorBufferInfo g4[7] = {
            {G.eg_xq.buf, 0, VK_WHOLE_SIZE}, {G.eg_xs.buf, 0, VK_WHOLE_SIZE},
            {g->wbuf, 0, VK_WHOLE_SIZE}, {g->sbuf, 0, VK_WHOLE_SIZE},
            {u->wbuf, 0, VK_WHOLE_SIZE}, {u->sbuf, 0, VK_WHOLE_SIZE},
            {G.eg_h.buf, ho, (VkDeviceSize)I * 4}};
        wr_desc(G.eg_gu4[c], 7, g4);
    } else {
        VkDescriptorBufferInfo gi[6] = {
            {G.eg_x.buf, xo, (VkDeviceSize)D * 4}, {g->wbuf, 0, VK_WHOLE_SIZE},
            {g->sbuf, 0, VK_WHOLE_SIZE}, {u->wbuf, 0, VK_WHOLE_SIZE},
            {u->sbuf, 0, VK_WHOLE_SIZE}, {G.eg_h.buf, ho, (VkDeviceSize)I * 4}};
        wr_desc(G.eg_gu[c], 6, gi);
    }
    if (dp4a_dn) {
        VkDescriptorBufferInfo d4[5] = {
            {G.eg_hq.buf, 0, VK_WHOLE_SIZE}, {G.eg_hs.buf, 0, VK_WHOLE_SIZE},
            {dn->wbuf, 0, VK_WHOLE_SIZE}, {dn->sbuf, 0, VK_WHOLE_SIZE},
            {G.eg_y.buf, yo, (VkDeviceSize)D * 4}};
        wr_desc(G.eg_dn4[c], 5, d4);
    } else {
        VkDescriptorBufferInfo di[4] = {
            {G.eg_h.buf, ho, (VkDeviceSize)I * 4}, {dn->wbuf, 0, VK_WHOLE_SIZE},
            {dn->sbuf, 0, VK_WHOLE_SIZE}, {G.eg_y.buf, yo, (VkDeviceSize)D * 4}};
        wr_desc(G.eg_dn[c], 4, di);
    }
    G.eg_cache_gw[c] = g->wbuf; G.eg_cache_gs[c] = g->sbuf;
    G.eg_cache_uw[c] = u->wbuf; G.eg_cache_us[c] = u->sbuf;
    G.eg_cache_dw[c] = dn->wbuf; G.eg_cache_ds[c] = dn->sbuf;
}

static void eg_pipe_bind_scratch(int count, int D, int I, int dp4a, int dp4a_dn,
                                 size_t xb, size_t hb) {
    if (dp4a) {
        VkDescriptorBufferInfo qx[3] = {{G.eg_x.buf, 0, (VkDeviceSize)xb},
                                        {G.eg_xq.buf, 0, (VkDeviceSize)(xb / 4)},
                                        {G.eg_xs.buf, 0, VK_WHOLE_SIZE}};
        wr_desc(G.dset_qr_x, 3, qx);
    }
    if (dp4a_dn) {
        VkDescriptorBufferInfo qh[3] = {{G.eg_h.buf, 0, (VkDeviceSize)hb},
                                        {G.eg_hq.buf, 0, (VkDeviceSize)(hb / 4)},
                                        {G.eg_hs.buf, 0, VK_WHOLE_SIZE}};
        wr_desc(G.dset_qr_h, 3, qh);
    }
    VkDescriptorBufferInfo brep[2] = {{G.route_nrm.buf, 0, VK_WHOLE_SIZE}, {G.eg_x.buf, 0, VK_WHOLE_SIZE}};
    wr_desc(G.dset_rep, 2, brep);
    /* Host fills moe_w on this path. */
    VkDescriptorBufferInfo bacc[3] = {{G.eg_y.buf, 0, VK_WHOLE_SIZE}, {G.moe_w.buf, 0, VK_WHOLE_SIZE},
                                      {G.stream.buf, 0, VK_WHOLE_SIZE}};
    wr_desc(G.dset_macc, 3, bacc);
    (void)count; (void)D; (void)I;
}

static int eg_pipe_record(int count, int D, int I, int fmt, int dfmt, int dp4a, int dp4a_dn,
                          ColiVkTensor *const *gates, ColiVkTensor *const *downs) {
    VKCHECK(vkResetCommandBuffer(G.eg_cmd, 0), "eg resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.eg_cmd, &begin), "eg beginCmd");
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_rep);
    vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_rep, 0, 1, &G.dset_rep, 0, NULL);
    struct PCRep pr = {count, D};
    vkCmdPushConstants(G.eg_cmd, G.plyt_rep, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pr), &pr);
    vkCmdDispatch(G.eg_cmd, (uint32_t)((D + 255) / 256), 1, 1);
    vkCmdPipelineBarrier(G.eg_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    if (dp4a) {
        struct PCQr qr = {count, D};
        vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_qr);
        vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_qr, 0, 1, &G.dset_qr_x, 0, NULL);
        vkCmdPushConstants(G.eg_cmd, G.plyt_qr, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(qr), &qr);
        vkCmdDispatch(G.eg_cmd, (uint32_t)count, 1, 1);
        vkCmdPipelineBarrier(G.eg_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dp4a ? G.pipe_gu_dp4a : G.pipe_gu);
    for (int c = 0; c < count; c++) {
        if (dp4a) {
            struct PCDp4a pc = {1, D, I, gates[c]->rowWords, gates[c]->gs, c};
            vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu4, 0, 1, &G.eg_gu4[c], 0, NULL);
            vkCmdPushConstants(G.eg_cmd, G.plyt_gu4, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        } else {
            struct PC pc = {fmt, 1, D, I, gates[c]->rowWords, gates[c]->gs};
            vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &G.eg_gu[c], 0, NULL);
            vkCmdPushConstants(G.eg_cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        }
        vkCmdDispatch(G.eg_cmd, (uint32_t)((I + 7) / 8), 1, 1);
    }
    vkCmdPipelineBarrier(G.eg_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    if (dp4a_dn) {
        struct PCQr qr = {count, I};
        vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_qr);
        vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_qr, 0, 1, &G.dset_qr_h, 0, NULL);
        vkCmdPushConstants(G.eg_cmd, G.plyt_qr, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(qr), &qr);
        vkCmdDispatch(G.eg_cmd, (uint32_t)count, 1, 1);
        vkCmdPipelineBarrier(G.eg_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dp4a_dn ? G.pipe_dn_dp4a : G.pipe);
    for (int c = 0; c < count; c++) {
        if (dp4a_dn) {
            struct PCDp4a pc = {1, I, D, downs[c]->rowWords, downs[c]->gs, c};
            vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_dn4, 0, 1, &G.eg_dn4[c], 0, NULL);
            vkCmdPushConstants(G.eg_cmd, G.plyt_dn4, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        } else {
            struct PC pc = {dfmt, 1, I, D, downs[c]->rowWords, downs[c]->gs};
            vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.eg_dn[c], 0, NULL);
            vkCmdPushConstants(G.eg_cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        }
        vkCmdDispatch(G.eg_cmd, (uint32_t)((D + 7) / 8), 1, 1);
    }
    vkCmdPipelineBarrier(G.eg_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_macc);
    vkCmdBindDescriptorSets(G.eg_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_macc, 0, 1, &G.dset_macc, 0, NULL);
    struct PCRep pa = {count, D};
    vkCmdPushConstants(G.eg_cmd, G.plyt_macc, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pa), &pa);
    vkCmdDispatch(G.eg_cmd, (uint32_t)((D + 255) / 256), 1, 1);
    VKCHECK(vkEndCommandBuffer(G.eg_cmd), "eg endCmd");
    return 1;
}

/* Expert group that reads device route_nrm, accumulates into stream, signals eg_sem.
 * With COLI_VK_EG_CACHE (default on): after the first record for a shape, subsequent
 * calls only delta-update expert weight descriptors and resubmit the same eg_cmd. */
static int eg_prepare_submit_pipe(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                                  ColiVkTensor *const *downs, const int *rows,
                                  const float *weights, int count, int D) {
    if (!G.ready || !G.shader_gu || !G.pipe_rep || !G.pipe_macc || count < 1 || count > 64) return 0;
    if (!G.stream_live || D != G.stream_D || !G.route_nrm.buf) return 0;
    ColiVkTensor *g0 = gates[0]; if (!g0) return 0;
    int I = g0->O, fmt = g0->fmt;
    if (g0->I != D || D > 6144) return 0;
    int dfmt = downs[0]->fmt;
    int grw = g0->rowWords, ggs = g0->gs, drw = downs[0]->rowWords, dgs = downs[0]->gs;
    int uniform_pc = 1;
    for (int c = 0; c < count; c++) {
        if (rows[c] != 1 || gates[c]->I != D || gates[c]->O != I || gates[c]->fmt != fmt ||
            ups[c]->I != D || ups[c]->O != I || ups[c]->fmt != fmt ||
            downs[c]->I != I || downs[c]->O != D || downs[c]->fmt != dfmt)
            return 0;
        /* cmd cache bakes uniform push constants — mixed gs/rowWords cannot reuse CB. */
        if (gates[c]->rowWords != grw || gates[c]->gs != ggs ||
            ups[c]->rowWords != grw || ups[c]->gs != ggs ||
            downs[c]->rowWords != drw || downs[c]->gs != dgs)
            uniform_pc = 0;
    }
    size_t xb = (size_t)count * D * 4, hb = (size_t)count * I * 4, yb = (size_t)count * D * 4;
    if (!scratch_reserve(&G.eg_x, xb) || !scratch_reserve(&G.eg_h, hb) ||
        !scratch_reserve(&G.eg_y, yb) ||
        !scratch_reserve(&G.moe_w, (size_t)count * 4 < 256 ? 256 : (size_t)count * 4)) return 0;
    memcpy(G.moe_w.ptr, weights, (size_t)count * 4);

    int dp4a_on = G.pipe_qr && g_qwen_opts.dp4a != 0;
    int dp4a = dp4a_on && fmt == 6 && D <= 6144;
    int dp4a_dn = dp4a_on && dfmt == 6 && I <= 8192;
    size_t sb_min = (size_t)count * 4 < 256 ? 256 : (size_t)count * 4;
    if (dp4a && !(scratch_reserve(&G.eg_xq, xb / 4) && scratch_reserve(&G.eg_xs, sb_min))) dp4a = 0;
    if (dp4a_dn && !(scratch_reserve(&G.eg_hq, hb / 4) && scratch_reserve(&G.eg_hs, sb_min))) dp4a_dn = 0;
    if (!eg_pool_ensure()) return 0;

    int shape_ok = G.eg_pipe_cache_on && G.eg_pipe_cmd_ready &&
        G.eg_pipe_n == count && G.eg_pipe_D == D && G.eg_pipe_I == I &&
        G.eg_pipe_fmt == fmt && G.eg_pipe_dfmt == dfmt &&
        G.eg_pipe_dp4a == dp4a && G.eg_pipe_dp4a_dn == dp4a_dn &&
        G.eg_pipe_grw == grw && G.eg_pipe_ggs == ggs &&
        G.eg_pipe_drw == drw && G.eg_pipe_dgs == dgs &&
        G.eg_pipe_xb == G.eg_x.buf && G.eg_pipe_hb == G.eg_h.buf &&
        G.eg_pipe_yb == G.eg_y.buf && G.eg_pipe_wb == G.moe_w.buf &&
        G.eg_pipe_sb == G.stream.buf && G.eg_pipe_nb == G.route_nrm.buf &&
        (!dp4a || (G.eg_pipe_xqb == G.eg_xq.buf && G.eg_pipe_xsb == G.eg_xs.buf)) &&
        (!dp4a_dn || (G.eg_pipe_hqb == G.eg_hq.buf && G.eg_pipe_hsb == G.eg_hs.buf));

    if (shape_ok) {
        /* A1: only weight bindings that changed. Scratch/rep/macc already valid. */
        for (int c = 0; c < count; c++)
            eg_pipe_bind_slot(c, dp4a, dp4a_dn, D, I, gates[c], ups[c], downs[c], 0);
        G.eg_pipe_hits++;
    } else {
        for (int c = 0; c < count; c++)
            eg_pipe_bind_slot(c, dp4a, dp4a_dn, D, I, gates[c], ups[c], downs[c], 1);
        eg_pipe_bind_scratch(count, D, I, dp4a, dp4a_dn, xb, hb);
        if (!eg_pipe_record(count, D, I, fmt, dfmt, dp4a, dp4a_dn, gates, downs)) return 0;
        /* Only cache the CB when every slot shares the baked push-constant fields. */
        G.eg_pipe_cmd_ready = G.eg_pipe_cache_on && uniform_pc;
        G.eg_pipe_n = count; G.eg_pipe_D = D; G.eg_pipe_I = I;
        G.eg_pipe_fmt = fmt; G.eg_pipe_dfmt = dfmt;
        G.eg_pipe_dp4a = dp4a; G.eg_pipe_dp4a_dn = dp4a_dn;
        G.eg_pipe_grw = grw; G.eg_pipe_ggs = ggs; G.eg_pipe_drw = drw; G.eg_pipe_dgs = dgs;
        G.eg_pipe_xb = G.eg_x.buf; G.eg_pipe_hb = G.eg_h.buf; G.eg_pipe_yb = G.eg_y.buf;
        G.eg_pipe_wb = G.moe_w.buf; G.eg_pipe_sb = G.stream.buf; G.eg_pipe_nb = G.route_nrm.buf;
        G.eg_pipe_xqb = G.eg_xq.buf; G.eg_pipe_xsb = G.eg_xs.buf;
        G.eg_pipe_hqb = G.eg_hq.buf; G.eg_pipe_hsb = G.eg_hs.buf;
        G.eg_pipe_misses++;
    }

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &G.eg_cmd,
        .signalSemaphoreCount = 1, .pSignalSemaphores = &G.eg_sem};
    VKCHECK(vkResetFences(G.dev, 1, &G.eg_fence), "eg resetFence");
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.eg_fence), "eg queueSubmit");
    G.eg_pending_yb = yb; G.eg_inflight = 1; G.eg_sem_armed = 1;
    return 1;
}

int coli_vk_expert_group_issue_pipe(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                                    ColiVkTensor *const *downs, const int *rows,
                                    const float *weights, int count, int D) {
    if (G.eg_inflight) return 0;
    return eg_prepare_submit_pipe(gates, ups, downs, rows, weights, count, D);
}

int coli_vk_expert_group_take_pipe(void) {
    return stream_wait_eg();
}

/* COLI_VK_FLASH: 1 = always use the chunked head-batched kernel, 0 = never,
 * unset = auto. Auto keeps the per-head kernel for short contexts (L still fits
 * L2, so its H-fold re-read is nearly free and one dispatch beats three) and
 * switches to the flash kernel once L outgrows cache and the read redundancy
 * starts hitting VRAM bandwidth for real. */
static int attn_flash_pick(int ntmax) {
    static int mode = -2;
    if (mode == -2) { mode = g_qwen_opts.flash; }
    if (!G.pipe_attf) return 0;
    if (mode == 0) return 0;
    if (mode == 1) return 1;
    return ntmax >= 8192;   /* measured crossover on Arc B390 (L ~ 16 MB > L2) */
}

/* Reserve the flash scratches, point the 8-binding set at this call's buffers
 * and record the 3 phases (qabs prepass -> chunk pass -> combine+value) with
 * compute barriers into the ALREADY-BEGUN main command buffer. ctx [S,H,V]
 * lands in ctxbuf. Returns 0 -> caller aborts (falls back to CPU). */
static int attn_flash_record(ColiVkTensor *t, VkKvLayer *kv, VkBuffer ctxbuf,
                             int fmt, int S, int H, int Q, int R, int V, int K,
                             int st0, int T, float scale) {
    int ntmax = T - st0;
    int nc = (ntmax + 255) / 256; if (nc > 64) nc = 64; if (nc < 1) nc = 1;
    int ch = (((ntmax + nc - 1) / nc) + 15) & ~15;    /* chunk = multiple of the TJ=16 tile */
    nc = (ntmax + ch - 1) / ch;
    if (!scratch_reserve(&G.att_qabs, (size_t)S * H * K * 4) ||
        !scratch_reserve(&G.att_part, (size_t)S * H * nc * (K + 2) * 4)) return 0;
    VkDescriptorBufferInfo bi[8] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = t->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = t->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = kv->bl, .range = VK_WHOLE_SIZE},
        {.buffer = kv->br, .range = VK_WHOLE_SIZE}, {.buffer = G.att_qabs.buf, .range = VK_WHOLE_SIZE},
        {.buffer = G.att_part.buf, .range = VK_WHOLE_SIZE}, {.buffer = ctxbuf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_attf, 8, bi);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_attf);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_attf, 0, 1, &G.dset_attf, 0, NULL);
    struct PCAttnF pc = {0, fmt, S, H, Q, R, V, K, st0, T, t->rowWords, t->gs, nc, ch, scale};
    vkCmdPushConstants(G.cmd, G.plyt_attf, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)H, (uint32_t)S, 1);          /* qabs prepass */
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    pc.mode = 1;
    vkCmdPushConstants(G.cmd, G.plyt_attf, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)(nc * ((H + 7) / 8)), (uint32_t)S, 1);   /* (chunk, head-block) pairs */
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    pc.mode = 2;
    vkCmdPushConstants(G.cmd, G.plyt_attf, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)H, (uint32_t)S, 1);          /* combine + value rows */
    return 1;
}

/* Record the legacy per-(s,h) fused kernel (one dispatch). Binding 5 is the old
 * score scratch: the fused shader never touches it, but the layout still wants
 * a valid buffer — the caller passes the (tiny) G.att_sc. */
static void attn_record_perhead(ColiVkTensor *t, VkKvLayer *kv, VkBuffer ctxbuf,
                                int fmt, int S, int H, int Q, int R, int V, int K,
                                int st0, int T, float scale) {
    VkDescriptorBufferInfo bi[7] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = t->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = t->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = kv->bl, .range = VK_WHOLE_SIZE},
        {.buffer = kv->br, .range = VK_WHOLE_SIZE}, {.buffer = G.att_sc.buf, .range = VK_WHOLE_SIZE},
        {.buffer = ctxbuf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset_att, 7, bi);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_att);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_att, 0, 1, &G.dset_att, 0, NULL);
    struct PCAttn pc = {fmt, S, H, Q, R, V, K, st0, T, t->rowWords, T - st0, scale, t->gs};
    vkCmdPushConstants(G.cmd, G.plyt_att, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(G.cmd, (uint32_t)H, (uint32_t)S, 1);     /* one workgroup per (head, row) */
}

/* Decode MLA absorption core for S causal query rows of one sequence, one submit:
 * ctx[s,h,:] = softmax((Wnope_h^T q_nope).L_t + q_rope.R_t) weighted latent context
 * projected through the value rows of kv_b. kv_b ([H*(Q+V), K]) uploads on first
 * call and stays resident; L/R rows [st0, T) must already be mirrored via
 * coli_vk_kv_row. Routes to the flash (chunked, head-batched) kernel on long
 * contexts when its shader is present. Returns 0 -> caller falls back to CPU. */
int coli_vk_attention_absorb(ColiVkTensor **kvb, const void *w, const float *sc, int fmt, int grp,
                             float *ctx, const float *q, int layer, int S, int H,
                             int Q, int R, int V, int K, int st0, int T, float scale) {
    if (!G.ready || !G.pipe_att || S < 1 || H < 1 || layer < 0 || layer >= VK_KV_LAYERS) return 0;
    if (Q > 256 || R > 64 || K > 512 || st0 < 0 || T - S - st0 < 0) return 0;  /* shared-array limits */
    VkKvLayer *kv = &G.kv[layer];
    if (!kv->bl || kv->rows < T || kv->K != K || kv->R != R) return 0;
    if (!upload_tensor(kvb, w, sc, fmt, K, H * (Q + V), grp)) return 0;
    ColiVkTensor *t = *kvb;
    int flash = attn_flash_pick(T - st0);
    size_t qb = (size_t)S * H * (Q + R) * 4, cb = (size_t)S * H * V * 4;
    if (!scratch_reserve(&G.x, qb) || !scratch_reserve_mt(&G.y, cb, G.memtype_cached) ||
        !scratch_reserve(&G.att_sc, 64)) return 0;    /* y (ctx) is read back -> cached */
    memcpy(G.x.ptr, q, qb);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    if (flash) {
        if (!attn_flash_record(t, kv, G.y.buf, fmt, S, H, Q, R, V, K, st0, T, scale)) return 0;
    } else {
        attn_record_perhead(t, kv, G.y.buf, fmt, S, H, Q, R, V, K, st0, T, scale);
    }
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double vp0 = G.eg_prof ? vk_now() : 0;
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { double vp1 = vk_now(); g_vsub_ms += vp1 - vp0; vp0 = vp1; }
    if (vk_fence_wait_loud(G.fence, "absorb") != VK_SUCCESS) { G.ready = 0; return 0; }
    if (G.eg_prof) { g_vwait_ms += vk_now() - vp0; vkprof_tick(); }
    memcpy(ctx, G.y.ptr, cb);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}

/* Two resident matmuls sharing the SAME input x in ONE submit (q_a + kv_a in the
 * attention prologue): one x staging, two dispatches, one fence — replaces two
 * full submit+wait roundtrips. Outputs y1 [S,O1] and y2 [S,O2] read back from
 * cached memory. Returns 0 -> caller falls back to the single-matmul path. */
int coli_vk_matmul_pair(ColiVkTensor **t1p, float *y1, const void *w1, const float *s1, int O1,
                        ColiVkTensor **t2p, float *y2, const void *w2, const float *s2, int O2,
                        int fmt, const float *x, int S, int I, int grp) {
    if (!G.ready || S < 1) return 0;
    if (!upload_tensor(t1p, w1, s1, fmt, I, O1, grp) || !upload_tensor(t2p, w2, s2, fmt, I, O2, grp)) return 0;
    ColiVkTensor *t1 = *t1p, *t2 = *t2p;
    size_t xb = (size_t)S * I * 4, yb1 = (size_t)S * O1 * 4, yb2 = (size_t)S * O2 * 4;
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve_mt(&G.y, yb1, G.memtype_cached) ||
        !scratch_reserve_mt(&G.y2, yb2, G.memtype_cached)) return 0;
    memcpy(G.x.ptr, x, xb);

    if (!G.pair_pool) {   /* one-time: a second 4-binding set (G.dset serves the first) */
        VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.pair_pool), "pair descPool");
        VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.pair_pool, .descriptorSetCount = 1, .pSetLayouts = &G.dsl};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa, &G.dset_pair), "pair descSet");
    }
    VkDescriptorBufferInfo b1[4] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = t1->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = t1->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
    VkDescriptorBufferInfo b2[4] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = t2->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = t2->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y2.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset, 4, b1);
    wr_desc(G.dset_pair, 4, b2);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    struct PC pc1 = {fmt, S, I, O1, t1->rowWords, t1->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), &pc1);
    vkCmdDispatch(G.cmd, (uint32_t)((O1 + 7) / 8), (uint32_t)S, 1);
    struct PC pc2 = {fmt, S, I, O2, t2->rowWords, t2->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset_pair, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
    vkCmdDispatch(G.cmd, (uint32_t)((O2 + 7) / 8), (uint32_t)S, 1);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double vp0 = G.eg_prof ? vk_now() : 0;
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { double vp1 = vk_now(); g_vsub_ms += vp1 - vp0; vp0 = vp1; }
    if (vk_fence_wait_loud(G.fence, "matmul_pair") != VK_SUCCESS) { G.ready = 0; return 0; }
    if (G.eg_prof) { g_vwait_ms += vk_now() - vp0; vkprof_tick(); }
    memcpy(y1, G.y.ptr, yb1);
    memcpy(y2, G.y2.ptr, yb2);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}


/* q-prep chain: [q_a + kv_a pair] -> rmsnorm(q_latent) -> [q_b], recorded in ONE
 * command buffer with compute barriers — one submit+fence where the engine paid
 * three (the middle CPU norm forced two roundtrips). Only q [S,Oqb] and the kv
 * latent [S,Okva] return to the host (RoPE + canonical KV append stay CPU-side).
 * The per-layer norm weights upload once into a tiny resident buffer (KV-mirror
 * pattern). All three tensors must share fmt (dense io is int8 in practice).
 * Returns 0 -> caller runs the 3-submit path (also when rmsnorm.spv is absent). */
int coli_vk_attn_qprep(int layer,
                       ColiVkTensor **qa,  const void *wqa,  const float *sqa,  int Oqa,
                       ColiVkTensor **kva, const void *wkva, const float *skva, int Okva,
                       ColiVkTensor **qb,  const void *wqb,  const float *sqb,  int Oqb,
                       int fmt, int grp, const float *lnw, float eps,
                       const float *x, int S, int I, float *q_out, float *kv_out, float *lat_out) {
    if (!G.ready || !G.shader_nrm || S < 1 || layer < 0 || layer >= VK_KV_LAYERS) return 0;
    if (!upload_tensor(qa, wqa, sqa, fmt, I, Oqa, grp) || !upload_tensor(kva, wkva, skva, fmt, I, Okva, grp) ||
        !upload_tensor(qb, wqb, sqb, fmt, Oqa, Oqb, grp)) return 0;
    ColiVkTensor *tqa = *qa, *tkv = *kva, *tqb = *qb;
    if (!G.lnbuf[layer]) {                       /* resident norm weights, uploaded once */
        void *lp; float p0 = G.prio; G.prio = 1.0f;
        int ok = alloc_hostvis((size_t)Oqa * 4, &G.lnbuf[layer], &G.lnmem[layer], &lp);
        G.prio = p0;
        if (!ok) { G.lnbuf[layer] = VK_NULL_HANDLE; return 0; }
        memcpy(lp, lnw, (size_t)Oqa * 4); G.lnlen[layer] = Oqa;
    }
    if (G.lnlen[layer] != Oqa) return 0;
    size_t xb = (size_t)S * I * 4, qb_b = (size_t)S * Oqb * 4, kvb_b = (size_t)S * Okva * 4;
    size_t lat = (size_t)S * Oqa * 4;
    if (!scratch_reserve(&G.x, xb) || !scratch_reserve_mt(&G.y, qb_b, G.memtype_cached) ||
        !scratch_reserve_mt(&G.y2, kvb_b, G.memtype_cached) ||
        !scratch_reserve(&G.qp1, lat) ||
        !scratch_reserve_mt(&G.qp2, lat, G.memtype_cached)) return 0;   /* normed latent reads back (DSA indexer) */
    memcpy(G.x.ptr, x, xb);

    VkDescriptorBufferInfo b1[4] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = tqa->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tqa->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.qp1.buf, .range = VK_WHOLE_SIZE}};
    VkDescriptorBufferInfo b2[4] = {
        {.buffer = G.x.buf, .range = VK_WHOLE_SIZE}, {.buffer = tkv->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tkv->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y2.buf, .range = VK_WHOLE_SIZE}};
    VkDescriptorBufferInfo bn[3] = {
        {.buffer = G.qp1.buf, .range = VK_WHOLE_SIZE}, {.buffer = G.lnbuf[layer], .range = VK_WHOLE_SIZE},
        {.buffer = G.qp2.buf, .range = VK_WHOLE_SIZE}};
    VkDescriptorBufferInfo b3[4] = {
        {.buffer = G.qp2.buf, .range = VK_WHOLE_SIZE}, {.buffer = tqb->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = tqb->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
    if (!G.pair_pool) {   /* the chain reuses the pair's 2nd matmul set */
        VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
        VKCHECK(vkCreateDescriptorPool(G.dev, &dpi, NULL, &G.pair_pool), "pair descPool");
        VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = G.pair_pool, .descriptorSetCount = 1, .pSetLayouts = &G.dsl};
        VKCHECK(vkAllocateDescriptorSets(G.dev, &dsa, &G.dset_pair), "pair descSet");
    }
    wr_desc(G.dset, 4, b1); wr_desc(G.dset_pair, 4, b2); wr_desc(G.dset_qp3, 4, b3);
    wr_desc(G.dset_nrm, 3, bn);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    struct PC pc1 = {fmt, S, I, Oqa, tqa->rowWords, tqa->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), &pc1);
    vkCmdDispatch(G.cmd, (uint32_t)((Oqa + 7) / 8), (uint32_t)S, 1);
    struct PC pc2 = {fmt, S, I, Okva, tkv->rowWords, tkv->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset_pair, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
    vkCmdDispatch(G.cmd, (uint32_t)((Okva + 7) / 8), (uint32_t)S, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_nrm);
    struct PCN pcn = {S, Oqa, eps};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_nrm, 0, 1, &G.dset_nrm, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt_nrm, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcn), &pcn);
    vkCmdDispatch(G.cmd, (uint32_t)S, 1, 1);
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    struct PC pc3 = {fmt, S, Oqa, Oqb, tqb->rowWords, tqb->gs};
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset_qp3, 0, NULL);
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc3), &pc3);
    vkCmdDispatch(G.cmd, (uint32_t)((Oqb + 7) / 8), (uint32_t)S, 1);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double vp0 = G.eg_prof ? vk_now() : 0;
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { double vp1 = vk_now(); g_vsub_ms += vp1 - vp0; vp0 = vp1; }
    if (vk_fence_wait_loud(G.fence, "attn_qprep") != VK_SUCCESS) { G.ready = 0; return 0; }
    if (G.eg_prof) { g_vwait_ms += vk_now() - vp0; vkprof_tick(); }
    memcpy(q_out, G.y.ptr, qb_b);
    memcpy(kv_out, G.y2.ptr, kvb_b);
    if (lat_out) memcpy(lat_out, G.qp2.ptr, lat);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}

/* Fused absorb attention + o-projection in ONE submit: the absorb kernel writes ctx
 * [S,H*V] to a device-only scratch, a barrier, then the resident o_proj ([Dout, H*V])
 * runs on it via the plain matmul pipeline — only out [S,Dout] returns to the host.
 * Kills the per-layer ctx readback + re-upload + second submit of the unfused path.
 * Returns 0 -> caller falls back (plain absorb or CPU). */
int coli_vk_attention_absorb_project(ColiVkTensor **kvb, const void *w, const float *sc, int fmt, int grp,
                                     ColiVkTensor **ot, const void *ow, const float *osc, int ofmt, int ogrp,
                                     float *out, const float *q, int layer, int S, int H,
                                     int Q, int R, int V, int K, int st0, int T, float scale,
                                     int Dout) {
    if (!G.ready || !G.pipe_att || S < 1 || H < 1 || layer < 0 || layer >= VK_KV_LAYERS) return 0;
    if (Q > 256 || R > 64 || K > 512 || st0 < 0 || T - S - st0 < 0 || Dout < 1) return 0;
    VkKvLayer *kv = &G.kv[layer];
    if (!kv->bl || kv->rows < T || kv->K != K || kv->R != R) return 0;
    if (!upload_tensor(kvb, w, sc, fmt, K, H * (Q + V), grp)) return 0;
    if (!upload_tensor(ot, ow, osc, ofmt, H * V, Dout, ogrp)) return 0;
    ColiVkTensor *t = *kvb, *to = *ot;
    int flash = attn_flash_pick(T - st0);
    size_t qb = (size_t)S * H * (Q + R) * 4, cb = (size_t)S * H * V * 4;
    size_t ob = (size_t)S * Dout * 4;
    if (!scratch_reserve(&G.x, qb) || !scratch_reserve(&G.att_ctx, cb) ||
        !scratch_reserve(&G.att_sc, 64) || !scratch_reserve_mt(&G.y, ob, G.memtype_cached)) return 0;
    memcpy(G.x.ptr, q, qb);

    VkDescriptorBufferInfo oi[4] = {
        {.buffer = G.att_ctx.buf, .range = VK_WHOLE_SIZE}, {.buffer = to->wbuf, .range = VK_WHOLE_SIZE},
        {.buffer = to->sbuf, .range = VK_WHOLE_SIZE}, {.buffer = G.y.buf, .range = VK_WHOLE_SIZE}};
    wr_desc(G.dset, 4, oi);

    VKCHECK(vkResetCommandBuffer(G.cmd, 0), "resetCmd");
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(G.cmd, &begin), "beginCmd");
    if (flash) {
        if (!attn_flash_record(t, kv, G.att_ctx.buf, fmt, S, H, Q, R, V, K, st0, T, scale)) return 0;
    } else {
        attn_record_perhead(t, kv, G.att_ctx.buf, fmt, S, H, Q, R, V, K, st0, T, scale);
    }
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    struct PC opc = {ofmt, S, H * V, Dout, to->rowWords, to->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(opc), &opc);
    vkCmdDispatch(G.cmd, (uint32_t)((Dout + 7) / 8), (uint32_t)S, 1);
    VKCHECK(vkEndCommandBuffer(G.cmd), "endCmd");

    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    VKCHECK(vkResetFences(G.dev, 1, &G.fence), "resetFence");
    double vp0 = G.eg_prof ? vk_now() : 0;
    VKCHECK(vkQueueSubmit(G.queue, 1, &si, G.fence), "queueSubmit");
    if (G.eg_prof) { double vp1 = vk_now(); g_vsub_ms += vp1 - vp0; vp0 = vp1; }
    if (vk_fence_wait_loud(G.fence, "absorb_project") != VK_SUCCESS) { G.ready = 0; return 0; }
    if (G.eg_prof) { g_vwait_ms += vk_now() - vp0; vkprof_tick(); }
    memcpy(out, G.y.ptr, ob);
    G.cmd_ready = 0; G.bound_tensor = NULL;   /* the shared command buffer/binding was clobbered */
    return 1;
}

void coli_vk_tensor_free(ColiVkTensor *t) {
    if (!t) return;
    if (G.bound_tensor == t) { G.bound_tensor = NULL; G.cmd_ready = 0; }  /* drop stale cache */
    // If the device was lost/disabled (the fence-timeout path sets G.ready=0), a submission
    // may still reference these buffers — do NOT vkDestroy into a dead device (GPU-side UAF).
    // Leak the GPU handles (we're degrading to CPU for the rest of the run) and reclaim the
    // host struct + counters only.
    if (G.ready) {
        if (t->wbuf) { vkDestroyBuffer(G.dev, t->wbuf, NULL); vkFreeMemory(G.dev, t->wmem, NULL); }
        if (t->sbuf) { vkDestroyBuffer(G.dev, t->sbuf, NULL); vkFreeMemory(G.dev, t->smem, NULL); }
    }
    // Mirror upload_tensor exactly (weights + scales), atomically — otherwise
    // used_bytes leaks the scales buffer on every free and drifts upward.
    __atomic_sub_fetch(&G.tensor_count, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&G.used_bytes, t->wbytes + scale_floats(t->fmt, t->I, t->O, t->gs) * sizeof(float), __ATOMIC_RELAXED);
    free(t);
}

size_t coli_vk_tensor_bytes(const ColiVkTensor *t) { return t ? t->wbytes : 0; }

void coli_vk_shutdown(void) {
    if (!G.ready) return;
    vkDeviceWaitIdle(G.dev);
    if (G.x.buf) { vkDestroyBuffer(G.dev, G.x.buf, NULL); vkFreeMemory(G.dev, G.x.mem, NULL); }
    if (G.y.buf) { vkDestroyBuffer(G.dev, G.y.buf, NULL); vkFreeMemory(G.dev, G.y.mem, NULL); }
    if (G.h.buf) { vkDestroyBuffer(G.dev, G.h.buf, NULL); vkFreeMemory(G.dev, G.h.mem, NULL); }
    if (G.eg_x.buf) { vkDestroyBuffer(G.dev, G.eg_x.buf, NULL); vkFreeMemory(G.dev, G.eg_x.mem, NULL); }
    if (G.eg_h.buf) { vkDestroyBuffer(G.dev, G.eg_h.buf, NULL); vkFreeMemory(G.dev, G.eg_h.mem, NULL); }
    if (G.eg_y.buf) { vkDestroyBuffer(G.dev, G.eg_y.buf, NULL); vkFreeMemory(G.dev, G.eg_y.mem, NULL); }
    if (G.eg_xq.buf) { vkDestroyBuffer(G.dev, G.eg_xq.buf, NULL); vkFreeMemory(G.dev, G.eg_xq.mem, NULL); }
    if (G.eg_xs.buf) { vkDestroyBuffer(G.dev, G.eg_xs.buf, NULL); vkFreeMemory(G.dev, G.eg_xs.mem, NULL); }
    if (G.eg_hq.buf) { vkDestroyBuffer(G.dev, G.eg_hq.buf, NULL); vkFreeMemory(G.dev, G.eg_hq.mem, NULL); }
    if (G.eg_hs.buf) { vkDestroyBuffer(G.dev, G.eg_hs.buf, NULL); vkFreeMemory(G.dev, G.eg_hs.mem, NULL); }
    if (G.att_sc.buf) { vkDestroyBuffer(G.dev, G.att_sc.buf, NULL); vkFreeMemory(G.dev, G.att_sc.mem, NULL); }
    if (G.att_ctx.buf) { vkDestroyBuffer(G.dev, G.att_ctx.buf, NULL); vkFreeMemory(G.dev, G.att_ctx.mem, NULL); }
    if (G.att_qabs.buf) { vkDestroyBuffer(G.dev, G.att_qabs.buf, NULL); vkFreeMemory(G.dev, G.att_qabs.mem, NULL); }
    if (G.att_part.buf) { vkDestroyBuffer(G.dev, G.att_part.buf, NULL); vkFreeMemory(G.dev, G.att_part.mem, NULL); }
    if (G.y2.buf) { vkDestroyBuffer(G.dev, G.y2.buf, NULL); vkFreeMemory(G.dev, G.y2.mem, NULL); }
    if (G.am_y.buf) { vkDestroyBuffer(G.dev, G.am_y.buf, NULL); vkFreeMemory(G.dev, G.am_y.mem, NULL); }
    if (G.am_pi.buf) { vkDestroyBuffer(G.dev, G.am_pi.buf, NULL); vkFreeMemory(G.dev, G.am_pi.mem, NULL); }
    if (G.am_pv.buf) { vkDestroyBuffer(G.dev, G.am_pv.buf, NULL); vkFreeMemory(G.dev, G.am_pv.mem, NULL); }
    if (G.gdn_in.buf) { vkDestroyBuffer(G.dev, G.gdn_in.buf, NULL); vkFreeMemory(G.dev, G.gdn_in.mem, NULL); }
    if (G.gdn_y.buf) { vkDestroyBuffer(G.dev, G.gdn_y.buf, NULL); vkFreeMemory(G.dev, G.gdn_y.mem, NULL); }
    coli_vk_gdn_free();
    if (G.pair_pool) vkDestroyDescriptorPool(G.dev, G.pair_pool, NULL);
    coli_vk_kv_reset();
    if (G.eg_pool) vkDestroyDescriptorPool(G.dev, G.eg_pool, NULL);
    vkDestroyFence(G.dev, G.fence, NULL);
    vkDestroyFence(G.dev, G.eg_fence, NULL);
    vkDestroyCommandPool(G.dev, G.cpool, NULL);
    vkDestroyDescriptorPool(G.dev, G.dpool, NULL);
    vkDestroyPipeline(G.dev, G.pipe, NULL);
    vkDestroyPipelineLayout(G.dev, G.plyt, NULL);
    vkDestroyDescriptorSetLayout(G.dev, G.dsl, NULL);
    vkDestroyShaderModule(G.dev, G.shader, NULL);
    if (G.shader_gu) {
        vkDestroyDescriptorPool(G.dev, G.dpool_gu, NULL);
        vkDestroyPipeline(G.dev, G.pipe_gu, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_gu, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_gu, NULL);
        vkDestroyShaderModule(G.dev, G.shader_gu, NULL);
    }
    if (G.shader_am) {
        vkDestroyDescriptorPool(G.dev, G.dpool_am, NULL);
        vkDestroyPipeline(G.dev, G.pipe_am, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_am, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_am, NULL);
        vkDestroyShaderModule(G.dev, G.shader_am, NULL);
    }
    if (G.route_x.buf) { vkDestroyBuffer(G.dev, G.route_x.buf, NULL); vkFreeMemory(G.dev, G.route_x.mem, NULL); }
    if (G.route_xo.buf) { vkDestroyBuffer(G.dev, G.route_xo.buf, NULL); vkFreeMemory(G.dev, G.route_xo.mem, NULL); }
    if (G.route_nrm.buf) { vkDestroyBuffer(G.dev, G.route_nrm.buf, NULL); vkFreeMemory(G.dev, G.route_nrm.mem, NULL); }
    if (G.att_delta.buf) { vkDestroyBuffer(G.dev, G.att_delta.buf, NULL); vkFreeMemory(G.dev, G.att_delta.mem, NULL); }
    if (G.stream.buf) { vkDestroyBuffer(G.dev, G.stream.buf, NULL); vkFreeMemory(G.dev, G.stream.mem, NULL); }
    if (G.moe_w.buf) { vkDestroyBuffer(G.dev, G.moe_w.buf, NULL); vkFreeMemory(G.dev, G.moe_w.mem, NULL); }
    if (G.gdn_ba.buf) { vkDestroyBuffer(G.dev, G.gdn_ba.buf, NULL); vkFreeMemory(G.dev, G.gdn_ba.mem, NULL); }
    if (G.gdn_bb.buf) { vkDestroyBuffer(G.dev, G.gdn_bb.buf, NULL); vkFreeMemory(G.dev, G.gdn_bb.mem, NULL); }
    if (G.route_idx.buf) { vkDestroyBuffer(G.dev, G.route_idx.buf, NULL); vkFreeMemory(G.dev, G.route_idx.mem, NULL); }
    if (G.route_val.buf) { vkDestroyBuffer(G.dev, G.route_val.buf, NULL); vkFreeMemory(G.dev, G.route_val.mem, NULL); }
    for (int i = 0; i < VK_KV_LAYERS; i++) {
        if (!G.qln_in[i]) continue;
        vkDestroyBuffer(G.dev, G.qln_in[i], NULL); vkFreeMemory(G.dev, G.qln_inm[i], NULL);
        vkDestroyBuffer(G.dev, G.qln_post[i], NULL); vkFreeMemory(G.dev, G.qln_postm[i], NULL);
    }
    if (G.final_nw) {
        vkDestroyBuffer(G.dev, G.final_nw, NULL); vkFreeMemory(G.dev, G.final_nwm, NULL);
        G.final_nw = VK_NULL_HANDLE; G.final_nwp = NULL; G.final_nD = 0;
    }
    if (G.tsq) { vkDestroyQueryPool(G.dev, G.tsq, NULL); G.tsq = VK_NULL_HANDLE; }
    for (int i = 0; i < VK_KV_LAYERS; i++) {
        if (!G.gdn_alog[i]) continue;
        vkDestroyBuffer(G.dev, G.gdn_alog[i], NULL); vkFreeMemory(G.dev, G.gdn_alogm[i], NULL);
        vkDestroyBuffer(G.dev, G.gdn_dtb[i], NULL); vkFreeMemory(G.dev, G.gdn_dtbm[i], NULL);
        vkDestroyBuffer(G.dev, G.gdn_gn[i], NULL); vkFreeMemory(G.dev, G.gdn_gnm[i], NULL);
    }
    if (G.pipe_rrnz) {
        vkDestroyDescriptorPool(G.dev, G.dpool_rrnz, NULL);
        vkDestroyPipeline(G.dev, G.pipe_rrnz, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_rrnz, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_rrnz, NULL);
        vkDestroyShaderModule(G.dev, G.shader_rrnz, NULL);
    }
    if (G.pipe_rep) {
        vkDestroyDescriptorPool(G.dev, G.dpool_rep, NULL);
        vkDestroyPipeline(G.dev, G.pipe_rep, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_rep, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_rep, NULL);
        vkDestroyShaderModule(G.dev, G.shader_rep, NULL);
    }
    if (G.pipe_macc) {
        vkDestroyDescriptorPool(G.dev, G.dpool_macc, NULL);
        vkDestroyPipeline(G.dev, G.pipe_macc, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_macc, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_macc, NULL);
        vkDestroyShaderModule(G.dev, G.shader_macc, NULL);
    }
    if (G.pipe_gdp) {
        vkDestroyDescriptorPool(G.dev, G.dpool_gdp, NULL);
        vkDestroyPipeline(G.dev, G.pipe_gdp, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_gdp, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_gdp, NULL);
        vkDestroyShaderModule(G.dev, G.shader_gdp, NULL);
    }
    if (G.pipe_topk) {
        vkDestroyDescriptorPool(G.dev, G.dpool_topk, NULL);
        vkDestroyPipeline(G.dev, G.pipe_topk, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_topk, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_topk, NULL);
        vkDestroyShaderModule(G.dev, G.shader_topk, NULL);
    }
    if (G.pipe_moe_ix) {
        vkDestroyDescriptorPool(G.dev, G.dpool_moe_ix, NULL);
        vkDestroyPipeline(G.dev, G.pipe_moe_ix, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_moe_ix, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_moe_ix, NULL);
        if (G.shader_moe_ix) vkDestroyShaderModule(G.dev, G.shader_moe_ix, NULL);
    }
    if (G.pipe_moe_pack) {
        vkDestroyDescriptorPool(G.dev, G.dpool_moe_pack, NULL);
        vkDestroyPipeline(G.dev, G.pipe_moe_pack, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_moe_pack, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_moe_pack, NULL);
        if (G.shader_moe_pack) vkDestroyShaderModule(G.dev, G.shader_moe_pack, NULL);
    }
    if (G.eg_ix_valid) {
        vkDestroyBuffer(G.dev, G.eg_ix_valid, NULL); vkFreeMemory(G.dev, G.eg_ix_validm, NULL);
        vkDestroyBuffer(G.dev, G.eg_ix_shgate, NULL); vkFreeMemory(G.dev, G.eg_ix_shgatem, NULL);
        vkDestroyBuffer(G.dev, G.eg_ix_dummy_w, NULL); vkFreeMemory(G.dev, G.eg_ix_dummym, NULL);
        vkDestroyBuffer(G.dev, G.eg_ix_dummy_s, NULL); vkFreeMemory(G.dev, G.eg_ix_dummy_sm, NULL);
    }
    if (G.pipe_mm_pool) vkDestroyDescriptorPool(G.dev, G.pipe_mm_pool, NULL);
    if (G.eg_sem) vkDestroySemaphore(G.dev, G.eg_sem, NULL);
    if (G.route_pool) vkDestroyDescriptorPool(G.dev, G.route_pool, NULL);
    if (G.pipe_nrmz) {
        vkDestroyPipeline(G.dev, G.pipe_nrmz, NULL);
        if (G.shader_nrmz) vkDestroyShaderModule(G.dev, G.shader_nrmz, NULL);
    }
    if (G.pipe_qr) {   /* the three DP4A expert kernels are created and torn down together */
        vkDestroyDescriptorPool(G.dev, G.dpool_qr, NULL);
        vkDestroyPipeline(G.dev, G.pipe_qr, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_qr, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_qr, NULL);
        vkDestroyDescriptorPool(G.dev, G.dpool_gu4, NULL);
        vkDestroyPipeline(G.dev, G.pipe_gu_dp4a, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_gu4, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_gu4, NULL);
        vkDestroyDescriptorPool(G.dev, G.dpool_dn4, NULL);
        vkDestroyPipeline(G.dev, G.pipe_dn_dp4a, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_dn4, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_dn4, NULL);
    }
    if (G.shader_qr) vkDestroyShaderModule(G.dev, G.shader_qr, NULL);
    if (G.shader_gu_dp4a) vkDestroyShaderModule(G.dev, G.shader_gu_dp4a, NULL);
    if (G.shader_dn_dp4a) vkDestroyShaderModule(G.dev, G.shader_dn_dp4a, NULL);
    if (G.shader_att) {
        vkDestroyDescriptorPool(G.dev, G.dpool_att, NULL);
        vkDestroyPipeline(G.dev, G.pipe_att, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_att, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_att, NULL);
        vkDestroyShaderModule(G.dev, G.shader_att, NULL);
    }
    if (G.shader_attf) {
        vkDestroyDescriptorPool(G.dev, G.dpool_attf, NULL);
        vkDestroyPipeline(G.dev, G.pipe_attf, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_attf, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_attf, NULL);
        vkDestroyShaderModule(G.dev, G.shader_attf, NULL);
    }
    if (G.shader_gdn) {
        vkDestroyDescriptorPool(G.dev, G.dpool_gdn, NULL);
        vkDestroyPipeline(G.dev, G.pipe_gdn, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_gdn, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_gdn, NULL);
        vkDestroyShaderModule(G.dev, G.shader_gdn, NULL);
    }
    if (G.shader_gqa) {
        vkDestroyDescriptorPool(G.dev, G.dpool_gqa, NULL);
        vkDestroyPipeline(G.dev, G.pipe_gqa, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_gqa, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_gqa, NULL);
        vkDestroyShaderModule(G.dev, G.shader_gqa, NULL);
    }
    if (G.shader_gdnconv) {
        vkDestroyDescriptorPool(G.dev, G.dpool_gdnconv, NULL);
        vkDestroyPipeline(G.dev, G.pipe_gdnconv, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_gdnconv, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_gdnconv, NULL);
        vkDestroyShaderModule(G.dev, G.shader_gdnconv, NULL);
    }
    if (G.shader_gdncv) {
        vkDestroyDescriptorPool(G.dev, G.dpool_gdncv, NULL);
        vkDestroyPipeline(G.dev, G.pipe_gdncv, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_gdncv, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_gdncv, NULL);
        vkDestroyShaderModule(G.dev, G.shader_gdncv, NULL);
    }
    if (G.gdnf_pool) vkDestroyDescriptorPool(G.dev, G.gdnf_pool, NULL);
    if (G.gdnf_qkv.buf) { vkDestroyBuffer(G.dev, G.gdnf_qkv.buf, NULL); vkFreeMemory(G.dev, G.gdnf_qkv.mem, NULL); }
    if (G.gdnf_z.buf) { vkDestroyBuffer(G.dev, G.gdnf_z.buf, NULL); vkFreeMemory(G.dev, G.gdnf_z.mem, NULL); }
    if (G.gdnf_cv.buf) { vkDestroyBuffer(G.dev, G.gdnf_cv.buf, NULL); vkFreeMemory(G.dev, G.gdnf_cv.mem, NULL); }
    if (G.gdnf_pr.buf) { vkDestroyBuffer(G.dev, G.gdnf_pr.buf, NULL); vkFreeMemory(G.dev, G.gdnf_pr.mem, NULL); }
    if (G.shader_qkv) {
        vkDestroyDescriptorPool(G.dev, G.dpool_qkv, NULL);
        vkDestroyPipeline(G.dev, G.pipe_qkv, NULL);
        vkDestroyPipelineLayout(G.dev, G.plyt_qkv, NULL);
        vkDestroyDescriptorSetLayout(G.dev, G.dsl_qkv, NULL);
        vkDestroyShaderModule(G.dev, G.shader_qkv, NULL);
    }
    if (G.gqaf_pool) vkDestroyDescriptorPool(G.dev, G.gqaf_pool, NULL);
    if (G.gqaf_qg.buf) { vkDestroyBuffer(G.dev, G.gqaf_qg.buf, NULL); vkFreeMemory(G.dev, G.gqaf_qg.mem, NULL); }
    if (G.gqaf_kb.buf) { vkDestroyBuffer(G.dev, G.gqaf_kb.buf, NULL); vkFreeMemory(G.dev, G.gqaf_kb.mem, NULL); }
    if (G.gqaf_vb.buf) { vkDestroyBuffer(G.dev, G.gqaf_vb.buf, NULL); vkFreeMemory(G.dev, G.gqaf_vb.mem, NULL); }
    for (int i = 0; i < VK_KV_LAYERS; i++) {
        if (G.gqa_qnw[i]) { vkDestroyBuffer(G.dev, G.gqa_qnw[i], NULL); vkFreeMemory(G.dev, G.gqa_qnwm[i], NULL); }
        if (G.gqa_knw[i]) { vkDestroyBuffer(G.dev, G.gqa_knw[i], NULL); vkFreeMemory(G.dev, G.gqa_knwm[i], NULL); }
    }
    for (VkWArena *a = g_warena; a;) {   /* weight arenas: unmapped/freed with the device */
        VkWArena *nx = a->next;
        vkUnmapMemory(G.dev, a->mem); vkFreeMemory(G.dev, a->mem, NULL);
        free(a); a = nx;
    }
    g_warena = NULL;
    vkDestroyDevice(G.dev, NULL);
    vkDestroyInstance(G.inst, NULL);
    memset(&G, 0, sizeof(G));
}

#ifdef VK_TEST
// ---- standalone GPU-vs-CPU validation + microbench --------------------------
#include <math.h>
#include <time.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9; }

static int g_ref_gs = 64;   /* fmt=4 group size the harness cases use */
static size_t ref_rowbytes(int fmt, int I) {
    return fmt == 1 ? (size_t)I : fmt == 5 ? (size_t)((I + 63) / 64) * 24 : (size_t)(I + 1) / 2;
}
static size_t ref_scales(int fmt, int I, int O) {   // scale COUNT (per-group for fmt 4/5)
    if (fmt == 5) return (size_t)O * (size_t)((I + 63) / 64);
    if (fmt == 4) return (size_t)O * (size_t)((I + g_ref_gs - 1) / g_ref_gs);
    return (size_t)O;
}
static float deq(const uint8_t *row, int fmt, int i) {
    if (fmt == 1) { int b = ((const int8_t *)row)[i]; return (float)b; }
    if (fmt == 5) {   // int3-g64: 16B low plane (2 bits) + 8B high plane (1 bit), v+4
        const uint8_t *lo = row + (size_t)(i >> 6) * 24, *hi = lo + 16; int j = i & 63;
        unsigned u = ((lo[j >> 2] >> ((j & 3) * 2)) & 3u) | (((hi[j >> 3] >> (j & 7)) & 1u) << 2);
        return (float)((int)u - 4); }
    uint8_t v = row[i >> 1]; int nib = (i & 1) ? (v >> 4) : (v & 15); return (float)(nib - 8);
}

static void cpu_ref(float *y, const float *x, const uint8_t *w, const float *sc,
                    int fmt, int S, int I, int O) {
    size_t rb = ref_rowbytes(fmt, I);
    int gw2 = fmt == 4 ? g_ref_gs : 64, ng = (I + gw2 - 1) / gw2;
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        double sum = 0; const uint8_t *row = w + (size_t)o * rb;
        if (fmt == 5 || fmt == 4) {   // per-group scales fold inside the sum
            for (int g = 0; g < ng; g++) {
                double a = 0; int end = (g + 1) * gw2 < I ? (g + 1) * gw2 : I;
                for (int i = g * gw2; i < end; i++) a += x[s * I + i] * deq(row, fmt, i);
                sum += a * sc[(size_t)o * ng + g];
            }
            y[s * O + o] = (float)sum;
        } else {
            for (int i = 0; i < I; i++) sum += x[s * I + i] * deq(row, fmt, i);
            y[s * O + o] = (float)(sum * sc[o]);
        }
    }
}

/* dequant dot of one weight row against x with that row's scales applied —
 * per-row for fmt 1/2, per 64-group for fmt=5. scb = tensor scale array, o = row. */
static double ref_dot(const float *x, const uint8_t *row, const float *scb, int o, int fmt, int I) {
    if (fmt == 5 || fmt == 4) {
        int gw2 = fmt == 4 ? g_ref_gs : 64;
        int ng = (I + gw2 - 1) / gw2; double sum = 0;
        for (int g = 0; g < ng; g++) {
            double a = 0; int end = (g + 1) * gw2 < I ? (g + 1) * gw2 : I;
            for (int i = g * gw2; i < end; i++) a += x[i] * deq(row, fmt, i);
            sum += a * scb[(size_t)o * ng + g];
        }
        return sum;
    }
    double sum = 0;
    for (int i = 0; i < I; i++) sum += x[i] * deq(row, fmt, i);
    return sum * scb[o];
}

static int run_case(int fmt, int S, int I, int O, int iters) {
    size_t rb = ref_rowbytes(fmt, I), nsc = ref_scales(fmt, I, O);
    float *x = malloc((size_t)S * I * sizeof(float));
    uint8_t *w = malloc(rb * O);
    float *sc = malloc(nsc * sizeof(float));
    float *yg = malloc((size_t)S * O * sizeof(float));
    float *yc = malloc((size_t)S * O * sizeof(float));
    for (int i = 0; i < S * I; i++) x[i] = (float)((rand() % 200 - 100) / 100.0);
    for (size_t i = 0; i < rb * O; i++) w[i] = rand() & 0xff;
    for (size_t o = 0; o < nsc; o++) sc[o] = 0.01f + (rand() % 100) / 10000.0f;

    ColiVkTensor *t = NULL;
    if (!coli_vk_matmul(&t, yg, x, w, sc, fmt, S, I, O, g_ref_gs)) { printf("matmul failed\n"); return 1; }
    cpu_ref(yc, x, w, sc, fmt, S, I, O);
    double maxerr = 0, maxrel = 0;
    for (int i = 0; i < S * O; i++) {
        double e = fabs(yg[i] - yc[i]); if (e > maxerr) maxerr = e;
        if (fabs(yc[i]) > 1e-2) { double r = e / fabs(yc[i]); if (r > maxrel) maxrel = r; }
    }
    // microbench (GPU)
    double t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_matmul(&t, yg, x, w, sc, fmt, S, I, O, g_ref_gs);
    double gpu_ms = (now() - t0) * 1000 / iters;
    // microbench (CPU ref, 1 iter — it's slow)
    double c0 = now(); cpu_ref(yc, x, w, sc, fmt, S, I, O); double cpu_ms = (now() - c0) * 1000;
    printf("fmt=%d S=%d I=%d O=%d | maxerr=%.4g maxrel=%.4g | gpu=%.3f ms  cpu_ref=%.3f ms\n",
           fmt, S, I, O, maxerr, maxrel, gpu_ms, cpu_ms);
    coli_vk_tensor_free(t);
    free(x); free(w); free(sc); free(yg); free(yc);
    return maxrel > 1e-3 ? 1 : 0;
}

/* Device argmax tail: the same projection through coli_vk_matmul (full logits read back)
 * and through coli_vk_matmul_argmax (only the winner comes back) must agree on the
 * winning index — including the tie rule, so the vector is fed duplicate maxima. */
static int run_argmax_case(int fmt, int I, int O, int dup) {
    size_t rb = ref_rowbytes(fmt, I), nsc = ref_scales(fmt, I, O);
    float *x = malloc((size_t)I * sizeof(float));
    uint8_t *w = malloc(rb * O);
    float *sc = malloc(nsc * sizeof(float));
    float *y = malloc((size_t)O * sizeof(float));
    for (int i = 0; i < I; i++) x[i] = (float)((rand() % 200 - 100) / 100.0);
    for (size_t i = 0; i < rb * O; i++) w[i] = rand() & 0xff;
    for (size_t o = 0; o < nsc; o++) sc[o] = 0.01f + (rand() % 100) / 10000.0f;

    ColiVkTensor *t = NULL;
    int rc = 1;
    if (!coli_vk_matmul(&t, y, x, w, sc, fmt, 1, I, O, g_ref_gs)) { printf("argmax: matmul failed\n"); goto done; }
    if (dup) {   /* force `dup` rows to share the max so the lowest-index rule is exercised */
        float mx = y[0]; for (int i = 1; i < O; i++) if (y[i] > mx) mx = y[i];
        size_t ng = nsc / (size_t)O;   /* sc is [O][ng]: scaling every group scales the logit */
        for (int k = 0; k < dup; k++) {
            int o = (int)((size_t)rand() * O / ((size_t)RAND_MAX + 1));
            if (fabsf(y[o]) < 1e-6f) continue;
            float f = mx / y[o];
            for (size_t g = 0; g < ng; g++) sc[(size_t)o * ng + g] *= f;
        }
        coli_vk_tensor_free(t); t = NULL;
        if (!coli_vk_matmul(&t, y, x, w, sc, fmt, 1, I, O, g_ref_gs)) { printf("argmax: matmul failed\n"); goto done; }
    }
    int want = 0; for (int i = 1; i < O; i++) if (y[i] > y[want]) want = i;   /* ascending scan */
    int got; float gv;
    if (!coli_vk_matmul_argmax(&t, w, sc, fmt, I, O, g_ref_gs, x, &got, &gv)) {
        printf("argmax unavailable (argmax.spv missing?)\n"); rc = 0; goto done;
    }
    rc = (got != want || gv != y[want]);
    printf("argmax fmt=%d I=%d O=%d dup=%d | host=%d(%.6f) device=%d(%.6f) %s\n",
           fmt, I, O, dup, want, y[want], got, gv, rc ? "FAIL" : "ok");
done:
    if (t) coli_vk_tensor_free(t);
    free(x); free(w); free(sc); free(y);
    return rc;
}

/* Batched throughput: record N dispatches in ONE command buffer, one submit + one
 * fence wait — the amortized per-matmul cost with the submit roundtrip spread across
 * the batch (how the real expert tier would drive it). Reuses the descriptor binding
 * left by a prior coli_vk_matmul call for this tensor/shape. */
static double bench_batched(ColiVkTensor *t, const float *x, int fmt, int S, int I, int O, int N) {
    size_t xb = (size_t)S * I * sizeof(float);
    memcpy(G.x.ptr, x, xb);
    vkResetCommandBuffer(G.cmd, 0);
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(G.cmd, &begin);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt, 0, 1, &G.dset, 0, NULL);
    struct PC pc = {fmt, S, I, O, t->rowWords, t->gs};
    vkCmdPushConstants(G.cmd, G.plyt, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    for (int i = 0; i < N; i++) {
        vkCmdDispatch(G.cmd, (uint32_t)((O + 7) / 8), (uint32_t)S, 1);
        vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkEndCommandBuffer(G.cmd);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    for (int warm = 0; warm < 2; warm++) {
        vkResetFences(G.dev, 1, &G.fence); vkQueueSubmit(G.queue, 1, &si, G.fence);
        vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, 10000000000ULL);
    }
    int iters = 10; double t0 = now();
    for (int k = 0; k < iters; k++) {
        vkResetFences(G.dev, 1, &G.fence); vkQueueSubmit(G.queue, 1, &si, G.fence);
        vkWaitForFences(G.dev, 1, &G.fence, VK_TRUE, 10000000000ULL);
    }
    G.cmd_ready = 0;   /* we clobbered the cached command buffer */
    return (now() - t0) * 1000.0 / iters / N;   /* ms per matmul, roundtrip amortized */
}

/* Fused gate+up correctness vs CPU ref: hidden = silu(gate*x)*(up*x). */
static int run_gate_up(int fmt, int S, int D, int I) {
    size_t rb = ref_rowbytes(fmt, D), nsc = ref_scales(fmt, D, I);
    float *x = malloc((size_t)S*D*4); uint8_t *gw = malloc(rb*I), *uw = malloc(rb*I);
    float *gs = malloc(nsc*4), *us = malloc(nsc*4);
    float *hg = malloc((size_t)S*I*4), *hc = malloc((size_t)S*I*4);
    for (int i = 0; i < S*D; i++) x[i] = (rand()%200-100)/100.0f;
    for (size_t i = 0; i < rb*I; i++) { gw[i] = rand()&0xff; uw[i] = rand()&0xff; }
    for (size_t o = 0; o < nsc; o++) { gs[o] = 0.01f+(rand()%100)/10000.0f; us[o] = 0.01f+(rand()%100)/10000.0f; }
    ColiVkTensor *tg = NULL, *tu = NULL;
    if (!coli_vk_gate_up(&tg, &tu, hg, x, gw, gs, uw, us, fmt, S, D, I, g_ref_gs)) { printf("gate_up failed\n"); return 1; }
    for (int s = 0; s < S; s++) for (int o = 0; o < I; o++) {
        float gt = (float)ref_dot(x+(size_t)s*D, gw+(size_t)o*rb, gs, o, fmt, D);
        float ut = (float)ref_dot(x+(size_t)s*D, uw+(size_t)o*rb, us, o, fmt, D);
        hc[s*I+o] = (gt/(1.0f+expf(-gt)))*ut;
    }
    double maxrel = 0;
    for (int i = 0; i < S*I; i++) { double e = fabs(hg[i]-hc[i]); if (fabs(hc[i])>1e-2) { double r = e/fabs(hc[i]); if (r>maxrel) maxrel = r; } }
    printf("gate_up(fused) fmt=%d S=%d D=%d I=%d | maxrel=%.4g\n", fmt, S, D, I, maxrel);
    coli_vk_tensor_free(tg); coli_vk_tensor_free(tu);
    free(x); free(gw); free(uw); free(gs); free(us); free(hg); free(hc);
    return maxrel > 1e-3 ? 1 : 0;
}

/* Batched throughput of the fused gate_up (N dispatches / one submit). */
static double bench_gu_batched(ColiVkTensor *tg, const float *x, int fmt, int S, int D, int I, int N) {
    memcpy(G.x.ptr, x, (size_t)S*D*sizeof(float));
    vkResetCommandBuffer(G.cmd, 0);
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(G.cmd, &begin);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gu);
    vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &G.dset_gu, 0, NULL);
    struct PC pc = {fmt, S, D, I, tg->rowWords, tg->gs};
    vkCmdPushConstants(G.cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    for (int i = 0; i < N; i++) {
        vkCmdDispatch(G.cmd, (uint32_t)((I+7)/8), (uint32_t)S, 1);
        vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkEndCommandBuffer(G.cmd);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    for (int w = 0; w < 2; w++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    int iters = 10; double t0 = now();
    for (int k = 0; k < iters; k++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    return (now()-t0)*1000.0/iters/N;
}

/* FAIR fused-gate_up throughput: cycle K DISTINCT experts (own descriptor set each) so
 * weights come from VRAM, not L2 — matching ROCm's expert_group reading distinct experts.
 * Returns ms per gate_up (one expert). */
static double bench_experts_fair(int fmt, int D, int I, int K, int Npass) {
    if (K > 32) K = 32;
    size_t rb = ref_rowbytes(fmt, D), nsc = ref_scales(fmt, D, I);
    ColiVkTensor *tg[32] = {0}, *tu[32] = {0};
    float *h = malloc((size_t)I*4), *x = malloc((size_t)D*4);
    for (int i = 0; i < D; i++) x[i] = (rand()%200-100)/100.0f;
    for (int c = 0; c < K; c++) {
        uint8_t *gw = malloc(rb*I), *uw = malloc(rb*I); float *gs = malloc(nsc*4), *us = malloc(nsc*4);
        for (size_t i = 0; i < rb*I; i++) { gw[i] = rand()&0xff; uw[i] = rand()&0xff; }
        for (size_t o = 0; o < nsc; o++) { gs[o] = 0.01f; us[o] = 0.01f; }
        coli_vk_gate_up(&tg[c], &tu[c], h, x, gw, gs, uw, us, fmt, 1, D, I, g_ref_gs);   /* uploads distinct experts */
        free(gw); free(uw); free(gs); free(us);
    }
    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = (uint32_t)(6*K)};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = (uint32_t)K, .poolSizeCount = 1, .pPoolSizes = &ps};
    VkDescriptorPool pool; vkCreateDescriptorPool(G.dev, &dpi, NULL, &pool);
    VkDescriptorSetLayout lays[32]; VkDescriptorSet sets[32]; for (int c = 0; c < K; c++) lays[c] = G.dsl_gu;
    VkDescriptorSetAllocateInfo dsa = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool, .descriptorSetCount = (uint32_t)K, .pSetLayouts = lays};
    vkAllocateDescriptorSets(G.dev, &dsa, sets);
    memcpy(G.x.ptr, x, (size_t)D*4);
    for (int c = 0; c < K; c++) {
        VkDescriptorBufferInfo bi[6] = {{.buffer=G.x.buf,.range=VK_WHOLE_SIZE},{.buffer=tg[c]->wbuf,.range=VK_WHOLE_SIZE},{.buffer=tg[c]->sbuf,.range=VK_WHOLE_SIZE},{.buffer=tu[c]->wbuf,.range=VK_WHOLE_SIZE},{.buffer=tu[c]->sbuf,.range=VK_WHOLE_SIZE},{.buffer=G.h.buf,.range=VK_WHOLE_SIZE}};
        VkWriteDescriptorSet w[6]; for (int i = 0; i < 6; i++) w[i] = (VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=sets[c],.dstBinding=(uint32_t)i,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi[i]};
        vkUpdateDescriptorSets(G.dev, 6, w, 0, NULL);
    }
    vkResetCommandBuffer(G.cmd, 0);
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(G.cmd, &begin);
    vkCmdBindPipeline(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.pipe_gu);
    struct PC pc = {fmt, 1, D, I, tg[0]->rowWords, tg[0]->gs};
    vkCmdPushConstants(G.cmd, G.plyt_gu, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    for (int pass = 0; pass < Npass; pass++) for (int c = 0; c < K; c++) {
        vkCmdBindDescriptorSets(G.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, G.plyt_gu, 0, 1, &sets[c], 0, NULL);
        vkCmdDispatch(G.cmd, (uint32_t)((I+7)/8), 1, 1);
        vkCmdPipelineBarrier(G.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    }
    vkEndCommandBuffer(G.cmd);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.cmd};
    for (int w = 0; w < 2; w++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    int iters = 10; double t0 = now();
    for (int k = 0; k < iters; k++) { vkResetFences(G.dev,1,&G.fence); vkQueueSubmit(G.queue,1,&si,G.fence); vkWaitForFences(G.dev,1,&G.fence,VK_TRUE,10000000000ULL); }
    double ms = (now()-t0)*1000.0/iters/((double)Npass*K);
    vkDestroyDescriptorPool(G.dev, pool, NULL);
    for (int c = 0; c < K; c++) { coli_vk_tensor_free(tg[c]); coli_vk_tensor_free(tu[c]); }
    free(h); free(x); G.cmd_ready = 0; G.bound_tensor = NULL;
    return ms;
}

/* Full expert-group correctness (vs CPU ref) + fair throughput: K distinct experts,
 * one submit, hidden on-device. The real comparison to ROCm's coli_cuda_expert_group. */
static int run_expert_group(int fmt, int D, int I, int K) {
    if (K > 64) K = 64;
    size_t gu_rb = ref_rowbytes(fmt, D), gu_sc = ref_scales(fmt, D, I);
    size_t d_rb  = ref_rowbytes(fmt, I), d_sc  = ref_scales(fmt, I, D);
    ColiVkTensor *tg[64] = {0}, *tu[64] = {0}, *td[64] = {0};
    uint8_t *hgw[64], *huw[64], *hdw[64]; float *hgs[64], *hus[64], *hds[64];
    float *x = malloc((size_t)K*D*4), *yg = malloc((size_t)K*D*4), *yc = malloc((size_t)K*D*4);
    float *tmp = malloc((size_t)(D > I ? D : I) * 4);
    for (int i = 0; i < K*D; i++) x[i] = (rand()%200-100)/100.0f;
    for (int c = 0; c < K; c++) {
        hgw[c] = malloc(gu_rb*I); huw[c] = malloc(gu_rb*I); hdw[c] = malloc(d_rb*D);
        for (size_t i = 0; i < gu_rb*I; i++) { hgw[c][i] = rand()&0xff; huw[c][i] = rand()&0xff; }
        for (size_t i = 0; i < d_rb*D; i++) hdw[c][i] = rand()&0xff;
        hgs[c] = malloc(gu_sc*4); hus[c] = malloc(gu_sc*4); hds[c] = malloc(d_sc*4);
        for (size_t o = 0; o < gu_sc; o++) { hgs[c][o] = 0.01f+(rand()%100)/10000.0f; hus[c][o] = 0.01f+(rand()%100)/10000.0f; }
        for (size_t o = 0; o < d_sc; o++) hds[c][o] = 0.01f+(rand()%100)/10000.0f;
        coli_vk_matmul(&tg[c], tmp, x, hgw[c], hgs[c], fmt, 1, D, I, g_ref_gs);   /* upload gate  (D->I) */
        coli_vk_matmul(&tu[c], tmp, x, huw[c], hus[c], fmt, 1, D, I, g_ref_gs);   /* upload up    (D->I) */
        coli_vk_matmul(&td[c], tmp, x, hdw[c], hds[c], fmt, 1, I, D, g_ref_gs);   /* upload down  (I->D) */
    }
    int rows[64]; for (int c = 0; c < K; c++) rows[c] = 1;
    if (!coli_vk_expert_group(tg, tu, td, rows, K, yg, x)) { printf("expert_group failed\n"); return 1; }
    float *hid = malloc((size_t)I*4);
    for (int c = 0; c < K; c++) {
        float *xc = x + (size_t)c*D;
        for (int o = 0; o < I; o++) {
            float gt = (float)ref_dot(xc, hgw[c]+(size_t)o*gu_rb, hgs[c], o, fmt, D);
            float ut = (float)ref_dot(xc, huw[c]+(size_t)o*gu_rb, hus[c], o, fmt, D);
            hid[o] = (gt/(1.0f+expf(-gt)))*ut;
        }
        for (int d = 0; d < D; d++)
            yc[c*D+d] = (float)ref_dot(hid, hdw[c]+(size_t)d*d_rb, hds[c], d, fmt, I);
    }
    double maxrel = 0;
    for (int i = 0; i < K*D; i++) { double e = fabs(yg[i]-yc[i]); if (fabs(yc[i])>1e-2) { double r = e/fabs(yc[i]); if (r>maxrel) maxrel = r; } }
    coli_vk_expert_group(tg, tu, td, rows, K, yg, x);   /* warm + leaves G.eg_cmd recorded */
    /* async issue/take must reproduce the sync result exactly (same buffers/records) */
    {
        float *ya = malloc((size_t)K*D*4);
        if (!coli_vk_expert_group_issue(tg, tu, td, rows, K, x) || !coli_vk_expert_group_take(ya)) {
            printf("expert_group issue/take failed\n"); maxrel = 1;
        } else {
            double amr = 0;
            for (int i = 0; i < K*D; i++) { double e = fabs(ya[i]-yg[i]); if (fabs(yg[i])>1e-2) { double r = e/fabs(yg[i]); if (r>amr) amr = r; } }
            if (amr > 1e-6) { printf("issue/take deviates from sync: %.4g\n", amr); maxrel = 1; }
        }
        free(ya);
    }
    int iters = 20; double t0 = now();
    for (int k = 0; k < iters; k++) coli_vk_expert_group(tg, tu, td, rows, K, yg, x);
    double ms = (now()-t0)*1000.0/iters/K;
    /* GPU-only: re-submit the already-recorded command buffer (skips per-call host setup:
     * descriptor updates + recording), isolating raw GPU throughput. */
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &G.eg_cmd};
    for (int w = 0; w < 2; w++) { vkResetFences(G.dev,1,&G.eg_fence); vkQueueSubmit(G.queue,1,&si,G.eg_fence); vkWaitForFences(G.dev,1,&G.eg_fence,VK_TRUE,10000000000ULL); }
    double g0 = now();
    for (int k = 0; k < iters; k++) { vkResetFences(G.dev,1,&G.eg_fence); vkQueueSubmit(G.queue,1,&si,G.eg_fence); vkWaitForFences(G.dev,1,&G.eg_fence,VK_TRUE,10000000000ULL); }
    double gpums = (now()-g0)*1000.0/iters/K;
    printf("FULL VK expert_group fmt=%d %2d experts | maxrel=%.4g | per-call %.4f  GPU-only %.4f ms/expert (ROCm 0.179)\n",
           fmt, K, maxrel, ms, gpums);
    free(hid); free(x); free(yg); free(yc); free(tmp);
    for (int c = 0; c < K; c++) {
        coli_vk_tensor_free(tg[c]); coli_vk_tensor_free(tu[c]); coli_vk_tensor_free(td[c]);
        free(hgw[c]); free(huw[c]); free(hdw[c]); free(hgs[c]); free(hus[c]); free(hds[c]);
    }
    /* The gate_up -> down chain accumulates fp32 rounding across two long reductions
     * (D=6144 then I) vs the double-precision CPU ref; 1-2e-3 shows up at any K
     * depending on the random draw, and is fine for greedy argmax. 3e-3 keeps the
     * gate honest without flagging the known fp behavior as a failure. */
    return maxrel > 3e-3 ? 1 : 0;
}

/* MLA absorb attention vs a CPU ref that mirrors glm.c's absorb loop exactly:
 * qabs = sum_d q[d]*deq(row rbase+d)*ws, scores over cache rows [st0, T-S+s],
 * softmax, weighted latent, value-row projection. */
static int run_absorb(int fmt, int S, int H, int Q, int R, int V, int K, int st0, int T, int layer) {
    size_t rb = ref_rowbytes(fmt, K);
    int O = H * (Q + V), ngK = (K + 63) / 64;
    size_t nws = ref_scales(fmt, K, O);
    uint8_t *w = malloc(rb * O); float *ws = malloc(nws * 4);
    float *q = malloc((size_t)S * H * (Q + R) * 4);
    float *L = malloc((size_t)T * K * 4), *Rr = malloc((size_t)T * R * 4);
    float *cg = malloc((size_t)S * H * V * 4), *cc = malloc((size_t)S * H * V * 4);
    for (size_t i = 0; i < rb * (size_t)O; i++) w[i] = rand() & 0xff;
    for (size_t o = 0; o < nws; o++) ws[o] = 0.01f + (rand() % 100) / 10000.0f;
    for (int i = 0; i < S * H * (Q + R); i++) q[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < T * K; i++) L[i] = (rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < T * R; i++) Rr[i] = (rand() % 200 - 100) / 100.0f;
    float scale = 0.13f;
    if (!coli_vk_kv_ensure(layer, T, K, R)) { printf("kv_ensure failed\n"); return 1; }
    for (int t = 0; t < T; t++)
        if (!coli_vk_kv_row(layer, t, L + (size_t)t * K, Rr + (size_t)t * R)) { printf("kv_row failed\n"); return 1; }
    ColiVkTensor *kvb = NULL;
    if (!coli_vk_attention_absorb(&kvb, w, ws, fmt, g_ref_gs, cg, q, layer, S, H, Q, R, V, K, st0, T, scale)) {
        printf("absorb failed\n"); return 1; }
    float *qabs = malloc((size_t)K * 4), *clat = malloc((size_t)K * 4), *sc = malloc((size_t)(T - st0) * 4);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        const float *qp = q + ((size_t)s * H + h) * (Q + R), *qr = qp + Q;
        int rbase = h * (Q + V), nt = (T - S + s + 1) - st0;
        for (int i = 0; i < K; i++) qabs[i] = 0;
        for (int d = 0; d < Q; d++) { const uint8_t *row = w + (size_t)(rbase + d) * rb;
            for (int i = 0; i < K; i++) {
                float sw = fmt == 5 ? ws[(size_t)(rbase + d) * ngK + (i >> 6)]
                         : fmt == 4 ? ws[(size_t)(rbase + d) * ((K + g_ref_gs - 1) / g_ref_gs) + i / g_ref_gs]
                         : ws[rbase + d];
                qabs[i] += qp[d] * deq(row, fmt, i) * sw; } }
        for (int j = 0; j < nt; j++) { int t = st0 + j;
            double a = 0;
            for (int i = 0; i < K; i++) a += qabs[i] * L[(size_t)t * K + i];
            for (int d = 0; d < R; d++) a += qr[d] * Rr[(size_t)t * R + d];
            sc[j] = (float)(a * scale); }
        float mx = sc[0]; for (int j = 1; j < nt; j++) if (sc[j] > mx) mx = sc[j];
        double sum = 0; for (int j = 0; j < nt; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
        for (int i = 0; i < K; i++) clat[i] = 0;
        for (int j = 0; j < nt; j++) { float a = (float)(sc[j] / sum);
            for (int i = 0; i < K; i++) clat[i] += a * L[(size_t)(st0 + j) * K + i]; }
        for (int v = 0; v < V; v++)
            cc[((size_t)s * H + h) * V + v] =
                (float)ref_dot(clat, w + (size_t)(rbase + Q + v) * rb, ws, rbase + Q + v, fmt, K);
    }
    double maxrel = 0, maxerr = 0;
    for (int i = 0; i < S * H * V; i++) { double e = fabs(cg[i] - cc[i]); if (e > maxerr) maxerr = e;
        if (fabs(cc[i]) > 1e-2) { double r = e / fabs(cc[i]); if (r > maxrel) maxrel = r; } }
    double t0 = now(); int iters = 20;   /* per-call cost, the engine pattern (one submit/layer) */
    for (int k = 0; k < iters; k++)
        coli_vk_attention_absorb(&kvb, w, ws, fmt, g_ref_gs, cg, q, layer, S, H, Q, R, V, K, st0, T, scale);
    double ms = (now() - t0) * 1000 / iters;
    printf("absorb fmt=%d S=%d H=%d Q=%d R=%d V=%d K=%d st0=%d T=%d | maxerr=%.4g maxrel=%.4g | %.3f ms/call\n",
           fmt, S, H, Q, R, V, K, st0, T, maxerr, maxrel, ms);
    /* fused absorb+project vs (CPU ctx ref) @ (CPU o ref) */
    int Dout = 512; size_t orb = ref_rowbytes(fmt, H * V), onsc = ref_scales(fmt, H * V, Dout);
    uint8_t *owt = malloc(orb * Dout); float *osc = malloc(onsc * 4);
    float *og = malloc((size_t)S * Dout * 4), *oc = malloc((size_t)S * Dout * 4);
    for (size_t i = 0; i < orb * (size_t)Dout; i++) owt[i] = rand() & 0xff;
    for (size_t o = 0; o < onsc; o++) osc[o] = 0.01f + (rand() % 100) / 10000.0f;
    ColiVkTensor *ot = NULL;
    if (!coli_vk_attention_absorb_project(&kvb, w, ws, fmt, g_ref_gs, &ot, owt, osc, fmt, g_ref_gs,
            og, q, layer, S, H, Q, R, V, K, st0, T, scale, Dout)) { printf("absorb_project failed\n"); return 1; }
    cpu_ref(oc, cc, owt, osc, fmt, S, H * V, Dout);
    double pmaxrel = 0;
    for (int i = 0; i < S * Dout; i++) { double e = fabs(og[i] - oc[i]);
        if (fabs(oc[i]) > 1e-2) { double r = e / fabs(oc[i]); if (r > pmaxrel) pmaxrel = r; } }
    t0 = now();
    for (int k = 0; k < iters; k++)
        coli_vk_attention_absorb_project(&kvb, w, ws, fmt, g_ref_gs, &ot, owt, osc, fmt, g_ref_gs,
            og, q, layer, S, H, Q, R, V, K, st0, T, scale, Dout);
    printf("absorb+project fused                  | maxrel=%.4g | %.3f ms/call (unfused absorb was %.3f)\n",
           pmaxrel, (now() - t0) * 1000 / iters, ms);
    coli_vk_tensor_free(ot); free(owt); free(osc); free(og); free(oc);
    coli_vk_tensor_free(kvb);
    free(w); free(ws); free(q); free(L); free(Rr); free(cg); free(cc); free(qabs); free(clat); free(sc);
    return (maxrel > 2e-3 || pmaxrel > 5e-3) ? 1 : 0;
}


/* q-prep chain vs CPU ref: matmul(qa) -> rmsnorm -> matmul(qb), kv_a alongside. */
static int run_qprep(int fmt, int S, int I, int Oqa, int Okva, int Oqb) {
    size_t rba = ref_rowbytes(fmt, I), rbb = ref_rowbytes(fmt, Oqa);
    size_t nsa = ref_scales(fmt, I, Oqa), nsk = ref_scales(fmt, I, Okva), nsb = ref_scales(fmt, Oqa, Oqb);
    uint8_t *wa = malloc(rba * Oqa), *wk = malloc(rba * Okva), *wb = malloc(rbb * Oqb);
    float *sa = malloc(nsa * 4), *sk = malloc(nsk * 4), *sb = malloc(nsb * 4);
    float *ln = malloc((size_t)Oqa * 4), *x = malloc((size_t)S * I * 4);
    float *qg = malloc((size_t)S * Oqb * 4), *kvg = malloc((size_t)S * Okva * 4);
    float *lat = malloc((size_t)S * Oqa * 4), *qc = malloc((size_t)S * Oqb * 4), *kvc = malloc((size_t)S * Okva * 4);
    for (size_t i = 0; i < rba * (size_t)Oqa; i++) wa[i] = rand() & 0xff;
    for (size_t i = 0; i < rba * (size_t)Okva; i++) wk[i] = rand() & 0xff;
    for (size_t i = 0; i < rbb * (size_t)Oqb; i++) wb[i] = rand() & 0xff;
    for (size_t o = 0; o < nsa; o++) sa[o] = 0.01f + (rand() % 100) / 10000.0f;
    for (size_t o = 0; o < nsk; o++) sk[o] = 0.01f + (rand() % 100) / 10000.0f;
    for (size_t o = 0; o < nsb; o++) sb[o] = 0.01f + (rand() % 100) / 10000.0f;
    for (int i = 0; i < Oqa; i++) ln[i] = 0.5f + (rand() % 100) / 100.0f;
    for (int i = 0; i < S * I; i++) x[i] = (rand() % 2000 - 1000) / 500.0f;
    ColiVkTensor *ta = NULL, *tk = NULL, *tb = NULL;
    static int qp_layer = 140;         /* distinct high slot per case: the per-layer ln
                                        * cache is upload-once by design (engine weights
                                        * are immutable); reuse here would mix cases */
    int layer = qp_layer++;
    if (!coli_vk_attn_qprep(layer, &ta, wa, sa, Oqa, &tk, wk, sk, Okva, &tb, wb, sb, Oqb,
                            fmt, g_ref_gs, ln, 1e-6f, x, S, I, qg, kvg, NULL)) {
        printf("qprep unavailable (rmsnorm.spv missing?)\n"); return 1; }
    cpu_ref(lat, x, wa, sa, fmt, S, I, Oqa);
    cpu_ref(kvc, x, wk, sk, fmt, S, I, Okva);
    for (int s = 0; s < S; s++) {                       /* rmsnorm rows like colibri.c */
        double ms = 0; float *r = lat + (size_t)s * Oqa;
        for (int i = 0; i < Oqa; i++) ms += (double)r[i] * r[i];
        float rr = 1.0f / sqrtf((float)(ms / Oqa) + 1e-6f);
        for (int i = 0; i < Oqa; i++) r[i] = r[i] * rr * ln[i];
    }
    cpu_ref(qc, lat, wb, sb, fmt, S, Oqa, Oqb);
    float mq = 0, mk = 0;
    for (int i = 0; i < S * Oqb; i++) { float d = fabsf(qg[i] - qc[i]) / (fabsf(qc[i]) + 1e-3f); if (d > mq) mq = d; }
    for (int i = 0; i < S * Okva; i++) { float d = fabsf(kvg[i] - kvc[i]) / (fabsf(kvc[i]) + 1e-3f); if (d > mk) mk = d; }
    printf("qprep fmt=%d S=%d I=%d (%d->%d, kv %d) | maxrel q %.4g kv %.4g\n", fmt, S, I, Oqa, Oqb, Okva, mq, mk);
    coli_vk_tensor_free(ta); coli_vk_tensor_free(tk); coli_vk_tensor_free(tb);
    free(wa); free(wk); free(wb); free(sa); free(sk); free(sb); free(ln); free(x);
    free(qg); free(kvg); free(lat); free(qc); free(kvc);
    /* q crosses TWO quantized reductions + the norm; random +-8-nibble rows are
     * cancellation-heavy, so fp32-vs-f64 divergence amplifies ~10x vs one GEMV
     * (same reasoning as the expert_group 3e-3 threshold). Engine-level logit
     * comparison on real weights is the tight check. */
    return mq > 1e-2f || mk > 1e-3f;
}

int main(int argc, char **argv) {
    qwen_opts_init_defaults();
    const char *spv = argc > 1 ? argv[1] : "shaders/qmatmul.spv";
    if (!coli_vk_init(spv)) { printf("vk init failed\n"); return 1; }
    srand(1234);
    int bad = 0;
    /* COLI_VK_TEST_BALLAST=N: allocate N idle 4 MB device buffers before benching.
     * Probes whether per-submit cost scales with the process's ALLOCATION COUNT
     * (RADV/amdgpu CS buffer-list accounting) independent of bytes — the suspected
     * mechanism behind decode attention degrading with expert-tier size even with
     * VRAM to spare (7.9s @2.6k BOs -> 15.6s @4.3k with 2.9 GB free). */
    {
        int nb = g_qwen_opts.test_ballast;
        for (int i = 0; i < nb; i++) {
            VkBuffer b; VkDeviceMemory m;
            if (!alloc_hostvis_mt(4u << 20, &b, &m, NULL, G.memtype)) { printf("ballast stop at %d\n", i); break; }
        }
        if (nb) printf("ballast: %d x 4 MB idle allocations\n", nb);
    }
    bad |= run_case(1, 1, 6144, 1536, 50);   // int8 expert gate/up shape (S=1 decode)
    bad |= run_case(2, 1, 6144, 1536, 50);   // int4 expert
    bad |= run_case(1, 1, 1536, 6144, 50);   // down proj shape
    bad |= run_case(2, 8, 6144, 1536, 20);   // prefill/MTP batch
    bad |= run_case(1, 1, 512, 512, 100);    // small
    bad |= run_case(2, 1, 16384, 6144, 20);  // o_proj shape: I > xsh capacity (unstaged path)
    /* int3-g64 (fmt=5): per-group scales through every dense shape incl. tail groups */
    bad |= run_case(5, 1, 6144, 2048, 50);   // int3 expert gate/up shape
    bad |= run_case(5, 1, 2048, 6144, 50);   // int3 down proj shape
    bad |= run_case(5, 8, 6144, 2048, 20);   // int3 batch
    bad |= run_case(5, 1, 100, 64, 20);      // partial tail group (I%64 != 0)
    bad |= run_case(5, 1, 16384, 6144, 20);  // int3 o_proj shape (unstaged path)
    /* lm_head tail: device argmax vs a host scan of the same logits */
    bad |= run_argmax_case(1, 2048, 248320, 0);   // Qwen3.5-35B lm_head shape
    bad |= run_argmax_case(1, 2048, 248320, 8);   // ... with ties
    bad |= run_argmax_case(2, 512, 199, 0);       // fewer rows than the reduction has threads
    /* Batched (amortized) throughput on the int4 expert shapes — the real expert-tier pattern. */
    {
        int I = 6144, O = 2048;   /* our gate/up dims */
        float *x = malloc((size_t)I * 4); uint8_t *w = malloc((size_t)(I + 1) / 2 * O); float *sc = malloc((size_t)O * 4);
        for (int i = 0; i < I; i++) x[i] = (rand() % 200 - 100) / 100.0f;
        for (size_t i = 0; i < (size_t)(I + 1) / 2 * O; i++) w[i] = rand() & 0xff;
        for (int o = 0; o < O; o++) sc[o] = 0.01f + (rand() % 100) / 10000.0f;
        float *y = malloc((size_t)O * 4);
        ColiVkTensor *t = NULL; coli_vk_matmul(&t, y, x, w, sc, 2, 1, I, O, 0);   /* bind */
        printf("BATCHED int4 S=1 6144->2048 (our gate/up): %.4f ms/matmul (N=64, one submit)\n",
               bench_batched(t, x, 2, 1, I, O, 64));
        coli_vk_tensor_free(t); free(x); free(w); free(sc); free(y);
        I = 2048; O = 6144;   /* our down dims */
        x = malloc((size_t)I * 4); w = malloc((size_t)(I + 1) / 2 * O); sc = malloc((size_t)O * 4);
        for (int i = 0; i < I; i++) x[i] = (rand() % 200 - 100) / 100.0f;
        for (size_t i = 0; i < (size_t)(I + 1) / 2 * O; i++) w[i] = rand() & 0xff;
        for (int o = 0; o < O; o++) sc[o] = 0.01f + (rand() % 100) / 10000.0f;
        y = malloc((size_t)O * 4);
        t = NULL; coli_vk_matmul(&t, y, x, w, sc, 2, 1, I, O, 0);
        printf("BATCHED int4 S=1 2048->6144 (our down):    %.4f ms/matmul (N=64, one submit)\n",
               bench_batched(t, x, 2, 1, I, O, 64));
        coli_vk_tensor_free(t); free(x); free(w); free(sc); free(y);
    }
    /* FUSED gate+up: correctness + batched throughput (vs 2x separate gate/up). */
    {
        int D = 6144, I = 2048;
        bad |= run_gate_up(2, 1, D, I);
        bad |= run_gate_up(5, 1, D, I);   // int3-g64 fused pair (per-group scales)
        size_t rb = (size_t)(D + 1) / 2;
        float *x = malloc((size_t)D*4); uint8_t *gw = malloc(rb*I), *uw = malloc(rb*I);
        float *gs = malloc((size_t)I*4), *us = malloc((size_t)I*4), *h = malloc((size_t)I*4);
        for (int i = 0; i < D; i++) x[i] = (rand()%200-100)/100.0f;
        for (size_t i = 0; i < rb*I; i++) { gw[i] = rand()&0xff; uw[i] = rand()&0xff; }
        for (int o = 0; o < I; o++) { gs[o] = 0.01f; us[o] = 0.01f; }
        ColiVkTensor *tg = NULL, *tu = NULL; coli_vk_gate_up(&tg, &tu, h, x, gw, gs, uw, us, 2, 1, D, I, 0);
        printf("BATCHED fused gate_up int4 6144->2048:     %.4f ms (N=64, SAME expert = L2-cached)\n",
               bench_gu_batched(tg, x, 2, 1, D, I, 64));
        coli_vk_tensor_free(tg); coli_vk_tensor_free(tu); free(x); free(gw); free(uw); free(gs); free(us); free(h);
    }
    /* FAIR: cycle 8 distinct experts (VRAM reads, not L2) — matches ROCm expert_group. */
    printf("FAIR fused gate_up int4 6144->2048 (8 distinct experts): %.4f ms/expert\n",
           bench_experts_fair(2, 6144, 2048, 8, 8));
    printf("FAIR fused gate_up int3 6144->2048 (8 distinct experts): %.4f ms/expert\n",
           bench_experts_fair(5, 6144, 2048, 8, 8));
    /* FULL expert_group: the real primitive. Sweep K to see if per-expert cost is fixed
     * per-call overhead (drops with K) or per-dispatch (constant). */
    bad |= run_expert_group(2, 6144, 2048, 1);
    bad |= run_expert_group(2, 6144, 2048, 8);
    bad |= run_expert_group(2, 6144, 2048, 32);
    /* int3-g64 expert group: correctness + the 0.86x-bytes throughput question.
     * count=1 included — it is the SHARED-expert path shape in the engine. */
    bad |= run_qprep(1, 1, 6144, 1536, 576, 16384);   /* GLM q_a/kv_a/q_b decode shapes */
    bad |= run_qprep(1, 11, 6144, 1536, 576, 16384);  /* prefill batch */
    bad |= run_qprep(1, 2, 6144, 1536, 576, 16384);   /* S=2 (MTP verify) */
    bad |= run_qprep(2, 1, 6144, 1536, 576, 16384);   /* int4 dense variant */
    /* fmt=4 grouped int4 (#298 semantics), gs=64 across real shapes + gs=32 sanity */
    g_ref_gs = 64;
    bad |= run_case(4, 1, 6144, 2048, 50);
    bad |= run_case(4, 1, 2048, 6144, 50);
    bad |= run_case(4, 8, 6144, 1536, 20);
    bad |= run_case(4, 1, 16384, 6144, 10);
    g_ref_gs = 32;
    bad |= run_case(4, 1, 6144, 2048, 20);
    g_ref_gs = 64;
    bad |= run_expert_group(4, 6144, 2048, 8);
    bad |= run_expert_group(5, 6144, 2048, 1);
    bad |= run_expert_group(5, 6144, 2048, 8);
    bad |= run_expert_group(5, 6144, 2048, 32);
    /* Fused same-input matmul pair (the q_a + kv_a prologue pattern), int4 AND int3. */
    for (int pi = 0; pi < 3; pi++) {
        int pf = pi == 0 ? 2 : pi == 1 ? 5 : 4;
        int I = 6144, O1 = 2048, O2 = 576, S = 1;
        size_t rb = ref_rowbytes(pf, I);
        size_t n1 = ref_scales(pf, I, O1), n2 = ref_scales(pf, I, O2);
        uint8_t *w1 = malloc(rb * O1), *w2 = malloc(rb * O2);
        float *s1 = malloc(n1 * 4), *s2 = malloc(n2 * 4), *x = malloc((size_t)I * 4);
        float *y1 = malloc((size_t)O1 * 4), *y2 = malloc((size_t)O2 * 4);
        float *c1 = malloc((size_t)O1 * 4), *c2 = malloc((size_t)O2 * 4);
        for (size_t i = 0; i < rb * (size_t)O1; i++) w1[i] = rand() & 0xff;
        for (size_t i = 0; i < rb * (size_t)O2; i++) w2[i] = rand() & 0xff;
        for (size_t o = 0; o < n1; o++) s1[o] = 0.01f + (rand() % 100) / 10000.0f;
        for (size_t o = 0; o < n2; o++) s2[o] = 0.01f + (rand() % 100) / 10000.0f;
        for (int i = 0; i < I; i++) x[i] = (rand() % 200 - 100) / 100.0f;
        ColiVkTensor *t1 = NULL, *t2 = NULL;
        if (!coli_vk_matmul_pair(&t1, y1, w1, s1, O1, &t2, y2, w2, s2, O2, pf, x, S, I, g_ref_gs)) {
            printf("matmul_pair fmt=%d failed\n", pf); bad = 1;
        } else {
            cpu_ref(c1, x, w1, s1, pf, S, I, O1); cpu_ref(c2, x, w2, s2, pf, S, I, O2);
            double mr = 0;
            for (int i = 0; i < O1; i++) { double e = fabs(y1[i]-c1[i]); if (fabs(c1[i])>1e-2) { double r=e/fabs(c1[i]); if (r>mr) mr=r; } }
            for (int i = 0; i < O2; i++) { double e = fabs(y2[i]-c2[i]); if (fabs(c2[i])>1e-2) { double r=e/fabs(c2[i]); if (r>mr) mr=r; } }
            double t0 = now();
            for (int k = 0; k < 30; k++) coli_vk_matmul_pair(&t1, y1, w1, s1, O1, &t2, y2, w2, s2, O2, pf, x, S, I, g_ref_gs);
            printf("matmul_pair fmt=%d 6144->(2048,576)   | maxrel=%.4g | %.3f ms/pair-call\n", pf, mr, (now()-t0)*1000/30);
            bad |= mr > 1e-3;
        }
        coli_vk_tensor_free(t1); coli_vk_tensor_free(t2);
        free(w1); free(w2); free(s1); free(s2); free(x); free(y1); free(y2); free(c1); free(c2);
    }
    /* MLA absorb attention core (GLM decode shape + window/causal/int8 variants). */
    if (G.pipe_att) {
        bad |= run_absorb(2, 1, 64, 192, 64, 256, 512, 0, 300, 0);    // GLM-5.2 decode
        bad |= run_absorb(2, 1, 64, 192, 64, 256, 512, 17, 300, 1);   // kv_start window
        bad |= run_absorb(2, 2, 64, 192, 64, 256, 512, 0, 300, 2);    // S=2 causal (MTP verify)
        bad |= run_absorb(1, 1, 8, 128, 32, 64, 256, 0, 64, 3);       // int8, odd dims
        bad |= run_absorb(2, 1, 64, 192, 64, 256, 512, 0, 2000, 4);   // long context
        bad |= run_absorb(5, 1, 64, 192, 64, 256, 512, 0, 300, 5);    // int3-g64 kv_b + o
        bad |= run_absorb(5, 2, 64, 192, 64, 256, 512, 17, 300, 6);   // int3 S=2 causal + window
        bad |= run_absorb(4, 1, 64, 192, 64, 256, 512, 0, 300, 7);    // grouped-int4 kv_b + o
        bad |= run_absorb(4, 2, 64, 192, 64, 256, 512, 17, 300, 8);   // fmt=4 S=2 causal + window
    }
    printf(bad ? "FAIL\n" : "PASS\n");
    coli_vk_shutdown();
    return bad;
}
#endif /* VK_TEST */
