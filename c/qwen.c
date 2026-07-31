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

/* Qwen3.5/3.6-MoE (e.g. Qwen3.6-35B-A3B) inference engine in plain C.
 *
 * Architecture (distilled from HF transformers modeling_qwen3_5_moe.py): hybrid
 * decoder — repeating (3x GatedDeltaNet "linear attention" + 1x gated full
 * attention), EVERY layer followed by a 256-expert top-8 MoE with a
 * sigmoid-gated shared expert.
 *
 * Container produced by tools/convert_qwen35_moe.py:
 *   dense weights float32, routed experts int4 (fmt 2, colibri nibble packing)
 *   or int8 (fmt 1) with per-row scales in companion ".qs" tensors.
 *
 * Dense stays resident in RAM (f32). Routed experts are read on demand from
 * the container (already quantized -> tiny reads) into a per-layer LRU cache,
 * olmoe.c-style. With COLI_VULKAN the routed experts run on a PINNED VRAM
 * tier aligned with the GLM engine (colibri.c vk_registry_fill): the
 * top-COLI_VK_EXPERTS heat-ranked experts (persistent <SNAP>/.coli_usage
 * history) upload ONCE at startup at eviction priority 0.4, the fill stops
 * while COLI_VK_RESERVE_GB of device-local budget remains, and NOTHING
 * uploads on the decode path — each token's tier-resident experts run as ONE
 * coli_vk_expert_group submit, the rest on CPU. First run has no history:
 * tier stays empty, usage seeds as you generate.
 *
 * Modes (--snap <container> required):
 *   --prompt "..." [--ngen n]   text generation via tok.h (ChatML by default)
 *   qwen <cache> [ref.json]     oracle: greedy match against full_ids
 *   PPL=1                       teacher-forced NLL over full_ids
 *   LOGITS_OUT=f                dump per-step logits (f32) vs HF
 *
 * Tokenizer: container/tokenizer.json (copied by convert_qwen35_moe.py). Same
 * byte-level BPE / cl100k family as GLM — tok.h handles it unchanged. Chat
 * template (CHAT_TEMPLATE=1 default):
 *   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
 * Stop on <|im_end|> / <|endoftext|>. CHAT_TEMPLATE=0 = raw prompt bytes.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "st.h"
#include "compat.h"
#include "tok.h"
#include "qwen_opts.h"
#ifdef COLI_VULKAN
#include "backend_vulkan.h"
#endif

/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, vocab;
    int n_heads, n_kv_heads, head_dim, rot;          /* full attention */
    float theta, eps;
    int gdn_kh, gdn_kd, gdn_vh, gdn_vd, conv_k;      /* GatedDeltaNet */
    int key_dim, value_dim, conv_dim;                /* derived */
    int n_experts, topk, inter, sh_inter;            /* MoE */
    int expert_bits;                                 /* 4 (fmt 2) or 8 (fmt 1) */
    int expert_fmt;                                  /* 1=int8, 2=int4 per-row, 6=grouped-asym int4 (GPTQ) */
    int group_size;                                  /* fmt=6 group size (multiple of 8); 0 otherwise */
    unsigned char *is_full;                          /* [n_layers] layer_types */
} Cfg;

/* ---------- dense weight with an optional GPU-resident quantized copy ----------
 * With COLI_VULKAN + COLI_VK_DENSE (default on) the f32 dense matmuls quantize
 * once at startup (int8 per-row for projections/lm_head, the container's expert
 * fmt for the shared expert so it can join the expert_group submit) and run on
 * the GPU via coli_vk_matmul / coli_vk_matmul_pair; the f32 original is freed.
 * The quantized host copy stays as the CPU fallback if the VK device dies. */
typedef struct {
#ifdef COLI_VULKAN
    ColiVkTensor *t;               /* GPU-resident copy (NULL = not uploaded) */
#else
    void *t;
#endif
    uint8_t *q;                    /* quantized host copy (upload source + CPU fallback) */
    float *s;                      /* scales: per-row (fmt 1/2) or per-group [O,ng,2] (fmt 6) */
    int fmt;                       /* 1=int8, 2=int4, 6=grouped-asym int4 */
    int gs;                        /* fmt=6 group size (0 otherwise) */
} VkDense;

/* ---------- per-layer dense weights (resident f32) ---------- */
typedef struct {
    float *in_ln, *post_ln;
    /* full attention */
    float *q, *k, *v, *o, *qn, *kn;
    /* GatedDeltaNet */
    float *qkv, *z, *b, *a, *conv, *dtb, *alog, *gnorm, *outp;
    /* MoE dense side */
    float *router, *sh_g, *sh_u, *sh_d, *sh_gate;
    /* GPU-resident quantized copies (zeroed by calloc; filled by vk_dense_init) */
    VkDense gq, gk, gv, go;        /* full attention q/k/v/o projections */
    VkDense gqkv, gz, gout;        /* GatedDeltaNet in_proj_qkv / in_proj_z / out_proj */
    VkDense gsg, gsu, gsd;         /* shared expert gate/up/down (expert fmt, joins expert_group) */
    VkDense grouter;               /* MoE router int8 — fused into attn/GDN full submit */
    VkDense gba, gbb;              /* GDN in_proj_a / in_proj_b (stream path, device pack) */
} Layer;

/* ---------- routed-expert LRU cache (quantized weights as stored) ---------- */
typedef struct {
    int eid;
    uint8_t *g, *u, *d;            /* fmt 2: nibble-packed, fmt 1: int8 */
    float *gs, *us, *ds;           /* per-row scales */
    uint64_t used;
} Slot;
typedef struct { Slot *slots; int n, cap; } LCache;

typedef struct {
    Cfg c;
    shards S;
    float *embed, *lm_head, *final_norm;
    Layer *L;
    LCache *cache;
    uint64_t clock, hits, miss;
    /* full-attention KV cache: only for full layers, [n_kv][max_t][head_dim] */
    float **K, **V; int max_t;
    /* GDN recurrent state per linear layer */
    float **gdnS;                  /* [gdn_vh * gdn_kd * gdn_vd] f32 */
    float **convst;                /* [conv_k][conv_dim] ring, slot conv_k-1 = newest */
    int token_count;
    double dense_load_s;
    /* persistent expert selection counts (GLM's "cache that learns"): accumulate in
     * <SNAP>/.coli_usage across sessions, rank the startup VK tier fill */
    uint32_t *eusage;              /* [n_layers * n_experts] */
    char usage_path[2176];
    VkDense glmh;                  /* lm_head int8 GPU copy (embed stays f32 for the lookup) */
#ifdef COLI_VULKAN
    ColiVkTensor **vkreg;          /* [n_layers * n_experts * 3] gate/up/down VRAM copies;
                                    * PINNED tier filled once at startup, never touched at decode */
    int vk_on;
    int vk_reg_n;                  /* experts resident on the VK tier */
    int vk_gdn_on;                 /* GatedDeltaNet recurrence runs on the GPU (device-resident state) */
    int vk_gdn_full_on;            /* whole GDN block fused on the GPU (conv+qknorm+delta+outproj, one submit) */
    int vk_gqa_on;                 /* full GQA attention runs on the GPU (device KV mirror) */
    int vk_gqa_full_on;            /* whole GQA block fused (q/k/v matmul+norm+rope+attn+oproj, one submit) */
    int vk_route_on;               /* in/post RMSNorm + router fused into the attn/GDN full submit */
    int vk_stream_on;              /* P0/P1/P2: device resid stream + eg→route semaphore */
    int vk_stream_argmax;          /* stream path left resid on device for norm+argmax fuse */
    /* Persistent decode scratch (Phase 1): avoid per-token malloc in the hot path. */
    float *scratch_x, *scratch_nrm, *scratch_tmp, *scratch_last;
    int scratch_cap;               /* floats per buffer (= hidden when allocated) */
#endif
} Model;

/* ---------- utility ---------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
#if defined(__APPLE__)
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0*1024.0); }
#else
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0); } /* Linux KB; Win compat also KB */
#endif
static float *falloc(int64_t n) { float *p = malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM %lld floats\n",(long long)n);exit(1);} return p; }
static float *fcalloc(int64_t n) { float *p = calloc(n,sizeof(float)); if(!p){fprintf(stderr,"OOM %lld floats\n",(long long)n);exit(1);} return p; }
static int   *ialloc(int64_t n)  { int *p = malloc(n*sizeof(int)); if(!p){fprintf(stderr,"OOM %lld ints\n",(long long)n);exit(1);} return p; }
static inline float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }
static inline float siluf_(float x) { return x / (1.f + expf(-x)); }
/* softplus matching torch F.softplus default (beta=1, threshold=20) */
static inline float softplusf_(float x) { return x > 20.f ? x : log1pf(expf(x)); }

/* y[S,O] = x[S,I] @ W^T, W [O,I] row-major f32 */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xs[i] * w[i];
            y[(int64_t)s * O + o] = acc;
        }
    }
}

/* y[O] = dequant(W) @ x with colibri fmt-2 int4 packing: element j of a row is
 * nibble j&7 of uint32 word j>>3 (little-endian bytes: byte j>>1, low nibble
 * for even j), value = nibble-8, weight ~= value*scale[row]. */
static void matmul_q4(float *y, const float *x, const uint8_t *q, const float *scale, int I, int O) {
    int rb = I / 2;                       /* bytes per row */
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q + (int64_t)o * rb;
        float acc = 0.f;
        for (int j = 0; j < rb; j++) {
            uint8_t b = w[j];
            acc += x[2*j]   * (float)((int)(b & 0xF) - 8);
            acc += x[2*j+1] * (float)((int)(b >> 4)  - 8);
        }
        y[o] = acc * scale[o];
    }
}
/* y[O] = grouped-asymmetric int4 (fmt=6, GPTQ-style): sz holds [scale_g, zero_g]
 * interleaved per group (ng = ceil(I/gs) groups/row); weight = (nibble - zero_g)*scale_g.
 * Distributes the group scale over inputs like the Vulkan qmatmul.comp fmt=6 branch. */
static void matmul_q4g_asym(float *y, const float *x, const uint8_t *q, const float *sz, int I, int O, int gs) {
    int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = q + (int64_t)o * rb;
        const float *szr = sz + (int64_t)o * ng * 2;
        float acc = 0.f;
        for (int g = 0; g < ng; g++) {
            int i0 = g * gs, i1 = i0 + gs; if (i1 > I) i1 = I;
            float scale = szr[2*g], zero = szr[2*g+1], awn = 0.f, ax = 0.f;
            for (int i = i0; i < i1; i++) {
                int nib = (i & 1) ? (w[i>>1] >> 4) : (w[i>>1] & 0xF);
                awn += x[i] * (float)nib; ax += x[i];
            }
            acc += scale * (awn - zero * ax);
        }
        y[o] = acc;
    }
}
static void matmul_q8(float *y, const float *x, const uint8_t *q, const float *scale, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = (const int8_t *)q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}

/* Per-row absmax quantization into colibri fmt 1 (int8) or fmt 2 (int4, two
 * values per byte, stored value = v+8 with v in [-8,7], low nibble = even i). */
static void quant_rows(const float *W, int I, int O, int fmt, int gs, uint8_t *q, float *s) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        if (fmt == 6) {                /* grouped-asymmetric int4: per gs inputs a scale+zero */
            int ng = (I + gs - 1) / gs;
            uint8_t *row = q + (int64_t)o * ((I + 1) / 2);
            float *sr = s + (int64_t)o * ng * 2;
            for (int g = 0; g < ng; g++) {
                int i0 = g*gs, i1 = i0 + gs; if (i1 > I) i1 = I;
                float wmin = 1e30f, wmax = -1e30f;
                for (int i = i0; i < i1; i++) { float v = w[i]; if (v < wmin) wmin = v; if (v > wmax) wmax = v; }
                float scale = (wmax - wmin) / 15.f; if (scale < 1e-8f) scale = 1e-8f;
                float zero = -wmin / scale;
                sr[2*g] = scale; sr[2*g+1] = zero;
                for (int i = i0; i < i1; i += 2) {   /* gs even -> pairs align to bytes, no RMW */
                    int q0 = (int)lrintf(w[i]/scale + zero);   q0 = q0 < 0 ? 0 : q0 > 15 ? 15 : q0;
                    int q1 = 0;
                    if (i+1 < i1) { q1 = (int)lrintf(w[i+1]/scale + zero); q1 = q1 < 0 ? 0 : q1 > 15 ? 15 : q1; }
                    row[i/2] = (uint8_t)(q0 | (q1 << 4));
                }
            }
            continue;
        }
        float amax = 0.f;
        for (int i = 0; i < I; i++) { float a = fabsf(w[i]); if (a > amax) amax = a; }
        if (fmt == 1) {
            float sc = amax > 0 ? amax / 127.f : 1.f;
            int8_t *row = (int8_t *)q + (int64_t)o * I;
            for (int i = 0; i < I; i++) {
                int v = (int)lrintf(w[i] / sc);
                row[i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
            }
            s[o] = sc;
        } else {                       /* fmt 2: int4 nibble pack, value = v+8 */
            float sc = amax > 0 ? amax / 7.f : 1.f;
            uint8_t *row = q + (int64_t)o * ((I + 1) / 2);
            for (int i = 0; i < I; i += 2) {
                int v0 = (int)lrintf(w[i] / sc);
                v0 = v0 > 7 ? 7 : v0 < -8 ? -8 : v0;
                int v1 = 0;
                if (i + 1 < I) {
                    v1 = (int)lrintf(w[i+1] / sc);
                    v1 = v1 > 7 ? 7 : v1 < -8 ? -8 : v1;
                }
                row[i/2] = (uint8_t)((v0 + 8) | ((v1 + 8) << 4));
            }
            s[o] = sc;
        }
    }
}

/* Dense matmul dispatch: GPU-resident quantized copy first, quantized CPU
 * fallback second (VK device died mid-run), original f32 last (VK off). */
static void mm_dense(VkDense *d, const float *Wf, float *y, const float *x, int S, int I, int O) {
#ifdef COLI_VULKAN
    if (d->t && coli_vk_matmul(&d->t, y, x, d->q, d->s, d->fmt, S, I, O, d->gs)) return;
#endif
    if (d->q) {
        for (int s = 0; s < S; s++) {
            if (d->fmt == 6) matmul_q4g_asym(y + (int64_t)s*O, x + (int64_t)s*I, d->q, d->s, I, O, d->gs);
            else (d->fmt == 1 ? matmul_q8 : matmul_q4)(y + (int64_t)s*O, x + (int64_t)s*I, d->q, d->s, I, O);
        }
        return;
    }
    matmul(y, x, Wf, S, I, O);
}

/* Two dense matmuls sharing one input x: one submit via coli_vk_matmul_pair
 * when both live on the GPU, otherwise two mm_dense calls. */
static void mm_dense2(VkDense *d1, const float *W1, float *y1, int O1,
                      VkDense *d2, const float *W2, float *y2, int O2,
                      const float *x, int S, int I) {
#ifdef COLI_VULKAN
    if (d1->t && d2->t && d1->fmt == d2->fmt && d1->gs == d2->gs &&
        coli_vk_matmul_pair(&d1->t, y1, d1->q, d1->s, O1,
                            &d2->t, y2, d2->q, d2->s, O2, d1->fmt, x, S, I, d1->gs)) return;
#endif
    mm_dense(d1, W1, y1, x, S, I, O1);
    mm_dense(d2, W2, y2, x, S, I, O2);
}

/* zero-centered RMSNorm (Qwen3_5MoeRMSNorm): out = x * rsqrt(mean(x^2)+eps) * (1+w).
 * The engine is all-f32, so the "multiply in f32 before cast" subtlety is free. */
static void rmsnorm_zc(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * (1.f + w[i]);
}

static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ---------- config / weight loading ---------- */
static double req_num(jval *r, const char *k){
    jval *v=json_get(r,k);
    if(!v||v->t!=J_NUM){ fprintf(stderr,"config.json: missing or non-numeric \"%s\"\n",k); exit(1); }
    return v->num;
}
static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n<0 || n>(64L<<20)){ fprintf(stderr,"%s: bad size\n",path); exit(1); }
    char *buf = malloc((size_t)n+1); if(!buf){fprintf(stderr,"OOM cfg\n");exit(1);}
    if(fread(buf,1,(size_t)n,f)!=(size_t)n){ fprintf(stderr,"%s: short read\n",path); exit(1); } buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);
    c->hidden    = (int)req_num(r,"hidden_size");
    c->n_layers  = (int)req_num(r,"num_hidden_layers");
    c->vocab     = (int)req_num(r,"vocab_size");
    c->n_heads   = (int)req_num(r,"num_attention_heads");
    c->n_kv_heads= (int)req_num(r,"num_key_value_heads");
    c->head_dim  = (int)req_num(r,"head_dim");
    jval *prf = json_get(r,"partial_rotary_factor");
    c->rot       = (int)(c->head_dim * (prf && prf->t==J_NUM ? (float)prf->num : 0.25f));
    jval *th = json_get(r,"rope_theta");   c->theta = th ? (float)th->num : 1e7f;
    jval *ep = json_get(r,"rms_norm_eps"); c->eps   = ep ? (float)ep->num : 1e-6f;
    c->gdn_kh  = (int)req_num(r,"linear_num_key_heads");
    c->gdn_kd  = (int)req_num(r,"linear_key_head_dim");
    c->gdn_vh  = (int)req_num(r,"linear_num_value_heads");
    c->gdn_vd  = (int)req_num(r,"linear_value_head_dim");
    c->conv_k  = (int)req_num(r,"linear_conv_kernel_dim");
    c->key_dim   = c->gdn_kh * c->gdn_kd;
    c->value_dim = c->gdn_vh * c->gdn_vd;
    c->conv_dim  = 2*c->key_dim + c->value_dim;
    c->n_experts = (int)req_num(r,"num_experts");
    c->topk      = (int)req_num(r,"num_experts_per_tok");
    c->inter     = (int)req_num(r,"moe_intermediate_size");
    c->sh_inter  = (int)req_num(r,"shared_expert_intermediate_size");
    jval *eb = json_get(r,"expert_bits");
    c->expert_bits = eb && eb->t==J_NUM ? (int)eb->num : 4;
    if (c->expert_bits != 4 && c->expert_bits != 8) { fprintf(stderr,"expert_bits must be 4 or 8\n"); exit(1); }
    jval *gsz = json_get(r,"group_size"); c->group_size = gsz && gsz->t==J_NUM ? (int)gsz->num : 0;
    jval *ef = json_get(r,"expert_fmt");
    c->expert_fmt = ef && ef->t==J_NUM ? (int)ef->num : (c->expert_bits==4 ? 2 : 1);
    if (c->expert_fmt == 6 && (c->group_size < 8 || c->group_size % 8)) {
        fprintf(stderr,"expert_fmt=6 needs group_size multiple of 8\n"); exit(1); }
    if (c->expert_fmt != 1 && c->expert_fmt != 2 && c->expert_fmt != 6) {
        fprintf(stderr,"unsupported expert_fmt %d\n", c->expert_fmt); exit(1); }
    /* range checks: dims drive mallocs below */
    if(c->hidden<1||c->hidden>(1<<20)||c->n_layers<1||c->n_layers>4096||c->vocab<1||c->vocab>(1<<24)
       ||c->n_heads<1||c->n_kv_heads<1||c->n_heads%c->n_kv_heads||c->head_dim<2||c->rot<2||c->rot>c->head_dim
       ||c->gdn_kh<1||c->gdn_vh<1||c->gdn_vh%c->gdn_kh||c->conv_k<1||c->conv_k>16
       ||c->gdn_vh>64||c->gdn_vd>512   /* b/a[64], kv_mem/delta[512] stack buffers in gdn_step */
       ||c->n_experts<1||c->n_experts>(1<<20)||c->topk<1||c->topk>c->n_experts||c->topk>64
       ||c->inter<2||c->inter%2||c->sh_inter<1){
        fprintf(stderr,"config.json: dimension out of range\n"); exit(1); }
    jval *lt = json_get(r,"layer_types");
    c->is_full = calloc(c->n_layers, 1);
    if (lt && lt->t==J_ARR && lt->len==c->n_layers) {
        for (int i = 0; i < c->n_layers; i++)
            c->is_full[i] = lt->kids[i]->t==J_STR && !strcmp(lt->kids[i]->str,"full_attention");
    } else {
        for (int i = 0; i < c->n_layers; i++) c->is_full[i] = ((i+1)%4)==0;
    }
    free(buf); free(arena);
}

static float *load_t(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);
    return p;
}

/* Same format as GLM's telemetry.h usage_load/stats_dump_q: one "layer eid count"
 * line per expert. Load ADDS into memory; save rewrites totals -> accumulates. */
static int64_t usage_load(Model *m) {
    FILE *f = fopen(m->usage_path, "r"); if (!f) return 0;
    Cfg *c = &m->c; int l, e; uint32_t cnt; int64_t tot = 0;
    while (fscanf(f, "%d %d %u", &l, &e, &cnt) == 3)
        if (l >= 0 && l < c->n_layers && e >= 0 && e < c->n_experts) {
            m->eusage[(size_t)l * c->n_experts + e] += cnt; tot += cnt;
        }
    fclose(f); return tot;
}
static void usage_save(Model *m) {
    FILE *f = fopen(m->usage_path, "w"); if (!f) return;
    Cfg *c = &m->c;
    for (int l = 0; l < c->n_layers; l++)
        for (int e = 0; e < c->n_experts; e++) {
            uint32_t u = m->eusage[(size_t)l * c->n_experts + e];
            if (u) fprintf(f, "%d %d %u\n", l, e, u);
        }
    fclose(f);
}

static void model_init(Model *m, const char *snap, int cap) {
    memset(m, 0, sizeof(*m));
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    double t0 = now_s();
    m->embed      = load_t(m, "model.embed_tokens.weight");
    m->final_norm = load_t(m, "model.norm.weight");
    m->lm_head    = st_has(&m->S, "lm_head.weight") ? load_t(m, "lm_head.weight") : m->embed;  /* tied */
    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        #define LD(field, suffix) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t(m,nm)
        LD(in_ln,  "input_layernorm.weight");
        LD(post_ln,"post_attention_layernorm.weight");
        if (c->is_full[i]) {
            LD(q, "self_attn.q_proj.weight"); LD(k, "self_attn.k_proj.weight");
            LD(v, "self_attn.v_proj.weight"); LD(o, "self_attn.o_proj.weight");
            LD(qn,"self_attn.q_norm.weight"); LD(kn,"self_attn.k_norm.weight");
        } else {
            LD(qkv, "linear_attn.in_proj_qkv.weight");
            LD(z,   "linear_attn.in_proj_z.weight");
            LD(b,   "linear_attn.in_proj_b.weight");
            LD(a,   "linear_attn.in_proj_a.weight");
            LD(conv,"linear_attn.conv1d.weight");      /* [conv_dim, conv_k] */
            LD(dtb, "linear_attn.dt_bias");
            LD(alog,"linear_attn.A_log");
            LD(gnorm,"linear_attn.norm.weight");
            LD(outp,"linear_attn.out_proj.weight");
        }
        LD(router, "mlp.gate.weight");
        LD(sh_g, "mlp.shared_expert.gate_proj.weight");
        LD(sh_u, "mlp.shared_expert.up_proj.weight");
        LD(sh_d, "mlp.shared_expert.down_proj.weight");
        LD(sh_gate, "mlp.shared_expert_gate.weight");
        #undef LD
    }
    m->cache = calloc(c->n_layers, sizeof(LCache));
    for (int i = 0; i < c->n_layers; i++) {
        m->cache[i].cap = cap;
        m->cache[i].slots = calloc(cap, sizeof(Slot));
    }
#ifdef COLI_VULKAN
    m->vkreg = calloc((size_t)c->n_layers * c->n_experts * 3, sizeof(ColiVkTensor*));
#endif
    m->eusage = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint32_t));
    if (!m->eusage) { fprintf(stderr, "OOM eusage\n"); exit(1); }
    snprintf(m->usage_path, sizeof(m->usage_path), "%s/.coli_usage", snap);
    int64_t hist = usage_load(m);
    if (hist > 0) printf("[USAGE] expert history: %lld selections (%s)\n", (long long)hist, m->usage_path);
    m->dense_load_s = now_s() - t0;
}

/* ---------- routed-expert cache ---------- */
static void slot_alloc(Model *m, Slot *s) {
    if (s->g) return;
    Cfg *c = &m->c;
    int div = c->expert_bits == 4 ? 2 : 1;
    int64_t ng = (int64_t)c->inter * c->hidden / div;   /* gate/up rows bytes */
    int64_t nd = (int64_t)c->hidden * c->inter / div;
    uint8_t *w = malloc(ng + ng + nd);
    if (!w) { fprintf(stderr,"OOM expert slot\n"); exit(1); }
    s->g = w; s->u = w + ng; s->d = w + ng + ng;
    /* scale slab: per-row (fmt 1/2) = O floats; grouped-asym (fmt 6) = O*ceil(I/gs)*2
     * ([scale,zero] per group). gate/up: O=inter I=hidden; down: O=hidden I=inter. */
    int64_t sg, su, sd;
    if (c->expert_fmt == 6) {
        int gsz = c->group_size;
        int64_t ngh = (c->hidden + gsz - 1) / gsz, ngi = (c->inter + gsz - 1) / gsz;
        sg = su = (int64_t)c->inter * ngh * 2; sd = (int64_t)c->hidden * ngi * 2;
    } else { sg = su = c->inter; sd = c->hidden; }
    float *sc = falloc(sg + su + sd);
    s->gs = sc; s->us = sc + sg; s->ds = sc + sg + su;
}

static void expert_read(Model *m, int layer, int eid, Slot *s) {
    Cfg *c = &m->c;
    int div = c->expert_bits == 4 ? 2 : 1;
    static const char *pj[3] = {"gate_proj","up_proj","down_proj"};
    uint8_t *dst[3]; float *sd[3]; int64_t wb[3], sn[3];
    dst[0]=s->g; dst[1]=s->u; dst[2]=s->d;
    sd[0]=s->gs; sd[1]=s->us; sd[2]=s->ds;
    wb[0]=wb[1]=(int64_t)c->inter*c->hidden/div; wb[2]=(int64_t)c->hidden*c->inter/div;
    if (c->expert_fmt == 6) {
        int gsz = c->group_size;
        int64_t ngh = (c->hidden + gsz - 1) / gsz, ngi = (c->inter + gsz - 1) / gsz;
        sn[0]=sn[1]=(int64_t)c->inter*ngh*2; sn[2]=(int64_t)c->hidden*ngi*2;
    } else { sn[0]=sn[1]=c->inter; sn[2]=c->hidden; }
    char nm[256];
    for (int j = 0; j < 3; j++) {
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,pj[j]);
        st_tensor *tw = st_find(&m->S, nm);
        if (!tw || tw->nbytes != wb[j]) {
            fprintf(stderr,"%s: %lld bytes, expected %lld — wrong container?\n",
                    nm,(long long)(tw?tw->nbytes:-1),(long long)wb[j]); exit(1); }
        st_read_raw(&m->S, nm, dst[j], 1);
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.%s.weight.qs",layer,eid,pj[j]);
        st_tensor *ts = st_find(&m->S, nm);
        if (!ts || ts->numel != sn[j]) {
            fprintf(stderr,"%s: %lld elems, expected %lld\n",nm,(long long)(ts?ts->numel:-1),(long long)sn[j]); exit(1); }
        st_read_f32(&m->S, nm, sd[j], 1);
    }
}

static Slot *expert_get(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer];
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) {
        m->hits++; lc->slots[i].used = ++m->clock; return &lc->slots[i];
    }
    m->miss++;
    Slot *s;
    if (lc->n < lc->cap) { s = &lc->slots[lc->n++]; s->eid = -1; slot_alloc(m, s); }
    else {
        int lru = 0;
        for (int i = 1; i < lc->n; i++) if (lc->slots[i].used < lc->slots[lru].used) lru = i;
        s = &lc->slots[lru];
    }
    expert_read(m, layer, eid, s);
    s->eid = eid; s->used = ++m->clock;
    return s;
}

#ifdef COLI_VULKAN
/* Is this expert pinned on the VK tier? (GLM's vk_reg_served: skip its RAM
 * slot and disk I/O entirely — the tier serves it at decode.) */
static inline ColiVkTensor **vk_reg_at(Model *m, int layer, int eid) {
    return m->vkreg + ((size_t)layer * m->c.n_experts + eid) * 3;
}
static inline int vk_reg_served(Model *m, int layer, int eid) {
    if (!m->vk_on) return 0;
    ColiVkTensor **r = vk_reg_at(m, layer, eid);
    return r[0] && r[1] && r[2];
}
#endif

/* ---------- full attention (spec section C) ---------- */
static void rope_partial(float *x, int pos, const Cfg *c) {
    int hr = c->rot / 2;                       /* 32 */
    for (int j = 0; j < hr; j++) {
        float inv = powf(c->theta, -2.0f * j / c->rot);
        float ang = (float)pos * inv, cs = cosf(ang), sn = sinf(ang);
        float lo = x[j], hi = x[j+hr];
        x[j]    = lo*cs - hi*sn;               /* half-half NeoX rotation on the first rot dims */
        x[j+hr] = hi*cs + lo*sn;
    }
}

static void attention(Model *m, Layer *l, int kvi, const float *x, int S, int pos_base, float *out) {
    Cfg *c = &m->c;
    int H = c->n_heads, KH = c->n_kv_heads, hd = c->head_dim, D = c->hidden;
    int qo = H * hd * 2, ko = KH * hd;         /* q_proj emits [q|gate] per head */
#ifdef COLI_VULKAN
    /* Whole GQA block on the GPU in ONE submit: q/k/v matmul -> q/k-norm+rope+KV-write ->
     * attention -> o-proj, all on device (the device KV mirror becomes canonical). */
    if (m->vk_gqa_full_on && l->gq.t && l->gk.t && l->gv.t && l->go.t) {
        if (coli_vk_gqa_full(kvi, x, D, l->gq.t, l->gk.t, l->gv.t, l->go.t, out, S, H, KH, hd,
                             c->rot, pos_base, m->max_t, c->eps, c->theta, D))
            return;
        m->vk_gqa_full_on = 0;    /* device lost: drop to the staged path */
    }
#endif
    float *qg = falloc((int64_t)S*qo), *k = falloc((int64_t)S*ko), *v = falloc((int64_t)S*ko);
    mm_dense2(&l->gq, l->q, qg, qo, &l->gk, l->k, k, ko, x, S, D);   /* q+k: one submit */
    mm_dense(&l->gv, l->v, v, x, S, D, ko);
    for (int s = 0; s < S; s++) {
        int pos = pos_base + s;
        for (int h = 0; h < H; h++) {          /* per-head: zero-centered q_norm, then partial RoPE */
            float *qh = qg + (int64_t)s*qo + h*2*hd;      /* [q(hd) | gate(hd)] */
            rmsnorm_zc(qh, qh, l->qn, hd, c->eps);
            rope_partial(qh, pos, c);
        }
        for (int h = 0; h < KH; h++) {
            float *kh = k + (int64_t)s*ko + h*hd;
            rmsnorm_zc(kh, kh, l->kn, hd, c->eps);
            rope_partial(kh, pos, c);
            memcpy(m->K[kvi] + ((int64_t)h*m->max_t + pos)*hd, kh, hd*sizeof(float));
            memcpy(m->V[kvi] + ((int64_t)h*m->max_t + pos)*hd, v + (int64_t)s*ko + h*hd, hd*sizeof(float));
#ifdef COLI_VULKAN
            if (m->vk_gqa_on)                  /* mirror normed+roped k / raw v to the device cache */
                coli_vk_kv_row(kvi, h*m->max_t + pos, kh, v + (int64_t)s*ko + h*hd);
#endif
        }
    }
    float scale = 1.f / sqrtf((float)hd);
#ifdef COLI_VULKAN
    /* score/softmax/context on the iGPU (host K/V stays canonical, so a device
     * loss just drops the flag and the CPU loop below serves the rest). */
    if (m->vk_gqa_on) {
        /* fused GQA + o-projection: ctx stays on device, one submit (o_proj resident) */
        if (l->go.t && coli_vk_gqa_project(kvi, qg, l->go.t, out, S, H, KH, hd,
                                           m->max_t, 0, pos_base+S, scale, D)) {
            free(qg); free(k); free(v);
            return;
        }
        float *ctxg = falloc((int64_t)S*H*hd);   /* fallback: separate attention + o-proj */
        if (coli_vk_gqa(kvi, qg, ctxg, S, H, KH, hd, m->max_t, 0, pos_base+S, scale)) {
            mm_dense(&l->go, l->o, out, ctxg, S, H*hd, D);
            free(qg); free(k); free(v); free(ctxg);
            return;
        }
        free(ctxg);
        m->vk_gqa_on = 0;
    }
#endif
    int gq = H / KH;                           /* GQA: q head h -> kv head h/gq (block mapping) */
    float *ctx = falloc((int64_t)S*H*hd);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < H; h++) {
        for (int s = 0; s < S; s++) {
            float scl[8192]; float *scp = (pos_base+S) <= 8192 ? scl : falloc(pos_base+S);
            int qpos = pos_base + s, kh = h / gq;
            const float *qv = qg + (int64_t)s*qo + h*2*hd;
            for (int t = 0; t <= qpos; t++) {
                const float *kv = m->K[kvi] + ((int64_t)kh*m->max_t + t)*hd;
                float acc = 0; for (int d = 0; d < hd; d++) acc += qv[d]*kv[d];
                scp[t] = acc * scale;
            }
            softmax_row(scp, qpos+1);
            float *cx = ctx + (int64_t)s*H*hd + h*hd;
            for (int d = 0; d < hd; d++) cx[d] = 0;
            for (int t = 0; t <= qpos; t++) {
                const float *vr = m->V[kvi] + ((int64_t)kh*m->max_t + t)*hd;
                float a = scp[t];
                for (int d = 0; d < hd; d++) cx[d] += a * vr[d];
            }
            const float *gate = qv + hd;       /* output gate: attn *= sigmoid(gate), pre-o_proj */
            for (int d = 0; d < hd; d++) cx[d] *= sigmoidf_(gate[d]);
            if (scp != scl) free(scp);
        }
    }
    mm_dense(&l->go, l->o, out, ctx, S, H*hd, D);
    free(qg); free(k); free(v); free(ctx);
}

/* ---------- GatedDeltaNet single-token recurrence (spec section D) ---------- */
static void gdn_step(Model *m, Layer *l, int gi, const float *x, float *out) {
    Cfg *c = &m->c;
    int CD = c->conv_dim, K = c->conv_k;
    int KH = c->gdn_kh, KD = c->gdn_kd, VH = c->gdn_vh, VD = c->gdn_vd;
#ifdef COLI_VULKAN
    /* Whole GDN block on the GPU in ONE submit: only the tiny b/a projections (decay,
     * beta, from x) stay on the CPU; conv1d, q/k-norm, delta rule and out-proj all run
     * on device with qkv/z/cv/y never touching the host. */
    if (m->vk_gdn_full_on && l->gqkv.t && l->gz.t && l->gout.t) {
        float b[64], a[64];
        matmul(b, x, l->b, 1, c->hidden, VH);
        matmul(a, x, l->a, 1, c->hidden, VH);
        float *params = falloc(2*VH + VD);        /* [decay(VH) | beta(VH) | gnorm(VD)] */
        for (int h = 0; h < VH; h++) {
            params[h]      = expf(-expf(l->alog[h]) * softplusf_(a[h] + l->dtb[h]));  /* decay */
            params[VH + h] = sigmoidf_(b[h]);                                          /* beta  */
        }
        memcpy(params + 2*VH, l->gnorm, VD*sizeof(float));
        int ok = coli_vk_gdn_full(gi, x, c->hidden, l->gqkv.t, l->gz.t, params, l->gout.t, out,
                                  KH, KD, VH, VD, CD, K, c->eps, c->hidden);
        free(params);
        if (ok) return;
        m->vk_gdn_full_on = 0;    /* device lost: drop to the staged path below */
    }
#endif
    float *qkv = falloc(CD), *cv = falloc(CD);
    float *z = falloc(c->value_dim);
    mm_dense2(&l->gqkv, l->qkv, qkv, CD, &l->gz, l->z, z, c->value_dim, x, 1, c->hidden);  /* one submit */
    /* causal conv1d update: shift ring, append, convolve (no bias, silu) */
    float *st = m->convst[gi];
    memmove(st, st + CD, (size_t)(K-1)*CD*sizeof(float));
    memcpy(st + (size_t)(K-1)*CD, qkv, CD*sizeof(float));
    #pragma omp parallel for schedule(static)
    for (int ch = 0; ch < CD; ch++) {
        const float *w = l->conv + (int64_t)ch*K;
        float acc = 0;
        for (int j = 0; j < K; j++) acc += w[j] * st[(size_t)j*CD + ch];
        cv[ch] = siluf_(acc);
    }
    free(qkv);
    float b[64], a[64];
    matmul(b, x, l->b, 1, c->hidden, VH);   /* tiny [VH,D] projections: stay f32/CPU */
    matmul(a, x, l->a, 1, c->hidden, VH);
    /* normalized q/k per source (k) head: l2norm (eps on the SUM), then q /= sqrt(KD) */
    float *qn = falloc(KH*KD), *kn = falloc(KH*KD);
    for (int h = 0; h < KH; h++) {
        const float *qs = cv + h*KD, *ks = cv + c->key_dim + h*KD;
        float sq = 0, sk = 0;
        for (int d = 0; d < KD; d++) { sq += qs[d]*qs[d]; sk += ks[d]*ks[d]; }
        float rq = 1.f/sqrtf(sq + c->eps) / sqrtf((float)KD), rk = 1.f/sqrtf(sk + c->eps);
        for (int d = 0; d < KD; d++) { qn[h*KD+d] = qs[d]*rq; kn[h*KD+d] = ks[d]*rk; }
    }
    float *y = falloc(VH*VD);
    /* per-v-head decay + beta (needed by both the GPU and CPU recurrence paths) */
    float decv[64], betv[64];
    for (int h = 0; h < VH; h++) {
        betv[h] = sigmoidf_(b[h]);
        decv[h] = expf(-expf(l->alog[h]) * softplusf_(a[h] + l->dtb[h]));
    }
#ifdef COLI_VULKAN
    /* delta rule + gated RMSNorm on the iGPU against the device-resident state.
     * All-or-nothing per layer (state lives on-device across the whole sequence);
     * a failure means the device was lost, so we drop the flag and stay on CPU. */
    if (m->vk_gdn_on) {
        /* fused delta-rule + out-projection: y stays on device, one submit (out_proj resident) */
        if (l->gout.t && coli_vk_gdn_project(gi, qn, kn, cv + 2*c->key_dim, z, decv, betv, l->gnorm,
                                             l->gout.t, out, KH, KD, VH, VD, c->eps, c->hidden)) {
            free(cv); free(z); free(qn); free(kn); free(y);
            return;
        }
        if (coli_vk_gdn(gi, qn, kn, cv + 2*c->key_dim, z, decv, betv, l->gnorm, y,
                        KH, KD, VH, VD, c->eps)) {   /* fallback: separate recurrence + out-proj */
            mm_dense(&l->gout, l->outp, out, y, 1, c->value_dim, c->hidden);
            free(cv); free(z); free(qn); free(kn); free(y);
            return;
        }
        m->vk_gdn_on = 0;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < VH; h++) {             /* delta rule, all f32; v-head h uses k-head h/2 */
        int src = h / (VH / KH);
        const float *qh = qn + src*KD, *kh = kn + src*KD, *vh = cv + 2*c->key_dim + h*VD;
        float beta = betv[h], decay = decv[h];
        float *S = m->gdnS[gi] + (size_t)h*KD*VD;      /* [KD][VD] row-major */
        float kv_mem[512];                             /* VD <= 512 */
        for (int vd = 0; vd < VD; vd++) kv_mem[vd] = 0;
        for (int kd = 0; kd < KD; kd++) {
            float *row = S + (size_t)kd*VD; float kk = kh[kd];
            for (int vd = 0; vd < VD; vd++) { row[vd] *= decay; kv_mem[vd] += row[vd]*kk; }
        }
        float delta[512];
        for (int vd = 0; vd < VD; vd++) delta[vd] = (vh[vd] - kv_mem[vd]) * beta;
        float *yh = y + (size_t)h*VD;
        for (int vd = 0; vd < VD; vd++) yh[vd] = 0;
        for (int kd = 0; kd < KD; kd++) {
            float *row = S + (size_t)kd*VD; float kk = kh[kd], qq = qh[kd];
            for (int vd = 0; vd < VD; vd++) { row[vd] += kk*delta[vd]; yh[vd] += row[vd]*qq; }
        }
        /* gated RMSNorm (plain weight, silu(z) gate) per v-head */
        const float *zh = z + (size_t)h*VD;
        double ms = 0; for (int vd = 0; vd < VD; vd++) ms += (double)yh[vd]*yh[vd];
        float r = 1.f / sqrtf((float)(ms/VD) + c->eps);
        for (int vd = 0; vd < VD; vd++) yh[vd] = yh[vd]*r*l->gnorm[vd] * siluf_(zh[vd]);
    }
    mm_dense(&l->gout, l->outp, out, y, 1, c->value_dim, c->hidden);
    free(cv); free(z); free(qn); free(kn); free(y);
}

/* One expert's fused MLP on CPU (fmt-aware): hidden = silu(gate(x))*up(x), y = down(hidden).
 * Covers int8 (fmt 1), per-row int4 (fmt 2), grouped-asym int4 (fmt 6 / GPTQ). */
static void expert_mlp_cpu(Cfg *c, Slot *s, const float *x, float *g, float *u, float *y,
                           int D, int I) {
    if (c->expert_fmt == 6) {
        int gs = c->group_size;
        matmul_q4g_asym(g, x, s->g, s->gs, D, I, gs); matmul_q4g_asym(u, x, s->u, s->us, D, I, gs);
        for (int i = 0; i < I; i++) g[i] = siluf_(g[i]) * u[i];
        matmul_q4g_asym(y, g, s->d, s->ds, I, D, gs);
    } else if (c->expert_fmt == 2) {
        matmul_q4(g, x, s->g, s->gs, D, I); matmul_q4(u, x, s->u, s->us, D, I);
        for (int i = 0; i < I; i++) g[i] = siluf_(g[i]) * u[i];
        matmul_q4(y, g, s->d, s->ds, I, D);
    } else {
        matmul_q8(g, x, s->g, s->gs, D, I); matmul_q8(u, x, s->u, s->us, D, I);
        for (int i = 0; i < I; i++) g[i] = siluf_(g[i]) * u[i];
        matmul_q8(y, g, s->d, s->ds, I, D);
    }
}

#ifdef COLI_VULKAN
static long g_vk_srv = 0, g_vk_unsrv = 0;   /* debug: routed experts served by VRAM tier vs not */
#endif
/* ---------- MoE (spec section E): routed top-8 + sigmoid-gated shared expert ---------- */
/* Softmax + top-k + expert apply. `pr` is the raw router logits [E] (mutated in place).
 * Used by moe_token and by the fused attn/GDN+route path that already produced pr on GPU. */
static void moe_from_logits(Model *m, Layer *l, int layer, const float *x, float *pr, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, TK = c->topk, I = c->inter;
    softmax_row(pr, E);                        /* softmax BEFORE topk, full width */
    int idx[64]; float val[64];
    for (int kk = 0; kk < TK; kk++) {
        int best = -1; float bv = -1.f;
        for (int e = 0; e < E; e++) {
            int taken = 0; for (int j = 0; j < kk; j++) if (idx[j]==e){taken=1;break;}
            if (!taken && pr[e] > bv) { bv = pr[e]; best = e; }
        }
        idx[kk] = best; val[kk] = bv;
    }
    { float sm=0; for(int kk=0;kk<TK;kk++) sm+=val[kk]; for(int kk=0;kk<TK;kk++) val[kk]/=sm; }  /* unconditional renorm */
    for (int d = 0; d < D; d++) out[d] = 0;
    for (int kk = 0; kk < TK; kk++) m->eusage[(size_t)layer * E + idx[kk]]++;  /* the cache that learns */

    /* Tier-served experts skip the RAM cache AND the container read (GLM's
     * vk_reg_served): the pinned VRAM copy is the only copy we need. */
    Slot *sl[64];
    for (int kk = 0; kk < TK; kk++)
#ifdef COLI_VULKAN
        if (vk_reg_served(m, layer, idx[kk])) { sl[kk] = NULL; g_vk_srv++; }
        else { sl[kk] = expert_get(m, layer, idx[kk]); g_vk_unsrv++; }
#else
        sl[kk] = expert_get(m, layer, idx[kk]);
#endif

    /* shared-expert scalar gate: needed by both the GPU and CPU paths */
    float shgt = 0; for (int d = 0; d < D; d++) shgt += l->sh_gate[d] * x[d];
    shgt = sigmoidf_(shgt);
    int sh_done = 0;

#ifdef COLI_VULKAN
    /* ONE expert_group submit: every tier-pinned routed expert PLUS the shared
     * expert (same shape/fmt, gm = -1) — the shared expert rides the batch for
     * free instead of costing three dense f32 matmuls on the CPU. */
    if (m->vk_on) {
        ColiVkTensor *vg[64], *vu[64], *vd[64]; int rows[64], gm[64], ng = 0;
        for (int kk = 0; kk < TK; kk++) {
            if (sl[kk]) continue;
            ColiVkTensor **r = vk_reg_at(m, layer, idx[kk]);
            vg[ng]=r[0]; vu[ng]=r[1]; vd[ng]=r[2]; rows[ng]=1; gm[ng]=kk; ng++;
        }
        /* Fold the shared expert into the routed group ONLY if it shares their quant
         * fmt — the fused gate_up shader decodes the whole batch with one fmt, so a
         * fmt-2 shared expert mixed with fmt-6 routed experts would reject the entire
         * group (dropping every expert to CPU). When they differ, leave the shared
         * expert to the separate mm_dense GPU path below (sh_done stays 0). */
        if (l->gsg.t && l->gsu.t && l->gsd.t && c->sh_inter == I && ng < 64 &&
            l->gsg.fmt == c->expert_fmt) {
            vg[ng]=l->gsg.t; vu[ng]=l->gsu.t; vd[ng]=l->gsd.t; rows[ng]=1; gm[ng]=-1; ng++;
        }
        if (ng > 0) {
            float *xg = falloc((int64_t)ng*D), *yg = falloc((int64_t)ng*D);
            for (int j = 0; j < ng; j++) memcpy(xg + (int64_t)j*D, x, D*sizeof(float));
            int issued = coli_vk_expert_group_issue(vg, vu, vd, rows, ng, xg);
            /* CPU share */
            float *g = falloc(I), *u = falloc(I), *hh = falloc(D);
            for (int kk = 0; kk < TK; kk++) {
                int on_gpu = 0;
                if (issued) for (int j = 0; j < ng; j++) if (gm[j]==kk) { on_gpu=1; break; }
                if (on_gpu) continue;
                /* !issued can leave tier-served experts without a RAM slot: fetch now */
                Slot *s = sl[kk] ? sl[kk] : expert_get(m, layer, idx[kk]);
                expert_mlp_cpu(c, s, x, g, u, hh, D, I);
                for (int d = 0; d < D; d++) out[d] += val[kk] * hh[d];
            }
            free(g); free(u); free(hh);
            if (issued && coli_vk_expert_group_take(yg)) {
                for (int j = 0; j < ng; j++) {
                    float w = gm[j] < 0 ? shgt : val[gm[j]];   /* shared entry: sigmoid gate */
                    const float *yr = yg + (int64_t)j*D;
                    for (int d = 0; d < D; d++) out[d] += w * yr[d];
                    if (gm[j] < 0) sh_done = 1;
                }
            } else if (issued) {
                /* group failed mid-flight: recompute the GPU share on CPU
                 * (the !issued case is already covered by the loop above,
                 * which skipped nothing when on_gpu never became 1) */
                fprintf(stderr,"[VK] expert group failed mid-flight — recomputing on CPU\n");
                float *g2 = falloc(I), *u2 = falloc(I), *h2 = falloc(D);
                for (int j = 0; j < ng; j++) {
                    if (gm[j] < 0) continue;   /* shared: recomputed below (sh_done stays 0) */
                    Slot *s = sl[gm[j]] ? sl[gm[j]] : expert_get(m, layer, idx[gm[j]]);
                    expert_mlp_cpu(c, s, x, g2, u2, h2, D, I);
                    for (int d = 0; d < D; d++) out[d] += val[gm[j]] * h2[d];
                }
                free(g2); free(u2); free(h2);
            }
            free(xg); free(yg);
            goto shared;
        }
    }
#endif
    {   /* pure CPU path */
        float *g = falloc(I), *u = falloc(I), *hh = falloc(D);
        for (int kk = 0; kk < TK; kk++) {
            Slot *s = sl[kk];
            expert_mlp_cpu(c, s, x, g, u, hh, D, I);
            for (int d = 0; d < D; d++) out[d] += val[kk] * hh[d];
        }
        free(g); free(u); free(hh);
    }
#ifdef COLI_VULKAN
shared:
#endif
    if (!sh_done) {   /* shared expert + scalar sigmoid gate (f32, or quantized fallback) */
        int SI = c->sh_inter;
        float *g = falloc(SI), *u = falloc(SI), *sh = falloc(D);
        mm_dense2(&l->gsg, l->sh_g, g, SI, &l->gsu, l->sh_u, u, SI, x, 1, D);
        for (int i = 0; i < SI; i++) g[i] = siluf_(g[i]) * u[i];
        mm_dense(&l->gsd, l->sh_d, sh, g, 1, SI, D);
        for (int d = 0; d < D; d++) out[d] += shgt * sh[d];
        free(g); free(u); free(sh);
    }
}

static void moe_token(Model *m, Layer *l, int layer, const float *x, float *out) {
    Cfg *c = &m->c;
    float *pr = falloc(c->n_experts);
    mm_dense(&l->grouter, l->router, pr, x, 1, c->hidden, c->n_experts);
    moe_from_logits(m, l, layer, x, pr, out);
    free(pr);
}

#ifdef COLI_VULKAN
/* Stream-path MoE: idx/val already come from device softmax+topk on the route
 * submit. expert_group reads device nrm, folds weighted outputs into the device
 * residual, and leaves a GPU semaphore for the next layer's route. */
static void moe_from_topk_pipe(Model *m, Layer *l, int layer, const int *idx, const float *val) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, TK = c->topk, I = c->inter;
    const float *x = coli_vk_stream_nrm();
    if (!x) return;
    for (int kk = 0; kk < TK; kk++) m->eusage[(size_t)layer * E + idx[kk]]++;

    Slot *sl[64];
    for (int kk = 0; kk < TK; kk++)
        if (vk_reg_served(m, layer, idx[kk])) { sl[kk] = NULL; g_vk_srv++; }
        else { sl[kk] = expert_get(m, layer, idx[kk]); g_vk_unsrv++; }

    float shgt = 0; for (int d = 0; d < D; d++) shgt += l->sh_gate[d] * x[d];
    shgt = sigmoidf_(shgt);
    int sh_done = 0;

    ColiVkTensor *vg[64], *vu[64], *vd[64]; int rows[64], gm[64], ng = 0;
    float wts[64];
    int sh_on_gpu = 0;
    for (int kk = 0; kk < TK; kk++) {
        if (sl[kk]) continue;
        ColiVkTensor **r = vk_reg_at(m, layer, idx[kk]);
        vg[ng]=r[0]; vu[ng]=r[1]; vd[ng]=r[2]; rows[ng]=1; gm[ng]=kk; wts[ng]=val[kk]; ng++;
    }
    if (l->gsg.t && l->gsu.t && l->gsd.t && c->sh_inter == I && ng < 64 &&
        l->gsg.fmt == c->expert_fmt) {
        vg[ng]=l->gsg.t; vu[ng]=l->gsu.t; vd[ng]=l->gsd.t; rows[ng]=1; gm[ng]=-1; wts[ng]=shgt;
        ng++; sh_on_gpu = 1;
    }

    int need_cpu = !sh_on_gpu;
    for (int kk = 0; kk < TK; kk++) if (sl[kk]) need_cpu = 1;

    int issued = 0;
    if (m->vk_on && ng > 0)
        issued = coli_vk_expert_group_issue_pipe(vg, vu, vd, rows, wts, ng, D);
    if (issued && sh_on_gpu) sh_done = 1;

    if (need_cpu || !issued) {
        float *cpu_y = fcalloc(D);
        float *g = falloc(I), *u = falloc(I), *hh = falloc(D);
        for (int kk = 0; kk < TK; kk++) {
            int on_gpu = 0;
            if (issued) for (int j = 0; j < ng; j++) if (gm[j]==kk) { on_gpu=1; break; }
            if (on_gpu) continue;
            Slot *s = sl[kk] ? sl[kk] : expert_get(m, layer, idx[kk]);
            expert_mlp_cpu(c, s, x, g, u, hh, D, I);
            for (int d = 0; d < D; d++) cpu_y[d] += val[kk] * hh[d];
        }
        free(g); free(u); free(hh);
        if (!sh_done) {
            int SI = c->sh_inter;
            float *g2 = falloc(SI), *u2 = falloc(SI), *sh = falloc(D);
            mm_dense2(&l->gsg, l->sh_g, g2, SI, &l->gsu, l->sh_u, u2, SI, x, 1, D);
            for (int i = 0; i < SI; i++) g2[i] = siluf_(g2[i]) * u2[i];
            mm_dense(&l->gsd, l->sh_d, sh, g2, 1, SI, D);
            for (int d = 0; d < D; d++) cpu_y[d] += shgt * sh[d];
            free(g2); free(u2); free(sh);
        }
        if (issued) coli_vk_expert_group_take_pipe();
        coli_vk_stream_add(cpu_y, D);
        free(cpu_y);
    }
    /* else: all experts on GPU — leave eg in flight; next route waits on eg_sem */
}
#endif

/* ---------- MoE, prefill form: one expert_group entry per DISTINCT expert ----------
 * moe_token gives every (token, expert) pair its own rows=1 entry, so prefilling S tokens
 * re-streams an expert's weights once per token that picked it. Below, all S tokens are
 * routed first and then grouped BY EXPERT, so each expert's weights are read once with
 * every token that wants them riding along as rows. That also lifts the projections out
 * of the mat-vec regime (rows>1) where they are purely bandwidth-bound. Experts the VRAM
 * tier does not serve go to the CPU, batched the same way: one container/LRU fetch, then
 * all of that expert's tokens. */

/* The routing plan shared by moe_batch's GPU and CPU halves: the tokens that picked
 * expert e are epair[ecnt[e] .. ecnt[e+1]), each entry an index into tidx/tval. */
typedef struct {
    Model *m; Layer *l; int layer, S, D, I, TK;
    const float *xs; float *out;
    const int *ecnt, *epair; const float *tval;
} MoeB;

static inline int moeb_rows(const MoeB *b, int e) { return b->ecnt[e+1] - b->ecnt[e]; }

/* Token index and routing weight of expert e's r-th row. */
static inline void moeb_row(const MoeB *b, int e, int r, int *tok, float *w) {
    int i = b->epair[b->ecnt[e] + r];
    *tok = i / b->TK; *w = b->tval[i];
}

static inline void moeb_add(const MoeB *b, int tok, float w, const float *y) {
    float *o = b->out + (int64_t)tok * b->D;
    for (int d = 0; d < b->D; d++) o[d] += w * y[d];
}

/* Expert e's rows [r0, r0+nr) on the CPU: fetch its weights once, then apply them to
 * every token that routed to it. */
static void moeb_cpu(const MoeB *b, int e, int r0, int nr) {
    Slot *s = expert_get(b->m, b->layer, e);
    float *g = falloc(b->I), *u = falloc(b->I), *hh = falloc(b->D);
    for (int r = 0; r < nr; r++) {
        int tok; float w; moeb_row(b, e, r0 + r, &tok, &w);
        expert_mlp_cpu(&b->m->c, s, b->xs + (int64_t)tok * b->D, g, u, hh, b->D, b->I);
        moeb_add(b, tok, w, hh);
    }
    free(g); free(u); free(hh);
}

#define MOEB_MAXE    64     /* coli_vk_expert_group caps one submit at 64 experts */
#define MOEB_MAXROWS 1024   /* and this caps its packed x/hidden/y scratch (~18 MB at D=2048) */

/* Prefill MoE given raw router logits [S,E] (mutated in-place by softmax). Groups
 * tokens by expert so each weight stream is read once with rows>1 — also used by the
 * route-fuse path which already computed pr on device. */
static void moe_batch_from_logits(Model *m, Layer *l, int layer, const float *xs,
                                  float *pr, int S, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, TK = c->topk, I = c->inter;

    int *tidx = ialloc((int64_t)S * TK);
    float *tval = falloc((int64_t)S * TK);
    for (int s = 0; s < S; s++) {
        float *p = pr + (int64_t)s * E;
        softmax_row(p, E);
        int *id = tidx + s*TK; float *vl = tval + s*TK;
        for (int kk = 0; kk < TK; kk++) {
            int best = -1; float bv = -1.f;
            for (int e = 0; e < E; e++) {
                int taken = 0; for (int j = 0; j < kk; j++) if (id[j] == e) { taken = 1; break; }
                if (!taken && p[e] > bv) { bv = p[e]; best = e; }
            }
            id[kk] = best; vl[kk] = bv;
        }
        float sm = 0; for (int kk = 0; kk < TK; kk++) sm += vl[kk];
        for (int kk = 0; kk < TK; kk++) vl[kk] /= sm;
        for (int kk = 0; kk < TK; kk++) m->eusage[(size_t)layer * E + id[kk]]++;
    }
    memset(out, 0, (size_t)S * D * sizeof(float));

    /* bucket the (token, slot) pairs by expert: counts -> offsets -> flat pair list */
    int *ecnt = ialloc(E + 1), *ecur = ialloc(E), *epair = ialloc((int64_t)S * TK);
    memset(ecnt, 0, (size_t)(E + 1) * sizeof(int));
    memset(ecur, 0, (size_t)E * sizeof(int));
    for (int64_t i = 0; i < (int64_t)S * TK; i++) ecnt[tidx[i]]++;
    int acc = 0;
    for (int e = 0; e < E; e++) { int n = ecnt[e]; ecnt[e] = acc; acc += n; }
    ecnt[E] = acc;
    for (int64_t i = 0; i < (int64_t)S * TK; i++) { int e = tidx[i]; epair[ecnt[e] + ecur[e]++] = (int)i; }

    MoeB b = {m, l, layer, S, D, I, TK, xs, out, ecnt, epair, tval};

    /* split the routed experts by where their weights live */
    int *gpu = ialloc(E), *cpu = ialloc(E), ngpu = 0, ncpu = 0;
    for (int e = 0; e < E; e++) {
        int nr = moeb_rows(&b, e);
        if (!nr) continue;
#ifdef COLI_VULKAN
        if (vk_reg_served(m, layer, e)) { gpu[ngpu++] = e; g_vk_srv += nr; continue; }
        g_vk_unsrv += nr;
#endif
        cpu[ncpu++] = e;
    }

    int cpu_i = 0;
#ifdef COLI_VULKAN
    /* GPU chunks, each one submit: pack the chunk's rows, issue, spend the flight on the
     * CPU experts (moe_token's issue -> CPU share -> take, spread over the chunks), join. */
    int nchunk = (ngpu + MOEB_MAXE - 1) / MOEB_MAXE, per = nchunk ? (ncpu + nchunk - 1) / nchunk : 0;
    for (int gi = 0, roff = 0; gi < ngpu; ) {
        ColiVkTensor *vg[MOEB_MAXE], *vu[MOEB_MAXE], *vd[MOEB_MAXE];
        int rows[MOEB_MAXE], jid[MOEB_MAXE], jbeg[MOEB_MAXE], ng = 0, total = 0;
        while (gi < ngpu && ng < MOEB_MAXE && total < MOEB_MAXROWS) {
            int e = gpu[gi], have = moeb_rows(&b, e) - roff;      /* an expert with more rows
                                                                  * than fit is split here and
                                                                  * resumed in the next chunk */
            int nr = have < MOEB_MAXROWS - total ? have : MOEB_MAXROWS - total;
            ColiVkTensor **r = vk_reg_at(m, layer, e);
            vg[ng] = r[0]; vu[ng] = r[1]; vd[ng] = r[2];
            rows[ng] = nr; jid[ng] = e; jbeg[ng] = roff; ng++; total += nr;
            roff += nr;
            if (roff == moeb_rows(&b, e)) { gi++; roff = 0; }
        }
        float *xg = falloc((int64_t)total*D), *yg = falloc((int64_t)total*D);
        int *rtok = ialloc(total); float *rw = falloc(total);
        for (int j = 0, p = 0; j < ng; j++)
            for (int r = 0; r < rows[j]; r++, p++) {
                moeb_row(&b, jid[j], jbeg[j] + r, &rtok[p], &rw[p]);
                memcpy(xg + (int64_t)p*D, xs + (int64_t)rtok[p]*D, D*sizeof(float));
            }
        int issued = coli_vk_expert_group_issue(vg, vu, vd, rows, ng, xg);
        for (int k = 0; k < per && cpu_i < ncpu; k++, cpu_i++)
            moeb_cpu(&b, cpu[cpu_i], 0, moeb_rows(&b, cpu[cpu_i]));
        if (issued && coli_vk_expert_group_take(yg)) {
            for (int r = 0; r < total; r++) moeb_add(&b, rtok[r], rw[r], yg + (int64_t)r*D);
        } else {
            if (issued) fprintf(stderr, "[VK] expert group failed mid-flight — recomputing on CPU\n");
            for (int j = 0; j < ng; j++) moeb_cpu(&b, jid[j], jbeg[j], rows[j]);
        }
        free(xg); free(yg); free(rtok); free(rw);
    }
#endif
    for (; cpu_i < ncpu; cpu_i++) moeb_cpu(&b, cpu[cpu_i], 0, moeb_rows(&b, cpu[cpu_i]));

    /* shared expert over all S rows at once: three submits for the whole batch instead
     * of three per token, and its own weights stream once */
    {
        int SI = c->sh_inter;
        float *g = falloc((int64_t)S*SI), *u = falloc((int64_t)S*SI), *sh = falloc((int64_t)S*D);
        mm_dense2(&l->gsg, l->sh_g, g, SI, &l->gsu, l->sh_u, u, SI, xs, S, D);
        for (int64_t i = 0; i < (int64_t)S*SI; i++) g[i] = siluf_(g[i]) * u[i];
        mm_dense(&l->gsd, l->sh_d, sh, g, S, SI, D);
        for (int s = 0; s < S; s++) {
            const float *x = xs + (int64_t)s*D; float gt = 0;
            for (int d = 0; d < D; d++) gt += l->sh_gate[d] * x[d];
            moeb_add(&b, s, sigmoidf_(gt), sh + (int64_t)s*D);
        }
        free(g); free(u); free(sh);
    }
    free(tidx); free(tval); free(ecnt); free(ecur); free(epair); free(gpu); free(cpu);
}

static void moe_batch(Model *m, Layer *l, int layer, const float *xs, int S, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts;
    float *pr = falloc((int64_t)S * E);
    mm_dense(&l->grouter, l->router, pr, xs, S, D, E);
    moe_batch_from_logits(m, l, layer, xs, pr, S, out);
    free(pr);
}

/* ---------- one forward step over S new tokens ---------- */
static FILE *g_logits_f = NULL;
static void dump_logits(Model *m, const float *logit) {
    if (g_logits_f) fwrite(logit, sizeof(float), m->c.vocab, g_logits_f);
}

#ifdef COLI_VULKAN
/* Grow persistent decode scratch to at least `need` floats per buffer. */
static void scratch_ensure(Model *m, int need) {
    if (m->scratch_cap >= need) return;
    free(m->scratch_x); free(m->scratch_nrm); free(m->scratch_tmp); free(m->scratch_last);
    m->scratch_x = falloc(need); m->scratch_nrm = falloc(need);
    m->scratch_tmp = falloc(need); m->scratch_last = falloc(need);
    m->scratch_cap = need;
}
#endif

/* Runs the decoder stack over S new tokens and leaves the final-normed hidden state of
 * the LAST token in last[hidden] — the shared body of step() and step_top1().
 * Stream+greedy path may leave resid on device (m->vk_stream_argmax) and skip filling last. */
static void decode_stack(Model *m, const int *ids, int S, int pos_base, float *last) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts;
#ifdef COLI_VULKAN
    m->vk_stream_argmax = 0;
    float *x, *nrm, *tmp;
    if (S == 1) {
        scratch_ensure(m, D);
        x = m->scratch_x; nrm = m->scratch_nrm; tmp = m->scratch_tmp;
    } else {
        x = falloc((int64_t)S*D); nrm = falloc((int64_t)S*D); tmp = falloc((int64_t)S*D);
    }
#else
    float *x = falloc((int64_t)S*D);
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
#endif
    for (int s = 0; s < S; s++) memcpy(x + (int64_t)s*D, m->embed + (int64_t)ids[s]*D, D*sizeof(float));
    int kvi = 0, gi = 0;
#ifdef COLI_VULKAN
    /* Decode stream path (S=1): resid stays on device; only pr[E] comes back for top-k. */
    int use_stream = m->vk_stream_on && S == 1 && coli_vk_stream_begin(x, D);
    if (use_stream) {
        for (int i = 0; i < c->n_layers; i++) {
            Layer *l = &m->L[i];
            int idx[64]; float val[64];
            int ok = 0;
            if (c->is_full[i] && m->vk_gqa_full_on && l->gq.t && l->gk.t && l->gv.t && l->go.t && l->grouter.t) {
                ok = coli_vk_gqa_full_route_pipe(kvi, D, l->gq.t, l->gk.t, l->gv.t, l->go.t,
                                                 l->grouter.t, i, idx, val, c->topk,
                                                 c->n_heads, c->n_kv_heads,
                                                 c->head_dim, c->rot, pos_base, m->max_t, c->eps, c->theta,
                                                 D, E);
                if (ok) kvi++;
                else m->vk_gqa_full_on = 0;
            } else if (!c->is_full[i] && m->vk_gdn_full_on && l->gqkv.t && l->gz.t && l->gout.t &&
                       l->gba.t && l->gbb.t && l->grouter.t) {
                ok = coli_vk_gdn_full_route_pipe(gi, D, l->gqkv.t, l->gz.t, l->gba.t, l->gbb.t,
                                                 l->gout.t, l->grouter.t, i, idx, val, c->topk,
                                                 c->gdn_kh, c->gdn_kd, c->gdn_vh, c->gdn_vd,
                                                 c->conv_dim, c->conv_k, c->eps, D, E);
                if (ok) gi++;
                else m->vk_gdn_full_on = 0;
            }
            if (!ok) {
                coli_vk_stream_end(x, D);
                use_stream = 0; m->vk_stream_on = 0;
                /* Restart this layer on the host-resid path below. */
                for (int j = i; j < c->n_layers; j++) {
                    Layer *lj = &m->L[j];
                    rmsnorm_zc(nrm, x, lj->in_ln, D, c->eps);
                    if (c->is_full[j]) { attention(m, lj, kvi, nrm, 1, pos_base, tmp); kvi++; }
                    else { gdn_step(m, lj, gi, nrm, tmp); gi++; }
                    for (int d = 0; d < D; d++) x[d] += tmp[d];
                    rmsnorm_zc(nrm, x, lj->post_ln, D, c->eps);
                    moe_token(m, lj, j, nrm, tmp);
                    for (int d = 0; d < D; d++) x[d] += tmp[d];
                }
                goto stack_done;
            }
            if (coli_vk_stream_moe_fused()) {
                /* Async ping-pong: idx not ready until stream_end drain. */
                if (!coli_vk_stream_ix_pending()) {
                    for (int kk = 0; kk < c->topk; kk++)
                        m->eusage[(size_t)i * E + idx[kk]]++;
                }
            } else
                moe_from_topk_pipe(m, l, i, idx, val);
        }
        /* Prefer device final-norm+argmax: drain without host resid copy. */
        coli_vk_stream_end(NULL, D);
        if (coli_vk_stream_ix_pending()) {
            for (int i = 0; i < c->n_layers; i++) {
                int hidx[64];
                int n = coli_vk_stream_ix_hist(i, hidx, c->topk);
                for (int kk = 0; kk < n; kk++)
                    m->eusage[(size_t)i * E + hidx[kk]]++;
            }
        }
        m->vk_stream_argmax = 1;
        m->token_count += S;
        /* scratch owned by Model — do not free */
        return;
    }
#endif
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
#ifdef COLI_VULKAN
        /* Fused layer submit: (in_ln+)attn/GDN + residual + post_ln + router — one fence,
         * then only expert_group remains. Collapses the per-layer CPU rmsnorm/router tax. */
        if (m->vk_route_on && l->grouter.t) {
            float *pr = falloc((int64_t)S * E);
            int ok = 0;
            if (c->is_full[i] && m->vk_gqa_full_on && l->gq.t && l->gk.t && l->gv.t && l->go.t) {
                ok = coli_vk_gqa_full_route(kvi, x, D, l->gq.t, l->gk.t, l->gv.t, l->go.t,
                                            l->grouter.t, i, x, nrm, pr, S, c->n_heads, c->n_kv_heads,
                                            c->head_dim, c->rot, pos_base, m->max_t, c->eps, c->theta,
                                            D, E);
                if (ok) kvi++;
                else m->vk_gqa_full_on = 0;
            } else if (!c->is_full[i] && m->vk_gdn_full_on && S == 1 &&
                       l->gqkv.t && l->gz.t && l->gout.t) {
                /* in_ln stays on the host so the tiny b/a projections can feed params */
                rmsnorm_zc(nrm, x, l->in_ln, D, c->eps);
                int KH = c->gdn_kh, VH = c->gdn_vh, VD = c->gdn_vd;
                float b[64], a[64];
                matmul(b, nrm, l->b, 1, D, VH);
                matmul(a, nrm, l->a, 1, D, VH);
                float *params = falloc(2*VH + VD);
                for (int h = 0; h < VH; h++) {
                    params[h]      = expf(-expf(l->alog[h]) * softplusf_(a[h] + l->dtb[h]));
                    params[VH + h] = sigmoidf_(b[h]);
                }
                memcpy(params + 2*VH, l->gnorm, VD*sizeof(float));
                ok = coli_vk_gdn_full_route(gi, nrm, x, D, l->gqkv.t, l->gz.t, params, l->gout.t,
                                            l->grouter.t, i, x, nrm, pr, KH, c->gdn_kd, VH, VD,
                                            c->conv_dim, c->conv_k, c->eps, D, E);
                free(params);
                if (ok) gi++;
                else m->vk_gdn_full_on = 0;
            }
            if (ok) {
                if (S > 1)
                    moe_batch_from_logits(m, l, i, nrm, pr, S, tmp);  /* prefill: batch by expert */
                else
                    moe_from_logits(m, l, i, nrm, pr, tmp);
                for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
                free(pr);
                continue;
            }
            free(pr);
            /* fall through to the staged path (and disable route if both full paths died) */
            if (!m->vk_gqa_full_on && !m->vk_gdn_full_on) m->vk_route_on = 0;
        }
#endif
        for (int s = 0; s < S; s++) rmsnorm_zc(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        if (c->is_full[i]) {
            attention(m, l, kvi, nrm, S, pos_base, tmp);
            kvi++;
        } else {
#ifdef COLI_VULKAN
            /* Prefill: chunked GDN on one CB per up-to-64 tokens (fewer fences). */
            int gdn_seq = 0;
            if (S > 1 && m->vk_gdn_full_on && l->gqkv.t && l->gz.t && l->gout.t) {
                int VH = c->gdn_vh, VD = c->gdn_vd, prs = 2 * VH + VD;
                float *params = falloc((int64_t)S * prs);
                float b[64], a[64];
                for (int s = 0; s < S; s++) {
                    const float *xs = nrm + (int64_t)s * D;
                    float *ps = params + (int64_t)s * prs;
                    matmul(b, xs, l->b, 1, D, VH);
                    matmul(a, xs, l->a, 1, D, VH);
                    for (int h = 0; h < VH; h++) {
                        ps[h]      = expf(-expf(l->alog[h]) * softplusf_(a[h] + l->dtb[h]));
                        ps[VH + h] = sigmoidf_(b[h]);
                    }
                    memcpy(ps + 2 * VH, l->gnorm, VD * sizeof(float));
                }
                gdn_seq = coli_vk_gdn_full_seq(gi, nrm, S, D, l->gqkv.t, l->gz.t, params,
                                               l->gout.t, tmp, c->gdn_kh, c->gdn_kd, VH, VD,
                                               c->conv_dim, c->conv_k, c->eps, D);
                free(params);
                if (!gdn_seq) m->vk_gdn_full_on = 0;
            }
            if (!gdn_seq)
#endif
            for (int s = 0; s < S; s++)        /* recurrence is inherently sequential over tokens */
                gdn_step(m, l, gi, nrm + (int64_t)s*D, tmp + (int64_t)s*D);
            gi++;
        }
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        for (int s = 0; s < S; s++) rmsnorm_zc(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        if (S > 1) moe_batch(m, l, i, nrm, S, tmp);       /* prefill: batch rows per expert */
        else       moe_token(m, l, i, nrm, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
    }
#ifdef COLI_VULKAN
stack_done:
#endif
    m->token_count += S;
    rmsnorm_zc(last, x + (int64_t)(S-1)*D, m->final_norm, D, c->eps);
#ifdef COLI_VULKAN
    if (S != 1) { free(x); free(nrm); free(tmp); }
#else
    free(x); free(nrm); free(tmp);
#endif
}

static float *step(Model *m, const int *ids, int S, int pos_base) {
    Cfg *c = &m->c;
    float *last = falloc(c->hidden);
    decode_stack(m, ids, S, pos_base, last);
#ifdef COLI_VULKAN
    /* Stream greedy path skipped host final-norm; materialize for logit consumers. */
    if (m->vk_stream_argmax) {
        m->vk_stream_argmax = 0;
        if (!coli_vk_stream_copy_resid(last, c->hidden))
            memset(last, 0, (size_t)c->hidden * sizeof(float));
        rmsnorm_zc(last, last, m->final_norm, c->hidden, c->eps);
    }
#endif
    float *logit = falloc(c->vocab);
    mm_dense(&m->glmh, m->lm_head, logit, last, 1, c->hidden, c->vocab);
    free(last);
    return logit;
}

/* Greedy pick that keeps lm_head's output on the device: the projection and the argmax
 * reduction ride one submit and only the winning token comes back, instead of a
 * vocab-sized vector plus a host-side scan of it. Falls back to the logit path when the
 * device argmax is unavailable — or when logits are being dumped and we need them all. */
static int step_top1(Model *m, const int *ids, int S, int pos_base) {
    Cfg *c = &m->c; int D = c->hidden;
#ifdef COLI_VULKAN
    float *last;
    if (S == 1) { scratch_ensure(m, D); last = m->scratch_last; }
    else last = falloc(D);
#else
    float *last = falloc(D);
#endif
    decode_stack(m, ids, S, pos_base, last);
#ifdef COLI_VULKAN
    if (!g_logits_f && m->glmh.t) {
        int idx;
        /* Phase 1: final RMSNorm + lm_head + argmax fused; resid never leaves GPU. */
        if (m->vk_stream_argmax &&
            coli_vk_stream_norm_argmax(&m->glmh.t, m->glmh.q, m->glmh.s, m->glmh.fmt,
                                       D, c->vocab, m->glmh.gs, c->eps, &idx, NULL)) {
            m->vk_stream_argmax = 0;
            return idx;
        }
        if (m->vk_stream_argmax) {
            /* Device fuse failed — materialize resid + CPU final-norm into last. */
            if (!coli_vk_stream_copy_resid(last, D))
                memset(last, 0, (size_t)D * sizeof(float));
            rmsnorm_zc(last, last, m->final_norm, D, c->eps);
            m->vk_stream_argmax = 0;
        }
        if (coli_vk_matmul_argmax(&m->glmh.t, m->glmh.q, m->glmh.s, m->glmh.fmt,
                                  D, c->vocab, m->glmh.gs, last, &idx, NULL)) {
            if (S != 1) free(last);
            return idx;
        }
    } else if (m->vk_stream_argmax) {
        if (!coli_vk_stream_copy_resid(last, D))
            memset(last, 0, (size_t)D * sizeof(float));
        rmsnorm_zc(last, last, m->final_norm, D, c->eps);
        m->vk_stream_argmax = 0;
    }
#endif
    /* the stack already advanced the KV/GDN state, so finish from the same hidden state */
    float *logit = falloc(c->vocab);
    mm_dense(&m->glmh, m->lm_head, logit, last, 1, D, c->vocab);
#ifdef COLI_VULKAN
    if (S != 1) free(last);
#else
    free(last);
#endif
    dump_logits(m, logit);
    int best = 0; float bv = logit[0];
    for (int i = 1; i < c->vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
    free(logit);
    return best;
}

static void state_alloc(Model *m, int max_t) {
    Cfg *c = &m->c;
    m->max_t = max_t;
    int nfull = 0, ngdn = 0;
    for (int i = 0; i < c->n_layers; i++) c->is_full[i] ? nfull++ : ngdn++;
    m->K = calloc(nfull, sizeof(float*)); m->V = calloc(nfull, sizeof(float*));
    for (int i = 0; i < nfull; i++) {
        m->K[i] = falloc((int64_t)c->n_kv_heads * max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->n_kv_heads * max_t * c->head_dim);
    }
    m->gdnS   = calloc(ngdn, sizeof(float*));
    m->convst = calloc(ngdn, sizeof(float*));
    for (int i = 0; i < ngdn; i++) {
        m->gdnS[i]   = fcalloc((int64_t)c->gdn_vh * c->gdn_kd * c->gdn_vd);
        m->convst[i] = fcalloc((int64_t)c->conv_k * c->conv_dim);
    }
#ifdef COLI_VULKAN
    /* device KV mirror for the GPU GQA path (row = kv_head*max_t + pos, hd K + hd V) */
    if (m->vk_gqa_on) {
        int ok = 1, kvi = 0;
        for (int li = 0; li < c->n_layers && ok; li++) {
            if (!c->is_full[li]) continue;
            ok = coli_vk_kv_ensure(kvi, c->n_kv_heads * max_t, c->head_dim, c->head_dim);
            if (ok && coli_vk_gqa_full_available())   /* resident q/k norm weights for the fused block */
                coli_vk_gqa_norm_weight(kvi, m->L[li].qn, m->L[li].kn, c->head_dim);
            kvi++;
        }
        m->vk_gqa_on = ok;
        printf("[VK] GQA attention %s\n",
               m->vk_gqa_on ? "ENABLED (device KV mirror)" : "on CPU (KV mirror alloc failed)");
        int want_full = g_qwen_opts.gqa_fuse;
        m->vk_gqa_full_on = ok && want_full && coli_vk_gqa_full_available();
        if (m->vk_gqa_full_on)
            printf("[VK] GQA block fusion ENABLED (qkv+norm+rope+attn+oproj in one submit)\n");
    }
    /* MoE prep fusion needs resident routers (vk_dense_init) + in/post LN weights + a
     * full attn/GDN block to hang the residual+rmsnorm+router tail onto. */
    if (coli_vk_moe_route_available()) {
        int ok = 1;
        for (int li = 0; li < c->n_layers && ok; li++) {
            if (!m->L[li].grouter.t) { ok = 0; break; }
            ok = coli_vk_layer_norm_weight(li, m->L[li].in_ln, m->L[li].post_ln, c->hidden);
        }
        int want = g_qwen_opts.route_fuse;
        m->vk_route_on = ok && want && (m->vk_gqa_full_on || m->vk_gdn_full_on);
        if (m->vk_route_on)
            printf("[VK] MoE prep fusion ENABLED (in/post RMSNorm+router in attn/GDN submit)\n");
        /* Device residual stream: needs route fusion + GDN b/a resident + pack shaders. */
        if (m->vk_route_on && coli_vk_stream_available()) {
            int sok = 1, gi = 0;
            for (int li = 0; li < c->n_layers && sok; li++) {
                if (c->is_full[li]) continue;
                if (!m->L[li].gba.t || !m->L[li].gbb.t) { sok = 0; break; }
                sok = coli_vk_gdn_ba_weight(gi, m->L[li].alog, m->L[li].dtb, m->L[li].gnorm,
                                            c->gdn_vh, c->gdn_vd);
                gi++;
            }
            int want_s = g_qwen_opts.stream;
            m->vk_stream_on = sok && want_s;
            if (m->vk_stream_on)
                printf("[VK] decode stream ENABLED (device topk, nrm→eg, resid accumulate, eg→route sem)\n");
        }
    }
#endif
}

/* ---------- greedy generation / teacher-forced NLL (olmoe.c pattern) ---------- */

/* Oracle path: fill out[0..np+n_new) starting from prompt; no EOS stop.
 * prefill_s / decode_s / n_decode are optional; decode timing covers only the S=1 steps. */
static void generate(Model *m, const int *prompt, int np, int n_new, int *out,
                     double *prefill_s, double *decode_s, int *n_decode) {
    state_alloc(m, np + n_new);
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    double t0 = now_s();
    int best = step_top1(m, prompt, np, 0);
    if (prefill_s) *prefill_s = now_s() - t0;
    int len = np, nd = 0; double t_dec = 0;
    for (int s = 0; s < n_new; s++) {
        out[len++] = best;
        if (s == n_new - 1) break;
        t0 = now_s();
        best = step_top1(m, &best, 1, len - 1);
        t_dec += now_s() - t0;
        nd++;
    }
    if (decode_s) *decode_s = t_dec;
    if (n_decode) *n_decode = nd;
}

static void print_usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [options] [cache_cap] [ref.json]\n"
        "\n"
        "Required:\n"
        "  --snap DIR              model container directory\n"
        "\n"
        "Text generation:\n"
        "  --prompt TEXT           user prompt (text mode)\n"
        "  --system TEXT           system prompt (ChatML)\n"
        "  --ngen N                max new tokens (default 64)\n"
        "  --chat / --no-chat      ChatML template (default on)\n"
        "  --ppl                   perplexity mode (needs ref.json)\n"
        "  --logits PATH           dump logits binary\n"
        "\n"
        "Vulkan / iGPU:\n"
        "  --shaders PATH          SPIR-V (e.g. shaders/qmatmul.spv)\n"
        "  --vulkan / --no-vulkan  enable Vulkan (default on if built with VK)\n"
        "  --stream / --no-stream  decode residual stream (default on)\n"
        "  --moe-ix / --no-moe-ix  descriptor-indexed MoE + dual-CB\n"
        "  --experts N             max hot experts pinned in VRAM (default 1024)\n"
        "  --reserve-gb F          device budget reserve GB (default 3)\n"
        "  --dense / --no-dense    dense matmul offload (default on)\n"
        "  --dense-bits N          projection weights 8|4 bit (default 8)\n"
        "  --lmhead-bits N         lm_head weights 8|4 bit (default 8)\n"
        "  --dense-gs N            int4 group size (default 128)\n"
        "  --gdn / --no-gdn        GDN on GPU (default on)\n"
        "  --gdn-fuse / --no-gdn-fuse\n"
        "  --gqa / --no-gqa        full-attn GQA on GPU (default on)\n"
        "  --gqa-fuse / --no-gqa-fuse\n"
        "  --route-fuse / --no-route-fuse\n"
        "  --topk / --no-topk      device softmax top-k (default on)\n"
        "  --moe-ix-pp / --no-moe-ix-pp\n"
        "  --eg-cache / --no-eg-cache\n"
        "  --spin-us N             fence spin µs (default 300; 0=block)\n"
        "  --dp4a / --no-dp4a\n"
        "  --flash MODE            -1=auto 0=off 1=on\n"
        "  --eg-stats / --eg-dbg / --vk-prof\n"
        "\n"
        "Positional:\n"
        "  cache_cap               CPU expert LRU depth per layer (default 32)\n"
        "  ref.json                oracle / PPL token ids (if no --prompt)\n"
        "\n"
        "Example:\n"
        "  %s --snap <SNAP> --shaders shaders/qmatmul.spv --moe-ix \\\n"
        "     --experts 7284 --prompt \"你好\" --ngen 33 8\n",
        argv0, argv0);
}

static const char *qwen_user_prompt(void) {
    return g_qwen_opts.prompt;
}

/* Qwen ChatML wrap (matches HF apply_chat_template for a single user turn +
 * optional --system and add_generation_prompt=True). Caller frees. */
static char *qwen_chat_wrap(const char *user) {
    const char *sys = g_qwen_opts.system;
    size_t ul = strlen(user), sl = sys ? strlen(sys) : 0;
    /* <|im_start|>system\n...\n<|im_end|>\n  + user turn + assistant prefix */
    size_t need = ul + sl + 128;
    char *buf = malloc(need);
    if (!buf) { fprintf(stderr, "OOM chat wrap\n"); exit(1); }
    size_t o = 0;
    if (sys && sys[0]) {
        o += (size_t)snprintf(buf + o, need - o, "<|im_start|>system\n%s<|im_end|>\n", sys);
    }
    o += (size_t)snprintf(buf + o, need - o, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user);
    (void)o;
    return buf;
}

/* Text generation: tokenize, prefill, greedy decode with stop tokens, stream
 * detokenized text. Returns number of new tokens produced (excluding prompt).
 * Prefill and decode wall times are split: the first next-token comes from the
 * prompt forward, every subsequent token is a pure S=1 decode step. */
static int generate_text(Model *m, Tok *T, const int *prompt, int np, int n_new,
                         const int *stop_ids, int n_stop,
                         double *prefill_s, double *decode_s, int *n_decode) {
    state_alloc(m, np + n_new + 2);
    double t0 = now_s();
    int best = step_top1(m, prompt, np, 0);
    if (prefill_s) *prefill_s = now_s() - t0;
    int produced = 0, nd = 0; double t_dec = 0;
    for (int s = 0; s < n_new; s++) {
        int is_stop = 0;
        for (int k = 0; k < n_stop; k++) if (best == stop_ids[k]) { is_stop = 1; break; }
        if (is_stop) break;
        /* Detokenize outside decode timing (Phase 1.4); fflush once per token is fine. */
        if (best >= 0 && best < T->n_ids && !T->id_special[best]) {
            char dec[256]; int dn = tok_decode(T, &best, 1, dec, (int)sizeof(dec) - 1);
            if (dn > 0) { fwrite(dec, 1, (size_t)dn, stdout); fflush(stdout); }
        }
        produced++;
        if (s == n_new - 1) break;
        t0 = now_s();
        best = step_top1(m, &best, 1, np + produced - 1);
        t_dec += now_s() - t0;
        nd++;
    }
    if (decode_s) *decode_s = t_dec;
    if (n_decode) *n_decode = nd;
    return produced;
}

static int tf_nll(Model *m, const int *full, int nfull, int np, double *nll_out) {
    Cfg *c = &m->c;
    state_alloc(m, nfull);
    double nll = 0; int scored = 0;
    float *logit = step(m, full, np, 0);
    dump_logits(m, logit);
    for (int i = np; i < nfull; i++) {
        float mx = logit[0]; for (int v = 1; v < c->vocab; v++) if (logit[v] > mx) mx = logit[v];
        double Z = 0; for (int v = 0; v < c->vocab; v++) Z += exp((double)logit[v] - mx);
        nll += -((double)logit[full[i]] - mx - log(Z));
        scored++;
        free(logit); logit = NULL;
        if (i == nfull - 1) break;
        logit = step(m, &full[i], 1, i);
        dump_logits(m, logit);
    }
    if (logit) free(logit);
    *nll_out = nll / scored;
    return scored;
}

static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    if (!a || a->t != J_ARR) { fprintf(stderr, "ref.json: missing array \"%s\"\n", key); exit(1); }
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

static void vk_maybe_init(Model *m) {
#ifdef COLI_VULKAN
    if (g_qwen_opts.vulkan) {
        const char *sp = g_qwen_opts.shaders;
        m->vk_on = coli_vk_init(sp ? sp : "shaders/qmatmul.spv");
        printf("[VK] expert tier %s\n", m->vk_on ? "ENABLED (pinned, filled from usage history at startup)"
                                                 : "unavailable — CPU only");
        /* GatedDeltaNet recurrence on the GPU: allocate device-resident per-GDN-layer
         * state (zeroed = fresh sequence). --no-gdn keeps it on the CPU. */
        int want_gdn = g_qwen_opts.gdn;
        if (m->vk_on && want_gdn && coli_vk_gdn_available()) {
            Cfg *c = &m->c; int ok = 1, gi = 0;
            for (int i = 0; i < c->n_layers && ok; i++)
                if (!c->is_full[i]) ok = coli_vk_gdn_state_ensure(gi++, c->gdn_vh, c->gdn_kd, c->gdn_vd);
            m->vk_gdn_on = ok;
            printf("[VK] GatedDeltaNet recurrence %s\n",
                   m->vk_gdn_on ? "ENABLED (device-resident state)" : "on CPU (state alloc failed)");
            /* Fused whole-GDN-block path: also needs a device conv ring + resident conv
             * weights per GDN layer. COLI_VK_GDN_FUSE=0 keeps conv/qk-norm on the CPU. */
            int want_full = g_qwen_opts.gdn_fuse;
            if (m->vk_gdn_on && want_full && coli_vk_gdn_full_available()) {
                int okf = 1, gi2 = 0;
                for (int i = 0; i < c->n_layers && okf; i++)
                    if (!c->is_full[i]) {
                        okf = coli_vk_gdn_conv_ensure(gi2, c->conv_k, c->conv_dim)
                           && coli_vk_gdn_conv_weight(gi2, m->L[i].conv, c->conv_k, c->conv_dim);
                        gi2++;
                    }
                m->vk_gdn_full_on = okf;
                printf("[VK] GatedDeltaNet block fusion %s\n",
                       m->vk_gdn_full_on ? "ENABLED (conv+qknorm+delta+outproj in one submit)" : "on staged path");
            }
        }
        /* GQA attention eligibility (device KV mirror ensured later in state_alloc,
         * where max_t is known). COLI_VK_GQA=0 keeps it on the CPU. */
        int want_gqa = g_qwen_opts.gqa;
        m->vk_gqa_on = m->vk_on && want_gqa && coli_vk_gqa_available();
    }
#else
    (void)m;
#endif
}

#ifdef COLI_VULKAN
/* Pinned VK expert tier fill, aligned with GLM's vk_registry_fill (colibri.c):
 * upload the top-COLI_VK_EXPERTS heat-ranked routed experts ONCE at startup —
 * stable residency, ZERO uploads on the decode critical path, and a hard cap
 * on iGPU memory instead of "everything ever touched stays resident".
 * Tier weights use eviction-priority 0.4 (scratch/KV 1.0, default 0.75) so an
 * oversubscribed heap sheds cold experts first, and the fill stops while the
 * device-local budget still holds COLI_VK_RESERVE_GB (default 3) for the
 * lazily-allocated scratch + staging. */
typedef struct { uint32_t u; int layer, eid; } VkCand;
static int vk_cand_cmp(const void *a, const void *b) {
    uint32_t ua = ((const VkCand*)a)->u, ub = ((const VkCand*)b)->u;
    return ua < ub ? 1 : ua > ub ? -1 : 0;
}
static void vk_tier_fill(Model *m) {
    Cfg *c = &m->c; int E = c->n_experts, NL = c->n_layers;
    if (!m->vk_on) return;
    /* GLM defaults to 320 experts (~19 MB each). Qwen int4 experts are ~1.5 MB,
     * so 1024 (~1.6 GB) keeps a good hit rate at a fraction of the memory. */
    int budget = g_qwen_opts.experts;
    if (budget <= 0) { printf("[VK] expert tier: disabled (--experts<=0) — all experts on CPU\n"); return; }
    int64_t nz = 0;
    for (int64_t i = 0; i < (int64_t)NL * E; i++) if (m->eusage[i]) nz++;
    if (!nz) {
        printf("[VK] expert tier: no usage history yet — tier empty this run "
               "(it seeds from %s as you use the model)\n", m->usage_path);
        return;
    }
    VkCand *cand = malloc((size_t)nz * sizeof(VkCand)); if (!cand) return;
    int64_t n = 0;
    for (int l = 0; l < NL; l++) for (int e = 0; e < E; e++)
        if (m->eusage[(size_t)l * E + e]) cand[n++] = (VkCand){m->eusage[(size_t)l * E + e], l, e};
    qsort(cand, (size_t)n, sizeof(VkCand), vk_cand_cmp);
    double reserve = g_qwen_opts.reserve_gb;
    int fmt = c->expert_fmt, gsz = c->expert_fmt == 6 ? c->group_size : 0;
    Slot tmp; memset(&tmp, 0, sizeof(tmp)); tmp.eid = -1; slot_alloc(m, &tmp);
    double t0 = now_s(); int64_t bytes = 0; int stopped = 0; double vu = 0, vb = 0;
    coli_vk_alloc_priority(0.4f);                /* evictable class */
    for (int64_t i = 0; i < n && m->vk_reg_n < budget; i++) {
        if (reserve > 0 && (m->vk_reg_n & 7) == 0 && coli_vk_mem_budget(&vu, &vb)
            && vb - vu < reserve) { stopped = 1; break; }
        int layer = cand[i].layer, eid = cand[i].eid;
        expert_read(m, layer, eid, &tmp);
        ColiVkTensor **r = vk_reg_at(m, layer, eid);
        if (!coli_vk_tensor_ensure(&r[0], tmp.g, tmp.gs, fmt, c->hidden, c->inter, gsz) ||
            !coli_vk_tensor_ensure(&r[1], tmp.u, tmp.us, fmt, c->hidden, c->inter, gsz) ||
            !coli_vk_tensor_ensure(&r[2], tmp.d, tmp.ds, fmt, c->inter, c->hidden, gsz)) {
            for (int j = 0; j < 3; j++) if (r[j]) { coli_vk_tensor_free(r[j]); r[j] = NULL; }
            fprintf(stderr, "[VK] expert tier: VRAM full after %d experts\n", m->vk_reg_n);
            break;
        }
        bytes += coli_vk_tensor_bytes(r[0]) + coli_vk_tensor_bytes(r[1]) + coli_vk_tensor_bytes(r[2]);
        m->vk_reg_n++;
    }
    coli_vk_alloc_priority(0.75f);               /* back to the default class */
    free(tmp.g); free(tmp.gs);                   /* g owns the weight slab, gs the scale slab */
    free(cand);
    printf("[VK] expert tier: %d hot experts resident (%.2f GB VRAM, %.1fs, top of %lld-entry history)\n",
           m->vk_reg_n, bytes / 1e9, now_s() - t0, (long long)n);
    if (stopped)
        printf("[VK] expert tier: budget stop at %d experts — %.1f of %.1f GB device-local used, %.1f GB reserved (--reserve-gb)\n",
               m->vk_reg_n, vu, vb, reserve);
}

/* Quantize one f32 dense weight and upload it resident. On success the caller
 * frees the f32 original; the quantized host copy stays (upload source is not
 * read again, but it is the CPU fallback if the VK device dies mid-run). */
static int vkd_make(VkDense *d, const float *W, int I, int O, int fmt, int gs) {
    size_t wb = (fmt == 1 ? (size_t)I : (size_t)((I + 1) / 2)) * (size_t)O;
    size_t sf = fmt == 6 ? (size_t)O * ((I + gs - 1) / gs) * 2 : (size_t)O;
    d->q = malloc(wb); d->s = malloc(sf * sizeof(float));
    if (!d->q || !d->s) { free(d->q); free(d->s); d->q = NULL; d->s = NULL; return 0; }
    quant_rows(W, I, O, fmt, gs, d->q, d->s);
    d->fmt = fmt; d->gs = fmt == 6 ? gs : 0;
    if (!coli_vk_tensor_ensure(&d->t, d->q, d->s, fmt, I, O, d->gs)) {
        /* VRAM full: drop the quantized copy, keep the f32 CPU path untouched */
        free(d->q); free(d->s); d->q = NULL; d->s = NULL; d->t = NULL; return 0;
    }
    return 1;
}

/* GLM's vk_dense_preload, adapted: quantize the f32 dense working set (attention
 * and GDN projections, shared expert in the container's expert fmt so it can
 * join the expert_group submit, plus lm_head) and upload it BEFORE the tier fill,
 * so the budget-capped tier sizes itself to the true remainder. Frees the f32
 * originals on success — the host keeps only the quantized fallback copies.
 * Width comes from --dense-bits / --lmhead-bits: 8 = per-row int8 (fmt 1),
 * 4 = grouped-asymmetric int4 (fmt 6). Dense is the largest per-token weight
 * stream at decode, so int4 roughly halves its bandwidth. */
static void vk_dense_init(Model *m) {
    Cfg *c = &m->c;
    if (!m->vk_on) return;
    if (!g_qwen_opts.dense) {
        printf("[VK] dense offload disabled (--no-dense)\n"); return;
    }
    int efmt = c->expert_bits == 4 ? 2 : 1;
    /* Shared expert takes the ROUTED experts' fmt so it can ride their fused
     * expert_group submit (the group decodes the whole batch with one fmt; a
     * fmt mismatch would reject the group and drop every expert to CPU). */
    int sfmt = c->expert_fmt == 6 ? 6 : efmt, sgs = c->expert_fmt == 6 ? c->group_size : 0;
    if (g_qwen_opts.dense_bits != 8 && g_qwen_opts.dense_bits != 4)
        fprintf(stderr, "[VK] --dense-bits %d unsupported (8|4) — using 8\n", g_qwen_opts.dense_bits);
    if (g_qwen_opts.lmhead_bits != 8 && g_qwen_opts.lmhead_bits != 4)
        fprintf(stderr, "[VK] --lmhead-bits %d unsupported (8|4) — using 8\n", g_qwen_opts.lmhead_bits);
    /* fmt=6 needs gs % 8 == 0 (a packed uint32 must not straddle a group). */
    int dgs = g_qwen_opts.dense_gs;
    if (dgs < 8 || dgs % 8) {
        fprintf(stderr, "[VK] --dense-gs %d invalid (>=8, multiple of 8) — using 128\n", dgs);
        dgs = 128;
    }
    int dfmt = g_qwen_opts.dense_bits == 4 ? 6 : 1;
    int dgs_use = dfmt == 6 ? dgs : 0;
    int hfmt = g_qwen_opts.lmhead_bits == 4 ? 6 : 1;
    int hgs_use = hfmt == 6 ? dgs : 0;
    if (dfmt == 6 || hfmt == 6)
        printf("[VK] dense quant: projections int%d, lm_head int%d (fmt6 gs=%d)\n",
               g_qwen_opts.dense_bits == 4 ? 4 : 8,
               g_qwen_opts.lmhead_bits == 4 ? 4 : 8, dgs);
    double t0 = now_s(); int64_t bytes = 0; int nt = 0, full = 0;
    for (int i = 0; i < c->n_layers && !full; i++) {
        Layer *l = &m->L[i];
        struct { VkDense *d; float **w; int I, O, fmt, gs; } ts[] = {
            {&l->gq,   &l->q,    c->hidden,               c->n_heads*c->head_dim*2,  dfmt, dgs_use},
            {&l->gk,   &l->k,    c->hidden,               c->n_kv_heads*c->head_dim, dfmt, dgs_use},
            {&l->gv,   &l->v,    c->hidden,               c->n_kv_heads*c->head_dim, dfmt, dgs_use},
            {&l->go,   &l->o,    c->n_heads*c->head_dim,  c->hidden,                 dfmt, dgs_use},
            {&l->gqkv, &l->qkv,  c->hidden,               c->conv_dim,               dfmt, dgs_use},
            {&l->gz,   &l->z,    c->hidden,               c->value_dim,              dfmt, dgs_use},
            {&l->gout, &l->outp, c->value_dim,            c->hidden,                 dfmt, dgs_use},
            {&l->gsg,  &l->sh_g, c->hidden,               c->sh_inter,            sfmt, sgs},
            {&l->gsu,  &l->sh_u, c->hidden,               c->sh_inter,            sfmt, sgs},
            {&l->gsd,  &l->sh_d, c->sh_inter,             c->hidden,              sfmt, sgs},
            /* router picks the experts — keep it int8, it is tiny and error-sensitive */
            {&l->grouter, &l->router, c->hidden,          c->n_experts,           1,    0},
            {&l->gba,  &l->a,     c->hidden,               c->gdn_vh,               1,    0},
            {&l->gbb,  &l->b,     c->hidden,               c->gdn_vh,               1,    0},
        };
        for (size_t k = 0; k < sizeof(ts)/sizeof(ts[0]); k++) {
            if (!*ts[k].w) continue;               /* layer type doesn't have this proj */
            if (!vkd_make(ts[k].d, *ts[k].w, ts[k].I, ts[k].O, ts[k].fmt, ts[k].gs)) {
                fprintf(stderr, "[VK] dense init: VRAM full at layer %d — remaining stay f32/CPU\n", i);
                full = 1; break;
            }
            /* Keep GDN b/a f32 for the non-stream / CPU fallback path (tiny vs hidden). */
            if (ts[k].d != &l->gba && ts[k].d != &l->gbb) {
                free(*ts[k].w); *ts[k].w = NULL;
            }
            bytes += coli_vk_tensor_bytes(ts[k].d->t); nt++;
        }
    }
    if (!full && vkd_make(&m->glmh, m->lm_head, c->hidden, c->vocab, hfmt, hgs_use)) {
        if (m->lm_head != m->embed) { free(m->lm_head); m->lm_head = NULL; }  /* tied: embed stays */
        bytes += coli_vk_tensor_bytes(m->glmh.t); nt++;
    }
    if (m->final_norm && coli_vk_final_norm_weight(m->final_norm, c->hidden))
        printf("[VK] final RMSNorm weight resident (stream→argmax fuse)\n");
    if (nt) printf("[VK] dense offload: %d tensors quantized + resident (%.2f GB VRAM, %.1fs) — f32 originals freed\n",
                   nt, bytes / 1e9, now_s() - t0);
}
#endif

static void vk_maybe_shutdown(Model *m) {
#ifdef COLI_VULKAN
    if (m->vk_on) {
        size_t used = 0, nten = 0;
        coli_vk_mem_info(&used, &nten);
        double ugb = 0, bgb = 0;
        int has_b = coli_vk_mem_budget(&ugb, &bgb);
        printf("[VK] tier-resident experts: %d | resident tensors: %zu (%.3f GB tracked)\n",
               m->vk_reg_n, nten, used / 1e9);
        if (has_b) printf("[VK] device-local budget: used=%.3f GB / budget=%.3f GB\n", ugb, bgb);
        coli_vk_shutdown();
    }
#else
    (void)m;
#endif
}

static int run_text_mode(Model *m, const char *snap, const char *user, int ngen) {
    char tkp[2048]; snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
    { FILE *tf = fopen(tkp, "rb");
      if (!tf) {
          fprintf(stderr, "missing %s — convert_qwen35_moe.py copies it from the HF snapshot\n", tkp);
          return 1;
      }
      fclose(tf); }
    Tok T; tok_load(&T, tkp);
    int templ = g_qwen_opts.chat_template;
    char *wrapped = NULL;
    const char *text = user;
    if (templ) { wrapped = qwen_chat_wrap(user); text = wrapped; }

    int cap = (int)strlen(text) + 64;
    int *pids = malloc((size_t)cap * sizeof(int));
    int np = tok_encode(&T, text, (int)strlen(text), pids, cap);
    if (np < 1) { fprintf(stderr, "prompt empty after tokenization\n"); free(pids); free(wrapped); return 1; }

    int stops[4], n_stop = 0;
    int im_end = tok_id_of(&T, "<|im_end|>");
    int eot    = tok_id_of(&T, "<|endoftext|>");
    if (im_end >= 0) stops[n_stop++] = im_end;
    if (eot >= 0 && eot != im_end) stops[n_stop++] = eot;
    if (n_stop < 1) fprintf(stderr, "warning: no <|im_end|>/<|endoftext|> in tokenizer — will run full NGEN\n");

    printf("prompt: %d tokens | generating up to %d | stop_ids=%d", np, ngen, n_stop);
    for (int i = 0; i < n_stop; i++) printf(" %d", stops[i]);
    printf(" | chat_template=%d\n", templ);
    fputs("---\n", stdout); fflush(stdout);

    double t_pf = 0, t_dec = 0; int n_dec = 0;
    int produced = generate_text(m, &T, pids, np, ngen, stops, n_stop, &t_pf, &t_dec, &n_dec);
    double tot = m->hits + m->miss;
    /* tok/s is DECODE ONLY: prefill is reported separately and never folded in. */
    printf("\n---\n%d new tokens | prefill %.2fs (prompt %d) | decode %d tok in %.2fs (%.2f tok/s) | expert hit %.1f%% (hit=%llu miss=%llu)\n",
           produced, t_pf, np, n_dec, t_dec, n_dec > 0 ? n_dec / t_dec : 0.0,
           tot ? 100.0 * m->hits / tot : 0.0,
           (unsigned long long)m->hits, (unsigned long long)m->miss);
#ifdef COLI_VULKAN
    coli_vk_decode_prof(t_dec * 1000.0, n_dec);
    if (g_qwen_opts.eg_stats) coli_vk_route_cache_stats();
    { long tt = g_vk_srv + g_vk_unsrv;
      printf("[VK] routed experts: tier-served %ld (%.1f%%), CPU %ld\n",
             g_vk_srv, tt ? 100.0 * g_vk_srv / tt : 0.0, g_vk_unsrv); }
#endif
    printf("peak RSS: %.2f GB\n", rss_gb());
    free(pids); free(wrapped);
    return 0;
}

int main(int argc, char **argv) {
    int i;
    qwen_opts_init_defaults();

#define NEED_ARG() do { if (i + 1 >= argc) { fprintf(stderr, "%s: missing value\n", a); return 1; } } while (0)
    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { print_usage(argv[0]); return 0; }
        if (!strcmp(a, "--snap")) { NEED_ARG(); g_qwen_opts.snap = argv[++i]; }
        else if (!strcmp(a, "--prompt")) { NEED_ARG(); g_qwen_opts.prompt = argv[++i]; }
        else if (!strcmp(a, "--system")) { NEED_ARG(); g_qwen_opts.system = argv[++i]; }
        else if (!strcmp(a, "--ngen")) { NEED_ARG(); g_qwen_opts.ngen = atoi(argv[++i]); }
        else if (!strcmp(a, "--shaders")) { NEED_ARG(); g_qwen_opts.shaders = argv[++i]; }
        else if (!strcmp(a, "--experts")) { NEED_ARG(); g_qwen_opts.experts = atoi(argv[++i]); }
        else if (!strcmp(a, "--reserve-gb")) { NEED_ARG(); g_qwen_opts.reserve_gb = atof(argv[++i]); }
        else if (!strcmp(a, "--logits")) { NEED_ARG(); g_qwen_opts.logits = argv[++i]; }
        else if (!strcmp(a, "--ref")) { NEED_ARG(); g_qwen_opts.ref = argv[++i]; }
        else if (!strcmp(a, "--spin-us")) { NEED_ARG(); g_qwen_opts.spin_us = atol(argv[++i]); }
        else if (!strcmp(a, "--flash")) { NEED_ARG(); g_qwen_opts.flash = atoi(argv[++i]); }
        else if (!strcmp(a, "--ppl")) { g_qwen_opts.ppl = 1; }
        else if (!strcmp(a, "--chat")) { g_qwen_opts.chat_template = 1; }
        else if (!strcmp(a, "--no-chat")) { g_qwen_opts.chat_template = 0; }
        else if (!strcmp(a, "--vulkan")) { g_qwen_opts.vulkan = 1; }
        else if (!strcmp(a, "--no-vulkan")) { g_qwen_opts.vulkan = 0; }
        else if (!strcmp(a, "--stream")) { g_qwen_opts.stream = 1; }
        else if (!strcmp(a, "--no-stream")) { g_qwen_opts.stream = 0; }
        else if (!strcmp(a, "--moe-ix")) { g_qwen_opts.moe_ix = 1; }
        else if (!strcmp(a, "--no-moe-ix")) { g_qwen_opts.moe_ix = 0; }
        else if (!strcmp(a, "--dense")) { g_qwen_opts.dense = 1; }
        else if (!strcmp(a, "--no-dense")) { g_qwen_opts.dense = 0; }
        else if (!strcmp(a, "--dense-bits")) { NEED_ARG(); g_qwen_opts.dense_bits = atoi(argv[++i]); }
        else if (!strcmp(a, "--lmhead-bits")) { NEED_ARG(); g_qwen_opts.lmhead_bits = atoi(argv[++i]); }
        else if (!strcmp(a, "--dense-gs")) { NEED_ARG(); g_qwen_opts.dense_gs = atoi(argv[++i]); }
        else if (!strcmp(a, "--gdn")) { g_qwen_opts.gdn = 1; }
        else if (!strcmp(a, "--no-gdn")) { g_qwen_opts.gdn = 0; }
        else if (!strcmp(a, "--gdn-fuse")) { g_qwen_opts.gdn_fuse = 1; }
        else if (!strcmp(a, "--no-gdn-fuse")) { g_qwen_opts.gdn_fuse = 0; }
        else if (!strcmp(a, "--gqa")) { g_qwen_opts.gqa = 1; }
        else if (!strcmp(a, "--no-gqa")) { g_qwen_opts.gqa = 0; }
        else if (!strcmp(a, "--gqa-fuse")) { g_qwen_opts.gqa_fuse = 1; }
        else if (!strcmp(a, "--no-gqa-fuse")) { g_qwen_opts.gqa_fuse = 0; }
        else if (!strcmp(a, "--route-fuse")) { g_qwen_opts.route_fuse = 1; }
        else if (!strcmp(a, "--no-route-fuse")) { g_qwen_opts.route_fuse = 0; }
        else if (!strcmp(a, "--topk")) { g_qwen_opts.topk = 1; }
        else if (!strcmp(a, "--no-topk")) { g_qwen_opts.topk = 0; }
        else if (!strcmp(a, "--moe-ix-pp")) { g_qwen_opts.moe_ix_pp = 1; }
        else if (!strcmp(a, "--no-moe-ix-pp")) { g_qwen_opts.moe_ix_pp = 0; }
        else if (!strcmp(a, "--eg-cache")) { g_qwen_opts.eg_cache = 1; }
        else if (!strcmp(a, "--no-eg-cache")) { g_qwen_opts.eg_cache = 0; }
        else if (!strcmp(a, "--dp4a")) { g_qwen_opts.dp4a = 1; }
        else if (!strcmp(a, "--no-dp4a")) { g_qwen_opts.dp4a = 0; }
        else if (!strcmp(a, "--eg-stats")) { g_qwen_opts.eg_stats = 1; }
        else if (!strcmp(a, "--eg-dbg")) { g_qwen_opts.eg_dbg = 1; }
        else if (!strcmp(a, "--vk-prof")) { g_qwen_opts.vk_prof = 1; }
        else if (a[0] == '-') {
            fprintf(stderr, "unknown option: %s\n\n", a);
            print_usage(argv[0]);
            return 1;
        } else if (a[0] >= '0' && a[0] <= '9') {
            g_qwen_opts.cap = atoi(a);
        } else if (!g_qwen_opts.ref) {
            g_qwen_opts.ref = a;
        } else {
            fprintf(stderr, "unexpected argument: %s\n", a);
            return 1;
        }
    }
#undef NEED_ARG

    if (!g_qwen_opts.snap) {
        fprintf(stderr, "missing --snap DIR\n\n");
        print_usage(argv[0]);
        return 1;
    }
    if (g_qwen_opts.cap < 8) g_qwen_opts.cap = 8;
    if (g_qwen_opts.ngen < 1) g_qwen_opts.ngen = 1;

    const char *snap = g_qwen_opts.snap;
    int cap = g_qwen_opts.cap;
    int ngen = g_qwen_opts.ngen;

    Model m; model_init(&m, snap, cap);
    Cfg *c = &m.c;
    int nfl = 0; for (int i2 = 0; i2 < c->n_layers; i2++) nfl += c->is_full[i2];
    printf("== Qwen3.5-MoE C engine | hidden=%d layers=%d (%d full-attn) experts=%d top%d int%d | cache=%d/layer ==\n",
           c->hidden, c->n_layers, nfl, c->n_experts, c->topk, c->expert_bits, cap);
    printf("dense loaded in %.1fs\n", m.dense_load_s);
    vk_maybe_init(&m);
#ifdef COLI_VULKAN
    if (m.vk_on) {
        vk_dense_init(&m);           /* dense claims VRAM first (GLM lesson: the tier
                                      * fill then sizes itself to the true remainder) */
        vk_tier_fill(&m);            /* pinned tier: fill once, before any decode */
        if (g_qwen_opts.moe_ix &&
            c->expert_fmt == 6 && c->group_size > 0) {
            if (coli_vk_eg_table_init(c->n_layers, c->n_experts, c->hidden, c->inter, c->group_size)) {
                for (int li = 0; li < c->n_layers; li++) {
                    Layer *l = &m.L[li];
                    if (l->gsg.t && l->gsu.t && l->gsd.t && c->sh_inter == c->inter)
                        coli_vk_eg_table_set_shared(li, l->gsg.t, l->gsu.t, l->gsd.t, l->sh_gate, c->hidden);
                    for (int e = 0; e < c->n_experts; e++)
                        if (vk_reg_served(&m, li, e)) {
                            ColiVkTensor **r = vk_reg_at(&m, li, e);
                            coli_vk_eg_table_set(li, e, r[0], r[1], r[2]);
                        }
                }
            }
        }
    }
#endif

    if (g_qwen_opts.logits) {
        g_logits_f = fopen(g_qwen_opts.logits, "wb");
        if (!g_logits_f) perror(g_qwen_opts.logits);
    }

    const char *user = qwen_user_prompt();
    int rc = 0;
    if (user) {
        rc = run_text_mode(&m, snap, user, ngen);
    } else {
        /* Oracle / PPL mode needs a ref.json of token ids. */
        const char *refpath = g_qwen_opts.ref ? g_qwen_opts.ref : "ref.json";
        FILE *f = fopen(refpath, "rb");
        if (!f) {
            fprintf(stderr,
                "no --prompt and cannot open %s\n"
                "  text mode:   %s --snap <dir> --prompt \"你好\" --ngen 32 %d\n"
                "  oracle mode: %s --snap <dir> %d ref.json\n",
                refpath, argv[0], cap, argv[0], cap);
            vk_maybe_shutdown(&m);
            return 1;
        }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        char *buf = malloc((size_t)n + 1);
        if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "%s: short read\n", refpath); return 1; }
        buf[n] = 0; fclose(f);
        char *arena = NULL; jval *ref = json_parse(buf, &arena);
        int np, nfull;
        int *prompt = read_int_array(ref, "prompt_ids", &np);
        int *full   = read_int_array(ref, "full_ids", &nfull);
        int n_new = nfull - np;

        if (g_qwen_opts.ppl) {
            double nll; double t = now_s();
            int scored = tf_nll(&m, full, nfull, np, &nll);
            double dt = now_s() - t;
            double tot = m.hits + m.miss;
            printf("TF-NLL: %.4f nats/token over %d tokens | ppl = %.2f\n", nll, scored, exp(nll));
            printf("Expert cache hit rate: %.1f%% (hit=%llu miss=%llu)\n", tot ? 100.0 * m.hits / tot : 0.0,
                   (unsigned long long)m.hits, (unsigned long long)m.miss);
            printf("Speed: %.2f tok/s (%.1fs for %d tokens)\n", scored / dt, dt, scored);
        } else {
            int *out = malloc((size_t)(np + n_new) * sizeof(int));
            double t_pf = 0, t_dec = 0; int n_dec = 0;
            generate(&m, prompt, np, n_new, out, &t_pf, &t_dec, &n_dec);
            int match = 0;
            printf("\nReference: "); for (int j = np; j < nfull; j++) printf("%d ", full[j]);
            printf("\nC engine : "); for (int j = np; j < nfull; j++) { printf("%d ", out[j]); if (out[j] == full[j]) match++; }
            printf("\nMatching tokens: %d/%d\n", match, n_new);
            double tot = m.hits + m.miss;
            printf("Expert cache hit rate: %.1f%% (hit=%llu miss=%llu)\n", tot ? 100.0 * m.hits / tot : 0.0,
                   (unsigned long long)m.hits, (unsigned long long)m.miss);
            printf("Prefill: %.2fs (prompt %d) | decode: %d tok in %.2fs (%.2f tok/s)\n",
                   t_pf, np, n_dec, t_dec, n_dec > 0 ? n_dec / t_dec : 0.0);
        }
        free(buf); free(arena);
    }

    usage_save(&m);                  /* the cache that learns: history feeds the next run's tier */
    if (g_logits_f) fclose(g_logits_f);
    vk_maybe_shutdown(&m);
    return rc;
}
