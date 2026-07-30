/*
 * Copyright 2026 qwen3.c contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Runtime options formerly read via getenv (COLI_*, NGEN, ...). Defaults match
 * the previous env-unset semantics. Filled by CLI in qwen.c.
 */

#ifndef QWEN_OPTS_H
#define QWEN_OPTS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QwenOpts {
    /* paths / strings (owned by argv or static; not freed) */
    const char *snap;
    const char *prompt;
    const char *system;
    const char *shaders;
    const char *logits;
    const char *ref;

    int   ngen;           /* default 64 */
    int   chat_template;  /* default 1 */
    int   ppl;            /* default 0 */
    int   cap;            /* CPU LRU depth; default 32 */

    /* Vulkan engine (qwen.c) — defaults match former getenv logic */
    int    vulkan;        /* default 1 when built with COLI_VULKAN */
    int    stream;        /* default 1 (was: unset ⇒ on) */
    int    moe_ix;        /* default 0 (was: only if COLI_VK_MOE_IX=1) */
    int    experts;       /* default 1024 */
    double reserve_gb;    /* default 3.0 */
    int    dense;         /* default 1 (was: COLI_VK_DENSE=0 disables) */
    int    gdn;           /* default 1 */
    int    gdn_fuse;      /* default 1 */
    int    gqa;           /* default 1 */
    int    gqa_fuse;      /* default 1 */
    int    route_fuse;    /* default 1 */

    /* Vulkan backend (backend_vulkan.c) */
    int  topk;            /* default 1 */
    int  moe_ix_pp;       /* default 1 */
    int  eg_cache;        /* default 1 */
    long spin_us;         /* default 300; 0 = blocking wait */
    int  dp4a;            /* default 0 */
    int  flash;           /* default -1 (auto); 0/1 force */
    int  eg_stats;        /* default 0 */
    int  eg_dbg;          /* default 0 */
    int  vk_prof;         /* default 0 */
    int  test_ballast;    /* default 0 (self-test only) */
} QwenOpts;

extern QwenOpts g_qwen_opts;

/* Reset all fields to former getenv-unset defaults. Call before CLI parse. */
void qwen_opts_init_defaults(void);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_OPTS_H */
