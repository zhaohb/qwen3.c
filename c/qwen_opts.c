/*
 * Copyright 2026 qwen3.c contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "qwen_opts.h"

QwenOpts g_qwen_opts;

void qwen_opts_init_defaults(void) {
    QwenOpts z = {0};
    z.ngen = 64;
    z.chat_template = 1;
    z.cap = 32;
#ifdef COLI_VULKAN
    z.vulkan = 1;
#else
    z.vulkan = 0;
#endif
    z.stream = 1;
    z.moe_ix = 0;
    z.experts = 1024;
    z.reserve_gb = 3.0;
    z.dense = 1;
    z.gdn = 1;
    z.gdn_fuse = 1;
    z.gqa = 1;
    z.gqa_fuse = 1;
    z.route_fuse = 1;
    z.topk = 1;
    z.moe_ix_pp = 1;
    z.eg_cache = 1;
    z.spin_us = 300;
    z.dp4a = 0;
    z.flash = -1;
    g_qwen_opts = z;
}
