# qwen3.c — Qwen3 35B-A3B MoE on Vulkan iGPU (standalone)
# From: Colibrì (Apache-2.0) qwen engine + Vulkan backend extract
#
#   make qwen VK=1          # Linux / MSYS2
#   make qwen VK=1 ARCH=native
#
# Windows MinGW (no make): use ../scripts/build_win.bat

DETECT_CC := $(if $(filter default,$(origin CC)),gcc,$(CC))
TRIPLET   := $(shell $(DETECT_CC) -dumpmachine 2>/dev/null)
MINGW     := $(findstring mingw,$(TRIPLET))
CYGWIN    := $(findstring cygwin,$(TRIPLET))
IS_WIN    := $(MINGW)$(CYGWIN)
ifeq ($(TRIPLET),)
IS_WIN := $(if $(filter Windows_NT,$(OS)),1,)
endif

ifneq ($(IS_WIN),)
CC      ?= gcc
ARCH    ?= native
CFLAGS   = -D_FILE_OFFSET_BITS=64 -O2 -std=c11 -march=$(ARCH) -Wall -Wno-unused-parameter -Wno-unused-function
LDFLAGS  = -lm -lpsapi
EXE      = .exe
else
CC      ?= gcc
ARCH    ?= native
CFLAGS   = -O2 -std=c11 -march=$(ARCH) -Wall -Wno-unused-parameter -Wno-unused-function
LDFLAGS  = -lm
EXE      =
endif

VK       ?= 0
VK_OBJ    =
VK_SPV    =
GLSLC    ?= glslc
GV       ?= glslangValidator

SPV_LIST = qmatmul qmatmul_gate_up \
           rmsnorm gdn_delta gqa_attn rmsnorm_zc rope_partial conv1d \
           gdn_conv gdn_delta_cv gqa_qkv \
           qmatmul_dp4a qmatmul_gate_up_dp4a quant_rows qmatmul_down_dp4a \
           argmax residual_rmsnorm_zc \
           replicate_rows moe_accumulate gdn_pack_params \
           softmax_topk moe_indexed moe_pack_w

ifeq ($(VK),1)
CFLAGS  += -DCOLI_VULKAN
LDFLAGS += -lvulkan
VK_OBJ   = backend_vulkan.o
VK_SPV   = $(addprefix shaders/,$(addsuffix .spv,$(SPV_LIST)))
endif

.PHONY: all qwen shaders clean

all: qwen$(EXE)

qwen: qwen$(EXE)

qwen$(EXE): qwen.c qwen_opts.o st.h json.h compat.h tok.h tok_unicode.h tok_unicode_o200k.h qwen_opts.h $(VK_OBJ) $(VK_SPV)
	$(CC) $(CFLAGS) -I. qwen.c qwen_opts.o $(VK_OBJ) -o $@ $(LDFLAGS)

qwen_opts.o: qwen_opts.c qwen_opts.h
	$(CC) $(CFLAGS) -I. -c qwen_opts.c -o $@

backend_vulkan.o: backend_vulkan.c backend_vulkan.h qwen_opts.h
	$(CC) $(CFLAGS) -I. -c backend_vulkan.c -o $@

shaders/%.spv: shaders/%.comp
	@command -v $(GLSLC) >/dev/null 2>&1 && $(GLSLC) -fshader-stage=compute --target-env=vulkan1.2 $< -o $@ || \
	 (command -v $(GV) >/dev/null 2>&1 && $(GV) --target-env vulkan1.2 -V $< -o $@) || \
	 { echo "need glslc or glslangValidator to build $<" >&2; exit 1; }

shaders: $(VK_SPV)

clean:
	rm -f qwen$(EXE) qwen_opts.o backend_vulkan.o shaders/*.spv
