# cuda-lab -- build for whatever card the pod happens to hand you.
#
#   make            build everything
#   make ARCH=90a   force an arch (e.g. compile Hopper code on the A4000)
#   make ptx K=01_vecadd    dump PTX
#   make sass K=01_vecadd   dump SASS

# Detect compute capability from the live GPU; fall back to 86 (A4000/A5000/3090)
# so the tree still compiles on a machine with no GPU attached.
ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
          | head -1 | tr -d '.' || echo 86)
ifeq ($(strip $(ARCH)),)
  ARCH := 86
endif

NVCC      := nvcc
# -lineinfo is mandatory: without it ncu cannot map counters back to your source.
# -Xptxas -v prints registers/thread and smem/block -- the inputs to occupancy.
NVCCFLAGS := -std=c++17 -O3 -arch=sm_$(ARCH) -lineinfo -Xptxas -v \
             --expt-relaxed-constexpr -Icommon

KERNEL_DIRS := $(sort $(notdir $(patsubst %/main.cu,%,$(wildcard kernels/*/main.cu))))
BINS        := $(addprefix bin/,$(KERNEL_DIRS))

.PHONY: all clean list
all: $(BINS)

bin/%: kernels/%/main.cu common/check.cuh common/harness.cuh
	@mkdir -p bin results
	$(NVCC) $(NVCCFLAGS) $< -o $@

list:
	@echo "arch: sm_$(ARCH)"
	@echo "kernels: $(KERNEL_DIRS)"

# --- inspection targets: these need NO GPU, run them locally ----------------
ptx:
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -ptx kernels/$(K)/main.cu -o build/$(K).ptx
	@echo "-> build/$(K).ptx"

sass:
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -cubin kernels/$(K)/main.cu -o build/$(K).cubin
	cuobjdump -sass build/$(K).cubin > build/$(K).sass
	@echo "-> build/$(K).sass"

regs:
	@$(NVCC) $(NVCCFLAGS) -c kernels/$(K)/main.cu -o /dev/null 2>&1 | grep -E 'ptxas info|registers|smem'

clean:
	rm -rf bin build
