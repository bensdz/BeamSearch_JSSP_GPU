# =============================================================================
#  Makefile — JSSP Beam Search Solvers (CPU, GPU, Hybrid)
#
#  Targets:
#    make compile            — Build all solver executables
#    make run-ich            — Run sequential BS-ICH solver
#    make run-pms            — Run parallel BS-PMS solver (MPI ranks)
#    make run-pms-bench      — Run BS-PMS Lawrence benchmark suite
#    make run-feasibility    — Run CPU feasibility profiling study
#    make run-time-width     — Run CPU time vs beam width study
#    make run-gpu-v1         — Run single-phase GPU solver
#    make run-gpu-v1-bench   — Run GPU v1 Lawrence benchmark suite
#    make run-gpu-v2         — Run two-phase GPU solver
#    make run-gpu-v2-bench   — Run GPU v2 Lawrence benchmark suite
#    make run-hybrid-v1      — Run hybrid v1 (CPU+GPU) solver
#    make run-hybrid-v1-bench— Run hybrid v1 Lawrence benchmark suite
#    make run-hybrid-v2      — Run hybrid v2 (CPU+GPU two-phase) solver
#    make run-hybrid-v2-bench— Run hybrid v2 Lawrence benchmark suite
#    make clean              — Remove compiled binaries
#
#  Overridable variables:
#    ARCH           — GPU compute capability   (default: sm_80)
#    NP             — MPI processes for Hybrid (default: 4)
#    NP_PMS         — MPI processes for PMS/GPU(default: 20)
#    OMP_THREADS    — OpenMP threads per rank  (default: 4)
#    INSTANCE       — Instance number or path  (default: 33)
#    BEAM_WIDTH     — Beam width               (default: 5000)
#    ACCEL_FACTOR   — CPU accel factor         (default: 34)
# =============================================================================

# ── Toolchain ────────────────────────────────────────────────────────────────
CXX         := g++
MPICXX      := mpicxx
NVCC        := nvcc
MPIRUN      := mpirun

# ── Architecture & Optimisation ──────────────────────────────────────────────
ARCH        ?= sm_80
CXX_FLAGS   := -O3 -std=c++11
NVCC_FLAGS  := -O3 -arch=$(ARCH) --expt-relaxed-constexpr
OMP_FLAGS   := -Xcompiler -fopenmp
LINK_LIBS   := -lgomp -lmpi

# ── Runtime defaults ──────────────────────────────────────────────────────────
NP          ?= 4
NP_PMS      ?= 20
OMP_THREADS ?= 4
INSTANCE    ?= 33
BEAM_WIDTH  ?= 5000
ACCEL_FACTOR?= 34

# ── Source files → executables ───────────────────────────────────────────────

# CPU solvers
BS_ICH_SRCS          := cpu/v0_ich/main.cpp cpu/v0_ich/solver.cpp cpu/common/instance_parser.cpp
BS_ICH_BIN           := BS-ICH

BS_PMS_SRCS          := cpu/v1_pms/main.cpp cpu/v1_pms/solver.cpp cpu/common/instance_parser.cpp
BS_PMS_BIN           := BS-PMS

BS_PMS_BENCH_SRCS    := cpu/v1_pms/benchmark.cpp cpu/v1_pms/solver.cpp cpu/common/instance_parser.cpp
BS_PMS_BENCH_BIN     := BS-PMS-bench

# CPU profiling tools
CPU_FEAS_SRCS        := cpu/profiling/gpu_feasibility.cpp cpu/common/instance_parser.cpp
CPU_FEAS_BIN         := exec_cpu_feasibility

CPU_TIME_WIDTH_SRCS  := cpu/profiling/time_width_study.cpp cpu/v1_pms/solver.cpp cpu/common/instance_parser.cpp
CPU_TIME_WIDTH_BIN   := exec_cpu_time_width

# GPU solvers
GPU_V1_SRCS          := gpu/v1/main.cu gpu/v1/solver.cu cpu/common/instance_parser.cpp
GPU_V1_BIN           := exec_gpu_v1

GPU_V1_BENCH_SRCS    := gpu/v1/benchmark.cu gpu/v1/solver.cu cpu/common/instance_parser.cpp
GPU_V1_BENCH_BIN     := exec_gpu_v1_bench

GPU_V2_SRCS          := gpu/v2_twophases/main.cu gpu/v2_twophases/solver.cu cpu/common/instance_parser.cpp
GPU_V2_BIN           := exec_gpu_v2

GPU_V2_BENCH_SRCS    := gpu/v2_twophases/benchmark.cu gpu/v2_twophases/solver.cu cpu/common/instance_parser.cpp
GPU_V2_BENCH_BIN     := exec_gpu_v2_bench

# Hybrid solvers
HYBRID_V1_SRCS       := hybrid/v1/main.cu hybrid/v1/solver.cu gpu/v1/solver.cu cpu/v1_pms/solver.cpp cpu/common/instance_parser.cpp
HYBRID_V1_BIN        := exechybrid_v1

HYBRID_V1_BENCH_SRCS := hybrid/v1/benchmark.cu hybrid/v1/solver.cu gpu/v1/solver.cu cpu/v1_pms/solver.cpp cpu/common/instance_parser.cpp
HYBRID_V1_BENCH_BIN  := exechybrid_v1_bench

HYBRID_V2_SRCS       := hybrid/v2_twophases/main.cu hybrid/v2_twophases/solver.cu gpu/v2_twophases/solver.cu cpu/v1_pms/solver.cpp cpu/common/instance_parser.cpp
HYBRID_V2_BIN        := exechybrid_v2

HYBRID_V2_BENCH_SRCS := hybrid/v2_twophases/benchmark.cu hybrid/v2_twophases/solver.cu gpu/v2_twophases/solver.cu cpu/v1_pms/solver.cpp cpu/common/instance_parser.cpp
HYBRID_V2_BENCH_BIN  := exechybrid_v2_bench

# =============================================================================
#  Phony targets
# =============================================================================
.PHONY: compile clean help \
        run-ich run-pms run-pms-bench run-feasibility run-time-width \
        run-gpu-v1 run-gpu-v1-bench run-gpu-v2 run-gpu-v2-bench \
        run-hybrid-v1 run-hybrid-v1-bench run-hybrid-v2 run-hybrid-v2-bench

# ── compile: build all executables ───────────────────────────────────────────
compile: $(BS_ICH_BIN) $(BS_PMS_BIN) $(BS_PMS_BENCH_BIN) \
         $(CPU_FEAS_BIN) $(CPU_TIME_WIDTH_BIN) \
         $(GPU_V1_BIN) $(GPU_V1_BENCH_BIN) $(GPU_V2_BIN) $(GPU_V2_BENCH_BIN) \
         $(HYBRID_V1_BIN) $(HYBRID_V1_BENCH_BIN) $(HYBRID_V2_BIN) $(HYBRID_V2_BENCH_BIN)
	@echo ""
	@echo "  ✓  Build complete."
	@echo "      $(BS_ICH_BIN)            — Sequential BS-ICH solver"
	@echo "      $(BS_PMS_BIN)            — Parallel BS-PMS solver"
	@echo "      $(BS_PMS_BENCH_BIN)      — BS-PMS Lawrence benchmark suite"
	@echo "      $(CPU_FEAS_BIN)  — CPU feasibility profiling"
	@echo "      $(CPU_TIME_WIDTH_BIN) — CPU time vs beam width study"
	@echo "      $(GPU_V1_BIN)         — GPU v1 single-instance solver"
	@echo "      $(GPU_V1_BENCH_BIN)   — GPU v1 Lawrence benchmark suite"
	@echo "      $(GPU_V2_BIN)         — GPU v2 two-phase solver"
	@echo "      $(GPU_V2_BENCH_BIN)   — GPU v2 Lawrence benchmark suite"
	@echo "      $(HYBRID_V1_BIN)       — Hybrid v1 solver"
	@echo "      $(HYBRID_V1_BENCH_BIN) — Hybrid v1 Lawrence benchmark suite"
	@echo "      $(HYBRID_V2_BIN)       — Hybrid v2 two-phase solver"
	@echo "      $(HYBRID_V2_BENCH_BIN) — Hybrid v2 Lawrence benchmark suite"
	@echo ""

# ── CPU build rules ──────────────────────────────────────────────────────────
$(BS_ICH_BIN): $(BS_ICH_SRCS)
	$(CXX) $(CXX_FLAGS) -Icpu/common -o $@ $^

$(BS_PMS_BIN): $(BS_PMS_SRCS)
	$(MPICXX) $(CXX_FLAGS) -Icpu/common -o $@ $^

$(BS_PMS_BENCH_BIN): $(BS_PMS_BENCH_SRCS)
	$(MPICXX) $(CXX_FLAGS) -Icpu/common -o $@ $^

$(CPU_FEAS_BIN): $(CPU_FEAS_SRCS)
	$(MPICXX) $(CXX_FLAGS) -Icpu/common -o $@ $^

$(CPU_TIME_WIDTH_BIN): $(CPU_TIME_WIDTH_SRCS)
	$(MPICXX) $(CXX_FLAGS) -Icpu/common -Icpu/v1_pms -o $@ $^

# ── GPU build rules ──────────────────────────────────────────────────────────
$(GPU_V1_BIN): $(GPU_V1_SRCS)
	$(NVCC) $(NVCC_FLAGS) -Icpu/common -Igpu/common -o $@ $^ -lmpi

$(GPU_V1_BENCH_BIN): $(GPU_V1_BENCH_SRCS)
	$(NVCC) $(NVCC_FLAGS) -Icpu/common -Igpu/common -o $@ $^ -lmpi

$(GPU_V2_BIN): $(GPU_V2_SRCS)
	$(NVCC) $(NVCC_FLAGS) -Icpu/common -Igpu/common -o $@ $^ -lmpi

$(GPU_V2_BENCH_BIN): $(GPU_V2_BENCH_SRCS)
	$(NVCC) $(NVCC_FLAGS) -Icpu/common -Igpu/common -o $@ $^ -lmpi

# ── Hybrid build rules ───────────────────────────────────────────────────────
$(HYBRID_V1_BIN): $(HYBRID_V1_SRCS)
	$(NVCC) $(NVCC_FLAGS) $(OMP_FLAGS) -Icpu/common -Icpu/v1_pms -Igpu/common -Igpu/v1 -o $@ $^ $(LINK_LIBS)

$(HYBRID_V1_BENCH_BIN): $(HYBRID_V1_BENCH_SRCS)
	$(NVCC) $(NVCC_FLAGS) $(OMP_FLAGS) -Icpu/common -Icpu/v1_pms -Igpu/common -Igpu/v1 -o $@ $^ $(LINK_LIBS)

$(HYBRID_V2_BIN): $(HYBRID_V2_SRCS)
	$(NVCC) $(NVCC_FLAGS) $(OMP_FLAGS) -Icpu/common -Icpu/v1_pms -Igpu/common -Igpu/v2_twophases -o $@ $^ $(LINK_LIBS)

$(HYBRID_V2_BENCH_BIN): $(HYBRID_V2_BENCH_SRCS)
	$(NVCC) $(NVCC_FLAGS) $(OMP_FLAGS) -Icpu/common -Icpu/v1_pms -Igpu/common -Igpu/v2_twophases -o $@ $^ $(LINK_LIBS)

# ── Execution targets ────────────────────────────────────────────────────────
run-ich: $(BS_ICH_BIN)
	./$(BS_ICH_BIN) $(INSTANCE) $(BEAM_WIDTH)

run-pms: $(BS_PMS_BIN)
	$(MPIRUN) -n $(NP_PMS) ./$(BS_PMS_BIN) $(INSTANCE) $(BEAM_WIDTH)

run-pms-bench: $(BS_PMS_BENCH_BIN)
	$(MPIRUN) -n $(NP_PMS) ./$(BS_PMS_BENCH_BIN)

run-feasibility: $(CPU_FEAS_BIN)
	$(MPIRUN) -n $(NP_PMS) ./$(CPU_FEAS_BIN) $(INSTANCE)

run-time-width: $(CPU_TIME_WIDTH_BIN)
	$(MPIRUN) -n $(NP_PMS) ./$(CPU_TIME_WIDTH_BIN)

run-gpu-v1: $(GPU_V1_BIN)
	$(MPIRUN) -n $(NP) ./$(GPU_V1_BIN) $(INSTANCE) $(BEAM_WIDTH)

run-gpu-v1-bench: $(GPU_V1_BENCH_BIN)
	$(MPIRUN) -n $(NP) ./$(GPU_V1_BENCH_BIN)

run-gpu-v2: $(GPU_V2_BIN)
	$(MPIRUN) -n $(NP) ./$(GPU_V2_BIN) $(INSTANCE) $(BEAM_WIDTH)

run-gpu-v2-bench: $(GPU_V2_BENCH_BIN)
	$(MPIRUN) -n $(NP) ./$(GPU_V2_BENCH_BIN)

run-hybrid-v1: $(HYBRID_V1_BIN)
	OMP_NUM_THREADS=$(OMP_THREADS) OMP_PLACES=cores OMP_PROC_BIND=close \
	$(MPIRUN) -n $(NP) ./$(HYBRID_V1_BIN) $(INSTANCE) $(BEAM_WIDTH) $(ACCEL_FACTOR)

run-hybrid-v1-bench: $(HYBRID_V1_BENCH_BIN)
	OMP_NUM_THREADS=$(OMP_THREADS) OMP_PLACES=cores OMP_PROC_BIND=close \
	$(MPIRUN) -n $(NP) ./$(HYBRID_V1_BENCH_BIN)

run-hybrid-v2: $(HYBRID_V2_BIN)
	OMP_NUM_THREADS=$(OMP_THREADS) OMP_PLACES=cores OMP_PROC_BIND=close \
	$(MPIRUN) -n $(NP) ./$(HYBRID_V2_BIN) $(INSTANCE) $(BEAM_WIDTH) $(ACCEL_FACTOR)

run-hybrid-v2-bench: $(HYBRID_V2_BENCH_BIN)
	OMP_NUM_THREADS=$(OMP_THREADS) OMP_PLACES=cores OMP_PROC_BIND=close \
	$(MPIRUN) -n $(NP) ./$(HYBRID_V2_BENCH_BIN)

# ── clean: remove built binaries ─────────────────────────────────────────────
clean:
	rm -f $(BS_ICH_BIN) $(BS_PMS_BIN) $(BS_PMS_BENCH_BIN) \
	      $(CPU_FEAS_BIN) $(CPU_TIME_WIDTH_BIN) \
	      $(GPU_V1_BIN) $(GPU_V1_BENCH_BIN) $(GPU_V2_BIN) $(GPU_V2_BENCH_BIN) \
	      $(HYBRID_V1_BIN) $(HYBRID_V1_BENCH_BIN) $(HYBRID_V2_BIN) $(HYBRID_V2_BENCH_BIN)
	@echo "  ✓  Cleaned."

# ── help: print usage information ────────────────────────────────────────────
help:
	@echo ""
	@echo "  Usage:"
	@echo "    make compile                  Build all executables"
	@echo "    make run-ich                  Run BS-ICH solver (sequential)"
	@echo "    make run-pms                  Run BS-PMS solver (MPI)"
	@echo "    make run-pms-bench            Run BS-PMS Lawrence benchmark suite"
	@echo "    make run-feasibility          Run CPU feasibility profiling study"
	@echo "    make run-time-width           Run CPU time vs beam width study"
	@echo "    make run-gpu-v1               Run GPU v1 solver"
	@echo "    make run-gpu-v1-bench         Run GPU v1 Lawrence benchmark suite"
	@echo "    make run-gpu-v2               Run GPU v2 two-phase solver"
	@echo "    make run-gpu-v2-bench         Run GPU v2 Lawrence benchmark suite"
	@echo "    make run-hybrid-v1            Run Hybrid v1 solver"
	@echo "    make run-hybrid-v1-bench      Run Hybrid v1 Lawrence benchmark suite"
	@echo "    make run-hybrid-v2            Run Hybrid v2 two-phase solver"
	@echo "    make run-hybrid-v2-bench      Run Hybrid v2 Lawrence benchmark suite"
	@echo "    make clean                    Remove binaries"
	@echo ""
