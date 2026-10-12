#  Targets:
#    make compile     — Build all solver executables
#    make run         — Run the hybrid_large_beam solver via mpirun
#    make run-pms     — Run Parallel Multi-Strategy Beam Search (BS-PMS) with 20 MPI ranks
#    make run-ich     — Run sequential Beam Search Iterative Construction Heuristic (BS-ICH)
#    make run-bench   — Run the hybrid Lawrence benchmark suite
#    make clean       — Remove compiled binaries
#
#  Overridable variables (pass on command line, e.g. make run-pms NP_PMS=20):
#    ARCH           — GPU compute capability  (default: sm_80)
#    NP             — Number of MPI processes for Hybrid (default: 4)
#    NP_PMS         — Number of MPI processes for BS-PMS (default: 20)
#    OMP_THREADS    — OpenMP threads per rank  (default: 4)
#    INSTANCE       — Instance number or path  (default: 33)
#    BEAM_WIDTH     — Beam width               (default: 5000)
#    ACCEL_FACTOR   — CPU accel factor, 0=auto (default: 0)
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
ACCEL_FACTOR?= 0

# ── Source files → executables ───────────────────────────────────────────────
HYBRID_LB_SRC    := hybrid/v2_twophases/hybrid_large_beam.cu
HYBRID_LB_BIN    := exechybrid_lb

BENCH_HYBRID_SRC := hybrid/v1/benchmark_hybrid.cu
BENCH_HYBRID_BIN := exechybrid_bench

BS_ICH_SRCS      := cpu/v0_ich/main.cpp cpu/v0_ich/solver.cpp cpu/common/instance_parser.cpp
BS_ICH_BIN       := BS-ICH

BS_PMS_SRCS      := cpu/v1_pms/main.cpp cpu/v1_pms/solver.cpp cpu/common/instance_parser.cpp
BS_PMS_BIN       := BS-PMS

BS_PMS_BENCH_SRCS:= cpu/v1_pms/benchmark.cpp cpu/v1_pms/solver.cpp cpu/common/instance_parser.cpp
BS_PMS_BENCH_BIN := BS-PMS-bench

CPU_FEAS_SRCS    := cpu/profiling/gpu_feasibility.cpp cpu/common/instance_parser.cpp
CPU_FEAS_BIN     := exec_cpu_feasibility

CPU_TIME_WIDTH_SRCS := cpu/profiling/time_width_study.cpp cpu/v1_pms/solver.cpp cpu/common/instance_parser.cpp
CPU_TIME_WIDTH_BIN  := exec_cpu_time_width

GPU_V1_SRCS      := gpu/v1/main.cu gpu/v1/solver.cu cpu/common/instance_parser.cpp
GPU_V1_BIN       := exec_gpu_v1

GPU_V1_BENCH_SRCS:= gpu/v1/benchmark.cu gpu/v1/solver.cu cpu/common/instance_parser.cpp
GPU_V1_BENCH_BIN := exec_gpu_v1_bench

GPU_V2_SRCS      := gpu/v2_twophases/main.cu gpu/v2_twophases/solver.cu cpu/common/instance_parser.cpp
GPU_V2_BIN       := exec_gpu_v2

GPU_V2_BENCH_SRCS:= gpu/v2_twophases/benchmark.cu gpu/v2_twophases/solver.cu cpu/common/instance_parser.cpp
GPU_V2_BENCH_BIN := exec_gpu_v2_bench

# =============================================================================
#  Phony targets
# =============================================================================
.PHONY: compile run run-pms run-ich run-bench run-pms-bench run-feasibility run-time-width run-gpu-v1 run-gpu-v1-bench run-gpu-v2 run-gpu-v2-bench clean help

# ── compile: build all executables ───────────────────────────────────────────
compile: $(HYBRID_LB_BIN) $(BENCH_HYBRID_BIN) $(BS_ICH_BIN) $(BS_PMS_BIN) $(BS_PMS_BENCH_BIN) $(CPU_FEAS_BIN) $(CPU_TIME_WIDTH_BIN) $(GPU_V1_BIN) $(GPU_V1_BENCH_BIN) $(GPU_V2_BIN) $(GPU_V2_BENCH_BIN)
	@echo ""
	@echo "  ✓  Build complete."
	@echo "      $(HYBRID_LB_BIN)      — single-instance hybrid solver"
	@echo "      $(BENCH_HYBRID_BIN)  — Lawrence benchmark suite"
	@echo "      $(BS_ICH_BIN)             — Sequential BS-ICH solver"
	@echo "      $(BS_PMS_BIN)             — Parallel Multi-Strategy (BS-PMS) solver"
	@echo "      $(BS_PMS_BENCH_BIN)       — BS-PMS Lawrence benchmark suite"
	@echo "      $(CPU_FEAS_BIN)   — CPU feasibility profiling"
	@echo "      $(CPU_TIME_WIDTH_BIN)  — CPU time vs beam width study"
	@echo "      $(GPU_V1_BIN)          — GPU v1 single-instance solver"
	@echo "      $(GPU_V1_BENCH_BIN)    — GPU v1 Lawrence benchmark suite"
	@echo "      $(GPU_V2_BIN)          — GPU v2 two-phase solver"
	@echo "      $(GPU_V2_BENCH_BIN)    — GPU v2 Lawrence benchmark suite"
	@echo ""

$(HYBRID_LB_BIN): $(HYBRID_LB_SRC)
	$(NVCC) $(NVCC_FLAGS) $(OMP_FLAGS) -o $@ $< $(LINK_LIBS)

$(BENCH_HYBRID_BIN): $(BENCH_HYBRID_SRC)
	$(NVCC) $(NVCC_FLAGS) $(OMP_FLAGS) -o $@ $< $(LINK_LIBS)

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

$(GPU_V1_BIN): $(GPU_V1_SRCS)
	$(NVCC) $(NVCC_FLAGS) -Icpu/common -Igpu/common -o $@ $^

$(GPU_V1_BENCH_BIN): $(GPU_V1_BENCH_SRCS)
	$(NVCC) $(NVCC_FLAGS) -Icpu/common -Igpu/common -o $@ $^

$(GPU_V2_BIN): $(GPU_V2_SRCS)
	$(NVCC) $(NVCC_FLAGS) -Icpu/common -Igpu/common -o $@ $^

$(GPU_V2_BENCH_BIN): $(GPU_V2_BENCH_SRCS)
	$(NVCC) $(NVCC_FLAGS) -Icpu/common -Igpu/common -o $@ $^

# ── run: launch the hybrid_large_beam solver ─────────────────────────────────
run: $(HYBRID_LB_BIN)
	@echo ""
	@echo "  ▶  Launching Hybrid CPU+GPU solver"
	@echo "     Instance : $(INSTANCE)  |  Beam : $(BEAM_WIDTH)  |  Ranks : $(NP)"
	@echo "     Accel    : $(ACCEL_FACTOR) (0=auto)  |  Time limit : $(TIME_LIMIT)s"
	@echo ""
	OMP_NUM_THREADS=$(OMP_THREADS) OMP_PLACES=cores OMP_PROC_BIND=close \
	$(MPIRUN) -n $(NP) ./$(HYBRID_LB_BIN) $(INSTANCE) $(BEAM_WIDTH) $(ACCEL_FACTOR)
# ── run-pms: launch BS-PMS with 20 MPI ranks ─────────────────────────────────
run-pms: $(BS_PMS_BIN)
	@echo ""
	@echo "  ▶  Launching BS-PMS (Parallel Multi-Strategy Beam Search) with $(NP_PMS) MPI ranks"
	@echo "     Instance : $(INSTANCE)  |  Beam : $(BEAM_WIDTH)  |  Ranks : $(NP_PMS)"
	@echo ""
	$(MPIRUN) -n $(NP_PMS) ./$(BS_PMS_BIN) $(INSTANCE) $(BEAM_WIDTH)

# ── run-ich: launch BS-ICH (sequential) ──────────────────────────────────────
run-ich: $(BS_ICH_BIN)
	@echo ""
	@echo "  ▶  Launching BS-ICH (Sequential Beam Search Iterative Construction Heuristic)"
	@echo "     Instance : $(INSTANCE)  |  Beam : $(BEAM_WIDTH)"
	@echo ""
	./$(BS_ICH_BIN) $(INSTANCE) $(BEAM_WIDTH)

# ── run-bench: launch the Lawrence benchmark suite ───────────────────────────
run-bench: $(BENCH_HYBRID_BIN)
	@echo ""
	@echo "  ▶  Launching hybrid Lawrence benchmark ($(NP) MPI ranks)"
	@echo ""
	OMP_NUM_THREADS=$(OMP_THREADS) OMP_PLACES=cores OMP_PROC_BIND=close \
	$(MPIRUN) -n $(NP) ./$(BENCH_HYBRID_BIN)

# ── run-pms-bench: launch the BS-PMS Lawrence benchmark suite ────────────────
run-pms-bench: $(BS_PMS_BENCH_BIN)
	@echo ""
	@echo "  ▶  Launching BS-PMS Lawrence benchmark ($(NP_PMS) MPI ranks)"
	@echo ""
	$(MPIRUN) -n $(NP_PMS) ./$(BS_PMS_BENCH_BIN)

# ── run-feasibility: launch CPU feasibility study ────────────────────────────
run-feasibility: $(CPU_FEAS_BIN)
	@echo ""
	@echo "  ▶  Launching CPU feasibility study on instance $(INSTANCE)"
	@echo ""
	$(MPIRUN) -n $(NP_PMS) ./$(CPU_FEAS_BIN) $(INSTANCE)

# ── run-time-width: launch CPU time vs beam width study ──────────────────────
run-time-width: $(CPU_TIME_WIDTH_BIN)
	@echo ""
	@echo "  ▶  Launching CPU time vs beam width study"
	@echo ""
	$(MPIRUN) -n $(NP_PMS) ./$(CPU_TIME_WIDTH_BIN)

# ── run-gpu-v1: launch GPU v1 single-instance solver ─────────────────────────
run-gpu-v1: $(GPU_V1_BIN)
	@echo ""
	@echo "  ▶  Launching GPU v1 solver"
	@echo "     Instance : $(INSTANCE)  |  Beam : $(BEAM_WIDTH)"
	@echo ""
	./$(GPU_V1_BIN) $(INSTANCE) $(BEAM_WIDTH)

# ── run-gpu-v1-bench: launch GPU v1 Lawrence benchmark suite ─────────────────
run-gpu-v1-bench: $(GPU_V1_BENCH_BIN)
	@echo ""
	@echo "  ▶  Launching GPU v1 Lawrence benchmark suite"
	@echo ""
	./$(GPU_V1_BENCH_BIN)

# ── run-gpu-v2: launch GPU v2 two-phase solver ───────────────────────────────
run-gpu-v2: $(GPU_V2_BIN)
	@echo ""
	@echo "  ▶  Launching GPU v2 two-phase solver"
	@echo "     Instance : $(INSTANCE)  |  Beam : $(BEAM_WIDTH)"
	@echo ""
	./$(GPU_V2_BIN) $(INSTANCE) $(BEAM_WIDTH)

# ── run-gpu-v2-bench: launch GPU v2 Lawrence benchmark suite ─────────────────
run-gpu-v2-bench: $(GPU_V2_BENCH_BIN)
	@echo ""
	@echo "  ▶  Launching GPU v2 Lawrence benchmark suite"
	@echo ""
	./$(GPU_V2_BENCH_BIN)

# ── clean: remove built binaries ─────────────────────────────────────────────
clean:
	rm -f $(HYBRID_LB_BIN) $(BENCH_HYBRID_BIN) $(BS_ICH_BIN) $(BS_PMS_BIN) $(BS_PMS_BENCH_BIN) $(CPU_FEAS_BIN) $(CPU_TIME_WIDTH_BIN) $(GPU_V1_BIN) $(GPU_V1_BENCH_BIN) $(GPU_V2_BIN) $(GPU_V2_BENCH_BIN)
	@echo "  ✓  Cleaned."

# ── help: print usage information ────────────────────────────────────────────
help:
	@echo ""
	@echo "  Usage:"
	@echo "    make compile                  Build all executables"
	@echo "    make run [VAR=val ...]        Run hybrid_large_beam solver"
	@echo "    make run-pms [VAR=val ...]    Run BS-PMS solver (20 MPI processes)"
	@echo "    make run-ich [VAR=val ...]    Run BS-ICH solver (sequential)"
	@echo "    make run-bench                Run hybrid Lawrence benchmark suite"
	@echo "    make run-pms-bench            Run BS-PMS Lawrence benchmark suite"
	@echo "    make run-feasibility          Run CPU feasibility profiling study"
	@echo "    make run-time-width           Run CPU time vs beam width study"
	@echo "    make run-gpu-v1               Run GPU v1 solver"
	@echo "    make run-gpu-v1-bench         Run GPU v1 Lawrence benchmark suite"
	@echo "    make run-gpu-v2               Run GPU v2 two-phase solver"
	@echo "    make run-gpu-v2-bench         Run GPU v2 Lawrence benchmark suite"
	@echo "    make clean                    Remove binaries"
	@echo ""
	@echo "  Overridable variables:"
	@echo "    ARCH          GPU arch         (default: sm_80)"
	@echo "    NP            MPI ranks hybrid (default: 4)"
	@echo "    NP_PMS        MPI ranks BS-PMS (default: 20)"
	@echo "    INSTANCE      Instance number  (default: 33)"
	@echo "    BEAM_WIDTH    GPU beam width   (default: 5000)"
	@echo ""
