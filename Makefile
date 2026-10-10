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

BS_ICH_SRC       := cpu/v0/BS-ICH.cpp
BS_ICH_BIN       := BS-ICH

BS_PMS_SRC       := cpu/v0/BS-PMS.cpp
BS_PMS_BIN       := BS-PMS

# =============================================================================
#  Phony targets
# =============================================================================
.PHONY: compile run run-pms run-ich run-bench clean help

# ── compile: build all executables ───────────────────────────────────────────
compile: $(HYBRID_LB_BIN) $(BENCH_HYBRID_BIN) $(BS_ICH_BIN) $(BS_PMS_BIN)
	@echo ""
	@echo "  ✓  Build complete."
	@echo "      $(HYBRID_LB_BIN)      — single-instance hybrid solver"
	@echo "      $(BENCH_HYBRID_BIN)  — Lawrence benchmark suite"
	@echo "      $(BS_ICH_BIN)             — Sequential BS-ICH solver"
	@echo "      $(BS_PMS_BIN)             — Parallel Multi-Strategy (BS-PMS) solver"
	@echo ""

$(HYBRID_LB_BIN): $(HYBRID_LB_SRC)
	$(NVCC) $(NVCC_FLAGS) $(OMP_FLAGS) -o $@ $< $(LINK_LIBS)

$(BENCH_HYBRID_BIN): $(BENCH_HYBRID_SRC)
	$(NVCC) $(NVCC_FLAGS) $(OMP_FLAGS) -o $@ $< $(LINK_LIBS)

$(BS_ICH_BIN): $(BS_ICH_SRC)
	$(CXX) $(CXX_FLAGS) -o $@ $<

$(BS_PMS_BIN): $(BS_PMS_SRC)
	$(MPICXX) $(CXX_FLAGS) -o $@ $<

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

# ── clean: remove built binaries ─────────────────────────────────────────────
clean:
	rm -f $(HYBRID_LB_BIN) $(BENCH_HYBRID_BIN) $(BS_ICH_BIN) $(BS_PMS_BIN)
	@echo "  ✓  Cleaned."

# ── help: print usage information ────────────────────────────────────────────
help:
	@echo ""
	@echo "  Usage:"
	@echo "    make compile                  Build all executables"
	@echo "    make run [VAR=val ...]        Run hybrid_large_beam solver"
	@echo "    make run-pms [VAR=val ...]    Run BS-PMS solver (20 MPI processes)"
	@echo "    make run-ich [VAR=val ...]    Run BS-ICH solver (sequential)"
	@echo "    make run-bench                Run Lawrence benchmark suite"
	@echo "    make clean                    Remove binaries"
	@echo ""
	@echo "  Overridable variables:"
	@echo "    ARCH          GPU arch         (default: sm_80)"
	@echo "    NP            MPI ranks hybrid (default: 4)"
	@echo "    NP_PMS        MPI ranks BS-PMS (default: 20)"
	@echo "    INSTANCE      Instance number  (default: 33)"
	@echo "    BEAM_WIDTH    GPU beam width   (default: 5000)"
	@echo ""
