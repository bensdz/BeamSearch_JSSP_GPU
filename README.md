# GPU-Accelerated Multi-Strategy Beam Search for the Blocking Job-Shop Scheduling Problem (BJSSP)

This repository contains a high-performance, parallelized framework for solving the **Blocking Job-Shop Scheduling Problem (BJSSP)**. The solver combines **Beam Search Iterative Construction Heuristics (BS-ICH)** with **Parallel Multi-Strategy (PMS-BS)** search to find near-optimal schedules. It is accelerated using NVIDIA GPU hardware (via CUDA) and distributed across multi-GPU/multi-node clusters (via MPI and OpenMP).

---

## Table of Contents

1. [Problem Context: BJSSP](#problem-context-bjssp)
2. [Algorithmic Core](#algorithmic-core)
3. [Reorganized Project Structure](#reorganized-project-structure)
4. [Walkthrough of Solver Architectures & Versions](#walkthrough-of-solver-architectures--versions)
   - [CPU Solvers (`cpu/`)](#1-cpu-solvers-cpuv0-and-cpuprofiling)
   - [GPU Solvers (`gpu/`)](#2-gpu-solvers-gpuv1-and-gpuv2_twophases)
   - [Hybrid CPU-GPU Solvers (`hybrid/`)](#3-hybrid-cpu-gpu-solvers-hybridv1-and-hybridv2_twophases)
   - [Profiling & Feasibility Studies](#4-profiling--feasibility-studies)
5. [Compilation Guide](#compilation-guide)
   - [Using Makefile](#1-using-makefile-recommended)
   - [Manual Compilation Commands](#2-manual-compilation-commands)
6. [Execution Guide](#execution-guide)
   - [Command-Line Parameters](#command-line-parameters)
   - [Using Makefile Shortcuts](#using-makefile-shortcuts)
   - [Interactive Execution](#interactive-execution)
   - [Using Makefile Shortcuts](#using-makefile-shortcuts)
   - [Slurm Cluster Submission](#slurm-cluster-submission)
7. [Benchmark Instances & Results](#benchmark-instances--results)

---

## Problem Context: BJSSP

The **Blocking Job-Shop Scheduling Problem (BJSSP)** is an extension of the classical Job-Shop Scheduling Problem (JSSP). The critical difference is the **Blocking Constraint**:

- There are **no intermediate buffers/storage** between machines.
- Once an operation finishes processing, it cannot move to a buffer. It must remain on its current machine, **blocking** it, until the next machine required in the job sequence becomes free.
- This lack of buffers leads to **deadlock cycles** (circular wait states between jobs).
- The primary objective is to minimize the **Makespan ($C_{\max}$)**, which is the total time required to complete all operations.

Our solvers resolve deadlocks dynamically by detecting circular wait chains and scheduling the operations involved in a deadlock simultaneously (referred to as **cycle swapping** or `check_swap`).

---

## Algorithmic Core

Our framework provides CPU, GPU, and Hybrid algorithms based on constructive search and search space pruning:

1. **Beam Search Iterative Construction Heuristic (BS-ICH)**:
   - Builds the schedule operation-by-operation.
   - At each step, it identifies candidate operations to schedule.
   - Instead of exploring all branches, it only keeps the top-$K$ best candidate schedules (where $K$ is the **Beam Width**) and discards the rest.
2. **Parallel Multi-Strategy Beam Search (BS-PMS / PMS-BS)**:
   - Diversifies the search by running multiple independent search instances (MPI ranks) concurrently.
   - Each rank uses a unique seed and priority bias to explore different regions of the scheduling search space.
   - Real-time bound sharing via `MPI_Allreduce` (and thread-level lock-free atomics in hybrid modes) allows ranks to prune unpromising branches early if they exceed the best makespan found by any rank.

---

## Reorganized Project Structure

The repository is modularly organized into dedicated directories for CPU baselines, GPU implementations, hybrid solvers, instances, and benchmark logs:

```text
GPU_BS_JSSP/
├── README.md                          # Project documentation
├── Makefile                           # Unified build and execution shortcuts
├── submit_cpu.sh                      # Slurm cluster script for CPU solver (BS-PMS)
├── submit_gpu.sh                      # Slurm cluster script for GPU solver (v1)
├── submit_hybrid.sh                   # Slurm cluster script for Hybrid solver (v1 / v2)
│
├── cpu/                               # CPU implementations & profiling
│   ├── v0/
│   │   ├── BS-ICH.cpp                 # Sequential Beam Search Iterative Construction Heuristic
│   │   ├── BS-PMS.cpp                 # Parallel Multi-Strategy Beam Search (MPI)
│   │   └── benchmark_cpu.cpp          # CPU benchmark runner for Lawrence instances
│   └── profiling/
│       ├── gpu_feasability_profiling.cpp # Amdahl's Law CPU profiler (parallel fraction)
│       └── time_width_study.cpp       # Beam width scaling analysis (varying K)
│
├── gpu/                               # Pure GPU implementations (CUDA + MPI)
│   ├── v1/
│   │   ├── gpu-version.cu             # Baseline GPU solver (materializes full child states)
│   │   └── benchmark.cu               # GPU v1 benchmark runner (la01 - la40)
│   └── v2_twophases/
│       ├── gpu_version_large_beam.cu  # Memory-optimized Two-Phase expansion (K >= 25600)
│       └── benchmark_lbversion.cu     # GPU v2 benchmark runner (Taillard instances)
│
├── hybrid/                            # Concurrent CPU + GPU co-processing (CUDA + OpenMP + MPI)
│   ├── v1/
│   │   ├── hybrid_version.cu          # Concurrent CPU+GPU solver with atomic bound sharing
│   │   ├── hybrid_withstats.cu        # Hybrid solver with per-rank throughput & multi-LB scoring
│   │   └── benchmark_hybrid.cu        # Hybrid v1 benchmark runner for Lawrence instances
│   └── v2_twophases/
│       ├── hybrid_large_beam.cu       # Two-phase GPU expansion + CPU search for huge beams
│       └── benchmark_hybrid_lb.cu     # Two-phase hybrid benchmark runner (Taillard instances)
│
├── lawrence_instances/                # Lawrence (1984) benchmark instances
│   └── la01.txt ... la40.txt
│
├── taillard_instances/                # Taillard (1993) benchmark instances
│   └── ta01 ... ta70, ta71.txt ... ta80.txt
│
└── benchmark_results/                 # Pre-computed benchmark results & profiling logs
    ├── beam_width_study.txt           # Scaling study across varying beam widths
    ├── benchmarkv3.txt                # Large beam GPU results on Taillard (ta01-ta80)
    ├── benchmarkhybrid*.txt           # Hybrid CPU-GPU benchmark logs
    ├── two-phase-hybrid-results.txt   # Two-phase hybrid solver benchmark results
    ├── feasability_gpu.txt            # Feasibility profiling outputs
    ├── intra_kernel_profiling.txt     # GPU kernel execution profiling breakdown
    ├── la33CPU_K25600.txt             # CPU benchmark run on la33 (K=25600)
    └── ta80CPU25600.txt               # CPU benchmark run on ta80 (K=25600)
```

---

## Walkthrough of Solver Architectures & Versions

### 1. CPU Solvers (`cpu/v0` and `cpu/profiling`)

- **[BS-ICH.cpp](./cpu/v0/BS-ICH.cpp)**:
  - Sequential C++ implementation of the Beam Search Iterative Construction Heuristic.
  - Constructs schedules operation-by-operation using a priority queue bounded to beam width $K$.
  - Generates reproducible, single-threaded baseline solutions with deterministic seeding.
- **[BS-PMS.cpp](./cpu/v0/BS-PMS.cpp)**:
  - Distributed CPU Parallel Multi-Strategy Beam Search using MPI.
  - Multiple MPI ranks execute the constructive heuristic concurrently with distinct seeds and exploration strategies.
  - Periodic `MPI_Allreduce` bound sharing prunes unpromising search trees across ranks.
- **[benchmark_cpu.cpp](./cpu/v0/benchmark_cpu.cpp)**:
  - Automated test harness running the CPU solver across the Lawrence dataset (`la01`–`la40`).

### 2. GPU Solvers (`gpu/v1` and `gpu/v2_twophases`)

#### Version 1: Standard GPU Solver (`gpu/v1/`)

- **Files**: [gpu-version.cu](./gpu/v1/gpu-version.cu) / [benchmark.cu](./gpu/v1/benchmark.cu)
- **Mechanism**:
  - Serializes the scheduling state (`ScheduleSnapshot`) into flat integer arrays to maximize coalesced GPU memory access.
  - For beam width $K$, parent states are copied to device global memory.
  - The GPU generates all valid child operations (up to $\beta$ branches per parent) and writes **full child states** into VRAM ($K \times \beta$ states materialized).
  - Uses **NVIDIA CUB Radix Sort** (`cub::DeviceRadixSort::SortPairs`) to rank child candidates by evaluation score.
  - Top-$K$ states are gathered via `kernel_gather_topk` to form the beam for the next level.
  - _Best for_: Small to moderate beam widths ($K \le 5000$).

#### Version 2: Large Beam Two-Phase Solver (`gpu/v2_twophases/`)

- **Files**: [gpu_version_large_beam.cu](./gpu/v2_twophases/gpu_version_large_beam.cu) / [benchmark_lbversion.cu](./gpu/v2_twophases/benchmark_lbversion.cu)
- **Motivation**:
  - Materializing all $K \times \beta$ complete child states causes GPU out-of-memory errors for large beam widths ($K \ge 25600$) or large instances (e.g., Taillard $100 \times 20$).
- **Two-Phase Solution**:
  - **Phase 1: Scoring (`kernel_score_children`)**: Threads simulate transitions in GPU registers/shared memory and output _only_ a 4-byte float score and child index to VRAM.
  - **Radix Sort**: CUB Radix Sort operates solely on the compact score array to identify the top-$K$ indices.
  - **Phase 2: Targeted Materialization (`kernel_materialize_topk`)**: Only the surviving top-$K$ candidates are materialized into full `ScheduleSnapshot` buffers for the next iteration.
  - _Result_: Achieves a **$10\times$ to $30\times$ reduction in global memory footprint**, unlocking beam widths up to $K = 50000+$.

### 3. Hybrid CPU-GPU Solvers (`hybrid/v1` and `hybrid/v2_twophases`)

#### Hybrid Version 1 (`hybrid/v1/`)

- **Files**: [hybrid_version.cu](./hybrid/v1/hybrid_version.cu) / [hybrid_withstats.cu](./hybrid/v1/hybrid_withstats.cu) / [benchmark_hybrid.cu](./hybrid/v1/benchmark_hybrid.cu)
- **Mechanism**:
  - Each MPI rank launches an asynchronous CPU thread (`std::thread` / OpenMP) alongside the CUDA stream.
  - The GPU processes a large beam width $K_{\text{gpu}}$, while the CPU processes a smaller beam width $K_{\text{cpu}} = K_{\text{gpu}} / \text{accel\_factor}$.
  - CPU and GPU workers continuously exchange progress through a lock-free `std::atomic<int> g_shared_best` makespan bound, pruning unpromising paths in real time.
  - `hybrid_withstats.cu` adds throughput counters and multi-strategy Lower-Bound (LB) scoring modes:
    - Mode 0 (`LB_CUR`): $C_{\max} / \text{num\_scheduled}$ (baseline)
    - Mode 1 (`LB_M`): Machine-based lower bound
    - Mode 2 (`LB_MJ`): Machine + Job lower bound
    - Mode 3 (`LB_HT`): Carlier Head-Tail bound

#### Hybrid Version 2: Two-Phase Large Beam (`hybrid/v2_twophases/`)

- **Files**: [hybrid_large_beam.cu](./hybrid/v2_twophases/hybrid_large_beam.cu) / [benchmark_hybrid_lb.cu](./hybrid/v2_twophases/benchmark_hybrid_lb.cu)
- **Mechanism**:
  - Unifies the memory-efficient **Two-Phase GPU expansion** with the concurrent CPU search.
  - Supports automatic dynamic calibration of the CPU beam width based on host machine performance (`accel_factor = 0`).
  - Capable of solving massive problem instances (e.g., `ta71`–`ta80`) with huge beam widths on multi-node HPC clusters.

### 4. Profiling & Feasibility Studies

- **[gpu_feasability_profiling.cpp](./cpu/profiling/gpu_feasability_profiling.cpp)**:
  - Instruments the CPU execution of BS-ICH with high-resolution timers.
  - Measures the candidate expansion phase vs. sequential steps (sorting, MPI overhead).
  - Demonstrates that the parallel fraction $P$ is **$98.1\%$ to $98.5\%$**, proving high theoretical speedup via Amdahl's Law.
- **[time_width_study.cpp](./cpu/profiling/time_width_study.cpp)**:
  - Explores scaling behavior across beam widths ($K = 100$ to $26000$) on benchmark instances.

---

## Compilation Guide

### Prerequisites

- **CUDA Toolkit** (`nvcc`, architecture `sm_80` or target GPU compute capability).
- **MPI Library** (OpenMPI or MPICH with `mpicxx` / `mpiCC`).
- **C++ Compiler** with C++11 and OpenMP support (e.g. GCC $\ge 9$).

---

### 1. Using Makefile (Recommended)

A root [Makefile](./Makefile) provides ready-to-use targets to build the solvers:

```bash
# Compile all core solvers (exechybrid_lb, exechybrid_bench, BS-ICH, BS-PMS)
make compile

# Custom GPU architecture (e.g., sm_70 for V100, sm_80 for A100, sm_90 for H100)
make compile ARCH=sm_80
```

---

### 2. Manual Compilation Commands

You can also compile each executable directly from the repository root:

#### CPU Solvers (`cpu/v0`)

```bash
# Sequential BS-ICH (Standard C++11)
g++ -O3 -std=c++11 -o BS-ICH cpu/v0/BS-ICH.cpp

# Parallel BS-PMS (MPI)
mpicxx -O3 -std=c++11 -o BS-PMS cpu/v0/BS-PMS.cpp

# CPU Benchmark Harness
mpicxx -O3 -std=c++11 -o cpu_benchmark cpu/v0/benchmark_cpu.cpp
```

#### GPU Solvers (`gpu/`)

```bash
# Standard GPU Solver (v1)
nvcc -O3 -arch=sm_80 -o gpu_v1_solver gpu/v1/gpu-version.cu -lmpi --expt-relaxed-constexpr

# Large Beam Two-Phase Solver (v2)
nvcc -O3 -arch=sm_80 -o gpu_v2_solver gpu/v2_twophases/gpu_version_large_beam.cu -lmpi --expt-relaxed-constexpr

# GPU Benchmark Runners
nvcc -O3 -arch=sm_80 -o gpu_v1_benchmark gpu/v1/benchmark.cu -lmpi --expt-relaxed-constexpr
nvcc -O3 -arch=sm_80 -o gpu_v2_benchmark gpu/v2_twophases/benchmark_lbversion.cu -lmpi --expt-relaxed-constexpr
```

#### Hybrid CPU-GPU Solvers (`hybrid/`)

```bash
# Standard Hybrid Solver (v1)
nvcc -O3 -arch=sm_80 -Xcompiler -fopenmp -o hybrid_v1_solver hybrid/v1/hybrid_version.cu -lgomp -lmpi --expt-relaxed-constexpr

# Two-Phase Large Beam Hybrid Solver (v2)
nvcc -O3 -arch=sm_80 -Xcompiler -fopenmp -o exechybrid_lb hybrid/v2_twophases/hybrid_large_beam.cu -lgomp -lmpi --expt-relaxed-constexpr

# Hybrid Benchmark Runners
nvcc -O3 -arch=sm_80 -Xcompiler -fopenmp -o exechybrid_bench hybrid/v1/benchmark_hybrid.cu -lgomp -lmpi --expt-relaxed-constexpr
nvcc -O3 -arch=sm_80 -Xcompiler -fopenmp -o hybrid_v2_benchmark hybrid/v2_twophases/benchmark_hybrid_lb.cu -lgomp -lmpi --expt-relaxed-constexpr
```

#### Profiling Tools (`cpu/profiling/`)

```bash
# Feasibility Profiler
mpicxx -O3 -o feasability_profiler cpu/profiling/gpu_feasability_profiling.cpp

# Beam Width Scaling Study
nvcc -O3 -arch=sm_80 -o time_width_study cpu/profiling/time_width_study.cpp -lmpi --expt-relaxed-constexpr
```

---

## Execution Guide

### Instance Path Note

> [!NOTE]
> The source files construct instance paths relative to their subfolder location:
>
> - Lawrence instances are expected at `../../lawrence_instances/la<id>.txt`
> - Taillard instances are expected at `../../taillard_instances/ta<id>.txt`
>
> If executing binaries from their respective subdirectories (e.g., inside `cpu/v0/`, `gpu/v1/`, `gpu/v2_twophases/`, `hybrid/v1/`, or `hybrid/v2_twophases/`), relative paths resolve naturally. When running via the root `Makefile`, the build targets handle instance lookups appropriately.

---

### Command-Line Parameters

- **Sequential CPU Solver (`BS-ICH`)**:

  ```bash
  ./BS-ICH <instance_id> <beam_width>
  # Example: Lawrence instance la33 with beam width 5000
  ./BS-ICH 33 5000
  ```

- **Parallel CPU Solver (`BS-PMS`)**:

  ```bash
  mpirun -n <ranks> ./BS-PMS <instance_id> <beam_width>
  # Example: 20 MPI ranks on la33 with beam width 5000
  mpirun -n 20 ./BS-PMS 33 5000
  ```

- **GPU Solvers (`gpu-version.cu`, `gpu_version_large_beam.cu`)**:

  ```bash
  mpirun -n <ranks> ./gpu_v1_solver <instance_id> <beam_width>
  # Example: 4 MPI ranks (1 per GPU) on la33 with beam width 2000
  mpirun -n 4 ./gpu_v1_solver 33 2000
  ```

- **Hybrid Solvers (`hybrid_version.cu`, `hybrid_large_beam.cu`)**:
  ```bash
  # Arguments: <instance_id> <gpu_beam_width> <accel_factor> <time_limit_sec> <lb_mode>
  # accel_factor: CPU beam ratio (0 = auto-calibrate dynamically)
  # lb_mode: "auto" or 0..4
  mpirun -n 4 ./exechybrid_lb 71 5000 0 1900 auto
  ```

---

### Using Makefile Shortcuts

The [Makefile](./Makefile) defines simple shortcuts for compiling and executing searches with configurable parameters:

| Command          | Description                                                                            | Default Settings                                           |
| :--------------- | :------------------------------------------------------------------------------------- | :--------------------------------------------------------- |
| `make compile`   | Compiles all primary solvers (`exechybrid_lb`, `exechybrid_bench`, `BS-ICH`, `BS-PMS`) | `ARCH=sm_80`                                               |
| `make run`       | Runs `exechybrid_lb` (Two-Phase Hybrid CPU+GPU)                                        | `NP=4`, `INSTANCE=33`, `BEAM_WIDTH=5000`, `ACCEL_FACTOR=0` |
| `make run-pms`   | Runs `BS-PMS` with 20 MPI processes                                                    | `NP_PMS=20`, `INSTANCE=33`, `BEAM_WIDTH=5000`              |
| `make run-ich`   | Runs `BS-ICH` sequential constructive heuristic                                        | `INSTANCE=33`, `BEAM_WIDTH=5000`                           |
| `make run-bench` | Runs the Lawrence hybrid benchmark suite                                               | `NP=4`                                                     |
| `make clean`     | Removes compiled binaries                                                              | —                                                          |
| `make help`      | Displays available targets and overridable variables                                   | —                                                          |

#### Overriding Parameters at Runtime

Pass custom values on the command line:

```bash
# Run BS-PMS on instance 15 with beam width 10000 across 16 MPI ranks
make run-pms INSTANCE=15 BEAM_WIDTH=10000 NP_PMS=16

# Run Hybrid Large Beam on Taillard instance 71 with 8 MPI ranks and 8 OpenMP threads
make run INSTANCE=71 BEAM_WIDTH=25600 NP=8 OMP_THREADS=8

# Run sequential BS-ICH on la10
make run-ich INSTANCE=10 BEAM_WIDTH=2000
```

---

### Slurm Cluster Submission

Pre-configured Slurm batch submission scripts are provided in the root directory for HPC clusters:

#### 1. CPU Benchmark (`submit_cpu.sh`)

```bash
sbatch submit_cpu.sh
```

Compiles and executes [cpu/v0/BS-PMS.cpp](./cpu/v0/BS-PMS.cpp) across multiple MPI ranks.

#### 2. GPU Benchmark (`submit_gpu.sh`)

```bash
sbatch submit_gpu.sh
```

Compiles [gpu/v1/gpu-version.cu](./gpu/v1/gpu-version.cu) and launches jobs across allocated GPU nodes (4 MPI ranks per node, 1 GPU per rank).

#### 3. Hybrid CPU-GPU Benchmark (`submit_hybrid.sh`)

```bash
sbatch submit_hybrid.sh
```

Compiles [hybrid/v1/hybrid_version.cu](./hybrid/v1/hybrid_version.cu) (or [hybrid/v2_twophases/hybrid_large_beam.cu](./hybrid/v2_twophases/hybrid_large_beam.cu)) with OpenMP multithreading mapped to Slurm CPU cores:

```bash
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

---

## Benchmark Instances & Results

### Instance Format

The benchmark files in [lawrence_instances/](./lawrence_instances/) and [taillard_instances/](./taillard_instances/) follow the standard OR-Library layout:

```text
<num_jobs> <num_machines>
<machine_id_0> <duration_0> <machine_id_1> <duration_1> ...
```

- Each subsequent line defines a job's sequence of machine operations and processing times.
- Machine IDs in the files are 0-indexed and parsed into 1-indexed representations internally.

### Pre-Computed Results (`benchmark_results/`)

The [benchmark_results/](./benchmark_results/) directory contains comprehensive logs from experimental evaluations:

| Result File                                                                                                             | Description                                                                          |
| :---------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- |
| [benchmarkv3.txt](./benchmark_results/benchmarkv3.txt)                                                                  | Two-Phase Large Beam GPU solver results across Taillard instances `ta01`–`ta80`.     |
| [two-phase-hybrid-results.txt](./benchmark_results/two-phase-hybrid-results.txt)                                        | Detailed evaluation logs for the Two-Phase Hybrid CPU+GPU solver on large instances. |
| [beam_width_study.txt](./benchmark_results/beam_width_study.txt)                                                        | Study analyzing execution time and makespan scaling as beam width $K$ grows.         |
| [benchmarkhybrid\*.txt](./benchmark_results/)                                                                           | Multi-run logs evaluating the concurrent hybrid CPU-GPU approach.                    |
| [feasability_gpu.txt](./benchmark_results/feasability_gpu.txt)                                                          | High-resolution CPU profiling logs calculating Amdahl's parallel fraction ($P$).     |
| [intra_kernel_profiling.txt](./benchmark_results/intra_kernel_profiling.txt)                                            | Breakdown of kernel stage timings (expansion, scoring, Radix sort, gather).          |
| [la33CPU_K25600.txt](./benchmark_results/la33CPU_K25600.txt) / [ta80CPU25600.txt](./benchmark_results/ta80CPU25600.txt) | CPU baseline logs for large beam width ($K=25600$) on `la33` and `ta80`.             |
