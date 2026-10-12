# GPU-Accelerated Multi-Strategy Beam Search for the Job-Shop Scheduling Problem (JSSP)

This repository contains a high-performance parallel framework for solving the **Job-Shop Scheduling Problem (JSSP)** with unlimited intermediate buffers. The framework integrates **Beam Search Iterative Construction Heuristics (BS-ICH)** with **Parallel Multi-Strategy Beam Search (BS-PMS)**, accelerated on NVIDIA GPUs with CUDA and scaled across CPU cores, multiple GPUs, and compute clusters using OpenMP and MPI.

---

## Table of Contents

1. [Problem Formulation](#problem-formulation)
2. [Algorithmic Core](#algorithmic-core)
3. [Repository Structure](#repository-structure)
4. [Solver Architectures](#solver-architectures)
   - [CPU Solvers (`cpu/`)](#cpu-solvers-cpu)
   - [GPU Solvers (`gpu/`)](#gpu-solvers-gpu)
   - [Hybrid Solvers (`hybrid/`)](#hybrid-solvers-hybrid)
   - [Profiling Tools (`cpu/profiling/`)](#profiling-tools-cpuprofiling)
5. [Compilation](#compilation)
6. [Execution Guide](#execution-guide)
7. [Slurm Cluster Scripts](#slurm-cluster-scripts)
8. [Benchmark Instances](#benchmark-instances)

---

## Problem Formulation

In the classical **Job-Shop Scheduling Problem (JSSP)**:
- A set of $n$ jobs must be processed on a set of $m$ machines.
- Each job consists of an ordered sequence of operations, each requiring a specific machine for a given duration.
- Each machine can process at most one operation at a time (non-preemptive).
- Machines have unlimited intermediate buffers: when an operation completes, it releases its machine immediately.
- The objective is to minimize the **Makespan ($C_{\max}$)**, the completion time of the entire schedule.

---

## Algorithmic Core

The framework implements two primary constructive beam search variants:

1. **Sequential Beam Search (BS-ICH)**:
   - Constructs schedules operation-by-operation.
   - At each level, all ready operations are expanded across candidates in the current beam.
   - Retains the top-$K$ states according to heuristic scoring (makespan evaluation with progress normalization).

2. **Parallel Multi-Strategy Beam Search (BS-PMS)**:
   - Uses machine-focused branching to focus search exploration on bottleneck resources.
   - Diversifies exploration by running multiple independent MPI ranks with distinct random seeds.
   - Synchronizes the global best makespan across ranks via `MPI_Allreduce` for branch-and-bound pruning.

---

## Repository Structure

```text
GPU_BS_JSSP/
├── Makefile                           # Unified build system
├── submit_cpu.sh                      # Slurm cluster script for CPU solver
├── submit_gpu.sh                      # Slurm cluster script for multi-GPU solver
├── submit_hybrid.sh                   # Slurm cluster script for multi-GPU hybrid solver
│
├── cpu/
│   ├── common/
│   │   ├── jssp_types.h               # Core data structures (Operation, JSSPInstance, JobShopState)
│   │   ├── instance_parser.h          # Benchmark instance file parser header
│   │   └── instance_parser.cpp        # Instance parser implementation
│   ├── v0_ich/
│   │   ├── solver.h                   # Sequential BS-ICH solver header
│   │   ├── solver.cpp                 # Sequential BS-ICH beam search implementation
│   │   └── main.cpp                   # Single-instance CLI runner
│   ├── v1_pms/
│   │   ├── solver.h                   # Parallel BS-PMS solver header
│   │   ├── solver.cpp                 # Machine-focused BS-PMS implementation
│   │   ├── main.cpp                   # MPI single-instance CLI runner
│   │   └── benchmark.cpp              # Full Lawrence suite benchmark runner
│   └── profiling/
│       ├── gpu_feasibility.cpp        # Parallel fraction profiler (Amdahl's law study)
│       └── time_width_study.cpp       # Beam width scaling analysis (varying K)
│
├── gpu/
│   ├── common/
│   │   ├── gpu_common.cuh             # GPU constants, OperationInfo, and device upload
│   │   └── gpu_snapshot.cuh           # Compact flat snapshot layout and dev_schedule_operation
│   ├── v1/
│   │   ├── solver.cuh                 # Single-phase GPU solver header
│   │   ├── solver.cu                  # Expansion kernel + CUB radix sort + Gather kernel
│   │   ├── main.cu                    # Single-instance multi-GPU CLI runner
│   │   └── benchmark.cu               # Lawrence suite multi-GPU benchmark runner
│   └── v2_twophases/
│       ├── solver.cuh                 # Two-phase memory-optimized solver header
│       ├── solver.cu                  # Phase 1: score evaluation; Phase 2: top-K materialization
│       ├── main.cu                    # Single-instance multi-GPU CLI runner
│       └── benchmark.cu               # Lawrence suite multi-GPU benchmark runner
│
├── hybrid/
│   ├── v1/
│   │   ├── solver.cuh                 # Hybrid v1 solver header
│   │   ├── solver.cu                  # OpenMP coordination (GPU v1 + CPU PMS)
│   │   ├── main.cu                    # Multi-node multi-GPU single-instance CLI runner
│   │   └── benchmark.cu               # Lawrence suite multi-GPU benchmark runner
│   └── v2_twophases/
│       ├── solver.cuh                 # Hybrid v2 solver header
│       ├── solver.cu                  # OpenMP coordination (GPU v2 two-phase + CPU PMS)
│       ├── main.cu                    # Multi-node multi-GPU single-instance CLI runner
│       └── benchmark.cu               # Lawrence suite multi-GPU benchmark runner
│
├── lawrence_instances/                # Lawrence benchmark files (la01.txt - la40.txt)
└── taillard_instances/                # Taillard benchmark files (ta01 - ta80)
```

---

## Solver Architectures

### CPU Solvers (`cpu/`)
- **`v0_ich`**: Standard sequential beam search. Good for baselines and single-core evaluation.
- **`v1_pms`**: Machine-focused branching with MPI parallel multi-strategy search and global bound synchronization.

### GPU Solvers (`gpu/`)
- **`v1` (Single-Phase)**: Expands each candidate into $B$ branches in device memory, evaluates scores, sorts with CUB radix sort, and gathers winning states. Optimal for $K \le 5000$.
- **`v2_twophases` (Two-Phase Memory-Optimized)**:
  - *Phase 1*: Computes branch scores directly into compact score arrays without writing full state snapshots to device memory.
  - *Sort*: Radix sorts scores and candidate indices.
  - *Phase 2*: Materializes only the top-$K$ winning snapshots directly into the next beam buffer using ping-pong buffers. Supports very large beam widths ($K \ge 50000$) with minimal memory footprint.

### Hybrid Solvers (`hybrid/`)
- Each MPI rank coordinates CPU cores and GPU hardware simultaneously:
  - Thread 0 drives the GPU beam search pass.
  - Worker threads run CPU multi-strategy passes with calibrated beam widths.
  - `MPI_Allreduce` shares bounds across cluster ranks after each iteration to aggressively prune search subtrees.

### Profiling Tools (`cpu/profiling/`)
- **`gpu_feasibility.cpp`**: Profiles CPU execution by separating parallelizable candidate evaluation and sorting from sequential state extraction, computing parallel ratios and theoretical speedups.
- **`time_width_study.cpp`**: Analyzes execution time and solution quality as beam width $K$ scales from $100$ to $25600$ across benchmark instances.

---

## Compilation

Build all targets using the top-level `Makefile`:

```bash
# Build all CPU, GPU, and Hybrid executables
make compile

# Clean binaries
make clean
```

### Overridable Makefile Variables
- `ARCH`: Target GPU compute capability (default: `sm_80`)
- `NP`: Number of MPI ranks for hybrid execution (default: `4`)
- `NP_PMS`: Number of MPI ranks for PMS/GPU execution (default: `20`)
- `OMP_THREADS`: Number of OpenMP threads per rank (default: `4`)
- `INSTANCE`: Instance identifier or path (default: `33`)
- `BEAM_WIDTH`: Search beam width $K$ (default: `5000`)
- `ACCEL_FACTOR`: CPU beam scaling divisor (default: `34`)

---

## Execution Guide

### CPU Solvers
```bash
# Sequential BS-ICH
make run-ich INSTANCE=33 BEAM_WIDTH=5000

# Parallel BS-PMS (MPI)
make run-pms INSTANCE=33 BEAM_WIDTH=5000 NP_PMS=20

# Run full Lawrence benchmark suite on CPU
make run-pms-bench NP_PMS=20

# Profiling studies
make run-feasibility INSTANCE=33 NP_PMS=20
make run-time-width NP_PMS=20
```

### GPU Solvers
```bash
# Single-phase GPU solver (v1)
make run-gpu-v1 INSTANCE=33 BEAM_WIDTH=1000 NP=4

# Two-phase GPU solver (v2)
make run-gpu-v2 INSTANCE=33 BEAM_WIDTH=5000 NP=4

# Lawrence benchmark suites on GPU
make run-gpu-v1-bench NP=4
make run-gpu-v2-bench NP=4
```

### Hybrid Solvers
```bash
# Hybrid v1 (single-phase GPU + CPU OpenMP threads)
make run-hybrid-v1 INSTANCE=33 BEAM_WIDTH=5000 ACCEL_FACTOR=34 NP=4 OMP_THREADS=4

# Hybrid v2 (two-phase GPU + CPU OpenMP threads)
make run-hybrid-v2 INSTANCE=33 BEAM_WIDTH=5000 ACCEL_FACTOR=34 NP=4 OMP_THREADS=4

# Lawrence benchmark suites with Hybrid solver
make run-hybrid-v1-bench NP=4 OMP_THREADS=4
make run-hybrid-v2-bench NP=4 OMP_THREADS=4
```

---

## Slurm Cluster Scripts

The repository includes pre-configured Slurm batch scripts for multi-GPU and multi-node compute clusters:

- **[`submit_cpu.sh`](submit_cpu.sh)**: Allocates CPU nodes and launches MPI parallel multi-strategy search across all assigned tasks.
- **[`submit_gpu.sh`](submit_gpu.sh)**: Allocates a GPU node with 4 GPUs, binding one MPI rank per GPU for distributed GPU search.
- **[`submit_hybrid.sh`](submit_hybrid.sh)**: Allocates 4 nodes with 4 GPUs each (16 GPUs total), binding 4 OpenMP threads per MPI rank for distributed hybrid search.

Submit jobs to the Slurm queue:
```bash
sbatch submit_cpu.sh
sbatch submit_gpu.sh
sbatch submit_hybrid.sh
```

---

## Benchmark Instances

- **Lawrence (1984)**: `lawrence_instances/la01.txt` through `la40.txt` (sizes from $10 \times 5$ to $30 \times 10$).
- **Taillard (1993)**: `taillard_instances/ta01` through `ta80` (sizes from $15 \times 15$ to $100 \times 20$).
