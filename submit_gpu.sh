#!/bin/bash
#SBATCH --account=exalab
#SBATCH --partition=dc-gpu-devel
#SBATCH --job-name=jssp_gpu_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=1:59:00
#SBATCH --exclusive
#SBATCH --output=R-benchmark-gpu.o%j
#SBATCH --error=R-benchmark-gpu.e%j

# Load required modules
module purge
module load GCC
module load CUDA
module load OpenMPI

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $SLURM_NNODES"
echo "Tasks:         $SLURM_NTASKS"
echo "GPUs/node:     4"
echo "Node list:     $SLURM_JOB_NODELIST"
echo "Working dir:   $(pwd)"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4
echo "========================================"

# Compile GPU v2 solver
nvcc -O3 -arch=sm_80 \
     -Icpu/common -Igpu/common \
     -o gpu_exec \
     gpu/v2_twophases/main.cu gpu/v2_twophases/solver.cu cpu/common/instance_parser.cpp \
     -lmpi \
     --expt-relaxed-constexpr

# Run multi-GPU solver across allocated GPUs
mpirun -n $SLURM_NTASKS ./gpu_exec 33 5000
