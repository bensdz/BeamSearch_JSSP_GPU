#!/bin/bash
#SBATCH --account=exalab
#SBATCH --partition=dc-gpu
#SBATCH --job-name=jssp_hybrid_benchmark
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --output=outputs/out_R-benchmark-hybrid.o%j
#SBATCH --error=outputs/err_R-benchmark-hybrid.e%j

# Load required modules
module purge
module load GCC
module load CUDA
module load OpenMPI

# Match OpenMP threads to allocated cores per task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close

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

# Compile hybrid solver with OpenMP
nvcc -O3 -arch=sm_80 \
     -Xcompiler -fopenmp \
     -Icpu/common -Icpu/v1_pms -Igpu/common -Igpu/v2_twophases \
     -o hybrid_exec \
     hybrid/v2_twophases/main.cu hybrid/v2_twophases/solver.cu \
     gpu/v2_twophases/solver.cu cpu/v1_pms/solver.cpp cpu/common/instance_parser.cpp \
     -lgomp -lmpi \
     --expt-relaxed-constexpr

# Run hybrid solver across all allocated nodes and GPUs
srun -n $SLURM_NTASKS ./hybrid_exec 33 5000 34 100