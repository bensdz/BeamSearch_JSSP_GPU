#!/bin/bash
#SBATCH --account=exalab
#SBATCH --partition=dc-gpu
#SBATCH --job-name=bjssp_gpu_benchmark
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --output=R-benchmark-gpu.o%j
#SBATCH --error=R-benchmark-gpu.e%j

# Load required modules
module purge
module load GCC
module load CUDA
module load OpenMPI

# Match OpenMP threads to allocated cores per task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores            # Pin OpenMP threads to CPU cores
export OMP_PROC_BIND=close         # Keep threads close for shared cache benefits

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $SLURM_NNODES"
echo "Tasks:         $SLURM_NTASKS"
echo "Node list:     $SLURM_JOB_NODELIST"
echo "Working dir:   $(pwd)"
echo "========================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4
echo "========================================"

# Compile the hybrid code with OpenMP support
nvcc -O3 -arch=sm_80 \
     -Xcompiler -fopenmp \
     -o hybrid_exec \
     hybrid/v1/hybrid_version.cu \
     -lgomp -lmpi \
     --expt-relaxed-constexpr

# Command Line Arguments:
# 1. Instance Number (e.g., 33 for la33.txt)
# 2. GPU Beam Width (e.g., 5000)
# 3. CPU Accel Factor (0 = Auto-calibrate dynamically based on machine speed)
# 4. Time limit in seconds for the instance (Number of iterations if running benchmark)
# 5. Lower Bound Mode ("auto" sets different threads to use different advanced heuristics)

echo "Starting Hybrid CPU+GPU execution..."
srun -n $SLURM_NTASKS ./exechybrid 71 5000 0 1900 auto