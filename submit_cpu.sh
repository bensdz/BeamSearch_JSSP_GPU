#!/bin/bash
#SBATCH --account=exalab
#SBATCH --partition=dc-gpu
#SBATCH --job-name=bjssp_cpu_benchmark
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=7:00:00
#SBATCH --output=R-benchmark-cpu.o%j
#SBATCH --error=R-benchmark-cpu.e%j

# Load required modules (CPU only)
module purge
module load GCC
module load OpenMPI

echo "========================================"
echo "CPU BENCHMARK (on GPU partition)"
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $SLURM_NNODES"
echo "Tasks:         $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "Node list:     $SLURM_JOB_NODELIST"
echo "Working dir:   $(pwd)"
echo "========================================"

# Show CPU info
echo "CPU Information:"
lscpu | grep -E "CPU|Core|Socket"
echo "========================================"

# Compile BS-PMS.cpp with standard C++ compiler (no CUDA)
echo "Compiling BS-PMS.cpp for CPU..."
mpiCC -O3 \
     -o execkbest_cpu \
     cpu/v0/BS-PMS.cpp

if [ $? -ne 0 ]; then
    echo "CPU compilation failed!"
    exit 1
fi

echo "CPU compilation successful!"
echo "Running CPU benchmark with $SLURM_NTASKS MPI ranks..."
echo "========================================"

# Run benchmark
mpirun -n $SLURM_NTASKS ./execkbest_cpu 33 25600

echo "========================================"
echo "CPU benchmark completed!"
