#include "solver.cuh"
#include "../../cpu/common/instance_parser.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Multi-GPU setup: bind each MPI rank to an available GPU
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > 0) {
        cudaSetDevice(my_rank % num_gpus);
    }

    char instance_file[512] = "./la33.txt";
    int  gpu_beam_width = 5000;
    int  accel_factor   = 34;
    int  num_iterations = 100;

    // Parse command line arguments
    if (argc > 1) {
        if (!resolve_instance_file(argv[1], instance_file, sizeof(instance_file))) {
            if (my_rank == 0)
                fprintf(stderr, "Error: cannot locate instance file: %s\n", argv[1]);
            MPI_Finalize();
            return 1;
        }
    }
    if (argc > 2) gpu_beam_width = std::max(1, atoi(argv[2]));
    if (argc > 3) accel_factor   = std::max(1, atoi(argv[3]));
    if (argc > 4) num_iterations = std::max(1, atoi(argv[4]));

    int cpu_beam_width = std::max(1, gpu_beam_width / accel_factor);

    // Load problem instance
    JSSPInstance inst;
    if (!parse_instance(instance_file, inst)) {
        MPI_Finalize();
        return 1;
    }

    int num_omp_threads = omp_get_max_threads();

    if (my_rank == 0) {
        printf("=== Hybrid CPU+GPU Beam Search (v2 Two-Phases) ===\n");
        printf("Instance: %s  |  Jobs: %d  |  Machines: %d\n", instance_file, inst.nbjob, inst.nbmachine);
        printf("GPU Beam: %d  |  CPU Beam: %d  |  Accel Factor: %d\n", gpu_beam_width, cpu_beam_width, accel_factor);
        printf("MPI Ranks: %d  |  GPUs/Node: %d  |  OpenMP Threads/Rank: %d  |  Iterations: %d\n",
               world_size, num_gpus, num_omp_threads, num_iterations);
        printf("===================================================\n");
    }

    clock_t t0 = clock();
    int local_best  = 500000;
    int global_best = 500000;

    for (int iter = 0; iter < num_iterations; iter++) {
        int iter_best = run_hybrid_iteration_v2(inst, gpu_beam_width, cpu_beam_width, global_best, my_rank, iter);

        if (iter_best < local_best)
            local_best = iter_best;

        // Synchronize best makespan across all MPI ranks
        MPI_Allreduce(&local_best, &global_best, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (my_rank == 0 && (iter + 1) % 20 == 0)
            printf("  iter %3d / %3d | best makespan = %d\n", iter + 1, num_iterations, global_best);
    }

    clock_t t1 = clock();
    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

    if (my_rank == 0) {
        printf("===================================================\n");
        printf("Best Makespan  : %d\n", global_best);
        printf("Execution Time : %.2f seconds\n", elapsed);
    }

    MPI_Finalize();
    return 0;
}
