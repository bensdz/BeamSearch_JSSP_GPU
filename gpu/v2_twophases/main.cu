#include "solver.cuh"
#include "../../cpu/common/instance_parser.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <cuda_runtime.h>

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
    int  beam_width = 5000;
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
    if (argc > 2) beam_width     = atoi(argv[2]);
    if (argc > 3) num_iterations = atoi(argv[3]);

    // Load problem instance
    JSSPInstance inst;
    if (!parse_instance(instance_file, inst)) {
        MPI_Finalize();
        return 1;
    }

    if (my_rank == 0) {
        printf("=== GPU Beam Search (v2 Two-Phases) ===\n");
        printf("Instance: %s  |  Jobs: %d  |  Machines: %d\n", instance_file, inst.nbjob, inst.nbmachine);
        printf("Beam Width (K): %d  |  MPI Ranks: %d  |  GPUs/Node: %d  |  Iterations: %d\n",
               beam_width, world_size, num_gpus, num_iterations);
        printf("=======================================\n");
    }

    clock_t t0 = clock();
    int local_best  = 500000;
    int global_best = 500000;

    for (int iter = 0; iter < num_iterations; iter++) {
        int seed = 12345 + my_rank * 1000 + iter;
        int makespan = run_gpu_beam_search_v2(inst, beam_width, global_best, seed);

        if (makespan < local_best)
            local_best = makespan;

        // Synchronize best makespan across all ranks
        MPI_Allreduce(&local_best, &global_best, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }

    clock_t t1 = clock();
    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

    if (my_rank == 0) {
        printf("=======================================\n");
        printf("Best Makespan  : %d\n", global_best);
        printf("Execution Time : %.2f seconds\n", elapsed);
    }

    MPI_Finalize();
    return 0;
}
