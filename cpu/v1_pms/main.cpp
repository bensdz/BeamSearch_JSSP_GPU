#include "solver.h"
#include "../common/instance_parser.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpi.h>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    char instance_file[512] = "./la33.txt";
    int  beam_width = 5000;
    int  num_iterations = 500;

    // Parse command line arguments
    if (argc > 1) {
        if (!resolve_instance_file(argv[1], instance_file, sizeof(instance_file))) {
            if (my_rank == 0)
                std::cerr << "Error: cannot locate instance file for: " << argv[1] << std::endl;
            MPI_Finalize();
            return 1;
        }
    }
    if (argc > 2) {
        beam_width = atoi(argv[2]);
        if (beam_width < 1) {
            if (my_rank == 0)
                std::cerr << "Error: beam_width must be >= 1" << std::endl;
            MPI_Finalize();
            return 1;
        }
    }

    // Load instance
    JSSPInstance inst;
    if (!parse_instance(instance_file, inst)) {
        MPI_Finalize();
        return 1;
    }

    if (my_rank == 0) {
        printf("  Instance : %s  (%d jobs x %d machines)\n",
               instance_file, inst.nbjob, inst.nbmachine);
        printf("  Beam     : %d  |  Ranks : %d  |  Iterations : %d\n",
               beam_width, world_size, num_iterations);
    }

    // Parallel beam search with different random seeds per rank
    srand((unsigned)time(NULL) + my_rank);
    clock_t t0 = clock();

    int local_best  = 500000;
    int global_best = 500000;

    for (int iter = 0; iter < num_iterations; iter++) {
        int makespan = beam_search_pms(inst, beam_width, global_best);

        if (makespan < local_best)
            local_best = makespan;

        // Share minimum makespan across ranks
        MPI_Allreduce(&local_best, &global_best, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (my_rank == 0 && iter % 50 == 0)
            printf("  iter %4d | global best = %d\n", iter, global_best);
    }

    clock_t t1 = clock();
    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

    // Print summary results
    if (my_rank == 0) {
        printf("  ===========================\n");
        printf("  Best Makespan  : %d\n", global_best);
        printf("  Execution Time : %.2f seconds\n", elapsed);
    }

    MPI_Finalize();
    return 0;
}
