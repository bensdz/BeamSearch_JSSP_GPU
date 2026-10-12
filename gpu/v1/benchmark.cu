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

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > 0) {
        cudaSetDevice(my_rank % num_gpus);
    }

    if (my_rank == 0) {
        printf("\n%-8s %-30s %-6s %-9s %-8s %-12s %-10s\n",
               "Kbest", "Instance", "Jobs", "Machines", "Ops", "BestSpan", "Time(s)");
        printf("-------------------------------------------------------------------------------\n");
    }

    const char* instance_ids[] = {
        "01","02","03","04","05","06","07","08","09","10",
        "11","12","13","14","15","16","17","18","19","20",
        "21","22","23","24","25","26","27","28","29","30",
        "31","32","33","34","35","36","37","38","39","40"
    };
    const int num_instances = 40;
    const int num_iterations = 50;

    for (int idx = 0; idx < num_instances; idx++) {
        char filepath[512];
        if (!resolve_instance_file(instance_ids[idx], filepath, sizeof(filepath))) {
            if (my_rank == 0) printf("SKIP la%s  (file not found)\n", instance_ids[idx]);
            continue;
        }

        // Load problem instance
        JSSPInstance inst;
        if (!parse_instance(filepath, inst))
            continue;

        int beam_width = (idx < 10) ? 1000 : 5000;

        clock_t t0 = clock();
        int local_best  = 500000;
        int global_best = 500000;

        for (int iter = 0; iter < num_iterations; iter++) {
            int seed = 12345 + my_rank * 1000 + iter;
            int makespan = run_gpu_beam_search_v1(inst, beam_width, global_best, seed);

            if (makespan < local_best)
                local_best = makespan;

            MPI_Allreduce(&local_best, &global_best, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        }

        clock_t t1 = clock();
        double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

        if (my_rank == 0) {
            printf("%-8d %-30s %-6d %-9d %-8d %-12d %-10.2f\n",
                   beam_width, filepath,
                   inst.nbjob, inst.nbmachine, inst.nboperation,
                   global_best, elapsed);
        }
    }

    if (my_rank == 0)
        printf("\nAll Lawrence instances finished.\n");

    MPI_Finalize();
    return 0;
}