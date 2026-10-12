#include "../common/jssp_types.h"
#include "../common/instance_parser.h"
#include "../v1_pms/solver.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <mpi.h>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Instances to benchmark
    const char* instance_ids[] = {"10", "15", "21", "31"};
    const int num_instances = 4;

    const int widths[] = {100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600};
    const int num_widths = 9;
    const int iterations_per_width = 100;

    if (my_rank == 0) {
        printf("\n%-8s %-30s %-6s %-9s %-8s %-12s %-10s\n",
               "Width", "Instance", "Jobs", "Machines", "Ops", "BestSpan", "Time(s)");
        printf("-------------------------------------------------------------------------------\n");
    }

    for (int j = 0; j < num_instances; j++) {
        char instance_path[512];
        if (!resolve_instance_file(instance_ids[j], instance_path, sizeof(instance_path))) {
            if (my_rank == 0)
                printf("SKIP la%s (file not found)\n", instance_ids[j]);
            continue;
        }

        // Load problem instance
        JSSPInstance inst;
        if (!parse_instance(instance_path, inst))
            continue;

        if (my_rank == 0)
            printf("\n==== Instance %s ====\n", instance_path);

        // Sweep beam widths
        for (int w = 0; w < num_widths; w++) {
            int beam_width = widths[w];

            srand((unsigned)time(NULL) + my_rank + w);
            clock_t wall_start = clock();

            int local_best = 500000;
            int global_best = 500000;

            for (int iter = 0; iter < iterations_per_width; iter++) {
                int makespan = beam_search_pms(inst, beam_width, global_best);
                if (makespan < local_best)
                    local_best = makespan;

                // Synchronize global best across ranks
                MPI_Allreduce(&local_best, &global_best, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            }

            clock_t wall_end = clock();
            double elapsed = (double)(wall_end - wall_start) / CLOCKS_PER_SEC;

            if (my_rank == 0) {
                printf("%-8d %-30s %-6d %-9d %-8d %-12d %-10.2f\n",
                       beam_width,
                       instance_path,
                       inst.nbjob,
                       inst.nbmachine,
                       inst.nboperation,
                       global_best,
                       elapsed);
            }
        }
    }

    if (my_rank == 0)
        printf("\nTime vs Beam Width study finished.\n");

    MPI_Finalize();
    return 0;
}
