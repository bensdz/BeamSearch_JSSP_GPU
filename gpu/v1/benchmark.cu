#include "solver.cuh"
#include "../../cpu/common/instance_parser.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv)
{
    printf("\n%-8s %-30s %-6s %-9s %-8s %-12s %-10s\n",
           "Kbest", "Instance", "Jobs", "Machines", "Ops", "BestSpan", "Time(s)");
    printf("-------------------------------------------------------------------------------\n");

    const char* instance_ids[] = {
        "01","02","03","04","05","06","07","08","09","10",
        "11","12","13","14","15","16","17","18","19","20",
        "21","22","23","24","25","26","27","28","29","30",
        "31","32","33","34","35","36","37","38","39","40"
    };
    const int num_instances = 40;

    for (int idx = 0; idx < num_instances; idx++) {
        char filepath[512];
        if (!resolve_instance_file(instance_ids[idx], filepath, sizeof(filepath))) {
            printf("SKIP la%s  (file not found)\n", instance_ids[idx]);
            continue;
        }

        // Load problem instance
        JSSPInstance inst;
        if (!parse_instance(filepath, inst))
            continue;

        // Beam width based on problem size
        int beam_width = (idx < 10) ? 1000 : 5000;

        clock_t t0 = clock();
        int best_makespan = run_gpu_beam_search_v1(inst, beam_width, 500000, 12345 + idx);
        clock_t t1 = clock();
        double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

        printf("%-8d %-30s %-6d %-9d %-8d %-12d %-10.2f\n",
               beam_width, filepath,
               inst.nbjob, inst.nbmachine, inst.nboperation,
               best_makespan, elapsed);
    }

    printf("\nAll Lawrence instances finished.\n");
    return 0;
}