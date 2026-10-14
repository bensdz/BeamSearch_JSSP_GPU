#include "solver.cuh"
#include "../../gpu/v1/solver.cuh"
#include "../../cpu/v1_pms/solver.h"

#include <omp.h>
#include <algorithm>
#include <cstdlib>
#include <ctime>

// Run one hybrid iteration with OpenMP thread cooperation
int run_hybrid_iteration_v1(
    const JSSPInstance& inst,
    int gpu_beam_width,
    int cpu_beam_width,
    int best_known,
    int rank,
    int iteration
) {
    int best_iter = best_known;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int result = 500000;

        if (tid == 0) {
            // GPU beam search pass on primary thread
            int seed = rand();
            result = run_gpu_beam_search_v1(inst, gpu_beam_width, best_known, seed);
        } else {
            // CPU beam search pass on worker thread
            srand((unsigned)time(NULL) + rank * 1000 + iteration * 100 + tid);
            result = beam_search_pms(inst, cpu_beam_width, best_known);
        }

        #pragma omp critical
        {
            if (result < best_iter)
                best_iter = result;
        }
    }

    return best_iter;
}
