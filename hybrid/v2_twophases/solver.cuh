#pragma once

#include "../../cpu/common/jssp_types.h"

// Run one hybrid iteration (GPU v2 two-phase on thread 0, CPU PMS on worker threads)
int run_hybrid_iteration_v2(
    const JSSPInstance& inst,
    int gpu_beam_width,
    int cpu_beam_width,
    int best_known,
    int rank,
    int iteration
);
