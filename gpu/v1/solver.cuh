#pragma once

#include "../../cpu/common/jssp_types.h"

// Run single-phase GPU beam search pass
int run_gpu_beam_search_v1(
    const JSSPInstance& inst,
    int beam_width,
    int best_known,
    int random_seed,
    int* out_best_sequence = nullptr
);
