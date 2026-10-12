#pragma once

#include "../common/jssp_types.h"

// Run beam search with iterative construction heuristic (BS-ICH)
int beam_search_ich(const JSSPInstance& inst, int beam_width);
