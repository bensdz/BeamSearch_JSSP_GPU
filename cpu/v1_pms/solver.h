#pragma once

#include "../common/jssp_types.h"

// Run beam search with machine-focused branching (BS-PMS)
int beam_search_pms(const JSSPInstance& inst, int beam_width, int best_known);
