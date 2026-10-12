#include "solver.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

// Run beam search with iterative construction heuristic (BS-ICH)
int beam_search_ich(const JSSPInstance& inst, int beam_width)
{
    const int total_ops = inst.nbjob * inst.nbmachine;

    // Seed beam with the initial state
    std::vector<JobShopState> beam;
    beam.push_back(create_initial_state(inst));

    std::vector<JobShopState> next_beam;

    // Expand beam one operation at a time
    for (int step = 0; step < total_ops; step++) {
        next_beam.clear();

        for (const auto& current : beam) {
            // Branch across all currently ready operations
            for (size_t idx = 0; idx < current.c.size(); idx++) {
                JobShopState child = current;
                int op = current.c[idx];

                // Remove selected operation from ready list
                child.c.erase(child.c.begin() + idx);

                // Schedule operation
                schedule_operation(op, inst, child);

                // Lower score is better
                child.score = (double)child.cmax + (rand() % 100) * 0.01;

                // Keep top-K states in beam
                insert_into_beam(next_beam, child, beam_width);
            }
        }

        beam = next_beam;
        if (beam.empty()) break;
    }

    // Find best makespan in final beam
    int best = 500000;
    for (const auto& state : beam) {
        if (state.cmax < best)
            best = state.cmax;
    }

    return best;
}
