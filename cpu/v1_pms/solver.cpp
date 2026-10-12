#include "solver.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

// Run beam search pass with machine-focused branching
int beam_search_pms(const JSSPInstance& inst, int beam_width, int best_known)
{
    const int total_ops = inst.nbjob * inst.nbmachine;

    // Initialize beam with start state
    std::vector<JobShopState> candidates;
    candidates.reserve(beam_width);
    candidates.push_back(create_initial_state(inst));

    std::vector<JobShopState> next_beam;
    next_beam.reserve(beam_width);

    int best_makespan = 500000;

    // Expand beam until all operations are scheduled
    for (int step = 1; step <= total_ops + 10 && !candidates.empty(); step++) {
        next_beam.clear();

        for (int ic = 0; ic < (int)candidates.size(); ic++) {
            const std::vector<int>& ready = candidates[ic].c;
            int nb = (int)ready.size();

            if (nb > 0) {
                // Select a pivot machine from ready operations
                int pivot = rand() % nb;
                int target_mc = inst.operations[ready[pivot]].num_mc;

                // Branch on ready operations for target machine
                for (int r = 0; r < nb; r++) {
                    if (inst.operations[ready[r]].num_mc != target_mc)
                        continue;

                    JobShopState child = candidates[ic];
                    int op = ready[r];

                    // Remove scheduled operation from ready list
                    auto it = std::find(child.c.begin(), child.c.end(), op);
                    if (it != child.c.end())
                        child.c.erase(it);

                    // Schedule operation
                    schedule_operation(op, inst, child);

                    // Normalized score with tie-breaking noise
                    double noise = (double)(rand() % 1000) / 1000000.0;
                    child.score = ((double)child.cmax / child.seq.size()) + noise;

                    // Prune candidates worse than best known makespan
                    if (child.cmax < best_known)
                        insert_into_beam(next_beam, child, beam_width);

                    // Update best complete solution
                    if ((int)child.seq.size() == total_ops && child.cmax < best_makespan)
                        best_makespan = child.cmax;
                }
            }
        }

        candidates.swap(next_beam);
    }

    return best_makespan;
}
