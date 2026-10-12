#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>

// Operation within a job
struct Operation {
    int num_op;   // 1-indexed operation ID
    int num_mc;   // 1-indexed machine ID
    int dure;     // processing duration
    int num_j;    // 1-indexed job ID
};

// JSSP instance specification
struct JSSPInstance {
    int nbjob       = 0;
    int nbmachine   = 0;
    int nboperation = 0;

    std::vector<Operation> operations;
    std::vector<std::vector<int>> opr_mach;
};

// Beam search state
class JobShopState {
public:
    std::vector<int> tr;    // Machine ready times (1-indexed)
    std::vector<int> seq;   // Scheduled operation sequence
    std::vector<int> c;     // Ready operations
    std::vector<int> tf;    // Operation finish times

    int    cmax  = 500000;  // Current makespan
    double score = 0.0;     // Selection score

    JobShopState() = default;
};

// Max-heap comparator for pruning worst candidates
struct CompareTask {
    bool operator()(const JobShopState& a, const JobShopState& b) const {
        return a.score < b.score;
    }
};

// Insert state into bounded beam
inline void insert_into_beam(std::vector<JobShopState>& beam,
                             const JobShopState& state, int k)
{
    if ((int)beam.size() < k) {
        beam.push_back(state);
        std::push_heap(beam.begin(), beam.end(), CompareTask{});
    } else if (state.score < beam.front().score) {
        std::pop_heap(beam.begin(), beam.end(), CompareTask{});
        beam.back() = state;
        std::push_heap(beam.begin(), beam.end(), CompareTask{});
    }
}

// Create initial state with the first operation of each job ready
inline JobShopState create_initial_state(const JSSPInstance& inst) {
    JobShopState s;
    s.tr.assign(inst.nbmachine + 1, 0);
    s.tf.assign(inst.nboperation, 0);
    s.cmax  = 0;
    s.score = 0.0;

    for (int j = 0; j < inst.nbjob; j++)
        s.c.push_back(j * inst.nbmachine + 1);

    return s;
}

// Schedule an operation and update machine time, finish time, and ready operations
inline void schedule_operation(int op, const JSSPInstance& inst, JobShopState& state)
{
    const int mc = inst.operations[op].num_mc;
    const int M  = inst.nbmachine;

    state.seq.push_back(op);

    int job_pred_finish = (op % M == 1) ? 0 : state.tf[op - 1];
    int start = std::max(state.tr[mc], job_pred_finish);

    state.tr[mc] = start + inst.operations[op].dure;
    state.tf[op] = state.tr[mc];

    // Add next operation in job to ready list if not completed
    if (op % M != 0)
        state.c.push_back(op + 1);

    // Update makespan
    state.cmax = 0;
    for (int m = 1; m <= M; m++)
        if (state.tr[m] > state.cmax)
            state.cmax = state.tr[m];
}

// Find index of value in vector, returns -1 if not found
inline int find_val(const std::vector<int>& a, int v) {
    auto it = std::find(a.begin(), a.end(), v);
    return (it != a.end()) ? (int)std::distance(a.begin(), it) : -1;
}
