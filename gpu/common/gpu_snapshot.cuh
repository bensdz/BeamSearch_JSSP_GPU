#pragma once

#include <cstring>
#include <cuda_runtime.h>
#include "gpu_common.cuh"

// Offset descriptor for flat integer snapshot representation
struct SnapshotLayout {
    int off_machine_ready;
    int off_op_finish;
    int off_precedence_ready;
    int off_ready_count;
    int off_sequence;
    int off_sequence_count;
    int off_makespan;
    int total_ints;
};

// Compute snapshot offsets based on instance dimensions
inline SnapshotLayout compute_snapshot_layout(int num_jobs, int num_machines, int num_ops) {
    SnapshotLayout layout;
    layout.off_machine_ready     = 0;
    layout.off_op_finish         = layout.off_machine_ready + (num_machines + 1);
    layout.off_precedence_ready  = layout.off_op_finish + (num_ops + 1);
    layout.off_ready_count       = layout.off_precedence_ready + num_jobs;
    layout.off_sequence          = layout.off_ready_count + 1;
    layout.off_sequence_count    = layout.off_sequence + (num_ops + 1);
    layout.off_makespan          = layout.off_sequence_count + 1;
    layout.total_ints            = layout.off_makespan + 1;
    return layout;
}

// Pack initial empty state with first operation of each job ready
inline void pack_initial_snapshot(const JSSPInstance& inst, int* flat, const SnapshotLayout& layout) {
    std::memset(flat, 0, layout.total_ints * sizeof(int));

    // First operation of each job is initially ready
    for (int j = 0; j < inst.nbjob; j++) {
        flat[layout.off_precedence_ready + j] = j * inst.nbmachine + 1;
    }
    flat[layout.off_ready_count]    = inst.nbjob;
    flat[layout.off_sequence_count] = 0;
    flat[layout.off_makespan]       = 0;
}

// Device operation scheduling function
__device__ inline void dev_schedule_operation(
    int op,
    int *machine_ready_time,
    int *op_finish_time,
    int *precedence_ready_ops,
    int &num_ready,
    int *seq,
    int &num_scheduled,
    int num_machines_1idx
) {
    int mc = c_ops[op].machine_id;
    int job_pred_finish = (op % num_machines_1idx == 1) ? 0 : op_finish_time[op - 1];
    int start = max(machine_ready_time[mc], job_pred_finish);
    int finish = start + c_ops[op].duration;

    machine_ready_time[mc] = finish;
    op_finish_time[op]     = finish;
    seq[num_scheduled++]   = op;

    // Add next operation in job to ready list if not completed
    if (op % num_machines_1idx != 0) {
        precedence_ready_ops[num_ready++] = op + 1;
    }
}
