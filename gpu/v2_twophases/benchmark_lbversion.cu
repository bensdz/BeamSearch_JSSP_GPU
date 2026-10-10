#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <climits>
#include <cfloat>

#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>

using namespace std;

// CUDA API error wrapper
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t _e = (call);                                      \
        if (_e != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_e));      \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Runtime buffer bounds
static int BUF_OPS      = 0;
static int BUF_MACHINES = 0;
static int BUF_JOBS     = 0;
static int BUF_BEAM     = 0;

// Absolute compile-time upper bounds
#define HARD_MAX_OPS       2100
#define HARD_MAX_MACHINES   26
#define HARD_MAX_JOBS       100
#define HARD_MAX_BEAM     50000

// Operation descriptor
struct OperationInfo {
    int machine_id;
    int duration;
    int job_id;
};

// Constant memory for instance data
__constant__ OperationInfo c_ops[HARD_MAX_OPS];
__constant__ int c_op_count[HARD_MAX_MACHINES];

// Global device memory for op order (moved from constant memory to avoid 64KB limit)
static int *d_op_order = nullptr;

// Flat integer array layout for scheduling state snapshots (gpu)
static int SNAP_OFF_MACHINE_OCCUPIED_OP = 0;
static int SNAP_OFF_MACHINE_READY_TIME  = 0;
static int SNAP_OFF_CRITICAL_PARENT_OP  = 0;
static int SNAP_OFF_OP_START_TIME       = 0;
static int SNAP_OFF_OP_FINISH_TIME      = 0;
static int SNAP_OFF_PRECEDENCE_READY    = 0;
static int SNAP_OFF_READY_COUNT         = 0;
static int SNAP_OFF_SEQUENCE            = 0;
static int SNAP_OFF_SEQUENCE_COUNT      = 0;
static int SNAP_OFF_MAKESPAN            = 0;
static int SNAPSHOT_INTS                = 0;

// Compute snapshot byte offsets based on buffer sizes
static void compute_snapshot_layout() {
    SNAP_OFF_MACHINE_OCCUPIED_OP = 0;
    SNAP_OFF_MACHINE_READY_TIME  = BUF_MACHINES;
    SNAP_OFF_CRITICAL_PARENT_OP  = 2 * BUF_MACHINES;
    SNAP_OFF_OP_START_TIME       = 2 * BUF_MACHINES +     BUF_OPS;
    SNAP_OFF_OP_FINISH_TIME      = 2 * BUF_MACHINES + 2 * BUF_OPS;
    SNAP_OFF_PRECEDENCE_READY    = 2 * BUF_MACHINES + 3 * BUF_OPS;
    SNAP_OFF_READY_COUNT         = 2 * BUF_MACHINES + 4 * BUF_OPS;
    SNAP_OFF_SEQUENCE            = 2 * BUF_MACHINES + 4 * BUF_OPS + 1;
    SNAP_OFF_SEQUENCE_COUNT      = 2 * BUF_MACHINES + 5 * BUF_OPS + 1;
    SNAP_OFF_MAKESPAN            = 2 * BUF_MACHINES + 5 * BUF_OPS + 2;
    SNAPSHOT_INTS                = 2 * BUF_MACHINES + 5 * BUF_OPS + 3;
}

// Host-side representation of a snapshot
struct ScheduleSnapshot {
    int machine_occupied_op[HARD_MAX_MACHINES];
    int machine_ready_time [HARD_MAX_MACHINES];
    int critical_parent_op [HARD_MAX_OPS];
    int op_start_time      [HARD_MAX_OPS];
    int op_finish_time     [HARD_MAX_OPS];
    int precedence_ready_ops[HARD_MAX_OPS];
    int num_precedence_ready;
    int seq                [HARD_MAX_OPS];
    int num_scheduled;
    int makespan;
};

// Serialize host snapshot to flat GPU array
static void pack_snapshot(const ScheduleSnapshot &s, int *flat) {
    memcpy(flat + SNAP_OFF_MACHINE_OCCUPIED_OP, s.machine_occupied_op,  BUF_MACHINES * sizeof(int));
    memcpy(flat + SNAP_OFF_MACHINE_READY_TIME,  s.machine_ready_time,   BUF_MACHINES * sizeof(int));
    memcpy(flat + SNAP_OFF_CRITICAL_PARENT_OP,  s.critical_parent_op,   BUF_OPS      * sizeof(int));
    memcpy(flat + SNAP_OFF_OP_START_TIME,       s.op_start_time,        BUF_OPS      * sizeof(int));
    memcpy(flat + SNAP_OFF_OP_FINISH_TIME,      s.op_finish_time,       BUF_OPS      * sizeof(int));
    memcpy(flat + SNAP_OFF_PRECEDENCE_READY,    s.precedence_ready_ops, BUF_OPS      * sizeof(int));
    flat[SNAP_OFF_READY_COUNT]    = s.num_precedence_ready;
    memcpy(flat + SNAP_OFF_SEQUENCE,            s.seq,                  BUF_OPS      * sizeof(int));
    flat[SNAP_OFF_SEQUENCE_COUNT] = s.num_scheduled;
    flat[SNAP_OFF_MAKESPAN]       = s.makespan;
}

// Host configuration and instance data
static char instance_path[256]   = "./la33.txt";
static int  beam_width            = 1000;
static int  global_best_makespan  = 500000;

static int num_jobs      = 0;
static int num_machines  = 0;
static int num_ops_total = 0;

static OperationInfo all_ops[HARD_MAX_OPS];
static vector<int>   ops_by_machine[HARD_MAX_MACHINES];
static int machine_op_order[HARD_MAX_MACHINES * HARD_MAX_JOBS];
static int machine_op_count[HARD_MAX_MACHINES];

// Pinned host buffer for the beam array
static int *h_input_flat = nullptr;

// GPU device buffers for beam search and CUB sorting
static int    *d_beam_snapshots    = nullptr;
static int    *d_beam_snapshots_2  = nullptr; // Double buffer for materialize step
static float  *d_child_scores      = nullptr;
static float  *d_child_scores_out  = nullptr;
static int    *d_child_indices     = nullptr;
static int    *d_child_indices_out = nullptr;
static void   *d_cub_temp_storage  = nullptr;
static size_t  cub_temp_bytes      = 0;
static int    *d_global_best_makespan = nullptr;
static int    *d_best_solution        = nullptr;

// Host copies of the best schedule
static int h_global_best_seq[HARD_MAX_OPS];
static int h_global_op_start[HARD_MAX_OPS];
static int h_global_op_finish[HARD_MAX_OPS];

// Verifies complete schedule against BJSSP constraints on CPU
static bool cpu_validate_schedule(const int* seq, int num_ops_scheduled,
                                   int num_machines_val, int num_ops_tot)
{
    if (num_ops_scheduled == 0) return false;

    bool valid = true;

    for (int i = 1; i < num_ops_tot - 1; i++) {
        int start_i    = h_global_op_start[i];
        int duration_i = all_ops[i].duration;
        int machine_i  = all_ops[i].machine_id;

        // Machine stays blocked until successor starts
        int finish_i;
        bool is_last_in_job = (i % num_machines_val == 0);
        if (!is_last_in_job) {
            finish_i = h_global_op_start[i + 1];
        } else {
            finish_i = start_i + duration_i;
        }

        // Check processing time fits
        if (finish_i < start_i + duration_i) {
            printf("VALIDATION FAILED: Op %d completes before its processing time "
                   "(start=%d, duration=%d, finish=%d)\n",
                   i, start_i, duration_i, finish_i);
            valid = false;
        }

        // Check machine overlap
        for (int j = i + 1; j < num_ops_tot - 1; j++) {
            if (all_ops[j].machine_id != machine_i) continue;

            int start_j  = h_global_op_start[j];
            int finish_j = (j % num_machines_val != 0)
                           ? h_global_op_start[j + 1]
                           : start_j + all_ops[j].duration;

            if (!(finish_i <= start_j || finish_j <= start_i)) {
                printf("VALIDATION FAILED: Machine %d overlap – Op %d [%d,%d) vs Op %d [%d,%d)\n",
                       machine_i, i, start_i, finish_i, j, start_j, finish_j);
                valid = false;
            }
        }
    }

    // Recompute makespan
    int verified_makespan = 0;
    for (int i = 1; i < num_ops_tot - 1; i++) {
        int finish_i = (i % num_machines_val != 0)
                       ? h_global_op_start[i + 1]
                       : h_global_op_start[i] + all_ops[i].duration;
        if (finish_i > verified_makespan) {
            verified_makespan = finish_i;
            if (valid)
                printf("Makespan pushed to %d by Op %d (start=%d)\n",
                       verified_makespan, i, h_global_op_start[i]);
        }
    }

    if (valid)
        printf("VALIDATION SUCCESS: BJSSP constraints verified. Makespan = %d\n",
               verified_makespan);
    return valid;
}

// Linear array search
__device__ static int dev_find_in_array(const int *arr, int len, int val) {
    for (int i = 0; i < len; i++) if (arr[i] == val) return i;
    return -1;
}

// Get rank of op_id on its assigned machine
__device__ static int find_op_rank_on_machine(int op_id, int machine, int machine_order_stride, const int *d_op_order_ptr) {
    int num_ops_on_machine = c_op_count[machine];
    for (int rank = 0; rank < num_ops_on_machine; rank++)
        if (__ldg(&d_op_order_ptr[machine * machine_order_stride + rank]) == op_id)
            return rank;
    return -1;
}

// Schedule chosen operation respecting BJSSP blocking rules
__device__ static void traiter(
    int op_id,
    int *machine_occupied_op, int *machine_ready_time, int *critical_parent_op,
    int *op_start_time, int *op_finish_time,
    int *precedence_ready_ops, int &num_ready,
    int *seq, int &num_scheduled,
    int num_machines_1idx, int num_ops_tot, int machine_order_stride,
    const int *d_op_order_ptr
) {
    int cascade_stack[HARD_MAX_OPS];
    int cascade_top = 0;
    int current = op_id;

    do {
        int machine = c_ops[current].machine_id;
        seq[num_scheduled++] = current;

        // Determine earliest legal start time respecting BJSSP blocking rules
        int start;
        bool is_first_in_job = (current % num_machines_1idx == 1);

        if (!is_first_in_job) {
            int predecessor_op     = current - 1;
            int predecessor_machine = c_ops[predecessor_op].machine_id;

            if (machine_ready_time[machine] <= machine_ready_time[predecessor_machine]) {
                // Wait for job-predecessor machine to release
                start = machine_ready_time[predecessor_machine];
                critical_parent_op[current] = predecessor_op;
            } else {
                // Wait for our machine to be free
                start = machine_ready_time[machine];
                int rank_on_machine = find_op_rank_on_machine(current, machine, machine_order_stride, d_op_order_ptr);
                int prev_op_on_same_machine = (rank_on_machine > 0)
                    ? __ldg(&d_op_order_ptr[machine * machine_order_stride + rank_on_machine - 1])
                    : -100;
                if (prev_op_on_same_machine != -100 &&
                    prev_op_on_same_machine % num_machines_1idx != 0)
                    prev_op_on_same_machine++;
                critical_parent_op[current] = prev_op_on_same_machine;
            }
        } else {
            // First job op relies only on machine schedule
            int rank_on_machine = find_op_rank_on_machine(current, machine, machine_order_stride, d_op_order_ptr);
            if (rank_on_machine > 0) {
                int prev_op_on_same_machine =
                    __ldg(&d_op_order_ptr[machine * machine_order_stride + rank_on_machine - 1]);
                if (prev_op_on_same_machine % num_machines_1idx != 0)
                    prev_op_on_same_machine++;
                critical_parent_op[current] = prev_op_on_same_machine;
            } else {
                critical_parent_op[current] = 0;
            }
            start = machine_ready_time[machine];
        }

        // Commit schedule and block machine
        op_start_time[current]       = start;
        machine_ready_time[machine]  = start + c_ops[current].duration;
        machine_occupied_op[machine] = current;

        // Release job-predecessor's machine
        if (!is_first_in_job) {
            int predecessor_machine = c_ops[current - 1].machine_id;
            machine_occupied_op[predecessor_machine] = 0;
            machine_ready_time[predecessor_machine]  = start;
        }

        // Unlock next job op or update makespan
        bool is_last_in_job = (current % num_machines_1idx == 0);
        if (!is_last_in_job) {
            precedence_ready_ops[num_ready++] = current + 1;
        } else {
            op_finish_time[current] = machine_ready_time[machine];
            if (op_finish_time[current] > op_start_time[num_ops_tot - 1])
                op_start_time[num_ops_tot - 1] = op_finish_time[current];
            machine_occupied_op[machine] = 0;
        }

        current = (cascade_top > 0) ? cascade_stack[--cascade_top] : -1;
    } while (current != -1);
}

// Schedule an operation involved in a deadlock cycle
__device__ static void traiter_swap(
    int op_id,
    int *machine_occupied_op, int *machine_ready_time,
    int *precedence_ready_ops, int &num_ready,
    int *seq, int &num_scheduled,
    int num_machines_1idx, int *op_start_time
) {
    int machine = c_ops[op_id].machine_id;
    seq[num_scheduled++] = op_id;

    int sync_start       = machine_ready_time[machine];
    op_start_time[op_id] = sync_start;
    machine_ready_time[machine]  = sync_start + c_ops[op_id].duration;
    machine_occupied_op[machine] = op_id;

    bool is_last_in_job = (op_id % num_machines_1idx == 0);
    if (!is_last_in_job)
        precedence_ready_ops[num_ready++] = op_id + 1;
    else
        machine_occupied_op[machine] = 0;
}

// Detect largest deadlock cycle (circular blocking chain)
__device__ static int solinit_swap_new(
    const int *machine_occupied_op,
    int *out_cycle,
    int num_machines_1idx, int num_ops_tot, int num_machines_actual
) {
    int best_cycle_len = 0;

    for (int start_machine = 1; start_machine <= num_machines_actual; start_machine++) {
        int op_on_start = machine_occupied_op[start_machine];
        if (op_on_start == 0) continue;

        int next_op_in_job = op_on_start + 1;
        if (next_op_in_job <= 0 || next_op_in_job >= num_ops_tot) continue;

        int next_machine_needed = c_ops[next_op_in_job].machine_id;
        if (machine_occupied_op[next_machine_needed] == 0) continue;

        // Follow required next-machine chain
        int chain[HARD_MAX_MACHINES];
        int chain_len = 0;
        chain[chain_len++] = start_machine;
        int current_machine = start_machine;
        int cycle_start_idx = -1;

        do {
            int op_cur        = machine_occupied_op[current_machine];
            int next_op       = op_cur + 1;
            if (next_op <= 0 || next_op >= num_ops_tot) { chain_len = 0; break; }

            int needed_machine   = c_ops[next_op].machine_id;
            int op_on_needed     = machine_occupied_op[needed_machine];
            int next_op_needed_2 = op_on_needed + 1;

            bool needed_machine_still_blocked =
                (op_on_needed != 0) &&
                (op_on_needed < num_ops_tot) &&
                (next_op_needed_2 > 0) &&
                (next_op_needed_2 < num_ops_tot) &&
                (machine_occupied_op[c_ops[next_op_needed_2].machine_id] > 0);

            if (!needed_machine_still_blocked) { chain_len = 0; break; }

            cycle_start_idx = dev_find_in_array(chain, chain_len, needed_machine);
            if (cycle_start_idx == -1) {
                if (chain_len < HARD_MAX_MACHINES) chain[chain_len++] = needed_machine;
                current_machine = needed_machine;
            }
        } while (cycle_start_idx == -1 && chain_len < HARD_MAX_MACHINES);

        if (cycle_start_idx >= 0) {
            int cycle_len = chain_len - cycle_start_idx;
            if (cycle_len >= 2 && cycle_len > best_cycle_len) {
                best_cycle_len = cycle_len;
                for (int i = 0; i < cycle_len; i++)
                    out_cycle[i] = chain[cycle_start_idx + i];
            }
        }
    }
    return best_cycle_len;
}

// Resolve deadlock by synchronizing blocked machines
__device__ static void check_swap(
    int *machine_occupied_op, int *machine_ready_time, int *critical_parent_op,
    int *precedence_ready_ops, int &num_ready,
    int *seq, int &num_scheduled,
    int num_machines_1idx, int num_ops_tot, int num_machines_actual,
    int *op_start_time
) {
    int cycle_machines[HARD_MAX_MACHINES];
    int cycle_len = solinit_swap_new(machine_occupied_op, cycle_machines,
                                     num_machines_1idx, num_ops_tot, num_machines_actual);
    if (cycle_len <= 1) return;

    // Sync time is latest ready time in cycle
    int sync_time = 0, pivot_op_id = 0;
    for (int i = 0; i < cycle_len; i++) {
        int mc = cycle_machines[i];
        if (mc != 0 && machine_ready_time[mc] > sync_time) {
            sync_time    = machine_ready_time[mc];
            pivot_op_id  = machine_occupied_op[mc];
        }
    }
    pivot_op_id++;

    for (int i = 0; i < cycle_len; i++)
        if (cycle_machines[i] != 0)
            machine_ready_time[cycle_machines[i]] = sync_time;

    int blocked_ops[HARD_MAX_MACHINES];
    for (int i = 0; i < cycle_len; i++) {
        int mc = cycle_machines[i];
        blocked_ops[i] = (mc != 0) ? machine_occupied_op[mc] + 1 : 0;
    }

    // Update critical path
    for (int i = 0; i < cycle_len; i++) {
        int op_id = blocked_ops[i];
        if (op_id == 0) continue;
        int mc = c_ops[op_id].machine_id;
        critical_parent_op[op_id] = (op_id == pivot_op_id)
            ? pivot_op_id - 1
            : machine_occupied_op[mc] + 1;
    }

    // Schedule cycle ops simultaneously
    for (int i = 0; i < cycle_len; i++) {
        int op_id = blocked_ops[i];
        if (op_id == 0) continue;
        int pos = dev_find_in_array(precedence_ready_ops, num_ready, op_id);
        if (pos >= 0)
            precedence_ready_ops[pos] = precedence_ready_ops[--num_ready];

        traiter_swap(op_id, machine_occupied_op, machine_ready_time,
                     precedence_ready_ops, num_ready,
                     seq, num_scheduled,
                     num_machines_1idx, op_start_time);
    }
}

// Phase 1 Kernel: Expand beam candidates and compute SCORES ONLY (1 thread per branch evaluated sequentially in a grid loop)
__global__ void kernel_score_children(
    const int  *d_in_beam,
    float      *d_out_scores,
    int        *d_global_best,
    int        *d_best_sol,
    const int  *d_op_order_ptr,
    int         num_candidates,
    int         num_branches_per_candidate,
    int         SNAPSHOT_INTS_val,
    int off_machine_occupied, int off_machine_ready, int off_critical_parent,
    int off_op_start, int off_op_finish, int off_precedence_ready, int off_ready_count,
    int off_sequence, int off_sequence_count, int off_makespan,
    int num_machines_1idx,
    int num_ops_tot,
    int total_real_ops,
    int machine_order_stride,
    int random_seed
) {
    for (int candidate_id = blockIdx.x * blockDim.x + threadIdx.x; 
         candidate_id < num_candidates; 
         candidate_id += blockDim.x * gridDim.x) {

        int parent_base = candidate_id * SNAPSHOT_INTS_val;

        // Initialize scores for all potential branches to Infinity
        for (int b = 0; b < num_branches_per_candidate; b++) {
            d_out_scores[candidate_id * num_branches_per_candidate + b] = 1e9f;
        }

        // Determine eligible ops
        int free_ops_buf[HARD_MAX_JOBS];
        int num_free = 0;
        int num_ready = d_in_beam[parent_base + off_ready_count];

        for (int i = 0; i < num_ready; i++) {
            int op = d_in_beam[parent_base + off_precedence_ready + i];
            if (op >= 1 && op < num_ops_tot &&
                d_in_beam[parent_base + off_machine_occupied + c_ops[op].machine_id] == 0)
                free_ops_buf[num_free++] = op;
        }

        int candidate_ops[HARD_MAX_JOBS];
        int num_valid_ops = 0;

        if (num_free > 0) {
            unsigned int hash = random_seed + candidate_id;
            hash ^= hash >> 16; hash *= 0x7feb352d;
            hash ^= hash >> 15; hash *= 0x846ca68b;
            hash ^= hash >> 16;
            int chosen_machine = c_ops[free_ops_buf[hash % num_free]].machine_id;

            for (int i = 0; i < num_ready; i++) {
                int op = d_in_beam[parent_base + off_precedence_ready + i];
                if (op >= 1 && op < num_ops_tot &&
                    c_ops[op].machine_id == chosen_machine &&
                    d_in_beam[parent_base + off_machine_occupied + chosen_machine] == 0)
                    candidate_ops[num_valid_ops++] = op;
            }
        }

        // Allocate thread-local state ONCE
        int machine_occupied_op[HARD_MAX_MACHINES];
        int machine_ready_time [HARD_MAX_MACHINES];
        int critical_parent_op [HARD_MAX_OPS];
        int op_start_time      [HARD_MAX_OPS];
        int op_finish_time     [HARD_MAX_OPS];
        int precedence_ready_ops[HARD_MAX_OPS];
        int seq                [HARD_MAX_OPS];

        // Deadlock fallback branch
        if (num_valid_ops == 0) {
            for (int m = 0; m <= num_machines_1idx; m++) {
                machine_occupied_op[m] = d_in_beam[parent_base + off_machine_occupied + m];
                machine_ready_time[m]  = d_in_beam[parent_base + off_machine_ready    + m];
            }
            for (int op = 0; op < num_ops_tot; op++) {
                critical_parent_op[op] = d_in_beam[parent_base + off_critical_parent + op];
                op_start_time[op]      = d_in_beam[parent_base + off_op_start        + op];
                op_finish_time[op]     = d_in_beam[parent_base + off_op_finish       + op];
            }
            int cur_num_ready = num_ready;
            for (int i = 0; i < cur_num_ready; i++)
                precedence_ready_ops[i] = d_in_beam[parent_base + off_precedence_ready + i];
            int num_scheduled = d_in_beam[parent_base + off_sequence_count];
            for (int i = 0; i < num_scheduled; i++)
                seq[i] = d_in_beam[parent_base + off_sequence + i];

            check_swap(machine_occupied_op, machine_ready_time, critical_parent_op,
                       precedence_ready_ops, cur_num_ready,
                       seq, num_scheduled,
                       num_machines_1idx, num_ops_tot, num_machines_1idx, op_start_time);

            int cmax = 0;
            for (int m = 1; m <= num_machines_1idx; m++)
                if (machine_ready_time[m] > cmax) cmax = machine_ready_time[m];

            if (cmax >= *d_global_best) continue;

            float base_score = (num_scheduled == 0) ? (float)cmax : ((float)cmax / (float)num_scheduled);
            unsigned int noise_hash = random_seed ^ (candidate_id * 0x85ebca6b);
            noise_hash ^= noise_hash >> 16; noise_hash *= 0x85ebca6b;
            noise_hash ^= noise_hash >> 16; noise_hash *= 0xc2b2ae35;
            noise_hash ^= noise_hash >> 16;
            float sibling_noise = (float)(noise_hash % 1000) / 1000000000.0f;
            d_out_scores[candidate_id * num_branches_per_candidate] = base_score + sibling_noise;

            if (num_scheduled == total_real_ops) {
                int old_best = atomicMin(d_global_best, cmax);
                if (cmax < old_best) {
                    d_best_sol[off_makespan] = cmax;
                    d_best_sol[off_sequence_count] = num_scheduled;
                    d_best_sol[off_ready_count] = cur_num_ready;
                    for (int m = 0; m <= num_machines_1idx; m++) {
                        d_best_sol[off_machine_occupied + m] = machine_occupied_op[m];
                        d_best_sol[off_machine_ready    + m] = machine_ready_time[m];
                    }
                    for (int op = 0; op < num_ops_tot; op++) {
                        d_best_sol[off_critical_parent + op] = critical_parent_op[op];
                        d_best_sol[off_op_start        + op] = op_start_time[op];
                        d_best_sol[off_op_finish       + op] = op_finish_time[op];
                    }
                    for (int i = 0; i < cur_num_ready; i++)
                        d_best_sol[off_precedence_ready + i] = precedence_ready_ops[i];
                    for (int i = 0; i < num_scheduled; i++)
                        d_best_sol[off_sequence + i] = seq[i];
                }
            }
            continue;
        }

        // Process valid branches sequentially
        for (int branch_id = 0; branch_id < num_valid_ops; branch_id++) {
            // Hydrate state from parent
            for (int m = 0; m <= num_machines_1idx; m++) {
                machine_occupied_op[m] = d_in_beam[parent_base + off_machine_occupied + m];
                machine_ready_time[m]  = d_in_beam[parent_base + off_machine_ready    + m];
            }
            for (int op = 0; op < num_ops_tot; op++) {
                critical_parent_op[op] = d_in_beam[parent_base + off_critical_parent + op];
                op_start_time[op]      = d_in_beam[parent_base + off_op_start        + op];
                op_finish_time[op]     = d_in_beam[parent_base + off_op_finish       + op];
            }
            int cur_num_ready = num_ready;
            for (int i = 0; i < cur_num_ready; i++)
                precedence_ready_ops[i] = d_in_beam[parent_base + off_precedence_ready + i];
            int num_scheduled = d_in_beam[parent_base + off_sequence_count];
            for (int i = 0; i < num_scheduled; i++)
                seq[i] = d_in_beam[parent_base + off_sequence + i];

            int chosen_op = candidate_ops[branch_id];
            int ready_pos = dev_find_in_array(precedence_ready_ops, cur_num_ready, chosen_op);
            if (ready_pos >= 0)
                precedence_ready_ops[ready_pos] = precedence_ready_ops[--cur_num_ready];

            traiter(chosen_op,
                    machine_occupied_op, machine_ready_time, critical_parent_op,
                    op_start_time, op_finish_time,
                    precedence_ready_ops, cur_num_ready,
                    seq, num_scheduled,
                    num_machines_1idx, num_ops_tot, machine_order_stride, d_op_order_ptr);

            check_swap(machine_occupied_op, machine_ready_time, critical_parent_op,
                       precedence_ready_ops, cur_num_ready,
                       seq, num_scheduled,
                       num_machines_1idx, num_ops_tot, num_machines_1idx, op_start_time);

            int cmax = 0;
            for (int m = 1; m <= num_machines_1idx; m++)
                if (machine_ready_time[m] > cmax) cmax = machine_ready_time[m];

            if (cmax >= *d_global_best) continue;

            float base_score = (num_scheduled == 0) ? (float)cmax : ((float)cmax / (float)num_scheduled);
            unsigned int noise_hash = random_seed ^ (candidate_id * 0x85ebca6b) ^ (branch_id * 0xc2b2ae35);
            noise_hash ^= noise_hash >> 16; noise_hash *= 0x85ebca6b;
            noise_hash ^= noise_hash >> 16; noise_hash *= 0xc2b2ae35;
            noise_hash ^= noise_hash >> 16;
            float sibling_noise = (float)(noise_hash % 1000) / 1000000000.0f;
            d_out_scores[candidate_id * num_branches_per_candidate + branch_id] = base_score + sibling_noise;

            if (num_scheduled == total_real_ops) {
                int old_best = atomicMin(d_global_best, cmax);
                if (cmax < old_best) {
                    d_best_sol[off_makespan] = cmax;
                    d_best_sol[off_sequence_count] = num_scheduled;
                    d_best_sol[off_ready_count] = cur_num_ready;
                    for (int m = 0; m <= num_machines_1idx; m++) {
                        d_best_sol[off_machine_occupied + m] = machine_occupied_op[m];
                        d_best_sol[off_machine_ready    + m] = machine_ready_time[m];
                    }
                    for (int op = 0; op < num_ops_tot; op++) {
                        d_best_sol[off_critical_parent + op] = critical_parent_op[op];
                        d_best_sol[off_op_start        + op] = op_start_time[op];
                        d_best_sol[off_op_finish       + op] = op_finish_time[op];
                    }
                    for (int i = 0; i < cur_num_ready; i++)
                        d_best_sol[off_precedence_ready + i] = precedence_ready_ops[i];
                    for (int i = 0; i < num_scheduled; i++)
                        d_best_sol[off_sequence + i] = seq[i];
                }
            }
        }
    }
}

// Phase 2 Kernel: Materialize ONLY the top-K child snapshots directly into the new beam buffer
__global__ void kernel_materialize_topk(
    const int *d_in_beam,
    const int *d_sorted_indices,
    int       *d_out_beam,
    int        num_to_gather,
    int        num_branches_per_candidate,
    int        SNAPSHOT_INTS_val,
    int off_machine_occupied, int off_machine_ready, int off_critical_parent,
    int off_op_start, int off_op_finish, int off_precedence_ready, int off_ready_count,
    int off_sequence, int off_sequence_count, int off_makespan,
    int num_machines_1idx,
    int num_ops_tot,
    int machine_order_stride,
    int random_seed,
    const int *d_op_order_ptr
) {
    for (int rank = blockIdx.x * blockDim.x + threadIdx.x; 
         rank < num_to_gather; 
         rank += blockDim.x * gridDim.x) {

        int src_child_idx = d_sorted_indices[rank];
        int candidate_id  = src_child_idx / num_branches_per_candidate;
        int target_branch = src_child_idx % num_branches_per_candidate;

        int parent_base = candidate_id * SNAPSHOT_INTS_val;

        // Recompute eligible ops to find the chosen_op
        int free_ops_buf[HARD_MAX_JOBS];
        int num_free = 0;
        int num_ready = d_in_beam[parent_base + off_ready_count];

        for (int i = 0; i < num_ready; i++) {
            int op = d_in_beam[parent_base + off_precedence_ready + i];
            if (op >= 1 && op < num_ops_tot &&
                d_in_beam[parent_base + off_machine_occupied + c_ops[op].machine_id] == 0)
                free_ops_buf[num_free++] = op;
        }

        int candidate_ops[HARD_MAX_JOBS];
        int num_valid_ops = 0;

        if (num_free > 0) {
            unsigned int hash = random_seed + candidate_id;
            hash ^= hash >> 16; hash *= 0x7feb352d;
            hash ^= hash >> 15; hash *= 0x846ca68b;
            hash ^= hash >> 16;
            int chosen_machine = c_ops[free_ops_buf[hash % num_free]].machine_id;

            for (int i = 0; i < num_ready; i++) {
                int op = d_in_beam[parent_base + off_precedence_ready + i];
                if (op >= 1 && op < num_ops_tot &&
                    c_ops[op].machine_id == chosen_machine &&
                    d_in_beam[parent_base + off_machine_occupied + chosen_machine] == 0)
                    candidate_ops[num_valid_ops++] = op;
            }
        }

        int out_base = rank * SNAPSHOT_INTS_val;

        // Local arrays for hydration and simulation
        int machine_occupied_op[HARD_MAX_MACHINES];
        int machine_ready_time [HARD_MAX_MACHINES];
        int critical_parent_op [HARD_MAX_OPS];
        int op_start_time      [HARD_MAX_OPS];
        int op_finish_time     [HARD_MAX_OPS];
        int precedence_ready_ops[HARD_MAX_OPS];
        int seq                [HARD_MAX_OPS];

        // Hydrate from parent
        for (int m = 0; m <= num_machines_1idx; m++) {
            machine_occupied_op[m] = d_in_beam[parent_base + off_machine_occupied + m];
            machine_ready_time[m]  = d_in_beam[parent_base + off_machine_ready    + m];
        }
        for (int op = 0; op < num_ops_tot; op++) {
            critical_parent_op[op] = d_in_beam[parent_base + off_critical_parent + op];
            op_start_time[op]      = d_in_beam[parent_base + off_op_start        + op];
            op_finish_time[op]     = d_in_beam[parent_base + off_op_finish       + op];
        }
        int cur_num_ready = num_ready;
        for (int i = 0; i < cur_num_ready; i++)
            precedence_ready_ops[i] = d_in_beam[parent_base + off_precedence_ready + i];
        int num_scheduled = d_in_beam[parent_base + off_sequence_count];
        for (int i = 0; i < num_scheduled; i++)
            seq[i] = d_in_beam[parent_base + off_sequence + i];

        if (num_valid_ops == 0) {
            check_swap(machine_occupied_op, machine_ready_time, critical_parent_op,
                       precedence_ready_ops, cur_num_ready,
                       seq, num_scheduled,
                       num_machines_1idx, num_ops_tot, num_machines_1idx, op_start_time);
        } else {
            int chosen_op = candidate_ops[target_branch];
            int ready_pos = dev_find_in_array(precedence_ready_ops, cur_num_ready, chosen_op);
            if (ready_pos >= 0)
                precedence_ready_ops[ready_pos] = precedence_ready_ops[--cur_num_ready];

            traiter(chosen_op,
                    machine_occupied_op, machine_ready_time, critical_parent_op,
                    op_start_time, op_finish_time,
                    precedence_ready_ops, cur_num_ready,
                    seq, num_scheduled,
                    num_machines_1idx, num_ops_tot, machine_order_stride, d_op_order_ptr);

            check_swap(machine_occupied_op, machine_ready_time, critical_parent_op,
                       precedence_ready_ops, cur_num_ready,
                       seq, num_scheduled,
                       num_machines_1idx, num_ops_tot, num_machines_1idx, op_start_time);
        }

        int cmax = 0;
        for (int m = 1; m <= num_machines_1idx; m++)
            if (machine_ready_time[m] > cmax) cmax = machine_ready_time[m];

        d_out_beam[out_base + off_makespan]       = cmax;
        d_out_beam[out_base + off_sequence_count] = num_scheduled;
        d_out_beam[out_base + off_ready_count]    = cur_num_ready;
        for (int m = 0; m <= num_machines_1idx; m++) {
            d_out_beam[out_base + off_machine_occupied + m] = machine_occupied_op[m];
            d_out_beam[out_base + off_machine_ready    + m] = machine_ready_time[m];
        }
        for (int op = 0; op < num_ops_tot; op++) {
            d_out_beam[out_base + off_critical_parent + op] = critical_parent_op[op];
            d_out_beam[out_base + off_op_start        + op] = op_start_time[op];
            d_out_beam[out_base + off_op_finish       + op] = op_finish_time[op];
        }
        for (int i = 0; i < cur_num_ready; i++)
            d_out_beam[out_base + off_precedence_ready + i] = precedence_ready_ops[i];
        for (int i = 0; i < num_scheduled; i++)
            d_out_beam[out_base + off_sequence + i] = seq[i];
    }
}

// Initialize identity permutation for sorting
__global__ void init_indices(int *indices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) indices[i] = i;
}

// Upload data to constant memory and allocate GPU buffers
static void allocate_gpu_buffers() {
    CUDA_CHECK(cudaMemcpyToSymbol(c_ops,      all_ops,
                                  HARD_MAX_OPS                      * sizeof(OperationInfo)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_op_count, machine_op_count,
                                  HARD_MAX_MACHINES                 * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_op_order, HARD_MAX_MACHINES * HARD_MAX_JOBS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_op_order, machine_op_order,
                          HARD_MAX_MACHINES * HARD_MAX_JOBS * sizeof(int), cudaMemcpyHostToDevice));

    size_t beam_bytes  = (size_t)BUF_BEAM * SNAPSHOT_INTS * sizeof(int);
    size_t score_bytes = (size_t)BUF_BEAM * BUF_JOBS * sizeof(float);
    size_t index_bytes = (size_t)BUF_BEAM * BUF_JOBS * sizeof(int);

    // Ensure we have enough VRAM
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    // Calculate total requested memory
    // CUB sort bytes is typically a few MBs
    size_t total_needed = beam_bytes * 2 + score_bytes * 2 + index_bytes * 2 + (10*1024*1024);
    
    printf("\n--- Memory Check ---\n");
    printf("Requested Beam Width: %d\n", BUF_BEAM);
    printf("Estimated VRAM required: %.2f MB\n", (double)total_needed / (1024*1024));
    printf("Available VRAM: %.2f MB\n", (double)free_mem / (1024*1024));
    
    if (total_needed > free_mem) {
        fprintf(stderr, "\nWARNING: Insufficient VRAM! You need %.2f MB but only have %.2f MB free.\n", 
                (double)total_needed / (1024*1024), (double)free_mem / (1024*1024));
        fprintf(stderr, "Consider reducing beam_width or freeing up GPU memory.\n\n");
        exit(EXIT_FAILURE);
    } else {
        printf("VRAM Check passed.\n\n");
    }

    CUDA_CHECK(cudaMalloc(&d_beam_snapshots,    beam_bytes));
    CUDA_CHECK(cudaMalloc(&d_beam_snapshots_2,  beam_bytes));
    CUDA_CHECK(cudaMalloc(&d_child_scores,      score_bytes));
    CUDA_CHECK(cudaMalloc(&d_child_scores_out,  score_bytes));
    CUDA_CHECK(cudaMalloc(&d_child_indices,     index_bytes));
    CUDA_CHECK(cudaMalloc(&d_child_indices_out, index_bytes));
    CUDA_CHECK(cudaMalloc(&d_global_best_makespan, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_solution, SNAPSHOT_INTS * sizeof(int)));

    // Determine CUB radix sort scratch space
    cub::DeviceRadixSort::SortPairs(
        nullptr, cub_temp_bytes,
        d_child_scores, d_child_scores_out,
        d_child_indices, d_child_indices_out,
        BUF_BEAM * BUF_JOBS
    );
    CUDA_CHECK(cudaMalloc(&d_cub_temp_storage, cub_temp_bytes));

    CUDA_CHECK(cudaMallocHost(&h_input_flat, SNAPSHOT_INTS * sizeof(int)));
}

static void free_gpu_buffers() {
    cudaFree(d_op_order);
    cudaFree(d_beam_snapshots);
    cudaFree(d_beam_snapshots_2);
    cudaFree(d_child_scores);
    cudaFree(d_child_scores_out);
    cudaFree(d_child_indices);
    cudaFree(d_child_indices_out);
    cudaFree(d_cub_temp_storage);
    cudaFree(d_global_best_makespan);
    cudaFree(d_best_solution);
    cudaFreeHost(h_input_flat);
}

// Read OR-Library instance file
static void load_instance() {
    FILE *f = fopen(instance_path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", instance_path); exit(1); }

    fscanf(f, "%d %d\n", &num_jobs, &num_machines);
    num_ops_total = num_jobs * num_machines + 2;

    BUF_OPS      = num_ops_total;
    BUF_MACHINES = num_machines + 1;
    BUF_JOBS     = num_jobs;

    memset(all_ops, 0, sizeof(all_ops));
    all_ops[0] = {0, 0, 0};

    int op_id = 1;
    for (int job = 1; job <= num_jobs; job++) {
        for (int step = 1; step <= num_machines; step++) {
            int raw_machine_id, duration;
            fscanf(f, "%d %d", &raw_machine_id, &duration);
            int machine_1idx = raw_machine_id + 1;
            all_ops[op_id] = {machine_1idx, duration, job};
            ops_by_machine[machine_1idx].push_back(op_id);
            op_id++;
        }
    }
    all_ops[num_jobs * num_machines + 1] = {0, 0, 0};
    fclose(f);

    memset(machine_op_order, 0, sizeof(machine_op_order));
    memset(machine_op_count, 0, sizeof(machine_op_count));
    for (int m = 1; m <= num_machines; m++) {
        int n = (int)ops_by_machine[m].size();
        machine_op_count[m] = n;
        for (int rank = 0; rank < n; rank++)
            machine_op_order[m * HARD_MAX_JOBS + rank] = ops_by_machine[m][rank];
    }
}

// Create initial state (first op of each job ready)
static ScheduleSnapshot make_root_snapshot() {
    ScheduleSnapshot s{};
    s.num_precedence_ready = num_jobs;
    for (int j = 0; j < num_jobs; j++)
        s.precedence_ready_ops[j] = j * num_machines + 1;
    s.makespan     = 0;
    s.num_scheduled = 0;
    return s;
}

// Execute one full beam search pass
static int run_beam_construction(int /*iteration_number_unused*/) {
    int pass_best_makespan = global_best_makespan;
    CUDA_CHECK(cudaMemcpy(d_global_best_makespan, &pass_best_makespan,
                          sizeof(int), cudaMemcpyHostToDevice));

    ScheduleSnapshot root = make_root_snapshot();
    pack_snapshot(root, h_input_flat);
    CUDA_CHECK(cudaMemcpy(d_beam_snapshots, h_input_flat,
                          SNAPSHOT_INTS * sizeof(int), cudaMemcpyHostToDevice));

    int current_beam_size = 1;
    int max_depth         = num_jobs * num_machines + 10;
    int total_real_ops    = num_jobs * num_machines;

    for (int depth = 1; depth < max_depth && current_beam_size > 0; depth++) {
        int random_seed = rand();

        int total_children = current_beam_size * BUF_JOBS;

        // Launch grid-stride Phase 1 Kernel (128 threads/block, max 80 blocks ~ 10,240 threads)
        int score_blocks = std::min(80, (current_beam_size + 127) / 128);
        if(score_blocks == 0) score_blocks = 1;
        kernel_score_children<<<score_blocks, 128>>>(
            d_beam_snapshots, d_child_scores,
            d_global_best_makespan, d_best_solution, d_op_order,
            current_beam_size, BUF_JOBS, SNAPSHOT_INTS,
            SNAP_OFF_MACHINE_OCCUPIED_OP, SNAP_OFF_MACHINE_READY_TIME,
            SNAP_OFF_CRITICAL_PARENT_OP,
            SNAP_OFF_OP_START_TIME, SNAP_OFF_OP_FINISH_TIME,
            SNAP_OFF_PRECEDENCE_READY, SNAP_OFF_READY_COUNT,
            SNAP_OFF_SEQUENCE, SNAP_OFF_SEQUENCE_COUNT, SNAP_OFF_MAKESPAN,
            num_machines, num_ops_total, total_real_ops, HARD_MAX_JOBS, random_seed
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Init indices for sorting
        int threads_per_block = 256;
        int index_init_blocks = (total_children + threads_per_block - 1) / threads_per_block;
        init_indices<<<index_init_blocks, threads_per_block>>>(d_child_indices, total_children);

        // Radix sort children by score
        cub::DeviceRadixSort::SortPairs(
            d_cub_temp_storage, cub_temp_bytes,
            d_child_scores,     d_child_scores_out,
            d_child_indices,    d_child_indices_out,
            total_children
        );

        int candidates_to_check = min(total_children, beam_width);
        
        // Dynamically allocated to handle large beam_width without stack overflow
        std::vector<float> h_top_scores(candidates_to_check);
        CUDA_CHECK(cudaMemcpy(h_top_scores.data(), d_child_scores_out,
                              candidates_to_check * sizeof(float), cudaMemcpyDeviceToHost));

        // Count valid unpruned children
        int valid_count = 0;
        while (valid_count < candidates_to_check && h_top_scores[valid_count] < 1e8f)
            valid_count++;

        if (valid_count == 0) break;

        // Launch grid-stride Phase 2 Kernel to materialize top-K
        int materialize_blocks = std::min(80, (valid_count + 127) / 128);
        if(materialize_blocks == 0) materialize_blocks = 1;
        kernel_materialize_topk<<<materialize_blocks, 128>>>(
            d_beam_snapshots, d_child_indices_out, d_beam_snapshots_2,
            valid_count, BUF_JOBS, SNAPSHOT_INTS,
            SNAP_OFF_MACHINE_OCCUPIED_OP, SNAP_OFF_MACHINE_READY_TIME,
            SNAP_OFF_CRITICAL_PARENT_OP,
            SNAP_OFF_OP_START_TIME, SNAP_OFF_OP_FINISH_TIME,
            SNAP_OFF_PRECEDENCE_READY, SNAP_OFF_READY_COUNT,
            SNAP_OFF_SEQUENCE, SNAP_OFF_SEQUENCE_COUNT, SNAP_OFF_MAKESPAN,
            num_machines, num_ops_total, HARD_MAX_JOBS, random_seed, d_op_order
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        current_beam_size = valid_count;
        
        // Swap primary and secondary beam buffers
        std::swap(d_beam_snapshots, d_beam_snapshots_2);
    }

    CUDA_CHECK(cudaMemcpy(&pass_best_makespan, d_global_best_makespan,
                          sizeof(int), cudaMemcpyDeviceToHost));

    if (pass_best_makespan < global_best_makespan) {
        CUDA_CHECK(cudaMemcpy(h_input_flat, d_best_solution,
                              SNAPSHOT_INTS * sizeof(int), cudaMemcpyDeviceToHost));
        if (h_input_flat[SNAP_OFF_SEQUENCE_COUNT] == total_real_ops) {
            for (int i = 0; i < total_real_ops; i++)
                h_global_best_seq[i] = h_input_flat[SNAP_OFF_SEQUENCE + i];
            for (int i = 0; i < num_ops_total; i++) {
                h_global_op_start[i]  = h_input_flat[SNAP_OFF_OP_START_TIME + i];
                h_global_op_finish[i] = h_input_flat[SNAP_OFF_OP_FINISH_TIME + i];
            }
        }
    }

    return pass_best_makespan;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int num_ranks, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        fprintf(stderr, "No GPU found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(my_rank % num_gpus));

    // Disable stdout buffering so output appears immediately
    setbuf(stdout, NULL);

    // Increase GPU stack size for deep array allocations - 96KB
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 98304));

    if (my_rank == 0) {
        printf("\n%-8s %-12s %-6s %-9s %-8s %-12s %-10s\n",
               "Width", "Instance", "Jobs", "Machines", "Ops", "BestSpan", "Time(s)");
        printf("-------------------------------------------------------------------------------\n");
        fflush(stdout);
    }

    const char* instances_to_run[] = {
        "71","72","73","74","75","76","77","78","79","80"
    };

    for (int j = 0; j < 10; j++) {

        // All ranks must agree on instance before proceeding
        MPI_Barrier(MPI_COMM_WORLD);

        instance_path[0] = '\0';
        strcat(instance_path, "../../taillard_instances/ta");
        strcat(instance_path, instances_to_run[j]);
        strcat(instance_path, ".txt");

        load_instance();

        // Sync after load to ensure all ranks have the same instance data
        MPI_Barrier(MPI_COMM_WORLD);

        clock_t wall_start = clock();

        beam_width = 25600;
        BUF_BEAM   = beam_width;

        compute_snapshot_layout();

        // Sync after layout computation before allocating
        MPI_Barrier(MPI_COMM_WORLD);

        allocate_gpu_buffers();

        global_best_makespan = 500000;
        int my_best     = global_best_makespan;
        int shared_best = global_best_makespan;

        srand((unsigned)time(NULL) + my_rank * 1000 + j);

        for (int iter = 0; iter < 50; iter++) {
            int makespan = run_beam_construction(iter);
            if (makespan < my_best) my_best = makespan;

            MPI_Allreduce(&my_best, &shared_best, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            global_best_makespan = shared_best;
        }

        clock_t wall_end = clock();
        double elapsed = (double)(wall_end - wall_start) / CLOCKS_PER_SEC;

        if (my_rank == 0) {
            printf("%-8d %-12s %-6d %-9d %-8d %-12d %-10.2f\n",
                   beam_width,
                   instance_path,
                   num_jobs,
                   num_machines,
                   num_ops_total,
                   global_best_makespan,
                   elapsed);
            fflush(stdout);
        }

        // Ensure all GPU work is done before freeing
        CUDA_CHECK(cudaDeviceSynchronize());

        free_gpu_buffers();

        // Null all pointers after free to prevent double-free segfault
        d_op_order           = NULL;
        d_beam_snapshots     = NULL;
        d_beam_snapshots_2   = NULL;
        d_child_scores       = NULL;
        d_child_scores_out   = NULL;
        d_child_indices      = NULL;
        d_child_indices_out  = NULL;
        d_cub_temp_storage   = NULL;
        d_global_best_makespan = NULL;
        d_best_solution      = NULL;
        h_input_flat         = NULL;

        // Sync all ranks before starting next instance
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (my_rank == 0) {
        printf("\nAll batches finished.\n");
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
