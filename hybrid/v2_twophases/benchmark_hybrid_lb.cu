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
#include <string>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>
#include <atomic>
#include <omp.h>
#include <mutex>

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

// ---- Lower-bound scoring modes (from File 1) ----
enum LbMode {
    LB_CUR = 0,  // cmax / num_scheduled
    LB_M   = 1,  // machine bound
    LB_MJ  = 2,  // machine + job bound
    LB_HT  = 3,  // Head-Tail bound
    LB_CP  = 4,  // Critical-path bound
    LB_N_MODES = 5
};
static const char *lb_mode_name[] = {
    "CUR(cmax/sched)", "LB_M(mach)", "LB_MJ(mach+job)", "LB_HT(head-tail)", "LB_CP(crit-path)"
};
static int g_lb_mode_cli = -1; // -1 = auto, else forced

// Throughput Stats
struct ThroughputStats {
    long long partial_states;
    long long complete_solutions;
    long long beam_iterations;
    double    elapsed_sec;
};
static ThroughputStats rank_stats = {0, 0, 0, 0.0};
static std::mutex stats_mutex;

// Operation descriptor
struct OperationInfo {
    int machine_id;
    int duration;
    int job_id;
};

// Constant memory for instance data
__constant__ OperationInfo c_ops[HARD_MAX_OPS];
__constant__ int c_op_count[HARD_MAX_MACHINES];
__constant__ int c_mach_total[HARD_MAX_MACHINES];
__constant__ int c_job_total[HARD_MAX_JOBS];
__constant__ int c_tail_static[HARD_MAX_OPS];

// Global device memory for op order
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
// New offsets for LB counters:
static int SNAP_OFF_MACH_REM            = 0;
static int SNAP_OFF_JOB_READY           = 0;
static int SNAP_OFF_JOB_REM             = 0;
static int SNAPSHOT_INTS                = 0;

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
    // Append LB arrays
    SNAP_OFF_MACH_REM            = 2 * BUF_MACHINES + 5 * BUF_OPS + 3;
    SNAP_OFF_JOB_READY           = 3 * BUF_MACHINES + 5 * BUF_OPS + 3;
    SNAP_OFF_JOB_REM             = 3 * BUF_MACHINES + 5 * BUF_OPS + 3 + BUF_JOBS;
    SNAPSHOT_INTS                = 3 * BUF_MACHINES + 5 * BUF_OPS + 3 + 2 * BUF_JOBS;
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
    // New LB states
    int mach_rem           [HARD_MAX_MACHINES];
    int job_ready          [HARD_MAX_JOBS];
    int job_rem            [HARD_MAX_JOBS];
};

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
    memcpy(flat + SNAP_OFF_MACH_REM,            s.mach_rem,             BUF_MACHINES * sizeof(int));
    memcpy(flat + SNAP_OFF_JOB_READY,           s.job_ready,            BUF_JOBS     * sizeof(int));
    memcpy(flat + SNAP_OFF_JOB_REM,             s.job_rem,              BUF_JOBS     * sizeof(int));
}

// Host configuration and instance data
static char instance_path[256]   = "./la33.txt";
static int  beam_width            = 1000;
static int  accel_factor          = 0;   // Auto-calibrate if <= 0
static double wall_budget_sec     = 60.0;
static std::atomic<int> g_shared_best(500000);
static int  global_best_makespan  = 500000;

static int num_jobs      = 0;
static int num_machines  = 0;
static int num_ops_total = 0;

static OperationInfo all_ops[HARD_MAX_OPS];
static vector<int>   ops_by_machine[HARD_MAX_MACHINES];
static int machine_op_order[HARD_MAX_MACHINES * HARD_MAX_JOBS];
static int machine_op_count[HARD_MAX_MACHINES];

// Instance-level totals for LB init
static int host_mach_total[HARD_MAX_MACHINES];
static int host_job_total[HARD_MAX_JOBS];
static int host_tail_static[HARD_MAX_OPS];

// Pinned host buffer for the beam array
static int *h_input_flat = nullptr;

// GPU device buffers
static int    *d_beam_snapshots    = nullptr;
static int    *d_beam_snapshots_2  = nullptr;
static float  *d_child_scores      = nullptr;
static float  *d_child_scores_out  = nullptr;
static int    *d_child_indices     = nullptr;
static int    *d_child_indices_out = nullptr;
static void   *d_cub_temp_storage  = nullptr;
static size_t  cub_temp_bytes      = 0;
static int    *d_global_best_makespan = nullptr;
static int    *d_best_solution        = nullptr;

static int h_global_best_seq[HARD_MAX_OPS];
static int h_global_op_start[HARD_MAX_OPS];
static int h_global_op_finish[HARD_MAX_OPS];

static int h_cpu_best_seq  [HARD_MAX_OPS];
static int h_cpu_op_start  [HARD_MAX_OPS];
static int h_cpu_op_finish [HARD_MAX_OPS];
static std::mutex cpu_best_mutex;
static int h_cpu_best_makespan = 500000;

// ---- CPU HELPER FUNCTIONS ----

static int cpu_find_in_array(const int* arr, int len, int val) {
    for (int i = 0; i < len; i++) if (arr[i] == val) return i;
    return -1;
}

static int cpu_find_op_rank(int op_id, int machine) {
    int n = machine_op_count[machine];
    for (int r = 0; r < n; r++)
        if (machine_op_order[machine * HARD_MAX_JOBS + r] == op_id) return r;
    return -1;
}

static int cpu_solinit_swap_new(const int* moc, int* out_cycle, int nm, int nops, int num_mach) {
    int best_len = 0;
    for (int sm = 1; sm <= num_mach; sm++) {
        int op_on = moc[sm]; if (op_on == 0) continue;
        int nxt = op_on + 1; if (nxt <= 0 || nxt >= nops) continue;
        int nmc = all_ops[nxt].machine_id; if (moc[nmc] == 0) continue;
        int chain[HARD_MAX_MACHINES], clen = 0, cstart = -1;
        chain[clen++] = sm; int cur = sm;
        do {
            int oc = moc[cur], nx = oc+1;
            if (nx <= 0 || nx >= nops) { clen = 0; break; }
            int nmc2 = all_ops[nx].machine_id, on2 = moc[nmc2], nx2 = on2+1;
            bool blk = (on2!=0)&&(on2<nops)&&(nx2>0)&&(nx2<nops)&&(moc[all_ops[nx2].machine_id]>0);
            if (!blk) { clen = 0; break; }
            cstart = cpu_find_in_array(chain, clen, nmc2);
            if (cstart == -1) { if (clen < HARD_MAX_MACHINES) chain[clen++] = nmc2; cur = nmc2; }
        } while (cstart == -1 && clen < HARD_MAX_MACHINES);
        if (cstart >= 0) { int cl = clen-cstart; if (cl >= 2 && cl > best_len) { best_len = cl; for (int i=0;i<cl;i++) out_cycle[i] = chain[cstart+i]; } }
    }
    return best_len;
}


// -----------------------------------------------------------------------------
// CPU / GPU LB Functions
// -----------------------------------------------------------------------------

static inline double cpu_compute_score(int cmax, int num_scheduled, int lb_mode, 
                                       const int* mrt, const int* m_rem, const int* j_ready, const int* j_rem,
                                       const int* pr, int nr, int num_ops_tot) {
    if (lb_mode == LB_CUR) {
        return (num_scheduled == 0) ? (double)cmax : ((double)cmax / num_scheduled);
    }
    int mach_LB = cmax;
    for (int m = 1; m <= num_machines; m++) {
        int v = mrt[m] + m_rem[m];
        if (v > mach_LB) mach_LB = v;
    }
    if (lb_mode == LB_M) return (double)mach_LB;

    if (lb_mode == LB_MJ) {
        int lb = mach_LB;
        for (int j = 0; j < num_jobs; j++) {
            int v = j_ready[j] + j_rem[j];
            if (v > lb) lb = v;
        }
        return (double)lb;
    }

    int m_minhead[HARD_MAX_MACHINES+2], m_mintail[HARD_MAX_MACHINES+2];
    for (int m = 0; m <= num_machines + 1; m++) {
        m_minhead[m] = INT_MAX; m_mintail[m] = INT_MAX;
    }
    int lb_cp = 0;
    
    for (int i = 0; i < nr; i++) {
        int op = pr[i];
        if (op < 1 || op >= num_ops_tot) continue;
        int j1 = all_ops[op].job_id;
        int last_in_job = j1 * num_machines;
        for (int op_id = op; op_id <= last_in_job; op_id++) {
            int m = all_ops[op_id].machine_id;
            int d = all_ops[op_id].duration;
            int tail = host_tail_static[op_id];
            int head = j_ready[j1-1];
            if (mrt[m] > head) head = mrt[m];
            if (head < m_minhead[m]) m_minhead[m] = head;
            if (tail < m_mintail[m]) m_mintail[m] = tail;
            if (lb_mode == LB_CP) {
                int v = head + d + tail;
                if (v > lb_cp) lb_cp = v;
            }
        }
    }
    if (lb_mode == LB_HT) {
        int lb_ht = 0;
        for (int m = 1; m <= num_machines; m++) {
            if (m_rem[m] > 0 && m_minhead[m] != INT_MAX) {
                int v = m_minhead[m] + m_rem[m] + m_mintail[m];
                if (v > lb_ht) lb_ht = v;
            }
        }
        return (double)max(mach_LB, lb_ht);
    }
    return (double)max(mach_LB, lb_cp);
}

__device__ static int gpu_compute_lb(
    int cmax, int num_scheduled,
    const int *mrt, const int *mach_rem,
    const int *job_ready, const int *job_rem,
    const int *pr, int num_ready,
    int num_machines_1idx, int num_jobs_actual, int num_ops_tot,
    int lb_mode)
{
    if (lb_mode == 0) return cmax; // LB_CUR logic handled by caller

    int lb = cmax;
    for (int m = 1; m <= num_machines_1idx; m++) {
        int v = mrt[m] + mach_rem[m];
        if (v > lb) lb = v;
    }
    if (lb_mode == 1) return lb;
    if (lb_mode == 2) {
        for (int j = 0; j < num_jobs_actual; j++) {
            int v = job_ready[j] + job_rem[j];
            if (v > lb) lb = v;
        }
        return lb;
    }

    int m_minhead[HARD_MAX_MACHINES+2];
    int m_mintail[HARD_MAX_MACHINES+2];
    for (int m = 0; m < HARD_MAX_MACHINES+2; m++) {
        m_minhead[m] = INT_MAX;
        m_mintail[m] = INT_MAX;
    }
    int lb_cp = 0;
    for (int i = 0; i < num_ready; i++) {
        int op = pr[i];
        if (op < 1 || op >= num_ops_tot) continue;
        int j1 = c_ops[op].job_id;
        int last_in_job = j1 * num_machines_1idx;
        for (int op_id = op; op_id <= last_in_job; op_id++) {
            int m = c_ops[op_id].machine_id;
            int d = c_ops[op_id].duration;
            int t = c_tail_static[op_id];
            int head = job_ready[j1-1];
            if (mrt[m] > head) head = mrt[m];
            if (head < m_minhead[m]) m_minhead[m] = head;
            if (t    < m_mintail[m]) m_mintail[m] = t;
            if (lb_mode == 4) {
                int v = head + d + t;
                if (v > lb_cp) lb_cp = v;
            }
        }
    }
    if (lb_mode == 3) { 
        for (int m = 1; m <= num_machines_1idx; m++) {
            if (mach_rem[m] > 0 && m_minhead[m] != INT_MAX) {
                int v = m_minhead[m] + mach_rem[m] + m_mintail[m];
                if (v > lb) lb = v;
            }
        }
        return lb;
    }
    if (lb_cp > lb) lb = lb_cp;
    return lb;
}

// -----------------------------------------------------------------------------
// Validation
// -----------------------------------------------------------------------------
static bool cpu_validate_schedule(const int* seq, int num_ops_scheduled,
                                   int num_machines_val, int num_ops_tot)
{
    if (num_ops_scheduled == 0) return false;
    bool valid = true;
    for (int i = 1; i < num_ops_tot - 1; i++) {
        int start_i = h_global_op_start[i];
        int duration_i = all_ops[i].duration;
        int machine_i = all_ops[i].machine_id;
        int finish_i;
        bool is_last_in_job = (i % num_machines_val == 0);
        if (!is_last_in_job) {
            finish_i = h_global_op_start[i + 1];
        } else {
            finish_i = start_i + duration_i;
        }
        if (finish_i < start_i + duration_i) {
            printf("VALIDATION FAILED: Op %d completes before its processing time\n", i);
            valid = false;
        }
        for (int j = i + 1; j < num_ops_tot - 1; j++) {
            if (all_ops[j].machine_id != machine_i) continue;
            int start_j = h_global_op_start[j];
            int finish_j = (j % num_machines_val != 0) ? h_global_op_start[j + 1] : start_j + all_ops[j].duration;
            if (!(finish_i <= start_j || finish_j <= start_i)) {
                printf("VALIDATION FAILED: Machine %d overlap\n", machine_i);
                valid = false;
            }
        }
    }
    return valid;
}

// Linear array search
__device__ static int dev_find_in_array(const int *arr, int len, int val) {
    for (int i = 0; i < len; i++) if (arr[i] == val) return i;
    return -1;
}

__device__ static int find_op_rank_on_machine(int op_id, int machine, int machine_order_stride, const int *d_op_order_ptr) {
    int num_ops_on_machine = c_op_count[machine];
    for (int rank = 0; rank < num_ops_on_machine; rank++)
        if (__ldg(&d_op_order_ptr[machine * machine_order_stride + rank]) == op_id)
            return rank;
    return -1;
}

// -----------------------------------------------------------------------------
// GPU Engine Core
// -----------------------------------------------------------------------------
__device__ static void traiter(
    int op_id,
    int *machine_occupied_op, int *machine_ready_time, int *critical_parent_op,
    int *op_start_time, int *op_finish_time,
    int *precedence_ready_ops, int &num_ready,
    int *seq, int &num_scheduled,
    int num_machines_1idx, int num_ops_tot, int machine_order_stride,
    const int *d_op_order_ptr,
    int *mach_rem, int *job_ready, int *job_rem
) {
    int cascade_stack[HARD_MAX_OPS];
    int cascade_top = 0;
    int current = op_id;

    do {
        int machine = c_ops[current].machine_id;
        seq[num_scheduled++] = current;

        int start;
        bool is_first_in_job = (current % num_machines_1idx == 1);

        if (!is_first_in_job) {
            int predecessor_op     = current - 1;
            int predecessor_machine = c_ops[predecessor_op].machine_id;

            if (machine_ready_time[machine] <= machine_ready_time[predecessor_machine]) {
                start = machine_ready_time[predecessor_machine];
                critical_parent_op[current] = predecessor_op;
            } else {
                start = machine_ready_time[machine];
                int rank_on_machine = find_op_rank_on_machine(current, machine, machine_order_stride, d_op_order_ptr);
                int prev_op_on_same_machine = (rank_on_machine > 0) ? __ldg(&d_op_order_ptr[machine * machine_order_stride + rank_on_machine - 1]) : -100;
                if (prev_op_on_same_machine != -100 && prev_op_on_same_machine % num_machines_1idx != 0) prev_op_on_same_machine++;
                critical_parent_op[current] = prev_op_on_same_machine;
            }
        } else {
            int rank_on_machine = find_op_rank_on_machine(current, machine, machine_order_stride, d_op_order_ptr);
            if (rank_on_machine > 0) {
                int prev_op_on_same_machine = __ldg(&d_op_order_ptr[machine * machine_order_stride + rank_on_machine - 1]);
                if (prev_op_on_same_machine % num_machines_1idx != 0) prev_op_on_same_machine++;
                critical_parent_op[current] = prev_op_on_same_machine;
            } else {
                critical_parent_op[current] = 0;
            }
            start = machine_ready_time[machine];
        }

        op_start_time[current]       = start;
        machine_ready_time[machine]  = start + c_ops[current].duration;
        machine_occupied_op[machine] = current;

        // -- LB Counters Update --
        int jid = c_ops[current].job_id;
        int d   = c_ops[current].duration;
        mach_rem[machine] -= d;
        job_rem[jid - 1]  -= d;
        job_ready[jid - 1] = machine_ready_time[machine];

        if (!is_first_in_job) {
            int predecessor_machine = c_ops[current - 1].machine_id;
            machine_occupied_op[predecessor_machine] = 0;
            machine_ready_time[predecessor_machine]  = start;
        }

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

__device__ static void traiter_swap(
    int op_id,
    int *machine_occupied_op, int *machine_ready_time,
    int *precedence_ready_ops, int &num_ready,
    int *seq, int &num_scheduled,
    int num_machines_1idx, int *op_start_time,
    int *mach_rem, int *job_ready, int *job_rem
) {
    int machine = c_ops[op_id].machine_id;
    seq[num_scheduled++] = op_id;

    int sync_start       = machine_ready_time[machine];
    op_start_time[op_id] = sync_start;
    machine_ready_time[machine]  = sync_start + c_ops[op_id].duration;
    machine_occupied_op[machine] = op_id;

    // -- LB Counters Update --
    int jid = c_ops[op_id].job_id;
    int d   = c_ops[op_id].duration;
    mach_rem[machine] -= d;
    job_rem[jid - 1]  -= d;
    job_ready[jid - 1] = machine_ready_time[machine];

    bool is_last_in_job = (op_id % num_machines_1idx == 0);
    if (!is_last_in_job)
        precedence_ready_ops[num_ready++] = op_id + 1;
    else
        machine_occupied_op[machine] = 0;
}

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
                (op_on_needed != 0) && (op_on_needed < num_ops_tot) &&
                (next_op_needed_2 > 0) && (next_op_needed_2 < num_ops_tot) &&
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

__device__ static void check_swap(
    int *machine_occupied_op, int *machine_ready_time, int *critical_parent_op,
    int *precedence_ready_ops, int &num_ready,
    int *seq, int &num_scheduled,
    int num_machines_1idx, int num_ops_tot, int num_machines_actual,
    int *op_start_time,
    int *mach_rem, int *job_ready, int *job_rem
) {
    int cycle_machines[HARD_MAX_MACHINES];
    int cycle_len = solinit_swap_new(machine_occupied_op, cycle_machines,
                                     num_machines_1idx, num_ops_tot, num_machines_actual);
    if (cycle_len <= 1) return;

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
        if (cycle_machines[i] != 0) machine_ready_time[cycle_machines[i]] = sync_time;

    int blocked_ops[HARD_MAX_MACHINES];
    for (int i = 0; i < cycle_len; i++) {
        int mc = cycle_machines[i];
        blocked_ops[i] = (mc != 0) ? machine_occupied_op[mc] + 1 : 0;
    }

    for (int i = 0; i < cycle_len; i++) {
        int op_id = blocked_ops[i];
        if (op_id == 0) continue;
        int mc = c_ops[op_id].machine_id;
        critical_parent_op[op_id] = (op_id == pivot_op_id) ? pivot_op_id - 1 : machine_occupied_op[mc] + 1;
    }

    for (int i = 0; i < cycle_len; i++) {
        int op_id = blocked_ops[i];
        if (op_id == 0) continue;
        int pos = dev_find_in_array(precedence_ready_ops, num_ready, op_id);
        if (pos >= 0) precedence_ready_ops[pos] = precedence_ready_ops[--num_ready];

        traiter_swap(op_id, machine_occupied_op, machine_ready_time, precedence_ready_ops, num_ready,
                     seq, num_scheduled, num_machines_1idx, op_start_time, mach_rem, job_ready, job_rem);
    }
}

// -----------------------------------------------------------------------------
// Phase 1 Kernel
// -----------------------------------------------------------------------------
__global__ void kernel_score_children(
    const int  *d_in_beam,
    float      *d_out_scores,
    int        *d_global_best,
    int        *d_best_sol,
    const int  *d_op_order_ptr,
    int         num_candidates, int num_branches_per_candidate, int SNAPSHOT_INTS_val,
    int off_machine_occupied, int off_machine_ready, int off_critical_parent,
    int off_op_start, int off_op_finish, int off_precedence_ready, int off_ready_count,
    int off_sequence, int off_sequence_count, int off_makespan,
    int off_mach_rem, int off_job_ready, int off_job_rem,
    int num_machines_1idx, int num_jobs_actual, int num_ops_tot,
    int total_real_ops, int machine_order_stride, int random_seed, int lb_mode
) {
    for (int candidate_id = blockIdx.x * blockDim.x + threadIdx.x; 
         candidate_id < num_candidates; 
         candidate_id += blockDim.x * gridDim.x) {

        int parent_base = candidate_id * SNAPSHOT_INTS_val;

        for (int b = 0; b < num_branches_per_candidate; b++)
            d_out_scores[candidate_id * num_branches_per_candidate + b] = 1e9f;

        int free_ops_buf[HARD_MAX_JOBS], num_free = 0;
        int num_ready = d_in_beam[parent_base + off_ready_count];

        for (int i = 0; i < num_ready; i++) {
            int op = d_in_beam[parent_base + off_precedence_ready + i];
            if (op >= 1 && op < num_ops_tot && d_in_beam[parent_base + off_machine_occupied + c_ops[op].machine_id] == 0)
                free_ops_buf[num_free++] = op;
        }

        int candidate_ops[HARD_MAX_JOBS], num_valid_ops = 0;

        if (num_free > 0) {
            unsigned int hash = random_seed + candidate_id;
            hash ^= hash >> 16; hash *= 0x7feb352d; hash ^= hash >> 15; hash *= 0x846ca68b; hash ^= hash >> 16;
            int chosen_machine = c_ops[free_ops_buf[hash % num_free]].machine_id;

            for (int i = 0; i < num_ready; i++) {
                int op = d_in_beam[parent_base + off_precedence_ready + i];
                if (op >= 1 && op < num_ops_tot && c_ops[op].machine_id == chosen_machine &&
                    d_in_beam[parent_base + off_machine_occupied + chosen_machine] == 0)
                    candidate_ops[num_valid_ops++] = op;
            }
        }

        int machine_occupied_op[HARD_MAX_MACHINES], machine_ready_time [HARD_MAX_MACHINES];
        int critical_parent_op [HARD_MAX_OPS], op_start_time [HARD_MAX_OPS], op_finish_time [HARD_MAX_OPS];
        int precedence_ready_ops[HARD_MAX_OPS], seq [HARD_MAX_OPS];
        int mach_rem[HARD_MAX_MACHINES], job_ready[HARD_MAX_JOBS], job_rem[HARD_MAX_JOBS];

        if (num_valid_ops == 0) { // Deadlock fallback
            for (int m = 0; m <= num_machines_1idx; m++) {
                machine_occupied_op[m] = d_in_beam[parent_base + off_machine_occupied + m];
                machine_ready_time[m]  = d_in_beam[parent_base + off_machine_ready    + m];
                mach_rem[m]            = d_in_beam[parent_base + off_mach_rem         + m];
            }
            for (int j = 0; j < num_jobs_actual; j++) {
                job_ready[j] = d_in_beam[parent_base + off_job_ready + j];
                job_rem[j]   = d_in_beam[parent_base + off_job_rem   + j];
            }
            for (int op = 0; op < num_ops_tot; op++) {
                critical_parent_op[op] = d_in_beam[parent_base + off_critical_parent + op];
                op_start_time[op]      = d_in_beam[parent_base + off_op_start        + op];
                op_finish_time[op]     = d_in_beam[parent_base + off_op_finish       + op];
            }
            int cur_num_ready = num_ready;
            for (int i = 0; i < cur_num_ready; i++) precedence_ready_ops[i] = d_in_beam[parent_base + off_precedence_ready + i];
            int num_scheduled = d_in_beam[parent_base + off_sequence_count];
            for (int i = 0; i < num_scheduled; i++) seq[i] = d_in_beam[parent_base + off_sequence + i];

            check_swap(machine_occupied_op, machine_ready_time, critical_parent_op,
                       precedence_ready_ops, cur_num_ready, seq, num_scheduled,
                       num_machines_1idx, num_ops_tot, num_machines_1idx, op_start_time, mach_rem, job_ready, job_rem);

            int cmax = 0;
            for (int m = 1; m <= num_machines_1idx; m++) if (machine_ready_time[m] > cmax) cmax = machine_ready_time[m];

            if (cmax >= *d_global_best) continue;

            float base_score;
            if (lb_mode == 0) {
                base_score = (num_scheduled == 0) ? (float)cmax : ((float)cmax / num_scheduled);
            } else {
                int lb = gpu_compute_lb(cmax, num_scheduled, machine_ready_time, mach_rem, job_ready, job_rem,
                                        precedence_ready_ops, cur_num_ready, num_machines_1idx, num_jobs_actual, num_ops_tot, lb_mode);
                base_score = (float)lb;
            }

            unsigned int nh = random_seed ^ (candidate_id * 0x85ebca6b);
            nh ^= nh >> 16; nh *= 0x85ebca6b; nh ^= nh >> 16; nh *= 0xc2b2ae35; nh ^= nh >> 16;
            float sibling_noise = (float)(nh % 1000) / 1000000000.0f;
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
                    for (int i = 0; i < cur_num_ready; i++) d_best_sol[off_precedence_ready + i] = precedence_ready_ops[i];
                    for (int i = 0; i < num_scheduled; i++) d_best_sol[off_sequence + i] = seq[i];
                }
            }
            continue;
        }

        // Process valid branches sequentially
        for (int branch_id = 0; branch_id < num_valid_ops; branch_id++) {
            for (int m = 0; m <= num_machines_1idx; m++) {
                machine_occupied_op[m] = d_in_beam[parent_base + off_machine_occupied + m];
                machine_ready_time[m]  = d_in_beam[parent_base + off_machine_ready    + m];
                mach_rem[m]            = d_in_beam[parent_base + off_mach_rem         + m];
            }
            for (int j = 0; j < num_jobs_actual; j++) {
                job_ready[j] = d_in_beam[parent_base + off_job_ready + j];
                job_rem[j]   = d_in_beam[parent_base + off_job_rem   + j];
            }
            for (int op = 0; op < num_ops_tot; op++) {
                critical_parent_op[op] = d_in_beam[parent_base + off_critical_parent + op];
                op_start_time[op]      = d_in_beam[parent_base + off_op_start        + op];
                op_finish_time[op]     = d_in_beam[parent_base + off_op_finish       + op];
            }
            int cur_num_ready = num_ready;
            for (int i = 0; i < cur_num_ready; i++) precedence_ready_ops[i] = d_in_beam[parent_base + off_precedence_ready + i];
            int num_scheduled = d_in_beam[parent_base + off_sequence_count];
            for (int i = 0; i < num_scheduled; i++) seq[i] = d_in_beam[parent_base + off_sequence + i];

            int chosen_op = candidate_ops[branch_id];
            int ready_pos = dev_find_in_array(precedence_ready_ops, cur_num_ready, chosen_op);
            if (ready_pos >= 0) precedence_ready_ops[ready_pos] = precedence_ready_ops[--cur_num_ready];

            traiter(chosen_op, machine_occupied_op, machine_ready_time, critical_parent_op,
                    op_start_time, op_finish_time, precedence_ready_ops, cur_num_ready,
                    seq, num_scheduled, num_machines_1idx, num_ops_tot, machine_order_stride, d_op_order_ptr,
                    mach_rem, job_ready, job_rem);

            check_swap(machine_occupied_op, machine_ready_time, critical_parent_op, precedence_ready_ops, cur_num_ready,
                       seq, num_scheduled, num_machines_1idx, num_ops_tot, num_machines_1idx, op_start_time,
                       mach_rem, job_ready, job_rem);

            int cmax = 0;
            for (int m = 1; m <= num_machines_1idx; m++) if (machine_ready_time[m] > cmax) cmax = machine_ready_time[m];

            if (cmax >= *d_global_best) continue;

            float base_score;
            if (lb_mode == 0) {
                base_score = (num_scheduled == 0) ? (float)cmax : ((float)cmax / num_scheduled);
            } else {
                int lb = gpu_compute_lb(cmax, num_scheduled, machine_ready_time, mach_rem, job_ready, job_rem,
                                        precedence_ready_ops, cur_num_ready, num_machines_1idx, num_jobs_actual, num_ops_tot, lb_mode);
                base_score = (float)lb;
            }

            unsigned int nh = random_seed ^ (candidate_id * 0x85ebca6b) ^ (branch_id * 0xc2b2ae35);
            nh ^= nh >> 16; nh *= 0x85ebca6b; nh ^= nh >> 16; nh *= 0xc2b2ae35; nh ^= nh >> 16;
            float sibling_noise = (float)(nh % 1000) / 1000000000.0f;
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
                    for (int i = 0; i < cur_num_ready; i++) d_best_sol[off_precedence_ready + i] = precedence_ready_ops[i];
                    for (int i = 0; i < num_scheduled; i++) d_best_sol[off_sequence + i] = seq[i];
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Phase 2 Kernel
// -----------------------------------------------------------------------------
__global__ void kernel_materialize_topk(
    const int *d_in_beam, const int *d_sorted_indices, int *d_out_beam,
    int num_to_gather, int num_branches_per_candidate, int SNAPSHOT_INTS_val,
    int off_machine_occupied, int off_machine_ready, int off_critical_parent,
    int off_op_start, int off_op_finish, int off_precedence_ready, int off_ready_count,
    int off_sequence, int off_sequence_count, int off_makespan,
    int off_mach_rem, int off_job_ready, int off_job_rem,
    int num_machines_1idx, int num_jobs_actual, int num_ops_tot, int machine_order_stride, int random_seed, const int *d_op_order_ptr
) {
    for (int rank = blockIdx.x * blockDim.x + threadIdx.x; rank < num_to_gather; rank += blockDim.x * gridDim.x) {
        int src_child_idx = d_sorted_indices[rank];
        int candidate_id  = src_child_idx / num_branches_per_candidate;
        int target_branch = src_child_idx % num_branches_per_candidate;
        int parent_base = candidate_id * SNAPSHOT_INTS_val;

        int free_ops_buf[HARD_MAX_JOBS], num_free = 0;
        int num_ready = d_in_beam[parent_base + off_ready_count];

        for (int i = 0; i < num_ready; i++) {
            int op = d_in_beam[parent_base + off_precedence_ready + i];
            if (op >= 1 && op < num_ops_tot && d_in_beam[parent_base + off_machine_occupied + c_ops[op].machine_id] == 0)
                free_ops_buf[num_free++] = op;
        }

        int candidate_ops[HARD_MAX_JOBS], num_valid_ops = 0;
        if (num_free > 0) {
            unsigned int hash = random_seed + candidate_id;
            hash ^= hash >> 16; hash *= 0x7feb352d; hash ^= hash >> 15; hash *= 0x846ca68b; hash ^= hash >> 16;
            int chosen_machine = c_ops[free_ops_buf[hash % num_free]].machine_id;

            for (int i = 0; i < num_ready; i++) {
                int op = d_in_beam[parent_base + off_precedence_ready + i];
                if (op >= 1 && op < num_ops_tot && c_ops[op].machine_id == chosen_machine &&
                    d_in_beam[parent_base + off_machine_occupied + chosen_machine] == 0)
                    candidate_ops[num_valid_ops++] = op;
            }
        }

        int out_base = rank * SNAPSHOT_INTS_val;
        int machine_occupied_op[HARD_MAX_MACHINES], machine_ready_time [HARD_MAX_MACHINES];
        int critical_parent_op [HARD_MAX_OPS], op_start_time [HARD_MAX_OPS], op_finish_time [HARD_MAX_OPS];
        int precedence_ready_ops[HARD_MAX_OPS], seq [HARD_MAX_OPS];
        int mach_rem[HARD_MAX_MACHINES], job_ready[HARD_MAX_JOBS], job_rem[HARD_MAX_JOBS];

        for (int m = 0; m <= num_machines_1idx; m++) {
            machine_occupied_op[m] = d_in_beam[parent_base + off_machine_occupied + m];
            machine_ready_time[m]  = d_in_beam[parent_base + off_machine_ready    + m];
            mach_rem[m]            = d_in_beam[parent_base + off_mach_rem         + m];
        }
        for (int j = 0; j < num_jobs_actual; j++) {
            job_ready[j] = d_in_beam[parent_base + off_job_ready + j];
            job_rem[j]   = d_in_beam[parent_base + off_job_rem   + j];
        }
        for (int op = 0; op < num_ops_tot; op++) {
            critical_parent_op[op] = d_in_beam[parent_base + off_critical_parent + op];
            op_start_time[op]      = d_in_beam[parent_base + off_op_start        + op];
            op_finish_time[op]     = d_in_beam[parent_base + off_op_finish       + op];
        }
        int cur_num_ready = num_ready;
        for (int i = 0; i < cur_num_ready; i++) precedence_ready_ops[i] = d_in_beam[parent_base + off_precedence_ready + i];
        int num_scheduled = d_in_beam[parent_base + off_sequence_count];
        for (int i = 0; i < num_scheduled; i++) seq[i] = d_in_beam[parent_base + off_sequence + i];

        if (num_valid_ops == 0) {
            check_swap(machine_occupied_op, machine_ready_time, critical_parent_op,
                       precedence_ready_ops, cur_num_ready, seq, num_scheduled,
                       num_machines_1idx, num_ops_tot, num_machines_1idx, op_start_time, mach_rem, job_ready, job_rem);
        } else {
            int chosen_op = candidate_ops[target_branch];
            int ready_pos = dev_find_in_array(precedence_ready_ops, cur_num_ready, chosen_op);
            if (ready_pos >= 0) precedence_ready_ops[ready_pos] = precedence_ready_ops[--cur_num_ready];

            traiter(chosen_op, machine_occupied_op, machine_ready_time, critical_parent_op,
                    op_start_time, op_finish_time, precedence_ready_ops, cur_num_ready,
                    seq, num_scheduled, num_machines_1idx, num_ops_tot, machine_order_stride, d_op_order_ptr,
                    mach_rem, job_ready, job_rem);

            check_swap(machine_occupied_op, machine_ready_time, critical_parent_op, precedence_ready_ops, cur_num_ready,
                       seq, num_scheduled, num_machines_1idx, num_ops_tot, num_machines_1idx, op_start_time,
                       mach_rem, job_ready, job_rem);
        }

        int cmax = 0;
        for (int m = 1; m <= num_machines_1idx; m++) if (machine_ready_time[m] > cmax) cmax = machine_ready_time[m];

        d_out_beam[out_base + off_makespan]       = cmax;
        d_out_beam[out_base + off_sequence_count] = num_scheduled;
        d_out_beam[out_base + off_ready_count]    = cur_num_ready;
        for (int m = 0; m <= num_machines_1idx; m++) {
            d_out_beam[out_base + off_machine_occupied + m] = machine_occupied_op[m];
            d_out_beam[out_base + off_machine_ready    + m] = machine_ready_time[m];
            d_out_beam[out_base + off_mach_rem         + m] = mach_rem[m];
        }
        for (int j = 0; j < num_jobs_actual; j++) {
            d_out_beam[out_base + off_job_ready + j] = job_ready[j];
            d_out_beam[out_base + off_job_rem   + j] = job_rem[j];
        }
        for (int op = 0; op < num_ops_tot; op++) {
            d_out_beam[out_base + off_critical_parent + op] = critical_parent_op[op];
            d_out_beam[out_base + off_op_start        + op] = op_start_time[op];
            d_out_beam[out_base + off_op_finish       + op] = op_finish_time[op];
        }
        for (int i = 0; i < cur_num_ready; i++) d_out_beam[out_base + off_precedence_ready + i] = precedence_ready_ops[i];
        for (int i = 0; i < num_scheduled; i++) d_out_beam[out_base + off_sequence + i] = seq[i];
    }
}

__global__ void init_indices(int *indices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) indices[i] = i;
}

// -----------------------------------------------------------------------------
// Buffer & Init
// -----------------------------------------------------------------------------
static void allocate_gpu_buffers() {
    CUDA_CHECK(cudaMemcpyToSymbol(c_ops, all_ops, HARD_MAX_OPS * sizeof(OperationInfo)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_op_count, machine_op_count, HARD_MAX_MACHINES * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_mach_total, host_mach_total, HARD_MAX_MACHINES * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_job_total, host_job_total, HARD_MAX_JOBS * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tail_static, host_tail_static, HARD_MAX_OPS * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_op_order, HARD_MAX_MACHINES * HARD_MAX_JOBS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_op_order, machine_op_order, HARD_MAX_MACHINES * HARD_MAX_JOBS * sizeof(int), cudaMemcpyHostToDevice));

    size_t beam_bytes  = (size_t)BUF_BEAM * SNAPSHOT_INTS * sizeof(int);
    size_t score_bytes = (size_t)BUF_BEAM * BUF_JOBS * sizeof(float);
    size_t index_bytes = (size_t)BUF_BEAM * BUF_JOBS * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_beam_snapshots,    beam_bytes));
    CUDA_CHECK(cudaMalloc(&d_beam_snapshots_2,  beam_bytes));
    CUDA_CHECK(cudaMalloc(&d_child_scores,      score_bytes));
    CUDA_CHECK(cudaMalloc(&d_child_scores_out,  score_bytes));
    CUDA_CHECK(cudaMalloc(&d_child_indices,     index_bytes));
    CUDA_CHECK(cudaMalloc(&d_child_indices_out, index_bytes));
    CUDA_CHECK(cudaMalloc(&d_global_best_makespan, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_solution, SNAPSHOT_INTS * sizeof(int)));

    cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes, d_child_scores, d_child_scores_out, d_child_indices, d_child_indices_out, BUF_BEAM * BUF_JOBS);
    CUDA_CHECK(cudaMalloc(&d_cub_temp_storage, cub_temp_bytes));
    CUDA_CHECK(cudaMallocHost(&h_input_flat, SNAPSHOT_INTS * sizeof(int)));
}

static void free_gpu_buffers() {
    cudaFree(d_op_order); cudaFree(d_beam_snapshots); cudaFree(d_beam_snapshots_2);
    cudaFree(d_child_scores); cudaFree(d_child_scores_out); cudaFree(d_child_indices);
    cudaFree(d_child_indices_out); cudaFree(d_cub_temp_storage); cudaFree(d_global_best_makespan);
    cudaFree(d_best_solution); cudaFreeHost(h_input_flat);
}

static void load_instance() {
    FILE *f = fopen(instance_path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", instance_path); exit(1); }

    fscanf(f, "%d %d\n", &num_jobs, &num_machines);
    num_ops_total = num_jobs * num_machines + 2;
    BUF_OPS = num_ops_total; BUF_MACHINES = num_machines + 1; BUF_JOBS = num_jobs;

    memset(all_ops, 0, sizeof(all_ops));
    all_ops[0] = {0, 0, 0};
    memset(host_mach_total, 0, sizeof(host_mach_total));
    memset(host_job_total, 0, sizeof(host_job_total));
    memset(host_tail_static, 0, sizeof(host_tail_static));

    int op_id = 1;
    for (int job = 1; job <= num_jobs; job++) {
        for (int step = 1; step <= num_machines; step++) {
            int raw_machine_id, duration;
            fscanf(f, "%d %d", &raw_machine_id, &duration);
            int m_1idx = raw_machine_id + 1;
            all_ops[op_id] = {m_1idx, duration, job};
            ops_by_machine[m_1idx].push_back(op_id);
            host_mach_total[m_1idx] += duration;
            host_job_total[job - 1] += duration;
            op_id++;
        }
    }
    all_ops[num_jobs * num_machines + 1] = {0, 0, 0};
    fclose(f);

    for (int m = 1; m <= num_machines; m++) {
        int n = (int)ops_by_machine[m].size();
        machine_op_count[m] = n;
        for (int rank = 0; rank < n; rank++) machine_op_order[m * HARD_MAX_JOBS + rank] = ops_by_machine[m][rank];
    }
    for (int j = 1; j <= num_jobs; j++) {
        int last = j * num_machines;
        host_tail_static[last] = 0;
        for (int k = num_machines - 1; k >= 1; k--) {
            int op_cur = (j - 1) * num_machines + k;
            host_tail_static[op_cur] = host_tail_static[op_cur + 1] + all_ops[op_cur + 1].duration;
        }
    }
}

static ScheduleSnapshot make_root_snapshot() {
    ScheduleSnapshot s{};
    s.num_precedence_ready = num_jobs;
    for (int j = 0; j < num_jobs; j++) s.precedence_ready_ops[j] = j * num_machines + 1;
    s.makespan = 0;
    s.num_scheduled = 0;
    for (int m = 1; m <= num_machines; m++) s.mach_rem[m] = host_mach_total[m];
    for (int j = 0; j < num_jobs; j++)      s.job_rem[j] = host_job_total[j];
    return s;
}

// -----------------------------------------------------------------------------
// Pass Execution (GPU & CPU)
// -----------------------------------------------------------------------------
static int run_beam_construction(int lb_mode) {
    int pass_best_makespan = global_best_makespan;
    CUDA_CHECK(cudaMemcpy(d_global_best_makespan, &pass_best_makespan, sizeof(int), cudaMemcpyHostToDevice));

    ScheduleSnapshot root = make_root_snapshot();
    pack_snapshot(root, h_input_flat);
    CUDA_CHECK(cudaMemcpy(d_beam_snapshots, h_input_flat, SNAPSHOT_INTS * sizeof(int), cudaMemcpyHostToDevice));

    int current_beam_size = 1;
    int max_depth = num_jobs * num_machines + 10;
    int total_real_ops = num_jobs * num_machines;
    long long local_partial_states = 0;

    for (int depth = 1; depth < max_depth && current_beam_size > 0; depth++) {
        int random_seed = rand();
        
        int gpu_cur; CUDA_CHECK(cudaMemcpy(&gpu_cur, d_global_best_makespan, sizeof(int), cudaMemcpyDeviceToHost));
        int prev = g_shared_best.load(std::memory_order_relaxed);
        while (gpu_cur < prev && !g_shared_best.compare_exchange_weak(prev, gpu_cur, std::memory_order_release, std::memory_order_relaxed)) {}

        int total_children = current_beam_size * BUF_JOBS;
        local_partial_states += total_children;

        int score_blocks = std::min(80, (current_beam_size + 127) / 128);
        if(score_blocks == 0) score_blocks = 1;
        kernel_score_children<<<score_blocks, 128>>>(
            d_beam_snapshots, d_child_scores, d_global_best_makespan, d_best_solution, d_op_order,
            current_beam_size, BUF_JOBS, SNAPSHOT_INTS,
            SNAP_OFF_MACHINE_OCCUPIED_OP, SNAP_OFF_MACHINE_READY_TIME, SNAP_OFF_CRITICAL_PARENT_OP,
            SNAP_OFF_OP_START_TIME, SNAP_OFF_OP_FINISH_TIME, SNAP_OFF_PRECEDENCE_READY, SNAP_OFF_READY_COUNT,
            SNAP_OFF_SEQUENCE, SNAP_OFF_SEQUENCE_COUNT, SNAP_OFF_MAKESPAN,
            SNAP_OFF_MACH_REM, SNAP_OFF_JOB_READY, SNAP_OFF_JOB_REM,
            num_machines, num_jobs, num_ops_total, total_real_ops, HARD_MAX_JOBS, random_seed, lb_mode
        );
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

        int threads_per_block = 256;
        int index_init_blocks = (total_children + threads_per_block - 1) / threads_per_block;
        init_indices<<<index_init_blocks, threads_per_block>>>(d_child_indices, total_children);

        cub::DeviceRadixSort::SortPairs(d_cub_temp_storage, cub_temp_bytes, d_child_scores, d_child_scores_out, d_child_indices, d_child_indices_out, total_children);

        int candidates_to_check = min(total_children, beam_width);
        std::vector<float> h_top_scores(candidates_to_check);
        CUDA_CHECK(cudaMemcpy(h_top_scores.data(), d_child_scores_out, candidates_to_check * sizeof(float), cudaMemcpyDeviceToHost));

        int valid_count = 0;
        while (valid_count < candidates_to_check && h_top_scores[valid_count] < 1e8f) valid_count++;
        if (valid_count == 0) break;

        int materialize_blocks = std::min(80, (valid_count + 127) / 128);
        if(materialize_blocks == 0) materialize_blocks = 1;
        kernel_materialize_topk<<<materialize_blocks, 128>>>(
            d_beam_snapshots, d_child_indices_out, d_beam_snapshots_2, valid_count, BUF_JOBS, SNAPSHOT_INTS,
            SNAP_OFF_MACHINE_OCCUPIED_OP, SNAP_OFF_MACHINE_READY_TIME, SNAP_OFF_CRITICAL_PARENT_OP,
            SNAP_OFF_OP_START_TIME, SNAP_OFF_OP_FINISH_TIME, SNAP_OFF_PRECEDENCE_READY, SNAP_OFF_READY_COUNT,
            SNAP_OFF_SEQUENCE, SNAP_OFF_SEQUENCE_COUNT, SNAP_OFF_MAKESPAN,
            SNAP_OFF_MACH_REM, SNAP_OFF_JOB_READY, SNAP_OFF_JOB_REM,
            num_machines, num_jobs, num_ops_total, HARD_MAX_JOBS, random_seed, d_op_order
        );
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

        current_beam_size = valid_count;
        std::swap(d_beam_snapshots, d_beam_snapshots_2);
    }

    CUDA_CHECK(cudaMemcpy(&pass_best_makespan, d_global_best_makespan, sizeof(int), cudaMemcpyDeviceToHost));
    if (pass_best_makespan < global_best_makespan) {
        CUDA_CHECK(cudaMemcpy(h_input_flat, d_best_solution, SNAPSHOT_INTS * sizeof(int), cudaMemcpyDeviceToHost));
        if (h_input_flat[SNAP_OFF_SEQUENCE_COUNT] == total_real_ops) {
            for (int i = 0; i < total_real_ops; i++) h_global_best_seq[i] = h_input_flat[SNAP_OFF_SEQUENCE + i];
            for (int i = 0; i < num_ops_total; i++) {
                h_global_op_start[i]  = h_input_flat[SNAP_OFF_OP_START_TIME + i];
                h_global_op_finish[i] = h_input_flat[SNAP_OFF_OP_FINISH_TIME + i];
            }
        }
    }
    
    std::lock_guard<std::mutex> lock(stats_mutex);
    rank_stats.partial_states += local_partial_states;
    rank_stats.complete_solutions++;
    rank_stats.beam_iterations++;

    return pass_best_makespan;
}

static void cpu_traiter_wrapper(int op_id,
    int* moc, int* mrt, int* cpo, int* ost, int* oft, int* pr, int& nr, int* seq, int& ns, int nm, int nops,
    int* mach_rem, int* job_ready, int* job_rem)
{
    int current = op_id;
    do {
        int machine = all_ops[current].machine_id;
        seq[ns++] = current;
        bool is_first = (current % nm == 1);
        int start;
        if (!is_first) {
            int pred = current - 1, pred_mc = all_ops[pred].machine_id;
            if (mrt[machine] <= mrt[pred_mc]) { start = mrt[pred_mc]; cpo[current] = pred; }
            else {
                start = mrt[machine]; int rank = cpu_find_op_rank(current, machine);
                int prev = -100; if (rank > 0) { prev = machine_op_order[machine*HARD_MAX_JOBS+rank-1]; if (prev%nm!=0) prev++; }
                cpo[current] = prev;
            }
        } else {
            int rank = cpu_find_op_rank(current, machine);
            if (rank > 0) { int prev = machine_op_order[machine*HARD_MAX_JOBS+rank-1]; if (prev%nm!=0) prev++; cpo[current] = prev; }
            else cpo[current] = 0;
            start = mrt[machine];
        }
        ost[current] = start;
        mrt[machine] = start + all_ops[current].duration;
        moc[machine] = current;

        // -- LB Counters Update --
        int jid = all_ops[current].job_id;
        int d = all_ops[current].duration;
        mach_rem[machine] -= d;
        job_rem[jid - 1] -= d;
        job_ready[jid - 1] = mrt[machine];

        if (!is_first) { int pmc = all_ops[current-1].machine_id; moc[pmc] = 0; mrt[pmc] = start; }
        if (current % nm != 0) { pr[nr++] = current + 1; }
        else { oft[current] = mrt[machine]; if (oft[current] > ost[nops-1]) ost[nops-1] = oft[current]; moc[machine] = 0; }
        current = -1;
    } while (current != -1);
}

static void cpu_traiter_swap_wrapper(int op_id, int* moc, int* mrt, int* pr, int& nr, int* seq, int& ns, int nm, int* ost, int* mach_rem, int* job_ready, int* job_rem) {
    int machine = all_ops[op_id].machine_id;
    seq[ns++] = op_id;
    ost[op_id] = mrt[machine];
    mrt[machine] += all_ops[op_id].duration;
    moc[machine] = op_id;

    // -- LB Counters Update --
    int jid = all_ops[op_id].job_id;
    int d = all_ops[op_id].duration;
    mach_rem[machine] -= d;
    job_rem[jid - 1] -= d;
    job_ready[jid - 1] = mrt[machine];

    if (op_id % nm != 0) pr[nr++] = op_id + 1;
    else moc[machine] = 0;
}

static void cpu_check_swap_wrapper(int* moc, int* mrt, int* cpo, int* pr, int& nr, int* seq, int& ns, int nm, int nops, int num_mach, int* ost, int* mach_rem, int* job_ready, int* job_rem) {
    int cyc[HARD_MAX_MACHINES];
    int cl = cpu_solinit_swap_new(moc, cyc, nm, nops, num_mach);
    if (cl <= 1) return;
    int st = 0, piv = 0;
    for (int i=0;i<cl;i++) { int mc=cyc[i]; if (mc!=0&&mrt[mc]>st) { st=mrt[mc]; piv=moc[mc]; } }
    piv++;
    for (int i=0;i<cl;i++) if (cyc[i]!=0) mrt[cyc[i]]=st;
    int bops[HARD_MAX_MACHINES];
    for (int i=0;i<cl;i++) { int mc=cyc[i]; bops[i]=(mc!=0)?moc[mc]+1:0; }
    for (int i=0;i<cl;i++) { int op=bops[i]; if (!op) continue; int mc=all_ops[op].machine_id; cpo[op]=(op==piv)?piv-1:moc[mc]+1; }
    for (int i=0;i<cl;i++) { int op=bops[i]; if (!op) continue; int p=cpu_find_in_array(pr,nr,op); if (p>=0) pr[p]=pr[--nr]; cpu_traiter_swap_wrapper(op,moc,mrt,pr,nr,seq,ns,nm,ost,mach_rem,job_ready,job_rem); }
}

static int run_beam_construction_cpu(int cpu_bw, unsigned int* seed, int lb_mode)
{
    const int nm = num_machines, nops = num_ops_total, nj = num_jobs;
    const int total_real = nj * nm, max_depth = total_real + 10;
    long long local_partial_states = 0;

    vector<int> beam((size_t)cpu_bw * SNAPSHOT_INTS, 0);
    int max_ch = cpu_bw * nj;
    vector<int> children((size_t)max_ch * SNAPSHOT_INTS, 0);
    vector<float> scores(max_ch, 1e9f);
    vector<int> indices(max_ch);

    ScheduleSnapshot root = make_root_snapshot();
    pack_snapshot(root, beam.data());

    int bsz = 1, cpu_best = 500000;

    for (int depth = 1; depth < max_depth && bsz > 0; depth++) {
        fill(scores.begin(), scores.end(), 1e9f);
        for (int cand = 0; cand < bsz; cand++) {
            const int* par = beam.data() + cand * SNAPSHOT_INTS;
            int free_ops[HARD_MAX_JOBS], nfree = 0;
            int nrp = par[SNAP_OFF_READY_COUNT];
            for (int i = 0; i < nrp; i++) {
                int op = par[SNAP_OFF_PRECEDENCE_READY+i];
                if (op>=1 && op<nops && par[SNAP_OFF_MACHINE_OCCUPIED_OP+all_ops[op].machine_id]==0)
                    free_ops[nfree++] = op;
            }
            if (nfree > 0) {
                int chosen_mc = all_ops[free_ops[rand_r(seed)%nfree]].machine_id;
                int cops[HARD_MAX_JOBS], ncops = 0;
                for (int i = 0; i < nrp; i++) {
                    int op = par[SNAP_OFF_PRECEDENCE_READY+i];
                    if (op>=1 && op<nops && all_ops[op].machine_id==chosen_mc && par[SNAP_OFF_MACHINE_OCCUPIED_OP+chosen_mc]==0)
                        cops[ncops++] = op;
                }
                for (int b = 0; b < ncops; b++) {
                    local_partial_states++;
                    int moc[HARD_MAX_MACHINES]={},mrt[HARD_MAX_MACHINES]={};
                    int cpo[HARD_MAX_OPS]={},ost[HARD_MAX_OPS]={},oft[HARD_MAX_OPS]={};
                    int pr[HARD_MAX_OPS]={},sq[HARD_MAX_OPS]={};
                    int m_rem[HARD_MAX_MACHINES]={}, j_ready[HARD_MAX_JOBS]={}, j_rem[HARD_MAX_JOBS]={};

                    for (int m=0;m<=nm;m++) { moc[m]=par[SNAP_OFF_MACHINE_OCCUPIED_OP+m]; mrt[m]=par[SNAP_OFF_MACHINE_READY_TIME+m]; m_rem[m]=par[SNAP_OFF_MACH_REM+m]; }
                    for (int j=0;j<nj;j++) { j_ready[j]=par[SNAP_OFF_JOB_READY+j]; j_rem[j]=par[SNAP_OFF_JOB_REM+j]; }
                    for (int op=0;op<nops;op++) { cpo[op]=par[SNAP_OFF_CRITICAL_PARENT_OP+op]; ost[op]=par[SNAP_OFF_OP_START_TIME+op]; oft[op]=par[SNAP_OFF_OP_FINISH_TIME+op]; }
                    int nr2=par[SNAP_OFF_READY_COUNT]; for (int i=0;i<nr2;i++) pr[i]=par[SNAP_OFF_PRECEDENCE_READY+i];
                    int ns2=par[SNAP_OFF_SEQUENCE_COUNT]; for (int i=0;i<ns2;i++) sq[i]=par[SNAP_OFF_SEQUENCE+i];
                    
                    int cop=cops[b]; int pos=cpu_find_in_array(pr,nr2,cop); if (pos>=0) pr[pos]=pr[--nr2];
                    cpu_traiter_wrapper(cop,moc,mrt,cpo,ost,oft,pr,nr2,sq,ns2,nm,nops, m_rem, j_ready, j_rem);
                    cpu_check_swap_wrapper(moc,mrt,cpo,pr,nr2,sq,ns2,nm,nops,nm,ost, m_rem, j_ready, j_rem);
                    
                    int cmax=0; for (int m=1;m<=nm;m++) if (mrt[m]>cmax) cmax=mrt[m];
                    if (cmax>=cpu_best) continue;
                    
                    double sc = cpu_compute_score(cmax, ns2, lb_mode, mrt, m_rem, j_ready, j_rem, pr, nr2, nops);
                    unsigned nh=*seed^(cand*0x85ebca6b)^(b*0xc2b2ae35); nh^=nh>>16; nh*=0x85ebca6b; nh^=nh>>16;
                    float base_score = sc + (float)(nh%1000)/1e9f;

                    int ci=cand*nj+b; int* out=children.data()+ci*SNAPSHOT_INTS;
                    for (int m=0;m<=nm;m++) { out[SNAP_OFF_MACHINE_OCCUPIED_OP+m]=moc[m]; out[SNAP_OFF_MACHINE_READY_TIME+m]=mrt[m]; out[SNAP_OFF_MACH_REM+m]=m_rem[m]; }
                    for (int j=0;j<nj;j++) { out[SNAP_OFF_JOB_READY+j]=j_ready[j]; out[SNAP_OFF_JOB_REM+j]=j_rem[j]; }
                    for (int op=0;op<nops;op++) { out[SNAP_OFF_CRITICAL_PARENT_OP+op]=cpo[op]; out[SNAP_OFF_OP_START_TIME+op]=ost[op]; out[SNAP_OFF_OP_FINISH_TIME+op]=oft[op]; }
                    for (int i=0;i<nr2;i++) out[SNAP_OFF_PRECEDENCE_READY+i]=pr[i];
                    out[SNAP_OFF_READY_COUNT]=nr2; for (int i=0;i<ns2;i++) out[SNAP_OFF_SEQUENCE+i]=sq[i];
                    out[SNAP_OFF_SEQUENCE_COUNT]=ns2; out[SNAP_OFF_MAKESPAN]=cmax;
                    scores[ci]=base_score;

                    if (ns2==total_real) { 
                        int pv=g_shared_best.load(memory_order_relaxed); 
                        while (cmax<pv&&!g_shared_best.compare_exchange_weak(pv,cmax,memory_order_release,memory_order_relaxed)){} 
                        if (cmax<cpu_best) { 
                            cpu_best=cmax; 
                            std::lock_guard<std::mutex> lock(cpu_best_mutex);
                            if (cmax < h_cpu_best_makespan) {
                                h_cpu_best_makespan = cmax;
                                for (int i=0;i<total_real;i++) h_cpu_best_seq[i]=sq[i]; 
                                for (int i=0;i<nops;i++) { h_cpu_op_start[i]=ost[i]; h_cpu_op_finish[i]=oft[i]; } 
                            }
                        } 
                    }
                }
            } else {
                local_partial_states++;
                int moc[HARD_MAX_MACHINES]={},mrt[HARD_MAX_MACHINES]={};
                int cpo[HARD_MAX_OPS]={},ost[HARD_MAX_OPS]={},oft[HARD_MAX_OPS]={};
                int pr[HARD_MAX_OPS]={},sq[HARD_MAX_OPS]={};
                int m_rem[HARD_MAX_MACHINES]={}, j_ready[HARD_MAX_JOBS]={}, j_rem[HARD_MAX_JOBS]={};

                for (int m=0;m<=nm;m++) { moc[m]=par[SNAP_OFF_MACHINE_OCCUPIED_OP+m]; mrt[m]=par[SNAP_OFF_MACHINE_READY_TIME+m]; m_rem[m]=par[SNAP_OFF_MACH_REM+m];}
                for (int j=0;j<nj;j++) { j_ready[j]=par[SNAP_OFF_JOB_READY+j]; j_rem[j]=par[SNAP_OFF_JOB_REM+j]; }
                for (int op=0;op<nops;op++) { cpo[op]=par[SNAP_OFF_CRITICAL_PARENT_OP+op]; ost[op]=par[SNAP_OFF_OP_START_TIME+op]; oft[op]=par[SNAP_OFF_OP_FINISH_TIME+op]; }
                int nr2=par[SNAP_OFF_READY_COUNT]; for (int i=0;i<nr2;i++) pr[i]=par[SNAP_OFF_PRECEDENCE_READY+i];
                int ns2=par[SNAP_OFF_SEQUENCE_COUNT]; for (int i=0;i<ns2;i++) sq[i]=par[SNAP_OFF_SEQUENCE+i];
                
                cpu_check_swap_wrapper(moc,mrt,cpo,pr,nr2,sq,ns2,nm,nops,nm,ost,m_rem,j_ready,j_rem);
                int cmax=0; for (int m=1;m<=nm;m++) if (mrt[m]>cmax) cmax=mrt[m];
                if (cmax>=cpu_best) continue; 
                
                double sc = cpu_compute_score(cmax, ns2, lb_mode, mrt, m_rem, j_ready, j_rem, pr, nr2, nops);
                scores[cand*nj] = sc;

                int ci=cand*nj; int* out=children.data()+ci*SNAPSHOT_INTS;
                for (int m=0;m<=nm;m++) { out[SNAP_OFF_MACHINE_OCCUPIED_OP+m]=moc[m]; out[SNAP_OFF_MACHINE_READY_TIME+m]=mrt[m]; out[SNAP_OFF_MACH_REM+m]=m_rem[m]; }
                for (int j=0;j<nj;j++) { out[SNAP_OFF_JOB_READY+j]=j_ready[j]; out[SNAP_OFF_JOB_REM+j]=j_rem[j]; }
                for (int op=0;op<nops;op++) { out[SNAP_OFF_CRITICAL_PARENT_OP+op]=cpo[op]; out[SNAP_OFF_OP_START_TIME+op]=ost[op]; out[SNAP_OFF_OP_FINISH_TIME+op]=oft[op]; }
                for (int i=0;i<nr2;i++) out[SNAP_OFF_PRECEDENCE_READY+i]=pr[i];
                out[SNAP_OFF_READY_COUNT]=nr2; for (int i=0;i<ns2;i++) out[SNAP_OFF_SEQUENCE+i]=sq[i];
                out[SNAP_OFF_SEQUENCE_COUNT]=ns2; out[SNAP_OFF_MAKESPAN]=cmax;

                if (ns2==total_real) { 
                    int pv=g_shared_best.load(memory_order_relaxed); 
                    while (cmax<pv&&!g_shared_best.compare_exchange_weak(pv,cmax,memory_order_release,memory_order_relaxed)){} 
                    if (cmax<cpu_best) { 
                        cpu_best=cmax; 
                        std::lock_guard<std::mutex> lock(cpu_best_mutex);
                        if (cmax < h_cpu_best_makespan) {
                            h_cpu_best_makespan = cmax;
                            for (int i=0;i<total_real;i++) h_cpu_best_seq[i]=sq[i]; 
                            for (int i=0;i<nops;i++) { h_cpu_op_start[i]=ost[i]; h_cpu_op_finish[i]=oft[i]; } 
                        }
                    } 
                }
            }
        }
        int tch=bsz*nj;
        for (int i=0;i<tch;i++) indices[i]=i;
        sort(indices.begin(),indices.begin()+tch,[&](int a,int b){return scores[a]<scores[b];});
        int valid=0, keep=min(tch,cpu_bw);
        while (valid<keep && scores[indices[valid]]<1e8f) valid++;
        if (valid==0) break;
        for (int i=0;i<valid;i++) memcpy(beam.data()+i*SNAPSHOT_INTS, children.data()+indices[i]*SNAPSHOT_INTS, SNAPSHOT_INTS*sizeof(int));
        bsz=valid;
    }
    
    std::lock_guard<std::mutex> lock(stats_mutex);
    rank_stats.partial_states += local_partial_states;
    rank_stats.complete_solutions++;
    rank_stats.beam_iterations++;
    
    return cpu_best;
}



// -----------------------------------------------------------------------------
// MAIN 
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int num_iterations = 50; // Default iterations per instance

    // Note: argv[1] is intentionally ignored here since we loop automatically.
    if (argc > 1) {
        if (argc>2 && argv[2]) beam_width=atoi(argv[2]);
        if (argc>3 && argv[3]) accel_factor=atoi(argv[3]);
        if (argc>4 && argv[4]) num_iterations=atoi(argv[4]);
        if (argc>5 && argv[5]) {
            if (strcmp(argv[5], "auto") == 0) g_lb_mode_cli = -1;
            else g_lb_mode_cli = atoi(argv[5]);
        }
    }
    
    if (beam_width<1) { fprintf(stderr,"beam_width must be >= 1\n"); return 1; }
    if (beam_width>HARD_MAX_BEAM) { fprintf(stderr,"beam_width > HARD_MAX_BEAM\n"); return 1; }

    MPI_Init(&argc,&argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

    int num_gpus=0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus==0) { fprintf(stderr,"No GPU found.\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    CUDA_CHECK(cudaSetDevice(world_rank%num_gpus));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize,32768));

    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    
    if (world_rank==0) {
        printf("=== Heterogeneous Hybrid CPU+GPU Beam Search (BATCH BENCHMARK) ===\n");
        printf("Target: Taillard Instances ta01.txt to ta80.txt\n");
        printf("GPU beam_width: %d\n", beam_width);
        printf("CPU accel_factor: %s\n", accel_factor > 0 ? std::to_string(accel_factor).c_str() : "Auto-calibrated");
        printf("Iterations PER INSTANCE: %d\n", num_iterations);
        printf("MPI ranks: %d  |  GPUs: %d  |  OpenMP threads/rank: %d\n",world_size,num_gpus,num_threads);
        printf("LB mode: %s\n", g_lb_mode_cli < 0 ? "auto (heterogeneous per thread)" : lb_mode_name[g_lb_mode_cli]);
        printf("Architecture: Independent Node Search (Zero MPI overhead inside loop)\n");
        printf("==================================================================\n\n");
    }

    // Benchmark loop: Instances 01 to 80
    for (int inst = 1; inst <= 80; inst++) {
        // Format instance name (Change "ta" to "la" if using Lawrence instances)
        sprintf(instance_path, "../../taillard_instances/ta%02d.txt", inst);

        // Check if file exists, skip if missing
        ifstream chk(instance_path); 
        if (!chk) { 
            if (world_rank == 0) printf("Skipping %s (file not found)\n", instance_path);
            continue; 
        }

        if (world_rank == 0) printf("\n>>> Starting Instance: %s <<<\n", instance_path);

        // Load and Allocate for THIS instance
        load_instance();
        BUF_BEAM = beam_width;
        compute_snapshot_layout();
        allocate_gpu_buffers();

        int gpu_bw = beam_width;
        int cpu_bw;

        // Reset global shared variables for the new instance
        g_shared_best.store(500000, memory_order_relaxed);
        global_best_makespan = 500000;
        h_cpu_best_makespan = 500000; 

        // Reset throughput statistics
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            rank_stats = {0, 0, 0, 0.0};
        }

        // Auto-Calibrated CPU Beam Width
        if (accel_factor > 0) {
            cpu_bw = max(1, beam_width / accel_factor);
        } else {
            if (world_rank == 0) printf("Auto-calibrating CPU beam width for %s...\n", instance_path);
            unsigned int calib_seed = 12345 + inst;
            double calib_start = MPI_Wtime();
            
            run_beam_construction_cpu(50, &calib_seed, 0); 
            
            double calib_t = MPI_Wtime() - calib_start;
            double target = 0.5;
            cpu_bw = (calib_t > 0.0) ? max(50, min(8000, (int)(50.0 * target / calib_t))) : 500;
            
            // Clean up states messed up by the calibration run
            g_shared_best.store(500000, memory_order_relaxed);
            h_cpu_best_makespan = 500000;
            {
                std::lock_guard<std::mutex> lock(stats_mutex);
                rank_stats = {0, 0, 0, 0.0};
            }
        }

        srand((unsigned)time(NULL) + world_rank + inst);
        
        int my_gpu_local = 500000, my_cpu_local = 500000;
        int iter = 0;
        
        double loop_start = MPI_Wtime();
        double next_print = loop_start + 10.0;

        // Independent Search - Fixed Iterations
        for (iter = 0; iter < num_iterations; iter++) {
            int gm = 500000;
            int cm = 500000;

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int lb_mode = (g_lb_mode_cli >= 0) ? g_lb_mode_cli : ((world_rank * num_threads + tid) % LB_N_MODES);

                if (tid == 0) {
                    /* GPU beam search*/
                    int local_gm = run_beam_construction(lb_mode);
                    if (local_gm < gm) gm = local_gm;
                } else {
                    /* CPU beam search*/
                    unsigned int local_seed = (unsigned int)time(NULL) + world_rank * 1000 + iter * 100 + tid + inst;
                    int local_cm = run_beam_construction_cpu(cpu_bw, &local_seed, lb_mode);
                    #pragma omp critical
                    {
                        if (local_cm < cm) cm = local_cm;
                    }
                }
            }

            if (gm < my_gpu_local) my_gpu_local = gm;
            if (cm < my_cpu_local) my_cpu_local = cm;

            int node_best = min(my_gpu_local, my_cpu_local);
            int prev = g_shared_best.load(std::memory_order_relaxed);
            while (node_best < prev && !g_shared_best.compare_exchange_weak(prev, node_best, std::memory_order_release, std::memory_order_relaxed)) {}

            if (world_rank == 0) {
                double now = MPI_Wtime();
                if (now >= next_print) {
                    printf("[%s] iter %4d/%d  | Node 0 Best: %d  (%.1f s elapsed)\n", 
                           instance_path, iter + 1, num_iterations, node_best, now - loop_start);
                    fflush(stdout);
                    next_print = now + 10.0;
                }
            }
        }
        
        rank_stats.elapsed_sec = MPI_Wtime() - loop_start;

        // Iterations finished for this instance. Gather global best across all MPI ranks.
        int local_final = g_shared_best.load(memory_order_relaxed);
        int global_final;
        MPI_Allreduce(&local_final, &global_final, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        long long sum_partial = 0, sum_complete = 0;
        double ps_rate = (rank_stats.elapsed_sec > 0) ? rank_stats.partial_states / rank_stats.elapsed_sec : 0;
        double cs_rate = (rank_stats.elapsed_sec > 0) ? rank_stats.complete_solutions / rank_stats.elapsed_sec : 0;
        double sum_ps_rate = 0, sum_cs_rate = 0;
        
        MPI_Reduce(&rank_stats.partial_states, &sum_partial, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&rank_stats.complete_solutions, &sum_complete, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&ps_rate, &sum_ps_rate, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&cs_rate, &sum_cs_rate, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            printf("\n=== RESULT FOR %s ===\n", instance_path);
            printf("Best makespan            : %d\n", global_final);
            printf("Total execution time     : %.2f s\n", rank_stats.elapsed_sec);
            printf("Total partial states     : %lld\n", sum_partial);
            printf("Aggregate states/s       : %.0f\n", sum_ps_rate);
            printf("------------------------------------------------------------------\n");
        }

        int is_best_rank = (local_final == global_final) ? world_rank : world_size;
        int validating_rank;
        MPI_Allreduce(&is_best_rank, &validating_rank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (world_rank == validating_rank) {
            printf("Rank %d validating the global best solution for %s:\n", world_rank, instance_path);
            if (my_cpu_local == local_final) {
                cpu_validate_schedule(h_cpu_best_seq, num_jobs * num_machines, num_machines, num_ops_total);
            } else {
                cpu_validate_schedule(h_global_best_seq, num_jobs * num_machines, num_machines, num_ops_total);
            }
        }

        // Free device memory before the next loop iteration safely
        free_gpu_buffers();
        
        // Barrier so fast nodes don't race ahead into the next instance
        MPI_Barrier(MPI_COMM_WORLD);
    } // End of instances loop

    MPI_Finalize();
    return 0;
}