
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <climits>
#include <cfloat>
#include <atomic>
#include <omp.h>

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <queue>
#include <random>
#include <fstream>
#include <unordered_map>
#include <ctime>
#include <cmath>
#include <thread>
#include <mutex>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

using namespace std;

/* =========================================================================
 *  CUDA error check
 * ========================================================================= */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* =========================================================================
 *  Shared global best (updated by both CPU and GPU sides)
 * ========================================================================= */
static std::atomic<int> g_shared_best(500000);   // INT best makespan seen so far

/* =========================================================================
 *  Instance description (read once, used by both sides)
 * ========================================================================= */
static char instance_path[256] = "./la33.txt";
static int  accel_factor       = 34;

/* =========================================================================
 * =========================================================================
 *  CUDA Beam Search (from gpu-version.cpp)
 * =========================================================================
 * ========================================================================= */

/* ---------- GPU sizing constants --------------------------------------- */
static int BUF_OPS      = 0;
static int BUF_MACHINES = 0;
static int BUF_JOBS     = 0;
static int BUF_BEAM     = 0;

#define HARD_MAX_OPS      1000
#define HARD_MAX_MACHINES   50
#define HARD_MAX_JOBS       50
#define HARD_MAX_BEAM     30000

struct OperationInfo { int machine_id, duration, job_id; };

__constant__ OperationInfo c_ops[HARD_MAX_OPS];
__constant__ int c_op_order[HARD_MAX_MACHINES * HARD_MAX_JOBS];
__constant__ int c_op_count[HARD_MAX_MACHINES];

/* Flat snapshot layout offsets */
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

static void pack_snapshot(const ScheduleSnapshot& s, int* flat) {
    memcpy(flat+SNAP_OFF_MACHINE_OCCUPIED_OP, s.machine_occupied_op,  BUF_MACHINES*sizeof(int));
    memcpy(flat+SNAP_OFF_MACHINE_READY_TIME,  s.machine_ready_time,   BUF_MACHINES*sizeof(int));
    memcpy(flat+SNAP_OFF_CRITICAL_PARENT_OP,  s.critical_parent_op,   BUF_OPS*sizeof(int));
    memcpy(flat+SNAP_OFF_OP_START_TIME,       s.op_start_time,        BUF_OPS*sizeof(int));
    memcpy(flat+SNAP_OFF_OP_FINISH_TIME,      s.op_finish_time,       BUF_OPS*sizeof(int));
    memcpy(flat+SNAP_OFF_PRECEDENCE_READY,    s.precedence_ready_ops, BUF_OPS*sizeof(int));
    flat[SNAP_OFF_READY_COUNT]    = s.num_precedence_ready;
    memcpy(flat+SNAP_OFF_SEQUENCE,            s.seq,                  BUF_OPS*sizeof(int));
    flat[SNAP_OFF_SEQUENCE_COUNT] = s.num_scheduled;
    flat[SNAP_OFF_MAKESPAN]       = s.makespan;
}

/* ---- GPU instance data ------------------------------------------------ */
static int num_jobs     = 0;
static int num_machines = 0;
static int num_ops_total= 0;
static int beam_width   = 1000;
static int gpu_best     = 500000;

static OperationInfo all_ops_gpu[HARD_MAX_OPS];
static vector<int>   ops_by_machine_gpu[HARD_MAX_MACHINES];
static int           machine_op_order[HARD_MAX_MACHINES * HARD_MAX_JOBS];
static int           machine_op_count[HARD_MAX_MACHINES];

/* ---- Device buffers -------------------------------------------------- */
static int   *d_beam_snapshots    = nullptr;
static int   *d_child_snapshots   = nullptr;
static float *d_child_scores      = nullptr;
static float *d_child_scores_out  = nullptr;
static int   *d_child_indices     = nullptr;
static int   *d_child_indices_out = nullptr;
static void  *d_cub_temp_storage  = nullptr;
static size_t cub_temp_bytes      = 0;
static int   *d_global_best_makespan = nullptr;
static int   *d_best_solution        = nullptr;
static int   *h_input_flat           = nullptr;

static int h_global_best_seq  [HARD_MAX_OPS];
static int h_global_op_start  [HARD_MAX_OPS];
static int h_global_op_finish [HARD_MAX_OPS];

static int h_cpu_best_seq  [HARD_MAX_OPS];
static int h_cpu_op_start  [HARD_MAX_OPS];
static int h_cpu_op_finish [HARD_MAX_OPS];
static std::mutex cpu_best_mutex;
static int h_cpu_best_makespan = 500000;

static bool cpu_validate_schedule(const int* seq, const int* op_start,
                                   int num_ops_scheduled, int num_machines_val, int num_ops_tot)
{
    if (num_ops_scheduled == 0) return false;

    bool valid = true;

    for (int i = 1; i < num_ops_tot - 1; i++) {
        int start_i    = op_start[i];
        int duration_i = all_ops_gpu[i].duration;
        int machine_i  = all_ops_gpu[i].machine_id;

        int finish_i;
        bool is_last_in_job = (i % num_machines_val == 0);
        if (!is_last_in_job) {
            finish_i = op_start[i + 1];
        } else {
            finish_i = start_i + duration_i;
        }

        if (finish_i < start_i + duration_i) {
            printf("VALIDATION FAILED: Op %d completes before its processing time "
                   "(start=%d, duration=%d, finish=%d)\n",
                   i, start_i, duration_i, finish_i);
            valid = false;
        }

        for (int j = i + 1; j < num_ops_tot - 1; j++) {
            if (all_ops_gpu[j].machine_id != machine_i) continue;

            int start_j  = op_start[j];
            int finish_j = (j % num_machines_val != 0)
                           ? op_start[j + 1]
                           : start_j + all_ops_gpu[j].duration;

            if (!(finish_i <= start_j || finish_j <= start_i)) {
                printf("VALIDATION FAILED: Machine %d overlap – Op %d [%d,%d) vs Op %d [%d,%d)\n",
                       machine_i, i, start_i, finish_i, j, start_j, finish_j);
                valid = false;
            }
        }
    }

    int verified_makespan = 0;
    for (int i = 1; i < num_ops_tot - 1; i++) {
        int finish_i = (i % num_machines_val != 0)
                       ? op_start[i + 1]
                       : op_start[i] + all_ops_gpu[i].duration;
        if (finish_i > verified_makespan) {
            verified_makespan = finish_i;
        }
    }

    if (valid)
        printf("VALIDATION SUCCESS: BJSSP constraints verified. Makespan = %d\n",
               verified_makespan);
    return valid;
}

/* ---- GPU device functions (identical to gpu-version.cpp) ------------- */
__device__ static int dev_find_in_array(const int* arr, int len, int val) {
    for (int i=0; i<len; i++) if (arr[i]==val) return i;
    return -1;
}
__device__ static int find_op_rank_on_machine(int op_id, int machine, int stride) {
    int n = c_op_count[machine];
    for (int r=0; r<n; r++)
        if (c_op_order[machine*stride+r]==op_id) return r;
    return -1;
}
__device__ static void traiter_gpu(
    int op_id,
    int* machine_occupied_op, int* machine_ready_time, int* critical_parent_op,
    int* op_start_time, int* op_finish_time,
    int* precedence_ready_ops, int& num_ready,
    int* seq, int& num_scheduled,
    int num_machines_1idx, int num_ops_tot, int machine_order_stride)
{
    int cascade_stack[HARD_MAX_OPS], cascade_top=0, current=op_id;
    do {
        int machine = c_ops[current].machine_id;
        seq[num_scheduled++] = current;
        bool is_first = (current % num_machines_1idx == 1);
        int start;
        if (!is_first) {
            int pred=current-1, pred_mc=c_ops[pred].machine_id;
            if (machine_ready_time[machine] <= machine_ready_time[pred_mc]) {
                start=machine_ready_time[pred_mc]; critical_parent_op[current]=pred;
            } else {
                start=machine_ready_time[machine];
                int rank=find_op_rank_on_machine(current,machine,machine_order_stride);
                int prev=-100;
                if (rank>0) { prev=c_op_order[machine*machine_order_stride+rank-1]; if (prev%num_machines_1idx!=0) prev++; }
                critical_parent_op[current]=prev;
            }
        } else {
            int rank=find_op_rank_on_machine(current,machine,machine_order_stride);
            if (rank>0) { int prev=c_op_order[machine*machine_order_stride+rank-1]; if (prev%num_machines_1idx!=0) prev++; critical_parent_op[current]=prev; }
            else critical_parent_op[current]=0;
            start=machine_ready_time[machine];
        }
        op_start_time[current]=start;
        machine_ready_time[machine]=start+c_ops[current].duration;
        machine_occupied_op[machine]=current;
        if (!is_first) { int pm=c_ops[current-1].machine_id; machine_occupied_op[pm]=0; machine_ready_time[pm]=start; }
        bool is_last=(current%num_machines_1idx==0);
        if (!is_last) { precedence_ready_ops[num_ready++]=current+1; }
        else {
            op_finish_time[current]=machine_ready_time[machine];
            if (op_finish_time[current]>op_start_time[num_ops_tot-1])
                op_start_time[num_ops_tot-1]=op_finish_time[current];
            machine_occupied_op[machine]=0;
        }
        current=(cascade_top>0)?cascade_stack[--cascade_top]:-1;
    } while (current!=-1);
}
__device__ static void traiter_swap_gpu(
    int op_id,
    int* machine_occupied_op, int* machine_ready_time,
    int* precedence_ready_ops, int& num_ready,
    int* seq, int& num_scheduled,
    int num_machines_1idx, int* op_start_time)
{
    int machine=c_ops[op_id].machine_id;
    seq[num_scheduled++]=op_id;
    int sync_start=machine_ready_time[machine];
    op_start_time[op_id]=sync_start;
    machine_ready_time[machine]=sync_start+c_ops[op_id].duration;
    machine_occupied_op[machine]=op_id;
    if (op_id%num_machines_1idx!=0) precedence_ready_ops[num_ready++]=op_id+1;
    else machine_occupied_op[machine]=0;
}
__device__ static int solinit_swap_new_gpu(
    const int* machine_occupied_op, int* out_cycle,
    int num_machines_1idx, int num_ops_tot, int num_machines_actual)
{
    int best_cycle_len=0;
    for (int sm=1; sm<=num_machines_actual; sm++) {
        int op_on_start=machine_occupied_op[sm];
        if (op_on_start==0) continue;
        int next_op_in_job=op_on_start+1;
        if (next_op_in_job<=0||next_op_in_job>=num_ops_tot) continue;
        int next_mc=c_ops[next_op_in_job].machine_id;
        if (machine_occupied_op[next_mc]==0) continue;
        int chain[HARD_MAX_MACHINES], chain_len=0, cycle_start_idx=-1;
        chain[chain_len++]=sm;
        int cur_mc=sm;
        do {
            int op_cur=machine_occupied_op[cur_mc];
            int next_op=op_cur+1;
            if (next_op<=0||next_op>=num_ops_tot) { chain_len=0; break; }
            int needed_mc=c_ops[next_op].machine_id;
            int op_on_needed=machine_occupied_op[needed_mc];
            int next_op2=op_on_needed+1;
            bool blocked=(op_on_needed!=0)&&(op_on_needed<num_ops_tot)&&(next_op2>0)&&(next_op2<num_ops_tot)&&(machine_occupied_op[c_ops[next_op2].machine_id]>0);
            if (!blocked) { chain_len=0; break; }
            cycle_start_idx=dev_find_in_array(chain,chain_len,needed_mc);
            if (cycle_start_idx==-1) { if (chain_len<HARD_MAX_MACHINES) chain[chain_len++]=needed_mc; cur_mc=needed_mc; }
        } while (cycle_start_idx==-1 && chain_len<HARD_MAX_MACHINES);
        if (cycle_start_idx>=0) {
            int cl=chain_len-cycle_start_idx;
            if (cl>=2&&cl>best_cycle_len) {
                best_cycle_len=cl;
                for (int i=0;i<cl;i++) out_cycle[i]=chain[cycle_start_idx+i];
            }
        }
    }
    return best_cycle_len;
}
__device__ static void check_swap_gpu(
    int* machine_occupied_op, int* machine_ready_time, int* critical_parent_op,
    int* precedence_ready_ops, int& num_ready,
    int* seq, int& num_scheduled,
    int num_machines_1idx, int num_ops_tot, int num_machines_actual,
    int* op_start_time)
{
    int cycle_machines[HARD_MAX_MACHINES];
    int cycle_len=solinit_swap_new_gpu(machine_occupied_op,cycle_machines,num_machines_1idx,num_ops_tot,num_machines_actual);
    if (cycle_len<=1) return;
    int sync_time=0, pivot_op=0;
    for (int i=0;i<cycle_len;i++) {
        int mc=cycle_machines[i];
        if (mc!=0&&machine_ready_time[mc]>sync_time) { sync_time=machine_ready_time[mc]; pivot_op=machine_occupied_op[mc]; }
    }
    pivot_op++;
    for (int i=0;i<cycle_len;i++) if (cycle_machines[i]!=0) machine_ready_time[cycle_machines[i]]=sync_time;
    int blocked_ops[HARD_MAX_MACHINES];
    for (int i=0;i<cycle_len;i++) { int mc=cycle_machines[i]; blocked_ops[i]=(mc!=0)?machine_occupied_op[mc]+1:0; }
    for (int i=0;i<cycle_len;i++) {
        int op=blocked_ops[i]; if (op==0) continue;
        int mc=c_ops[op].machine_id;
        critical_parent_op[op]=(op==pivot_op)?pivot_op-1:machine_occupied_op[mc]+1;
    }
    for (int i=0;i<cycle_len;i++) {
        int op=blocked_ops[i]; if (op==0) continue;
        int pos=dev_find_in_array(precedence_ready_ops,num_ready,op);
        if (pos>=0) precedence_ready_ops[pos]=precedence_ready_ops[--num_ready];
        traiter_swap_gpu(op,machine_occupied_op,machine_ready_time,precedence_ready_ops,num_ready,seq,num_scheduled,num_machines_1idx,op_start_time);
    }
}

/* ---- GPU kernels (identical to gpu-version.cpp) ---------------------- */
__global__ void kernel_expand_beam_layer(
    const int* d_in_beam, int* d_out_snapshots, float* d_out_scores,
    int* d_global_best, int* d_best_sol,
    int num_candidates, int num_branches_per_candidate, int SNAPSHOT_INTS_val,
    int off_moc, int off_mrt, int off_cpo,
    int off_ost, int off_oft, int off_pr, int off_rc,
    int off_seq, int off_sc, int off_ms,
    int num_machines_1idx, int num_ops_tot, int total_real_ops,
    int machine_order_stride, int random_seed)
{
    extern __shared__ int s_data[];
    int* s_parent_snap    = s_data;
    int* s_candidate_ops  = s_parent_snap + SNAPSHOT_INTS_val;
    int* s_num_cand_ptr   = s_candidate_ops + HARD_MAX_JOBS;

    int candidate_id = blockIdx.x, branch_id = threadIdx.x;
    if (candidate_id >= num_candidates) return;

    int child_flat = candidate_id * num_branches_per_candidate + branch_id;
    int out_base   = child_flat * SNAPSHOT_INTS_val;
    d_out_scores[child_flat] = 1e9f;
    d_out_snapshots[out_base + off_ms] = INT_MAX;

    for (int i=branch_id; i<SNAPSHOT_INTS_val; i+=blockDim.x)
        s_parent_snap[i]=d_in_beam[candidate_id*SNAPSHOT_INTS_val+i];
    __syncthreads();

    if (branch_id==0) {
        int free_ops_buf[HARD_MAX_JOBS]; int num_free=0;
        int num_ready=s_parent_snap[off_rc];
        for (int i=0;i<num_ready;i++) {
            int op=s_parent_snap[off_pr+i];
            if (op>=1&&op<num_ops_tot&&s_parent_snap[off_moc+c_ops[op].machine_id]==0)
                free_ops_buf[num_free++]=op;
        }
        if (num_free>0) {
            unsigned h=random_seed+candidate_id; h^=h>>16; h*=0x7feb352d; h^=h>>15; h*=0x846ca68b; h^=h>>16;
            int chosen_mc=c_ops[free_ops_buf[h%num_free]].machine_id;
            *s_num_cand_ptr=0;
            for (int i=0;i<num_ready;i++) {
                int op=s_parent_snap[off_pr+i];
                if (op>=1&&op<num_ops_tot&&c_ops[op].machine_id==chosen_mc&&s_parent_snap[off_moc+chosen_mc]==0)
                    s_candidate_ops[(*s_num_cand_ptr)++]=op;
            }
        } else *s_num_cand_ptr=0;
    }
    __syncthreads();
    int num_valid_ops=*s_num_cand_ptr;

    /* Deadlock fallback */
    if (num_valid_ops==0) {
        if (branch_id!=0) return;
        int moc[HARD_MAX_MACHINES]={}, mrt[HARD_MAX_MACHINES]={};
        int cpo[HARD_MAX_OPS]={}, ost[HARD_MAX_OPS]={}, oft[HARD_MAX_OPS]={};
        int pr[HARD_MAX_OPS]={}, seq_g[HARD_MAX_OPS]={};
        for (int m=0;m<=num_machines_1idx;m++) { moc[m]=s_parent_snap[off_moc+m]; mrt[m]=s_parent_snap[off_mrt+m]; }
        for (int op=0;op<num_ops_tot;op++) { cpo[op]=s_parent_snap[off_cpo+op]; ost[op]=s_parent_snap[off_ost+op]; oft[op]=s_parent_snap[off_oft+op]; }
        int num_ready=s_parent_snap[off_rc];
        for (int i=0;i<num_ready;i++) pr[i]=s_parent_snap[off_pr+i];
        int num_sched=s_parent_snap[off_sc];
        for (int i=0;i<num_sched;i++) seq_g[i]=s_parent_snap[off_seq+i];
        check_swap_gpu(moc,mrt,cpo,pr,num_ready,seq_g,num_sched,num_machines_1idx,num_ops_tot,num_machines_1idx,ost);
        int cmax=0; for (int m=1;m<=num_machines_1idx;m++) if (mrt[m]>cmax) cmax=mrt[m];
        if (cmax>=*d_global_best) return;
        float base=(num_sched==0)?(float)cmax:((float)cmax/(float)num_sched);
        unsigned nh=random_seed^(candidate_id*0x85ebca6b); nh^=nh>>16; nh*=0x85ebca6b; nh^=nh>>16; nh*=0xc2b2ae35; nh^=nh>>16;
        float noise=(float)(nh%1000)/1e9f;
        d_out_scores[candidate_id*num_branches_per_candidate]=base+noise;
        int* out=d_out_snapshots+(candidate_id*num_branches_per_candidate)*SNAPSHOT_INTS_val;
        for (int m=0;m<=num_machines_1idx;m++) { out[off_moc+m]=moc[m]; out[off_mrt+m]=mrt[m]; }
        for (int op=0;op<num_ops_tot;op++) { out[off_cpo+op]=cpo[op]; out[off_ost+op]=ost[op]; out[off_oft+op]=oft[op]; }
        for (int i=0;i<num_ready;i++) out[off_pr+i]=pr[i];
        out[off_rc]=num_ready;
        for (int i=0;i<num_sched;i++) out[off_seq+i]=seq_g[i];
        out[off_sc]=num_sched; out[off_ms]=cmax;
        if (num_sched==total_real_ops) { int old=atomicMin(d_global_best,cmax); if (cmax<old) for (int i=0;i<SNAPSHOT_INTS_val;i++) d_best_sol[i]=out[i]; }
        return;
    }

    if (branch_id>=num_valid_ops) return;
    int moc[HARD_MAX_MACHINES]={}, mrt[HARD_MAX_MACHINES]={};
    int cpo[HARD_MAX_OPS]={}, ost[HARD_MAX_OPS]={}, oft[HARD_MAX_OPS]={};
    int pr[HARD_MAX_OPS]={}, seq_g[HARD_MAX_OPS]={};
    for (int m=0;m<=num_machines_1idx;m++) { moc[m]=s_parent_snap[off_moc+m]; mrt[m]=s_parent_snap[off_mrt+m]; }
    for (int op=0;op<num_ops_tot;op++) { cpo[op]=s_parent_snap[off_cpo+op]; ost[op]=s_parent_snap[off_ost+op]; oft[op]=s_parent_snap[off_oft+op]; }
    int num_ready=s_parent_snap[off_rc];
    for (int i=0;i<num_ready;i++) pr[i]=s_parent_snap[off_pr+i];
    int num_sched=s_parent_snap[off_sc];
    for (int i=0;i<num_sched;i++) seq_g[i]=s_parent_snap[off_seq+i];
    int chosen_op=s_candidate_ops[branch_id];
    int rpos=dev_find_in_array(pr,num_ready,chosen_op);
    if (rpos>=0) pr[rpos]=pr[--num_ready];
    traiter_gpu(chosen_op,moc,mrt,cpo,ost,oft,pr,num_ready,seq_g,num_sched,num_machines_1idx,num_ops_tot,machine_order_stride);
    check_swap_gpu(moc,mrt,cpo,pr,num_ready,seq_g,num_sched,num_machines_1idx,num_ops_tot,num_machines_1idx,ost);
    int cmax=0; for (int m=1;m<=num_machines_1idx;m++) if (mrt[m]>cmax) cmax=mrt[m];
    if (cmax>=*d_global_best) return;
    float base=(num_sched==0)?(float)cmax:((float)cmax/(float)num_sched);
    unsigned nh=random_seed^(candidate_id*0x85ebca6b)^(branch_id*0xc2b2ae35); nh^=nh>>16; nh*=0x85ebca6b; nh^=nh>>16; nh*=0xc2b2ae35; nh^=nh>>16;
    float noise=(float)(nh%1000)/1e9f;
    d_out_scores[child_flat]=base+noise;
    int* out=d_out_snapshots+out_base;
    for (int m=0;m<=num_machines_1idx;m++) { out[off_moc+m]=moc[m]; out[off_mrt+m]=mrt[m]; }
    for (int op=0;op<num_ops_tot;op++) { out[off_cpo+op]=cpo[op]; out[off_ost+op]=ost[op]; out[off_oft+op]=oft[op]; }
    for (int i=0;i<num_ready;i++) out[off_pr+i]=pr[i];
    out[off_rc]=num_ready;
    for (int i=0;i<num_sched;i++) out[off_seq+i]=seq_g[i];
    out[off_sc]=num_sched; out[off_ms]=cmax;
    if (num_sched==total_real_ops) { int old=atomicMin(d_global_best,cmax); if (cmax<old) for (int i=0;i<SNAPSHOT_INTS_val;i++) d_best_sol[i]=out[i]; }
}

__global__ void init_indices_gpu(int* indices, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) indices[i]=i;
}

__global__ void kernel_gather_topk(
    const int* d_in_children, const int* d_sorted_indices,
    int* d_out_beam, int num_to_gather, int SNAPSHOT_INTS_val)
{
    int rank=blockIdx.x; if (rank>=num_to_gather) return;
    int src=d_sorted_indices[rank], tid=threadIdx.x;
    for (int i=tid;i<SNAPSHOT_INTS_val;i+=blockDim.x)
        d_out_beam[rank*SNAPSHOT_INTS_val+i]=d_in_children[src*SNAPSHOT_INTS_val+i];
}

/* ---- GPU init / cleanup --------------------------------------------- */
static void load_instance_gpu() {
    FILE* f=fopen(instance_path,"r");
    if (!f) { fprintf(stderr,"Cannot open %s\n",instance_path); exit(1); }
    fscanf(f,"%d %d\n",&num_jobs,&num_machines);
    num_ops_total=num_jobs*num_machines+2;
    BUF_OPS=num_ops_total; BUF_MACHINES=num_machines+1; BUF_JOBS=num_jobs;
    memset(all_ops_gpu,0,sizeof(all_ops_gpu));
    int op_id=1;
    for (int job=1;job<=num_jobs;job++)
        for (int step=1;step<=num_machines;step++) {
            int raw,dur; fscanf(f,"%d %d",&raw,&dur);
            int mc=raw+1; all_ops_gpu[op_id]={mc,dur,job};
            ops_by_machine_gpu[mc].push_back(op_id); op_id++;
        }
    all_ops_gpu[num_jobs*num_machines+1]={0,0,0};
    fclose(f);
    memset(machine_op_order,0,sizeof(machine_op_order));
    memset(machine_op_count,0,sizeof(machine_op_count));
    for (int m=1;m<=num_machines;m++) {
        int n=(int)ops_by_machine_gpu[m].size(); machine_op_count[m]=n;
        for (int r=0;r<n;r++) machine_op_order[m*HARD_MAX_JOBS+r]=ops_by_machine_gpu[m][r];
    }
}

static void allocate_gpu_buffers() {
    CUDA_CHECK(cudaMemcpyToSymbol(c_ops,all_ops_gpu,HARD_MAX_OPS*sizeof(OperationInfo)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_op_order,machine_op_order,HARD_MAX_MACHINES*HARD_MAX_JOBS*sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_op_count,machine_op_count,HARD_MAX_MACHINES*sizeof(int)));

    size_t beam_bytes  =(size_t)BUF_BEAM*SNAPSHOT_INTS*sizeof(int);
    size_t child_bytes =(size_t)BUF_BEAM*BUF_JOBS*SNAPSHOT_INTS*sizeof(int);
    size_t score_bytes =(size_t)BUF_BEAM*BUF_JOBS*sizeof(float);
    size_t index_bytes =(size_t)BUF_BEAM*BUF_JOBS*sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_beam_snapshots,   beam_bytes));
    CUDA_CHECK(cudaMalloc(&d_child_snapshots,  child_bytes));
    CUDA_CHECK(cudaMalloc(&d_child_scores,     score_bytes));
    CUDA_CHECK(cudaMalloc(&d_child_scores_out, score_bytes));
    CUDA_CHECK(cudaMalloc(&d_child_indices,    index_bytes));
    CUDA_CHECK(cudaMalloc(&d_child_indices_out,index_bytes));
    CUDA_CHECK(cudaMalloc(&d_global_best_makespan,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_solution,SNAPSHOT_INTS*sizeof(int)));

    cub::DeviceRadixSort::SortPairs(nullptr,cub_temp_bytes,
        d_child_scores,d_child_scores_out,
        d_child_indices,d_child_indices_out,
        BUF_BEAM*BUF_JOBS);
    CUDA_CHECK(cudaMalloc(&d_cub_temp_storage,cub_temp_bytes));
    CUDA_CHECK(cudaMallocHost(&h_input_flat,beam_bytes));
}

static void free_gpu_buffers() {
    cudaFree(d_beam_snapshots);  cudaFree(d_child_snapshots);
    cudaFree(d_child_scores);    cudaFree(d_child_scores_out);
    cudaFree(d_child_indices);   cudaFree(d_child_indices_out);
    cudaFree(d_cub_temp_storage);cudaFree(d_global_best_makespan);
    cudaFree(d_best_solution);   cudaFreeHost(h_input_flat);
}

static ScheduleSnapshot make_root_snapshot_gpu() {
    ScheduleSnapshot s{};
    s.num_precedence_ready=num_jobs;
    for (int j=0;j<num_jobs;j++) s.precedence_ready_ops[j]=j*num_machines+1;
    return s;
}

/* ---- GPU beam search pass -------------------------------------------- */
static int run_beam_construction() {
    int init_bound = g_shared_best.load(memory_order_relaxed);
    CUDA_CHECK(cudaMemcpy(d_global_best_makespan,&init_bound,sizeof(int),cudaMemcpyHostToDevice));

    ScheduleSnapshot root = make_root_snapshot_gpu();
    pack_snapshot(root,h_input_flat);
    CUDA_CHECK(cudaMemcpy(d_beam_snapshots,h_input_flat,SNAPSHOT_INTS*sizeof(int),cudaMemcpyHostToDevice));

    int current_beam_size=1;
    int max_depth=num_jobs*num_machines+10;
    int total_real_ops=num_jobs*num_machines;

    for (int depth=1; depth<max_depth && current_beam_size>0; depth++) {
        int random_seed=rand();
        size_t shared_mem_bytes=(SNAPSHOT_INTS+HARD_MAX_JOBS+1)*sizeof(int);

        kernel_expand_beam_layer<<<current_beam_size,BUF_JOBS,shared_mem_bytes>>>(
            d_beam_snapshots,d_child_snapshots,d_child_scores,
            d_global_best_makespan,d_best_solution,
            current_beam_size,BUF_JOBS,SNAPSHOT_INTS,
            SNAP_OFF_MACHINE_OCCUPIED_OP,SNAP_OFF_MACHINE_READY_TIME,
            SNAP_OFF_CRITICAL_PARENT_OP,
            SNAP_OFF_OP_START_TIME,SNAP_OFF_OP_FINISH_TIME,
            SNAP_OFF_PRECEDENCE_READY,SNAP_OFF_READY_COUNT,
            SNAP_OFF_SEQUENCE,SNAP_OFF_SEQUENCE_COUNT,SNAP_OFF_MAKESPAN,
            num_machines,num_ops_total,total_real_ops,HARD_MAX_JOBS,random_seed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Push GPU best to shared atomic so CPU side can use it */
        int gpu_cur; CUDA_CHECK(cudaMemcpy(&gpu_cur,d_global_best_makespan,sizeof(int),cudaMemcpyDeviceToHost));
        int prev=g_shared_best.load(memory_order_relaxed);
        while (gpu_cur<prev && !g_shared_best.compare_exchange_weak(prev,gpu_cur,memory_order_release,memory_order_relaxed)) {}

        int total_children=current_beam_size*BUF_JOBS;
        init_indices_gpu<<<(total_children+255)/256,256>>>(d_child_indices,total_children);
        cub::DeviceRadixSort::SortPairs(d_cub_temp_storage,cub_temp_bytes,
            d_child_scores,d_child_scores_out,
            d_child_indices,d_child_indices_out,total_children);

        int candidates_to_check=min(total_children,beam_width);
        float h_top_scores[HARD_MAX_BEAM];
        CUDA_CHECK(cudaMemcpy(h_top_scores,d_child_scores_out,candidates_to_check*sizeof(float),cudaMemcpyDeviceToHost));
        int valid_count=0;
        while (valid_count<candidates_to_check && h_top_scores[valid_count]<1e8f) valid_count++;
        if (valid_count==0) break;

        kernel_gather_topk<<<valid_count,256>>>(d_child_snapshots,d_child_indices_out,d_beam_snapshots,valid_count,SNAPSHOT_INTS);
        current_beam_size=valid_count;
    }

    int device_best;
    CUDA_CHECK(cudaMemcpy(&device_best,d_global_best_makespan,sizeof(int),cudaMemcpyDeviceToHost));

    /* GPU-only best: only count improvements the GPU itself found */
    int gpu_only = (device_best < init_bound) ? device_best : 500000;

    if (device_best<gpu_best) {
        gpu_best=device_best;
        CUDA_CHECK(cudaMemcpy(h_input_flat,d_best_solution,SNAPSHOT_INTS*sizeof(int),cudaMemcpyDeviceToHost));
        if (h_input_flat[SNAP_OFF_SEQUENCE_COUNT]==total_real_ops) {
            for (int i=0;i<total_real_ops;i++) h_global_best_seq[i]=h_input_flat[SNAP_OFF_SEQUENCE+i];
            for (int i=0;i<num_ops_total;i++) {
                h_global_op_start[i] =h_input_flat[SNAP_OFF_OP_START_TIME+i];
                h_global_op_finish[i]=h_input_flat[SNAP_OFF_OP_FINISH_TIME+i];
            }
        }
    }
    return gpu_only;
}

/* =========================================================================
 *  CPU Beam Search (same algorithm as GPU, smaller beam width)
 * ========================================================================= */

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

static void cpu_traiter(int op_id,
    int* moc, int* mrt, int* cpo, int* ost, int* oft,
    int* pr, int& nr, int* seq, int& ns, int nm, int nops)
{
    int current = op_id;
    do {
        int machine = all_ops_gpu[current].machine_id;
        seq[ns++] = current;
        bool is_first = (current % nm == 1);
        int start;
        if (!is_first) {
            int pred = current - 1, pred_mc = all_ops_gpu[pred].machine_id;
            if (mrt[machine] <= mrt[pred_mc]) {
                start = mrt[pred_mc]; cpo[current] = pred;
            } else {
                start = mrt[machine];
                int rank = cpu_find_op_rank(current, machine);
                int prev = -100;
                if (rank > 0) { prev = machine_op_order[machine*HARD_MAX_JOBS+rank-1]; if (prev%nm!=0) prev++; }
                cpo[current] = prev;
            }
        } else {
            int rank = cpu_find_op_rank(current, machine);
            if (rank > 0) { int prev = machine_op_order[machine*HARD_MAX_JOBS+rank-1]; if (prev%nm!=0) prev++; cpo[current] = prev; }
            else cpo[current] = 0;
            start = mrt[machine];
        }
        ost[current] = start;
        mrt[machine] = start + all_ops_gpu[current].duration;
        moc[machine] = current;
        if (!is_first) { int pmc = all_ops_gpu[current-1].machine_id; moc[pmc] = 0; mrt[pmc] = start; }
        if (current % nm != 0) { pr[nr++] = current + 1; }
        else { oft[current] = mrt[machine]; if (oft[current] > ost[nops-1]) ost[nops-1] = oft[current]; moc[machine] = 0; }
        current = -1;
    } while (current != -1);
}

static void cpu_traiter_swap(int op_id, int* moc, int* mrt, int* pr, int& nr, int* seq, int& ns, int nm, int* ost) {
    int machine = all_ops_gpu[op_id].machine_id;
    seq[ns++] = op_id;
    ost[op_id] = mrt[machine];
    mrt[machine] += all_ops_gpu[op_id].duration;
    moc[machine] = op_id;
    if (op_id % nm != 0) pr[nr++] = op_id + 1;
    else moc[machine] = 0;
}

static int cpu_solinit_swap_new(const int* moc, int* out_cycle, int nm, int nops, int num_mach) {
    int best_len = 0;
    for (int sm = 1; sm <= num_mach; sm++) {
        int op_on = moc[sm]; if (op_on == 0) continue;
        int nxt = op_on + 1; if (nxt <= 0 || nxt >= nops) continue;
        int nmc = all_ops_gpu[nxt].machine_id; if (moc[nmc] == 0) continue;
        int chain[HARD_MAX_MACHINES], clen = 0, cstart = -1;
        chain[clen++] = sm; int cur = sm;
        do {
            int oc = moc[cur], nx = oc+1;
            if (nx <= 0 || nx >= nops) { clen = 0; break; }
            int nmc2 = all_ops_gpu[nx].machine_id, on2 = moc[nmc2], nx2 = on2+1;
            bool blk = (on2!=0)&&(on2<nops)&&(nx2>0)&&(nx2<nops)&&(moc[all_ops_gpu[nx2].machine_id]>0);
            if (!blk) { clen = 0; break; }
            cstart = cpu_find_in_array(chain, clen, nmc2);
            if (cstart == -1) { if (clen < HARD_MAX_MACHINES) chain[clen++] = nmc2; cur = nmc2; }
        } while (cstart == -1 && clen < HARD_MAX_MACHINES);
        if (cstart >= 0) { int cl = clen-cstart; if (cl >= 2 && cl > best_len) { best_len = cl; for (int i=0;i<cl;i++) out_cycle[i] = chain[cstart+i]; } }
    }
    return best_len;
}

static void cpu_check_swap(int* moc, int* mrt, int* cpo, int* pr, int& nr, int* seq, int& ns, int nm, int nops, int num_mach, int* ost) {
    int cyc[HARD_MAX_MACHINES];
    int cl = cpu_solinit_swap_new(moc, cyc, nm, nops, num_mach);
    if (cl <= 1) return;
    int st = 0, piv = 0;
    for (int i=0;i<cl;i++) { int mc=cyc[i]; if (mc!=0&&mrt[mc]>st) { st=mrt[mc]; piv=moc[mc]; } }
    piv++;
    for (int i=0;i<cl;i++) if (cyc[i]!=0) mrt[cyc[i]]=st;
    int bops[HARD_MAX_MACHINES];
    for (int i=0;i<cl;i++) { int mc=cyc[i]; bops[i]=(mc!=0)?moc[mc]+1:0; }
    for (int i=0;i<cl;i++) { int op=bops[i]; if (!op) continue; int mc=all_ops_gpu[op].machine_id; cpo[op]=(op==piv)?piv-1:moc[mc]+1; }
    for (int i=0;i<cl;i++) { int op=bops[i]; if (!op) continue; int p=cpu_find_in_array(pr,nr,op); if (p>=0) pr[p]=pr[--nr]; cpu_traiter_swap(op,moc,mrt,pr,nr,seq,ns,nm,ost); }
}

/* ---- One full CPU beam search pass ---------------------------------- */
static int run_beam_construction_cpu(int cpu_bw, unsigned int* seed)
{
    const int nm = num_machines, nops = num_ops_total, nj = num_jobs;
    const int total_real = nj * nm, max_depth = total_real + 10;

    vector<int> beam((size_t)cpu_bw * SNAPSHOT_INTS, 0);
    int max_ch = cpu_bw * nj;
    vector<int> children((size_t)max_ch * SNAPSHOT_INTS, 0);
    vector<float> scores(max_ch, 1e9f);
    vector<int> indices(max_ch);

    ScheduleSnapshot root{};
    root.num_precedence_ready = nj;
    for (int j = 0; j < nj; j++) root.precedence_ready_ops[j] = j * nm + 1;
    pack_snapshot(root, beam.data());

    int bsz = 1, cpu_best = 500000;  /* track CPU-only findings */

    for (int depth = 1; depth < max_depth && bsz > 0; depth++) {
        fill(scores.begin(), scores.end(), 1e9f);
        for (int cand = 0; cand < bsz; cand++) {
            const int* par = beam.data() + cand * SNAPSHOT_INTS;
            int free_ops[HARD_MAX_JOBS], nfree = 0;
            int nrp = par[SNAP_OFF_READY_COUNT];
            for (int i = 0; i < nrp; i++) {
                int op = par[SNAP_OFF_PRECEDENCE_READY+i];
                if (op>=1 && op<nops && par[SNAP_OFF_MACHINE_OCCUPIED_OP+all_ops_gpu[op].machine_id]==0)
                    free_ops[nfree++] = op;
            }
            if (nfree > 0) {
                int chosen_mc = all_ops_gpu[free_ops[rand_r(seed)%nfree]].machine_id;
                int cops[HARD_MAX_JOBS], ncops = 0;
                for (int i = 0; i < nrp; i++) {
                    int op = par[SNAP_OFF_PRECEDENCE_READY+i];
                    if (op>=1 && op<nops && all_ops_gpu[op].machine_id==chosen_mc && par[SNAP_OFF_MACHINE_OCCUPIED_OP+chosen_mc]==0)
                        cops[ncops++] = op;
                }
                for (int b = 0; b < ncops; b++) {
                    int moc[HARD_MAX_MACHINES]={},mrt[HARD_MAX_MACHINES]={};
                    int cpo[HARD_MAX_OPS]={},ost[HARD_MAX_OPS]={},oft[HARD_MAX_OPS]={};
                    int pr[HARD_MAX_OPS]={},sq[HARD_MAX_OPS]={};
                    for (int m=0;m<=nm;m++) { moc[m]=par[SNAP_OFF_MACHINE_OCCUPIED_OP+m]; mrt[m]=par[SNAP_OFF_MACHINE_READY_TIME+m]; }
                    for (int op=0;op<nops;op++) { cpo[op]=par[SNAP_OFF_CRITICAL_PARENT_OP+op]; ost[op]=par[SNAP_OFF_OP_START_TIME+op]; oft[op]=par[SNAP_OFF_OP_FINISH_TIME+op]; }
                    int nr2=par[SNAP_OFF_READY_COUNT]; for (int i=0;i<nr2;i++) pr[i]=par[SNAP_OFF_PRECEDENCE_READY+i];
                    int ns2=par[SNAP_OFF_SEQUENCE_COUNT]; for (int i=0;i<ns2;i++) sq[i]=par[SNAP_OFF_SEQUENCE+i];
                    int cop=cops[b]; int pos=cpu_find_in_array(pr,nr2,cop); if (pos>=0) pr[pos]=pr[--nr2];
                    cpu_traiter(cop,moc,mrt,cpo,ost,oft,pr,nr2,sq,ns2,nm,nops);
                    cpu_check_swap(moc,mrt,cpo,pr,nr2,sq,ns2,nm,nops,nm,ost);
                    int cmax=0; for (int m=1;m<=nm;m++) if (mrt[m]>cmax) cmax=mrt[m];
                    if (cmax>=cpu_best) continue;  /* prune against CPU's own best, not GPU's */
                    float sc=(ns2==0)?(float)cmax:((float)cmax/(float)ns2);
                    unsigned nh=*seed^(cand*0x85ebca6b)^(b*0xc2b2ae35); nh^=nh>>16; nh*=0x85ebca6b; nh^=nh>>16;
                    sc+=(float)(nh%1000)/1e9f;
                    int ci=cand*nj+b; int* out=children.data()+ci*SNAPSHOT_INTS;
                    for (int m=0;m<=nm;m++) { out[SNAP_OFF_MACHINE_OCCUPIED_OP+m]=moc[m]; out[SNAP_OFF_MACHINE_READY_TIME+m]=mrt[m]; }
                    for (int op=0;op<nops;op++) { out[SNAP_OFF_CRITICAL_PARENT_OP+op]=cpo[op]; out[SNAP_OFF_OP_START_TIME+op]=ost[op]; out[SNAP_OFF_OP_FINISH_TIME+op]=oft[op]; }
                    for (int i=0;i<nr2;i++) out[SNAP_OFF_PRECEDENCE_READY+i]=pr[i];
                    out[SNAP_OFF_READY_COUNT]=nr2;
                    for (int i=0;i<ns2;i++) out[SNAP_OFF_SEQUENCE+i]=sq[i];
                    out[SNAP_OFF_SEQUENCE_COUNT]=ns2; out[SNAP_OFF_MAKESPAN]=cmax;
                    scores[ci]=sc;
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
                /* Deadlock fallback */
                int moc[HARD_MAX_MACHINES]={},mrt[HARD_MAX_MACHINES]={};
                int cpo[HARD_MAX_OPS]={},ost[HARD_MAX_OPS]={},oft[HARD_MAX_OPS]={};
                int pr[HARD_MAX_OPS]={},sq[HARD_MAX_OPS]={};
                for (int m=0;m<=nm;m++) { moc[m]=par[SNAP_OFF_MACHINE_OCCUPIED_OP+m]; mrt[m]=par[SNAP_OFF_MACHINE_READY_TIME+m]; }
                for (int op=0;op<nops;op++) { cpo[op]=par[SNAP_OFF_CRITICAL_PARENT_OP+op]; ost[op]=par[SNAP_OFF_OP_START_TIME+op]; oft[op]=par[SNAP_OFF_OP_FINISH_TIME+op]; }
                int nr2=par[SNAP_OFF_READY_COUNT]; for (int i=0;i<nr2;i++) pr[i]=par[SNAP_OFF_PRECEDENCE_READY+i];
                int ns2=par[SNAP_OFF_SEQUENCE_COUNT]; for (int i=0;i<ns2;i++) sq[i]=par[SNAP_OFF_SEQUENCE+i];
                cpu_check_swap(moc,mrt,cpo,pr,nr2,sq,ns2,nm,nops,nm,ost);
                int cmax=0; for (int m=1;m<=nm;m++) if (mrt[m]>cmax) cmax=mrt[m];
                if (cmax>=cpu_best) continue;  /* prune against CPU's own best, not GPU's */
                float sc=(ns2==0)?(float)cmax:((float)cmax/(float)ns2);
                int ci=cand*nj; int* out=children.data()+ci*SNAPSHOT_INTS;
                for (int m=0;m<=nm;m++) { out[SNAP_OFF_MACHINE_OCCUPIED_OP+m]=moc[m]; out[SNAP_OFF_MACHINE_READY_TIME+m]=mrt[m]; }
                for (int op=0;op<nops;op++) { out[SNAP_OFF_CRITICAL_PARENT_OP+op]=cpo[op]; out[SNAP_OFF_OP_START_TIME+op]=ost[op]; out[SNAP_OFF_OP_FINISH_TIME+op]=oft[op]; }
                for (int i=0;i<nr2;i++) out[SNAP_OFF_PRECEDENCE_READY+i]=pr[i];
                out[SNAP_OFF_READY_COUNT]=nr2;
                for (int i=0;i<ns2;i++) out[SNAP_OFF_SEQUENCE+i]=sq[i];
                out[SNAP_OFF_SEQUENCE_COUNT]=ns2; out[SNAP_OFF_MAKESPAN]=cmax;
                scores[ci]=sc;
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
    return cpu_best;
}



/* =========================================================================
 *  MAIN  — concurrent CPU + GPU beam search
 * ========================================================================= */
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int num_gpus=0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus==0) { fprintf(stderr,"No GPU found.\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    CUDA_CHECK(cudaSetDevice(world_rank%num_gpus));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize,32768));

    if (world_rank == 0) {
        printf("\n%-8s %-12s %-6s %-9s %-8s %-10s %-10s %-10s %-10s\n",
               "Width", "Instance", "Jobs", "Machines", "Ops", "GPU_Best", "CPU_Best", "Global", "Time(s)");
        printf("------------------------------------------------------------------------------------------------------\n");
    }

    const char* instances_to_run[] = {"01","02","03", "04", "05","06","07", "08","09","10","11", "12","13","14", "15","16","17", "18","19","20","21","22","23", "24","25","26", "27","28","29", "30","31","32","33", "34","35","36", "37","38","39", "40"};

    if (world_rank == 0) {
        printf("Starting hybrid benchmarks...\n");
        fflush(stdout);
    }

    for (int j = 0; j < 40; j++) {
        instance_path[0] = '\0';
        strcat(instance_path, "../../lawrence_instances/la");
        strcat(instance_path, instances_to_run[j]);
        strcat(instance_path, ".txt");

        load_instance_gpu();


        if (j < 10 ){
            beam_width = 1000;
        } else {
            beam_width = 25600;
        }
        
        accel_factor = 34;
        if (argc > 3 && argv[3]) accel_factor = atoi(argv[3]);
        if (accel_factor < 1) accel_factor = 34;

        int gpu_bw = beam_width;
        int cpu_bw = max(1, beam_width / accel_factor);

        BUF_BEAM = beam_width;
        compute_snapshot_layout();
        allocate_gpu_buffers();

        int num_threads = 1;
        #pragma omp parallel
        {
            #pragma omp single
            num_threads = omp_get_num_threads();
        }

        clock_t wall_start = clock();

        g_shared_best.store(500000, memory_order_relaxed);
        int my_gpu_local=500000, my_cpu_local=500000;
        int gpu_global_best=500000, cpu_global_best=500000;

        srand((unsigned)time(NULL) + world_rank);

        for (int iter = 0; iter < 100; iter++) {
            int gm = 500000;
            int cm = 500000;

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                if (tid == 0) {
                    /* GPU beam search on thread 0 */
                    int local_gm = run_beam_construction();
                    if (local_gm < gm) gm = local_gm;
                } else {
                    /* CPU beam search on other threads */
                    unsigned int local_seed = (unsigned int)time(NULL) + world_rank * 1000 + iter * 100 + tid;
                    int local_cm = run_beam_construction_cpu(cpu_bw, &local_seed);
                    #pragma omp critical
                    {
                        if (local_cm < cm) cm = local_cm;
                    }
                }
            }

            if (gm < my_gpu_local) my_gpu_local = gm;
            if (cm < my_cpu_local) my_cpu_local = cm;

            MPI_Allreduce(&my_gpu_local, &gpu_global_best, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&my_cpu_local, &cpu_global_best, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            int global_best = min(gpu_global_best, cpu_global_best);

            int prev = g_shared_best.load(memory_order_relaxed);
            while (global_best < prev && !g_shared_best.compare_exchange_weak(prev, global_best, memory_order_release, memory_order_relaxed)) {}
        }

        int local_final = g_shared_best.load(memory_order_relaxed);
        int global_final;
        MPI_Allreduce(&local_final, &global_final, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        clock_t wall_end = clock();
        double elapsed = (double)(wall_end - wall_start) / CLOCKS_PER_SEC;

        if (world_rank == 0) {
            printf("%-8d %-12s %-6d %-9d %-8d %-10d %-10d %-10d %-10.2f\n",
                   beam_width,
                   instance_path,
                   num_jobs,
                   num_machines,
                   num_ops_total,
                   gpu_global_best,
                   cpu_global_best,
                   global_final,
                   elapsed);
            fflush(stdout);
        }

        free_gpu_buffers();
    }

    if (world_rank == 0) printf("\nAll batches finished.\n");
    MPI_Finalize();
    return 0;
}