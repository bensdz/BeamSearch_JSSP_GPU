// hybrid-beam.cu  — with per-rank throughput stats + per-rank lower-bound scoring
//
// Scoring modes (selected per rank, see LbMode below):
//   0  CUR     score = cmax / num_scheduled                    (original)
//   1  LB_M    score = max(cmax, mach_ready[m] + mach_rem[m])  (machine bound)
//   2  LB_MJ   score = max(cmax, mach_LB, job_ready[j] + job_rem[j])  (both)
//
// The LB is computed in O(M + J) per node from three counters that are
// maintained incrementally inside gpu_traiter / cpu_traiter:
//   mach_rem[m]   — remaining duration to run on machine m
//   job_rem[j]    — remaining duration of job j
//   job_ready[j]  — finish time of the last scheduled op of job j
// All three are initialised at the root, decremented/updated on each dispatch,
// and round-trip through the schedule snapshot so beam children inherit them.
// The per-node LB cost is one short fixed-size loop — effectively zero.
//
// MPI usage:
//   argv[5] = "auto"  (default)    → rank r uses mode (r % 3)
//   argv[5] = "0" | "1" | "2"      → all ranks use that mode

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <climits>
#include <cfloat>
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
#include <limits>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
using namespace std;

// =============================================================================
// SECTION 1 – Configuration
// =============================================================================
static char instance_path[256] = "./la10.txt";
static int  gpu_beam_width     = 1000;
static int  cpu_beam_width     = 500;
static int  global_best_cmax   = 500000;
static int  num_iterations     = 50;
static const int CALIB_K       = 50;
static const int MAX_CPU_KBEST = 8000;

// ---- Lower-bound scoring mode ---------------------------------------------
enum LbMode {
    LB_CUR = 0,  // cmax / num_scheduled                     (original, baseline)
    LB_M   = 1,  // max(cmax, mach_ready + mach_rem)         (machine bound)
    LB_MJ  = 2,  // LB_M + job_ready + job_rem               (machine + job)
    LB_HT  = 3,  // Carlier head-tail per machine            (NEW — dominates LB_M)
    LB_CP  = 4,  // Critical-path max_i (head_i + p_i + tail_i)   (NEW)
    LB_N_MODES = 5
};
static int  g_lb_mode_cli = -1;     // -1 = auto (rank % LB_N_MODES), else forced
static int  g_lb_mode     = 0;      // resolved per-rank at startup
static const char *lb_mode_name[] = {
    "CUR(cmax/sched)", "LB_M(mach)", "LB_MJ(mach+job)", "LB_HT(head-tail)", "LB_CP(crit-path)"
};

// =============================================================================
// SECTION 1b – Throughput Statistics
// =============================================================================
// One instance per rank; populated during the search loop.
struct ThroughputStats {
    long long partial_states;    // beam nodes expanded (partial schedules evaluated)
    long long complete_solutions;// full schedules (seq.size()==nbjob*nbmachine) evaluated
    long long beam_iterations;   // number of run_*_beam() calls completed
    double    elapsed_sec;       // wall time of the search loop only
};

static ThroughputStats rank_stats = {0, 0, 0, 0.0};

// =============================================================================
// SECTION 2 – BJSSP Instance
// =============================================================================
static int nbjob, nbmachine, nboperation;

struct CpuOperation {
    int num_op, num_mc, dure, td, tf, tdp, tfp, num_j, type;
    double ds; int ops;
};
static vector<CpuOperation> list_opr;
static vector<int>          opr_mach[20];

struct OperationInfo { int machine_id, duration, job_id; };

#define HARD_MAX_OPS      1000
#define HARD_MAX_MACHINES   50
#define HARD_MAX_JOBS       50
#define HARD_MAX_BEAM     50000

__constant__ OperationInfo c_ops[HARD_MAX_OPS];
__constant__ int c_op_order[HARD_MAX_MACHINES * HARD_MAX_JOBS];
__constant__ int c_op_count[HARD_MAX_MACHINES];
// per-machine and per-job total durations, cached as constants for fast init
__constant__ int c_mach_total[HARD_MAX_MACHINES];
__constant__ int c_job_total [HARD_MAX_JOBS];
// static tail = sum of durations of ops strictly AFTER this one in its job
// Used by LB_HT and LB_CP — doesn't change with scheduling.
__constant__ int c_tail_static[HARD_MAX_OPS];

static OperationInfo gpu_all_ops[HARD_MAX_OPS];
static int           gpu_machine_op_order[HARD_MAX_MACHINES * HARD_MAX_JOBS];
static int           gpu_machine_op_count[HARD_MAX_MACHINES];
static int           gpu_mach_total[HARD_MAX_MACHINES];
static int           gpu_job_total [HARD_MAX_JOBS];
static int           gpu_tail_static[HARD_MAX_OPS];

// =============================================================================
// SECTION 3 – GPU Snapshot Layout
// =============================================================================
static int BUF_OPS=0, BUF_MACHINES=0, BUF_JOBS=0, BUF_BEAM=0;
static int SNAP_OFF_MACHINE_OCCUPIED_OP=0, SNAP_OFF_MACHINE_READY_TIME=0;
static int SNAP_OFF_CRITICAL_PARENT_OP=0,  SNAP_OFF_OP_START_TIME=0;
static int SNAP_OFF_OP_FINISH_TIME=0,      SNAP_OFF_PRECEDENCE_READY=0;
static int SNAP_OFF_READY_COUNT=0,         SNAP_OFF_SEQUENCE=0;
static int SNAP_OFF_SEQUENCE_COUNT=0,      SNAP_OFF_MAKESPAN=0;
// New offsets for LB counters:
static int SNAP_OFF_MACH_REM=0,            SNAP_OFF_JOB_READY=0;
static int SNAP_OFF_JOB_REM=0;
static int SNAPSHOT_INTS=0;

static void compute_snapshot_layout() {
    SNAP_OFF_MACHINE_OCCUPIED_OP = 0;
    SNAP_OFF_MACHINE_READY_TIME  = BUF_MACHINES;
    SNAP_OFF_CRITICAL_PARENT_OP  = 2*BUF_MACHINES;
    SNAP_OFF_OP_START_TIME       = 2*BUF_MACHINES +   BUF_OPS;
    SNAP_OFF_OP_FINISH_TIME      = 2*BUF_MACHINES + 2*BUF_OPS;
    SNAP_OFF_PRECEDENCE_READY    = 2*BUF_MACHINES + 3*BUF_OPS;
    SNAP_OFF_READY_COUNT         = 2*BUF_MACHINES + 4*BUF_OPS;
    SNAP_OFF_SEQUENCE            = 2*BUF_MACHINES + 4*BUF_OPS + 1;
    SNAP_OFF_SEQUENCE_COUNT      = 2*BUF_MACHINES + 5*BUF_OPS + 1;
    SNAP_OFF_MAKESPAN            = 2*BUF_MACHINES + 5*BUF_OPS + 2;
    // New layout: one BUF_MACHINES block (mach_rem) + two BUF_JOBS blocks (job_ready, job_rem)
    SNAP_OFF_MACH_REM            = 2*BUF_MACHINES + 5*BUF_OPS + 3;
    SNAP_OFF_JOB_READY           = 3*BUF_MACHINES + 5*BUF_OPS + 3;
    SNAP_OFF_JOB_REM             = 3*BUF_MACHINES + 5*BUF_OPS + 3 +   BUF_JOBS;
    SNAPSHOT_INTS                = 3*BUF_MACHINES + 5*BUF_OPS + 3 + 2*BUF_JOBS;
}

struct ScheduleSnapshot {
    int machine_occupied_op[HARD_MAX_MACHINES];
    int machine_ready_time [HARD_MAX_MACHINES];
    int critical_parent_op [HARD_MAX_OPS];
    int op_start_time      [HARD_MAX_OPS];
    int op_finish_time     [HARD_MAX_OPS];
    int precedence_ready_ops[HARD_MAX_OPS];
    int num_precedence_ready;
    int seq[HARD_MAX_OPS];
    int num_scheduled;
    int makespan;
    // New:
    int mach_rem  [HARD_MAX_MACHINES];   // remaining duration on each machine
    int job_ready [HARD_MAX_JOBS];       // finish time of last scheduled op of each job (0-indexed)
    int job_rem   [HARD_MAX_JOBS];       // remaining duration of each job
};

static void pack_snapshot(const ScheduleSnapshot &s, int *flat) {
    memcpy(flat+SNAP_OFF_MACHINE_OCCUPIED_OP, s.machine_occupied_op, BUF_MACHINES*sizeof(int));
    memcpy(flat+SNAP_OFF_MACHINE_READY_TIME,  s.machine_ready_time,  BUF_MACHINES*sizeof(int));
    memcpy(flat+SNAP_OFF_CRITICAL_PARENT_OP,  s.critical_parent_op,  BUF_OPS*sizeof(int));
    memcpy(flat+SNAP_OFF_OP_START_TIME,       s.op_start_time,       BUF_OPS*sizeof(int));
    memcpy(flat+SNAP_OFF_OP_FINISH_TIME,      s.op_finish_time,      BUF_OPS*sizeof(int));
    memcpy(flat+SNAP_OFF_PRECEDENCE_READY,    s.precedence_ready_ops,BUF_OPS*sizeof(int));
    flat[SNAP_OFF_READY_COUNT]    = s.num_precedence_ready;
    memcpy(flat+SNAP_OFF_SEQUENCE,            s.seq,                 BUF_OPS*sizeof(int));
    flat[SNAP_OFF_SEQUENCE_COUNT] = s.num_scheduled;
    flat[SNAP_OFF_MAKESPAN]       = s.makespan;
    memcpy(flat+SNAP_OFF_MACH_REM,            s.mach_rem,            BUF_MACHINES*sizeof(int));
    memcpy(flat+SNAP_OFF_JOB_READY,           s.job_ready,           BUF_JOBS*sizeof(int));
    memcpy(flat+SNAP_OFF_JOB_REM,             s.job_rem,             BUF_JOBS*sizeof(int));
}

static int    *d_beam_snapshots=nullptr, *d_child_snapshots=nullptr;
static float  *d_child_scores=nullptr,   *d_child_scores_out=nullptr;
static int    *d_child_indices=nullptr,  *d_child_indices_out=nullptr;
static void   *d_cub_temp_storage=nullptr;
static size_t  cub_temp_bytes=0;
static int    *d_global_best_makespan=nullptr, *d_best_solution=nullptr;
static int    *h_input_flat=nullptr;
static int h_global_best_seq[HARD_MAX_OPS]={};

#define CUDA_CHECK(call) do { \
    cudaError_t _e=(call); \
    if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s:%d  %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(EXIT_FAILURE);} \
} while(0)

// =============================================================================
// SECTION 4 – Instance Loading
// =============================================================================
static void load_instance() {
    FILE *f = fopen(instance_path,"r");
    if(!f){fprintf(stderr,"Cannot open %s\n",instance_path);exit(1);}
    fscanf(f,"%d %d\n",&nbjob,&nbmachine);
    nboperation = nbjob*nbmachine+2;
    BUF_OPS=nboperation; BUF_MACHINES=nbmachine+1; BUF_JOBS=nbjob;
    CpuOperation dummy={};
    list_opr.push_back(dummy);
    gpu_all_ops[0]={0,0,0};
    int op_id=1;
    for(int job=1;job<=nbjob;job++)
        for(int step=1;step<=nbmachine;step++){
            int raw_mc,dur; fscanf(f,"%d %d",&raw_mc,&dur);
            int mc_1idx=raw_mc+1;
            CpuOperation op={}; op.num_op=op_id; op.num_mc=mc_1idx; op.dure=dur; op.num_j=job;
            list_opr.push_back(op); opr_mach[mc_1idx].push_back(op_id);
            gpu_all_ops[op_id]={mc_1idx,dur,job}; op_id++;
        }
    CpuOperation last={}; last.num_op=nbjob*nbmachine+1; list_opr.push_back(last);
    gpu_all_ops[nbjob*nbmachine+1]={0,0,0};
    fclose(f);
    memset(gpu_machine_op_order,0,sizeof(gpu_machine_op_order));
    memset(gpu_machine_op_count,0,sizeof(gpu_machine_op_count));
    for(int m=1;m<=nbmachine;m++){
        int n=(int)opr_mach[m].size(); gpu_machine_op_count[m]=n;
        for(int r=0;r<n;r++) gpu_machine_op_order[m*HARD_MAX_JOBS+r]=opr_mach[m][r];
    }
    // Instance-level totals for LB init
    memset(gpu_mach_total,0,sizeof(gpu_mach_total));
    memset(gpu_job_total, 0,sizeof(gpu_job_total));
    for(int op_id=1; op_id<=nbjob*nbmachine; op_id++){
        int m = list_opr[op_id].num_mc;
        int j = list_opr[op_id].num_j;        // 1-indexed
        int d = list_opr[op_id].dure;
        gpu_mach_total[m]  += d;
        gpu_job_total[j-1] += d;              // store 0-indexed in new LB arrays
    }
    // Static tail per op: tail[op] = sum of durations of ops AFTER op in its job
    memset(gpu_tail_static, 0, sizeof(gpu_tail_static));
    for (int j = 1; j <= nbjob; j++) {
        // ops of job j are (j-1)*M+1 .. j*M  (1-indexed)
        int last = j * nbmachine;
        gpu_tail_static[last] = 0;
        for (int k = nbmachine - 1; k >= 1; k--) {
            int op_id = (j-1)*nbmachine + k;
            gpu_tail_static[op_id] = gpu_tail_static[op_id + 1] + list_opr[op_id + 1].dure;
        }
    }
}

// =============================================================================
// SECTION 5 – CPU Scheduling Functions
// =============================================================================
static vector<int> cpu_dispo(31,0),cpu_tr(31,0),cpu_seq,cpu_c,cpu_pere(302,0),cpu_td(302,0),cpu_tf(302,0);
static vector<int> cpu_ob[31];
// LB counters (maintained incrementally by cpu_traiter / cpu_traiter_swap)
static vector<int> cpu_mach_rem(31,0), cpu_job_ready(51,0), cpu_job_rem(51,0);
static vector<int> &dispo=cpu_dispo,&tr=cpu_tr,&seq=cpu_seq,*ob=cpu_ob;
static vector<int> &c=cpu_c,&pere=cpu_pere,&td=cpu_td,&tf=cpu_tf;
static vector<int> &mach_rem=cpu_mach_rem,&job_ready=cpu_job_ready,&job_rem=cpu_job_rem;

static int p1pv(vector<int>&a,int v){
    if(a.empty())return -1;
    auto res=find(a.begin(),a.end(),v);
    int d=(int)distance(a.begin(),res);
    return(d!=(int)a.size())?d:-1;
}

static int cpu_traiter(int cii){
    vector<int> afaire1; int ci=cii;
    do {
        int mci=list_opr[ci].num_mc; seq.push_back(ci);
        if((ci%nbmachine)!=1){
            int opp=ci-1,mach_opp=list_opr[opp].num_mc,cx=0;
            if(tr[mci]<=tr[mach_opp]){cx=tr[mach_opp];pere[ci]=opp;}
            else{cx=tr[mci];int h=p1pv(opr_mach[mci],ci);int pp=-100;
                if(h>0){pp=opr_mach[mci][h-1];if(pp%nbmachine!=0)pp++;}pere[ci]=pp;}
            tr[mach_opp]=cx;tf[opp]=cx;td[ci]=cx;tr[mci]=cx+list_opr[ci].dure;
        } else {
            int h=p1pv(opr_mach[mci],ci);int pp=-100;
            if(h>0){pp=opr_mach[mci][h-1];if(pp%nbmachine!=0)pp++;pere[ci]=pp;}else pere[ci]=0;
            td[ci]=tr[mci];tr[mci]=tr[mci]+list_opr[ci].dure;
        }
        // ── LB counter update (O(1)) ─────────────────────────────────────────
        {
            int j1 = list_opr[ci].num_j;             // 1-indexed job
            int d  = list_opr[ci].dure;
            mach_rem[mci]    -= d;
            job_rem[j1-1]    -= d;
            job_ready[j1-1]  = tr[mci];              // finish time of ci
        }
        dispo[mci]=ci;
        int mo=ci%nbmachine;
        if(mo!=1){int m=list_opr[ci-1].num_mc;dispo[m]=0;if(!ob[m].empty()){afaire1.push_back(ob[m][0]);ob[m].erase(ob[m].begin());}}
        if((ci%nbmachine)!=0){c.push_back(ci+1);}
        else{tf[ci]=tr[mci];if(tf[ci]>td[nboperation-1])td[nboperation-1]=tf[ci];dispo[mci]=0;}
        ci=afaire1.empty()?-1:afaire1.back();if(ci!=-1)afaire1.pop_back();
    } while(ci!=-1);
    return 5;
}

static int cpu_traiter_swap(int cii,vector<int>&afaire){
    int ci=cii,mci=list_opr[ci].num_mc; seq.push_back(ci);
    tr[mci]+=list_opr[ci].dure; dispo[mci]=ci;
    // ── LB counter update (O(1)) ─────────────────────────────────────────────
    {
        int j1 = list_opr[ci].num_j;
        int d  = list_opr[ci].dure;
        mach_rem[mci]    -= d;
        job_rem[j1-1]    -= d;
        job_ready[j1-1]  = tr[mci];
    }
    if(ci%nbmachine!=0)c.push_back(ci+1);else dispo[mci]=0;
    return afaire.empty()?0:afaire[0];
}

static vector<int> cpu_solinit_swap_new(){
    vector<vector<int>> all_swap; vector<int> v;
    for(int i=0;i<(int)dispo.size();i++){
        int f=-1,op=dispo[i]; if(op==0)continue;
        int op1=op+1; if(op1>=nboperation)continue;
        if(dispo[i]!=0&&dispo[list_opr[op1].num_mc]>0)f=i;
        if(f<0)continue;
        vector<int> swp; swp.push_back(f); int rt=0;
        do {
            int suc=list_opr[dispo[f]+1].num_mc,opx=dispo[suc]+1;
            if(dispo[suc]!=0&&dispo[suc]<=nboperation&&dispo[list_opr[opx].num_mc]>0){
                rt=p1pv(swp,suc);
                if(rt==-1){swp.push_back(suc);f=suc;}
                else{for(int k=0;k<rt;k++)swp.erase(swp.begin());}
            }else{swp.clear();rt=0;}
        }while(rt==-1);
        if(swp.size()>=2)all_swap.push_back(swp);
    }
    int best_sz=0,best_idx=0;
    for(int i=0;i<(int)all_swap.size();i++)if((int)all_swap[i].size()>best_sz){best_sz=(int)all_swap[i].size();best_idx=i;}
    return(best_sz>1)?all_swap[best_idx]:v;
}

static void cpu_check_swap(){
    vector<int> swapp=cpu_solinit_swap_new(); if(swapp.size()<=1)return;
    int swt=0,poo=0;
    for(int j=0;j<(int)swapp.size();j++){int m=swapp[j];if(m!=0&&tr[m]>swt){swt=tr[m];poo=dispo[m];}}
    poo++;
    for(int j=0;j<(int)swapp.size();j++)if(swapp[j]!=0)tr[swapp[j]]=swt;
    for(int j=0;j<(int)swapp.size();j++){int mm=swapp[j];if(mm!=0)swapp[j]=dispo[mm]+1;}
    for(int j=0;j<(int)swapp.size();j++){int op=swapp[j],mpo=list_opr[op].num_mc;if(op!=0)pere[op]=(op==poo)?poo-1:dispo[mpo]+1;}
    vector<int> afaire;
    for(int j=0;j<(int)swapp.size();j++){int cci=swapp[j];if(cci==0)continue;auto it=find(c.begin(),c.end(),cci);if(it!=c.end())c.erase(it);cpu_traiter_swap(cci,afaire);swapp[j]=0;}
}

// =============================================================================
// SECTION 6 – CPU Beam Data Structures
// =============================================================================
struct JobShopState {
    vector<int> dispo,tr,seq,c,pere,td,tf; vector<int> ob[31];
    vector<int> s_mach_rem, s_job_ready, s_job_rem;   // LB counters (saved)
    int cmax; double score; int64_t s_hash;
    void save_state(const vector<int>&d,const vector<int>&t,const vector<int>&s,vector<int> o[31],const vector<int>&cv,const vector<int>&p,const vector<int>&tds,const vector<int>&tfs,int64_t h){
        s_hash=h;dispo=d;tr=t;seq=s;c=cv;pere=p;td=tds;tf=tfs;
        s_mach_rem  = cpu_mach_rem;
        s_job_ready = cpu_job_ready;
        s_job_rem   = cpu_job_rem;
    }
    void load_state(vector<int>&d,vector<int>&t,vector<int>&s,vector<int> o[31],vector<int>&cv,vector<int>&p,vector<int>&tds,vector<int>&tfs)const{
        d=dispo;t=tr;s=seq;cv=c;p=pere;tds=td;tfs=tf;
        cpu_mach_rem  = s_mach_rem;
        cpu_job_ready = s_job_ready;
        cpu_job_rem   = s_job_rem;
    }
};
struct CompareTask{bool operator()(const JobShopState&a,const JobShopState&b)const{return a.score<b.score;}};
static vector<JobShopState> candidates;
static void in_element(vector<JobShopState>&heap,JobShopState elem,int k){
    if((int)heap.size()<k){heap.push_back(elem);push_heap(heap.begin(),heap.end(),CompareTask{});}
    else if(elem.score<heap.front().score){pop_heap(heap.begin(),heap.end(),CompareTask{});heap.back()=elem;push_heap(heap.begin(),heap.end(),CompareTask{});}
}

// CPU scoring — O(M + J) or O(total unscheduled ops) per call
static inline double cpu_compute_score(int cmax) {
    if (g_lb_mode == LB_CUR) {
        return ((double)cmax / max(1,(int)seq.size()));
    }
    // LB_M: machine bound (cheap, always computed)
    int mach_LB = cmax;
    for (int m = 1; m <= nbmachine; m++) {
        int v = cpu_tr[m] + cpu_mach_rem[m];
        if (v > mach_LB) mach_LB = v;
    }
    if (g_lb_mode == LB_M) return (double)mach_LB;

    if (g_lb_mode == LB_MJ) {
        int lb = mach_LB;
        for (int j = 0; j < nbjob; j++) {
            int v = cpu_job_ready[j] + cpu_job_rem[j];
            if (v > lb) lb = v;
        }
        return (double)lb;
    }

    // LB_HT and LB_CP: walk unscheduled ops via precedence_ready list + job chain
    // For each job j with next unscheduled op `op`, iterate ops op, op+1, ..., end-of-job.
    if (g_lb_mode == LB_HT || g_lb_mode == LB_CP) {
        int m_minhead[HARD_MAX_MACHINES+2], m_mintail[HARD_MAX_MACHINES+2];
        for (int m = 0; m <= nbmachine + 1; m++) {
            m_minhead[m] = INT_MAX;
            m_mintail[m] = INT_MAX;
        }
        int lb_cp = 0;
        // Identify the "next unscheduled op" for each active job from cpu_c
        for (size_t i = 0; i < cpu_c.size(); i++) {
            int op = cpu_c[i];
            if (op < 1 || op >= nboperation) continue;
            int j1 = list_opr[op].num_j;  // 1-indexed
            // walk from op through end of job
            int last_in_job = j1 * nbmachine;
            for (int op_id = op; op_id <= last_in_job; op_id++) {
                int m = list_opr[op_id].num_mc;
                int d = list_opr[op_id].dure;
                int tail = gpu_tail_static[op_id];
                // head: lower bound = max(job_ready[j], mach_ready[m])
                int head = cpu_job_ready[j1-1];
                if (cpu_tr[m] > head) head = cpu_tr[m];
                if (head < m_minhead[m]) m_minhead[m] = head;
                if (tail < m_mintail[m]) m_mintail[m] = tail;
                if (g_lb_mode == LB_CP) {
                    int v = head + d + tail;
                    if (v > lb_cp) lb_cp = v;
                }
            }
        }
        if (g_lb_mode == LB_HT) {
            int lb_ht = 0;
            for (int m = 1; m <= nbmachine; m++) {
                if (cpu_mach_rem[m] > 0 && m_minhead[m] != INT_MAX) {
                    int v = m_minhead[m] + cpu_mach_rem[m] + m_mintail[m];
                    if (v > lb_ht) lb_ht = v;
                }
            }
            int lb = mach_LB;                   // LB_HT ≥ LB_M, but keep max just in case
            if (lb_ht > lb) lb = lb_ht;
            return (double)lb;
        } else { // LB_CP
            int lb = mach_LB;                   // combine with LB_M for robustness
            if (lb_cp > lb) lb = lb_cp;
            return (double)lb;
        }
    }
    return (double)mach_LB;  // fallback
}

// =============================================================================
// SECTION 7 – CPU Beam Search  ← STATS ADDED HERE
// =============================================================================
static int run_cpu_beam(int /*iteration*/) {
    vector<JobShopState> top_elements;
    top_elements.reserve(cpu_beam_width);
    seq.clear(); c.clear();
    fill(dispo.begin(),dispo.end(),0); fill(tr.begin(),tr.end(),0);
    fill(td.begin(),td.end(),0);       fill(tf.begin(),tf.end(),0);
    for(int i=0;i<31;i++)ob[i].clear();
    for(int j=0;j<nbjob;j++)c.push_back(j*nbmachine+1);
    // ── init LB counters at the root ─────────────────────────────────────────
    if ((int)cpu_mach_rem.size()  < nbmachine+2) cpu_mach_rem.resize(nbmachine+2, 0);
    if ((int)cpu_job_rem.size()   < nbjob+2)     cpu_job_rem.resize(nbjob+2, 0);
    if ((int)cpu_job_ready.size() < nbjob+2)     cpu_job_ready.resize(nbjob+2, 0);
    fill(cpu_mach_rem.begin(),  cpu_mach_rem.end(),  0);
    fill(cpu_job_rem.begin(),   cpu_job_rem.end(),   0);
    fill(cpu_job_ready.begin(), cpu_job_ready.end(), 0);
    for (int m = 1; m <= nbmachine; m++) cpu_mach_rem[m] = gpu_mach_total[m];
    for (int j = 0; j < nbjob; j++)      cpu_job_rem[j]  = gpu_job_total[j];
    candidates.clear();
    JobShopState root; root.save_state(dispo,tr,seq,ob,c,pere,td,tf,0);
    root.cmax=0; root.score=0.0; candidates.push_back(root);
    int best=10000000,newop=1;

    while(newop<nbjob*nbmachine+10&&!candidates.empty()){
        newop++;
        for(int id_c=0;id_c<(int)candidates.size();id_c++){
            candidates[id_c].load_state(dispo,tr,seq,ob,c,pere,td,tf);
            vector<int> ready_opp;
            for(int i=0;i<(int)c.size();i++) if(dispo[list_opr[c[i]].num_mc]==0) ready_opp.push_back(c[i]);
            int nb=(int)ready_opp.size();
            if(nb==0){
                candidates[id_c].load_state(dispo,tr,seq,ob,c,pere,td,tf);
                cpu_check_swap();
                auto result=max_element(tr.begin(),tr.end()); int a=(int)distance(tr.begin(),result);
                int64_t h=0; JobShopState tmp;
                tmp.save_state(dispo,tr,seq,ob,c,pere,td,tf,h);
                tmp.cmax=tr[a]; tmp.score=cpu_compute_score(tr[a])+(double)(rand()%1000)/1e6;
                // ── STAT: count this as a partial state evaluated ──────────────
                rank_stats.partial_states++;
                if((int)seq.size()==nbjob*nbmachine) rank_stats.complete_solutions++;
                if(tmp.cmax<global_best_cmax) in_element(top_elements,tmp,cpu_beam_width);
                continue;
            }
            int ii0=rand()%nb, mtb=list_opr[ready_opp[ii0]].num_mc;
            for(int it=0;it<nb;it++){
                if(list_opr[ready_opp[it]].num_mc!=mtb) continue;
                candidates[id_c].load_state(dispo,tr,seq,ob,c,pere,td,tf);
                int ci=ready_opp[it];
                auto itt=find(c.begin(),c.end(),ci); if(itt!=c.end())c.erase(itt);
                if(dispo[list_opr[ci].num_mc]==0) cpu_traiter(ci);
                cpu_check_swap();
                auto result=max_element(tr.begin(),tr.end()); int a=(int)distance(tr.begin(),result);
                int64_t h=0; JobShopState tmp;
                tmp.save_state(dispo,tr,seq,ob,c,pere,td,tf,h);
                tmp.cmax=tr[a]; tmp.score=cpu_compute_score(tr[a])+(double)(rand()%1000)/1e6;
                // ── STAT: every child state evaluated ─────────────────────────
                rank_stats.partial_states++;
                if((int)seq.size()==nbjob*nbmachine){
                    rank_stats.complete_solutions++;
                    if(tr[a]<best) best=tr[a];
                }
                if(tmp.cmax<global_best_cmax) in_element(top_elements,tmp,cpu_beam_width);
            }
        }
        candidates.clear(); candidates=top_elements; top_elements.clear();
    }
    rank_stats.beam_iterations++;
    return best;
}

// =============================================================================
// SECTION 8 – GPU Kernels (unchanged from original)
// =============================================================================
__device__ static int dev_find_in_array(const int*arr,int len,int val){for(int i=0;i<len;i++)if(arr[i]==val)return i;return -1;}
__device__ static int find_op_rank_on_machine(int op_id,int machine,int stride){int n=c_op_count[machine];for(int r=0;r<n;r++)if(c_op_order[machine*stride+r]==op_id)return r;return -1;}

__device__ static void gpu_traiter(int op_id,int*machine_occupied_op,int*machine_ready_time,int*critical_parent_op,int*op_start_time,int*op_finish_time,int*precedence_ready_ops,int&num_ready,int*seq,int&num_scheduled,int num_machines_1idx,int num_ops_tot,int machine_order_stride,
                                   int*mach_rem,int*job_ready,int*job_rem){
    int cascade_stack[HARD_MAX_OPS],cascade_top=0,current=op_id;
    do {
        int machine=c_ops[current].machine_id; seq[num_scheduled++]=current;
        int start; bool is_first=(current%num_machines_1idx==1);
        if(!is_first){int pred_op=current-1,pred_mc=c_ops[pred_op].machine_id;
            if(machine_ready_time[machine]<=machine_ready_time[pred_mc]){start=machine_ready_time[pred_mc];critical_parent_op[current]=pred_op;}
            else{start=machine_ready_time[machine];int rank=find_op_rank_on_machine(current,machine,machine_order_stride);int prev=(rank>0)?c_op_order[machine*machine_order_stride+rank-1]:-100;if(prev!=-100&&prev%num_machines_1idx!=0)prev++;critical_parent_op[current]=prev;}
        } else {int rank=find_op_rank_on_machine(current,machine,machine_order_stride);if(rank>0){int prev=c_op_order[machine*machine_order_stride+rank-1];if(prev%num_machines_1idx!=0)prev++;critical_parent_op[current]=prev;}else critical_parent_op[current]=0;start=machine_ready_time[machine];}
        op_start_time[current]=start; machine_ready_time[machine]=start+c_ops[current].duration; machine_occupied_op[machine]=current;
        // ── LB counter update (O(1)) ─────────────────────────────────────────
        {
            int jid = c_ops[current].job_id;          // 1-indexed
            int d   = c_ops[current].duration;
            mach_rem[machine] -= d;
            job_rem [jid-1]   -= d;
            job_ready[jid-1]  = machine_ready_time[machine];  // finish time
        }
        if(!is_first){int pm=c_ops[current-1].machine_id;machine_occupied_op[pm]=0;machine_ready_time[pm]=start;}
        bool is_last=(current%num_machines_1idx==0);
        if(!is_last){precedence_ready_ops[num_ready++]=current+1;}
        else{op_finish_time[current]=machine_ready_time[machine];if(op_finish_time[current]>op_start_time[num_ops_tot-1])op_start_time[num_ops_tot-1]=op_finish_time[current];machine_occupied_op[machine]=0;}
        current=(cascade_top>0)?cascade_stack[--cascade_top]:-1;
    }while(current!=-1);
}

__device__ static void gpu_traiter_swap(int op_id,int*machine_occupied_op,int*machine_ready_time,int*precedence_ready_ops,int&num_ready,int*seq,int&num_scheduled,int num_machines_1idx,int*op_start_time,
                                        int*mach_rem,int*job_ready,int*job_rem){
    int machine=c_ops[op_id].machine_id; seq[num_scheduled++]=op_id;
    int ss=machine_ready_time[machine]; op_start_time[op_id]=ss;
    machine_ready_time[machine]=ss+c_ops[op_id].duration; machine_occupied_op[machine]=op_id;
    // ── LB counter update (O(1)) ─────────────────────────────────────────────
    {
        int jid = c_ops[op_id].job_id;
        int d   = c_ops[op_id].duration;
        mach_rem[machine] -= d;
        job_rem [jid-1]   -= d;
        job_ready[jid-1]  = machine_ready_time[machine];
    }
    if(op_id%num_machines_1idx!=0)precedence_ready_ops[num_ready++]=op_id+1;else machine_occupied_op[machine]=0;
}

__device__ static int gpu_solinit_swap_new(const int*machine_occupied_op,int*out_cycle,int num_machines_1idx,int num_ops_tot,int num_machines_actual){
    int best_cycle_len=0;
    for(int sm=1;sm<=num_machines_actual;sm++){
        int op_on_start=machine_occupied_op[sm]; if(op_on_start==0)continue;
        int next_op=op_on_start+1; if(next_op<=0||next_op>=num_ops_tot)continue;
        int next_mc=c_ops[next_op].machine_id; if(machine_occupied_op[next_mc]==0)continue;
        int chain[HARD_MAX_MACHINES],chain_len=0; chain[chain_len++]=sm;
        int cur_mc=sm,cycle_start=-1;
        do {
            int op_cur=machine_occupied_op[cur_mc],nxt=op_cur+1;
            if(nxt<=0||nxt>=num_ops_tot){chain_len=0;break;}
            int needed_mc=c_ops[nxt].machine_id,op_needed=machine_occupied_op[needed_mc],nxt2=op_needed+1;
            bool still_blk=(op_needed!=0)&&(op_needed<num_ops_tot)&&(nxt2>0)&&(nxt2<num_ops_tot)&&(machine_occupied_op[c_ops[nxt2].machine_id]>0);
            if(!still_blk){chain_len=0;break;}
            cycle_start=dev_find_in_array(chain,chain_len,needed_mc);
            if(cycle_start==-1){if(chain_len<HARD_MAX_MACHINES)chain[chain_len++]=needed_mc;cur_mc=needed_mc;}
        }while(cycle_start==-1&&chain_len<HARD_MAX_MACHINES);
        if(cycle_start>=0){int len=chain_len-cycle_start;if(len>=2&&len>best_cycle_len){best_cycle_len=len;for(int i=0;i<len;i++)out_cycle[i]=chain[cycle_start+i];}}
    }
    return best_cycle_len;
}

__device__ static void gpu_check_swap(int*machine_occupied_op,int*machine_ready_time,int*critical_parent_op,int*precedence_ready_ops,int&num_ready,int*seq,int&num_scheduled,int num_machines_1idx,int num_ops_tot,int num_machines_actual,int*op_start_time,
                                      int*mach_rem,int*job_ready,int*job_rem){
    int cycle[HARD_MAX_MACHINES];
    int cycle_len=gpu_solinit_swap_new(machine_occupied_op,cycle,num_machines_1idx,num_ops_tot,num_machines_actual);
    if(cycle_len<=1)return;
    int sync_time=0,pivot=0;
    for(int i=0;i<cycle_len;i++){int mc=cycle[i];if(mc!=0&&machine_ready_time[mc]>sync_time){sync_time=machine_ready_time[mc];pivot=machine_occupied_op[mc];}}
    pivot++;
    for(int i=0;i<cycle_len;i++)if(cycle[i]!=0)machine_ready_time[cycle[i]]=sync_time;
    int blocked[HARD_MAX_MACHINES];
    for(int i=0;i<cycle_len;i++)blocked[i]=(cycle[i]!=0)?machine_occupied_op[cycle[i]]+1:0;
    for(int i=0;i<cycle_len;i++){int op=blocked[i];if(op==0)continue;int mc=c_ops[op].machine_id;critical_parent_op[op]=(op==pivot)?pivot-1:machine_occupied_op[mc]+1;}
    for(int i=0;i<cycle_len;i++){int op=blocked[i];if(op==0)continue;int pos=dev_find_in_array(precedence_ready_ops,num_ready,op);if(pos>=0)precedence_ready_ops[pos]=precedence_ready_ops[--num_ready];gpu_traiter_swap(op,machine_occupied_op,machine_ready_time,precedence_ready_ops,num_ready,seq,num_scheduled,num_machines_1idx,op_start_time,mach_rem,job_ready,job_rem);}
}

// =============================================================================
// Device-side LB dispatcher — computes the score base from a partial state.
// Caller supplies: cmax, counters, and mode. Returns the base value (LB).
// =============================================================================
__device__ static int gpu_compute_lb(
    int cmax, int num_scheduled,
    const int *machine_ready_time, const int *mach_rem,
    const int *job_ready, const int *job_rem,
    const int *precedence_ready_ops, int num_ready,
    int num_machines_1idx, int num_jobs_actual, int num_ops_tot,
    int lb_mode)
{
    if (lb_mode == 0) {
        // LB_CUR: (cmax / num_scheduled) scaled to int*1000 for float storage later
        // We return cmax here and let the caller use the classic form
        // — handled at call site. Return cmax as a safe fallback.
        return cmax;
    }
    // LB_M: machine bound
    int lb = cmax;
    for (int m = 1; m <= num_machines_1idx; m++) {
        int v = machine_ready_time[m] + mach_rem[m];
        if (v > lb) lb = v;
    }
    if (lb_mode == 1) return lb;
    // LB_MJ: add job bound
    if (lb_mode == 2) {
        for (int j = 0; j < num_jobs_actual; j++) {
            int v = job_ready[j] + job_rem[j];
            if (v > lb) lb = v;
        }
        return lb;
    }
    // LB_HT and LB_CP: walk unscheduled ops via precedence_ready + job chain
    int m_minhead[HARD_MAX_MACHINES+2];
    int m_mintail[HARD_MAX_MACHINES+2];
    #pragma unroll
    for (int m = 0; m < HARD_MAX_MACHINES+2; m++) {
        m_minhead[m] = INT_MAX;
        m_mintail[m] = INT_MAX;
    }
    int lb_cp = 0;
    for (int i = 0; i < num_ready; i++) {
        int op = precedence_ready_ops[i];
        if (op < 1 || op >= num_ops_tot) continue;
        int j1 = c_ops[op].job_id;  // 1-indexed
        // ops of this job still to schedule: op, op+1, ..., j1*M  (1-indexed)
        int last_in_job = j1 * num_machines_1idx;
        for (int op_id = op; op_id <= last_in_job; op_id++) {
            int m = c_ops[op_id].machine_id;
            int d = c_ops[op_id].duration;
            int t = c_tail_static[op_id];
            int head = job_ready[j1-1];
            if (machine_ready_time[m] > head) head = machine_ready_time[m];
            if (head < m_minhead[m]) m_minhead[m] = head;
            if (t    < m_mintail[m]) m_mintail[m] = t;
            if (lb_mode == 4) {
                int v = head + d + t;
                if (v > lb_cp) lb_cp = v;
            }
        }
    }
    if (lb_mode == 3) { // LB_HT
        for (int m = 1; m <= num_machines_1idx; m++) {
            if (mach_rem[m] > 0 && m_minhead[m] != INT_MAX) {
                int v = m_minhead[m] + mach_rem[m] + m_mintail[m];
                if (v > lb) lb = v;
            }
        }
        return lb;
    }
    // LB_CP
    if (lb_cp > lb) lb = lb_cp;
    return lb;
}

__global__ void kernel_expand_beam_layer(
    const int*d_in_beam,int*d_out_snapshots,float*d_out_scores,int*d_global_best,int*d_best_sol,
    int num_candidates,int num_branches_per_candidate,int SNAPSHOT_INTS_val,
    int off_machine_occupied,int off_machine_ready,int off_critical_parent,
    int off_op_start,int off_op_finish,int off_precedence_ready,int off_ready_count,
    int off_sequence,int off_sequence_count,int off_makespan,
    int off_mach_rem, int off_job_ready, int off_job_rem,
    int num_machines_1idx,int num_ops_tot,int total_real_ops,int machine_order_stride,int random_seed,
    int lb_mode, int num_jobs_actual)
{
    extern __shared__ int s_data[];
    int*s_parent_snap=s_data,*s_candidate_ops=s_parent_snap+SNAPSHOT_INTS_val,*s_num_ptr=s_candidate_ops+HARD_MAX_JOBS;
    int candidate_id=blockIdx.x,branch_id=threadIdx.x;
    if(candidate_id>=num_candidates)return;
    int child_flat_index=candidate_id*num_branches_per_candidate+branch_id;
    int out_base=child_flat_index*SNAPSHOT_INTS_val;
    d_out_scores[child_flat_index]=1e9f; d_out_snapshots[out_base+off_makespan]=INT_MAX;
    for(int i=branch_id;i<SNAPSHOT_INTS_val;i+=blockDim.x) s_parent_snap[i]=d_in_beam[candidate_id*SNAPSHOT_INTS_val+i];
    __syncthreads();
    if(branch_id==0){
        int free_buf[HARD_MAX_JOBS],nf=0,nr=s_parent_snap[off_ready_count];
        for(int i=0;i<nr;i++){int op=s_parent_snap[off_precedence_ready+i];if(op>=1&&op<num_ops_tot&&s_parent_snap[off_machine_occupied+c_ops[op].machine_id]==0)free_buf[nf++]=op;}
        if(nf>0){unsigned int hh=(unsigned int)random_seed+(unsigned int)candidate_id;hh^=hh>>16;hh*=0x7feb352d;hh^=hh>>15;hh*=0x846ca68b;hh^=hh>>16;int chosen_mc=c_ops[free_buf[hh%nf]].machine_id;*s_num_ptr=0;for(int i=0;i<nr;i++){int op=s_parent_snap[off_precedence_ready+i];if(op>=1&&op<num_ops_tot&&c_ops[op].machine_id==chosen_mc&&s_parent_snap[off_machine_occupied+chosen_mc]==0)s_candidate_ops[(*s_num_ptr)++]=op;}}
        else *s_num_ptr=0;
    }
    __syncthreads();
    int num_valid_ops=*s_num_ptr;
    if(num_valid_ops==0){
        if(branch_id!=0)return;
        int machine_occupied_op[HARD_MAX_MACHINES]={},machine_ready_time[HARD_MAX_MACHINES]={},critical_parent_op[HARD_MAX_OPS]={},op_start_time[HARD_MAX_OPS]={},op_finish_time[HARD_MAX_OPS]={},precedence_ready_ops[HARD_MAX_OPS]={},seq[HARD_MAX_OPS]={};
        int mach_rem[HARD_MAX_MACHINES]={}, job_ready[HARD_MAX_JOBS]={}, job_rem[HARD_MAX_JOBS]={};
        for(int m=0;m<=num_machines_1idx;m++){machine_occupied_op[m]=s_parent_snap[off_machine_occupied+m];machine_ready_time[m]=s_parent_snap[off_machine_ready+m];mach_rem[m]=s_parent_snap[off_mach_rem+m];}
        for(int j=0;j<num_jobs_actual;j++){job_ready[j]=s_parent_snap[off_job_ready+j];job_rem[j]=s_parent_snap[off_job_rem+j];}
        for(int op=0;op<num_ops_tot;op++){critical_parent_op[op]=s_parent_snap[off_critical_parent+op];op_start_time[op]=s_parent_snap[off_op_start+op];op_finish_time[op]=s_parent_snap[off_op_finish+op];}
        int num_ready=s_parent_snap[off_ready_count];for(int i=0;i<num_ready;i++)precedence_ready_ops[i]=s_parent_snap[off_precedence_ready+i];
        int num_scheduled=s_parent_snap[off_sequence_count];for(int i=0;i<num_scheduled;i++)seq[i]=s_parent_snap[off_sequence+i];
        gpu_check_swap(machine_occupied_op,machine_ready_time,critical_parent_op,precedence_ready_ops,num_ready,seq,num_scheduled,num_machines_1idx,num_ops_tot,num_machines_1idx,op_start_time,mach_rem,job_ready,job_rem);
        int cmax=0;for(int m=1;m<=num_machines_1idx;m++)if(machine_ready_time[m]>cmax)cmax=machine_ready_time[m];
        if(cmax>=*d_global_best)return;
        // ── LB scoring: LB_CUR keeps the old form, others use gpu_compute_lb ──
        float base;
        if (lb_mode == 0) {
            base = (num_scheduled==0)?(float)cmax:((float)cmax/num_scheduled);
        } else {
            int lb = gpu_compute_lb(cmax, num_scheduled,
                                    machine_ready_time, mach_rem, job_ready, job_rem,
                                    precedence_ready_ops, num_ready,
                                    num_machines_1idx, num_jobs_actual, num_ops_tot,
                                    lb_mode);
            base = (float)lb;
        }
        unsigned int nh=(unsigned int)random_seed^((unsigned int)candidate_id*0x85ebca6b);nh^=nh>>16;nh*=0x85ebca6b;nh^=nh>>16;nh*=0xc2b2ae35;nh^=nh>>16;
        d_out_scores[candidate_id*num_branches_per_candidate]=base+(float)(nh%1000)/1e9f;
        int*out=d_out_snapshots+candidate_id*num_branches_per_candidate*SNAPSHOT_INTS_val;
        for(int m=0;m<=num_machines_1idx;m++){out[off_machine_occupied+m]=machine_occupied_op[m];out[off_machine_ready+m]=machine_ready_time[m];out[off_mach_rem+m]=mach_rem[m];}
        for(int j=0;j<num_jobs_actual;j++){out[off_job_ready+j]=job_ready[j];out[off_job_rem+j]=job_rem[j];}
        for(int op=0;op<num_ops_tot;op++){out[off_critical_parent+op]=critical_parent_op[op];out[off_op_start+op]=op_start_time[op];out[off_op_finish+op]=op_finish_time[op];}
        for(int i=0;i<num_ready;i++)out[off_precedence_ready+i]=precedence_ready_ops[i];
        out[off_ready_count]=num_ready;for(int i=0;i<num_scheduled;i++)out[off_sequence+i]=seq[i];out[off_sequence_count]=num_scheduled;out[off_makespan]=cmax;
        if(num_scheduled==total_real_ops){int old_best=atomicMin(d_global_best,cmax);if(cmax<old_best)for(int i=0;i<SNAPSHOT_INTS_val;i++)d_best_sol[i]=out[i];}
        return;
    }
    if(branch_id>=num_valid_ops)return;
    int machine_occupied_op[HARD_MAX_MACHINES]={},machine_ready_time[HARD_MAX_MACHINES]={},critical_parent_op[HARD_MAX_OPS]={},op_start_time[HARD_MAX_OPS]={},op_finish_time[HARD_MAX_OPS]={},precedence_ready_ops[HARD_MAX_OPS]={},seq[HARD_MAX_OPS]={};
    int mach_rem[HARD_MAX_MACHINES]={}, job_ready[HARD_MAX_JOBS]={}, job_rem[HARD_MAX_JOBS]={};
    for(int m=0;m<=num_machines_1idx;m++){machine_occupied_op[m]=s_parent_snap[off_machine_occupied+m];machine_ready_time[m]=s_parent_snap[off_machine_ready+m];mach_rem[m]=s_parent_snap[off_mach_rem+m];}
    for(int j=0;j<num_jobs_actual;j++){job_ready[j]=s_parent_snap[off_job_ready+j];job_rem[j]=s_parent_snap[off_job_rem+j];}
    for(int op=0;op<num_ops_tot;op++){critical_parent_op[op]=s_parent_snap[off_critical_parent+op];op_start_time[op]=s_parent_snap[off_op_start+op];op_finish_time[op]=s_parent_snap[off_op_finish+op];}
    int num_ready=s_parent_snap[off_ready_count];for(int i=0;i<num_ready;i++)precedence_ready_ops[i]=s_parent_snap[off_precedence_ready+i];
    int num_scheduled=s_parent_snap[off_sequence_count];for(int i=0;i<num_scheduled;i++)seq[i]=s_parent_snap[off_sequence+i];
    int chosen_op=s_candidate_ops[branch_id];int rpos=dev_find_in_array(precedence_ready_ops,num_ready,chosen_op);if(rpos>=0)precedence_ready_ops[rpos]=precedence_ready_ops[--num_ready];
    gpu_traiter(chosen_op,machine_occupied_op,machine_ready_time,critical_parent_op,op_start_time,op_finish_time,precedence_ready_ops,num_ready,seq,num_scheduled,num_machines_1idx,num_ops_tot,machine_order_stride,mach_rem,job_ready,job_rem);
    gpu_check_swap(machine_occupied_op,machine_ready_time,critical_parent_op,precedence_ready_ops,num_ready,seq,num_scheduled,num_machines_1idx,num_ops_tot,num_machines_1idx,op_start_time,mach_rem,job_ready,job_rem);
    int cmax=0;for(int m=1;m<=num_machines_1idx;m++)if(machine_ready_time[m]>cmax)cmax=machine_ready_time[m];
    if(cmax>=*d_global_best)return;
    // ── LB scoring: LB_CUR keeps the old form, others use gpu_compute_lb ────
    float base;
    if (lb_mode == 0) {
        base = (num_scheduled==0)?(float)cmax:((float)cmax/num_scheduled);
    } else {
        int lb = gpu_compute_lb(cmax, num_scheduled,
                                machine_ready_time, mach_rem, job_ready, job_rem,
                                precedence_ready_ops, num_ready,
                                num_machines_1idx, num_jobs_actual, num_ops_tot,
                                lb_mode);
        base = (float)lb;
    }
    unsigned int nh=(unsigned int)random_seed^((unsigned int)candidate_id*0x85ebca6b)^((unsigned int)branch_id*0xc2b2ae35);nh^=nh>>16;nh*=0x85ebca6b;nh^=nh>>16;nh*=0xc2b2ae35;nh^=nh>>16;
    d_out_scores[child_flat_index]=base+(float)(nh%1000)/1e9f;
    int*out=d_out_snapshots+out_base;
    for(int m=0;m<=num_machines_1idx;m++){out[off_machine_occupied+m]=machine_occupied_op[m];out[off_machine_ready+m]=machine_ready_time[m];out[off_mach_rem+m]=mach_rem[m];}
    for(int j=0;j<num_jobs_actual;j++){out[off_job_ready+j]=job_ready[j];out[off_job_rem+j]=job_rem[j];}
    for(int op=0;op<num_ops_tot;op++){out[off_critical_parent+op]=critical_parent_op[op];out[off_op_start+op]=op_start_time[op];out[off_op_finish+op]=op_finish_time[op];}
    for(int i=0;i<num_ready;i++)out[off_precedence_ready+i]=precedence_ready_ops[i];
    out[off_ready_count]=num_ready;for(int i=0;i<num_scheduled;i++)out[off_sequence+i]=seq[i];out[off_sequence_count]=num_scheduled;out[off_makespan]=cmax;
    if(num_scheduled==total_real_ops){int old_best=atomicMin(d_global_best,cmax);if(cmax<old_best)for(int i=0;i<SNAPSHOT_INTS_val;i++)d_best_sol[i]=out[i];}
}

__global__ void init_indices(int*indices,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)indices[i]=i;}
__global__ void kernel_gather_topk(const int*d_in,const int*d_sorted_idx,int*d_out,int num,int SINTS){
    int rank=blockIdx.x; if(rank>=num)return;
    int src=d_sorted_idx[rank],tid=threadIdx.x;
    for(int i=tid;i<SINTS;i+=blockDim.x)d_out[rank*SINTS+i]=d_in[src*SINTS+i];
}

// =============================================================================
// SECTION 9 – GPU Buffer Management
// =============================================================================
static void allocate_gpu_buffers(){
    CUDA_CHECK(cudaMemcpyToSymbol(c_ops,gpu_all_ops,HARD_MAX_OPS*sizeof(OperationInfo)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_op_order,gpu_machine_op_order,HARD_MAX_MACHINES*HARD_MAX_JOBS*sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_op_count,gpu_machine_op_count,HARD_MAX_MACHINES*sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_mach_total,gpu_mach_total,HARD_MAX_MACHINES*sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_job_total, gpu_job_total, HARD_MAX_JOBS   *sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_tail_static, gpu_tail_static, HARD_MAX_OPS*sizeof(int)));
    size_t beam_bytes=(size_t)BUF_BEAM*SNAPSHOT_INTS*sizeof(int);
    size_t child_bytes=(size_t)BUF_BEAM*BUF_JOBS*SNAPSHOT_INTS*sizeof(int);
    size_t score_bytes=(size_t)BUF_BEAM*BUF_JOBS*sizeof(float);
    size_t index_bytes=(size_t)BUF_BEAM*BUF_JOBS*sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_beam_snapshots,beam_bytes)); CUDA_CHECK(cudaMalloc(&d_child_snapshots,child_bytes));
    CUDA_CHECK(cudaMalloc(&d_child_scores,score_bytes));  CUDA_CHECK(cudaMalloc(&d_child_scores_out,score_bytes));
    CUDA_CHECK(cudaMalloc(&d_child_indices,index_bytes));  CUDA_CHECK(cudaMalloc(&d_child_indices_out,index_bytes));
    CUDA_CHECK(cudaMalloc(&d_global_best_makespan,sizeof(int))); CUDA_CHECK(cudaMalloc(&d_best_solution,SNAPSHOT_INTS*sizeof(int)));
    cub::DeviceRadixSort::SortPairs(nullptr,cub_temp_bytes,d_child_scores,d_child_scores_out,d_child_indices,d_child_indices_out,BUF_BEAM*BUF_JOBS);
    CUDA_CHECK(cudaMalloc(&d_cub_temp_storage,cub_temp_bytes));
    CUDA_CHECK(cudaMallocHost(&h_input_flat,beam_bytes));
}
static void free_gpu_buffers(){
    cudaFree(d_beam_snapshots);cudaFree(d_child_snapshots);cudaFree(d_child_scores);cudaFree(d_child_scores_out);
    cudaFree(d_child_indices);cudaFree(d_child_indices_out);cudaFree(d_cub_temp_storage);
    cudaFree(d_global_best_makespan);cudaFree(d_best_solution);cudaFreeHost(h_input_flat);
}

// =============================================================================
// SECTION 10 – GPU Beam Search Driver  ← STATS ADDED HERE
// =============================================================================
static ScheduleSnapshot make_root_snapshot(){
    ScheduleSnapshot s{};
    s.num_precedence_ready=nbjob;
    for(int j=0;j<nbjob;j++) s.precedence_ready_ops[j]=j*nbmachine+1;
    s.makespan=0; s.num_scheduled=0;
    // seed LB counters
    for(int m=1;m<=nbmachine;m++) s.mach_rem[m]  = gpu_mach_total[m];
    for(int j=0;j<nbjob;j++)      s.job_rem[j]   = gpu_job_total[j];
    // job_ready stays 0
    return s;
}

static int run_gpu_beam(int /*iter*/){
    int pass_best=global_best_cmax;
    CUDA_CHECK(cudaMemcpy(d_global_best_makespan,&pass_best,sizeof(int),cudaMemcpyHostToDevice));
    ScheduleSnapshot root=make_root_snapshot(); pack_snapshot(root,h_input_flat);
    CUDA_CHECK(cudaMemcpy(d_beam_snapshots,h_input_flat,SNAPSHOT_INTS*sizeof(int),cudaMemcpyHostToDevice));
    int cur_beam=1,max_depth=nbjob*nbmachine+10,total_ops=nbjob*nbmachine;

    for(int depth=1;depth<max_depth&&cur_beam>0;depth++){
        int rseed=rand();
        size_t shm=(SNAPSHOT_INTS+HARD_MAX_JOBS+1)*sizeof(int);
        kernel_expand_beam_layer<<<cur_beam,BUF_JOBS,shm>>>(
            d_beam_snapshots,d_child_snapshots,d_child_scores,d_global_best_makespan,d_best_solution,
            cur_beam,BUF_JOBS,SNAPSHOT_INTS,
            SNAP_OFF_MACHINE_OCCUPIED_OP,SNAP_OFF_MACHINE_READY_TIME,SNAP_OFF_CRITICAL_PARENT_OP,
            SNAP_OFF_OP_START_TIME,SNAP_OFF_OP_FINISH_TIME,SNAP_OFF_PRECEDENCE_READY,SNAP_OFF_READY_COUNT,
            SNAP_OFF_SEQUENCE,SNAP_OFF_SEQUENCE_COUNT,SNAP_OFF_MAKESPAN,
            SNAP_OFF_MACH_REM,SNAP_OFF_JOB_READY,SNAP_OFF_JOB_REM,
            nbmachine,nboperation,total_ops,HARD_MAX_JOBS,rseed,
            g_lb_mode, nbjob);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

        int total_children=cur_beam*BUF_JOBS;
        // ── STAT: each child is one schedule state evaluated ──────────────────
        rank_stats.partial_states += total_children;
        // Complete solutions are those that reached full depth; approximate as
        // cur_beam * BUF_JOBS at the final depth (generous upper bound for GPU).

        int tpb=256;
        init_indices<<<(total_children+tpb-1)/tpb,tpb>>>(d_child_indices,total_children);
        cub::DeviceRadixSort::SortPairs(d_cub_temp_storage,cub_temp_bytes,d_child_scores,d_child_scores_out,d_child_indices,d_child_indices_out,total_children);
        int cands=min(total_children,gpu_beam_width);
        float h_top[HARD_MAX_BEAM];
        CUDA_CHECK(cudaMemcpy(h_top,d_child_scores_out,cands*sizeof(float),cudaMemcpyDeviceToHost));
        int valid=0; while(valid<cands&&h_top[valid]<1e8f)valid++;
        if(valid==0) break;
        kernel_gather_topk<<<valid,256>>>(d_child_snapshots,d_child_indices_out,d_beam_snapshots,valid,SNAPSHOT_INTS);
        cur_beam=valid;
    }
    // Count one complete solution per beam pass (the best schedule found)
    rank_stats.complete_solutions++;
    rank_stats.beam_iterations++;

    CUDA_CHECK(cudaMemcpy(&pass_best,d_global_best_makespan,sizeof(int),cudaMemcpyDeviceToHost));
    if(pass_best<global_best_cmax){
        CUDA_CHECK(cudaMemcpy(h_input_flat,d_best_solution,SNAPSHOT_INTS*sizeof(int),cudaMemcpyDeviceToHost));
        if(h_input_flat[SNAP_OFF_SEQUENCE_COUNT]==total_ops)
            for(int i=0;i<total_ops;i++) h_global_best_seq[i]=h_input_flat[SNAP_OFF_SEQUENCE+i];
    }
    return pass_best;
}

// =============================================================================
// SECTION 11 – CPU Calibration
// =============================================================================
static double calibrate_cpu(int calib_k){
    int saved=cpu_beam_width; cpu_beam_width=calib_k;
    clock_t t0=clock(); run_cpu_beam(0); clock_t t1=clock();
    // Don't count calibration in stats
    rank_stats.partial_states=0; rank_stats.complete_solutions=0; rank_stats.beam_iterations=0;
    cpu_beam_width=saved;
    return (double)(t1-t0)/CLOCKS_PER_SEC;
}

// =============================================================================
// SECTION 12 – main()
// =============================================================================
int main(int argc,char**argv){
    if(argc>1&&argv[1]) snprintf(instance_path,sizeof(instance_path),"../../lawrence_instances/la%s.txt",argv[1]);
    if(argc>2&&argv[2]) gpu_beam_width=atoi(argv[2]);
    if(argc>3&&argv[3]) cpu_beam_width=atoi(argv[3]);
    if(argc>4&&argv[4]) num_iterations=atoi(argv[4]);
    // NEW: argv[5] = lb mode.  "auto" (or absent) = per-rank cycling; or 0,1,2
    if (argc > 5 && argv[5]) {
        if (strcmp(argv[5], "auto") == 0) g_lb_mode_cli = -1;
        else                              g_lb_mode_cli = atoi(argv[5]);
    }
    double wall_budget_sec=(double)num_iterations;

    if(gpu_beam_width<1||gpu_beam_width>HARD_MAX_BEAM){fprintf(stderr,"gpu_beam_width must be in [1,%d]\n",HARD_MAX_BEAM);return 1;}
    {ifstream chk(instance_path);if(!chk){fprintf(stderr,"File not found: %s\n",instance_path);return 1;}}

    MPI_Init(&argc,&argv);
    int world_size,my_rank;
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    srand((unsigned)time(NULL)+my_rank*1234567);

    // ── Resolve per-rank LB mode ─────────────────────────────────────────────
    if (g_lb_mode_cli >= 0 && g_lb_mode_cli < LB_N_MODES) {
        g_lb_mode = g_lb_mode_cli;
    } else {
        g_lb_mode = my_rank % LB_N_MODES;   // rank 0→CUR, 1→LB_M, 2→LB_MJ, 3→CUR, ...
    }

    int num_gpus=0;
    cudaGetDeviceCount(&num_gpus);
    bool is_gpu_rank=(num_gpus>0&&my_rank<num_gpus);
    bool is_cpu_rank=!is_gpu_rank;

    if(my_rank==0){
        printf("=== Hybrid Beam Search + Throughput Statistics + Lower-Bound Scoring ===\n");
        printf("Instance : %s\n",instance_path);
        printf("MPI ranks: %d  |  GPUs detected: %d\n",world_size,num_gpus);
        printf("GPU ranks: 0..%d  |  CPU ranks: %d..%d\n",max(0,num_gpus-1),num_gpus,world_size-1);
        printf("LB mode  : %s (per-rank cycling)\n",
               g_lb_mode_cli < 0 ? "auto" : lb_mode_name[g_lb_mode_cli]);
    }
    printf("[rank %d] LB mode = %s\n", my_rank, lb_mode_name[g_lb_mode]);

    load_instance(); compute_snapshot_layout();
    if(my_rank==0) printf("Jobs: %d  Machines: %d  Ops: %d\n",nbjob,nbmachine,nboperation);

    // ── GPU setup (no MPI) ────────────────────────────────────────────────────
    if(is_gpu_rank){
        CUDA_CHECK(cudaSetDevice(my_rank%num_gpus));
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize,32768));
        BUF_BEAM=gpu_beam_width; allocate_gpu_buffers();
        double wt0=MPI_Wtime(); run_gpu_beam(0);
        // Reset stats — warm-up not counted
        rank_stats={0,0,0,0.0};
        if(my_rank==0) printf("GPU warm-up time: %.3f s  (beam_width=%d)\n",MPI_Wtime()-wt0,gpu_beam_width);
    }

    // ── CPU calibration (no MPI) ──────────────────────────────────────────────
    if(is_cpu_rank){
        if(cpu_beam_width<=0){
            double calib_t=calibrate_cpu(CALIB_K);
            double target=0.5;
            cpu_beam_width=(calib_t>0.0)?max(CALIB_K,min(MAX_CPU_KBEST,(int)((double)CALIB_K*target/calib_t))):500;
            printf("[CPU rank %d] calib=%.3fs  target=%.1fs  → cpu_beam_width=%d\n",my_rank,calib_t,target,cpu_beam_width);
        } else {
            printf("[CPU rank %d] cpu_beam_width=%d (manual)\n",my_rank,cpu_beam_width);
        }
    }

    // ── Fully independent search loop — ZERO MPI ─────────────────────────────
    int local_best=global_best_cmax,global_best=global_best_cmax;
    double loop_start=MPI_Wtime(),next_print=loop_start+10.0;
    int iter=0;

    while((MPI_Wtime()-loop_start)<wall_budget_sec){
        int result=is_gpu_rank?run_gpu_beam(iter):run_cpu_beam(iter);
        if(result<local_best){local_best=result;global_best_cmax=local_best;}
        if(my_rank==0){double now=MPI_Wtime();if(now>=next_print){printf("Rank 0  iter %4d  makespan: %d  (%.1f s)\n",iter,local_best,now-loop_start);fflush(stdout);next_print=now+10.0;}}
        iter++;
    }
    rank_stats.elapsed_sec = MPI_Wtime()-loop_start;

    // ── Collect global best ───────────────────────────────────────────────────
    MPI_Reduce(&local_best,&global_best,1,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);

    // ── Per-rank stats ────────────────────────────────────────────────────────
    double ps_rate  = (rank_stats.elapsed_sec>0) ? rank_stats.partial_states   /rank_stats.elapsed_sec : 0;
    double cs_rate  = (rank_stats.elapsed_sec>0) ? rank_stats.complete_solutions/rank_stats.elapsed_sec : 0;
    double bi_rate  = (rank_stats.elapsed_sec>0) ? rank_stats.beam_iterations   /rank_stats.elapsed_sec : 0;

    // Barrier so per-rank lines are printed before the summary
    MPI_Barrier(MPI_COMM_WORLD);

    if(is_gpu_rank)
        printf("[GPU rank %d] lb=%s  makespan: %d"
               "  |  states/s: %10.0f"
               "  |  solutions/s: %7.1f"
               "  |  beam_iters/s: %6.2f"
               "  |  total_states: %lld"
               "  |  total_solutions: %lld\n",
               my_rank,lb_mode_name[g_lb_mode],local_best,ps_rate,cs_rate,bi_rate,
               rank_stats.partial_states,rank_stats.complete_solutions);
    else
        printf("[CPU rank %d] lb=%s  makespan: %d"
               "  |  states/s: %10.0f"
               "  |  solutions/s: %7.1f"
               "  |  beam_iters/s: %6.2f"
               "  |  beam_candidates: %d\n",
               my_rank,lb_mode_name[g_lb_mode],local_best,ps_rate,cs_rate,bi_rate,cpu_beam_width);

    // ── Aggregate summary on rank 0 ───────────────────────────────────────────
    // Gather all per-rank stats to rank 0 via Reduce
    long long sum_partial=0, sum_complete=0;
    double    sum_ps_rate=0, sum_cs_rate=0;
    MPI_Reduce(&rank_stats.partial_states,    &sum_partial,  1,MPI_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&rank_stats.complete_solutions,&sum_complete, 1,MPI_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&ps_rate, &sum_ps_rate, 1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&cs_rate, &sum_cs_rate, 1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

    if(my_rank==0){
        printf("\n=== RESULT ===\n");
        printf("Best makespan            : %d\n",global_best);
        printf("Wall budget              : %.0f s  (actual: %.1f s)\n",wall_budget_sec,rank_stats.elapsed_sec);
        printf("Roles                    : %d GPU rank(s), %d CPU rank(s)\n",num_gpus,world_size-num_gpus);
        printf("GPU beam width           : %d\n",gpu_beam_width);
        printf("\n--- Throughput Summary (all ranks combined) ---\n");
        printf("Total partial states     : %lld\n",   sum_partial);
        printf("Total complete solutions : %lld\n",   sum_complete);
        printf("Aggregate states/s       : %.0f\n",   sum_ps_rate);
        printf("Aggregate solutions/s    : %.1f\n",   sum_cs_rate);
        printf("GPU states/s (rank 0)    : %.0f  [%d ranks × beam_width=%d × depth=%d]\n",
               ps_rate, num_gpus, gpu_beam_width, nbjob*nbmachine);
        printf("CPU states/s (per rank)  : %.0f  [beam_candidates≈%d]\n",
               sum_ps_rate / max(1,world_size-num_gpus),cpu_beam_width);
    }

    if(is_gpu_rank) free_gpu_buffers();
    MPI_Finalize();
    return 0;
}