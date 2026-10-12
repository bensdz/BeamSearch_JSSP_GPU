#include "solver.cuh"
#include "../common/gpu_common.cuh"
#include "../common/gpu_snapshot.cuh"

#include <climits>
#include <cub/cub.cuh>

// Kernel: Expand beam candidates (1 block per parent, 1 thread per branch)
__global__ void kernel_expand_beam_layer(
    const int     *d_in_beam,
    int           *d_out_snapshots,
    float         *d_out_scores,
    int           *d_out_indices,
    int           *d_global_best,
    int           *d_best_sol,
    int            num_candidates,
    int            num_branches_per_candidate,
    SnapshotLayout layout,
    int            num_machines_1idx,
    int            num_ops_tot,
    int            total_real_ops,
    int            random_seed
) {
    extern __shared__ int s_data[];
    int *s_parent_snap        = s_data;
    int *s_candidate_ops      = s_parent_snap + layout.total_ints;
    int *s_num_candidates_ptr = s_candidate_ops + num_branches_per_candidate;

    int candidate_id = blockIdx.x;
    int branch_id    = threadIdx.x;

    if (candidate_id >= num_candidates) return;

    int child_flat_index = candidate_id * num_branches_per_candidate + branch_id;
    int out_base         = child_flat_index * layout.total_ints;

    d_out_scores[child_flat_index]  = 1e9f;
    d_out_indices[child_flat_index] = child_flat_index;
    d_out_snapshots[out_base + layout.off_makespan] = INT_MAX;

    // Load parent snapshot into shared memory
    for (int i = branch_id; i < layout.total_ints; i += blockDim.x) {
        s_parent_snap[i] = d_in_beam[candidate_id * layout.total_ints + i];
    }
    __syncthreads();

    // Thread 0 picks target bottleneck machine
    if (branch_id == 0) {
        int num_ready = s_parent_snap[layout.off_ready_count];
        if (num_ready > 0) {
            unsigned int hash = random_seed + candidate_id;
            hash ^= hash >> 16; hash *= 0x7feb352d;
            hash ^= hash >> 15; hash *= 0x846ca68b;
            hash ^= hash >> 16;
            int chosen_op = s_parent_snap[layout.off_precedence_ready + (hash % num_ready)];
            int chosen_machine = c_ops[chosen_op].machine_id;

            int count = 0;
            for (int i = 0; i < num_ready; i++) {
                int op = s_parent_snap[layout.off_precedence_ready + i];
                if (c_ops[op].machine_id == chosen_machine) {
                    s_candidate_ops[count++] = op;
                }
            }
            *s_num_candidates_ptr = count;
        } else {
            *s_num_candidates_ptr = 0;
        }
    }
    __syncthreads();

    int valid_ops = *s_num_candidates_ptr;
    if (branch_id >= valid_ops) return;

    int chosen_op = s_candidate_ops[branch_id];

    // Hydrate state from parent shared memory
    int machine_ready_time[HARD_MAX_MACHINES];
    for (int m = 0; m <= num_machines_1idx; m++) {
        machine_ready_time[m] = s_parent_snap[layout.off_machine_ready + m];
    }

    int op_finish_time[HARD_MAX_OPS];
    for (int op = 0; op < num_ops_tot; op++) {
        op_finish_time[op] = s_parent_snap[layout.off_op_finish + op];
    }

    int precedence_ready_ops[HARD_MAX_JOBS];
    int num_ready = s_parent_snap[layout.off_ready_count];
    for (int i = 0; i < num_ready; i++) {
        precedence_ready_ops[i] = s_parent_snap[layout.off_precedence_ready + i];
    }

    int seq[HARD_MAX_OPS];
    int num_scheduled = s_parent_snap[layout.off_sequence_count];
    for (int i = 0; i < num_scheduled; i++) {
        seq[i] = s_parent_snap[layout.off_sequence + i];
    }

    // Remove chosen operation from ready list
    for (int i = 0; i < num_ready; i++) {
        if (precedence_ready_ops[i] == chosen_op) {
            precedence_ready_ops[i] = precedence_ready_ops[--num_ready];
            break;
        }
    }

    // Schedule operation
    dev_schedule_operation(chosen_op, machine_ready_time, op_finish_time,
                           precedence_ready_ops, num_ready, seq, num_scheduled,
                           num_machines_1idx);

    // Compute makespan
    int cmax = 0;
    for (int m = 1; m <= num_machines_1idx; m++) {
        if (machine_ready_time[m] > cmax) cmax = machine_ready_time[m];
    }

    // Prune candidates worse than best known makespan
    if (cmax < *d_global_best) {
        float base_score = (num_scheduled == 0) ? (float)cmax : ((float)cmax / (float)num_scheduled);
        unsigned int noise_hash = random_seed ^ (candidate_id * 0x85ebca6b) ^ (branch_id * 0xc2b2ae35);
        noise_hash ^= noise_hash >> 16; noise_hash *= 0x85ebca6b;
        noise_hash ^= noise_hash >> 16; noise_hash *= 0xc2b2ae35;
        noise_hash ^= noise_hash >> 16;
        float sibling_noise = (float)(noise_hash % 1000) / 1000000000.0f;
        d_out_scores[child_flat_index] = base_score + sibling_noise;

        // Write child snapshot to device buffer
        int *out = d_out_snapshots + out_base;
        for (int m = 0; m <= num_machines_1idx; m++) {
            out[layout.off_machine_ready + m] = machine_ready_time[m];
        }
        for (int op = 0; op < num_ops_tot; op++) {
            out[layout.off_op_finish + op] = op_finish_time[op];
        }
        for (int i = 0; i < num_ready; i++) {
            out[layout.off_precedence_ready + i] = precedence_ready_ops[i];
        }
        out[layout.off_ready_count] = num_ready;
        for (int i = 0; i < num_scheduled; i++) {
            out[layout.off_sequence + i] = seq[i];
        }
        out[layout.off_sequence_count] = num_scheduled;
        out[layout.off_makespan] = cmax;

        // Update best complete makespan
        if (num_scheduled == total_real_ops) {
            int old_best = atomicMin(d_global_best, cmax);
            if (cmax < old_best) {
                for (int i = 0; i < layout.total_ints; i++) {
                    d_best_sol[i] = out[i];
                }
            }
        }
    }
}

// Kernel: Gather top-K snapshots from sorted indices
__global__ void kernel_gather_topk(
    const int *d_child_snapshots,
    const int *d_sorted_indices,
    int       *d_out_beam,
    int        num_to_gather,
    int        snapshot_ints
) {
    int rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank < num_to_gather) {
        int src_child_idx = d_sorted_indices[rank];
        const int *src = d_child_snapshots + src_child_idx * snapshot_ints;
        int *dst = d_out_beam + rank * snapshot_ints;
        for (int i = 0; i < snapshot_ints; i++) {
            dst[i] = src[i];
        }
    }
}

int run_gpu_beam_search_v1(
    const JSSPInstance& inst,
    int beam_width,
    int best_known,
    int random_seed,
    int* out_best_sequence
) {
    const int total_ops = inst.nbjob * inst.nbmachine;
    const int branches_per_candidate = inst.nbjob;
    SnapshotLayout layout = compute_snapshot_layout(inst.nbjob, inst.nbmachine, inst.nboperation);

    // Upload instance to GPU constant memory
    copy_instance_to_device(inst);

    // Allocate GPU buffers
    int *d_beam_snapshots = nullptr;
    int *d_child_snapshots = nullptr;
    float *d_child_scores = nullptr;
    float *d_child_scores_out = nullptr;
    int *d_child_indices = nullptr;
    int *d_child_indices_out = nullptr;
    int *d_global_best = nullptr;
    int *d_best_sol = nullptr;

    int max_children = beam_width * branches_per_candidate;

    CUDA_CHECK(cudaMalloc(&d_beam_snapshots, beam_width * layout.total_ints * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_child_snapshots, max_children * layout.total_ints * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_child_scores, max_children * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_child_scores_out, max_children * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_child_indices, max_children * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_child_indices_out, max_children * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_global_best, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_sol, layout.total_ints * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_global_best, &best_known, sizeof(int), cudaMemcpyHostToDevice));

    // Allocate CUB temporary storage
    void *d_cub_temp = nullptr;
    size_t cub_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, cub_bytes,
                                    d_child_scores, d_child_scores_out,
                                    d_child_indices, d_child_indices_out,
                                    max_children);
    CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_bytes));

    // Initialize initial state on host and upload
    std::vector<int> h_initial_flat(layout.total_ints);
    pack_initial_snapshot(inst, h_initial_flat.data(), layout);
    CUDA_CHECK(cudaMemcpy(d_beam_snapshots, h_initial_flat.data(),
                          layout.total_ints * sizeof(int), cudaMemcpyHostToDevice));

    int current_num_candidates = 1;
    size_t shared_mem_bytes = (layout.total_ints + branches_per_candidate + 1) * sizeof(int);

    // Main beam construction loop
    for (int step = 0; step < total_ops; step++) {
        int total_children = current_num_candidates * branches_per_candidate;

        kernel_expand_beam_layer<<<current_num_candidates, branches_per_candidate, shared_mem_bytes>>>(
            d_beam_snapshots,
            d_child_snapshots,
            d_child_scores,
            d_child_indices,
            d_global_best,
            d_best_sol,
            current_num_candidates,
            branches_per_candidate,
            layout,
            inst.nbmachine,
            inst.nboperation,
            total_ops,
            random_seed + step
        );
        CUDA_CHECK(cudaGetLastError());

        // Sort children by score
        cub::DeviceRadixSort::SortPairs(d_cub_temp, cub_bytes,
                                        d_child_scores, d_child_scores_out,
                                        d_child_indices, d_child_indices_out,
                                        total_children);
        CUDA_CHECK(cudaGetLastError());

        int num_to_gather = std::min(total_children, beam_width);
        int threads = 256;
        int blocks = (num_to_gather + threads - 1) / threads;

        kernel_gather_topk<<<blocks, threads>>>(
            d_child_snapshots,
            d_child_indices_out,
            d_beam_snapshots,
            num_to_gather,
            layout.total_ints
        );
        CUDA_CHECK(cudaGetLastError());

        current_num_candidates = num_to_gather;
    }

    int h_best_makespan = 500000;
    CUDA_CHECK(cudaMemcpy(&h_best_makespan, d_global_best, sizeof(int), cudaMemcpyDeviceToHost));

    if (out_best_sequence != nullptr && h_best_makespan < 500000) {
        std::vector<int> h_sol(layout.total_ints);
        CUDA_CHECK(cudaMemcpy(h_sol.data(), d_best_sol, layout.total_ints * sizeof(int), cudaMemcpyDeviceToHost));
        std::memcpy(out_best_sequence, h_sol.data() + layout.off_sequence, total_ops * sizeof(int));
    }

    // Free device allocations
    CUDA_CHECK(cudaFree(d_cub_temp));
    CUDA_CHECK(cudaFree(d_best_sol));
    CUDA_CHECK(cudaFree(d_global_best));
    CUDA_CHECK(cudaFree(d_child_indices_out));
    CUDA_CHECK(cudaFree(d_child_indices));
    CUDA_CHECK(cudaFree(d_child_scores_out));
    CUDA_CHECK(cudaFree(d_child_scores));
    CUDA_CHECK(cudaFree(d_child_snapshots));
    CUDA_CHECK(cudaFree(d_beam_snapshots));

    return h_best_makespan;
}
