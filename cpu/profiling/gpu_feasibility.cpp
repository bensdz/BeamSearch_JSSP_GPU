#include "../common/jssp_types.h"
#include "../common/instance_parser.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>
#include <algorithm>
#include <mpi.h>

// Profile single beam search run, measuring parallel and sequential durations
int profile_beam_search_pass(const JSSPInstance& inst, int beam_width, int best_known,
                            double& out_total_time, double& out_par_time, double& out_seq_time)
{
    const int total_ops = inst.nbjob * inst.nbmachine;
    auto t_start_total = std::chrono::high_resolution_clock::now();

    // Initialize beam
    std::vector<JobShopState> candidates;
    candidates.reserve(beam_width);
    candidates.push_back(create_initial_state(inst));

    std::vector<JobShopState> next_beam;
    next_beam.reserve(beam_width);

    double par_time = 0.0;
    double seq_overhead = 0.0;
    int best_makespan = 500000;

    // Search expansion loop
    for (int step = 1; step <= total_ops + 10 && !candidates.empty(); step++) {
        auto t_start_layer = std::chrono::high_resolution_clock::now();
        next_beam.clear();

        for (int ic = 0; ic < (int)candidates.size(); ic++) {
            const std::vector<int>& ready = candidates[ic].c;
            int nb = (int)ready.size();

            if (nb > 0) {
                // Pick target machine
                int pivot = rand() % nb;
                int target_mc = inst.operations[ready[pivot]].num_mc;

                // Branch on ready operations for target machine
                for (int r = 0; r < nb; r++) {
                    if (inst.operations[ready[r]].num_mc != target_mc)
                        continue;

                    // Measure state copy overhead
                    auto t_c1 = std::chrono::high_resolution_clock::now();
                    JobShopState child = candidates[ic];
                    auto t_c2 = std::chrono::high_resolution_clock::now();
                    seq_overhead += std::chrono::duration<double>(t_c2 - t_c1).count();

                    int op = ready[r];

                    // Remove op from ready list
                    auto it = std::find(child.c.begin(), child.c.end(), op);
                    if (it != child.c.end())
                        child.c.erase(it);

                    // Schedule operation
                    schedule_operation(op, inst, child);

                    // Heuristic score
                    double noise = (double)(rand() % 1000) / 1000000.0;
                    child.score = ((double)child.cmax / child.seq.size()) + noise;

                    // Insert into beam
                    if (child.cmax < best_known)
                        insert_into_beam(next_beam, child, beam_width);

                    // Update best complete makespan
                    if ((int)child.seq.size() == total_ops && child.cmax < best_makespan)
                        best_makespan = child.cmax;
                }
            }
        }

        // Swap beam levels
        candidates.swap(next_beam);

        auto t_end_layer = std::chrono::high_resolution_clock::now();
        double layer_duration = std::chrono::duration<double>(t_end_layer - t_start_layer).count();
        par_time += (layer_duration - seq_overhead);
        seq_overhead = 0.0;
    }

    auto t_end_total = std::chrono::high_resolution_clock::now();
    out_total_time = std::chrono::duration<double>(t_end_total - t_start_total).count();
    out_par_time = par_time;
    out_seq_time = out_total_time - par_time;

    return best_makespan;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    char instance_file[512] = "./la33.txt";
    if (argc > 1) {
        if (!resolve_instance_file(argv[1], instance_file, sizeof(instance_file))) {
            if (my_rank == 0)
                fprintf(stderr, "Error: cannot resolve instance: %s\n", argv[1]);
            MPI_Finalize();
            return 1;
        }
    }

    // Load instance
    JSSPInstance inst;
    if (!parse_instance(instance_file, inst)) {
        MPI_Finalize();
        return 1;
    }

    // Print table header
    if (my_rank == 0) {
        printf("\n--- JSSP CPU Feasibility Study ---");
        printf("\nInstance: %s (%d jobs x %d machines)", instance_file, inst.nbjob, inst.nbmachine);
        printf("\nRuns per K: 100\n\n");
        printf("%-6s | %-8s | %-8s | %-12s | %-12s | %-12s | %-10s\n",
               "K", "Min Cmx", "Avg Cmx", "Avg Total(s)", "Avg Par(s)", "Avg Seq(us)", "Par Ratio");
        printf("-------|----------|----------|--------------|--------------|--------------|------------\n");
    }

    const int num_runs = 100;

    // Sweep beam width from 500 to 5000
    for (int k_val = 500; k_val <= 5000; k_val += 500) {
        int local_min_cmax = 1000000;
        double sum_cmax = 0.0;
        double sum_total_time = 0.0;
        double sum_par_time = 0.0;
        double sum_seq_time = 0.0;
        int successful_runs = 0;

        srand((unsigned)time(NULL) + my_rank + k_val);

        for (int run = 0; run < num_runs; run++) {
            double total_time = 0.0, par_time = 0.0, seq_time = 0.0;
            int cm = profile_beam_search_pass(inst, k_val, 500000, total_time, par_time, seq_time);

            if (cm < local_min_cmax)
                local_min_cmax = cm;

            sum_cmax += cm;
            sum_total_time += total_time;
            sum_par_time += par_time;
            sum_seq_time += seq_time;
            successful_runs++;
        }

        // Aggregate statistics across MPI ranks
        int global_min_cmax = 0;
        double global_sum_cmax = 0.0;
        double global_sum_total = 0.0;
        double global_sum_par = 0.0;
        double global_sum_seq = 0.0;
        int total_runs = 0;

        MPI_Reduce(&local_min_cmax, &global_min_cmax, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&sum_cmax, &global_sum_cmax, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&sum_total_time, &global_sum_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&sum_par_time, &global_sum_par, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&sum_seq_time, &global_sum_seq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&successful_runs, &total_runs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (my_rank == 0 && total_runs > 0) {
            double avg_cmax = global_sum_cmax / total_runs;
            double avg_total = global_sum_total / total_runs;
            double avg_par = global_sum_par / total_runs;
            double avg_seq = global_sum_seq / total_runs;
            double par_ratio = (avg_total > 0.0) ? (avg_par / avg_total) * 100.0 : 0.0;

            printf("%-6d | %-8d | %-8.1f | %-12.4f | %-12.4f | %-12.1f | %-8.4f%%\n",
                   k_val, global_min_cmax, avg_cmax, avg_total, avg_par, avg_seq * 1e6, par_ratio);
        }
    }

    if (my_rank == 0)
        printf("-------------------------------------------------------------------------------\n");

    MPI_Finalize();
    return 0;
}
