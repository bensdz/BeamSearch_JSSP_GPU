#include "solver.cuh"
#include "../../cpu/common/instance_parser.h"

#include <iostream>
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv)
{
    char instance_file[512] = "./la33.txt";
    int  beam_width = 1000;

    // Parse command line arguments
    if (argc > 1) {
        if (!resolve_instance_file(argv[1], instance_file, sizeof(instance_file))) {
            std::cerr << "Error: cannot locate instance file for: " << argv[1] << std::endl;
            return 1;
        }
    }
    if (argc > 2) {
        beam_width = atoi(argv[2]);
        if (beam_width < 1) {
            std::cerr << "Error: beam_width must be >= 1" << std::endl;
            return 1;
        }
    }

    // Load problem instance
    JSSPInstance inst;
    if (!parse_instance(instance_file, inst))
        return 1;

    std::cout << "  Instance File : " << instance_file << std::endl;
    std::cout << "  Jobs          : " << inst.nbjob     << std::endl;
    std::cout << "  Machines      : " << inst.nbmachine  << std::endl;
    std::cout << "  Beam Width (K): " << beam_width      << std::endl;

    // Run GPU beam search
    clock_t t0 = clock();
    int seed = 12345;

    int best_makespan = run_gpu_beam_search_v1(inst, beam_width, 500000, seed);

    clock_t t1 = clock();
    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

    // Print results
    std::cout << "  ===========================" << std::endl;
    std::cout << "  Best Makespan    : " << best_makespan << std::endl;
    std::cout << "  Execution Time   : " << elapsed << " seconds" << std::endl;

    return 0;
}
