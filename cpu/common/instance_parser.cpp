#include "instance_parser.h"

#include <cstdio>
#include <cstring>
#include <fstream>

// Resolve instance path or benchmark name to a valid file path
bool resolve_instance_file(const char* input_arg, char* out_path, int path_size)
{
    // Check direct path
    strncpy(out_path, input_arg, path_size - 1);
    out_path[path_size - 1] = '\0';
    { std::ifstream f(out_path); if (f.good()) return true; }

    const char* formats[] = {
        "./la%s.txt",
        "./ta%s.txt",
        "./lawrence_instances/la%s.txt",
        "../lawrence_instances/la%s.txt",
        "../../lawrence_instances/la%s.txt",
        "./taillard_instances/ta%s",
        "../taillard_instances/ta%s",
        "../../taillard_instances/ta%s"
    };

    // Check candidate search locations
    for (const char* fmt : formats) {
        snprintf(out_path, path_size, fmt, input_arg);
        std::ifstream f(out_path);
        if (f.good()) return true;
    }

    return false;
}

// Parse JSSP instance file
bool parse_instance(const char* filepath, JSSPInstance& inst)
{
    FILE* f = fopen(filepath, "r");
    if (!f) {
        fprintf(stderr, "Error: cannot open instance file: %s\n", filepath);
        return false;
    }

    if (fscanf(f, "%d %d", &inst.nbjob, &inst.nbmachine) != 2) {
        fprintf(stderr, "Error: cannot read header from %s\n", filepath);
        fclose(f);
        return false;
    }

    inst.nboperation = inst.nbjob * inst.nbmachine + 2;

    inst.operations.clear();
    inst.opr_mach.clear();
    inst.opr_mach.resize(inst.nbmachine + 1);

    // Dummy start operation
    Operation dummy;
    dummy.num_op = 0;
    dummy.num_mc = 0;
    dummy.dure   = 0;
    dummy.num_j  = 0;
    inst.operations.push_back(dummy);

    int idf_op = 1;
    for (int job = 1; job <= inst.nbjob; job++) {
        for (int step = 1; step <= inst.nbmachine; step++) {
            int mc = 0, dur = 0;
            fscanf(f, "%d %d", &mc, &dur);

            Operation op;
            op.num_op = idf_op;
            op.num_mc = mc + 1; // 1-indexed machine
            op.dure   = dur;
            op.num_j  = job;

            inst.operations.push_back(op);
            inst.opr_mach[mc + 1].push_back(idf_op);
            idf_op++;
        }
    }

    // Dummy end operation
    Operation end_op;
    end_op.num_op = inst.nbjob * inst.nbmachine + 1;
    end_op.num_mc = 0;
    end_op.dure   = 0;
    end_op.num_j  = 0;
    inst.operations.push_back(end_op);

    fclose(f);
    return true;
}
