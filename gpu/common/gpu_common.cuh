#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "../../cpu/common/jssp_types.h"

// CUDA API error check wrapper
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t _e = (call);                                      \
        if (_e != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_e));      \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Upper bounds for device memory allocations
#define HARD_MAX_OPS       2100
#define HARD_MAX_MACHINES   32
#define HARD_MAX_JOBS      100

// Operation descriptor
struct OperationInfo {
    int machine_id;
    int duration;
    int job_id;
};

// Constant memory for problem instance operations
__constant__ OperationInfo c_ops[HARD_MAX_OPS];

// Copy instance operations to device constant memory
inline void copy_instance_to_device(const JSSPInstance& inst) {
    OperationInfo h_ops[HARD_MAX_OPS];
    int count = std::min((int)inst.operations.size(), HARD_MAX_OPS);

    for (int i = 0; i < count; i++) {
        h_ops[i].machine_id = inst.operations[i].num_mc;
        h_ops[i].duration   = inst.operations[i].dure;
        h_ops[i].job_id     = inst.operations[i].num_j;
    }

    CUDA_CHECK(cudaMemcpyToSymbol(c_ops, h_ops, count * sizeof(OperationInfo)));
}
