#include <mma.h>

#include "kernel.cuh"
#include "matrixSetting.hpp"

using namespace nvcuda::wmma;

__global__ void compSddmmCoo(const size_t M, const size_t N, const size_t K,
                             const float *matrixA, const float *matrixB, const float *matrixS, float *matrixP) {


    // Compute dense matrix multiplication using Tensor core
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half , row_major> aFrag;
}
