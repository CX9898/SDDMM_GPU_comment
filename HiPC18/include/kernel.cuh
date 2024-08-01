#pragma once

__global__ void compSddmmCoo(const size_t M, const size_t N, const size_t K,
                             const float *matrixA, const float *matrixB, const float *matrixS, float *matrixP);