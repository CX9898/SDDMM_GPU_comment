#pragma once

__global__ void compSddmmCoo(const int M, const int N, const int K,
                             const half *matrixA, const half *matrixB,
                             const float *matrixS,
                             float *matrixP);