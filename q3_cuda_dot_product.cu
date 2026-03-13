#include <stdio.h>
#include <cuda.h>

#define N (1 << 20)
#define THREADS 256

__global__ void dotProductKernel(float *A, float *B, float *partial) {
    __shared__ float cache[THREADS];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0f;
    while (tid < N) {
        temp += A[tid] * B[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (cacheIndex < s)
            cache[cacheIndex] += cache[cacheIndex + s];
        __syncthreads();
    }

    if (cacheIndex == 0)
        partial[blockIdx.x] = cache[0];
}

int main() {
    float *h_A, *h_B, *h_partial;
    float *d_A, *d_B, *d_partial;

    int blocks = 32;

    h_A = (float*)malloc(N * sizeof(float));
    h_B = (float*)malloc(N * sizeof(float));
    h_partial = (float*)malloc(blocks * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_partial, blocks * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    dotProductKernel<<<blocks, THREADS>>>(d_A, d_B, d_partial);

    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float dot = 0.0f;
    for (int i = 0; i < blocks; i++)
        dot += h_partial[i];

    printf("Vector size: %d\n", N);
    printf("CUDA Dot Product Result: %f\n", dot);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partial);
    free(h_A);
    free(h_B);
    free(h_partial);

    return 0;
}
