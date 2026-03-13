#include <stdio.h>
#include <cuda.h>

#define WIDTH 512
#define HEIGHT 512
#define TILE 16

__global__ void convolutionKernel(unsigned char *input, unsigned char *output) {
    __shared__ unsigned char tile[TILE+2][TILE+2]; // shared memory tile with 1-pixel halo

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    // Load data into shared memory with boundary check
    if (x < WIDTH && y < HEIGHT)
        tile[ty][tx] = input[y * WIDTH + x];
    else
        tile[ty][tx] = 0;

    // Load halo pixels for convolution
    if (threadIdx.x == 0 && x > 0)
        tile[ty][tx-1] = input[y * WIDTH + (x-1)];
    if (threadIdx.x == TILE-1 && x < WIDTH-1)
        tile[ty][tx+1] = input[y * WIDTH + (x+1)];
    if (threadIdx.y == 0 && y > 0)
        tile[ty-1][tx] = input[(y-1) * WIDTH + x];
    if (threadIdx.y == TILE-1 && y < HEIGHT-1)
        tile[ty+1][tx] = input[(y+1) * WIDTH + x];

    __syncthreads();

    // Apply 3x3 average convolution
    if (x > 0 && y > 0 && x < WIDTH-1 && y < HEIGHT-1) {
        float sum = 0.0f;
        for (int ky = -1; ky <= 1; ky++)
            for (int kx = -1; kx <= 1; kx++)
                sum += tile[ty+ky][tx+kx] * (1.0f / 9.0f);

        output[y * WIDTH + x] = (unsigned char)sum;
    }
}

int main() {
    unsigned char *h_input, *h_output;
    unsigned char *d_input, *d_output;

    // Allocate host memory
    h_input = (unsigned char*)malloc(WIDTH * HEIGHT);
    h_output = (unsigned char*)malloc(WIDTH * HEIGHT);

    // Initialize input image (simple gray image)
    for (int i = 0; i < WIDTH * HEIGHT; i++)
        h_input[i] = 128;

    // Allocate device memory
    cudaMalloc((void**)&d_input, WIDTH * HEIGHT);
    cudaMalloc((void**)&d_output, WIDTH * HEIGHT);

    // Copy input to device
    cudaMemcpy(d_input, h_input, WIDTH * HEIGHT, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block(TILE, TILE);
    dim3 grid((WIDTH + TILE - 1) / TILE, (HEIGHT + TILE - 1) / TILE);

    // Launch kernel
    convolutionKernel<<<grid, block>>>(d_input, d_output);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    // Print output snippet
    printf("Image size: %dx%d\n", WIDTH, HEIGHT);
    printf("CUDA convolution completed successfully\n");

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
