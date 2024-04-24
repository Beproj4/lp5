#include <stdio.h>
#include "device_launch_parameters.h"

__global__ void add_vectors(int *a, int *b, int *c, int n)

{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n)
    {

        c[index] = a[index] + b[index];
    }
}

int main()
{
    int n = 1000;

    int *a, *b, *c;       // host arrays
    int *d_a, *d_b, *d_c; // device arrays

    int size = n * sizeof(int);

    a = (int *)malloc(size);

    b = (int *)malloc(size);

    c = (int *)malloc(size);
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
        b[i] = 2 * i;
    }

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    int threads_per_block = 256;

    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    add_vectors<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, n);
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);

    cudaFree(d_b);

    cudaFree(d_c);

    for (int i = 0; i < n; i++)
    {
        printf("%d ", c[i]);
    }

    free(a);

    free(b);

    free(c);

    return 0;
}
