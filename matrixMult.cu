#include <stdio.h>

#define N 4 // size of matrices

#define THREADS_PER_BLOCK 32

__global__ void matrixMultiply(float *a, float *b, float *c, int n)

{

    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;

    for (int i = 0; i < n; i++)
    {
        sum += a[row * n + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
}

int main()
{

    float *a, *b, *c; // matrices

    float *d_a, *d_b, *d_c;
    a = (float *)malloc(N * N * sizeof(float));

    b = (float *)malloc(N * N * sizeof(float));

    c = (float *)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++)
    {

        for (int j = 0; j < N; j++)
        {

            a[i * N + j] = i * N + j;

            b[i * N + j] = j * N + i;

            c[i * N + j] = 0;
        }
    }
    cudaMalloc(&d_a, N * N * sizeof(float));

    cudaMalloc(&d_b, N * N * sizeof(float));

    cudaMalloc(&d_c, N * N * sizeof(float));

    cudaMemcpy(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridSize((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 blockSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    matrixMultiply<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {

            printf("%f ", c[i * N + j]);
        }
        printf("\n");
    }
    free(a);

    free(b);

    free(c);

    cudaFree(d_a);

    cudaFree(d_b);

    cudaFree(d_c);

    return 0;
}