#include <ctime>
#include <cstdio>
#include <cassert>
#include "immintrin.h"
#include "cuda.h"

#define SAMPLE_SIZE 8
#define MAX_LEN (1<<15)

typedef uint32_t u32;

// nvcc -O3 --compiler-options "-mavx2" compare.cu -o compare
// sbatch -n 8 -N 1 --gpus-per-node=1 -A standby --constraint=K run_test.sh

void avx(__m256i *c, __m256i *a, __m256i *b, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] ^ b[i];
}

__global__
void xor_gpu(u32 *c, u32 *a, u32 *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = a[i] ^ b[i];
}

__global__
void xor_gpu_ptx(u32 *c, u32 *a, u32 *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    asm ("xor.b32 %0, %1, %2;\n\t" : "=r"(c[i]) : "r"(a[i]), "r"(b[i]));
}

void cuda(ulong4 *c, ulong4 *a, ulong4 *b, int n, bool usePtx) {
    if (usePtx) xor_gpu_ptx<<<(8*n+1023)/1024, 1024>>>((u32*)c, (u32*)a, (u32*)b, 8*n);
    else xor_gpu<<<(8*n+1023)/1024, 1024>>>((u32*)c, (u32*)a, (u32*)b, 8*n);
    cudaDeviceSynchronize();
}

int main(int argc, char **argv) {
    struct timespec start, end;
    float duration = 0.0f;

    cudaFree(0);

    __m256i a[MAX_LEN], b[MAX_LEN], c[MAX_LEN];
    memset(a, 0xff, sizeof(a));
    memset(b, 0xff, sizeof(b));

    ulong4 *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sizeof(*d_a) * MAX_LEN);
    cudaMalloc(&d_b, sizeof(*d_b) * MAX_LEN);
    cudaMalloc(&d_c, sizeof(*d_c) * MAX_LEN);
    cudaMemset(d_a, 0xff, sizeof(*d_a) * MAX_LEN);
    cudaMemset(d_b, 0xff, sizeof(*d_b) * MAX_LEN);

    for (int n = 1; n <= MAX_LEN; n <<= 1) {
        printf("n = %d\n", n);

        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int k = 0; k < SAMPLE_SIZE; k++)
            avx(c, a, b, n);
        clock_gettime(CLOCK_MONOTONIC, &end);

        duration = (end.tv_sec - start.tv_sec) * 1000000.0f;
        duration += (end.tv_nsec - start.tv_nsec) / 1000.0f;
        printf("avx: %.2f µs\n", duration / SAMPLE_SIZE);

        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int k = 0; k < SAMPLE_SIZE; k++)
            cuda(d_c, d_a, d_b, n, false);
        clock_gettime(CLOCK_MONOTONIC, &end);

        duration = (end.tv_sec - start.tv_sec) * 1000000.0f;
        duration += (end.tv_nsec - start.tv_nsec) / 1000.0f;
        printf("gpu: %.2f µs\n", duration / SAMPLE_SIZE);

        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int k = 0; k < SAMPLE_SIZE; k++)
            cuda(d_c, d_a, d_b, n, true);
        clock_gettime(CLOCK_MONOTONIC, &end);

        duration = (end.tv_sec - start.tv_sec) * 1000000.0f;
        duration += (end.tv_nsec - start.tv_nsec) / 1000.0f;
        printf("ptx: %.2f µs\n", duration / SAMPLE_SIZE);
    }

    memset(b, 0x0, sizeof(b));
    assert(memcmp(b, c, sizeof(c)) == 0);

    cudaMemcpy(c, d_c, sizeof(c), cudaMemcpyDeviceToHost);
    assert(memcmp(b, c, sizeof(c)) == 0);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
