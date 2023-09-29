#ifndef __UTIL_CUH__
#define __UTIL_CUH__

#include <stdarg.h>
#include "cuda.h"
// #include "cuda_runtime.h"

typedef struct ulong2 u128;
typedef struct ulong4 u256;

__device__ u128 xor128(const u128 &lhs, const u128 &rhs);

__device__ u256 u256_xor(const u256 &lhs, const u256 &rhs);

// implement: _mm_slli_si128
__device__ u128 slli_si128(const u128 &a, uint8_t imm8);

// implement: _mm_srli_si128
__device__ u128 srli_si128(const u128 &a, uint8_t imm8);

// implement: _mm_shuffle_epi32
__device__ u128 shuffle_epi32(const u128 &a, uint8_t imm8);

// implement: _mm_clmulepi64_si128
__device__ u128 clmul(const u128 &a, const u128 &b, uint8_t imm8);

// implement: _mm256_permute2x128_si256
__device__ u256 permute2x128(const u256 &a, const u256 &b, int control);

// implement: _mm256_alignr_epi8
__device__ u256 alignr(const u256 &a, const u256 &b, int mask);

__device__
u128 _gf2ext128_reduce_gpu( u128 x0 , u128 x128 );

__global__
void gf2ext128_mul_gpu( uint8_t * c , const uint8_t * a , const uint8_t * b );

// implement:: _mm256_slli_si256
template <typename T>
__device__ u256 slli(const u256 &obj, int n) {
    u256 res;
    T *casted = (T*) &res;
    for (int i = 0; i < sizeof(u256) / sizeof(T); i++) {
        casted[i] = ((T*)&obj)[i] << n;
    }
    return res;
}

// implement: _mm256_srli_si256
template <typename T>
__device__ u256 srli(const u256 &obj, int n) {
    u256 res;
    T *casted = (T*) &res;
    for (int i = 0; i < sizeof(u256) / sizeof(T); i++) {
        casted[i] = ((T*)&obj)[i] >> n;
    }
    return res;
}

template <typename... Arguments>
__global__
void xor_gpu(u256* poly, int iBase, int offset, int offsetStride, Arguments... indices) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    poly += z * (1<<16);

    int currOffs = offset + y * offsetStride;
    int i = iBase + currOffs + x;
    for (const int index : {indices...}) {
        poly[i] = u256_xor(poly[i], poly[i + index]);
    }
}

#endif