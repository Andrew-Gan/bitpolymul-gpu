#ifndef __UTIL_CUH__
#define __UTIL_CUH__

#include <stdarg.h>
#include "cuda.h"
// #include "cuda_runtime.h"

typedef struct ulong4 u256;

__device__ u256 u256_xor(const u256 &lhs, const u256 &rhs) {
    u256 res;
    asm (
        "xor.b64     %0, %4, %8;\n\t"
        "xor.b64     %1, %5, %9;\n\t"
        "xor.b64     %2, %6, %10;\n\t"
        "xor.b64     %3, %7, %11;\n\t"
        : "=l"(res.x), "=l"(res.y), "=l"(res.z), "=l"(res.w)
        : "l"(lhs.x), "l"(lhs.y), "l"(lhs.z), "l"(lhs.w),
        "l"(rhs.x), "l"(rhs.y), "l"(rhs.z), "l"(rhs.w)
    );
    return res;
}

// implements:: _mm256_slli_si256
template <typename T>
__device__ u256 slli(const u256 &obj, int n) {
    u256 res;
    T *casted = (T*) &res;
    for (int i = 0; i < sizeof(u256) / sizeof(T); i++) {
        casted[i] = ((T*)&obj)[i] << n;
    }
    return res;
}

// implements: _mm256_srli_si256
template <typename T>
__device__ u256 srli(const u256 &obj, int n) {
    u256 res;
    T *casted = (T*) &res;
    for (int i = 0; i < sizeof(u256) / sizeof(T); i++) {
        casted[i] = ((T*)&obj)[i] >> n;
    }
    return res;
}

// implements: _mm256_permute2x128_si256
__device__ u256 permute2x128(const u256 &a, const u256 &b, uint8_t control) {
    u256 res;
    switch (control >> 4) {
        case 0b1000: res.z = a.x; res.w = a.y;
        case 0b0100: res.z = a.z; res.w = a.w;
        case 0b0010: res.z = b.x; res.w = b.y;
        case 0b0001: res.z = b.z; res.w = b.w;
    }
    switch ((control & 0b1111)) {
        case 0b1000: res.x = a.x; res.y = a.y;
        case 0b0100: res.x = a.z; res.y = a.w;
        case 0b0010: res.x = b.x; res.y = b.y;
        case 0b0001: res.x = b.z; res.y = b.w;
    }
    return res;
}

// implements: _mm256_alignr_epi8
__device__ u256 alignr(const u256 &a, const u256 &b, uint8_t mask) {
    ulong4 res;

    int offs = 0;
    int n = mask*8;
    if (n > 64) offs++;
    if (n > 128) offs++;

    u256 joint0 = { .x = b.x, .y = b.y, .z = a.x, .w = a.y };
    u256 joint1 = { .x = b.z, .y = b.w, .z = a.z, .w = a.w };
    uint64_t *joint0Data = (uint64_t*)&joint0;
    uint64_t *joint1Data = (uint64_t*)&joint1;

    for (int i = 0; i < 2; i++) {
        ((uint64_t *)&res)[i] |= joint0Data[i+offs] >> n;
        ((uint64_t *)&res)[i] |= joint0Data[i+offs+1] << (64-n);
    }
    for (int i = 0; i < 2; i++) {
        ((uint64_t *)&res)[i+2] |= joint1Data[i+offs] >> n;
        ((uint64_t *)&res)[i+2] |= joint1Data[i+offs+1] << (64-n);
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