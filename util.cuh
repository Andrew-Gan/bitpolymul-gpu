#ifndef __UTIL_CUH__
#define __UTIL_CUH__

#include <stdarg.h>
#include "cuda.h"
// #include "cuda_runtime.h"

class u256 {
private:
    const size_t size = 32;
    ulong4 data;

public:
    __host__ __device__ u256() {
        data.x = 0;
        data.y = 0;
        data.z = 0;
        data.w = 0;
    }

    __device__ u256(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
        data.x = a;
        data.y = b;
        data.z = c;
        data.w = d;
    }

    __device__ u256(ulong4 _data) : data(_data) {}

    __device__ u256 operator^(const u256 rhs) const {
        ulong4 res;
        asm (
            "xor.b64     %0, %4, %8;\n\t"
            "xor.b64     %1, %5, %9;\n\t"
            "xor.b64     %2, %6, %10;\n\t"
            "xor.b64     %3, %7, %11;\n\t"
            : "=l"(res.x), "=l"(res.y), "=l"(res.z), "=l"(res.w)
            : "l"(data.x), "l"(data.y), "l"(data.z), "l"(data.w),
            "l"(rhs.data.x), "l"(rhs.data.y), "l"(rhs.data.z), "l"(rhs.data.w)
        );
        return res;
    }

    __device__ u256& operator^=(const u256 rhs) {
        asm (
            "xor.b64     %0, %4, %8;\n\t"
            "xor.b64     %1, %5, %9;\n\t"
            "xor.b64     %2, %6, %10;\n\t"
            "xor.b64     %3, %7, %11;\n\t"
            : "=l"(data.x), "=l"(data.y), "=l"(data.z), "=l"(data.w)
            : "l"(data.x), "l"(data.y), "l"(data.z), "l"(data.w),
            "l"(rhs.data.x), "l"(rhs.data.y), "l"(rhs.data.z), "l"(rhs.data.w)
        );
        return *this;
    }

    // implements:: _mm256_slli_si256
    template <typename T>
    __device__
    u256 slli(int n) const {
        u256 res;
        T *castedData = (T*) &res.data;
        for (int i = 0; i < size / sizeof(T); i++) {
            castedData[i] = ((T*)&data)[i] << n;
        }
        return res;
    }

    // implements: _mm256_srli_si256
    template <typename T>
    __device__
    u256 srli(int n) const {
        u256 res;
        T *castedData = (T*) &res.data;
        for (int i = 0; i < size / sizeof(T); i++) {
            castedData[i] = ((T*)&data)[i] >> n;
        }
        return res;
    }

    // implements: _mm256_permute2x128_si256
    __device__ u256 permute2x128(u256 b, uint8_t control) {
        u256 res;
        switch (control >> 4) {
            case 0b1000: res.data.z = data.x; res.data.w = data.y;
            case 0b0100: res.data.z = data.z; res.data.w = data.w;
            case 0b0010: res.data.z = b.data.x; res.data.w = b.data.y;
            case 0b0001: res.data.z = b.data.z; res.data.w = b.data.w;
        }
        switch ((control & 0b1111)) {
            case 0b1000: res.data.x = data.x; res.data.y = data.y;
            case 0b0100: res.data.x = data.z; res.data.y = data.w;
            case 0b0010: res.data.x = b.data.x; res.data.y = b.data.y;
            case 0b0001: res.data.x = b.data.z; res.data.y = b.data.w;
        }
        return res;
    }

    // implements: _mm256_alignr_epi8
    __device__ u256 alignr(u256 b, uint8_t mask) {
        ulong4 res;

        int offs = 0;
        int n = mask*8;
        if (n > 64) offs++;
        if (n > 128) offs++;

        u256 joint0(b.data.x, b.data.y, data.x, data.y);
        u256 joint1(b.data.z, b.data.w, data.z, data.w);
        uint64_t *joint0Data = (uint64_t*)&joint0.data;
        uint64_t *joint1Data = (uint64_t*)&joint1.data;

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
};

template <typename... Arguments>
__global__
void xor_gpu(u256* poly, int base, Arguments... offsets) {
    int i = base + blockIdx.x * blockDim.x + threadIdx.x;
    for (const int offset : {offsets...}) {
        poly[i] ^= poly[i + offset];
    }
}

template <typename... Arguments>
__global__
void xor_gpu_flat(u256* poly, int iBase, int offset, int offsetStride, Arguments... indices) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int currOffs = offset + y * offsetStride;
    int i = iBase + currOffs + x;
    for (const int index : {indices...}) {
        poly[i] ^= poly[i + index];
    }
}

#endif