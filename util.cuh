#ifndef __UTIL_CUH__
#define __UTIL_CUH__

#include <stdarg.h>
#include "cuda.h"
// #include "cuda_runtime.h"

class u256 {
private:
    const size_t size = 32;
    uint64_t data[4];

public:
    __host__ __device__ u256() {
        memset(data, 0, size);
    }

    __device__ u256(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
        data[0] = a;
        data[1] = b;
        data[2] = c;
        data[3] = d;
    }

    __device__ u256 operator^(const u256 rhs) const {
        u256 res;
        for (int i = 0; i < 4; i++) {
            res.data[i] = data[i] ^ rhs.data[i];
        }
        return res;
    }

    __device__ u256& operator^=(const u256 rhs) {
        for (int i = 0; i < 4; i++) {
            data[i] ^= rhs.data[i];
        }
        return *this;
    }

    // implements:: _mm256_slli_si256
    template <typename T>
    __device__
    u256 slli(int n) const {
        u256 res;
        T *castedData = (T*) res.data;
        for (int i = 0; i < size / sizeof(T); i++) {
            castedData[i] = ((T*)data)[i] << n;
        }
        return res;
    }

    // implements: _mm256_srli_si256
    template <typename T>
    __device__
    u256 srli(int n) const {
        u256 res;
        T *castedData = (T*) res.data;
        for (int i = 0; i < size / sizeof(T); i++) {
            castedData[i] = ((T*)data)[i] >> n;
        }
        return res;
    }

    // implements: _mm256_permute2x128_si256
    __device__ u256 permute2x128(u256 b, uint8_t control) {
        u256 res;
        switch (control >> 4) {
            case 0b1000: res.data[2] = data[0]; res.data[3] = data[1];
            case 0b0100: res.data[2] = data[2]; res.data[3] = data[3];
            case 0b0010: res.data[2] = b.data[0]; res.data[3] = b.data[1];
            case 0b0001: res.data[2] = b.data[2]; res.data[3] = b.data[3];
        }
        switch ((control & 0b1111)) {
            case 0b1000: res.data[0] = data[0]; res.data[1] = data[1];
            case 0b0100: res.data[0] = data[2]; res.data[1] = data[3];
            case 0b0010: res.data[0] = b.data[0]; res.data[1] = b.data[1];
            case 0b0001: res.data[0] = b.data[2]; res.data[1] = b.data[3];
        }
        return res;
    }

    // implements: _mm256_alignr_epi8
    __device__ u256 alignr(u256 b, uint8_t mask) {
        u256 res;

        int offs = 0;
        int n = mask*8;
        if (n > 64) offs++;
        if (n > 128) offs++;

        u256 joint0(b.data[0], b.data[1], data[0], data[1]);
        u256 joint1(b.data[2], b.data[3], data[2], data[3]);

        for (int i = 0; i < 2; i++) {
            res.data[i] |= joint0.data[i+offs] >> n;
            res.data[i] |= joint0.data[i+offs+1] << (64-n);
        }
        for (int i = 0; i < 2; i++) {
            res.data[i+2] |= joint1.data[i+offs] >> n;
            res.data[i+2] |= joint1.data[i+offs+1] << (64-n);
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