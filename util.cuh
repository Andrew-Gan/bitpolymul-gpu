#include <stdarg.h>
#include "cuda.h"
#include "cuda_runtime.h"

typedef struct _u256 {
private:
    const size_t size = 32;
    uint64_t data[4];

public:
    __device__ _u256& operator^=(_u256 &rhs) {
        for (int i = 0; i < 4; i++) {
            data[i] ^= rhs.data[i];
        }
    }

    // implements:: _mm256_slli_si256
    template <typename T>
    __device__ _u256& slli(int n) {
        _u256 res;
        T *castedData = (T*) res.data;
        for (int i = 0; i < size / sizeof(T); i++) {
            castedData[i] << n;
        }
        return res;
    }

    // implements: _mm256_srli_si256
    template <typename T>
    __device__ _u256& srli(int n) {
        _u256 res;
        T *castedData = (T*) res.data;
        for (int i = 0; i < size / sizeof(T); i++) {
            castedData[i] >> n;
        }
        return res;
    }

    // implements: _mm256_permute2x128_si256
    __device__ _u256 permute2x128(_u256 b, uint8_t control) {
        _u256 res;
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
    __device__ _u256 alignr(_u256 b, uint8_t mask) {
        _u256 res;

        int offs = 0;
        int n = mask*8;
        if (n > 64) offs++;
        if (n > 128) offs++;

        _u256 joint0, joint1;
        joint0.data = {b.data[0], b.data[1], data[0], data[1]};
        joint1.data = {b.data[2], b.data[3], data[2], data[3]};

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
} u256;

template <typename... Arguments>
__global__
void xor_gpu(u256* poly, int base, Arguments... offsets) {
    int i = base + blockIdx.x * blockDim.x + threadIdx.x;
    for (const int offset : {offsets...}) {
        poly[i] ^= poly[i + offset];
    }
}

// __global__ static
// void xor_gpu(u256* poly, int base, int offs) {
//     int i = base + blockIdx.x * blockDim.x + threadIdx.x;
//     poly[i] ^= poly[i+offs];
// }

// __global__ static
// void xor_gpu(u256* poly, int base, int offs1, int offs2) {
//     int i = base + blockIdx.x * blockDim.x + threadIdx.x;
//     poly[i] ^= poly[i+offs1] ^ poly[i+offs2];
// }

// __global__ static
// void xor_gpu(u256* poly, int base, int offs1, int offs2, int offs3) {
//     int i = base + blockIdx.x * blockDim.x + threadIdx.x;
//     poly[i] ^= poly[i+offs1] ^ poly[i+offs2] ^ poly[i+offs3];
// }