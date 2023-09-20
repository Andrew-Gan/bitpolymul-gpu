#include <stdarg.h>
#include "cuda.h"
#include "cuda_runtime.h"

typedef struct _u256 {
private:
    uint64_t data[4];
public:
    __device__ _u256& operator^=(_u256 &rhs) {
        for (int i = 0; i < 4; i++) {
            data[i] ^= rhs.data[i];
        }
    }

    __device__ _u256& operator>>(int n) {
        if (n <= 0) return;
        if (n >= 256) memset(data, 0, 4 * sizeof(*data));

        int offs = 0;

        if (n > 64) offs++;
        if (n > 128) offs++;

        uint64_t buffer;

        for (int i = 0; i < 4; i++) {
            buffer = 0;
            if (i+offs < 4) buffer |= data[i+offs] >> n;
            if (i+offs+1 < 4) buffer |= data[i+offs+1] << (64-n);
            data[i] = buffer;
        }
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