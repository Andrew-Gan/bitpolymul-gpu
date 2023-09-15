#include <initializer_list>

typedef struct u256 {
private:
    data[4];
public:
    __device__ u256& operator^(u256 &rhs) {
        for (int i = 0; i < 4; i++) {
            data[i] ^= rhs.data[i];
        }
    }
};

__global__ static
void xor_gpu(u256* poly, int base, std::initializer_list<int> offs) {
    int i = base + blockIdx.x * blockDim.x + threadIdx.x;
    for (auto off : offs) {
        poly[i] ^= poly[i + off];
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