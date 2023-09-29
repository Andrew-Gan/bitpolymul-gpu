#include "util.h"

__device__ u128 xor128(const u128 &lhs, const u128 &rhs) {
    u128 res;
    asm (
        "xor.b64     %0, %2, %4;\n\t"
        "xor.b64     %1, %3, %5;\n\t"
        : "=l"(res.x), "=l"(res.y)
        : "l"(lhs.x), "l"(lhs.y),
        "l"(rhs.x), "l"(rhs.y)
    );
    return res;
}

__device__ void xor128_inplace(u128 &lhs, const u128 &rhs) {
    asm (
        "xor.b64     %0, %2, %4;\n\t"
        "xor.b64     %1, %3, %5;\n\t"
        : "=l"(lhs.x), "=l"(lhs.y)
        : "l"(lhs.x), "l"(lhs.y),
        "l"(rhs.x), "l"(rhs.y)
    );
}

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

// implement: _mm_slli_si128
__device__ u128 slli_si128(const u128 &a, uint8_t imm8) {
    u128 res = {.x = 0, .y = 0};
    if (imm8 >= 16) return res;
    uint8_t *a8 = (uint8_t*)&a;
    for (int i = 15; i >= imm8; i--) {
        ((uint8_t*)&res)[i] = a8[i - imm8];
    }
    return res;
}

// implement: _mm_srli_si128
__device__ u128 srli_si128(const u128 &a, uint8_t imm8) {
    u128 res = {.x = 0, .y = 0};
    if (imm8 >= 16) return res;
    uint8_t *a8 = (uint8_t*)&a;
    for (int i = 0; i < 16-imm8; i++) {
        ((uint8_t*)&res)[i] = a8[i + imm8];
    }
    return res;
}

// implement: _mm_shuffle_epi32
__device__ u128 shuffle_epi32(const u128 &a, uint8_t imm8) {
    u128 res;
    uint32_t *a32 = (uint32_t*)&a;
    uint32_t *res32 = (uint32_t*)&res;
    res32[0] = a32[imm8 & 0b11];
    res32[1] = a32[imm8>>2 & 0b11];
    res32[2] = a32[imm8>>4 & 0b11];
    res32[3] = a32[imm8>>6 & 0b11];
    return res;
}

// implement: _mm_clmulepi64_si128
__device__ u128 clmul(const u128 &a, const u128 &b, uint8_t imm8) {
    uint64_t a64 = imm8 & 0b00001 ? *(uint64_t*)&(a.y) : *(uint64_t*)&(a.x);
    uint64_t b64 = imm8 & 0b10000 ? *(uint64_t*)&(b.y) : *(uint64_t*)&(b.x);
    u128 c128 = {.x = 0, .y = 0};

    for (int i = 0; i < 64; i++) {
        if (a64 & 1) {
            c128.x ^= b64 << i;
            c128.y ^= b64 >> (64-i);
        }
        a64 >>= 1;
    }

    return c128;
}

// implement: _mm256_permute2x128_si256
__device__ u256 permute2x128(const u256 &a, const u256 &b, int control) {
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

// implement: _mm256_alignr_epi8
__device__ u256 alignr(const u256 &a, const u256 &b, int mask) {
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

__global__
void gf2ext128_mul_gpu(uint8_t *c, const uint8_t *a, const uint8_t *b) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c += i*16;
	a += i*16;
	b += i*16;

	u128 a0 = *(u128 const *)a;
	u128 b0 = *(u128 const *)b;
	u128 c0, c128;

    //
    c0 = clmul(a0, b0, 0x00);
    c128 = clmul(a0, b0, 0x11);

    //
    xor128_inplace(a0, srli_si128(a0, 8));
    xor128_inplace(b0, srli_si128(b0, 8));
    u128 _tt2 = xor128(clmul( a0, b0 , 0 ), xor128(c0, c128));
    xor128_inplace(c0, slli_si128( _tt2 , 8 ));
    xor128_inplace(c128, slli_si128( _tt2 , 8 ));

    //
    const u128 reducer = { .x = 0x87ULL, .y = 0x0ULL };
	u128 c64 = clmul( c128 , reducer , 1 );
	xor128_inplace(c128, shuffle_epi32(c64, 0xfe));
	xor128_inplace(c0, shuffle_epi32(c64, 0x4f));
	xor128_inplace(c0, clmul(c128, reducer, 0));

	memcpy(c, &c0, sizeof(c0));
}
