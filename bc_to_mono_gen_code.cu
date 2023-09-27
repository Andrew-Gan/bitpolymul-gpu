/*
Copyright (C) 2018 Wen-Ding Li

This file is part of BitPolyMul.

BitPolyMul is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

BitPolyMul is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with BitPolyMul.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "util.cuh"

__global__
void bc_to_mono_256_16_a(u256* poly) {
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t offset = 4 * x + 2;
    poly[offset-1] = u256_xor(poly[offset-1], poly[offset]);
    poly[offset] = u256_xor(poly[offset], poly[offset+1]);
}

void bc_to_mono_256_16(u256* poly, int logn) {
    uint64_t nBlock = (((1<<logn)-2)/4+1023)/1024;
    bc_to_mono_256_16_a<<<nBlock, 1024>>>(poly);

    for(int offset=(1<<2);offset<(1<<logn);offset+=(1<<(2+1))) {
        // for(int i=offset-3;i<=offset-1-0;++i)poly[i]^=poly[i+3];
        // for(int i=offset+(1<<2)-4;i<=offset+(1<<2)-1-3;++i)poly[i]^=poly[i+3];
        // xor_gpu<<<1, 3>>>(poly, offset-3, 3);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<2)-4, 3);
    }

    for(int offset=(1<<3);offset<(1<<logn);offset+=(1<<(3+1))) {
        // for(int i=offset-7;i<=offset-1-6;++i)poly[i]^=poly[i+7];
        // for(int i=offset-6;i<=offset-1-4;++i)poly[i]^=poly[i+6]^poly[i+7];
        // for(int i=offset-4;i<=offset-1-0;++i)poly[i]^=poly[i+4]^poly[i+6]^poly[i+7];
        // for(int i=offset+(1<<3)-8;i<=offset+(1<<3)-1-7;++i)poly[i]^=poly[i+4]^poly[i+6]^poly[i+7];
        // for(int i=offset+(1<<3)-7;i<=offset+(1<<3)-1-6;++i)poly[i]^=poly[i+4]^poly[i+6];
        // for(int i=offset+(1<<3)-6;i<=offset+(1<<3)-1-4;++i)poly[i]^=poly[i+4];
        // xor_gpu<<<1, 1>>>(poly, offset-7, 7);
        // xor_gpu<<<1, 2>>>(poly, offset-6, 6, 7);
        // xor_gpu<<<1, 4>>>(poly, offset-4, 4, 6, 7);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<3)-8, 4, 6, 7);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<3)-7, 4, 6);
        // xor_gpu<<<1, 2>>>(poly, offset+(1<<3)-6, 4);
    }

    for(int offset=(1<<4);offset<(1<<logn);offset+=(1<<(4+1))) {
        // for(int i=offset-15;i<=offset-1-0;++i)poly[i]^=poly[i+15];
        // for(int i=offset+(1<<4)-16;i<=offset+(1<<4)-1-15;++i)poly[i]^=poly[i+15];
        // xor_gpu<<<1, 15>>>(poly, offset-15, 15);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<4)-16, 15);
    }

    for(int offset=(1<<5);offset<(1<<logn);offset+=(1<<(5+1))) {
        // for(int i=offset-31;i<=offset-1-30;++i)poly[i]^=poly[i+31];
        // for(int i=offset-30;i<=offset-1-16;++i)poly[i]^=poly[i+30]^poly[i+31];
        // for(int i=offset-16;i<=offset-1-0;++i)poly[i]^=poly[i+16]^poly[i+30]^poly[i+31];
        // for(int i=offset+(1<<5)-32;i<=offset+(1<<5)-1-31;++i)poly[i]^=poly[i+16]^poly[i+30]^poly[i+31];
        // for(int i=offset+(1<<5)-31;i<=offset+(1<<5)-1-30;++i)poly[i]^=poly[i+16]^poly[i+30];
        // for(int i=offset+(1<<5)-30;i<=offset+(1<<5)-1-16;++i)poly[i]^=poly[i+16];
        // xor_gpu<<<1, 1>>>(poly, offset-31, 31);
        // xor_gpu<<<1, 14>>>(poly, offset-30, 30, 31);
        // xor_gpu<<<1, 16>>>(poly, offset-16, 16, 30, 31);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<5)-32, 16, 30, 31);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<5)-31, 16, 30);
        // xor_gpu<<<1, 14>>>(poly, offset+(1<<5)-30, 16);
    }

    for(int offset=(1<<6);offset<(1<<logn);offset+=(1<<(6+1))) {
        // for(int i=offset-63;i<=offset-1-60;++i)poly[i]^=poly[i+63];
        // for(int i=offset-60;i<=offset-1-48;++i)poly[i]^=poly[i+60]^poly[i+63];
        // for(int i=offset-48;i<=offset-1-0;++i)poly[i]^=poly[i+48]^poly[i+60]^poly[i+63];
        // for(int i=offset+(1<<6)-64;i<=offset+(1<<6)-1-63;++i)poly[i]^=poly[i+48]^poly[i+60]^poly[i+63];
        // for(int i=offset+(1<<6)-63;i<=offset+(1<<6)-1-60;++i)poly[i]^=poly[i+48]^poly[i+60];
        // for(int i=offset+(1<<6)-60;i<=offset+(1<<6)-1-48;++i)poly[i]^=poly[i+48];
        // xor_gpu<<<1, 3>>>(poly, offset-63, 63);
        // xor_gpu<<<1, 12>>>(poly, offset-60, 60, 63);
        // xor_gpu<<<1, 48>>>(poly, offset-48, 48, 60, 63);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<6)-64, 48, 60, 63);
        // xor_gpu<<<1, 3>>>(poly, offset+(1<<6)-63, 48, 60);
        // xor_gpu<<<1, 12>>>(poly, offset+(1<<6)-60, 48);
    }

    for(int offset=(1<<7);offset<(1<<logn);offset+=(1<<(7+1))) {
        // for(int i=offset-127;i<=offset-1-126;++i)poly[i]^=poly[i+127];
        // for(int i=offset-126;i<=offset-1-124;++i)poly[i]^=poly[i+126]^poly[i+127];
        // for(int i=offset-124;i<=offset-1-120;++i)poly[i]^=poly[i+124]^poly[i+126]^poly[i+127];
        // for(int i=offset-120;i<=offset-1-112;++i)poly[i]^=poly[i+120]^poly[i+124]^poly[i+126]^poly[i+127];
        // for(int i=offset-112;i<=offset-1-96;++i)poly[i]^=poly[i+112]^poly[i+120]^poly[i+124]^poly[i+126]^poly[i+127];
        // for(int i=offset-96;i<=offset-1-64;++i)poly[i]^=poly[i+96]^poly[i+112]^poly[i+120]^poly[i+124]^poly[i+126]^poly[i+127];
        // for(int i=offset-64;i<=offset-1-0;++i)poly[i]^=poly[i+64]^poly[i+96]^poly[i+112]^poly[i+120]^poly[i+124]^poly[i+126]^poly[i+127];
        // for(int i=offset+(1<<7)-128;i<=offset+(1<<7)-1-127;++i)poly[i]^=poly[i+64]^poly[i+96]^poly[i+112]^poly[i+120]^poly[i+124]^poly[i+126]^poly[i+127];
        // for(int i=offset+(1<<7)-127;i<=offset+(1<<7)-1-126;++i)poly[i]^=poly[i+64]^poly[i+96]^poly[i+112]^poly[i+120]^poly[i+124]^poly[i+126];
        // for(int i=offset+(1<<7)-126;i<=offset+(1<<7)-1-124;++i)poly[i]^=poly[i+64]^poly[i+96]^poly[i+112]^poly[i+120]^poly[i+124];
        // for(int i=offset+(1<<7)-124;i<=offset+(1<<7)-1-120;++i)poly[i]^=poly[i+64]^poly[i+96]^poly[i+112]^poly[i+120];
        // for(int i=offset+(1<<7)-120;i<=offset+(1<<7)-1-112;++i)poly[i]^=poly[i+64]^poly[i+96]^poly[i+112];
        // for(int i=offset+(1<<7)-112;i<=offset+(1<<7)-1-96;++i)poly[i]^=poly[i+64]^poly[i+96];
        // for(int i=offset+(1<<7)-96;i<=offset+(1<<7)-1-64;++i)poly[i]^=poly[i+64];
        // xor_gpu<<<1, 1>>>(poly, offset-127, 127);
        // xor_gpu<<<1, 2>>>(poly, offset-126, 126, 127);
        // xor_gpu<<<1, 4>>>(poly, offset-124, 124, 126, 127);
        // xor_gpu<<<1, 8>>>(poly, offset-120, 120, 124, 126, 127);
        // xor_gpu<<<1, 16>>>(poly, offset-112, 112, 120, 124, 126, 127);
        // xor_gpu<<<1, 32>>>(poly, offset-96, 96, 112, 120, 124, 126, 127);
        // xor_gpu<<<1, 64>>>(poly, offset-64, 64, 96, 112, 120, 124, 126, 127);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<7)-128, 64, 96, 112, 120, 124, 126, 127);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<7)-127, 64, 96, 112, 120, 124, 126);
        // xor_gpu<<<1, 2>>>(poly, offset+(1<<7)-126, 64, 96, 112, 120, 124);
        // xor_gpu<<<1, 4>>>(poly, offset+(1<<7)-124, 64, 96, 112, 120);
        // xor_gpu<<<1, 8>>>(poly, offset+(1<<7)-120, 64, 96, 112);
        // xor_gpu<<<1, 16>>>(poly, offset+(1<<7)-112, 64, 96);
        // xor_gpu<<<1, 32>>>(poly, offset+(1<<7)-96, 64);
    }

    for(int offset=(1<<8);offset<(1<<logn);offset+=(1<<(8+1))) {
        // for(int i=offset-255;i<=offset-1-0;++i)poly[i]^=poly[i+255];
        // for(int i=offset+(1<<8)-256;i<=offset+(1<<8)-1-255;++i)poly[i]^=poly[i+255];
        // xor_gpu<<<1, 255>>>(poly, offset-255, 255);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<8)-256, 255);
    }

    for(int offset=(1<<9);offset<(1<<logn);offset+=(1<<(9+1))) {
        // for(int i=offset-511;i<=offset-1-510;++i)poly[i]^=poly[i+511];
        // for(int i=offset-510;i<=offset-1-256;++i)poly[i]^=poly[i+510]^poly[i+511];
        // for(int i=offset-256;i<=offset-1-0;++i)poly[i]^=poly[i+256]^poly[i+510]^poly[i+511];
        // for(int i=offset+(1<<9)-512;i<=offset+(1<<9)-1-511;++i)poly[i]^=poly[i+256]^poly[i+510]^poly[i+511];
        // for(int i=offset+(1<<9)-511;i<=offset+(1<<9)-1-510;++i)poly[i]^=poly[i+256]^poly[i+510];
        // for(int i=offset+(1<<9)-510;i<=offset+(1<<9)-1-256;++i)poly[i]^=poly[i+256];
        // xor_gpu<<<1, 1>>>(poly, offset-511, 511);
        // xor_gpu<<<1, 254>>>(poly, offset-510, 510, 511);
        // xor_gpu<<<1, 256>>>(poly, offset-256, 256, 510, 511);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<9)-512, 256, 510, 511);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<9)-511, 256, 510);
        // xor_gpu<<<1, 254>>>(poly, offset+(1<<9)-510, 256);
    }

    for(int offset=(1<<10);offset<(1<<logn);offset+=(1<<(10+1))) {
        // for(int i=offset-1023;i<=offset-1-1020;++i)poly[i]^=poly[i+1023];
        // for(int i=offset-1020;i<=offset-1-768;++i)poly[i]^=poly[i+1020]^poly[i+1023];
        // for(int i=offset-768;i<=offset-1-0;++i)poly[i]^=poly[i+768]^poly[i+1020]^poly[i+1023];
        // for(int i=offset+(1<<10)-1024;i<=offset+(1<<10)-1-1023;++i)poly[i]^=poly[i+768]^poly[i+1020]^poly[i+1023];
        // for(int i=offset+(1<<10)-1023;i<=offset+(1<<10)-1-1020;++i)poly[i]^=poly[i+768]^poly[i+1020];
        // for(int i=offset+(1<<10)-1020;i<=offset+(1<<10)-1-768;++i)poly[i]^=poly[i+768];
        // xor_gpu<<<1, 3>>>(poly, offset-1023, 1023);
        // xor_gpu<<<1, 252>>>(poly, offset-1020, 1020, 1023);
        // xor_gpu<<<1, 768>>>(poly, offset-768, 768, 1020, 1023);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<10)-1024, 768, 1020, 1023);
        // xor_gpu<<<1, 3>>>(poly, offset+(1<<10)-1023, 768, 1020);
        // xor_gpu<<<1, 252>>>(poly, offset+(1<<10)-1020, 768);
    }

    for(int offset=(1<<11);offset<(1<<logn);offset+=(1<<(11+1))) {
        // for(int i=offset-2047;i<=offset-1-2046;++i)poly[i]^=poly[i+2047];
        // for(int i=offset-2046;i<=offset-1-2044;++i)poly[i]^=poly[i+2046]^poly[i+2047];
        // for(int i=offset-2044;i<=offset-1-2040;++i)poly[i]^=poly[i+2044]^poly[i+2046]^poly[i+2047];
        // for(int i=offset-2040;i<=offset-1-1792;++i)poly[i]^=poly[i+2040]^poly[i+2044]^poly[i+2046]^poly[i+2047];
        // for(int i=offset-1792;i<=offset-1-1536;++i)poly[i]^=poly[i+1792]^poly[i+2040]^poly[i+2044]^poly[i+2046]^poly[i+2047];
        // for(int i=offset-1536;i<=offset-1-1024;++i)poly[i]^=poly[i+1536]^poly[i+1792]^poly[i+2040]^poly[i+2044]^poly[i+2046]^poly[i+2047];
        // for(int i=offset-1024;i<=offset-1-0;++i)poly[i]^=poly[i+1024]^poly[i+1536]^poly[i+1792]^poly[i+2040]^poly[i+2044]^poly[i+2046]^poly[i+2047];
        // for(int i=offset+(1<<11)-2048;i<=offset+(1<<11)-1-2047;++i)poly[i]^=poly[i+1024]^poly[i+1536]^poly[i+1792]^poly[i+2040]^poly[i+2044]^poly[i+2046]^poly[i+2047];
        // for(int i=offset+(1<<11)-2047;i<=offset+(1<<11)-1-2046;++i)poly[i]^=poly[i+1024]^poly[i+1536]^poly[i+1792]^poly[i+2040]^poly[i+2044]^poly[i+2046];
        // for(int i=offset+(1<<11)-2046;i<=offset+(1<<11)-1-2044;++i)poly[i]^=poly[i+1024]^poly[i+1536]^poly[i+1792]^poly[i+2040]^poly[i+2044];
        // for(int i=offset+(1<<11)-2044;i<=offset+(1<<11)-1-2040;++i)poly[i]^=poly[i+1024]^poly[i+1536]^poly[i+1792]^poly[i+2040];
        // for(int i=offset+(1<<11)-2040;i<=offset+(1<<11)-1-1792;++i)poly[i]^=poly[i+1024]^poly[i+1536]^poly[i+1792];
        // for(int i=offset+(1<<11)-1792;i<=offset+(1<<11)-1-1536;++i)poly[i]^=poly[i+1024]^poly[i+1536];
        // for(int i=offset+(1<<11)-1536;i<=offset+(1<<11)-1-1024;++i)poly[i]^=poly[i+1024];
        // xor_gpu<<<1, 1>>>(poly, offset-2047, 2047);
        // xor_gpu<<<1, 2>>>(poly, offset-2046, 2046, 2047);
        // xor_gpu<<<1, 4>>>(poly, offset-2044, 2044, 2046, 2047);
        // xor_gpu<<<1, 248>>>(poly, offset-2040, 2040, 2044, 2046, 2047);
        // xor_gpu<<<1, 256>>>(poly, offset-1792, 1792, 2040, 2044, 2046, 2047);
        // xor_gpu<<<1, 512>>>(poly, offset-1536, 1536, 1792, 2040, 2044, 2046, 2047);
        // xor_gpu<<<1, 1024>>>(poly, offset-1024, 1024, 1536, 1792, 2040, 2044, 2046, 2047);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<11)-2048, 1024, 1536, 1792, 2040, 2044, 2046, 2047);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<11)-2047, 1024, 1536, 1792, 2040, 2044, 2046);
        // xor_gpu<<<1, 2>>>(poly, offset+(1<<11)-2046, 1024, 1536, 1792, 2040, 2044);
        // xor_gpu<<<1, 4>>>(poly, offset+(1<<11)-2044, 1024, 1536, 1792, 2040);
        // xor_gpu<<<1, 248>>>(poly, offset+(1<<11)-2040, 1024, 1536, 1792);
        // xor_gpu<<<1, 256>>>(poly, offset+(1<<11)-1792, 1024, 1536);
        // xor_gpu<<<1, 512>>>(poly, offset+(1<<11)-1536, 1024);
    }

    for(int offset=(1<<12);offset<(1<<logn);offset+=(1<<(12+1))) {
        // for(int i=offset-4095;i<=offset-1-4080;++i)poly[i]^=poly[i+4095];
        // for(int i=offset-4080;i<=offset-1-3840;++i)poly[i]^=poly[i+4080]^poly[i+4095];
        // for(int i=offset-3840;i<=offset-1-0;++i)poly[i]^=poly[i+3840]^poly[i+4080]^poly[i+4095];
        // for(int i=offset+(1<<12)-4096;i<=offset+(1<<12)-1-4095;++i)poly[i]^=poly[i+3840]^poly[i+4080]^poly[i+4095];
        // for(int i=offset+(1<<12)-4095;i<=offset+(1<<12)-1-4080;++i)poly[i]^=poly[i+3840]^poly[i+4080];
        // for(int i=offset+(1<<12)-4080;i<=offset+(1<<12)-1-3840;++i)poly[i]^=poly[i+3840];
        // xor_gpu<<<1, 15>>>(poly, offset-4095, 4095);
        // xor_gpu<<<1, 240>>>(poly, offset-4080, 4080, 4095);
        // xor_gpu<<<15, 256>>>(poly, offset-3840, 3840, 4080, 4095);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<12)-4096, 3840, 4080, 4095);
        // xor_gpu<<<1, 15>>>(poly, offset+(1<<12)-4095, 3840, 4080);
        // xor_gpu<<<1, 240>>>(poly, offset+(1<<12)-4080, 3840);
    }

    for(int offset=(1<<13);offset<(1<<logn);offset+=(1<<(13+1))) {
        // for(int i=offset-8191;i<=offset-1-8190;++i)poly[i]^=poly[i+8191];
        // for(int i=offset-8190;i<=offset-1-8176;++i)poly[i]^=poly[i+8190]^poly[i+8191];
        // for(int i=offset-8176;i<=offset-1-8160;++i)poly[i]^=poly[i+8176]^poly[i+8190]^poly[i+8191];
        // for(int i=offset-8160;i<=offset-1-7936;++i)poly[i]^=poly[i+8160]^poly[i+8176]^poly[i+8190]^poly[i+8191];
        // for(int i=offset-7936;i<=offset-1-7680;++i)poly[i]^=poly[i+7936]^poly[i+8160]^poly[i+8176]^poly[i+8190]^poly[i+8191];
        // for(int i=offset-7680;i<=offset-1-4096;++i)poly[i]^=poly[i+7680]^poly[i+7936]^poly[i+8160]^poly[i+8176]^poly[i+8190]^poly[i+8191];
        // for(int i=offset-4096;i<=offset-1-0;++i)poly[i]^=poly[i+4096]^poly[i+7680]^poly[i+7936]^poly[i+8160]^poly[i+8176]^poly[i+8190]^poly[i+8191];
        // for(int i=offset+(1<<13)-8192;i<=offset+(1<<13)-1-8191;++i)poly[i]^=poly[i+4096]^poly[i+7680]^poly[i+7936]^poly[i+8160]^poly[i+8176]^poly[i+8190]^poly[i+8191];
        // for(int i=offset+(1<<13)-8191;i<=offset+(1<<13)-1-8190;++i)poly[i]^=poly[i+4096]^poly[i+7680]^poly[i+7936]^poly[i+8160]^poly[i+8176]^poly[i+8190];
        // for(int i=offset+(1<<13)-8190;i<=offset+(1<<13)-1-8176;++i)poly[i]^=poly[i+4096]^poly[i+7680]^poly[i+7936]^poly[i+8160]^poly[i+8176];
        // for(int i=offset+(1<<13)-8176;i<=offset+(1<<13)-1-8160;++i)poly[i]^=poly[i+4096]^poly[i+7680]^poly[i+7936]^poly[i+8160];
        // for(int i=offset+(1<<13)-8160;i<=offset+(1<<13)-1-7936;++i)poly[i]^=poly[i+4096]^poly[i+7680]^poly[i+7936];
        // for(int i=offset+(1<<13)-7936;i<=offset+(1<<13)-1-7680;++i)poly[i]^=poly[i+4096]^poly[i+7680];
        // for(int i=offset+(1<<13)-7680;i<=offset+(1<<13)-1-4096;++i)poly[i]^=poly[i+4096];
        // xor_gpu<<<1, 1>>>(poly, offset-8191, 8191);
        // xor_gpu<<<1, 14>>>(poly, offset-8190, 8190, 8191);
        // xor_gpu<<<1, 16>>>(poly, offset-8176, 8176, 8190, 8191);
        // xor_gpu<<<1, 224>>>(poly, offset-8160, 8160, 8176, 8190, 8191);
        // xor_gpu<<<1, 256>>>(poly, offset-7936, 7936, 8160, 8176, 8190, 8191);
        // xor_gpu<<<7, 512>>>(poly, offset-7680, 7680, 7936, 8160, 8176, 8190, 8191);
        // xor_gpu<<<4, 1024>>>(poly, offset-4096, 4096, 7680, 7936, 8160, 8176, 8190, 8191);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<13)-8192, 4096, 7680, 7936, 8160, 8176, 8190, 8191);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<13)-8191, 4096, 7680, 7936, 8160, 8176, 8190);
        // xor_gpu<<<1, 14>>>(poly, offset+(1<<13)-8190, 4096, 7680, 7936, 8160, 8176);
        // xor_gpu<<<1, 16>>>(poly, offset+(1<<13)-8176, 4096, 7680, 7936, 8160);
        // xor_gpu<<<1, 224>>>(poly, offset+(1<<13)-8160, 4096, 7680, 7936);
        // xor_gpu<<<1, 256>>>(poly, offset+(1<<13)-7936, 4096, 7680);
        // xor_gpu<<<7, 512>>>(poly, offset+(1<<13)-7680, 4096);
    }

    for(int offset=(1<<14);offset<(1<<logn);offset+=(1<<(14+1))) {
        // for(int i=offset-16383;i<=offset-1-16380;++i)poly[i]^=poly[i+16383];
        // for(int i=offset-16380;i<=offset-1-16368;++i)poly[i]^=poly[i+16380]^poly[i+16383];
        // for(int i=offset-16368;i<=offset-1-16320;++i)poly[i]^=poly[i+16368]^poly[i+16380]^poly[i+16383];
        // for(int i=offset-16320;i<=offset-1-16128;++i)poly[i]^=poly[i+16320]^poly[i+16368]^poly[i+16380]^poly[i+16383];
        // for(int i=offset-16128;i<=offset-1-15360;++i)poly[i]^=poly[i+16128]^poly[i+16320]^poly[i+16368]^poly[i+16380]^poly[i+16383];
        // for(int i=offset-15360;i<=offset-1-12288;++i)poly[i]^=poly[i+15360]^poly[i+16128]^poly[i+16320]^poly[i+16368]^poly[i+16380]^poly[i+16383];
        // for(int i=offset-12288;i<=offset-1-0;++i)poly[i]^=poly[i+12288]^poly[i+15360]^poly[i+16128]^poly[i+16320]^poly[i+16368]^poly[i+16380]^poly[i+16383];
        // for(int i=offset+(1<<14)-16384;i<=offset+(1<<14)-1-16383;++i)poly[i]^=poly[i+12288]^poly[i+15360]^poly[i+16128]^poly[i+16320]^poly[i+16368]^poly[i+16380]^poly[i+16383];
        // for(int i=offset+(1<<14)-16383;i<=offset+(1<<14)-1-16380;++i)poly[i]^=poly[i+12288]^poly[i+15360]^poly[i+16128]^poly[i+16320]^poly[i+16368]^poly[i+16380];
        // for(int i=offset+(1<<14)-16380;i<=offset+(1<<14)-1-16368;++i)poly[i]^=poly[i+12288]^poly[i+15360]^poly[i+16128]^poly[i+16320]^poly[i+16368];
        // for(int i=offset+(1<<14)-16368;i<=offset+(1<<14)-1-16320;++i)poly[i]^=poly[i+12288]^poly[i+15360]^poly[i+16128]^poly[i+16320];
        // for(int i=offset+(1<<14)-16320;i<=offset+(1<<14)-1-16128;++i)poly[i]^=poly[i+12288]^poly[i+15360]^poly[i+16128];
        // for(int i=offset+(1<<14)-16128;i<=offset+(1<<14)-1-15360;++i)poly[i]^=poly[i+12288]^poly[i+15360];
        // for(int i=offset+(1<<14)-15360;i<=offset+(1<<14)-1-12288;++i)poly[i]^=poly[i+12288];
        // xor_gpu<<<1, 3>>>(poly, offset-16383, 16383);
        // xor_gpu<<<1, 12>>>(poly, offset-16380, 16380, 16383);
        // xor_gpu<<<1, 48>>>(poly, offset-16368, 16368, 16380, 16383);
        // xor_gpu<<<1, 192>>>(poly, offset-16320, 16320, 16368, 16380, 16383);
        // xor_gpu<<<1, 768>>>(poly, offset-16128, 16128, 16320, 16368, 16380, 16383);
        // xor_gpu<<<3, 1024>>>(poly, offset-15360, 15360, 16128, 16320, 16368, 16380, 16383);
        // xor_gpu<<<12, 1024>>>(poly, offset-12288, 12288, 15360, 16128, 16320, 16368, 16380, 16383);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<14)-16384, 12288, 15360, 16128, 16320, 16368, 16380, 16383);
        // xor_gpu<<<1, 3>>>(poly, offset+(1<<14)-16383, 12288, 15360, 16128, 16320, 16368, 16380);
        // xor_gpu<<<1, 12>>>(poly, offset+(1<<14)-16380, 12288, 15360, 16128, 16320, 16368);
        // xor_gpu<<<1, 48>>>(poly, offset+(1<<14)-16368, 12288, 15360, 16128, 16320);
        // xor_gpu<<<1, 192>>>(poly, offset+(1<<14)-16320, 12288, 15360, 16128);
        // xor_gpu<<<1, 768>>>(poly, offset+(1<<14)-16128, 12288, 15360);
        // xor_gpu<<<3, 1024>>>(poly, offset+(1<<14)-15360, 12288);
    }

    for(int offset=(1<<15);offset<(1<<logn);offset+=(1<<(15+1))) {
        // for(int i=offset-32767;i<=offset-1-32766;++i)poly[i]^=poly[i+32767];
        // for(int i=offset-32766;i<=offset-1-32764;++i)poly[i]^=poly[i+32766]^poly[i+32767];
        // for(int i=offset-32764;i<=offset-1-32760;++i)poly[i]^=poly[i+32764]^poly[i+32766]^poly[i+32767];
        // for(int i=offset-32760;i<=offset-1-32752;++i)poly[i]^=poly[i+32760]^poly[i+32764]^poly[i+32766]^poly[i+32767];
        // for(int i=offset-32752;i<=offset-1-32736;++i)poly[i]^=poly[i+32752]^poly[i+32760]^poly[i+32764]^poly[i+32766]^poly[i+32767];
        // for(int i=offset-32736;i<=offset-1-32704;++i)poly[i]^=poly[i+32736]^poly[i+32752]^poly[i+32760]^poly[i+32764]^poly[i+32766]^poly[i+32767];
        // for(int i=offset-32704;i<=offset-1-32640;++i)poly[i]^=poly[i+32704]^poly[i+32736]^poly[i+32752]^poly[i+32760]^poly[i+32764]^poly[i+32766]^poly[i+32767];
        // for(int i=offset-32640;i<=offset-1-32512;++i)poly[i]^=poly[i+32640]^poly[i+32704]^poly[i+32736]^poly[i+32752]^poly[i+32760]^poly[i+32764]^poly[i+32766]^poly[i+32767];
        // for(int i=offset-32512;i<=offset-1-32256;++i)poly[i]^=poly[i+32512]^poly[i+32640]^poly[i+32704]^poly[i+32736]^poly[i+32752]^poly[i+32760]^poly[i+32764]^poly[i+32766]^poly[i+32767];
        // for(int i=offset-32256;i<=offset-1-31744;++i)poly[i]^=poly[i+32256]^poly[i+32512]^poly[i+32640]^poly[i+32704]^poly[i+32736]^poly[i+32752]^poly[i+32760]^poly[i+32764]^poly[i+32766]^poly[i+32767];
        // for(int i=offset-31744;i<=offset-1-30720;++i)poly[i]^=poly[i+31744]^poly[i+32256]^poly[i+32512]^poly[i+32640]^poly[i+32704]^poly[i+32736]^poly[i+32752]^poly[i+32760]^poly[i+32764]^poly[i+32766]^poly[i+32767];
        // for(int i=offset-30720;i<=offset-1-28672;++i)poly[i]^=poly[i+30720]^poly[i+31744]^poly[i+32256]^poly[i+32512]^poly[i+32640]^poly[i+32704]^poly[i+32736]^poly[i+32752]^poly[i+32760]^poly[i+32764]^poly[i+32766]^poly[i+32767];
        // for(int i=offset-28672;i<=offset-1-24576;++i)poly[i]^=poly[i+28672]^poly[i+30720]^poly[i+31744]^poly[i+32256]^poly[i+32512]^poly[i+32640]^poly[i+32704]^poly[i+32736]^poly[i+32752]^poly[i+32760]^poly[i+32764]^poly[i+32766]^poly[i+32767];
        // for(int i=offset-24576;i<=offset-1-16384;++i)poly[i]^=poly[i+24576]^poly[i+28672]^poly[i+30720]^poly[i+31744]^poly[i+32256]^poly[i+32512]^poly[i+32640]^poly[i+32704]^poly[i+32736]^poly[i+32752]^poly[i+32760]^poly[i+32764]^poly[i+32766]^poly[i+32767];
        // for(int i=offset-16384;i<=offset-1-0;++i)poly[i]^=poly[i+16384]^poly[i+24576]^poly[i+28672]^poly[i+30720]^poly[i+31744]^poly[i+32256]^poly[i+32512]^poly[i+32640]^poly[i+32704]^poly[i+32736]^poly[i+32752]^poly[i+32760]^poly[i+32764]^poly[i+32766]^poly[i+32767];
        // for(int i=offset+(1<<15)-32768;i<=offset+(1<<15)-1-32767;++i)poly[i]^=poly[i+16384]^poly[i+24576]^poly[i+28672]^poly[i+30720]^poly[i+31744]^poly[i+32256]^poly[i+32512]^poly[i+32640]^poly[i+32704]^poly[i+32736]^poly[i+32752]^poly[i+32760]^poly[i+32764]^poly[i+32766]^poly[i+32767];
        // for(int i=offset+(1<<15)-32767;i<=offset+(1<<15)-1-32766;++i)poly[i]^=poly[i+16384]^poly[i+24576]^poly[i+28672]^poly[i+30720]^poly[i+31744]^poly[i+32256]^poly[i+32512]^poly[i+32640]^poly[i+32704]^poly[i+32736]^poly[i+32752]^poly[i+32760]^poly[i+32764]^poly[i+32766];
        // for(int i=offset+(1<<15)-32766;i<=offset+(1<<15)-1-32764;++i)poly[i]^=poly[i+16384]^poly[i+24576]^poly[i+28672]^poly[i+30720]^poly[i+31744]^poly[i+32256]^poly[i+32512]^poly[i+32640]^poly[i+32704]^poly[i+32736]^poly[i+32752]^poly[i+32760]^poly[i+32764];
        // for(int i=offset+(1<<15)-32764;i<=offset+(1<<15)-1-32760;++i)poly[i]^=poly[i+16384]^poly[i+24576]^poly[i+28672]^poly[i+30720]^poly[i+31744]^poly[i+32256]^poly[i+32512]^poly[i+32640]^poly[i+32704]^poly[i+32736]^poly[i+32752]^poly[i+32760];
        // for(int i=offset+(1<<15)-32760;i<=offset+(1<<15)-1-32752;++i)poly[i]^=poly[i+16384]^poly[i+24576]^poly[i+28672]^poly[i+30720]^poly[i+31744]^poly[i+32256]^poly[i+32512]^poly[i+32640]^poly[i+32704]^poly[i+32736]^poly[i+32752];
        // for(int i=offset+(1<<15)-32752;i<=offset+(1<<15)-1-32736;++i)poly[i]^=poly[i+16384]^poly[i+24576]^poly[i+28672]^poly[i+30720]^poly[i+31744]^poly[i+32256]^poly[i+32512]^poly[i+32640]^poly[i+32704]^poly[i+32736];
        // for(int i=offset+(1<<15)-32736;i<=offset+(1<<15)-1-32704;++i)poly[i]^=poly[i+16384]^poly[i+24576]^poly[i+28672]^poly[i+30720]^poly[i+31744]^poly[i+32256]^poly[i+32512]^poly[i+32640]^poly[i+32704];
        // for(int i=offset+(1<<15)-32704;i<=offset+(1<<15)-1-32640;++i)poly[i]^=poly[i+16384]^poly[i+24576]^poly[i+28672]^poly[i+30720]^poly[i+31744]^poly[i+32256]^poly[i+32512]^poly[i+32640];
        // for(int i=offset+(1<<15)-32640;i<=offset+(1<<15)-1-32512;++i)poly[i]^=poly[i+16384]^poly[i+24576]^poly[i+28672]^poly[i+30720]^poly[i+31744]^poly[i+32256]^poly[i+32512];
        // for(int i=offset+(1<<15)-32512;i<=offset+(1<<15)-1-32256;++i)poly[i]^=poly[i+16384]^poly[i+24576]^poly[i+28672]^poly[i+30720]^poly[i+31744]^poly[i+32256];
        // for(int i=offset+(1<<15)-32256;i<=offset+(1<<15)-1-31744;++i)poly[i]^=poly[i+16384]^poly[i+24576]^poly[i+28672]^poly[i+30720]^poly[i+31744];
        // for(int i=offset+(1<<15)-31744;i<=offset+(1<<15)-1-30720;++i)poly[i]^=poly[i+16384]^poly[i+24576]^poly[i+28672]^poly[i+30720];
        // for(int i=offset+(1<<15)-30720;i<=offset+(1<<15)-1-28672;++i)poly[i]^=poly[i+16384]^poly[i+24576]^poly[i+28672];
        // for(int i=offset+(1<<15)-28672;i<=offset+(1<<15)-1-24576;++i)poly[i]^=poly[i+16384]^poly[i+24576];
        // for(int i=offset+(1<<15)-24576;i<=offset+(1<<15)-1-16384;++i)poly[i]^=poly[i+16384];
        // xor_gpu<<<1, 1>>>(poly, offset-32767, 32767);
        // xor_gpu<<<1, 2>>>(poly, offset-32766, 32766, 32767);
        // xor_gpu<<<1, 4>>>(poly, offset-32764, 32764, 32766, 32767);
        // xor_gpu<<<1, 8>>>(poly, offset-32760, 32760, 32764, 32766, 32767);
        // xor_gpu<<<1, 16>>>(poly, offset-32752, 32752, 32760, 32764, 32766, 32767);
        // xor_gpu<<<1, 32>>>(poly, offset-32736, 32736, 32752, 32760, 32764, 32766, 32767);
        // xor_gpu<<<1, 64>>>(poly, offset-32704, 32704, 32736, 32752, 32760, 32764, 32766, 32767);
        // xor_gpu<<<1, 128>>>(poly, offset-32640, 32640, 32704, 32736, 32752, 32760, 32764, 32766, 32767);
        // xor_gpu<<<1, 256>>>(poly, offset-32512, 32512, 32640, 32704, 32736, 32752, 32760, 32764, 32766, 32767);
        // xor_gpu<<<1, 512>>>(poly, offset-32256, 32256, 32512, 32640, 32704, 32736, 32752, 32760, 32764, 32766, 32767);
        // xor_gpu<<<1, 1024>>>(poly, offset-31744, 31744, 32256, 32512, 32640, 32704, 32736, 32752, 32760, 32764, 32766, 32767);
        // xor_gpu<<<2, 1024>>>(poly, offset-30720, 30720, 31744, 32256, 32512, 32640, 32704, 32736, 32752, 32760, 32764, 32766, 32767);
        // xor_gpu<<<4, 1024>>>(poly, offset-28672, 28672, 30720, 31744, 32256, 32512, 32640, 32704, 32736, 32752, 32760, 32764, 32766, 32767);
        // xor_gpu<<<8, 1024>>>(poly, offset-24576, 24576, 28672, 30720, 31744, 32256, 32512, 32640, 32704, 32736, 32752, 32760, 32764, 32766, 32767);
        // xor_gpu<<<16, 1024>>>(poly, offset-16384, 16384, 24576, 28672, 30720, 31744, 32256, 32512, 32640, 32704, 32736, 32752, 32760, 32764, 32766, 32767);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<15)-32768, 16384, 24576, 28672, 30720, 31744, 32256, 32512, 32640, 32704, 32736, 32752, 32760, 32764, 32766, 32767);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<15)-32767, 16384, 24576, 28672, 30720, 31744, 32256, 32512, 32640, 32704, 32736, 32752, 32760, 32764, 32766);
        // xor_gpu<<<1, 2>>>(poly, offset+(1<<15)-32766, 16384, 24576, 28672, 30720, 31744, 32256, 32512, 32640, 32704, 32736, 32752, 32760, 32764);
        // xor_gpu<<<1, 4>>>(poly, offset+(1<<15)-32764, 16384, 24576, 28672, 30720, 31744, 32256, 32512, 32640, 32704, 32736, 32752, 32760);
        // xor_gpu<<<1, 8>>>(poly, offset+(1<<15)-32760, 16384, 24576, 28672, 30720, 31744, 32256, 32512, 32640, 32704, 32736, 32752);
        // xor_gpu<<<1, 16>>>(poly, offset+(1<<15)-32752, 16384, 24576, 28672, 30720, 31744, 32256, 32512, 32640, 32704, 32736);
        // xor_gpu<<<1, 32>>>(poly, offset+(1<<15)-32736, 16384, 24576, 28672, 30720, 31744, 32256, 32512, 32640, 32704);
        // xor_gpu<<<1, 64>>>(poly, offset+(1<<15)-32704, 16384, 24576, 28672, 30720, 31744, 32256, 32512, 32640);
        // xor_gpu<<<1, 128>>>(poly, offset+(1<<15)-32640, 16384, 24576, 28672, 30720, 31744, 32256, 32512);
        // xor_gpu<<<1, 256>>>(poly, offset+(1<<15)-32512, 16384, 24576, 28672, 30720, 31744, 32256);
        // xor_gpu<<<1, 512>>>(poly, offset+(1<<15)-32256, 16384, 24576, 28672, 30720, 31744);
        // xor_gpu<<<1, 1024>>>(poly, offset+(1<<15)-31744, 16384, 24576, 28672, 30720);
        // xor_gpu<<<2, 1024>>>(poly, offset+(1<<15)-30720, 16384, 24576, 28672);
        // xor_gpu<<<4, 1024>>>(poly, offset+(1<<15)-28672, 16384, 24576);
        // xor_gpu<<<8, 1024>>>(poly, offset+(1<<15)-24576, 16384);
    }
}

void bc_to_mono_256_19_17(u256* poly, int logn) {
    for(int offset=(1<<16);offset<(1<<logn);offset+=(1<<(16+1))) {
        // for(int i=offset-65535;i<=offset-1-0;++i)poly[i]^=poly[i+65535];
        // for(int i=offset+(1<<16)-65536;i<=offset+(1<<16)-1-65535;++i)poly[i]^=poly[i+65535];
        // xor_gpu<<<64, 1024>>>(poly, offset-65535, 65535);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<16)-65536, 65535);
    }

    for(int offset=(1<<17);offset<(1<<logn);offset+=(1<<(17+1))) {
        // for(int i=offset-131071;i<=offset-1-131070;++i)poly[i]^=poly[i+131071];
        // for(int i=offset-131070;i<=offset-1-65536;++i)poly[i]^=poly[i+131070]^poly[i+131071];
        // for(int i=offset-65536;i<=offset-1-0;++i)poly[i]^=poly[i+65536]^poly[i+131070]^poly[i+131071];
        // for(int i=offset+(1<<17)-131072;i<=offset+(1<<17)-1-131071;++i)poly[i]^=poly[i+65536]^poly[i+131070]^poly[i+131071];
        // for(int i=offset+(1<<17)-131071;i<=offset+(1<<17)-1-131070;++i)poly[i]^=poly[i+65536]^poly[i+131070];
        // for(int i=offset+(1<<17)-131070;i<=offset+(1<<17)-1-65536;++i)poly[i]^=poly[i+65536];
        // xor_gpu<<<1, 1>>>(poly, offset-131071, 131071);
        // xor_gpu<<<64, 1024>>>(poly, offset-131070, 131070, 131071);
        // xor_gpu<<<64, 1024>>>(poly, offset-65536, 65536, 131070, 131071);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<17)-131072, 65536, 131070, 131071);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<17)-131071, 65536, 131070);
        // xor_gpu<<<64, 1024>>>(poly, offset+(1<<17)-131070, 65536);
    }

    for(int offset=(1<<18);offset<(1<<logn);offset+=(1<<(18+1))) {
        // for(int i=offset-262143;i<=offset-1-262140;++i)poly[i]^=poly[i+262143];
        // for(int i=offset-262140;i<=offset-1-196608;++i)poly[i]^=poly[i+262140]^poly[i+262143];
        // for(int i=offset-196608;i<=offset-1-0;++i)poly[i]^=poly[i+196608]^poly[i+262140]^poly[i+262143];
        // for(int i=offset+(1<<18)-262144;i<=offset+(1<<18)-1-262143;++i)poly[i]^=poly[i+196608]^poly[i+262140]^poly[i+262143];
        // for(int i=offset+(1<<18)-262143;i<=offset+(1<<18)-1-262140;++i)poly[i]^=poly[i+196608]^poly[i+262140];
        // for(int i=offset+(1<<18)-262140;i<=offset+(1<<18)-1-196608;++i)poly[i]^=poly[i+196608];
        // xor_gpu<<<1, 3>>>(poly, offset-262143, 262143);
        // xor_gpu<<<64, 1024>>>(poly, offset-262140, 262140, 262143);
        // xor_gpu<<<192, 1024>>>(poly, offset-196608, 196608, 262140, 262143);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<18)-262144, 196608, 262140, 262143);
        // xor_gpu<<<1, 3>>>(poly, offset+(1<<18)-262143, 196608, 262140);
        // xor_gpu<<<64, 1024>>>(poly, offset+(1<<18)-262140, 196608);
    }
}

void bc_to_mono_256_30_20(u256* poly, int logn) {
    for(int offset=(1<<19);offset<(1<<logn);offset+=(1<<(19+1))) {
        // for(int i=offset-524287;i<=offset-1-524286;++i)poly[i]^=poly[i+524287];
        // for(int i=offset-524286;i<=offset-1-524284;++i)poly[i]^=poly[i+524286]^poly[i+524287];
        // for(int i=offset-524284;i<=offset-1-524280;++i)poly[i]^=poly[i+524284]^poly[i+524286]^poly[i+524287];
        // for(int i=offset-524280;i<=offset-1-458752;++i)poly[i]^=poly[i+524280]^poly[i+524284]^poly[i+524286]^poly[i+524287];
        // for(int i=offset-458752;i<=offset-1-393216;++i)poly[i]^=poly[i+458752]^poly[i+524280]^poly[i+524284]^poly[i+524286]^poly[i+524287];
        // for(int i=offset-393216;i<=offset-1-262144;++i)poly[i]^=poly[i+393216]^poly[i+458752]^poly[i+524280]^poly[i+524284]^poly[i+524286]^poly[i+524287];
        // for(int i=offset-262144;i<=offset-1-0;++i)poly[i]^=poly[i+262144]^poly[i+393216]^poly[i+458752]^poly[i+524280]^poly[i+524284]^poly[i+524286]^poly[i+524287];
        // for(int i=offset+(1<<19)-524288;i<=offset+(1<<19)-1-524287;++i)poly[i]^=poly[i+262144]^poly[i+393216]^poly[i+458752]^poly[i+524280]^poly[i+524284]^poly[i+524286]^poly[i+524287];
        // for(int i=offset+(1<<19)-524287;i<=offset+(1<<19)-1-524286;++i)poly[i]^=poly[i+262144]^poly[i+393216]^poly[i+458752]^poly[i+524280]^poly[i+524284]^poly[i+524286];
        // for(int i=offset+(1<<19)-524286;i<=offset+(1<<19)-1-524284;++i)poly[i]^=poly[i+262144]^poly[i+393216]^poly[i+458752]^poly[i+524280]^poly[i+524284];
        // for(int i=offset+(1<<19)-524284;i<=offset+(1<<19)-1-524280;++i)poly[i]^=poly[i+262144]^poly[i+393216]^poly[i+458752]^poly[i+524280];
        // for(int i=offset+(1<<19)-524280;i<=offset+(1<<19)-1-458752;++i)poly[i]^=poly[i+262144]^poly[i+393216]^poly[i+458752];
        // for(int i=offset+(1<<19)-458752;i<=offset+(1<<19)-1-393216;++i)poly[i]^=poly[i+262144]^poly[i+393216];
        // for(int i=offset+(1<<19)-393216;i<=offset+(1<<19)-1-262144;++i)poly[i]^=poly[i+262144];
        // xor_gpu<<<1, 1>>>(poly, offset-524287, 524287);
        // xor_gpu<<<1, 2>>>(poly, offset-524286, 524286, 524287);
        // xor_gpu<<<1, 4>>>(poly, offset-524284, 524284, 524286, 524287);
        // xor_gpu<<<64, 1024>>>(poly, offset-524280, 524280, 524284, 524286, 524287);
        // xor_gpu<<<64, 1024>>>(poly, offset-458752, 458752, 524280, 524284, 524286, 524287);
        // xor_gpu<<<128, 1024>>>(poly, offset-393216, 393216, 458752, 524280, 524284, 524286, 524287);
        // xor_gpu<<<256, 1024>>>(poly, offset-262144, 262144, 393216, 458752, 524280, 524284, 524286, 524287);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<19)-524288, 262144, 393216, 458752, 524280, 524284, 524286, 524287);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<19)-524287, 262144, 393216, 458752, 524280, 524284, 524286);
        // xor_gpu<<<1, 2>>>(poly, offset+(1<<19)-524286, 262144, 393216, 458752, 524280, 524284);
        // xor_gpu<<<1, 4>>>(poly, offset+(1<<19)-524284, 262144, 393216, 458752, 524280);
        // xor_gpu<<<64, 1024>>>(poly, offset+(1<<19)-524280, 262144, 393216, 458752);
        // xor_gpu<<<64, 1024>>>(poly, offset+(1<<19)-458752, 262144, 393216);
        // xor_gpu<<<128, 1024>>>(poly, offset+(1<<19)-393216, 262144);
    }

    for(int offset=(1<<20);offset<(1<<logn);offset+=(1<<(20+1))) {
        // for(int i=offset-1048575;i<=offset-1-1048560;++i)poly[i]^=poly[i+1048575];
        // for(int i=offset-1048560;i<=offset-1-983040;++i)poly[i]^=poly[i+1048560]^poly[i+1048575];
        // for(int i=offset-983040;i<=offset-1-0;++i)poly[i]^=poly[i+983040]^poly[i+1048560]^poly[i+1048575];
        // for(int i=offset+(1<<20)-1048576;i<=offset+(1<<20)-1-1048575;++i)poly[i]^=poly[i+983040]^poly[i+1048560]^poly[i+1048575];
        // for(int i=offset+(1<<20)-1048575;i<=offset+(1<<20)-1-1048560;++i)poly[i]^=poly[i+983040]^poly[i+1048560];
        // for(int i=offset+(1<<20)-1048560;i<=offset+(1<<20)-1-983040;++i)poly[i]^=poly[i+983040];
        // xor_gpu<<<1, 15>>>(poly, offset-1048575, 1048575);
        // xor_gpu<<<64, 1024>>>(poly, offset-1048560, 1048560, 1048575);
        // xor_gpu<<<960, 1024>>>(poly, offset-983040, 983040, 1048560, 1048575);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<20)-1048576, 983040, 1048560, 1048575);
        // xor_gpu<<<1, 15>>>(poly, offset+(1<<20)-1048575, 983040, 1048560);
        // xor_gpu<<<64, 1024>>>(poly, offset+(1<<20)-1048560, 983040);
    }

    for(int offset=(1<<21);offset<(1<<logn);offset+=(1<<(21+1))) {
        // for(int i=offset-2097151;i<=offset-1-2097150;++i)poly[i]^=poly[i+2097151];
        // for(int i=offset-2097150;i<=offset-1-2097136;++i)poly[i]^=poly[i+2097150]^poly[i+2097151];
        // for(int i=offset-2097136;i<=offset-1-2097120;++i)poly[i]^=poly[i+2097136]^poly[i+2097150]^poly[i+2097151];
        // for(int i=offset-2097120;i<=offset-1-2031616;++i)poly[i]^=poly[i+2097120]^poly[i+2097136]^poly[i+2097150]^poly[i+2097151];
        // for(int i=offset-2031616;i<=offset-1-1966080;++i)poly[i]^=poly[i+2031616]^poly[i+2097120]^poly[i+2097136]^poly[i+2097150]^poly[i+2097151];
        // for(int i=offset-1966080;i<=offset-1-1048576;++i)poly[i]^=poly[i+1966080]^poly[i+2031616]^poly[i+2097120]^poly[i+2097136]^poly[i+2097150]^poly[i+2097151];
        // for(int i=offset-1048576;i<=offset-1-0;++i)poly[i]^=poly[i+1048576]^poly[i+1966080]^poly[i+2031616]^poly[i+2097120]^poly[i+2097136]^poly[i+2097150]^poly[i+2097151];
        // for(int i=offset+(1<<21)-2097152;i<=offset+(1<<21)-1-2097151;++i)poly[i]^=poly[i+1048576]^poly[i+1966080]^poly[i+2031616]^poly[i+2097120]^poly[i+2097136]^poly[i+2097150]^poly[i+2097151];
        // for(int i=offset+(1<<21)-2097151;i<=offset+(1<<21)-1-2097150;++i)poly[i]^=poly[i+1048576]^poly[i+1966080]^poly[i+2031616]^poly[i+2097120]^poly[i+2097136]^poly[i+2097150];
        // for(int i=offset+(1<<21)-2097150;i<=offset+(1<<21)-1-2097136;++i)poly[i]^=poly[i+1048576]^poly[i+1966080]^poly[i+2031616]^poly[i+2097120]^poly[i+2097136];
        // for(int i=offset+(1<<21)-2097136;i<=offset+(1<<21)-1-2097120;++i)poly[i]^=poly[i+1048576]^poly[i+1966080]^poly[i+2031616]^poly[i+2097120];
        // for(int i=offset+(1<<21)-2097120;i<=offset+(1<<21)-1-2031616;++i)poly[i]^=poly[i+1048576]^poly[i+1966080]^poly[i+2031616];
        // for(int i=offset+(1<<21)-2031616;i<=offset+(1<<21)-1-1966080;++i)poly[i]^=poly[i+1048576]^poly[i+1966080];
        // for(int i=offset+(1<<21)-1966080;i<=offset+(1<<21)-1-1048576;++i)poly[i]^=poly[i+1048576];
        // xor_gpu<<<1, 1>>>(poly, offset-2097151, 2097151);
        // xor_gpu<<<1, 14>>>(poly, offset-2097150, 2097150, 2097151);
        // xor_gpu<<<1, 16>>>(poly, offset-2097136, 2097136, 2097150, 2097151);
        // xor_gpu<<<2047, 32>>>(poly, offset-2097120, 2097120, 2097136, 2097150, 2097151);
        // xor_gpu<<<64, 1024>>>(poly, offset-2031616, 2031616, 2097120, 2097136, 2097150, 2097151);
        // xor_gpu<<<896, 1024>>>(poly, offset-1966080, 1966080, 2031616, 2097120, 2097136, 2097150, 2097151);
        // xor_gpu<<<1024, 1024>>>(poly, offset-1048576, 1048576, 1966080, 2031616, 2097120, 2097136, 2097150, 2097151);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<21)-2097152, 1048576, 1966080, 2031616, 2097120, 2097136, 2097150, 2097151);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<21)-2097151, 1048576, 1966080, 2031616, 2097120, 2097136, 2097150);
        // xor_gpu<<<1, 14>>>(poly, offset+(1<<21)-2097150, 1048576, 1966080, 2031616, 2097120, 2097136);
        // xor_gpu<<<1, 16>>>(poly, offset+(1<<21)-2097136, 1048576, 1966080, 2031616, 2097120);
        // xor_gpu<<<2047, 32>>>(poly, offset+(1<<21)-2097120, 1048576, 1966080, 2031616);
        // xor_gpu<<<64, 1024>>>(poly, offset+(1<<21)-2031616, 1048576, 1966080);
        // xor_gpu<<<896, 1024>>>(poly, offset+(1<<21)-1966080, 1048576);
    }

    for(int offset=(1<<22);offset<(1<<logn);offset+=(1<<(22+1))) {
        // for(int i=offset-4194303;i<=offset-1-4194300;++i)poly[i]^=poly[i+4194303];
        // for(int i=offset-4194300;i<=offset-1-4194288;++i)poly[i]^=poly[i+4194300]^poly[i+4194303];
        // for(int i=offset-4194288;i<=offset-1-4194240;++i)poly[i]^=poly[i+4194288]^poly[i+4194300]^poly[i+4194303];
        // for(int i=offset-4194240;i<=offset-1-4128768;++i)poly[i]^=poly[i+4194240]^poly[i+4194288]^poly[i+4194300]^poly[i+4194303];
        // for(int i=offset-4128768;i<=offset-1-3932160;++i)poly[i]^=poly[i+4128768]^poly[i+4194240]^poly[i+4194288]^poly[i+4194300]^poly[i+4194303];
        // for(int i=offset-3932160;i<=offset-1-3145728;++i)poly[i]^=poly[i+3932160]^poly[i+4128768]^poly[i+4194240]^poly[i+4194288]^poly[i+4194300]^poly[i+4194303];
        // for(int i=offset-3145728;i<=offset-1-0;++i)poly[i]^=poly[i+3145728]^poly[i+3932160]^poly[i+4128768]^poly[i+4194240]^poly[i+4194288]^poly[i+4194300]^poly[i+4194303];
        // for(int i=offset+(1<<22)-4194304;i<=offset+(1<<22)-1-4194303;++i)poly[i]^=poly[i+3145728]^poly[i+3932160]^poly[i+4128768]^poly[i+4194240]^poly[i+4194288]^poly[i+4194300]^poly[i+4194303];
        // for(int i=offset+(1<<22)-4194303;i<=offset+(1<<22)-1-4194300;++i)poly[i]^=poly[i+3145728]^poly[i+3932160]^poly[i+4128768]^poly[i+4194240]^poly[i+4194288]^poly[i+4194300];
        // for(int i=offset+(1<<22)-4194300;i<=offset+(1<<22)-1-4194288;++i)poly[i]^=poly[i+3145728]^poly[i+3932160]^poly[i+4128768]^poly[i+4194240]^poly[i+4194288];
        // for(int i=offset+(1<<22)-4194288;i<=offset+(1<<22)-1-4194240;++i)poly[i]^=poly[i+3145728]^poly[i+3932160]^poly[i+4128768]^poly[i+4194240];
        // for(int i=offset+(1<<22)-4194240;i<=offset+(1<<22)-1-4128768;++i)poly[i]^=poly[i+3145728]^poly[i+3932160]^poly[i+4128768];
        // for(int i=offset+(1<<22)-4128768;i<=offset+(1<<22)-1-3932160;++i)poly[i]^=poly[i+3145728]^poly[i+3932160];
        // for(int i=offset+(1<<22)-3932160;i<=offset+(1<<22)-1-3145728;++i)poly[i]^=poly[i+3145728];
        // xor_gpu<<<1, 3>>>(poly, offset-4194303, 4194303);
        // xor_gpu<<<1, 12>>>(poly, offset-4194300, 4194300, 4194303);
        // xor_gpu<<<1, 48>>>(poly, offset-4194288, 4194288, 4194300, 4194303);
        // xor_gpu<<<1023, 64>>>(poly, offset-4194240, 4194240, 4194288, 4194300, 4194303);
        // xor_gpu<<<192, 1024>>>(poly, offset-4128768, 4128768, 4194240, 4194288, 4194300, 4194303);
        // xor_gpu<<<768, 1024>>>(poly, offset-3932160, 3932160, 4128768, 4194240, 4194288, 4194300, 4194303);
        // xor_gpu<<<3072, 1024>>>(poly, offset-3145728, 3145728, 3932160, 4128768, 4194240, 4194288, 4194300, 4194303);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<22)-4194304, 3145728, 3932160, 4128768, 4194240, 4194288, 4194300, 4194303);
        // xor_gpu<<<1, 3>>>(poly, offset+(1<<22)-4194303, 3145728, 3932160, 4128768, 4194240, 4194288, 4194300);
        // xor_gpu<<<1, 12>>>(poly, offset+(1<<22)-4194300, 3145728, 3932160, 4128768, 4194240, 4194288);
        // xor_gpu<<<1, 48>>>(poly, offset+(1<<22)-4194288, 3145728, 3932160, 4128768, 4194240);
        // xor_gpu<<<1023, 64>>>(poly, offset+(1<<22)-4194240, 3145728, 3932160, 4128768);
        // xor_gpu<<<192, 1024>>>(poly, offset+(1<<22)-4128768, 3145728, 3932160);
        // xor_gpu<<<768, 1024>>>(poly, offset+(1<<22)-3932160, 3145728);
    }

    for(int offset=(1<<23);offset<(1<<logn);offset+=(1<<(23+1))) {
        // for(int i=offset-8388607;i<=offset-1-8388606;++i)poly[i]^=poly[i+8388607];
        // for(int i=offset-8388606;i<=offset-1-8388604;++i)poly[i]^=poly[i+8388606]^poly[i+8388607];
        // for(int i=offset-8388604;i<=offset-1-8388600;++i)poly[i]^=poly[i+8388604]^poly[i+8388606]^poly[i+8388607];
        // for(int i=offset-8388600;i<=offset-1-8388592;++i)poly[i]^=poly[i+8388600]^poly[i+8388604]^poly[i+8388606]^poly[i+8388607];
        // for(int i=offset-8388592;i<=offset-1-8388576;++i)poly[i]^=poly[i+8388592]^poly[i+8388600]^poly[i+8388604]^poly[i+8388606]^poly[i+8388607];
        // for(int i=offset-8388576;i<=offset-1-8388544;++i)poly[i]^=poly[i+8388576]^poly[i+8388592]^poly[i+8388600]^poly[i+8388604]^poly[i+8388606]^poly[i+8388607];
        // for(int i=offset-8388544;i<=offset-1-8388480;++i)poly[i]^=poly[i+8388544]^poly[i+8388576]^poly[i+8388592]^poly[i+8388600]^poly[i+8388604]^poly[i+8388606]^poly[i+8388607];
        // for(int i=offset-8388480;i<=offset-1-8323072;++i)poly[i]^=poly[i+8388480]^poly[i+8388544]^poly[i+8388576]^poly[i+8388592]^poly[i+8388600]^poly[i+8388604]^poly[i+8388606]^poly[i+8388607];
        // for(int i=offset-8323072;i<=offset-1-8257536;++i)poly[i]^=poly[i+8323072]^poly[i+8388480]^poly[i+8388544]^poly[i+8388576]^poly[i+8388592]^poly[i+8388600]^poly[i+8388604]^poly[i+8388606]^poly[i+8388607];
        // for(int i=offset-8257536;i<=offset-1-8126464;++i)poly[i]^=poly[i+8257536]^poly[i+8323072]^poly[i+8388480]^poly[i+8388544]^poly[i+8388576]^poly[i+8388592]^poly[i+8388600]^poly[i+8388604]^poly[i+8388606]^poly[i+8388607];
        // for(int i=offset-8126464;i<=offset-1-7864320;++i)poly[i]^=poly[i+8126464]^poly[i+8257536]^poly[i+8323072]^poly[i+8388480]^poly[i+8388544]^poly[i+8388576]^poly[i+8388592]^poly[i+8388600]^poly[i+8388604]^poly[i+8388606]^poly[i+8388607];
        // for(int i=offset-7864320;i<=offset-1-7340032;++i)poly[i]^=poly[i+7864320]^poly[i+8126464]^poly[i+8257536]^poly[i+8323072]^poly[i+8388480]^poly[i+8388544]^poly[i+8388576]^poly[i+8388592]^poly[i+8388600]^poly[i+8388604]^poly[i+8388606]^poly[i+8388607];
        // for(int i=offset-7340032;i<=offset-1-6291456;++i)poly[i]^=poly[i+7340032]^poly[i+7864320]^poly[i+8126464]^poly[i+8257536]^poly[i+8323072]^poly[i+8388480]^poly[i+8388544]^poly[i+8388576]^poly[i+8388592]^poly[i+8388600]^poly[i+8388604]^poly[i+8388606]^poly[i+8388607];
        // for(int i=offset-6291456;i<=offset-1-4194304;++i)poly[i]^=poly[i+6291456]^poly[i+7340032]^poly[i+7864320]^poly[i+8126464]^poly[i+8257536]^poly[i+8323072]^poly[i+8388480]^poly[i+8388544]^poly[i+8388576]^poly[i+8388592]^poly[i+8388600]^poly[i+8388604]^poly[i+8388606]^poly[i+8388607];
        // for(int i=offset-4194304;i<=offset-1-0;++i)poly[i]^=poly[i+4194304]^poly[i+6291456]^poly[i+7340032]^poly[i+7864320]^poly[i+8126464]^poly[i+8257536]^poly[i+8323072]^poly[i+8388480]^poly[i+8388544]^poly[i+8388576]^poly[i+8388592]^poly[i+8388600]^poly[i+8388604]^poly[i+8388606]^poly[i+8388607];
        // for(int i=offset+(1<<23)-8388608;i<=offset+(1<<23)-1-8388607;++i)poly[i]^=poly[i+4194304]^poly[i+6291456]^poly[i+7340032]^poly[i+7864320]^poly[i+8126464]^poly[i+8257536]^poly[i+8323072]^poly[i+8388480]^poly[i+8388544]^poly[i+8388576]^poly[i+8388592]^poly[i+8388600]^poly[i+8388604]^poly[i+8388606]^poly[i+8388607];
        // for(int i=offset+(1<<23)-8388607;i<=offset+(1<<23)-1-8388606;++i)poly[i]^=poly[i+4194304]^poly[i+6291456]^poly[i+7340032]^poly[i+7864320]^poly[i+8126464]^poly[i+8257536]^poly[i+8323072]^poly[i+8388480]^poly[i+8388544]^poly[i+8388576]^poly[i+8388592]^poly[i+8388600]^poly[i+8388604]^poly[i+8388606];
        // for(int i=offset+(1<<23)-8388606;i<=offset+(1<<23)-1-8388604;++i)poly[i]^=poly[i+4194304]^poly[i+6291456]^poly[i+7340032]^poly[i+7864320]^poly[i+8126464]^poly[i+8257536]^poly[i+8323072]^poly[i+8388480]^poly[i+8388544]^poly[i+8388576]^poly[i+8388592]^poly[i+8388600]^poly[i+8388604];
        // for(int i=offset+(1<<23)-8388604;i<=offset+(1<<23)-1-8388600;++i)poly[i]^=poly[i+4194304]^poly[i+6291456]^poly[i+7340032]^poly[i+7864320]^poly[i+8126464]^poly[i+8257536]^poly[i+8323072]^poly[i+8388480]^poly[i+8388544]^poly[i+8388576]^poly[i+8388592]^poly[i+8388600];
        // for(int i=offset+(1<<23)-8388600;i<=offset+(1<<23)-1-8388592;++i)poly[i]^=poly[i+4194304]^poly[i+6291456]^poly[i+7340032]^poly[i+7864320]^poly[i+8126464]^poly[i+8257536]^poly[i+8323072]^poly[i+8388480]^poly[i+8388544]^poly[i+8388576]^poly[i+8388592];
        // for(int i=offset+(1<<23)-8388592;i<=offset+(1<<23)-1-8388576;++i)poly[i]^=poly[i+4194304]^poly[i+6291456]^poly[i+7340032]^poly[i+7864320]^poly[i+8126464]^poly[i+8257536]^poly[i+8323072]^poly[i+8388480]^poly[i+8388544]^poly[i+8388576];
        // for(int i=offset+(1<<23)-8388576;i<=offset+(1<<23)-1-8388544;++i)poly[i]^=poly[i+4194304]^poly[i+6291456]^poly[i+7340032]^poly[i+7864320]^poly[i+8126464]^poly[i+8257536]^poly[i+8323072]^poly[i+8388480]^poly[i+8388544];
        // for(int i=offset+(1<<23)-8388544;i<=offset+(1<<23)-1-8388480;++i)poly[i]^=poly[i+4194304]^poly[i+6291456]^poly[i+7340032]^poly[i+7864320]^poly[i+8126464]^poly[i+8257536]^poly[i+8323072]^poly[i+8388480];
        // for(int i=offset+(1<<23)-8388480;i<=offset+(1<<23)-1-8323072;++i)poly[i]^=poly[i+4194304]^poly[i+6291456]^poly[i+7340032]^poly[i+7864320]^poly[i+8126464]^poly[i+8257536]^poly[i+8323072];
        // for(int i=offset+(1<<23)-8323072;i<=offset+(1<<23)-1-8257536;++i)poly[i]^=poly[i+4194304]^poly[i+6291456]^poly[i+7340032]^poly[i+7864320]^poly[i+8126464]^poly[i+8257536];
        // for(int i=offset+(1<<23)-8257536;i<=offset+(1<<23)-1-8126464;++i)poly[i]^=poly[i+4194304]^poly[i+6291456]^poly[i+7340032]^poly[i+7864320]^poly[i+8126464];
        // for(int i=offset+(1<<23)-8126464;i<=offset+(1<<23)-1-7864320;++i)poly[i]^=poly[i+4194304]^poly[i+6291456]^poly[i+7340032]^poly[i+7864320];
        // for(int i=offset+(1<<23)-7864320;i<=offset+(1<<23)-1-7340032;++i)poly[i]^=poly[i+4194304]^poly[i+6291456]^poly[i+7340032];
        // for(int i=offset+(1<<23)-7340032;i<=offset+(1<<23)-1-6291456;++i)poly[i]^=poly[i+4194304]^poly[i+6291456];
        // for(int i=offset+(1<<23)-6291456;i<=offset+(1<<23)-1-4194304;++i)poly[i]^=poly[i+4194304];
        // xor_gpu<<<1, 1>>>(poly, offset-8388607, 8388607);
        // xor_gpu<<<1, 2>>>(poly, offset-8388606, 8388606, 8388607);
        // xor_gpu<<<1, 4>>>(poly, offset-8388604, 8388604, 8388606, 8388607);
        // xor_gpu<<<1, 8>>>(poly, offset-8388600, 8388600, 8388604, 8388606, 8388607);
        // xor_gpu<<<1, 16>>>(poly, offset-8388592, 8388592, 8388600, 8388604, 8388606, 8388607);
        // xor_gpu<<<1, 32>>>(poly, offset-8388576, 8388576, 8388592, 8388600, 8388604, 8388606, 8388607);
        // xor_gpu<<<1, 64>>>(poly, offset-8388544, 8388544, 8388576, 8388592, 8388600, 8388604, 8388606, 8388607);
        // xor_gpu<<<511, 128>>>(poly, offset-8388480, 8388480, 8388544, 8388576, 8388592, 8388600, 8388604, 8388606, 8388607);
        // xor_gpu<<<64, 1024>>>(poly, offset-8323072, 8323072, 8388480, 8388544, 8388576, 8388592, 8388600, 8388604, 8388606, 8388607);
        // xor_gpu<<<128, 1024>>>(poly, offset-8257536, 8257536, 8323072, 8388480, 8388544, 8388576, 8388592, 8388600, 8388604, 8388606, 8388607);
        // xor_gpu<<<256, 1024>>>(poly, offset-8126464, 8126464, 8257536, 8323072, 8388480, 8388544, 8388576, 8388592, 8388600, 8388604, 8388606, 8388607);
        // xor_gpu<<<512, 1024>>>(poly, offset-7864320, 7864320, 8126464, 8257536, 8323072, 8388480, 8388544, 8388576, 8388592, 8388600, 8388604, 8388606, 8388607);
        // xor_gpu<<<1024, 1024>>>(poly, offset-7340032, 7340032, 7864320, 8126464, 8257536, 8323072, 8388480, 8388544, 8388576, 8388592, 8388600, 8388604, 8388606, 8388607);
        // xor_gpu<<<2048, 1024>>>(poly, offset-6291456, 6291456, 7340032, 7864320, 8126464, 8257536, 8323072, 8388480, 8388544, 8388576, 8388592, 8388600, 8388604, 8388606, 8388607);
        // xor_gpu<<<4096, 1024>>>(poly, offset-4194304, 4194304, 6291456, 7340032, 7864320, 8126464, 8257536, 8323072, 8388480, 8388544, 8388576, 8388592, 8388600, 8388604, 8388606, 8388607);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<23)-8388608, 4194304, 6291456, 7340032, 7864320, 8126464, 8257536, 8323072, 8388480, 8388544, 8388576, 8388592, 8388600, 8388604, 8388606, 8388607);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<23)-8388607, 4194304, 6291456, 7340032, 7864320, 8126464, 8257536, 8323072, 8388480, 8388544, 8388576, 8388592, 8388600, 8388604, 8388606);
        // xor_gpu<<<1, 2>>>(poly, offset+(1<<23)-8388606, 4194304, 6291456, 7340032, 7864320, 8126464, 8257536, 8323072, 8388480, 8388544, 8388576, 8388592, 8388600, 8388604);
        // xor_gpu<<<1, 4>>>(poly, offset+(1<<23)-8388604, 4194304, 6291456, 7340032, 7864320, 8126464, 8257536, 8323072, 8388480, 8388544, 8388576, 8388592, 8388600);
        // xor_gpu<<<1, 8>>>(poly, offset+(1<<23)-8388600, 4194304, 6291456, 7340032, 7864320, 8126464, 8257536, 8323072, 8388480, 8388544, 8388576, 8388592);
        // xor_gpu<<<1, 16>>>(poly, offset+(1<<23)-8388592, 4194304, 6291456, 7340032, 7864320, 8126464, 8257536, 8323072, 8388480, 8388544, 8388576);
        // xor_gpu<<<1, 32>>>(poly, offset+(1<<23)-8388576, 4194304, 6291456, 7340032, 7864320, 8126464, 8257536, 8323072, 8388480, 8388544);
        // xor_gpu<<<1, 64>>>(poly, offset+(1<<23)-8388544, 4194304, 6291456, 7340032, 7864320, 8126464, 8257536, 8323072, 8388480);
        // xor_gpu<<<511, 128>>>(poly, offset+(1<<23)-8388480, 4194304, 6291456, 7340032, 7864320, 8126464, 8257536, 8323072);
        // xor_gpu<<<64, 1024>>>(poly, offset+(1<<23)-8323072, 4194304, 6291456, 7340032, 7864320, 8126464, 8257536);
        // xor_gpu<<<128, 1024>>>(poly, offset+(1<<23)-8257536, 4194304, 6291456, 7340032, 7864320, 8126464);
        // xor_gpu<<<256, 1024>>>(poly, offset+(1<<23)-8126464, 4194304, 6291456, 7340032, 7864320);
        // xor_gpu<<<512, 1024>>>(poly, offset+(1<<23)-7864320, 4194304, 6291456, 7340032);
        // xor_gpu<<<1024, 1024>>>(poly, offset+(1<<23)-7340032, 4194304, 6291456);
        // xor_gpu<<<2048, 1024>>>(poly, offset+(1<<23)-6291456, 4194304);
    }

    for(int offset=(1<<24);offset<(1<<logn);offset+=(1<<(24+1))) {
        // for(int i=offset-16777215;i<=offset-1-16776960;++i)poly[i]^=poly[i+16777215];
        // for(int i=offset-16776960;i<=offset-1-16711680;++i)poly[i]^=poly[i+16776960]^poly[i+16777215];
        // for(int i=offset-16711680;i<=offset-1-0;++i)poly[i]^=poly[i+16711680]^poly[i+16776960]^poly[i+16777215];
        // for(int i=offset+(1<<24)-16777216;i<=offset+(1<<24)-1-16777215;++i)poly[i]^=poly[i+16711680]^poly[i+16776960]^poly[i+16777215];
        // for(int i=offset+(1<<24)-16777215;i<=offset+(1<<24)-1-16776960;++i)poly[i]^=poly[i+16711680]^poly[i+16776960];
        // for(int i=offset+(1<<24)-16776960;i<=offset+(1<<24)-1-16711680;++i)poly[i]^=poly[i+16711680];
        // xor_gpu<<<1, 255>>>(poly, offset-16777215, 16777215);
        // xor_gpu<<<255, 256>>>(poly, offset-16776960, 16776960, 16777215);
        // xor_gpu<<<16320, 1024>>>(poly, offset-16711680, 16711680, 16776960, 16777215);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<24)-16777216, 16711680, 16776960, 16777215);
        // xor_gpu<<<1, 255>>>(poly, offset+(1<<24)-16777215, 16711680, 16776960);
        // xor_gpu<<<255, 256>>>(poly, offset+(1<<24)-16776960, 16711680);
    }

    for(int offset=(1<<25);offset<(1<<logn);offset+=(1<<(25+1))) {
        // for(int i=offset-33554431;i<=offset-1-33554430;++i)poly[i]^=poly[i+33554431];
        // for(int i=offset-33554430;i<=offset-1-33554176;++i)poly[i]^=poly[i+33554430]^poly[i+33554431];
        // for(int i=offset-33554176;i<=offset-1-33553920;++i)poly[i]^=poly[i+33554176]^poly[i+33554430]^poly[i+33554431];
        // for(int i=offset-33553920;i<=offset-1-33488896;++i)poly[i]^=poly[i+33553920]^poly[i+33554176]^poly[i+33554430]^poly[i+33554431];
        // for(int i=offset-33488896;i<=offset-1-33423360;++i)poly[i]^=poly[i+33488896]^poly[i+33553920]^poly[i+33554176]^poly[i+33554430]^poly[i+33554431];
        // for(int i=offset-33423360;i<=offset-1-16777216;++i)poly[i]^=poly[i+33423360]^poly[i+33488896]^poly[i+33553920]^poly[i+33554176]^poly[i+33554430]^poly[i+33554431];
        // for(int i=offset-16777216;i<=offset-1-0;++i)poly[i]^=poly[i+16777216]^poly[i+33423360]^poly[i+33488896]^poly[i+33553920]^poly[i+33554176]^poly[i+33554430]^poly[i+33554431];
        // for(int i=offset+(1<<25)-33554432;i<=offset+(1<<25)-1-33554431;++i)poly[i]^=poly[i+16777216]^poly[i+33423360]^poly[i+33488896]^poly[i+33553920]^poly[i+33554176]^poly[i+33554430]^poly[i+33554431];
        // for(int i=offset+(1<<25)-33554431;i<=offset+(1<<25)-1-33554430;++i)poly[i]^=poly[i+16777216]^poly[i+33423360]^poly[i+33488896]^poly[i+33553920]^poly[i+33554176]^poly[i+33554430];
        // for(int i=offset+(1<<25)-33554430;i<=offset+(1<<25)-1-33554176;++i)poly[i]^=poly[i+16777216]^poly[i+33423360]^poly[i+33488896]^poly[i+33553920]^poly[i+33554176];
        // for(int i=offset+(1<<25)-33554176;i<=offset+(1<<25)-1-33553920;++i)poly[i]^=poly[i+16777216]^poly[i+33423360]^poly[i+33488896]^poly[i+33553920];
        // for(int i=offset+(1<<25)-33553920;i<=offset+(1<<25)-1-33488896;++i)poly[i]^=poly[i+16777216]^poly[i+33423360]^poly[i+33488896];
        // for(int i=offset+(1<<25)-33488896;i<=offset+(1<<25)-1-33423360;++i)poly[i]^=poly[i+16777216]^poly[i+33423360];
        // for(int i=offset+(1<<25)-33423360;i<=offset+(1<<25)-1-16777216;++i)poly[i]^=poly[i+16777216];
        // xor_gpu<<<1, 1>>>(poly, offset-33554431, 33554431);
        // xor_gpu<<<1, 254>>>(poly, offset-33554430, 33554430, 33554431);
        // xor_gpu<<<1, 256>>>(poly, offset-33554176, 33554176, 33554430, 33554431);
        // xor_gpu<<<127, 512>>>(poly, offset-33553920, 33553920, 33554176, 33554430, 33554431);
        // xor_gpu<<<64, 1024>>>(poly, offset-33488896, 33488896, 33553920, 33554176, 33554430, 33554431);
        // xor_gpu<<<16256, 1024>>>(poly, offset-33423360, 33423360, 33488896, 33553920, 33554176, 33554430, 33554431);
        // xor_gpu<<<16384, 1024>>>(poly, offset-16777216, 16777216, 33423360, 33488896, 33553920, 33554176, 33554430, 33554431);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<25)-33554432, 16777216, 33423360, 33488896, 33553920, 33554176, 33554430, 33554431);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<25)-33554431, 16777216, 33423360, 33488896, 33553920, 33554176, 33554430);
        // xor_gpu<<<1, 254>>>(poly, offset+(1<<25)-33554430, 16777216, 33423360, 33488896, 33553920, 33554176);
        // xor_gpu<<<1, 256>>>(poly, offset+(1<<25)-33554176, 16777216, 33423360, 33488896, 33553920);
        // xor_gpu<<<127, 512>>>(poly, offset+(1<<25)-33553920, 16777216, 33423360, 33488896);
        // xor_gpu<<<64, 1024>>>(poly, offset+(1<<25)-33488896, 16777216, 33423360);
        // xor_gpu<<<16256, 1024>>>(poly, offset+(1<<25)-33423360, 16777216);
    }

    for(int offset=(1<<26);offset<(1<<logn);offset+=(1<<(26+1))) {
        // for(int i=offset-67108863;i<=offset-1-67108860;++i)poly[i]^=poly[i+67108863];
        // for(int i=offset-67108860;i<=offset-1-67108608;++i)poly[i]^=poly[i+67108860]^poly[i+67108863];
        // for(int i=offset-67108608;i<=offset-1-67107840;++i)poly[i]^=poly[i+67108608]^poly[i+67108860]^poly[i+67108863];
        // for(int i=offset-67107840;i<=offset-1-67043328;++i)poly[i]^=poly[i+67107840]^poly[i+67108608]^poly[i+67108860]^poly[i+67108863];
        // for(int i=offset-67043328;i<=offset-1-66846720;++i)poly[i]^=poly[i+67043328]^poly[i+67107840]^poly[i+67108608]^poly[i+67108860]^poly[i+67108863];
        // for(int i=offset-66846720;i<=offset-1-50331648;++i)poly[i]^=poly[i+66846720]^poly[i+67043328]^poly[i+67107840]^poly[i+67108608]^poly[i+67108860]^poly[i+67108863];
        // for(int i=offset-50331648;i<=offset-1-0;++i)poly[i]^=poly[i+50331648]^poly[i+66846720]^poly[i+67043328]^poly[i+67107840]^poly[i+67108608]^poly[i+67108860]^poly[i+67108863];
        // for(int i=offset+(1<<26)-67108864;i<=offset+(1<<26)-1-67108863;++i)poly[i]^=poly[i+50331648]^poly[i+66846720]^poly[i+67043328]^poly[i+67107840]^poly[i+67108608]^poly[i+67108860]^poly[i+67108863];
        // for(int i=offset+(1<<26)-67108863;i<=offset+(1<<26)-1-67108860;++i)poly[i]^=poly[i+50331648]^poly[i+66846720]^poly[i+67043328]^poly[i+67107840]^poly[i+67108608]^poly[i+67108860];
        // for(int i=offset+(1<<26)-67108860;i<=offset+(1<<26)-1-67108608;++i)poly[i]^=poly[i+50331648]^poly[i+66846720]^poly[i+67043328]^poly[i+67107840]^poly[i+67108608];
        // for(int i=offset+(1<<26)-67108608;i<=offset+(1<<26)-1-67107840;++i)poly[i]^=poly[i+50331648]^poly[i+66846720]^poly[i+67043328]^poly[i+67107840];
        // for(int i=offset+(1<<26)-67107840;i<=offset+(1<<26)-1-67043328;++i)poly[i]^=poly[i+50331648]^poly[i+66846720]^poly[i+67043328];
        // for(int i=offset+(1<<26)-67043328;i<=offset+(1<<26)-1-66846720;++i)poly[i]^=poly[i+50331648]^poly[i+66846720];
        // for(int i=offset+(1<<26)-66846720;i<=offset+(1<<26)-1-50331648;++i)poly[i]^=poly[i+50331648];
        // xor_gpu<<<1, 3>>>(poly, offset-67108863, 67108863);
        // xor_gpu<<<1, 252>>>(poly, offset-67108860, 67108860, 67108863);
        // xor_gpu<<<1, 768>>>(poly, offset-67108608, 67108608, 67108860, 67108863);
        // xor_gpu<<<63, 1024>>>(poly, offset-67107840, 67107840, 67108608, 67108860, 67108863);
        // xor_gpu<<<192, 1024>>>(poly, offset-67043328, 67043328, 67107840, 67108608, 67108860, 67108863);
        // xor_gpu<<<16128, 1024>>>(poly, offset-66846720, 66846720, 67043328, 67107840, 67108608, 67108860, 67108863);
        // xor_gpu<<<49152, 1024>>>(poly, offset-50331648, 50331648, 66846720, 67043328, 67107840, 67108608, 67108860, 67108863);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<26)-67108864, 50331648, 66846720, 67043328, 67107840, 67108608, 67108860, 67108863);
        // xor_gpu<<<1, 3>>>(poly, offset+(1<<26)-67108863, 50331648, 66846720, 67043328, 67107840, 67108608, 67108860);
        // xor_gpu<<<1, 252>>>(poly, offset+(1<<26)-67108860, 50331648, 66846720, 67043328, 67107840, 67108608);
        // xor_gpu<<<1, 768>>>(poly, offset+(1<<26)-67108608, 50331648, 66846720, 67043328, 67107840);
        // xor_gpu<<<63, 1024>>>(poly, offset+(1<<26)-67107840, 50331648, 66846720, 67043328);
        // xor_gpu<<<192, 1024>>>(poly, offset+(1<<26)-67043328, 50331648, 66846720);
        // xor_gpu<<<16128, 1024>>>(poly, offset+(1<<26)-66846720, 50331648);
    }

    for(int offset=(1<<27);offset<(1<<logn);offset+=(1<<(27+1))) {
        // for(int i=offset-134217727;i<=offset-1-134217726;++i)poly[i]^=poly[i+134217727];
        // for(int i=offset-134217726;i<=offset-1-134217724;++i)poly[i]^=poly[i+134217726]^poly[i+134217727];
        // for(int i=offset-134217724;i<=offset-1-134217720;++i)poly[i]^=poly[i+134217724]^poly[i+134217726]^poly[i+134217727];
        // for(int i=offset-134217720;i<=offset-1-134217472;++i)poly[i]^=poly[i+134217720]^poly[i+134217724]^poly[i+134217726]^poly[i+134217727];
        // for(int i=offset-134217472;i<=offset-1-134217216;++i)poly[i]^=poly[i+134217472]^poly[i+134217720]^poly[i+134217724]^poly[i+134217726]^poly[i+134217727];
        // for(int i=offset-134217216;i<=offset-1-134216704;++i)poly[i]^=poly[i+134217216]^poly[i+134217472]^poly[i+134217720]^poly[i+134217724]^poly[i+134217726]^poly[i+134217727];
        // for(int i=offset-134216704;i<=offset-1-134215680;++i)poly[i]^=poly[i+134216704]^poly[i+134217216]^poly[i+134217472]^poly[i+134217720]^poly[i+134217724]^poly[i+134217726]^poly[i+134217727];
        // for(int i=offset-134215680;i<=offset-1-134152192;++i)poly[i]^=poly[i+134215680]^poly[i+134216704]^poly[i+134217216]^poly[i+134217472]^poly[i+134217720]^poly[i+134217724]^poly[i+134217726]^poly[i+134217727];
        // for(int i=offset-134152192;i<=offset-1-134086656;++i)poly[i]^=poly[i+134152192]^poly[i+134215680]^poly[i+134216704]^poly[i+134217216]^poly[i+134217472]^poly[i+134217720]^poly[i+134217724]^poly[i+134217726]^poly[i+134217727];
        // for(int i=offset-134086656;i<=offset-1-133955584;++i)poly[i]^=poly[i+134086656]^poly[i+134152192]^poly[i+134215680]^poly[i+134216704]^poly[i+134217216]^poly[i+134217472]^poly[i+134217720]^poly[i+134217724]^poly[i+134217726]^poly[i+134217727];
        // for(int i=offset-133955584;i<=offset-1-133693440;++i)poly[i]^=poly[i+133955584]^poly[i+134086656]^poly[i+134152192]^poly[i+134215680]^poly[i+134216704]^poly[i+134217216]^poly[i+134217472]^poly[i+134217720]^poly[i+134217724]^poly[i+134217726]^poly[i+134217727];
        // for(int i=offset-133693440;i<=offset-1-117440512;++i)poly[i]^=poly[i+133693440]^poly[i+133955584]^poly[i+134086656]^poly[i+134152192]^poly[i+134215680]^poly[i+134216704]^poly[i+134217216]^poly[i+134217472]^poly[i+134217720]^poly[i+134217724]^poly[i+134217726]^poly[i+134217727];
        // for(int i=offset-117440512;i<=offset-1-100663296;++i)poly[i]^=poly[i+117440512]^poly[i+133693440]^poly[i+133955584]^poly[i+134086656]^poly[i+134152192]^poly[i+134215680]^poly[i+134216704]^poly[i+134217216]^poly[i+134217472]^poly[i+134217720]^poly[i+134217724]^poly[i+134217726]^poly[i+134217727];
        // for(int i=offset-100663296;i<=offset-1-67108864;++i)poly[i]^=poly[i+100663296]^poly[i+117440512]^poly[i+133693440]^poly[i+133955584]^poly[i+134086656]^poly[i+134152192]^poly[i+134215680]^poly[i+134216704]^poly[i+134217216]^poly[i+134217472]^poly[i+134217720]^poly[i+134217724]^poly[i+134217726]^poly[i+134217727];
        // for(int i=offset-67108864;i<=offset-1-0;++i)poly[i]^=poly[i+67108864]^poly[i+100663296]^poly[i+117440512]^poly[i+133693440]^poly[i+133955584]^poly[i+134086656]^poly[i+134152192]^poly[i+134215680]^poly[i+134216704]^poly[i+134217216]^poly[i+134217472]^poly[i+134217720]^poly[i+134217724]^poly[i+134217726]^poly[i+134217727];
        // for(int i=offset+(1<<27)-134217728;i<=offset+(1<<27)-1-134217727;++i)poly[i]^=poly[i+67108864]^poly[i+100663296]^poly[i+117440512]^poly[i+133693440]^poly[i+133955584]^poly[i+134086656]^poly[i+134152192]^poly[i+134215680]^poly[i+134216704]^poly[i+134217216]^poly[i+134217472]^poly[i+134217720]^poly[i+134217724]^poly[i+134217726]^poly[i+134217727];
        // for(int i=offset+(1<<27)-134217727;i<=offset+(1<<27)-1-134217726;++i)poly[i]^=poly[i+67108864]^poly[i+100663296]^poly[i+117440512]^poly[i+133693440]^poly[i+133955584]^poly[i+134086656]^poly[i+134152192]^poly[i+134215680]^poly[i+134216704]^poly[i+134217216]^poly[i+134217472]^poly[i+134217720]^poly[i+134217724]^poly[i+134217726];
        // for(int i=offset+(1<<27)-134217726;i<=offset+(1<<27)-1-134217724;++i)poly[i]^=poly[i+67108864]^poly[i+100663296]^poly[i+117440512]^poly[i+133693440]^poly[i+133955584]^poly[i+134086656]^poly[i+134152192]^poly[i+134215680]^poly[i+134216704]^poly[i+134217216]^poly[i+134217472]^poly[i+134217720]^poly[i+134217724];
        // for(int i=offset+(1<<27)-134217724;i<=offset+(1<<27)-1-134217720;++i)poly[i]^=poly[i+67108864]^poly[i+100663296]^poly[i+117440512]^poly[i+133693440]^poly[i+133955584]^poly[i+134086656]^poly[i+134152192]^poly[i+134215680]^poly[i+134216704]^poly[i+134217216]^poly[i+134217472]^poly[i+134217720];
        // for(int i=offset+(1<<27)-134217720;i<=offset+(1<<27)-1-134217472;++i)poly[i]^=poly[i+67108864]^poly[i+100663296]^poly[i+117440512]^poly[i+133693440]^poly[i+133955584]^poly[i+134086656]^poly[i+134152192]^poly[i+134215680]^poly[i+134216704]^poly[i+134217216]^poly[i+134217472];
        // for(int i=offset+(1<<27)-134217472;i<=offset+(1<<27)-1-134217216;++i)poly[i]^=poly[i+67108864]^poly[i+100663296]^poly[i+117440512]^poly[i+133693440]^poly[i+133955584]^poly[i+134086656]^poly[i+134152192]^poly[i+134215680]^poly[i+134216704]^poly[i+134217216];
        // for(int i=offset+(1<<27)-134217216;i<=offset+(1<<27)-1-134216704;++i)poly[i]^=poly[i+67108864]^poly[i+100663296]^poly[i+117440512]^poly[i+133693440]^poly[i+133955584]^poly[i+134086656]^poly[i+134152192]^poly[i+134215680]^poly[i+134216704];
        // for(int i=offset+(1<<27)-134216704;i<=offset+(1<<27)-1-134215680;++i)poly[i]^=poly[i+67108864]^poly[i+100663296]^poly[i+117440512]^poly[i+133693440]^poly[i+133955584]^poly[i+134086656]^poly[i+134152192]^poly[i+134215680];
        // for(int i=offset+(1<<27)-134215680;i<=offset+(1<<27)-1-134152192;++i)poly[i]^=poly[i+67108864]^poly[i+100663296]^poly[i+117440512]^poly[i+133693440]^poly[i+133955584]^poly[i+134086656]^poly[i+134152192];
        // for(int i=offset+(1<<27)-134152192;i<=offset+(1<<27)-1-134086656;++i)poly[i]^=poly[i+67108864]^poly[i+100663296]^poly[i+117440512]^poly[i+133693440]^poly[i+133955584]^poly[i+134086656];
        // for(int i=offset+(1<<27)-134086656;i<=offset+(1<<27)-1-133955584;++i)poly[i]^=poly[i+67108864]^poly[i+100663296]^poly[i+117440512]^poly[i+133693440]^poly[i+133955584];
        // for(int i=offset+(1<<27)-133955584;i<=offset+(1<<27)-1-133693440;++i)poly[i]^=poly[i+67108864]^poly[i+100663296]^poly[i+117440512]^poly[i+133693440];
        // for(int i=offset+(1<<27)-133693440;i<=offset+(1<<27)-1-117440512;++i)poly[i]^=poly[i+67108864]^poly[i+100663296]^poly[i+117440512];
        // for(int i=offset+(1<<27)-117440512;i<=offset+(1<<27)-1-100663296;++i)poly[i]^=poly[i+67108864]^poly[i+100663296];
        // for(int i=offset+(1<<27)-100663296;i<=offset+(1<<27)-1-67108864;++i)poly[i]^=poly[i+67108864];
        // xor_gpu<<<1, 1>>>(poly, offset-134217727, 134217727);
        // xor_gpu<<<1, 2>>>(poly, offset-134217726, 134217726, 134217727);
        // xor_gpu<<<1, 4>>>(poly, offset-134217724, 134217724, 134217726, 134217727);
        // xor_gpu<<<1, 248>>>(poly, offset-134217720, 134217720, 134217724, 134217726, 134217727);
        // xor_gpu<<<1, 256>>>(poly, offset-134217472, 134217472, 134217720, 134217724, 134217726, 134217727);
        // xor_gpu<<<1, 512>>>(poly, offset-134217216, 134217216, 134217472, 134217720, 134217724, 134217726, 134217727);
        // xor_gpu<<<1, 1024>>>(poly, offset-134216704, 134216704, 134217216, 134217472, 134217720, 134217724, 134217726, 134217727);
        // xor_gpu<<<62, 1024>>>(poly, offset-134215680, 134215680, 134216704, 134217216, 134217472, 134217720, 134217724, 134217726, 134217727);
        // xor_gpu<<<64, 1024>>>(poly, offset-134152192, 134152192, 134215680, 134216704, 134217216, 134217472, 134217720, 134217724, 134217726, 134217727);
        // xor_gpu<<<128, 1024>>>(poly, offset-134086656, 134086656, 134152192, 134215680, 134216704, 134217216, 134217472, 134217720, 134217724, 134217726, 134217727);
        // xor_gpu<<<256, 1024>>>(poly, offset-133955584, 133955584, 134086656, 134152192, 134215680, 134216704, 134217216, 134217472, 134217720, 134217724, 134217726, 134217727);
        // xor_gpu<<<15872, 1024>>>(poly, offset-133693440, 133693440, 133955584, 134086656, 134152192, 134215680, 134216704, 134217216, 134217472, 134217720, 134217724, 134217726, 134217727);
        // xor_gpu<<<16384, 1024>>>(poly, offset-117440512, 117440512, 133693440, 133955584, 134086656, 134152192, 134215680, 134216704, 134217216, 134217472, 134217720, 134217724, 134217726, 134217727);
        // xor_gpu<<<32768, 1024>>>(poly, offset-100663296, 100663296, 117440512, 133693440, 133955584, 134086656, 134152192, 134215680, 134216704, 134217216, 134217472, 134217720, 134217724, 134217726, 134217727);
        // xor_gpu<<<65536, 1024>>>(poly, offset-67108864, 67108864, 100663296, 117440512, 133693440, 133955584, 134086656, 134152192, 134215680, 134216704, 134217216, 134217472, 134217720, 134217724, 134217726, 134217727);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<27)-134217728, 67108864, 100663296, 117440512, 133693440, 133955584, 134086656, 134152192, 134215680, 134216704, 134217216, 134217472, 134217720, 134217724, 134217726, 134217727);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<27)-134217727, 67108864, 100663296, 117440512, 133693440, 133955584, 134086656, 134152192, 134215680, 134216704, 134217216, 134217472, 134217720, 134217724, 134217726);
        // xor_gpu<<<1, 2>>>(poly, offset+(1<<27)-134217726, 67108864, 100663296, 117440512, 133693440, 133955584, 134086656, 134152192, 134215680, 134216704, 134217216, 134217472, 134217720, 134217724);
        // xor_gpu<<<1, 4>>>(poly, offset+(1<<27)-134217724, 67108864, 100663296, 117440512, 133693440, 133955584, 134086656, 134152192, 134215680, 134216704, 134217216, 134217472, 134217720);
        // xor_gpu<<<1, 248>>>(poly, offset+(1<<27)-134217720, 67108864, 100663296, 117440512, 133693440, 133955584, 134086656, 134152192, 134215680, 134216704, 134217216, 134217472);
        // xor_gpu<<<1, 256>>>(poly, offset+(1<<27)-134217472, 67108864, 100663296, 117440512, 133693440, 133955584, 134086656, 134152192, 134215680, 134216704, 134217216);
        // xor_gpu<<<1, 512>>>(poly, offset+(1<<27)-134217216, 67108864, 100663296, 117440512, 133693440, 133955584, 134086656, 134152192, 134215680, 134216704);
        // xor_gpu<<<1, 1024>>>(poly, offset+(1<<27)-134216704, 67108864, 100663296, 117440512, 133693440, 133955584, 134086656, 134152192, 134215680);
        // xor_gpu<<<62, 1024>>>(poly, offset+(1<<27)-134215680, 67108864, 100663296, 117440512, 133693440, 133955584, 134086656, 134152192);
        // xor_gpu<<<64, 1024>>>(poly, offset+(1<<27)-134152192, 67108864, 100663296, 117440512, 133693440, 133955584, 134086656);
        // xor_gpu<<<128, 1024>>>(poly, offset+(1<<27)-134086656, 67108864, 100663296, 117440512, 133693440, 133955584);
        // xor_gpu<<<256, 1024>>>(poly, offset+(1<<27)-133955584, 67108864, 100663296, 117440512, 133693440);
        // xor_gpu<<<15872, 1024>>>(poly, offset+(1<<27)-133693440, 67108864, 100663296, 117440512);
        // xor_gpu<<<16384, 1024>>>(poly, offset+(1<<27)-117440512, 67108864, 100663296);
        // xor_gpu<<<32768, 1024>>>(poly, offset+(1<<27)-100663296, 67108864);
    }

    for(int offset=(1<<28);offset<(1<<logn);offset+=(1<<(28+1))) {
        // for(int i=offset-268435455;i<=offset-1-268435440;++i)poly[i]^=poly[i+268435455];
        // for(int i=offset-268435440;i<=offset-1-268435200;++i)poly[i]^=poly[i+268435440]^poly[i+268435455];
        // for(int i=offset-268435200;i<=offset-1-268431360;++i)poly[i]^=poly[i+268435200]^poly[i+268435440]^poly[i+268435455];
        // for(int i=offset-268431360;i<=offset-1-268369920;++i)poly[i]^=poly[i+268431360]^poly[i+268435200]^poly[i+268435440]^poly[i+268435455];
        // for(int i=offset-268369920;i<=offset-1-267386880;++i)poly[i]^=poly[i+268369920]^poly[i+268431360]^poly[i+268435200]^poly[i+268435440]^poly[i+268435455];
        // for(int i=offset-267386880;i<=offset-1-251658240;++i)poly[i]^=poly[i+267386880]^poly[i+268369920]^poly[i+268431360]^poly[i+268435200]^poly[i+268435440]^poly[i+268435455];
        // for(int i=offset-251658240;i<=offset-1-0;++i)poly[i]^=poly[i+251658240]^poly[i+267386880]^poly[i+268369920]^poly[i+268431360]^poly[i+268435200]^poly[i+268435440]^poly[i+268435455];
        // for(int i=offset+(1<<28)-268435456;i<=offset+(1<<28)-1-268435455;++i)poly[i]^=poly[i+251658240]^poly[i+267386880]^poly[i+268369920]^poly[i+268431360]^poly[i+268435200]^poly[i+268435440]^poly[i+268435455];
        // for(int i=offset+(1<<28)-268435455;i<=offset+(1<<28)-1-268435440;++i)poly[i]^=poly[i+251658240]^poly[i+267386880]^poly[i+268369920]^poly[i+268431360]^poly[i+268435200]^poly[i+268435440];
        // for(int i=offset+(1<<28)-268435440;i<=offset+(1<<28)-1-268435200;++i)poly[i]^=poly[i+251658240]^poly[i+267386880]^poly[i+268369920]^poly[i+268431360]^poly[i+268435200];
        // for(int i=offset+(1<<28)-268435200;i<=offset+(1<<28)-1-268431360;++i)poly[i]^=poly[i+251658240]^poly[i+267386880]^poly[i+268369920]^poly[i+268431360];
        // for(int i=offset+(1<<28)-268431360;i<=offset+(1<<28)-1-268369920;++i)poly[i]^=poly[i+251658240]^poly[i+267386880]^poly[i+268369920];
        // for(int i=offset+(1<<28)-268369920;i<=offset+(1<<28)-1-267386880;++i)poly[i]^=poly[i+251658240]^poly[i+267386880];
        // for(int i=offset+(1<<28)-267386880;i<=offset+(1<<28)-1-251658240;++i)poly[i]^=poly[i+251658240];
        // xor_gpu<<<1, 15>>>(poly, offset-268435455, 268435455);
        // xor_gpu<<<1, 240>>>(poly, offset-268435440, 268435440, 268435455);
        // xor_gpu<<<15, 256>>>(poly, offset-268435200, 268435200, 268435440, 268435455);
        // xor_gpu<<<60, 1024>>>(poly, offset-268431360, 268431360, 268435200, 268435440, 268435455);
        // xor_gpu<<<960, 1024>>>(poly, offset-268369920, 268369920, 268431360, 268435200, 268435440, 268435455);
        // xor_gpu<<<15360, 1024>>>(poly, offset-267386880, 267386880, 268369920, 268431360, 268435200, 268435440, 268435455);
        // xor_gpu<<<245760, 1024>>>(poly, offset-251658240, 251658240, 267386880, 268369920, 268431360, 268435200, 268435440, 268435455);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<28)-268435456, 251658240, 267386880, 268369920, 268431360, 268435200, 268435440, 268435455);
        // xor_gpu<<<1, 15>>>(poly, offset+(1<<28)-268435455, 251658240, 267386880, 268369920, 268431360, 268435200, 268435440);
        // xor_gpu<<<1, 240>>>(poly, offset+(1<<28)-268435440, 251658240, 267386880, 268369920, 268431360, 268435200);
        // xor_gpu<<<15, 256>>>(poly, offset+(1<<28)-268435200, 251658240, 267386880, 268369920, 268431360);
        // xor_gpu<<<60, 1024>>>(poly, offset+(1<<28)-268431360, 251658240, 267386880, 268369920);
        // xor_gpu<<<960, 1024>>>(poly, offset+(1<<28)-268369920, 251658240, 267386880);
        // xor_gpu<<<15360, 1024>>>(poly, offset+(1<<28)-267386880, 251658240);
    }

    for(int offset=(1<<29);offset<(1<<logn);offset+=(1<<(29+1))) {
        // for(int i=offset-536870911;i<=offset-1-536870910;++i)poly[i]^=poly[i+536870911];
        // for(int i=offset-536870910;i<=offset-1-536870896;++i)poly[i]^=poly[i+536870910]^poly[i+536870911];
        // for(int i=offset-536870896;i<=offset-1-536870880;++i)poly[i]^=poly[i+536870896]^poly[i+536870910]^poly[i+536870911];
        // for(int i=offset-536870880;i<=offset-1-536870656;++i)poly[i]^=poly[i+536870880]^poly[i+536870896]^poly[i+536870910]^poly[i+536870911];
        // for(int i=offset-536870656;i<=offset-1-536870400;++i)poly[i]^=poly[i+536870656]^poly[i+536870880]^poly[i+536870896]^poly[i+536870910]^poly[i+536870911];
        // for(int i=offset-536870400;i<=offset-1-536866816;++i)poly[i]^=poly[i+536870400]^poly[i+536870656]^poly[i+536870880]^poly[i+536870896]^poly[i+536870910]^poly[i+536870911];
        // for(int i=offset-536866816;i<=offset-1-536862720;++i)poly[i]^=poly[i+536866816]^poly[i+536870400]^poly[i+536870656]^poly[i+536870880]^poly[i+536870896]^poly[i+536870910]^poly[i+536870911];
        // for(int i=offset-536862720;i<=offset-1-536805376;++i)poly[i]^=poly[i+536862720]^poly[i+536866816]^poly[i+536870400]^poly[i+536870656]^poly[i+536870880]^poly[i+536870896]^poly[i+536870910]^poly[i+536870911];
        // for(int i=offset-536805376;i<=offset-1-536739840;++i)poly[i]^=poly[i+536805376]^poly[i+536862720]^poly[i+536866816]^poly[i+536870400]^poly[i+536870656]^poly[i+536870880]^poly[i+536870896]^poly[i+536870910]^poly[i+536870911];
        // for(int i=offset-536739840;i<=offset-1-535822336;++i)poly[i]^=poly[i+536739840]^poly[i+536805376]^poly[i+536862720]^poly[i+536866816]^poly[i+536870400]^poly[i+536870656]^poly[i+536870880]^poly[i+536870896]^poly[i+536870910]^poly[i+536870911];
        // for(int i=offset-535822336;i<=offset-1-534773760;++i)poly[i]^=poly[i+535822336]^poly[i+536739840]^poly[i+536805376]^poly[i+536862720]^poly[i+536866816]^poly[i+536870400]^poly[i+536870656]^poly[i+536870880]^poly[i+536870896]^poly[i+536870910]^poly[i+536870911];
        // for(int i=offset-534773760;i<=offset-1-520093696;++i)poly[i]^=poly[i+534773760]^poly[i+535822336]^poly[i+536739840]^poly[i+536805376]^poly[i+536862720]^poly[i+536866816]^poly[i+536870400]^poly[i+536870656]^poly[i+536870880]^poly[i+536870896]^poly[i+536870910]^poly[i+536870911];
        // for(int i=offset-520093696;i<=offset-1-503316480;++i)poly[i]^=poly[i+520093696]^poly[i+534773760]^poly[i+535822336]^poly[i+536739840]^poly[i+536805376]^poly[i+536862720]^poly[i+536866816]^poly[i+536870400]^poly[i+536870656]^poly[i+536870880]^poly[i+536870896]^poly[i+536870910]^poly[i+536870911];
        // for(int i=offset-503316480;i<=offset-1-268435456;++i)poly[i]^=poly[i+503316480]^poly[i+520093696]^poly[i+534773760]^poly[i+535822336]^poly[i+536739840]^poly[i+536805376]^poly[i+536862720]^poly[i+536866816]^poly[i+536870400]^poly[i+536870656]^poly[i+536870880]^poly[i+536870896]^poly[i+536870910]^poly[i+536870911];
        // for(int i=offset-268435456;i<=offset-1-0;++i)poly[i]^=poly[i+268435456]^poly[i+503316480]^poly[i+520093696]^poly[i+534773760]^poly[i+535822336]^poly[i+536739840]^poly[i+536805376]^poly[i+536862720]^poly[i+536866816]^poly[i+536870400]^poly[i+536870656]^poly[i+536870880]^poly[i+536870896]^poly[i+536870910]^poly[i+536870911];
        // for(int i=offset+(1<<29)-536870912;i<=offset+(1<<29)-1-536870911;++i)poly[i]^=poly[i+268435456]^poly[i+503316480]^poly[i+520093696]^poly[i+534773760]^poly[i+535822336]^poly[i+536739840]^poly[i+536805376]^poly[i+536862720]^poly[i+536866816]^poly[i+536870400]^poly[i+536870656]^poly[i+536870880]^poly[i+536870896]^poly[i+536870910]^poly[i+536870911];
        // for(int i=offset+(1<<29)-536870911;i<=offset+(1<<29)-1-536870910;++i)poly[i]^=poly[i+268435456]^poly[i+503316480]^poly[i+520093696]^poly[i+534773760]^poly[i+535822336]^poly[i+536739840]^poly[i+536805376]^poly[i+536862720]^poly[i+536866816]^poly[i+536870400]^poly[i+536870656]^poly[i+536870880]^poly[i+536870896]^poly[i+536870910];
        // for(int i=offset+(1<<29)-536870910;i<=offset+(1<<29)-1-536870896;++i)poly[i]^=poly[i+268435456]^poly[i+503316480]^poly[i+520093696]^poly[i+534773760]^poly[i+535822336]^poly[i+536739840]^poly[i+536805376]^poly[i+536862720]^poly[i+536866816]^poly[i+536870400]^poly[i+536870656]^poly[i+536870880]^poly[i+536870896];
        // for(int i=offset+(1<<29)-536870896;i<=offset+(1<<29)-1-536870880;++i)poly[i]^=poly[i+268435456]^poly[i+503316480]^poly[i+520093696]^poly[i+534773760]^poly[i+535822336]^poly[i+536739840]^poly[i+536805376]^poly[i+536862720]^poly[i+536866816]^poly[i+536870400]^poly[i+536870656]^poly[i+536870880];
        // for(int i=offset+(1<<29)-536870880;i<=offset+(1<<29)-1-536870656;++i)poly[i]^=poly[i+268435456]^poly[i+503316480]^poly[i+520093696]^poly[i+534773760]^poly[i+535822336]^poly[i+536739840]^poly[i+536805376]^poly[i+536862720]^poly[i+536866816]^poly[i+536870400]^poly[i+536870656];
        // for(int i=offset+(1<<29)-536870656;i<=offset+(1<<29)-1-536870400;++i)poly[i]^=poly[i+268435456]^poly[i+503316480]^poly[i+520093696]^poly[i+534773760]^poly[i+535822336]^poly[i+536739840]^poly[i+536805376]^poly[i+536862720]^poly[i+536866816]^poly[i+536870400];
        // for(int i=offset+(1<<29)-536870400;i<=offset+(1<<29)-1-536866816;++i)poly[i]^=poly[i+268435456]^poly[i+503316480]^poly[i+520093696]^poly[i+534773760]^poly[i+535822336]^poly[i+536739840]^poly[i+536805376]^poly[i+536862720]^poly[i+536866816];
        // for(int i=offset+(1<<29)-536866816;i<=offset+(1<<29)-1-536862720;++i)poly[i]^=poly[i+268435456]^poly[i+503316480]^poly[i+520093696]^poly[i+534773760]^poly[i+535822336]^poly[i+536739840]^poly[i+536805376]^poly[i+536862720];
        // for(int i=offset+(1<<29)-536862720;i<=offset+(1<<29)-1-536805376;++i)poly[i]^=poly[i+268435456]^poly[i+503316480]^poly[i+520093696]^poly[i+534773760]^poly[i+535822336]^poly[i+536739840]^poly[i+536805376];
        // for(int i=offset+(1<<29)-536805376;i<=offset+(1<<29)-1-536739840;++i)poly[i]^=poly[i+268435456]^poly[i+503316480]^poly[i+520093696]^poly[i+534773760]^poly[i+535822336]^poly[i+536739840];
        // for(int i=offset+(1<<29)-536739840;i<=offset+(1<<29)-1-535822336;++i)poly[i]^=poly[i+268435456]^poly[i+503316480]^poly[i+520093696]^poly[i+534773760]^poly[i+535822336];
        // for(int i=offset+(1<<29)-535822336;i<=offset+(1<<29)-1-534773760;++i)poly[i]^=poly[i+268435456]^poly[i+503316480]^poly[i+520093696]^poly[i+534773760];
        // for(int i=offset+(1<<29)-534773760;i<=offset+(1<<29)-1-520093696;++i)poly[i]^=poly[i+268435456]^poly[i+503316480]^poly[i+520093696];
        // for(int i=offset+(1<<29)-520093696;i<=offset+(1<<29)-1-503316480;++i)poly[i]^=poly[i+268435456]^poly[i+503316480];
        // for(int i=offset+(1<<29)-503316480;i<=offset+(1<<29)-1-268435456;++i)poly[i]^=poly[i+268435456];
        // xor_gpu<<<1, 1>>>(poly, offset-536870911, 536870911);
        // xor_gpu<<<1, 14>>>(poly, offset-536870910, 536870910, 536870911);
        // xor_gpu<<<1, 16>>>(poly, offset-536870896, 536870896, 536870910, 536870911);
        // xor_gpu<<<1, 224>>>(poly, offset-536870880, 536870880, 536870896, 536870910, 536870911);
        // xor_gpu<<<1, 256>>>(poly, offset-536870656, 536870656, 536870880, 536870896, 536870910, 536870911);
        // xor_gpu<<<7, 512>>>(poly, offset-536870400, 536870400, 536870656, 536870880, 536870896, 536870910, 536870911);
        // xor_gpu<<<4, 1024>>>(poly, offset-536866816, 536866816, 536870400, 536870656, 536870880, 536870896, 536870910, 536870911);
        // xor_gpu<<<56, 1024>>>(poly, offset-536862720, 536862720, 536866816, 536870400, 536870656, 536870880, 536870896, 536870910, 536870911);
        // xor_gpu<<<64, 1024>>>(poly, offset-536805376, 536805376, 536862720, 536866816, 536870400, 536870656, 536870880, 536870896, 536870910, 536870911);
        // xor_gpu<<<896, 1024>>>(poly, offset-536739840, 536739840, 536805376, 536862720, 536866816, 536870400, 536870656, 536870880, 536870896, 536870910, 536870911);
        // xor_gpu<<<1024, 1024>>>(poly, offset-535822336, 535822336, 536739840, 536805376, 536862720, 536866816, 536870400, 536870656, 536870880, 536870896, 536870910, 536870911);
        // xor_gpu<<<14336, 1024>>>(poly, offset-534773760, 534773760, 535822336, 536739840, 536805376, 536862720, 536866816, 536870400, 536870656, 536870880, 536870896, 536870910, 536870911);
        // xor_gpu<<<16384, 1024>>>(poly, offset-520093696, 520093696, 534773760, 535822336, 536739840, 536805376, 536862720, 536866816, 536870400, 536870656, 536870880, 536870896, 536870910, 536870911);
        // xor_gpu<<<229376, 1024>>>(poly, offset-503316480, 503316480, 520093696, 534773760, 535822336, 536739840, 536805376, 536862720, 536866816, 536870400, 536870656, 536870880, 536870896, 536870910, 536870911);
        // xor_gpu<<<262144, 1024>>>(poly, offset-268435456, 268435456, 503316480, 520093696, 534773760, 535822336, 536739840, 536805376, 536862720, 536866816, 536870400, 536870656, 536870880, 536870896, 536870910, 536870911);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<29)-536870912, 268435456, 503316480, 520093696, 534773760, 535822336, 536739840, 536805376, 536862720, 536866816, 536870400, 536870656, 536870880, 536870896, 536870910, 536870911);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<29)-536870911, 268435456, 503316480, 520093696, 534773760, 535822336, 536739840, 536805376, 536862720, 536866816, 536870400, 536870656, 536870880, 536870896, 536870910);
        // xor_gpu<<<1, 14>>>(poly, offset+(1<<29)-536870910, 268435456, 503316480, 520093696, 534773760, 535822336, 536739840, 536805376, 536862720, 536866816, 536870400, 536870656, 536870880, 536870896);
        // xor_gpu<<<1, 16>>>(poly, offset+(1<<29)-536870896, 268435456, 503316480, 520093696, 534773760, 535822336, 536739840, 536805376, 536862720, 536866816, 536870400, 536870656, 536870880);
        // xor_gpu<<<1, 224>>>(poly, offset+(1<<29)-536870880, 268435456, 503316480, 520093696, 534773760, 535822336, 536739840, 536805376, 536862720, 536866816, 536870400, 536870656);
        // xor_gpu<<<1, 256>>>(poly, offset+(1<<29)-536870656, 268435456, 503316480, 520093696, 534773760, 535822336, 536739840, 536805376, 536862720, 536866816, 536870400);
        // xor_gpu<<<7, 512>>>(poly, offset+(1<<29)-536870400, 268435456, 503316480, 520093696, 534773760, 535822336, 536739840, 536805376, 536862720, 536866816);
        // xor_gpu<<<4, 1024>>>(poly, offset+(1<<29)-536866816, 268435456, 503316480, 520093696, 534773760, 535822336, 536739840, 536805376, 536862720);
        // xor_gpu<<<56, 1024>>>(poly, offset+(1<<29)-536862720, 268435456, 503316480, 520093696, 534773760, 535822336, 536739840, 536805376);
        // xor_gpu<<<64, 1024>>>(poly, offset+(1<<29)-536805376, 268435456, 503316480, 520093696, 534773760, 535822336, 536739840);
        // xor_gpu<<<896, 1024>>>(poly, offset+(1<<29)-536739840, 268435456, 503316480, 520093696, 534773760, 535822336);
        // xor_gpu<<<1024, 1024>>>(poly, offset+(1<<29)-535822336, 268435456, 503316480, 520093696, 534773760);
        // xor_gpu<<<14336, 1024>>>(poly, offset+(1<<29)-534773760, 268435456, 503316480, 520093696);
        // xor_gpu<<<16384, 1024>>>(poly, offset+(1<<29)-520093696, 268435456, 503316480);
        // xor_gpu<<<229376, 1024>>>(poly, offset+(1<<29)-503316480, 268435456);
    }

    for(int offset=(1<<30);offset<(1<<logn);offset+=(1<<(30+1))) {
        // for(int i=offset-1073741823;i<=offset-1-1073741820;++i)poly[i]^=poly[i+1073741823];
        // for(int i=offset-1073741820;i<=offset-1-1073741808;++i)poly[i]^=poly[i+1073741820]^poly[i+1073741823];
        // for(int i=offset-1073741808;i<=offset-1-1073741760;++i)poly[i]^=poly[i+1073741808]^poly[i+1073741820]^poly[i+1073741823];
        // for(int i=offset-1073741760;i<=offset-1-1073741568;++i)poly[i]^=poly[i+1073741760]^poly[i+1073741808]^poly[i+1073741820]^poly[i+1073741823];
        // for(int i=offset-1073741568;i<=offset-1-1073740800;++i)poly[i]^=poly[i+1073741568]^poly[i+1073741760]^poly[i+1073741808]^poly[i+1073741820]^poly[i+1073741823];
        // for(int i=offset-1073740800;i<=offset-1-1073737728;++i)poly[i]^=poly[i+1073740800]^poly[i+1073741568]^poly[i+1073741760]^poly[i+1073741808]^poly[i+1073741820]^poly[i+1073741823];
        // for(int i=offset-1073737728;i<=offset-1-1073725440;++i)poly[i]^=poly[i+1073737728]^poly[i+1073740800]^poly[i+1073741568]^poly[i+1073741760]^poly[i+1073741808]^poly[i+1073741820]^poly[i+1073741823];
        // for(int i=offset-1073725440;i<=offset-1-1073676288;++i)poly[i]^=poly[i+1073725440]^poly[i+1073737728]^poly[i+1073740800]^poly[i+1073741568]^poly[i+1073741760]^poly[i+1073741808]^poly[i+1073741820]^poly[i+1073741823];
        // for(int i=offset-1073676288;i<=offset-1-1073479680;++i)poly[i]^=poly[i+1073676288]^poly[i+1073725440]^poly[i+1073737728]^poly[i+1073740800]^poly[i+1073741568]^poly[i+1073741760]^poly[i+1073741808]^poly[i+1073741820]^poly[i+1073741823];
        // for(int i=offset-1073479680;i<=offset-1-1072693248;++i)poly[i]^=poly[i+1073479680]^poly[i+1073676288]^poly[i+1073725440]^poly[i+1073737728]^poly[i+1073740800]^poly[i+1073741568]^poly[i+1073741760]^poly[i+1073741808]^poly[i+1073741820]^poly[i+1073741823];
        // for(int i=offset-1072693248;i<=offset-1-1069547520;++i)poly[i]^=poly[i+1072693248]^poly[i+1073479680]^poly[i+1073676288]^poly[i+1073725440]^poly[i+1073737728]^poly[i+1073740800]^poly[i+1073741568]^poly[i+1073741760]^poly[i+1073741808]^poly[i+1073741820]^poly[i+1073741823];
        // for(int i=offset-1069547520;i<=offset-1-1056964608;++i)poly[i]^=poly[i+1069547520]^poly[i+1072693248]^poly[i+1073479680]^poly[i+1073676288]^poly[i+1073725440]^poly[i+1073737728]^poly[i+1073740800]^poly[i+1073741568]^poly[i+1073741760]^poly[i+1073741808]^poly[i+1073741820]^poly[i+1073741823];
        // for(int i=offset-1056964608;i<=offset-1-1006632960;++i)poly[i]^=poly[i+1056964608]^poly[i+1069547520]^poly[i+1072693248]^poly[i+1073479680]^poly[i+1073676288]^poly[i+1073725440]^poly[i+1073737728]^poly[i+1073740800]^poly[i+1073741568]^poly[i+1073741760]^poly[i+1073741808]^poly[i+1073741820]^poly[i+1073741823];
        // for(int i=offset-1006632960;i<=offset-1-805306368;++i)poly[i]^=poly[i+1006632960]^poly[i+1056964608]^poly[i+1069547520]^poly[i+1072693248]^poly[i+1073479680]^poly[i+1073676288]^poly[i+1073725440]^poly[i+1073737728]^poly[i+1073740800]^poly[i+1073741568]^poly[i+1073741760]^poly[i+1073741808]^poly[i+1073741820]^poly[i+1073741823];
        // for(int i=offset-805306368;i<=offset-1-0;++i)poly[i]^=poly[i+805306368]^poly[i+1006632960]^poly[i+1056964608]^poly[i+1069547520]^poly[i+1072693248]^poly[i+1073479680]^poly[i+1073676288]^poly[i+1073725440]^poly[i+1073737728]^poly[i+1073740800]^poly[i+1073741568]^poly[i+1073741760]^poly[i+1073741808]^poly[i+1073741820]^poly[i+1073741823];
        // for(int i=offset+(1<<30)-1073741824;i<=offset+(1<<30)-1-1073741823;++i)poly[i]^=poly[i+805306368]^poly[i+1006632960]^poly[i+1056964608]^poly[i+1069547520]^poly[i+1072693248]^poly[i+1073479680]^poly[i+1073676288]^poly[i+1073725440]^poly[i+1073737728]^poly[i+1073740800]^poly[i+1073741568]^poly[i+1073741760]^poly[i+1073741808]^poly[i+1073741820]^poly[i+1073741823];
        // for(int i=offset+(1<<30)-1073741823;i<=offset+(1<<30)-1-1073741820;++i)poly[i]^=poly[i+805306368]^poly[i+1006632960]^poly[i+1056964608]^poly[i+1069547520]^poly[i+1072693248]^poly[i+1073479680]^poly[i+1073676288]^poly[i+1073725440]^poly[i+1073737728]^poly[i+1073740800]^poly[i+1073741568]^poly[i+1073741760]^poly[i+1073741808]^poly[i+1073741820];
        // for(int i=offset+(1<<30)-1073741820;i<=offset+(1<<30)-1-1073741808;++i)poly[i]^=poly[i+805306368]^poly[i+1006632960]^poly[i+1056964608]^poly[i+1069547520]^poly[i+1072693248]^poly[i+1073479680]^poly[i+1073676288]^poly[i+1073725440]^poly[i+1073737728]^poly[i+1073740800]^poly[i+1073741568]^poly[i+1073741760]^poly[i+1073741808];
        // for(int i=offset+(1<<30)-1073741808;i<=offset+(1<<30)-1-1073741760;++i)poly[i]^=poly[i+805306368]^poly[i+1006632960]^poly[i+1056964608]^poly[i+1069547520]^poly[i+1072693248]^poly[i+1073479680]^poly[i+1073676288]^poly[i+1073725440]^poly[i+1073737728]^poly[i+1073740800]^poly[i+1073741568]^poly[i+1073741760];
        // for(int i=offset+(1<<30)-1073741760;i<=offset+(1<<30)-1-1073741568;++i)poly[i]^=poly[i+805306368]^poly[i+1006632960]^poly[i+1056964608]^poly[i+1069547520]^poly[i+1072693248]^poly[i+1073479680]^poly[i+1073676288]^poly[i+1073725440]^poly[i+1073737728]^poly[i+1073740800]^poly[i+1073741568];
        // for(int i=offset+(1<<30)-1073741568;i<=offset+(1<<30)-1-1073740800;++i)poly[i]^=poly[i+805306368]^poly[i+1006632960]^poly[i+1056964608]^poly[i+1069547520]^poly[i+1072693248]^poly[i+1073479680]^poly[i+1073676288]^poly[i+1073725440]^poly[i+1073737728]^poly[i+1073740800];
        // for(int i=offset+(1<<30)-1073740800;i<=offset+(1<<30)-1-1073737728;++i)poly[i]^=poly[i+805306368]^poly[i+1006632960]^poly[i+1056964608]^poly[i+1069547520]^poly[i+1072693248]^poly[i+1073479680]^poly[i+1073676288]^poly[i+1073725440]^poly[i+1073737728];
        // for(int i=offset+(1<<30)-1073737728;i<=offset+(1<<30)-1-1073725440;++i)poly[i]^=poly[i+805306368]^poly[i+1006632960]^poly[i+1056964608]^poly[i+1069547520]^poly[i+1072693248]^poly[i+1073479680]^poly[i+1073676288]^poly[i+1073725440];
        // for(int i=offset+(1<<30)-1073725440;i<=offset+(1<<30)-1-1073676288;++i)poly[i]^=poly[i+805306368]^poly[i+1006632960]^poly[i+1056964608]^poly[i+1069547520]^poly[i+1072693248]^poly[i+1073479680]^poly[i+1073676288];
        // for(int i=offset+(1<<30)-1073676288;i<=offset+(1<<30)-1-1073479680;++i)poly[i]^=poly[i+805306368]^poly[i+1006632960]^poly[i+1056964608]^poly[i+1069547520]^poly[i+1072693248]^poly[i+1073479680];
        // for(int i=offset+(1<<30)-1073479680;i<=offset+(1<<30)-1-1072693248;++i)poly[i]^=poly[i+805306368]^poly[i+1006632960]^poly[i+1056964608]^poly[i+1069547520]^poly[i+1072693248];
        // for(int i=offset+(1<<30)-1072693248;i<=offset+(1<<30)-1-1069547520;++i)poly[i]^=poly[i+805306368]^poly[i+1006632960]^poly[i+1056964608]^poly[i+1069547520];
        // for(int i=offset+(1<<30)-1069547520;i<=offset+(1<<30)-1-1056964608;++i)poly[i]^=poly[i+805306368]^poly[i+1006632960]^poly[i+1056964608];
        // for(int i=offset+(1<<30)-1056964608;i<=offset+(1<<30)-1-1006632960;++i)poly[i]^=poly[i+805306368]^poly[i+1006632960];
        // for(int i=offset+(1<<30)-1006632960;i<=offset+(1<<30)-1-805306368;++i)poly[i]^=poly[i+805306368];
        // xor_gpu<<<1, 3>>>(poly, offset-1073741823, 1073741823);
        // xor_gpu<<<1, 12>>>(poly, offset-1073741820, 1073741820, 1073741823);
        // xor_gpu<<<1, 48>>>(poly, offset-1073741808, 1073741808, 1073741820, 1073741823);
        // xor_gpu<<<1, 192>>>(poly, offset-1073741760, 1073741760, 1073741808, 1073741820, 1073741823);
        // xor_gpu<<<1, 768>>>(poly, offset-1073741568, 1073741568, 1073741760, 1073741808, 1073741820, 1073741823);
        // xor_gpu<<<3, 1024>>>(poly, offset-1073740800, 1073740800, 1073741568, 1073741760, 1073741808, 1073741820, 1073741823);
        // xor_gpu<<<12, 1024>>>(poly, offset-1073737728, 1073737728, 1073740800, 1073741568, 1073741760, 1073741808, 1073741820, 1073741823);
        // xor_gpu<<<48, 1024>>>(poly, offset-1073725440, 1073725440, 1073737728, 1073740800, 1073741568, 1073741760, 1073741808, 1073741820, 1073741823);
        // xor_gpu<<<192, 1024>>>(poly, offset-1073676288, 1073676288, 1073725440, 1073737728, 1073740800, 1073741568, 1073741760, 1073741808, 1073741820, 1073741823);
        // xor_gpu<<<768, 1024>>>(poly, offset-1073479680, 1073479680, 1073676288, 1073725440, 1073737728, 1073740800, 1073741568, 1073741760, 1073741808, 1073741820, 1073741823);
        // xor_gpu<<<3072, 1024>>>(poly, offset-1072693248, 1072693248, 1073479680, 1073676288, 1073725440, 1073737728, 1073740800, 1073741568, 1073741760, 1073741808, 1073741820, 1073741823);
        // xor_gpu<<<12288, 1024>>>(poly, offset-1069547520, 1069547520, 1072693248, 1073479680, 1073676288, 1073725440, 1073737728, 1073740800, 1073741568, 1073741760, 1073741808, 1073741820, 1073741823);
        // xor_gpu<<<49152, 1024>>>(poly, offset-1056964608, 1056964608, 1069547520, 1072693248, 1073479680, 1073676288, 1073725440, 1073737728, 1073740800, 1073741568, 1073741760, 1073741808, 1073741820, 1073741823);
        // xor_gpu<<<196608, 1024>>>(poly, offset-1006632960, 1006632960, 1056964608, 1069547520, 1072693248, 1073479680, 1073676288, 1073725440, 1073737728, 1073740800, 1073741568, 1073741760, 1073741808, 1073741820, 1073741823);
        // xor_gpu<<<786432, 1024>>>(poly, offset-805306368, 805306368, 1006632960, 1056964608, 1069547520, 1072693248, 1073479680, 1073676288, 1073725440, 1073737728, 1073740800, 1073741568, 1073741760, 1073741808, 1073741820, 1073741823);
        // xor_gpu<<<1, 1>>>(poly, offset+(1<<30)-1073741824, 805306368, 1006632960, 1056964608, 1069547520, 1072693248, 1073479680, 1073676288, 1073725440, 1073737728, 1073740800, 1073741568, 1073741760, 1073741808, 1073741820, 1073741823);
        // xor_gpu<<<1, 3>>>(poly, offset+(1<<30)-1073741823, 805306368, 1006632960, 1056964608, 1069547520, 1072693248, 1073479680, 1073676288, 1073725440, 1073737728, 1073740800, 1073741568, 1073741760, 1073741808, 1073741820);
        // xor_gpu<<<1, 12>>>(poly, offset+(1<<30)-1073741820, 805306368, 1006632960, 1056964608, 1069547520, 1072693248, 1073479680, 1073676288, 1073725440, 1073737728, 1073740800, 1073741568, 1073741760, 1073741808);
        // xor_gpu<<<1, 48>>>(poly, offset+(1<<30)-1073741808, 805306368, 1006632960, 1056964608, 1069547520, 1072693248, 1073479680, 1073676288, 1073725440, 1073737728, 1073740800, 1073741568, 1073741760);
        // xor_gpu<<<1, 192>>>(poly, offset+(1<<30)-1073741760, 805306368, 1006632960, 1056964608, 1069547520, 1072693248, 1073479680, 1073676288, 1073725440, 1073737728, 1073740800, 1073741568);
        // xor_gpu<<<1, 768>>>(poly, offset+(1<<30)-1073741568, 805306368, 1006632960, 1056964608, 1069547520, 1072693248, 1073479680, 1073676288, 1073725440, 1073737728, 1073740800);
        // xor_gpu<<<3, 1024>>>(poly, offset+(1<<30)-1073740800, 805306368, 1006632960, 1056964608, 1069547520, 1072693248, 1073479680, 1073676288, 1073725440, 1073737728);
        // xor_gpu<<<12, 1024>>>(poly, offset+(1<<30)-1073737728, 805306368, 1006632960, 1056964608, 1069547520, 1072693248, 1073479680, 1073676288, 1073725440);
        // xor_gpu<<<48, 1024>>>(poly, offset+(1<<30)-1073725440, 805306368, 1006632960, 1056964608, 1069547520, 1072693248, 1073479680, 1073676288);
        // xor_gpu<<<192, 1024>>>(poly, offset+(1<<30)-1073676288, 805306368, 1006632960, 1056964608, 1069547520, 1072693248, 1073479680);
        // xor_gpu<<<768, 1024>>>(poly, offset+(1<<30)-1073479680, 805306368, 1006632960, 1056964608, 1069547520, 1072693248);
        // xor_gpu<<<3072, 1024>>>(poly, offset+(1<<30)-1072693248, 805306368, 1006632960, 1056964608, 1069547520);
        // xor_gpu<<<12288, 1024>>>(poly, offset+(1<<30)-1069547520, 805306368, 1006632960, 1056964608);
        // xor_gpu<<<49152, 1024>>>(poly, offset+(1<<30)-1056964608, 805306368, 1006632960);
        // xor_gpu<<<196608, 1024>>>(poly, offset+(1<<30)-1006632960, 805306368);
    }
}
