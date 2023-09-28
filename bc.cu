/*
Copyright (C) 2017 Ming-Shing Chen

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

#include "bc.h"

#include <stdio.h>

#include "util.cuh"

#define BC_CODE_GEN

static inline
unsigned get_num_blocks( unsigned poly_len , unsigned blk_size ) {
	return poly_len/blk_size;
}


static inline
unsigned deg_si( unsigned si ) {
	return (1<<si);
}

static inline
unsigned get_si_2_pow( unsigned si , unsigned deg ) {
	unsigned si_deg = (1<<si);
	unsigned r=1;
	while( (si_deg<<r) < deg ) {
		r += 1;
	}
	return (1<<(r-1));
}

static inline
unsigned get_max_si( unsigned deg ) {
	unsigned si = 0;
	unsigned si_attempt = 1;
	uint64_t deg64 = deg;
	while( deg64 > ((1ULL)<<si_attempt) ) {
		si = si_attempt;
		si_attempt <<= 1;
	}
	return si;
}


//////////////////////////////////////////////////////////////////////


//#include <x86intrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#include "bc_to_mono_gen_code.cu"
#include "bc_to_lch_gen_code.cu"


#define LOG2(X) ((unsigned) (8*sizeof (unsigned long long) - __builtin_clzll((X)) - 1))
#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

__global__
void xor_down_d_2(u256 *poly256, bc_sto_t *poly, unsigned st, unsigned diff) {
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	u256_xor(*(poly256-i-1), *(u256*)(poly+st+diff-(i*4)-4));
}

__global__
void xor_down_d_3(bc_sto_t *poly, unsigned st, unsigned diff, unsigned start) {
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x + start;
	poly[st-i-1] ^= poly[st-i-1+diff];
}

void xor_down( bc_sto_t * poly , unsigned st , unsigned len , unsigned diff )
{
#if 0
	for( unsigned i=0;i<len;i++) {
		poly[st-i-1] ^= poly[st-i-1+diff];
	}
#else
	while( ((unsigned long)(poly+st)) & 31 ) {
		poly[st-1] ^= poly[st-1+diff];
		st--;
		len--;
		if( 0 == len ) break;
	}
	u256 * poly256 = (u256*)(poly+st);
	unsigned _len = len>>2;
	uint64_t nBlock = (_len + 1023) / 1024;
	xor_down_d_2<<<nBlock, 1024>>>(poly256, poly, st, diff);
	nBlock = (len - (_len<<2) + 1023) / 1024;
	xor_down_d_3<<<nBlock, 1024>>>(poly, st, diff, _len<<2);
#endif
}

static inline
void poly_div( bc_sto_t * poly , unsigned n_terms , unsigned blk_size , unsigned si , unsigned pow )
{
	if( 0 == si ) return;
	unsigned si_degree = deg_si(si)*pow;
	unsigned deg_diff = si_degree - pow;
	unsigned deg_blk = get_num_blocks( n_terms , blk_size ) -1;
#if 1
	xor_down( poly , (deg_blk-deg_diff+1)*blk_size , (deg_blk-si_degree+1)*blk_size , deg_diff*blk_size );
#else
	for(unsigned i=deg_blk;i>=si_degree;i--) {
		for(int j=((int)blk_size)-1;j>=0;j--) {
			poly[(i-deg_diff)*blk_size+j] ^= poly[i*blk_size+j];
		}
	}
#endif
}

static inline
void represent_in_si( bc_sto_t * poly , unsigned n_terms , unsigned blk_size , unsigned si )
{
	if( 0 == si ) return;
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned degree_basic_form_si = deg_si(si);
	if( degree_basic_form_si > degree_in_blocks ) return;

#if 1
	unsigned pow = get_si_2_pow( si , degree_in_blocks );
	while( 0 < pow ) {
		for(unsigned i=0;i<n_terms;i+= blk_size*2*pow*deg_si(si) ) {
			poly_div( poly + i , blk_size*2*pow*deg_si(si) , blk_size , si , pow );
		}
		pow >>= 1;
	}
#else
	unsigned pow = get_si_2_pow( si , degree_in_blocks );
	poly_div( poly , n_terms , blk_size , si , pow );
	if( 1 < pow ) {
		represent_in_si( poly , pow*deg_si(si)*blk_size , blk_size , si );
		represent_in_si( poly + pow*deg_si(si)*blk_size , n_terms - pow*deg_si(si)*blk_size , blk_size , si );
	}
#endif
}


void _bc_to_lch( bc_sto_t * poly , unsigned n_terms , unsigned blk_size )
{
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned si = get_max_si( degree_in_blocks );
	represent_in_si( poly , n_terms , blk_size , si );

	unsigned new_blk_size = deg_si(si)*blk_size;
	_bc_to_lch( poly , n_terms , new_blk_size );
	for(unsigned i=0;i<n_terms;i+= new_blk_size ) {
		_bc_to_lch( poly + i , new_blk_size , blk_size );
	}
}


void bc_to_lch( bc_sto_t * poly , unsigned n_terms )
{
	_bc_to_lch( poly , n_terms , 1 );
}



/////////////////////////////////////

__global__
void xor_up_d_2(u256 *poly256, bc_sto_t *poly, unsigned st, unsigned diff) {
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	poly256[i] = u256_xor(poly256[i], *(u256*)(poly+st+diff+(i*4)));
}

__global__
void xor_up_d_3(bc_sto_t *poly, unsigned st, unsigned diff, unsigned start) {
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x + start;
	poly[st+i] ^= poly[st+i+diff];
}

static inline
void xor_up( bc_sto_t * poly , unsigned st , unsigned len , unsigned diff )
{
#if 0
	for( unsigned i=0;i<len;i++) {
		poly[st+i] ^= poly[st+i+diff];
	}
#else
	while( ((unsigned long)(poly+st)) & 31 ) {
		poly[st] ^= poly[st+diff];
		st++;
		len--;
		if( 0 == len ) break;
	}
	u256 * poly256 = (u256*)(poly+st);
	unsigned _len = len>>2;
	uint64_t nBlock = (_len + 1023) / 1024;
	xor_up_d_2<<<nBlock, 1024>>>(poly256, poly, st, diff);
	nBlock = (len - (_len<<2) + 1023) / 1024;
	xor_up_d_3<<<nBlock, 1024>>>(poly, st, diff, _len<<2);
#endif
}


static inline
void i_poly_div( bc_sto_t * poly , unsigned n_terms , unsigned blk_size , unsigned si , unsigned pow )
{
	if( 0 == si ) return;
	unsigned si_degree = deg_si(si)*pow;
	unsigned deg_diff = si_degree - pow;
	unsigned deg_blk = get_num_blocks( n_terms , blk_size ) -1;
#if 1
	xor_up( poly , (blk_size)*(si_degree-deg_diff) , (deg_blk-si_degree+1)*blk_size , deg_diff*blk_size );
#else
	for(unsigned i=si_degree;i<=deg_blk;i++) {
		for(unsigned j=0; j<blk_size ;j++) {
			poly[(i-deg_diff)*blk_size+j] ^= poly[i*blk_size+j];
		}
	}
#endif
}

static inline
void i_represent_in_si( bc_sto_t * poly , unsigned n_terms , unsigned blk_size , unsigned si )
{
	if( 0 == si ) return;
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned degree_basic_form_si = deg_si(si);
	if( degree_basic_form_si > degree_in_blocks ) return;

	unsigned pow = 1;
	while( pow*deg_si(si) <= degree_in_blocks ) {
		for(unsigned i=0;i<n_terms;i+= blk_size*2*pow*deg_si(si) ) {
			i_poly_div( poly + i , blk_size*2*pow*deg_si(si) , blk_size , si , pow );
		}
		pow *= 2;

	}
}


void _bc_to_mono( bc_sto_t * poly , unsigned n_terms , unsigned blk_size )
{
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned si = get_max_si( degree_in_blocks );


	unsigned new_blk_size = deg_si(si)*blk_size;
	for(unsigned i=0;i<n_terms;i+= new_blk_size ) {
		_bc_to_mono( poly + i , new_blk_size , blk_size );
	}
	_bc_to_mono( poly , n_terms , new_blk_size );
	i_represent_in_si( poly , n_terms , blk_size , si );
}


void bc_to_mono( bc_sto_t * poly , unsigned n_terms )
{
	_bc_to_mono( poly , n_terms , 1 );
}





//////////////////////////////////////////////


static inline
void xor_down_128( __m128i * poly , unsigned st , unsigned len , unsigned diff )
{
#if 0
	for( unsigned i=0;i<len;i++) {
		poly[st-i-1] ^= poly[st-i-1+diff];
	}
#else
	if( ((unsigned long)(poly+st)) & 31 ) {
		poly[st-1] ^= poly[st+diff-1];
		st--;
		len--;
	}
	__m256i * poly256 = (__m256i*)(poly+st);
	unsigned _len = len>>1;
	for( unsigned i=0;i<_len;i++ ) {
		*(poly256 - i-1) ^= _mm256_loadu_si256( (__m256i*)(poly+st+diff-(i*2)-2) );
	}
	if( len&1 ) {
		poly[st-len] ^= poly[st-len+diff];
	}
#endif
}



static inline
void poly_div_128( __m128i * poly , unsigned n_terms , unsigned blk_size , unsigned si , unsigned pow )
{
	if( 0 == si ) return;
	unsigned si_degree = deg_si(si)*pow;
	unsigned deg_diff = si_degree - pow;
	unsigned deg_blk = get_num_blocks( n_terms , blk_size ) -1;

	xor_down_128( poly , (deg_blk-deg_diff+1)*blk_size , (deg_blk-si_degree+1)*blk_size , deg_diff*blk_size );
}

static inline
void represent_in_si_128( __m128i * poly , unsigned n_terms , unsigned blk_size , unsigned si )
{
	if( 0 == si ) return;
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned degree_basic_form_si = deg_si(si);
	if( degree_basic_form_si > degree_in_blocks ) return;

#if 1
	unsigned pow = get_si_2_pow( si , degree_in_blocks );
	while( 0 < pow ) {
		for(unsigned i=0;i<n_terms;i+= blk_size*2*pow*deg_si(si) ) {
			poly_div_128( poly + i , blk_size*2*pow*deg_si(si) , blk_size , si , pow );
		}
		pow >>= 1;
	}
#else
	unsigned pow = get_si_2_pow( si , degree_in_blocks );
	poly_div( poly , n_terms , blk_size , si , pow );
	if( 1 < pow ) {
		represent_in_si( poly , pow*deg_si(si)*blk_size , blk_size , si );
		represent_in_si( poly + pow*deg_si(si)*blk_size , n_terms - pow*deg_si(si)*blk_size , blk_size , si );
	}
#endif
}


void _bc_to_lch_128( __m128i * poly , unsigned n_terms , unsigned blk_size )
{
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned si = get_max_si( degree_in_blocks );
	represent_in_si_128( poly , n_terms , blk_size , si );

	unsigned new_blk_size = deg_si(si)*blk_size;
	_bc_to_lch_128( poly , n_terms , new_blk_size );
	for(unsigned i=0;i<n_terms;i+= new_blk_size ) {
		_bc_to_lch_128( poly + i , new_blk_size , blk_size );
	}
}


void bc_to_lch_128( bc_sto_t * poly , unsigned n_terms )
{
	_bc_to_lch_128( (__m128i*) poly , n_terms , 1 );
}


///////////////////////////////////


static inline
void xor_up_128( __m128i * poly , unsigned st , unsigned len , unsigned diff )
{
#if 0
	for( unsigned i=0;i<len;i++) {
		poly[st+i] ^= poly[st+i+diff];
	}
#else
	if( ((unsigned long)(poly+st)) & 31 ) {
		poly[st] ^= poly[st+diff];
		st++;
		len--;
	}
	__m256i * poly256 = (__m256i*)(poly+st);
	unsigned _len = len>>1;
	for( unsigned i=0;i<_len;i++ ) {
		poly256[i] ^= _mm256_loadu_si256( (__m256i*)(poly+st+diff+(i*2)) );
	}
	if( len&1 ) {
		poly[st+len-1] ^= poly[st+len-1+diff];
	}
#endif
}


static inline
void i_poly_div_128( __m128i * poly , unsigned n_terms , unsigned blk_size , unsigned si , unsigned pow )
{
	if( 0 == si ) return;
	unsigned si_degree = deg_si(si)*pow;
	unsigned deg_diff = si_degree - pow;
	unsigned deg_blk = get_num_blocks( n_terms , blk_size ) -1;

#if 1
	xor_up_128( poly , blk_size*(si_degree-deg_diff) , (deg_blk-si_degree+1)*blk_size , deg_diff*blk_size );
#else
	for(unsigned i=si_degree;i<=deg_blk;i++) {
		for(unsigned j=0; j<blk_size ;j++) {
			poly[(i-deg_diff)*blk_size+j] ^= poly[i*blk_size+j];
		}
	}
#endif
}

static inline
void i_represent_in_si_128( __m128i * poly , unsigned n_terms , unsigned blk_size , unsigned si )
{
	if( 0 == si ) return;
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned degree_basic_form_si = deg_si(si);
	if( degree_basic_form_si > degree_in_blocks ) return;

	unsigned pow = 1;
	while( pow*deg_si(si) <= degree_in_blocks ) {
		for(unsigned i=0;i<n_terms;i+= blk_size*2*pow*deg_si(si) ) {
			i_poly_div_128( poly + i , blk_size*2*pow*deg_si(si) , blk_size , si , pow );
		}
		pow *= 2;
	}
}


void _bc_to_mono_128( __m128i * poly , unsigned n_terms , unsigned blk_size )
{

//printf("ibc: %d/%d\n", n_terms , blk_size );

	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;

//printf("deg: %d\n", degree_in_blocks);
	unsigned si = get_max_si( degree_in_blocks );
//printf("si: %d\n",si);

	unsigned new_blk_size = deg_si(si)*blk_size;
//printf("new blksize: %d\n", new_blk_size);
	for(unsigned i=0;i<n_terms;i+= new_blk_size ) {
		_bc_to_mono_128( poly + i , new_blk_size , blk_size );
	}
	_bc_to_mono_128( poly , n_terms , new_blk_size );
	i_represent_in_si_128( poly , n_terms , blk_size , si );
}


void bc_to_mono_128( bc_sto_t * poly , unsigned n_terms )
{

	_bc_to_mono_128( (__m128i*)poly , n_terms , 1 );
}





//////////////////////////////////////////////

// Beispiel:
// __xor_down_256<<<dim3(1, 1), src_idx - dest_idx>>>(...);
__global__
void __xor_down_256( u256 * poly , unsigned dest_idx , unsigned src_idx, unsigned len, unsigned unit ) {
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t pSize = gridDim.x * blockDim.x;

	if (i >= src_idx - dest_idx) return;
	poly += blockIdx.y * unit;

	for (uint64_t d = (dest_idx+len-pSize), s = (src_idx+len-pSize); d >= dest_idx && s >= src_idx; d -= pSize, s -= pSize) {
		poly[d+i] = u256_xor(poly[d+i], poly[s+i]);
	}
}

static inline
void xor_down_256( u256 * poly , unsigned st , unsigned len , unsigned diff )
{
	unsigned dest_st = st - len;
	unsigned src_st = st - len + diff;

	uint64_t nBlock = (diff + 1023) / 1024;
	__xor_down_256<<<nBlock, 1024>>>(poly, dest_st, src_st, len, 1);
//	for( unsigned i=0;i<len;i++) {
//		poly[st-i-1] ^= poly[st-i-1+diff];
//	}
}

static inline
void __xor_down_256_2( u256 * poly , unsigned len , unsigned l_st, unsigned num, unsigned unit ) {
	dim3 nBlock((len - l_st + 1023) / 1024, num);
	__xor_down_256<<<nBlock, 1024>>>(poly, l_st, len, len, unit);
}

// Beispiel:
// __xor_up_256<<<1, src_idx - dest_idx>>>(...);
__global__
void __xor_up_256( u256 * poly , unsigned dest_idx , unsigned src_idx , unsigned len, unsigned unit ) {
	uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t pSize = gridDim.x * blockDim.x;

	if (i >= src_idx - dest_idx) return;
	poly += blockIdx.y * unit;

	for (uint64_t d = dest_idx, s = src_idx; (d+i) < (dest_idx+len) && (s+i) < (src_idx+len); d += pSize, s += pSize) {
		poly[d+i] = u256_xor(poly[d+i], poly[s+i]);
	}
}

static inline
void xor_up_256( u256 * poly , unsigned st , unsigned len , unsigned diff )
{
	uint64_t nBlock = (diff + 1023) / 1024;
	__xor_up_256<<<nBlock, 1024>>>( poly , st , st + diff, len, 1 );
//	for( unsigned i=0;i<len;i++) {
//		poly[st+i] ^= poly[st+i+diff];
//	}
}

static inline
void __xor_up_256_2( u256 * poly , unsigned len , unsigned l_st, unsigned num, unsigned unit ){
	dim3 nBlock((len - l_st + 1023) / 1024, num);
	__xor_up_256<<<nBlock, 1024>>>(poly, l_st, len, len, unit);
//	for( unsigned i=0;i<len;i++) poly[l_st+i] ^= poly[len+i];
}



//////////////////////////////////////////////////////////////////////////



static inline
void poly_div_256( u256 * poly , unsigned n_terms , unsigned blk_size , unsigned si , unsigned pow )
{
	if( 0 == si ) return;
	unsigned si_degree = deg_si(si)*pow;
	unsigned deg_diff = si_degree - pow;
	unsigned deg_blk = get_num_blocks( n_terms , blk_size ) -1;

	xor_down_256( poly , (deg_blk-deg_diff+1)*blk_size , (deg_blk-si_degree+1)*blk_size , deg_diff*blk_size );
}

static inline
void represent_in_si_256( u256 * poly , unsigned n_terms , unsigned blk_size , unsigned si )
{
	if( 0 == si ) return;
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned degree_basic_form_si = deg_si(si);
	if( degree_basic_form_si > degree_in_blocks ) return;

#if 1
	unsigned pow = get_si_2_pow( si , degree_in_blocks );
	while( 0 < pow ) {
		for(unsigned i=0;i<n_terms;i+= blk_size*2*pow*deg_si(si) ) {
			poly_div_256( poly + i , blk_size*2*pow*deg_si(si) , blk_size , si , pow );
		}
		pow >>= 1;
	}
#else
	unsigned pow = get_si_2_pow( si , degree_in_blocks );
	poly_div( poly , n_terms , blk_size , si , pow );
	if( 1 < pow ) {
		represent_in_si( poly , pow*deg_si(si)*blk_size , blk_size , si );
		represent_in_si( poly + pow*deg_si(si)*blk_size , n_terms - pow*deg_si(si)*blk_size , blk_size , si );
	}
#endif
}


void _bc_to_lch_256( u256 * poly , unsigned n_terms , unsigned blk_size )
{

	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned si = get_max_si( degree_in_blocks );
	represent_in_si_256( poly , n_terms , blk_size , si );

	unsigned new_blk_size = deg_si(si)*blk_size;
	_bc_to_lch_256( poly , n_terms , new_blk_size );
	for(unsigned i=0;i<n_terms;i+= new_blk_size ) {
		_bc_to_lch_256( poly + i , new_blk_size , blk_size );
	}
}


void bc_to_lch_256( bc_sto_t * poly , unsigned n_terms )
{
	_bc_to_lch_256( (u256*) poly , n_terms , 1 );
}


///////////////////////////////////



static inline
void i_poly_div_256( u256 * poly , unsigned n_terms , unsigned blk_size , unsigned si , unsigned pow )
{
	if( 0 == si ) return;
	unsigned si_degree = deg_si(si)*pow;
	unsigned deg_diff = si_degree - pow;
	unsigned deg_blk = get_num_blocks( n_terms , blk_size ) -1;

	xor_up_256( poly , blk_size*(si_degree-deg_diff) , (deg_blk-si_degree+1)*blk_size , deg_diff*blk_size );
}

static inline
void i_represent_in_si_256( u256 * poly , unsigned n_terms , unsigned blk_size , unsigned si )
{
	if( 0 == si ) return;
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned degree_basic_form_si = deg_si(si);
	if( degree_basic_form_si > degree_in_blocks ) return;

	unsigned pow = 1;
	while( pow*deg_si(si) <= degree_in_blocks ) {
		for(unsigned i=0;i<n_terms;i+= blk_size*2*pow*deg_si(si) ) {
			i_poly_div_256( poly + i , blk_size*2*pow*deg_si(si) , blk_size , si , pow );
		}
		pow *= 2;
	}
}


void _bc_to_mono_256( u256 * poly , unsigned n_terms , unsigned blk_size )
{

	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned si = get_max_si( degree_in_blocks );

	unsigned new_blk_size = deg_si(si)*blk_size;
	for(unsigned i=0;i<n_terms;i+= new_blk_size ) {
		_bc_to_mono_256( poly + i , new_blk_size , blk_size );
	}
	_bc_to_mono_256( poly , n_terms , new_blk_size );
	i_represent_in_si_256( poly , n_terms , blk_size , si );
}


void bc_to_mono_256( bc_sto_t * poly , unsigned n_terms )
{

	_bc_to_mono_256( (u256*)poly , n_terms , 1 );
}





///////////////////////////////////////////////

#define NDEBUG
#include "assert.h"



//#include "byte_inline_func.h"

__device__
u256 _mm256_alignr_255bit_zerohigh( u256 zerohigh , u256 low )
{
	// __m256i l_shr_15 = _mm256_srli_epi16( low , 15 );
	// __m256i r_1 = _mm256_permute2x128_si256( l_shr_15 , zerohigh , 0x21 );
	// return _mm256_srli_si256( r_1 , 14 );

	u256 l_shr_15 = srli<uint16_t>(low, 15);
	u256 r_1 = permute2x128(l_shr_15, zerohigh, 0x21);
	return srli<uint8_t>(r_1, 14);
}

__device__
u256 _mm256_alignr_254bit_zerohigh( u256 zerohigh , u256 low )
{
	// __m256i l_shr_14 = _mm256_srli_epi16( low , 14 );
	// __m256i r_2 = _mm256_permute2x128_si256( l_shr_14 , zerohigh , 0x21 );
	// return _mm256_srli_si256( r_2 , 14 );

	u256 l_shr_14 = srli<uint16_t>(low, 14);
	u256 r_2 = permute2x128(l_shr_14, zerohigh, 0x21);
	return srli<uint8_t>(r_2, 14);
}

__device__
u256 _mm256_alignr_252bit_zerohigh( u256 zerohigh , u256 low )
{
	// __m256i l_shr_12 = _mm256_srli_epi16( low , 12 );
	// __m256i r_4 = _mm256_permute2x128_si256( l_shr_12 , zerohigh , 0x21 );
	// return _mm256_srli_si256( r_4 , 14 );

	u256 l_shr_12 = srli<uint16_t>(low, 12);
	u256 r_4 = permute2x128(l_shr_12, zerohigh, 0x21);
	return srli<uint8_t>(r_4, 14);
}

__device__
u256 _mm256_alignr_255bit( u256 high , u256 low )
{
	// __m256i l_shr_15 = _mm256_srli_epi16( low , 15 );
	// __m256i h_shr_15 = _mm256_srli_epi16( high , 15 );
	// __m256i h_shl_1 = _mm256_slli_epi16( high , 1 );
	// __m256i r = h_shl_1^_mm256_slli_si256( h_shr_15 , 2 );

	// __m256i r_1 = _mm256_permute2x128_si256( l_shr_15 , h_shr_15 , 0x21 );
	// r ^= _mm256_srli_si256( r_1 , 14 );
	// return r;

	u256 l_shr_15 = srli<uint16_t>(low, 15);
	u256 h_shr_15 = srli<uint16_t>(high, 15);
	u256 h_shl_1 = slli<uint16_t>(high, 1);
	u256 r = u256_xor(h_shl_1, slli<uint8_t>(h_shr_15, 2));
	return r;
}

__device__
u256 _mm256_alignr_254bit( u256 high , u256 low )
{
	// __m256i l_shr_14 = _mm256_srli_epi16( low , 14 );
	// __m256i h_shr_14 = _mm256_srli_epi16( high , 14 );
	// __m256i h_shl_2 = _mm256_slli_epi16( high , 2 );
	// __m256i r = h_shl_2^_mm256_slli_si256( h_shr_14 , 2 );

	// __m256i r_2 = _mm256_permute2x128_si256( l_shr_14 , h_shr_14 , 0x21 );
	// r ^= _mm256_srli_si256( r_2 , 14 );
	// return r;

	u256 l_shr_14 = srli<uint16_t>(low, 14);
	u256 h_shr_14 = srli<uint16_t>(high, 14);
	u256 h_shl_2 = slli<uint16_t>(high, 2);
	u256 r = u256_xor(h_shl_2, slli<uint8_t>(h_shr_14, 2));

	u256 r_2 = permute2x128(l_shr_14, h_shr_14 , 0x21);
	r = u256_xor(r, srli<uint8_t>(r_2, 14));
	return r;
}

__device__
u256 _mm256_alignr_252bit( u256 high , u256 low )
{
	// __m256i l_shr_12 = _mm256_srli_epi16( low , 12 );
	// __m256i h_shr_12 = _mm256_srli_epi16( high , 12 );
	// __m256i h_shl_4 = _mm256_slli_epi16( high , 4 );
	// __m256i r = h_shl_4^_mm256_slli_si256( h_shr_12 , 2 );

	// __m256i r_4 = _mm256_permute2x128_si256( l_shr_12 , h_shr_12 , 0x21 );
	// r ^= _mm256_srli_si256( r_4 , 14 );
	// return r;

	u256 l_shr_12 = srli<uint16_t>(low, 12);
	u256 h_shr_12 = srli<uint16_t>(high, 12);
	u256 h_shl_4 = srli<uint16_t>(high, 4);
	u256 r = u256_xor(h_shl_4, slli<uint8_t>(h_shr_12, 2));
	

	u256 r_4 = permute2x128(l_shr_12, h_shr_12 , 0x21);
	r = u256_xor(r, srli<uint8_t>(r_4, 14));
	return r;
}

__device__
u256 _mm256_alignr_31byte( u256 high , u256 low )
{
	// __m256i l0 = _mm256_permute2x128_si256( low , high , 0x21 );
	// return _mm256_alignr_epi8( high , l0 , 15 );

	u256 l0 = permute2x128(low, high, 0x21);
	return alignr(high, l0, 15);
	
}

__device__
u256 _mm256_alignr_30byte( u256 high , u256 low )
{
	// __m256i l0 = _mm256_permute2x128_si256( low , high , 0x21 );
	// return _mm256_alignr_epi8( high , l0 , 14 );

	u256 l0 = permute2x128(low, high, 0x21);
	return alignr(high, l0 , 14);
}

__device__
u256 _mm256_alignr_28byte( u256 high , u256 low )
{
	// __m256i l0 = _mm256_permute2x128_si256( low , high , 0x21 );
	// return _mm256_alignr_epi8( high , l0 , 12 );

	u256 l0 = permute2x128(low, high, 0x21);
	return alignr(high, l0 , 12);
}

__device__
u256 _mm256_alignr_24byte( u256 high , u256 low )
{
	// __m256i l0 = _mm256_permute2x128_si256( low , high , 0x21 );
	// return _mm256_alignr_epi8( high , l0 , 8 );

	u256 l0 = permute2x128(low, high, 0x21);
	return alignr(high, l0 , 8);
}

__device__
u256 _mm256_alignr_16byte( u256 high , u256 low )
{
	// return _mm256_permute2x128_si256( low , high , 0x21 );

	return permute2x128(low, high , 0x21);
}

__device__
u256 (*_sh_op[8]) (u256 h, u256 l) = {
	_mm256_alignr_255bit, _mm256_alignr_254bit, _mm256_alignr_252bit, _mm256_alignr_31byte, _mm256_alignr_30byte, _mm256_alignr_28byte, _mm256_alignr_24byte, _mm256_alignr_16byte
};

__device__
u256 (*_sh_op_zerohigh[8]) (u256 h, u256 l) = {
	_mm256_alignr_255bit_zerohigh , _mm256_alignr_254bit_zerohigh , _mm256_alignr_252bit_zerohigh , _mm256_alignr_31byte, _mm256_alignr_30byte, _mm256_alignr_28byte, _mm256_alignr_24byte, _mm256_alignr_16byte
};

__global__
void __sh_xor_down_a(u256 *poly256, unsigned unit, unsigned _op, u256 zero) {
	poly256 += (blockIdx.x * blockDim.x + threadIdx.x) * unit;
	poly256[(unit>>1)] = u256_xor(poly256[(unit>>1)], _sh_op_zerohigh[_op](zero,poly256[unit-1]));
}

__global__
void __sh_xor_down_b(u256 *poly256, unsigned unit, unsigned _op) {
	uint64_t i = blockIdx.y;
	poly256 += (blockIdx.x * blockDim.x + threadIdx.x) * unit;
	poly256[(unit>>1)-1-i] = u256_xor(poly256[(unit>>1)-1-i], _sh_op[_op](poly256[unit-1-i] , poly256[unit-2-i]));
}

__global__
void __sh_xor_down_c(u256 *poly256, unsigned unit, unsigned _op, u256 zero) {
	poly256 += (blockIdx.x * blockDim.x + threadIdx.x) * unit;
	poly256[0] = u256_xor(poly256[0], _sh_op[_op](poly256[unit>>1],zero));
}

static inline
void __sh_xor_down( u256* poly256 , unsigned unit , unsigned _op , u256 zero, unsigned num )
{
	__sh_xor_down_a<<<(num+1023)/1024, 1024>>>(poly256, unit, _op, zero);

	dim3 nBlock((num+1023)/1024, (unit>>1)-1);
	if (nBlock.x > 0 && nBlock.y > 0)
		__sh_xor_down_b<<<nBlock, 1024>>>(poly256, unit, _op);

	__sh_xor_down_c<<<(num+1023)/1024, 1024>>>(poly256, unit, _op, zero);
}


static
void varsub_x256( u256* poly256 , unsigned n_256 )
{
	if( 1 >= n_256 ) return;
	unsigned log_n = __builtin_ctz( n_256 );
	u256 zero;
	// __m256i zero = _mm256_setzero_si256();

	while( log_n > 8 ) {
		unsigned unit = 1<<log_n;
		unsigned num = n_256/unit;
		unsigned unit_2 = unit>>1;
		__xor_down_256_2( poly256 , unit_2 , (1<<(log_n-9)), num, unit );
		log_n--;
	}

	for(unsigned i=log_n; i>0 ; i--) {
		unsigned unit = (1<<i);
		unsigned num = n_256 / unit;
		__sh_xor_down( poly256 , unit , i-1 , zero, num );
	}
}

__global__
void __sh_xor_up_a(u256 *poly256, unsigned unit, unsigned _op, u256 zero) {
	poly256 += (blockIdx.x * blockDim.x + threadIdx.x) * unit;
	poly256[0] = u256_xor(poly256[0], _sh_op[_op](poly256[unit>>1],zero));
}

__global__
void __sh_xor_up_b(u256 *poly256, unsigned unit, unsigned _op) {
	uint64_t i = blockIdx.y;
	poly256 += (blockIdx.x * blockDim.x + threadIdx.x) * unit;
	poly256[i+1] = u256_xor(poly256[i+1], _sh_op[_op]( poly256[(unit>>1)+i+1] , poly256[(unit>>1)+i] ));
}

__global__
void __sh_xor_up_c(u256 *poly256, unsigned unit, unsigned _op, u256 zero) {
	poly256 += (blockIdx.x * blockDim.x + threadIdx.x) * unit;
	poly256[(unit>>1)] = u256_xor(poly256[(unit>>1)], _sh_op_zerohigh[_op](zero,poly256[unit-1]));
}

static inline
void __sh_xor_up( u256* poly256 , unsigned unit , unsigned _op , u256 zero, unsigned num )
{
	__sh_xor_up_a<<<(num+1023)/1024, 1024>>>(poly256, unit, _op, zero);

	dim3 nBlock((num+1023)/1024, (unit>>1)-1);
	if (nBlock.x > 0 && nBlock.y > 0)
		__sh_xor_up_b<<<nBlock, 1024>>>(poly256, unit, _op);

	__sh_xor_up_c<<<(num+1023)/1024, 1024>>>(poly256, unit, _op, zero);
}

static
void i_varsub_x256( u256* poly256 , unsigned n_256 )
{
	if( 1 >= n_256 ) return;
	unsigned log_n = __builtin_ctz( n_256 );
	u256 zero;
	// __m256i zero = _mm256_setzero_si256();

	unsigned _log_n = (log_n>8)? 8 : log_n;
	for(unsigned i=1; i<=_log_n ; i++) {
		unsigned unit = (1<<i);
		unsigned num = n_256 / unit;
		__sh_xor_up( poly256 , unit , i-1 , zero, num );
	}

	for(unsigned i=9;i<=log_n ; i++ ) {
		unsigned unit = 1<<i;
		unsigned num = n_256/unit;
		unsigned unit_2 = unit>>1;
		__xor_up_256_2( poly256 , unit_2 , (1<<(i-9)), num, unit );
	}
}


void bc_to_lch_2_unit256( bc_sto_t * poly , unsigned n_terms )
{
	assert( 0 == ( n_terms&(n_terms-1) ) );
	assert( 4 <= n_terms );

	u256 * poly256 = (u256*) poly;
	unsigned n_256 = n_terms>>2;

	// varsub_x256( poly256 , n_256 );

#ifdef BC_CODE_GEN
	int logn = LOG2(n_256);
	// bc_to_lch_256_30_12(poly256,logn);

	// for(int i=0;i<(1<<(MAX(0,logn-19)));++i){
	//     bc_to_lch_256_19_17(poly256+i*(1<<19),MIN(19,logn));
	// }

	struct timespec start, end;

	clock_gettime(CLOCK_MONOTONIC, &start);
	// for(int i=0;i<(1<<(MAX(0,logn-16)));++i){
	// 	bc_to_lch_256_16(poly256+i*(1<<16), MIN(16,logn));
	// }
	bc_to_lch_256_16(poly256, MIN(16,logn));
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &end);

	float loopDuration = (end.tv_sec - start.tv_sec) * 1000;
	loopDuration += (end.tv_nsec - start.tv_nsec) / 1000000.0;
	printf("Overall: %.2f ms\n", loopDuration);

#else
	_bc_to_lch_256( poly256 , n_256 , 1 );
#endif
}


void bc_to_mono_2_unit256( bc_sto_t * poly , unsigned n_terms )
{
	assert( 0 == ( n_terms&(n_terms-1) ) );
	assert( 4 <= n_terms );

	u256 * poly256 = (u256*) poly;
	unsigned n_256 = n_terms>>2;

#ifdef BC_CODE_GEN
        int logn = LOG2(n_256);
        for(int i=0;i<(1<<(MAX(0,logn-16)));++i){
	    	bc_to_mono_256_16(poly256+i*(1<<16), MIN(16,logn));
        }
        for(int i=0;i<(1<<(MAX(0,logn-19)));++i){
            bc_to_mono_256_19_17(poly256+i*(1<<19),MIN(19,logn));
        }
        bc_to_mono_256_30_20(poly256,logn);
#else
	_bc_to_mono_256( poly256 , n_256 , 1 );
#endif
	i_varsub_x256( poly256 , n_256 );
}


