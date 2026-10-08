// noid_math.h - GF(2^128) / Poseidon2b arithmetic for the NOID GPU engine.
//
// Compiles as CUDA device code (nvcc) AND as plain C++ (g++/MSVC) so that the
// exact same arithmetic can be validated on the CPU against noid_ref.cpp.
//
// Element layout: W4 {w0,w1,w2,w3}, w0 = coefficients x^0..x^31 (flat basis,
// polynomial basis modulo x^128 + x^7 + x^2 + x + 1).
//
// Carry-less multiplication without a CLMUL instruction (Pascal sm_61):
//   * 16x16 -> 32 carry-less products with ordinary 16-bit integer multiplies
//     (XMAD on Pascal) on operands split into 3 bit classes (bit i mod 3).
//     Each class product has at most 6 terms per position, so the integer
//     carries (value <= 6, 3 bits) never reach the next bit of the same class.
//   * Karatsuba 16 -> 32 -> 64 -> 128 (27 base products).
//   * The class split is linear, so the 9 Karatsuba operand words of a
//     128-bit operand are XORs of the split of its 4 words.
//
// Multiplication by the fixed MDS constants can use either the same routine
// with a pre-split constant operand (NOID_CMUL_XMAD) or shared-memory nibble
// tables (NOID_CMUL_TABLE).
#pragma once
#include <stdint.h>

#ifdef __CUDACC__
#define NOID_HD __host__ __device__ __forceinline__
#define NOID_HDM __host__ __device__ __forceinline__
#else
#define NOID_HD static inline
#define NOID_HDM inline
#endif

struct W4 { uint32_t w0, w1, w2, w3; };

NOID_HD W4 w4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { W4 r; r.w0 = a; r.w1 = b; r.w2 = c; r.w3 = d; return r; }
NOID_HD W4 w4x(W4 a, W4 b) { return w4(a.w0 ^ b.w0, a.w1 ^ b.w1, a.w2 ^ b.w2, a.w3 ^ b.w3); }
NOID_HD W4 w4x3(W4 a, W4 b, W4 c) { return w4(a.w0 ^ b.w0 ^ c.w0, a.w1 ^ b.w1 ^ c.w1, a.w2 ^ b.w2 ^ c.w2, a.w3 ^ b.w3 ^ c.w3); }

// funnel shift: high 32 bits of ((hi:lo) << s), 0 < s < 32
NOID_HD uint32_t fshl(uint32_t lo, uint32_t hi, int s)
{
#ifdef __CUDA_ARCH__
    return __funnelshift_l(lo, hi, s);
#else
    return (hi << s) | (lo >> (32 - s));
#endif
}

// 16x16 -> 32 integer multiply of half HA of a by half HB of b (0 = low, 1 = high).
// On the GPU this is a single XMAD with half selectors (PTX mul.wide.u16 on
// the 16-bit halves of the registers).
template <int HA, int HB>
NOID_HD uint32_t xm(uint32_t a, uint32_t b)
{
#ifdef __CUDA_ARCH__
    uint32_t r;
    if (HA == 0 && HB == 0)
        asm("{.reg .u16 al, ah, bl, bh; mov.b32 {al, ah}, %1; mov.b32 {bl, bh}, %2; mul.wide.u16 %0, al, bl;}" : "=r"(r) : "r"(a), "r"(b));
    else if (HA == 1 && HB == 1)
        asm("{.reg .u16 al, ah, bl, bh; mov.b32 {al, ah}, %1; mov.b32 {bl, bh}, %2; mul.wide.u16 %0, ah, bh;}" : "=r"(r) : "r"(a), "r"(b));
    else if (HA == 0)
        asm("{.reg .u16 al, ah, bl, bh; mov.b32 {al, ah}, %1; mov.b32 {bl, bh}, %2; mul.wide.u16 %0, al, bh;}" : "=r"(r) : "r"(a), "r"(b));
    else
        asm("{.reg .u16 al, ah, bl, bh; mov.b32 {al, ah}, %1; mov.b32 {bl, bh}, %2; mul.wide.u16 %0, ah, bl;}" : "=r"(r) : "r"(a), "r"(b));
    return r;
#else
    uint32_t x = HA ? (a >> 16) : (a & 0xFFFFu);
    uint32_t y = HB ? (b >> 16) : (b & 0xFFFFu);
    return x * y;
#endif
}

#define NOID_SPLIT0 0x92499249u   // bits = 0 mod 3 inside each 16-bit half
#define NOID_SPLIT1 0x24922492u   // bits = 1 mod 3
#define NOID_SPLIT2 0x49244924u   // bits = 2 mod 3
#define NOID_PCLS0  0x49249249u   // product bits = 0 mod 3 (32-bit)
#define NOID_PCLS1  0x92492492u
#define NOID_PCLS2  0x24924924u

// a 32-bit word split in 3 classes (both 16-bit halves at once)
struct S3 { uint32_t c0, c1, c2; };

NOID_HD S3 s3(uint32_t w) { S3 r; r.c0 = w & NOID_SPLIT0; r.c1 = w & NOID_SPLIT1; r.c2 = w & NOID_SPLIT2; return r; }
NOID_HD S3 s3x(S3 a, S3 b) { S3 r; r.c0 = a.c0 ^ b.c0; r.c1 = a.c1 ^ b.c1; r.c2 = a.c2 ^ b.c2; return r; }

// carry-less 16x16 -> 32 of half HA of a by half HB of b (split words).
// The 3 class sums are merged with two bit-selects (LOP3 with an immediate):
// positions = 0 mod 3 take t0, = 1 mod 3 take t1, = 2 mod 3 take t2.
template <int HA, int HB>
NOID_HD uint32_t cm16(S3 a, S3 b)
{
    uint32_t p00 = xm<HA, HB>(a.c0, b.c0), p01 = xm<HA, HB>(a.c0, b.c1), p02 = xm<HA, HB>(a.c0, b.c2);
    uint32_t p10 = xm<HA, HB>(a.c1, b.c0), p11 = xm<HA, HB>(a.c1, b.c1), p12 = xm<HA, HB>(a.c1, b.c2);
    uint32_t p20 = xm<HA, HB>(a.c2, b.c0), p21 = xm<HA, HB>(a.c2, b.c1), p22 = xm<HA, HB>(a.c2, b.c2);
    uint32_t t0 = p00 ^ p12 ^ p21;
    uint32_t t1 = p01 ^ p10 ^ p22;
    uint32_t t2 = p02 ^ p11 ^ p20;
    uint32_t sel = (t1 & NOID_PCLS1) | (t2 & ~NOID_PCLS1);
    return (t0 & NOID_PCLS0) | (sel & ~NOID_PCLS0);
}

// Karatsuba operand word: split of the word (both halves) and split of
// (low half ^ high half) in the low half.
struct KW { S3 s, m; };

NOID_HD KW kw(uint32_t w)
{
    KW r;
    r.s = s3(w);
    r.m = s3(w ^ (w >> 16));
    return r;
}
NOID_HD KW kwx(KW a, KW b) { KW r; r.s = s3x(a.s, b.s); r.m = s3x(a.m, b.m); return r; }

// carry-less 32x32 -> 64 (lo, hi)
NOID_HD void cm32(KW a, KW b, uint32_t& lo, uint32_t& hi)
{
    uint32_t L = cm16<0, 0>(a.s, b.s);
    uint32_t H = cm16<1, 1>(a.s, b.s);
    uint32_t M = cm16<0, 0>(a.m, b.m) ^ L ^ H;
    lo = L ^ (M << 16);
    hi = H ^ (M >> 16);
}

// carry-less 64x64 -> 128 of operand pairs (u0,u1) x (v0,v1)
NOID_HD void cm64(KW u0, KW u1, KW v0, KW v1, uint32_t r[4])
{
    uint32_t l0, l1, h0, h1, m0, m1;
    cm32(u0, v0, l0, l1);
    cm32(u1, v1, h0, h1);
    cm32(kwx(u0, u1), kwx(v0, v1), m0, m1);
    m0 ^= l0 ^ h0;
    m1 ^= l1 ^ h1;
    r[0] = l0;
    r[1] = l1 ^ m0;
    r[2] = h0 ^ m1;
    r[3] = h1;
}

// reduce a 256-bit carry-less product modulo x^128 + x^7 + x^2 + x + 1
NOID_HD W4 reduce256(const uint32_t r[8])
{
    // H = r4..r7 ; H*x^128 = H*(x^7+x^2+x+1)
    uint32_t h0 = r[4], h1 = r[5], h2 = r[6], h3 = r[7];
    uint32_t t0 = r[0] ^ h0 ^ (h0 << 1) ^ (h0 << 2) ^ (h0 << 7);
    uint32_t t1 = r[1] ^ h1 ^ fshl(h0, h1, 1) ^ fshl(h0, h1, 2) ^ fshl(h0, h1, 7);
    uint32_t t2 = r[2] ^ h2 ^ fshl(h1, h2, 1) ^ fshl(h1, h2, 2) ^ fshl(h1, h2, 7);
    uint32_t t3 = r[3] ^ h3 ^ fshl(h2, h3, 1) ^ fshl(h2, h3, 2) ^ fshl(h2, h3, 7);
    uint32_t ov = (h3 >> 31) ^ (h3 >> 30) ^ (h3 >> 25);      // x^128..x^134
    t0 ^= ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);
    return w4(t0, t1, t2, t3);
}

// Karatsuba operand: the 4 words of a 128-bit element, split
struct KOp { KW a0, a1, a2, a3; };

NOID_HD KOp kop(W4 a)
{
    KOp k; k.a0 = kw(a.w0); k.a1 = kw(a.w1); k.a2 = kw(a.w2); k.a3 = kw(a.w3);
    return k;
}

// full product of two split operands, reduced
NOID_HD W4 kmul(const KOp& a, const KOp& b)
{
    uint32_t L[4], H[4], M[4], r[8];
    cm64(a.a0, a.a1, b.a0, b.a1, L);
    cm64(a.a2, a.a3, b.a2, b.a3, H);
    cm64(kwx(a.a0, a.a2), kwx(a.a1, a.a3), kwx(b.a0, b.a2), kwx(b.a1, b.a3), M);
    for (int i = 0; i < 4; i++) M[i] ^= L[i] ^ H[i];
    r[0] = L[0];
    r[1] = L[1];
    r[2] = L[2] ^ M[0];
    r[3] = L[3] ^ M[1];
    r[4] = H[0] ^ M[2];
    r[5] = H[1] ^ M[3];
    r[6] = H[2];
    r[7] = H[3];
    return reduce256(r);
}

NOID_HD W4 gmul(W4 a, W4 b)
{
    KOp ka = kop(a), kb = kop(b);
    return kmul(ka, kb);
}

// bit spread of a 16-bit value into the even bits of a 32-bit word
NOID_HD uint32_t spread16(uint32_t x)
{
    x &= 0xFFFFu;
    x = (x | (x << 8)) & 0x00FF00FFu;
    x = (x | (x << 4)) & 0x0F0F0F0Fu;
    x = (x | (x << 2)) & 0x33333333u;
    x = (x | (x << 1)) & 0x55555555u;
    return x;
}

NOID_HD W4 gsqr(W4 a)
{
    uint32_t r[8];
    r[0] = spread16(a.w0); r[1] = spread16(a.w0 >> 16);
    r[2] = spread16(a.w1); r[3] = spread16(a.w1 >> 16);
    r[4] = spread16(a.w2); r[5] = spread16(a.w2 >> 16);
    r[6] = spread16(a.w3); r[7] = spread16(a.w3 >> 16);
    return reduce256(r);
}

// ---------------------------------------------------------------------------
// MDS constants (flat basis), derived from the tower-basis matrices:
//   MDS_FULL = [[5,7,1,3],[4,6,1,1],[1,3,5,7],[1,1,4,6]]  (tower integers)
//     with g = tower(2), h = tower(4):  5 = h+1, 7 = h+g+1, 3 = g+1, 6 = h+g
//     A = h(x0+x1)  B = h(x2+x3)  C = g x1  D = g x3  S = x0+x1+x2+x3
//     y0 = A+C+D+S  y1 = A+C+x2+x3  y2 = B+C+D+S  y3 = B+D+x0+x1
//   MDS_PARTIAL = diag(0x20,0x2000,0x200,0x800) + (ones - I):
//     y_i = S + (d_i + 1) x_i
// Constant indices: 0 = g, 1 = h, 2..5 = d_0+1 .. d_3+1
// ---------------------------------------------------------------------------
#define NOID_NCONST 6
