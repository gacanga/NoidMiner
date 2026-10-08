// noid_perm.h - Poseidon2b permutation (flat basis) built on noid_math.h.
//
// Templated on a constant-multiplier policy CM providing
//     W4 CM::mul(int c, W4 a)     // a * const[c], c in 0..NOID_NCONST-1
// and on a round-constant accessor RC providing
//     W4 RC::get(int lane, int round)
// so that the same code runs on the GPU (shared tables / constant memory)
// and on the CPU (validation).
//
// Code size matters on the GPU (instruction cache): the permutation is written
// so that there is a single instance of the S-box, of the 128-bit multiply and
// of the squaring; full rounds loop over the 4 lanes with a register rotation.
#pragma once
#include "noid_math.h"

#ifdef __CUDACC__
#define NOID_PRAGMA(x) _Pragma(#x)
#define NOID_UNROLL1 NOID_PRAGMA(unroll 1)
#define NOID_UNROLL NOID_PRAGMA(unroll)
#else
#define NOID_UNROLL1
#define NOID_UNROLL
#endif

// x^7 = (x * x^2) * x^4 with one multiply and one squaring instance
// NOID_EXPERIMENT (profiling builds only, results are WRONG):
//   1 = constant multiplications skipped (arithmetic only)
//   2 = S-box arithmetic replaced by XORs (table lookups only)
NOID_HD W4 sbox7(W4 x)
{
#if defined(NOID_EXPERIMENT) && NOID_EXPERIMENT == 2
    return w4(x.w1 ^ 0x9e3779b9u, x.w2 ^ x.w0, x.w3 + 7u, x.w0 ^ x.w3);
#endif
    W4 sq = x, x2 = x;
    NOID_UNROLL1
    for (int k = 0; k < 2; k++) {
        sq = gsqr(sq);
        if (k == 0) x2 = sq;
    }
    // sq = x^4
    W4 a = x, b = x2;
    NOID_UNROLL1
    for (int k = 0; k < 2; k++) {
        a = gmul(a, b);
        b = sq;
    }
    return a;
}

// MDS layer with a single constant-multiplier instance (4 iterations, rotation)
//   full   : A = h(x0+x1)  B = h(x2+x3)  C = g x1  D = g x3   (consts 1,1,0,0)
//   partial: P_i = (d_i + 1) x_i                              (consts 2,3,4,5)
template <class CM>
NOID_HD void mds(W4 s[4], bool full, CM& cm)
{
    W4 x01 = w4x(s[0], s[1]);
    W4 x23 = w4x(s[2], s[3]);
    W4 S = w4x(x01, x23);
    W4 u0, u1, u2, u3;
    if (full) { u0 = x01; u1 = x23; u2 = s[1]; u3 = s[3]; }
    else      { u0 = s[0]; u1 = s[1]; u2 = s[2]; u3 = s[3]; }
    NOID_UNROLL1
    for (int i = 0; i < 4; i++) {
        int c = full ? (i < 2 ? 1 : 0) : 2 + i;
        W4 p = cm.mul(c, u0);
        u0 = u1; u1 = u2; u2 = u3; u3 = p;
    }
    if (full) {
        // u0 = A, u1 = B, u2 = C, u3 = D
        W4 CD = w4x(u2, u3);
        s[0] = w4x3(u0, CD, S);
        s[1] = w4x3(u0, u2, x23);
        s[2] = w4x3(u1, CD, S);
        s[3] = w4x3(u1, u3, x01);
    } else {
        s[0] = w4x(S, u0);
        s[1] = w4x(S, u1);
        s[2] = w4x(S, u2);
        s[3] = w4x(S, u3);
    }
}

// round -1 is the initial MDS_FULL layer; rounds 0..3 and 62..65 are full
template <class CM, class RC>
NOID_HD void permute(W4 s[4], CM& cm, RC& rc)
{
    NOID_UNROLL1
    for (int r = -1; r < 66; r++) {
        const bool full = (r < 4) || (r >= 62);
        const int lanes = r < 0 ? 0 : (full ? 4 : 1);
        NOID_UNROLL1
        for (int i = 0; i < lanes; i++) {
            W4 y = sbox7(w4x(s[0], rc.get(i, r)));
            if (full) {               // rotate: after 4 steps lanes are back in place
                s[0] = s[1]; s[1] = s[2]; s[2] = s[3]; s[3] = y;
            } else {
                s[0] = y;
            }
        }
        mds(s, full, cm);
    }
}

// ---------------------------------------------------------------------------
// constant multiplier policies
// ---------------------------------------------------------------------------

// pre-split constant operands, multiplication with the generic Karatsuba core
struct CMulKop {
    const KOp* k;   // NOID_NCONST entries
    NOID_HDM W4 mul(int c, W4 a) { KOp ka = kop(a); return kmul(ka, k[c]); }
};

// nibble tables, 64-bit entries: tbl[(((c*2 + half)*32 + nibble)*16 + value)*2 + {0,1}]
//   half 0 = words (w0,w1), half 1 = words (w2,w3) of value*x^(4*nibble)*const.
// 16 entries x 8 bytes = 128 bytes = the 32 shared-memory banks: an LDS.64
// half-warp access never has a bank conflict.
#define NOID_TBL_WORDS (NOID_NCONST * 2 * 32 * 16 * 2)
struct CMulTable {
    const uint32_t* tbl;
    NOID_HDM W4 mul(int c, W4 a)
    {
#if defined(NOID_EXPERIMENT) && NOID_EXPERIMENT == 1
        return w4(a.w1 ^ (uint32_t)c, a.w2, a.w3 ^ a.w0, a.w0 + 3u);
#endif
#ifdef __CUDA_ARCH__
        const uint2* t = reinterpret_cast<const uint2*>(tbl) + c * (2 * 32 * 16);
#else
        const uint32_t* tw = tbl + c * (2 * 32 * 16 * 2);
#endif
        uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
        uint32_t w[4] = { a.w0, a.w1, a.w2, a.w3 };
        NOID_UNROLL
        for (int k = 0; k < 32; k++) {
            uint32_t v = (w[k >> 3] >> (4 * (k & 7))) & 15u;
#ifdef __CUDA_ARCH__
            uint2 lo = t[k * 16 + v];
            uint2 hi = t[512 + k * 16 + v];
            r0 ^= lo.x; r1 ^= lo.y; r2 ^= hi.x; r3 ^= hi.y;
#else
            const uint32_t* e0 = tw + (k * 16 + v) * 2;
            const uint32_t* e1 = tw + (512 + k * 16 + v) * 2;
            r0 ^= e0[0]; r1 ^= e0[1]; r2 ^= e1[0]; r3 ^= e1[1];
#endif
        }
        return w4(r0, r1, r2, r3);
    }
};

// ---------------------------------------------------------------------------
// "tower" engine (NOID_MODE_TOWER): partial rounds keep lanes 1..3 in the
// tower basis, where multiplying by the partial MDS constants (elements of the
// GF(2^16) subfield) acts independently on the eight 16-bit chunks.
// Lane 0 goes flat -> S-box -> tower once per partial round.
//
// Shared tables (u32 words):
//   lin  : 4 GF(2)-linear maps with 64-bit nibble entries (layout of CMulTable)
//          0 = * g (flat), 1 = * h (flat), 2 = tower->flat, 3 = flat->tower
//   chunk: 4 constants x 2 byte positions x 256 entries, 16-bit tower products
// ---------------------------------------------------------------------------
#define NOID_LIN_G   0
#define NOID_LIN_H   1
#define NOID_LIN_T2F 2
#define NOID_LIN_F2T 3
#define NOID_TOWER_LIN_WORDS   (4 * 2 * 32 * 16 * 2)          // 8192 u32 = 32 KiB
#define NOID_TOWER_CHUNK_WORDS (4 * 2 * 256)                   // 2048 u32 =  8 KiB
#define NOID_TOWER_TBL_WORDS   (NOID_TOWER_LIN_WORDS + NOID_TOWER_CHUNK_WORDS)

struct TowerTables {
    const uint32_t* tbl;
    // linear map m applied to a (same 64-bit nibble layout as CMulTable)
    NOID_HDM W4 lin(int m, W4 a)
    {
        CMulTable t{ tbl };
        return t.mul(m, a);
    }
    // a * P_c in the tower basis, chunk by chunk (16-bit chunks)
    NOID_HDM W4 chunk(int c, W4 a)
    {
        const uint32_t* t0 = tbl + NOID_TOWER_LIN_WORDS + c * 512;
        const uint32_t* t1 = t0 + 256;
        uint32_t w[4] = { a.w0, a.w1, a.w2, a.w3 };
        uint32_t r[4];
        NOID_UNROLL
        for (int i = 0; i < 4; i++) {
            uint32_t x = w[i];
            uint32_t lo = t0[x & 0xFFu] ^ t1[(x >> 8) & 0xFFu];
            uint32_t hi = t0[(x >> 16) & 0xFFu] ^ t1[x >> 24];
            r[i] = lo | (hi << 16);
        }
        return w4(r[0], r[1], r[2], r[3]);
    }
};

template <class TB>
NOID_HD void mds_full_tb(W4 s[4], TB& tb)
{
    W4 x01 = w4x(s[0], s[1]);
    W4 x23 = w4x(s[2], s[3]);
    W4 S = w4x(x01, x23);
    W4 u0 = x01, u1 = x23, u2 = s[1], u3 = s[3];
    NOID_UNROLL1
    for (int i = 0; i < 4; i++) {
        W4 p = tb.lin(i < 2 ? NOID_LIN_H : NOID_LIN_G, u0);
        u0 = u1; u1 = u2; u2 = u3; u3 = p;
    }
    W4 CD = w4x(u2, u3);
    s[0] = w4x3(u0, CD, S);
    s[1] = w4x3(u0, u2, x23);
    s[2] = w4x3(u1, CD, S);
    s[3] = w4x3(u1, u3, x01);
}

// lanes 1..3 through linear map m (rotation keeps a single call site)
template <class TB>
NOID_HD void lin3(W4& a, W4& b, W4& c, int m, TB& tb)
{
    NOID_UNROLL1
    for (int i = 0; i < 3; i++) {
        W4 p = tb.lin(m, a);
        a = b; b = c; c = p;
    }
}

template <class TB, class RC>
NOID_HD void permute_tower(W4 s[4], TB& tb, RC& rc)
{
    // initial MDS + 4 full rounds (flat)
    NOID_UNROLL1
    for (int r = -1; r < 4; r++) {
        NOID_UNROLL1
        for (int i = 0; i < (r < 0 ? 0 : 4); i++) {
            W4 y = sbox7(w4x(s[0], rc.get(i, r)));
            s[0] = s[1]; s[1] = s[2]; s[2] = s[3]; s[3] = y;
        }
        mds_full_tb(s, tb);
    }
    // 58 partial rounds: f0 flat, t1..t3 tower
    W4 f0 = s[0], t1 = s[1], t2 = s[2], t3 = s[3];
    lin3(t1, t2, t3, NOID_LIN_F2T, tb);
    NOID_UNROLL1
    for (int r = 4; r < 62; r++) {
        W4 y = sbox7(w4x(f0, rc.get(0, r)));
        W4 Y = tb.lin(NOID_LIN_F2T, y);
        W4 S = w4x(w4x(Y, t1), w4x(t2, t3));
        W4 n1 = w4x(S, tb.chunk(1, t1));
        W4 n2 = w4x(S, tb.chunk(2, t2));
        W4 n3 = w4x(S, tb.chunk(3, t3));
        W4 n0 = w4x(S, tb.chunk(0, Y));
        t1 = n1; t2 = n2; t3 = n3;
        f0 = tb.lin(NOID_LIN_T2F, n0);
    }
    lin3(t1, t2, t3, NOID_LIN_T2F, tb);
    s[0] = f0; s[1] = t1; s[2] = t2; s[3] = t3;
    // 4 full rounds (flat)
    NOID_UNROLL1
    for (int r = 62; r < 66; r++) {
        NOID_UNROLL1
        for (int i = 0; i < 4; i++) {
            W4 y = sbox7(w4x(s[0], rc.get(i, r)));
            s[0] = s[1]; s[1] = s[2]; s[2] = s[3]; s[3] = y;
        }
        mds_full_tb(s, tb);
    }
}

// builds the tables / split operands on the host from the flat constants
// consts[c] = flat value of constant c (see noid_math.h)
static inline void noid_build_const_tables(const W4 consts[NOID_NCONST], uint32_t* tbl /* NOID_TBL_WORDS */)
{
    for (int c = 0; c < NOID_NCONST; c++)
        for (int k = 0; k < 32; k++)
            for (int v = 0; v < 16; v++) {
                uint32_t in[4] = { 0, 0, 0, 0 };
                in[k >> 3] = (uint32_t)v << (4 * (k & 7));
                W4 p = gmul(w4(in[0], in[1], in[2], in[3]), consts[c]);
                uint32_t* t0 = tbl + ((c * 2 + 0) * 32 * 16 + k * 16 + v) * 2;
                uint32_t* t1 = tbl + ((c * 2 + 1) * 32 * 16 + k * 16 + v) * 2;
                t0[0] = p.w0; t0[1] = p.w1; t1[0] = p.w2; t1[1] = p.w3;
            }
}

static inline void noid_build_const_kops(const W4 consts[NOID_NCONST], KOp* k)
{
    for (int c = 0; c < NOID_NCONST; c++) k[c] = kop(consts[c]);
}

// tower engine tables. consts[0] = g, consts[1] = h (flat), t2f/f2t = columns of
// the basis conversions (column i = image of bit i), ptow[4] = tower values of
// the partial constants d_i + 1 (16-bit).
static inline void noid_build_tower_tables(const W4 consts[2], const W4 t2f[128], const W4 f2t[128],
                                           const uint32_t ptow[4], uint32_t* tbl /* NOID_TOWER_TBL_WORDS */)
{
    for (int m = 0; m < 4; m++)
        for (int k = 0; k < 32; k++)
            for (int v = 0; v < 16; v++) {
                uint32_t in[4] = { 0, 0, 0, 0 };
                in[k >> 3] = (uint32_t)v << (4 * (k & 7));
                W4 x = w4(in[0], in[1], in[2], in[3]);
                W4 p;
                if (m < 2) p = gmul(x, consts[m]);
                else {
                    const W4* cols = m == 2 ? t2f : f2t;
                    p = w4(0, 0, 0, 0);
                    for (int b = 0; b < 4; b++)
                        if ((v >> b) & 1) p = w4x(p, cols[4 * k + b]);
                }
                uint32_t* e0 = tbl + ((m * 2 + 0) * 32 * 16 + k * 16 + v) * 2;
                uint32_t* e1 = tbl + ((m * 2 + 1) * 32 * 16 + k * 16 + v) * 2;
                e0[0] = p.w0; e0[1] = p.w1; e1[0] = p.w2; e1[1] = p.w3;
            }
    // chunk tables: tower product P_c * (b << 8j) inside GF(2^16), computed as
    // flat products: tower -> flat, multiply, flat -> tower
    for (int c = 0; c < 4; c++) {
        W4 pf = w4(0, 0, 0, 0);
        for (int b = 0; b < 16; b++) if ((ptow[c] >> b) & 1) pf = w4x(pf, t2f[b]);
        for (int j = 0; j < 2; j++)
            for (int b = 0; b < 256; b++) {
                uint32_t xv = (uint32_t)b << (8 * j);
                W4 xf = w4(0, 0, 0, 0);
                for (int i = 0; i < 16; i++) if ((xv >> i) & 1) xf = w4x(xf, t2f[i]);
                W4 pr = gmul(xf, pf);
                // back to tower
                uint32_t pw[4] = { pr.w0, pr.w1, pr.w2, pr.w3 };
                W4 tw = w4(0, 0, 0, 0);
                for (int i = 0; i < 128; i++) if ((pw[i >> 5] >> (i & 31)) & 1) tw = w4x(tw, f2t[i]);
                tbl[NOID_TOWER_LIN_WORDS + c * 512 + j * 256 + b] = tw.w0 & 0xFFFFu;   // stays in GF(2^16)
            }
    }
}
