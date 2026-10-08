// test_math.cpp - validates gpu/noid_math.h + gpu/noid_perm.h on the CPU
// against the reference src/noid_ref.cpp.
//   g++ -O2 -std=c++17 -Isrc -Igpu tests/test_math.cpp src/noid_ref.cpp
#include <stdio.h>
#include <string.h>
#include <vector>
#include "noid_ref.h"
#include "noid_perm.h"

static W4 to_w4(noid::u128 v) { return w4((uint32_t)v.lo, (uint32_t)(v.lo >> 32), (uint32_t)v.hi, (uint32_t)(v.hi >> 32)); }
static noid::u128 to_u(W4 a) { return noid::mk(((uint64_t)a.w1 << 32) | a.w0, ((uint64_t)a.w3 << 32) | a.w2); }

struct Rng { uint64_t s; uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; } };

struct RCHost { NOID_HDM W4 get(int lane, int r) { return to_w4(NOID_RC[lane][r]); } };

int main()
{
    Rng rng{ 0x1234567887654321ULL };
    int bad = 0;
    for (int i = 0; i < 20000; i++) {
        noid::u128 a = noid::mk(rng.next(), rng.next()), b = noid::mk(rng.next(), rng.next());
        if (i == 0) { a = noid::mk(~0ULL, ~0ULL); b = a; }
        if (!noid::eq(to_u(gmul(to_w4(a), to_w4(b))), noid::gf_mul(a, b))) { if (bad++ < 5) printf("gmul mismatch %d\n", i); }
        if (!noid::eq(to_u(gsqr(to_w4(a))), noid::gf_sqr(a))) { if (bad++ < 5) printf("gsqr mismatch %d\n", i); }
    }
    printf("field ops: %s\n", bad ? "FAIL" : "ok");

    W4 consts[NOID_NCONST];
    consts[0] = to_w4(noid::tower_to_flat(noid::mk(2, 0)));
    consts[1] = to_w4(noid::tower_to_flat(noid::mk(4, 0)));
    for (int i = 0; i < 4; i++) consts[2 + i] = to_w4(noid::x_or(NOID_MDS_PARTIAL[i][i], noid::mk(1, 0)));

    std::vector<uint32_t> tbl(NOID_TBL_WORDS);
    noid_build_const_tables(consts, tbl.data());
    KOp kops[NOID_NCONST];
    noid_build_const_kops(consts, kops);
    CMulTable cmt{ tbl.data() };
    CMulKop cmk{ kops };
    RCHost rc;

    int pbad = 0;
    for (int i = 0; i < 200; i++) {
        noid::u128 s[4];
        for (int j = 0; j < 4; j++) s[j] = noid::mk(rng.next(), rng.next());
        W4 t1[4], t2[4];
        for (int j = 0; j < 4; j++) t1[j] = t2[j] = to_w4(s[j]);
        noid::permute_flat(s);
        permute(t1, cmt, rc);
        permute(t2, cmk, rc);
        for (int j = 0; j < 4; j++) {
            if (!noid::eq(to_u(t1[j]), s[j])) pbad++;
            if (!noid::eq(to_u(t2[j]), s[j])) pbad++;
        }
    }
    printf("permutation (table + kop policies): %s\n", pbad ? "FAIL" : "ok");

    // tower engine
    W4 t2f[128], f2t[128];
    for (int i = 0; i < 128; i++) { t2f[i] = to_w4(NOID_T2F[i]); f2t[i] = to_w4(NOID_F2T[i]); }
    uint32_t ptow[4] = { 0x21, 0x2001, 0x201, 0x801 };
    std::vector<uint32_t> ttbl(NOID_TOWER_TBL_WORDS);
    noid_build_tower_tables(consts, t2f, f2t, ptow, ttbl.data());
    // chunk products must stay inside GF(2^16): check against the reference
    int cbad = 0;
    for (int i = 0; i < 2000; i++) {
        noid::u128 x = noid::mk(rng.next(), rng.next());
        for (int c = 0; c < 4; c++) {
            noid::u128 want = noid::flat_to_tower(noid::gf_mul(noid::tower_to_flat(x), noid::tower_to_flat(noid::mk(ptow[c], 0))));
            TowerTables tb{ ttbl.data() };
            if (!noid::eq(to_u(tb.chunk(c, to_w4(x))), want)) cbad++;
        }
    }
    printf("tower chunk multiplication: %s\n", cbad ? "FAIL" : "ok");
    int tbad = 0;
    for (int i = 0; i < 200; i++) {
        noid::u128 s[4];
        for (int j = 0; j < 4; j++) s[j] = noid::mk(rng.next(), rng.next());
        W4 t[4];
        for (int j = 0; j < 4; j++) t[j] = to_w4(s[j]);
        noid::permute_flat(s);
        TowerTables tb{ ttbl.data() };
        permute_tower(t, tb, rc);
        for (int j = 0; j < 4; j++) if (!noid::eq(to_u(t[j]), s[j])) tbad++;
    }
    printf("permutation (tower engine): %s\n", tbad ? "FAIL" : "ok");
    pbad += cbad + tbad;
    return (bad || pbad) ? 1 : 0;
}
