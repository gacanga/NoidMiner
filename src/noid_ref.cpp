// noid_ref.cpp - portable CPU reference of the NOID Poseidon2b PoW (see noid_ref.h)
#include "noid_ref.h"
#include <stdio.h>
#include <string.h>
#include <string>

namespace noid {

// ---------------------------------------------------------------------------
// GF(2^128) flat basis
// ---------------------------------------------------------------------------
#if defined(__PCLMUL__) && (defined(__x86_64__) || defined(_M_X64))
#include <wmmintrin.h>
#include <emmintrin.h>
static inline void clmul64(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi)
{
    __m128i r = _mm_clmulepi64_si128(_mm_set_epi64x(0, (long long)a), _mm_set_epi64x(0, (long long)b), 0x00);
    lo = (uint64_t)_mm_cvtsi128_si64(r);
    hi = (uint64_t)_mm_cvtsi128_si64(_mm_unpackhi_epi64(r, r));
}
#else
static inline void clmul64(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi)
{
    uint64_t l = 0, h = 0;
    for (int i = 0; i < 64; i++) {
        if ((b >> i) & 1) {
            l ^= a << i;
            if (i) h ^= a >> (64 - i);
        }
    }
    lo = l; hi = h;
}
#endif

// reduce (r3 r2 r1 r0) modulo x^128 + x^7 + x^2 + x + 1
static inline u128 reduce256(uint64_t r0, uint64_t r1, uint64_t r2, uint64_t r3)
{
    // fold r3 (x^192..x^255): x^(192+i) = x^(64+i) * (x^7 + x^2 + x + 1)
    uint64_t t = r3;
    r1 ^= t ^ (t << 1) ^ (t << 2) ^ (t << 7);
    r2 ^= (t >> 63) ^ (t >> 62) ^ (t >> 57);
    // fold r2 (x^128..x^191) into r1/r0
    t = r2;
    r1 ^= (t >> 63) ^ (t >> 62) ^ (t >> 57);
    r0 ^= t ^ (t << 1) ^ (t << 2) ^ (t << 7);
    return mk(r0, r1);
}

u128 gf_mul(u128 a, u128 b)
{
    uint64_t l0, h0, l1, h1, lm, hm;
    clmul64(a.lo, b.lo, l0, h0);
    clmul64(a.hi, b.hi, l1, h1);
    clmul64(a.lo ^ a.hi, b.lo ^ b.hi, lm, hm);
    lm ^= l0 ^ l1;
    hm ^= h0 ^ h1;
    // product = (h1:l1) x^128 + (hm:lm) x^64 + (h0:l0)
    return reduce256(l0, h0 ^ lm, l1 ^ hm, h1);
}

static inline uint64_t spread32(uint32_t v)
{
    uint64_t x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x << 8))  & 0x00FF00FF00FF00FFULL;
    x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x << 2))  & 0x3333333333333333ULL;
    x = (x | (x << 1))  & 0x5555555555555555ULL;
    return x;
}

u128 gf_sqr(u128 a)
{
    uint64_t r0 = spread32((uint32_t)a.lo), r1 = spread32((uint32_t)(a.lo >> 32));
    uint64_t r2 = spread32((uint32_t)a.hi), r3 = spread32((uint32_t)(a.hi >> 32));
    return reduce256(r0, r1, r2, r3);
}

static inline u128 apply_cols(const u128 cols[128], u128 v)
{
    u128 r = mk(0, 0);
    for (int i = 0; i < 64; i++) {
        if ((v.lo >> i) & 1) r = x_or(r, cols[i]);
        if ((v.hi >> i) & 1) r = x_or(r, cols[64 + i]);
    }
    return r;
}

u128 tower_to_flat(u128 v) { return apply_cols(NOID_T2F, v); }
u128 flat_to_tower(u128 v) { return apply_cols(NOID_F2T, v); }

// ---------------------------------------------------------------------------
// Poseidon2b permutation (t = 4, x^7, 8 full + 58 partial rounds)
// ---------------------------------------------------------------------------
static inline u128 sbox7(u128 x)
{
    u128 x2 = gf_sqr(x);
    u128 x4 = gf_sqr(x2);
    u128 x6 = gf_mul(x, x2);
    return gf_mul(x6, x4);
}

static inline bool is_one(u128 c) { return c.lo == 1 && c.hi == 0; }

static void mds(u128 s[4], const u128 m[4][4])
{
    u128 in[4] = { s[0], s[1], s[2], s[3] };
    for (int i = 0; i < 4; i++) {
        u128 o = mk(0, 0);
        for (int j = 0; j < 4; j++)
            o = x_or(o, is_one(m[i][j]) ? in[j] : gf_mul(in[j], m[i][j]));
        s[i] = o;
    }
}

void permute_flat(u128 s[4])
{
    mds(s, NOID_MDS_FULL);
    for (int r = 0; r < NOID_N_ROUNDS; r++) {
        bool full = r < NOID_F_ROUNDS / 2 || r >= NOID_F_ROUNDS / 2 + NOID_P_ROUNDS;
        if (full) {
            for (int i = 0; i < 4; i++) s[i] = sbox7(x_or(s[i], NOID_RC[i][r]));
            mds(s, NOID_MDS_FULL);
        } else {
            s[0] = sbox7(x_or(s[0], NOID_RC[0][r]));
            mds(s, NOID_MDS_PARTIAL);
        }
    }
}

// ---------------------------------------------------------------------------
// sponge / digest
// ---------------------------------------------------------------------------
void u128_to_le(u128 v, uint8_t* p)
{
    for (int i = 0; i < 8; i++) { p[i] = (uint8_t)(v.lo >> (8 * i)); p[8 + i] = (uint8_t)(v.hi >> (8 * i)); }
}

u128 u128_from_le(const uint8_t* p)
{
    uint64_t lo = 0, hi = 0;
    for (int i = 7; i >= 0; i--) { lo = (lo << 8) | p[i]; hi = (hi << 8) | p[8 + i]; }
    return mk(lo, hi);
}

void pow_digest(const u128 fields[16], uint8_t out[32])
{
    u128 s[4] = { mk(0, 0), mk(0, 0), NOID_IV[0], NOID_IV[1] };
    for (int k = 0; k < 8; k++) {
        s[0] = x_or(s[0], tower_to_flat(fields[2 * k]));
        s[1] = x_or(s[1], tower_to_flat(fields[2 * k + 1]));
        permute_flat(s);
    }
    u128_to_le(flat_to_tower(s[0]), out);
    u128_to_le(flat_to_tower(s[1]), out + 16);
}

bool le256_lt(const uint8_t a[32], const uint8_t b[32])
{
    for (int i = 31; i >= 0; i--) {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
    }
    return false;
}

void prepare_job(Job& j, const u128 fields[16])
{
    for (int i = 0; i < 16; i++) j.fields[i] = fields[i];
    u128 s[4] = { mk(0, 0), mk(0, 0), NOID_IV[0], NOID_IV[1] };
    for (int k = 0; k < 5; k++) {
        s[0] = x_or(s[0], tower_to_flat(fields[2 * k]));
        s[1] = x_or(s[1], tower_to_flat(fields[2 * k + 1]));
        permute_flat(s);
    }
    for (int i = 0; i < 4; i++) j.mid[i] = s[i];
    j.f11 = tower_to_flat(fields[11]);
    j.f12 = tower_to_flat(fields[12]);
    j.f13 = tower_to_flat(fields[13]);
    j.f14 = tower_to_flat(fields[14]);
    j.f15 = tower_to_flat(fields[15]);
}

void job_digest(const Job& j, u128 nonce, uint8_t out[32])
{
    u128 s[4] = { j.mid[0], j.mid[1], j.mid[2], j.mid[3] };
    s[0] = x_or(s[0], tower_to_flat(nonce));
    s[1] = x_or(s[1], j.f11);
    permute_flat(s);
    s[0] = x_or(s[0], j.f12); s[1] = x_or(s[1], j.f13);
    permute_flat(s);
    s[0] = x_or(s[0], j.f14); s[1] = x_or(s[1], j.f15);
    permute_flat(s);
    u128_to_le(flat_to_tower(s[0]), out);
    u128_to_le(flat_to_tower(s[1]), out + 16);
}

// ---------------------------------------------------------------------------
// hex helpers
// ---------------------------------------------------------------------------
static int hexval(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool hex_to_bytes(const std::string& hex, uint8_t* out, size_t n)
{
    if (hex.size() != 2 * n) return false;
    for (size_t i = 0; i < n; i++) {
        int a = hexval(hex[2 * i]), b = hexval(hex[2 * i + 1]);
        if (a < 0 || b < 0) return false;
        out[i] = (uint8_t)(a * 16 + b);
    }
    return true;
}

std::string bytes_to_hex(const uint8_t* p, size_t n)
{
    static const char* d = "0123456789abcdef";
    std::string s(2 * n, '0');
    for (size_t i = 0; i < n; i++) { s[2 * i] = d[p[i] >> 4]; s[2 * i + 1] = d[p[i] & 15]; }
    return s;
}

// big-endian 32-hex-digit integer -> u128
static bool parse_u128_be(const std::string& h, u128& v)
{
    if (h.size() != 32) return false;
    uint64_t hi = 0, lo = 0;
    for (int i = 0; i < 16; i++) { int x = hexval(h[i]); if (x < 0) return false; hi = (hi << 4) | (uint64_t)x; }
    for (int i = 16; i < 32; i++) { int x = hexval(h[i]); if (x < 0) return false; lo = (lo << 4) | (uint64_t)x; }
    v = mk(lo, hi);
    return true;
}

int self_test_vectors(const char* path, int verbose)
{
    FILE* f = fopen(path, "r");
    if (!f) { printf("self-test: cannot open %s\n", path); return -1; }
    char line[2048];
    int n = 0, bad = 0;
    while (fgets(line, sizeof line, f)) {
        std::string l(line);
        while (!l.empty() && (l.back() == '\n' || l.back() == '\r' || l.back() == ' ')) l.pop_back();
        if (l.empty() || l[0] == '#') continue;
        size_t sp = l.find(' ');
        if (sp == std::string::npos) continue;
        std::string fs = l.substr(0, sp), dh = l.substr(sp + 1);
        u128 fields[16];
        size_t pos = 0;
        bool ok = true;
        for (int i = 0; i < 16 && ok; i++) {
            size_t c = fs.find(',', pos);
            std::string one = fs.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
            ok = parse_u128_be(one, fields[i]);
            pos = c + 1;
        }
        uint8_t want[32], got[32], got2[32];
        if (!ok || !hex_to_bytes(dh, want, 32)) { printf("self-test: bad line %d\n", n); bad++; n++; continue; }
        pow_digest(fields, got);
        Job j; prepare_job(j, fields);
        job_digest(j, fields[10], got2);
        if (memcmp(got, want, 32) != 0 || memcmp(got2, want, 32) != 0) {
            bad++;
            if (verbose) printf("self-test: vector %d MISMATCH\n  want %s\n  got  %s\n  job  %s\n", n,
                                dh.c_str(), bytes_to_hex(got, 32).c_str(), bytes_to_hex(got2, 32).c_str());
        }
        n++;
    }
    fclose(f);
    if (verbose) printf("self-test: %d vectors, %d failures\n", n, bad);
    return bad;
}

} // namespace noid
