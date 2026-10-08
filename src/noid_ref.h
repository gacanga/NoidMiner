// noid_ref.h - portable CPU reference of the ParanO(1)d (NOID) Poseidon2b PoW.
//
//   pow_digest = Poseidon2b sponge (rate 2, capacity IV "POWHDR__") over the
//                16 header fields (GF(2^128), tower basis), no padding,
//                squeeze state[0] || state[1] (tower basis, little-endian).
//   valid      : pow_digest < target, both read as 256-bit little-endian.
//
// The permutation runs in the flat (GCM polynomial) basis, exactly like the
// official noid_poseidon2b crate. Portable C++ (no __int128) so it builds with
// MinGW g++, MSVC and Linux g++.
#pragma once
#include <stdint.h>
#include <string>
#include "noid_consts.h"

namespace noid {

typedef noid_u128 u128;

static inline u128 mk(uint64_t lo, uint64_t hi) { u128 r; r.lo = lo; r.hi = hi; return r; }
static inline u128 x_or(u128 a, u128 b) { return mk(a.lo ^ b.lo, a.hi ^ b.hi); }
static inline bool is_zero(u128 a) { return (a.lo | a.hi) == 0; }
static inline bool eq(u128 a, u128 b) { return a.lo == b.lo && a.hi == b.hi; }

// flat-basis arithmetic modulo x^128 + x^7 + x^2 + x + 1
u128 gf_mul(u128 a, u128 b);
u128 gf_sqr(u128 a);

// basis conversions (GF(2)-linear maps)
u128 tower_to_flat(u128 v);
u128 flat_to_tower(u128 v);

// Poseidon2b permutation on a flat-basis state
void permute_flat(u128 s[4]);

// 16 header fields in tower basis -> 32-byte digest
void pow_digest(const u128 fields[16], uint8_t out[32]);

// a < b as 256-bit little-endian integers (equality fails)
bool le256_lt(const uint8_t a[32], const uint8_t b[32]);

// per-template precomputation shared with the GPU engine
struct Job {
    u128 fields[16];      // tower basis, as received (field 10 = nonce)
    u128 mid[4];          // flat state after absorbing blocks 0..4 (fields 0..9)
    u128 f11, f12, f13, f14, f15;   // flat basis
};
void prepare_job(Job& j, const u128 fields[16]);
// digest for this job with nonce (tower basis value, LE u128)
void job_digest(const Job& j, u128 nonce, uint8_t out[32]);

// hex helpers
bool hex_to_bytes(const std::string& hex, uint8_t* out, size_t n);
std::string bytes_to_hex(const uint8_t* p, size_t n);
u128 u128_from_le(const uint8_t* p);
void u128_to_le(u128 v, uint8_t* p);

// self test against tests/vectors.txt lines "f0,...,f15 digesthex"
// returns number of failures (prints details), -1 if file missing
int self_test_vectors(const char* path, int verbose);

} // namespace noid
