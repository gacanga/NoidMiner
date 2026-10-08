// noid_gpu.h - C API of noidgpu.dll (CUDA engine of NoidMiner).
// Plain C types only: the DLL is built with nvcc + MSVC, the miner with MinGW g++.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Per-template data prepared by the host (all field elements in FLAT basis,
// 4 x u32 little-endian words each).
typedef struct NoidGpuJob {
    uint32_t mid[16];        // sponge state after absorbing fields 0..9
    uint32_t f11[4];         // block 5 second lane
    uint32_t f12[4], f13[4]; // block 6
    uint32_t f14[4], f15[4]; // block 7
    uint32_t nonce_base[4];  // flat(prefix<<64 | hi32<<32); the GPU adds flat(lo32)
    uint32_t target_hi[2];   // candidate if digest bits 192..255 <= this (lo word, hi word)
    uint32_t reserved[2];
} NoidGpuJob;

// modes for the MDS constant multiplications
#define NOID_MODE_TABLE 0    // shared-memory nibble tables (default)
#define NOID_MODE_KOP   1    // Karatsuba with pre-split constants
#define NOID_MODE_TOWER 2    // partial rounds in the tower basis (16-bit chunk tables)

int         noid_gpu_device_count(void);
// returns NULL on failure (see noid_gpu_last_error(NULL))
void*       noid_gpu_create(int device, int mode, int threads_per_block, int blocks_per_sm, int nonces_per_thread);
// lanes = nonces interleaved per thread (1 or 2, table/kop engines only)
void*       noid_gpu_create2(int device, int mode, int threads_per_block, int blocks_per_sm, int nonces_per_thread, int lanes);
int         noid_gpu_lanes(void* ctx);
void        noid_gpu_destroy(void* ctx);
const char* noid_gpu_name(void* ctx);
const char* noid_gpu_last_error(void* ctx);
uint32_t    noid_gpu_batch(void* ctx);              // nonces per scan call (SM x bps x tpb x npt x lanes)
int         noid_gpu_set_nonces_per_thread(void* ctx, int npt);
int         noid_gpu_set_job(void* ctx, const NoidGpuJob* job);
// scans lo32 in [lo32_start, lo32_start + batch); returns 0 on success.
// found: candidate lo32 values (host must verify), n_found may exceed max_found.
int         noid_gpu_scan(void* ctx, uint32_t lo32_start, uint32_t* found, int max_found, int* n_found);
// test helper: final flat state lanes 0,1 (8 words per nonce) for count nonces
int         noid_gpu_digest_test(void* ctx, uint32_t lo32_start, int count, uint32_t* out_states);

#ifdef __cplusplus
}
#endif
