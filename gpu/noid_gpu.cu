// noid_gpu.cu - CUDA engine of NoidMiner (ParanO(1)d / NOID, Poseidon2b PoW).
//
// Target: NVIDIA Pascal (P104-100, sm_61), CUDA 12.9, built as noidgpu.dll.
// Each thread hashes nonces_per_thread nonces: 3 Poseidon2b permutations per
// nonce from the per-template midstate (fields 0..9 are nonce independent).
//
// Engine modes (constant multiplications of the MDS layers):
//   table : flat basis, shared-memory nibble tables            (48 KiB smem)
//   kop   : flat basis, Karatsuba with pre-split constants     (no smem)
//   tower : partial rounds with lanes 1..3 in the tower basis  (40 KiB smem)
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "noid_gpu.h"
#include "noid_perm.h"
#include "../src/noid_consts.h"

#define MAX_FOUND 1024

__constant__ W4 c_rc[4][66];
__constant__ W4 c_t2f_lo32[32];
__constant__ KOp c_kop[NOID_NCONST];
__constant__ NoidGpuJob c_job;

struct RCDev {
    __device__ __forceinline__ W4 get(int lane, int r) { return c_rc[lane][r]; }
};

struct CMulKopConst {
    __device__ __forceinline__ W4 mul(int c, W4 a) { KOp ka = kop(a); return kmul(ka, c_kop[c]); }
};

static __host__ __device__ __forceinline__ int smem_words(int mode)
{
    return mode == NOID_MODE_TABLE ? NOID_TBL_WORDS : (mode == NOID_MODE_TOWER ? NOID_TOWER_TBL_WORDS : 0);
}

__device__ __forceinline__ W4 ldw4(const uint32_t* p) { return w4(p[0], p[1], p[2], p[3]); }

__device__ __forceinline__ W4 t2f_lo32(uint32_t v)
{
    W4 r = w4(0, 0, 0, 0);
#pragma unroll
    for (int i = 0; i < 32; i++) {
        uint32_t m = 0u - ((v >> i) & 1u);
        W4 c = c_t2f_lo32[i];
        r.w0 ^= c.w0 & m; r.w1 ^= c.w1 & m; r.w2 ^= c.w2 & m; r.w3 ^= c.w3 & m;
    }
    return r;
}

// bits 64..127 of flat_to_tower(a): nibble tables in global memory (4 KiB)
__device__ __forceinline__ uint64_t f2t_top64(W4 a, const uint2* __restrict__ t)
{
    uint32_t lo = 0, hi = 0;
    uint32_t w[4] = { a.w0, a.w1, a.w2, a.w3 };
#pragma unroll
    for (int k = 0; k < 32; k++) {
        uint32_t v = (w[k >> 3] >> (4 * (k & 7))) & 15u;
        uint2 e = __ldg(&t[k * 16 + v]);
        lo ^= e.x; hi ^= e.y;
    }
    return ((uint64_t)hi << 32) | lo;
}

template <int MODE>
__device__ __forceinline__ void perm_any(W4 s[4], const uint32_t* stbl)
{
    RCDev rc;
    if (MODE == NOID_MODE_TABLE) {
        CMulTable cm{ stbl };
        permute(s, cm, rc);
    } else if (MODE == NOID_MODE_KOP) {
        CMulKopConst cm;
        permute(s, cm, rc);
    } else {
        TowerTables tb{ stbl };
        permute_tower(s, tb, rc);
    }
}

// 3 permutations per nonce (blocks 5, 6, 7) with a single permutation instance
template <int MODE>
__device__ __forceinline__ void hash_nonce(uint32_t lo32, W4 s[4], const uint32_t* stbl)
{
    W4 nf = w4x(ldw4(c_job.nonce_base), t2f_lo32(lo32));
    s[0] = w4x(ldw4(c_job.mid + 0), nf);
    s[1] = w4x(ldw4(c_job.mid + 4), ldw4(c_job.f11));
    s[2] = ldw4(c_job.mid + 8);
    s[3] = ldw4(c_job.mid + 12);
#pragma unroll 1
    for (int b = 0; b < 3; b++) {
        perm_any<MODE>(s, stbl);
        if (b == 0) { s[0] = w4x(s[0], ldw4(c_job.f12)); s[1] = w4x(s[1], ldw4(c_job.f13)); }
        if (b == 1) { s[0] = w4x(s[0], ldw4(c_job.f14)); s[1] = w4x(s[1], ldw4(c_job.f15)); }
    }
}

// N nonces per thread interleaved (table / kop engines)
template <int MODE, int N>
__device__ __forceinline__ void hash_nonce_n(const uint32_t lo32[N], W4 s[N][4], const uint32_t* stbl)
{
    RCDev rc;
    const W4 base = ldw4(c_job.nonce_base);
    const W4 m0 = ldw4(c_job.mid + 0), m1 = w4x(ldw4(c_job.mid + 4), ldw4(c_job.f11));
    const W4 m2 = ldw4(c_job.mid + 8), m3 = ldw4(c_job.mid + 12);
#pragma unroll
    for (int n = 0; n < N; n++) {
        s[n][0] = w4x(m0, w4x(base, t2f_lo32(lo32[n])));
        s[n][1] = m1;
        s[n][2] = m2;
        s[n][3] = m3;
    }
#pragma unroll 1
    for (int b = 0; b < 3; b++) {
        if (MODE == NOID_MODE_KOP) { CMulKopConst cm; permute_n<N>(s, cm, rc); }
        else { CMulTable cm{ stbl }; permute_n<N>(s, cm, rc); }
        if (b < 2) {
            const W4 a0 = ldw4(b == 0 ? c_job.f12 : c_job.f14);
            const W4 a1 = ldw4(b == 0 ? c_job.f13 : c_job.f15);
#pragma unroll
            for (int n = 0; n < N; n++) { s[n][0] = w4x(s[n][0], a0); s[n][1] = w4x(s[n][1], a1); }
        }
    }
}

template <int MODE>
__device__ __forceinline__ void load_tables(const uint32_t* __restrict__ gtbl, uint32_t* stbl)
{
    if (MODE != NOID_MODE_KOP) {
        const int n = smem_words(MODE);
        for (int i = threadIdx.x; i < n; i += blockDim.x) stbl[i] = gtbl[i];
        __syncthreads();
    }
}

template <int MODE, int N, int TPB, int MINB>
__global__ void __launch_bounds__(TPB, MINB)
scan_kernel(const uint32_t* __restrict__ gtbl, const uint2* __restrict__ f2t, uint32_t lo32_start, int npt, uint32_t* out)
{
    extern __shared__ uint32_t stbl[];
    load_tables<MODE>(gtbl, stbl);
    const uint32_t total = gridDim.x * blockDim.x;
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t target = ((uint64_t)c_job.target_hi[1] << 32) | c_job.target_hi[0];
    for (int it = 0; it < npt; it++) {
        if (N == 1 || MODE == NOID_MODE_TOWER) {
            uint32_t lo32 = lo32_start + gid + (uint32_t)it * total;
            W4 s[4];
            hash_nonce<MODE>(lo32, s, stbl);
            uint64_t top = f2t_top64(s[1], f2t);
            if (top <= target) {
                uint32_t idx = atomicAdd(out, 1u);
                if (idx < MAX_FOUND) out[1 + idx] = lo32;
            }
        } else {
            uint32_t lo32[N];
            W4 s[N][4];
#pragma unroll
            for (int n = 0; n < N; n++) lo32[n] = lo32_start + gid + (uint32_t)(it * N + n) * total;
            hash_nonce_n<MODE, N>(lo32, s, stbl);
#pragma unroll
            for (int n = 0; n < N; n++) {
                uint64_t top = f2t_top64(s[n][1], f2t);
                if (top <= target) {
                    uint32_t idx = atomicAdd(out, 1u);
                    if (idx < MAX_FOUND) out[1 + idx] = lo32[n];
                }
            }
        }
    }
}

template <int MODE>
__global__ void digest_kernel(const uint32_t* __restrict__ gtbl, uint32_t lo32_start, int count, uint32_t* out)
{
    extern __shared__ uint32_t stbl[];
    load_tables<MODE>(gtbl, stbl);
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    W4 s[4];
    hash_nonce<MODE>(lo32_start + gid, s, stbl);
    uint32_t* o = out + 8 * gid;
    o[0] = s[0].w0; o[1] = s[0].w1; o[2] = s[0].w2; o[3] = s[0].w3;
    o[4] = s[1].w0; o[5] = s[1].w1; o[6] = s[1].w2; o[7] = s[1].w3;
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
struct Ctx {
    int dev, mode, tpb, bps, npt, nsm, lanes;
    uint32_t* d_tbl;
    uint2* d_f2t;
    uint32_t* d_out;
    uint32_t* h_out;
    cudaStream_t st;
    char name[256];
    char err[512];
};

static char g_err[512];

static W4 u2w(noid_u128 v) { return w4((uint32_t)v.lo, (uint32_t)(v.lo >> 32), (uint32_t)v.hi, (uint32_t)(v.hi >> 32)); }

static noid_u128 apply_cols_host(const noid_u128* cols, uint64_t lo, uint64_t hi)
{
    noid_u128 r = { 0, 0 };
    for (int i = 0; i < 64; i++) {
        if ((lo >> i) & 1) { r.lo ^= cols[i].lo; r.hi ^= cols[i].hi; }
        if ((hi >> i) & 1) { r.lo ^= cols[64 + i].lo; r.hi ^= cols[64 + i].hi; }
    }
    return r;
}

static bool ck(Ctx* c, cudaError_t e, const char* what)
{
    if (e == cudaSuccess) return true;
    char* dst = c ? c->err : g_err;
    snprintf(dst, 512, "%s: %s", what, cudaGetErrorString(e));
    return false;
}

static bool upload_constants(Ctx* c)
{
    W4 rc[4][66];
    for (int l = 0; l < 4; l++)
        for (int r = 0; r < 66; r++) rc[l][r] = u2w(NOID_RC[l][r]);
    W4 t2f32[32];
    for (int i = 0; i < 32; i++) t2f32[i] = u2w(NOID_T2F[i]);
    W4 consts[NOID_NCONST];
    consts[0] = u2w(apply_cols_host(NOID_T2F, 2, 0));
    consts[1] = u2w(apply_cols_host(NOID_T2F, 4, 0));
    uint32_t ptow[4];
    for (int i = 0; i < 4; i++) {
        noid_u128 d = NOID_MDS_PARTIAL[i][i];
        d.lo ^= 1;
        consts[2 + i] = u2w(d);
        // tower value of d_i + 1 (diagonal 0x20, 0x2000, 0x200, 0x800 in the tower basis)
        noid_u128 tw = apply_cols_host(NOID_F2T, d.lo, d.hi);
        ptow[i] = (uint32_t)tw.lo;
    }
    KOp kops[NOID_NCONST];
    noid_build_const_kops(consts, kops);

    int words = smem_words(c->mode);
    uint32_t* tbl = NULL;
    if (words) {
        tbl = (uint32_t*)malloc((size_t)words * sizeof(uint32_t));
        if (c->mode == NOID_MODE_TABLE) noid_build_const_tables(consts, tbl);
        else {
            W4 t2f[128], f2t[128];
            for (int i = 0; i < 128; i++) { t2f[i] = u2w(NOID_T2F[i]); f2t[i] = u2w(NOID_F2T[i]); }
            noid_build_tower_tables(consts, t2f, f2t, ptow, tbl);
        }
    }
    uint2 f2t[512];
    for (int k = 0; k < 32; k++)
        for (int v = 0; v < 16; v++) {
            uint64_t lo = 0, hi = 0;
            if (k < 16) lo = (uint64_t)v << (4 * k); else hi = (uint64_t)v << (4 * (k - 16));
            noid_u128 t = apply_cols_host(NOID_F2T, lo, hi);
            f2t[k * 16 + v].x = (uint32_t)t.hi;
            f2t[k * 16 + v].y = (uint32_t)(t.hi >> 32);
        }
    bool ok = ck(c, cudaMemcpyToSymbol(c_rc, rc, sizeof rc), "rc")
           && ck(c, cudaMemcpyToSymbol(c_t2f_lo32, t2f32, sizeof t2f32), "t2f")
           && ck(c, cudaMemcpyToSymbol(c_kop, kops, sizeof kops), "kop")
           && (!words || ck(c, cudaMemcpy(c->d_tbl, tbl, (size_t)words * sizeof(uint32_t), cudaMemcpyHostToDevice), "tbl"))
           && ck(c, cudaMemcpy(c->d_f2t, f2t, sizeof f2t, cudaMemcpyHostToDevice), "f2t");
    free(tbl);
    return ok;
}

extern "C" int noid_gpu_device_count(void)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

extern "C" const char* noid_gpu_last_error(void* ctx)
{
    return ctx ? ((Ctx*)ctx)->err : g_err;
}

typedef void (*scan_fn)(const uint32_t*, const uint2*, uint32_t, int, uint32_t*);
typedef void (*digest_fn)(const uint32_t*, uint32_t, int, uint32_t*);

template <int MODE, int N>
static scan_fn pick_scan_m(int tpb)
{
    if (tpb == 384) return scan_kernel<MODE, N, 384, 2>;
    if (tpb == 512) return scan_kernel<MODE, N, 512, 2>;
    return scan_kernel<MODE, N, 256, 2>;
}

// lanes: nonces interleaved per thread (1 or 2; the tower engine always uses 1)
static scan_fn pick_scan(int mode, int tpb, int lanes)
{
    if (mode == NOID_MODE_TOWER) return pick_scan_m<NOID_MODE_TOWER, 1>(tpb);
    if (mode == NOID_MODE_KOP) return lanes == 2 ? pick_scan_m<NOID_MODE_KOP, 2>(tpb) : pick_scan_m<NOID_MODE_KOP, 1>(tpb);
    return lanes == 2 ? pick_scan_m<NOID_MODE_TABLE, 2>(tpb) : pick_scan_m<NOID_MODE_TABLE, 1>(tpb);
}

static digest_fn pick_digest(int mode)
{
    if (mode == NOID_MODE_KOP) return digest_kernel<NOID_MODE_KOP>;
    if (mode == NOID_MODE_TOWER) return digest_kernel<NOID_MODE_TOWER>;
    return digest_kernel<NOID_MODE_TABLE>;
}

extern "C" void* noid_gpu_create2(int device, int mode, int tpb, int bps, int npt, int lanes)
{
    Ctx* c = (Ctx*)calloc(1, sizeof(Ctx));
    c->dev = device;
    c->lanes = (lanes == 2 && mode != NOID_MODE_TOWER) ? 2 : 1;
    c->mode = (mode == NOID_MODE_KOP || mode == NOID_MODE_TOWER) ? mode : NOID_MODE_TABLE;
    c->tpb = (tpb == 384 || tpb == 512) ? tpb : 256;
    c->npt = npt > 0 ? npt : 16;
    cudaSetDevice(device);
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    cudaDeviceProp p;
    if (!ck(c, cudaGetDeviceProperties(&p, device), "props")) { strcpy(g_err, c->err); free(c); return NULL; }
    snprintf(c->name, sizeof c->name, "%.200s (sm_%d%d, %d SM)", p.name, p.major, p.minor, p.multiProcessorCount);
    c->nsm = p.multiProcessorCount;
    c->bps = bps > 0 ? bps : 2;
    int smem = smem_words(c->mode) * 4;
    if (smem) {
        if (!ck(c, cudaFuncSetAttribute(pick_scan(c->mode, c->tpb, c->lanes), cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "smem attr") ||
            !ck(c, cudaFuncSetAttribute(pick_digest(c->mode), cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "smem attr2")) {
            strcpy(g_err, c->err); free(c); return NULL;
        }
    }
    int words = smem_words(c->mode);
    bool ok = ck(c, cudaStreamCreateWithFlags(&c->st, cudaStreamNonBlocking), "stream")
           && ck(c, cudaMalloc(&c->d_tbl, (size_t)(words ? words : 1) * 4), "malloc tbl")
           && ck(c, cudaMalloc(&c->d_f2t, 512 * sizeof(uint2)), "malloc f2t")
           && ck(c, cudaMalloc(&c->d_out, (1 + MAX_FOUND) * 4), "malloc out")
           && ck(c, cudaMallocHost(&c->h_out, (1 + MAX_FOUND) * 4), "malloc host")
           && upload_constants(c);
    if (!ok) { strcpy(g_err, c->err); free(c); return NULL; }
    return c;
}

extern "C" void* noid_gpu_create(int device, int mode, int tpb, int bps, int npt)
{
    return noid_gpu_create2(device, mode, tpb, bps, npt, 1);
}

extern "C" int noid_gpu_lanes(void* ctx) { return ((Ctx*)ctx)->lanes; }

extern "C" void noid_gpu_destroy(void* ctx)
{
    Ctx* c = (Ctx*)ctx;
    if (!c) return;
    cudaSetDevice(c->dev);
    cudaFree(c->d_tbl); cudaFree(c->d_f2t); cudaFree(c->d_out); cudaFreeHost(c->h_out);
    cudaStreamDestroy(c->st);
    free(c);
}

extern "C" const char* noid_gpu_name(void* ctx) { return ((Ctx*)ctx)->name; }

extern "C" uint32_t noid_gpu_batch(void* ctx)
{
    Ctx* c = (Ctx*)ctx;
    return (uint32_t)c->nsm * c->bps * c->tpb * c->npt * c->lanes;
}

extern "C" int noid_gpu_set_nonces_per_thread(void* ctx, int npt)
{
    Ctx* c = (Ctx*)ctx;
    if (npt < 1) npt = 1;
    if (npt > 4096) npt = 4096;
    c->npt = npt;
    return 0;
}

extern "C" int noid_gpu_set_job(void* ctx, const NoidGpuJob* job)
{
    Ctx* c = (Ctx*)ctx;
    cudaSetDevice(c->dev);
    return ck(c, cudaMemcpyToSymbolAsync(c_job, job, sizeof(NoidGpuJob), 0, cudaMemcpyHostToDevice, c->st), "set job") ? 0 : -1;
}

extern "C" int noid_gpu_scan(void* ctx, uint32_t lo32_start, uint32_t* found, int max_found, int* n_found)
{
    Ctx* c = (Ctx*)ctx;
    cudaSetDevice(c->dev);
    *n_found = 0;
    if (!ck(c, cudaMemsetAsync(c->d_out, 0, 4, c->st), "memset")) return -1;
    dim3 grid(c->nsm * c->bps), block(c->tpb);
    pick_scan(c->mode, c->tpb, c->lanes)<<<grid, block, smem_words(c->mode) * 4, c->st>>>(c->d_tbl, c->d_f2t, lo32_start, c->npt, c->d_out);
    if (!ck(c, cudaGetLastError(), "launch")) return -1;
    if (!ck(c, cudaMemcpyAsync(c->h_out, c->d_out, (1 + MAX_FOUND) * 4, cudaMemcpyDeviceToHost, c->st), "copy out")) return -1;
    if (!ck(c, cudaStreamSynchronize(c->st), "sync")) return -1;
    int n = (int)c->h_out[0];
    *n_found = n;
    int m = n < MAX_FOUND ? n : MAX_FOUND;
    if (m > max_found) m = max_found;
    for (int i = 0; i < m; i++) found[i] = c->h_out[1 + i];
    return 0;
}

extern "C" int noid_gpu_digest_test(void* ctx, uint32_t lo32_start, int count, uint32_t* out_states)
{
    Ctx* c = (Ctx*)ctx;
    cudaSetDevice(c->dev);
    uint32_t* d = NULL;
    if (!ck(c, cudaMalloc(&d, (size_t)count * 32), "malloc test")) return -1;
    int blocks = (count + 255) / 256;
    pick_digest(c->mode)<<<blocks, 256, smem_words(c->mode) * 4, c->st>>>(c->d_tbl, lo32_start, count, d);
    bool ok = ck(c, cudaGetLastError(), "launch test")
           && ck(c, cudaMemcpyAsync(out_states, d, (size_t)count * 32, cudaMemcpyDeviceToHost, c->st), "copy test")
           && ck(c, cudaStreamSynchronize(c->st), "sync test");
    cudaFree(d);
    return ok ? 0 : -1;
}
