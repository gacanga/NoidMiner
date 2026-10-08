// NoidMiner - GPU miner for ParanO(1)d (NOID), Poseidon2b PoW, InnovLab pool
// protocol (parano1d-stratum-v1 over TLS). No developer fee.
//
//   noidminer.exe                     mine with noidminer.conf
//   noidminer.exe --test              CPU vectors + GPU correctness tests
//   noidminer.exe --bench [seconds]   hashrate benchmark on all selected GPUs
//
// Options (also accepted as key=value lines in noidminer.conf, without "--"):
//   -o / --pool URL      pool url (repeatable, failover order)
//   -u / --user ADDR.RIG -p / --pass X
//   --gpus 0,1,2|all     --mode tower|table|kop     --npt N (0 = auto)
//   --tpb 256|384|512 (threads per block)  --lanes 1|2 (nonces interleaved per thread)  --bps N (blocks per SM)   --target-ms N (scan duration, auto npt)
//   --cafile FILE        --ns-order auto|int|bytes   --stats N (seconds)
//   --log FILE           --args-file FILE (extra arguments read from a file)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <string>
#include <vector>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <signal.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "noid_ref.h"
#include "json.h"
#include "tls_conn.h"
#include "../gpu/noid_gpu.h"

#define NOIDMINER_VERSION "0.3.0"

using Clock = std::chrono::steady_clock;
using noid::u128;

// ---------------------------------------------------------------------------
// logging
// ---------------------------------------------------------------------------
static std::mutex g_log_mtx;
static FILE* g_logf = nullptr;

static void logf_(const char* fmt, ...)
{
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    time_t t = time(nullptr);
    struct tm tmv;
#ifdef _WIN32
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    char ts[32];
    strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", &tmv);
    std::lock_guard<std::mutex> lk(g_log_mtx);
    printf("[%s] %s\n", ts, msg);
    fflush(stdout);
    if (g_logf) { fprintf(g_logf, "[%s] %s\n", ts, msg); fflush(g_logf); }
}
#define LOG(...) logf_(__VA_ARGS__)

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------
struct Config {
    std::vector<std::string> pools;
    std::string user, pass = "x";
    std::vector<int> gpus;          // empty = all
    int mode = NOID_MODE_TABLE;
    int npt = 0;                    // 0 = auto
    int bps = 2;
    int tpb = 512;
    int lanes = 1;
    int target_ms = 200;
    std::string cafile;
    std::string ns_order = "auto";
    int stats = 30;
    std::string logfile;
    bool test = false;
    bool bench = false;
    int bench_seconds = 30;
    std::string exe_dir = ".";
};

static std::string trim(const std::string& s)
{
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

static bool apply_option(Config& c, const std::string& key, const std::string& val)
{
    std::string k = key;
    while (!k.empty() && k[0] == '-') k.erase(0, 1);
    if (k == "o" || k == "pool" || k == "url") c.pools.push_back(val);
    else if (k == "u" || k == "user") c.user = val;
    else if (k == "p" || k == "pass") c.pass = val;
    else if (k == "gpus" || k == "devices") {
        c.gpus.clear();
        if (val != "all" && !val.empty()) {
            size_t p = 0;
            while (p <= val.size()) {
                size_t q = val.find(',', p);
                std::string one = trim(val.substr(p, q == std::string::npos ? std::string::npos : q - p));
                if (!one.empty()) c.gpus.push_back(atoi(one.c_str()));
                if (q == std::string::npos) break;
                p = q + 1;
            }
        }
    }
    else if (k == "mode") c.mode = (val == "kop" || val == "1") ? NOID_MODE_KOP : ((val == "tower" || val == "2") ? NOID_MODE_TOWER : NOID_MODE_TABLE);
    else if (k == "npt") c.npt = atoi(val.c_str());
    else if (k == "bps") c.bps = atoi(val.c_str());
    else if (k == "tpb") c.tpb = atoi(val.c_str());
    else if (k == "lanes") c.lanes = atoi(val.c_str()) == 2 ? 2 : 1;
    else if (k == "target-ms" || k == "target_ms") c.target_ms = atoi(val.c_str());
    else if (k == "cafile") c.cafile = val;
    else if (k == "ns-order" || k == "ns_order") c.ns_order = val;
    else if (k == "stats") c.stats = atoi(val.c_str());
    else if (k == "log") c.logfile = val;
    else return false;
    return true;
}

static bool load_config_file(Config& c, const std::string& path)
{
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        std::string l = trim(line);
        if (l.empty() || l[0] == '#' || l[0] == ';') continue;
        size_t eq = l.find('=');
        if (eq == std::string::npos) continue;
        std::string k = trim(l.substr(0, eq)), v = trim(l.substr(eq + 1));
        if (!apply_option(c, k, v)) fprintf(stderr, "config: unknown key '%s'\n", k.c_str());
    }
    fclose(f);
    return true;
}

static std::string exe_directory(const char* argv0)
{
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(NULL, buf, MAX_PATH);
    std::string p = n ? std::string(buf, n) : std::string(argv0);
#else
    std::string p = argv0;
#endif
    size_t s = p.find_last_of("\\/");
    return s == std::string::npos ? "." : p.substr(0, s);
}

static bool file_exists(const std::string& p)
{
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// shared mining state
// ---------------------------------------------------------------------------
struct PoolJob {
    uint64_t seq = 0;
    std::string job_id, work_domain_id, assignment_id;
    u128 fields[16];
    uint8_t share_target[32];
    uint8_t block_target[32];
    std::string prefix_hex;
    int nonce_bits = 64;
    long long height = 0;
    Clock::time_point expires;
};

struct Share {
    uint64_t seq;
    double work;
    std::string job_id, work_domain_id, nonce_hex;
    int gpu;
    bool block;
};

static std::atomic<bool> g_running(true);
static std::mutex g_mtx;
static std::shared_ptr<const PoolJob> g_job;
static bool g_paused = true;
static uint64_t g_job_seq = 0;
static std::deque<Share> g_shares;
static std::atomic<int> g_ns_order(1);         // 1 = raw bytes (InnovLab), 0 = integer hex
static std::vector<uint64_t> g_valid_seqs;      // jobs still accepted by the pool
static bool g_ns_auto = true;

struct GpuStat {
    std::atomic<uint64_t> hashes{ 0 };
    std::atomic<uint64_t> found{ 0 };
    std::atomic<uint64_t> bad{ 0 };
    std::atomic<int> npt{ 0 };
    std::atomic<int> scan_ms{ 0 };
    std::atomic<bool> alive{ false };
    std::string name;
    int dev = -1;
};
static std::vector<std::unique_ptr<GpuStat>> g_stats;
static std::atomic<uint64_t> g_accepted(0), g_rejected(0), g_blocks(0), g_stale_dropped(0);
static double g_accepted_work = 0;
static std::atomic<uint64_t> g_pauses(0);               // sum of expected hashes of accepted shares

static double share_work(const uint8_t t[32])
{
    // expected hashes per share = 2^256 / target, from the top 64 bits
    uint64_t top = 0;
    for (int i = 31; i >= 24; i--) top = (top << 8) | t[i];
    return top ? 18446744073709551616.0 / (double)top : 0.0;
}

static void on_signal(int) { g_running = false; }

static const char* mode_name(int m) { return m == NOID_MODE_KOP ? "kop" : (m == NOID_MODE_TOWER ? "tower" : "table"); }

// upper 64 bits of the nonce from the pool prefix
static uint64_t prefix_value(const std::string& hex, int order)
{
    uint8_t b[8] = { 0 };
    if (!noid::hex_to_bytes(hex, b, 8)) return 0;
    uint64_t v = 0;
    if (order == 0) { for (int i = 0; i < 8; i++) v = (v << 8) | b[i]; }        // big-endian integer text
    else { for (int i = 7; i >= 0; i--) v = (v << 8) | b[i]; }                  // little-endian bytes
    return v;
}

static std::string nonce_hex(u128 n)
{
    uint8_t b[16];
    noid::u128_to_le(n, b);
    return noid::bytes_to_hex(b, 16);
}

static void make_gpu_job(const noid::Job& j, u128 nonce_base_tower, const uint8_t share_target[32], NoidGpuJob& g)
{
    memset(&g, 0, sizeof g);
    auto put = [](uint32_t* d, u128 v) { d[0] = (uint32_t)v.lo; d[1] = (uint32_t)(v.lo >> 32); d[2] = (uint32_t)v.hi; d[3] = (uint32_t)(v.hi >> 32); };
    for (int i = 0; i < 4; i++) put(g.mid + 4 * i, j.mid[i]);
    put(g.f11, j.f11); put(g.f12, j.f12); put(g.f13, j.f13); put(g.f14, j.f14); put(g.f15, j.f15);
    put(g.nonce_base, noid::tower_to_flat(nonce_base_tower));
    uint64_t t = 0;
    for (int i = 31; i >= 24; i--) t = (t << 8) | share_target[i];
    g.target_hi[0] = (uint32_t)t;
    g.target_hi[1] = (uint32_t)(t >> 32);
}

// ---------------------------------------------------------------------------
// GPU worker
// ---------------------------------------------------------------------------
static void gpu_worker(const Config& cfg, int idx, int dev)
{
    GpuStat& st = *g_stats[idx];
    int npt = cfg.npt > 0 ? cfg.npt : 8;
    void* g = noid_gpu_create2(dev, cfg.mode, cfg.tpb, cfg.bps, npt, cfg.lanes);
    if (!g) { LOG("GPU %d: init failed: %s", dev, noid_gpu_last_error(nullptr)); return; }
    st.name = noid_gpu_name(g);
    st.alive = true;
    LOG("GPU %d: %s, mode %s, %d threads/block, %d lane(s), %u nonces per scan", dev, st.name.c_str(), mode_name(cfg.mode), cfg.tpb, noid_gpu_lanes(g), noid_gpu_batch(g));

    std::mt19937_64 rng((uint64_t)Clock::now().time_since_epoch().count() ^ ((uint64_t)dev << 40));
    uint64_t cur_seq = 0;
    std::shared_ptr<const PoolJob> job;
    noid::Job cj;
    uint64_t upper = 0;
    uint32_t hi32 = 0, lo32 = 0;
    uint32_t hi32_mask = 0xFFFFFFFFu;
    int order = -1;
    std::vector<uint32_t> found(1024);

    while (g_running) {
        std::shared_ptr<const PoolJob> snap;
        bool paused;
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            snap = g_job;
            paused = g_paused;
        }
        if (paused || !snap || Clock::now() > snap->expires) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        int ord = g_ns_order.load();
        if (snap->seq != cur_seq || ord != order) {
            job = snap;
            cur_seq = snap->seq;
            order = ord;
            noid::prepare_job(cj, job->fields);
            upper = prefix_value(job->prefix_hex, order);
            int free_bits = job->nonce_bits;
            if (free_bits < 33) free_bits = 33;
            if (free_bits > 64) free_bits = 64;
            hi32_mask = free_bits >= 64 ? 0xFFFFFFFFu : ((1u << (free_bits - 32)) - 1u);
            // GPU index in the top bits of hi32 keeps the GPUs disjoint
            hi32 = (((uint32_t)idx << 27) | (uint32_t)(rng() & 0x07FFFFFFu)) & hi32_mask;
            lo32 = 0;
            NoidGpuJob gj;
            make_gpu_job(cj, noid::mk((uint64_t)hi32 << 32, upper), job->share_target, gj);
            if (noid_gpu_set_job(g, &gj) != 0) { LOG("GPU %d: set_job failed: %s", dev, noid_gpu_last_error(g)); break; }
        }
        uint32_t batch = noid_gpu_batch(g);
        if ((uint64_t)lo32 + batch > 0x100000000ULL) {
            hi32 = (hi32 + 1) & hi32_mask;
            lo32 = 0;
            NoidGpuJob gj;
            make_gpu_job(cj, noid::mk((uint64_t)hi32 << 32, upper), job->share_target, gj);
            noid_gpu_set_job(g, &gj);
        }
        int nf = 0;
        auto t0 = Clock::now();
        if (noid_gpu_scan(g, lo32, found.data(), (int)found.size(), &nf) != 0) {
            LOG("GPU %d: scan failed: %s", dev, noid_gpu_last_error(g));
            break;
        }
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        st.hashes += batch;
        st.scan_ms = (int)ms;
        if (nf > (int)found.size()) LOG("GPU %d: %d candidates in one scan (share target very easy), extra ones dropped", dev, nf);
        int m = nf < (int)found.size() ? nf : (int)found.size();
        for (int i = 0; i < m; i++) {
            u128 n = noid::mk(((uint64_t)hi32 << 32) | found[i], upper);
            uint8_t d[32];
            noid::job_digest(cj, n, d);
            if (!noid::le256_lt(d, job->share_target)) { st.bad++; continue; }
            st.found++;
            Share s;
            s.seq = job->seq;
            s.work = share_work(job->share_target);
            s.job_id = job->job_id;
            s.work_domain_id = job->work_domain_id;
            s.nonce_hex = nonce_hex(n);
            s.gpu = dev;
            s.block = noid::le256_lt(d, job->block_target);
            if (s.block) LOG("GPU %d: BLOCK candidate at height %lld, nonce %s", dev, job->height, s.nonce_hex.c_str());
            std::lock_guard<std::mutex> lk(g_mtx);
            bool valid = false;
            for (uint64_t v : g_valid_seqs) if (v == s.seq) valid = true;
            if (valid) g_shares.push_back(s); else g_stale_dropped++;
        }
        lo32 += batch;
        // auto intensity: keep each scan close to target_ms
        if (cfg.npt == 0 && ms > 1.0) {
            int want = (int)(npt * cfg.target_ms / ms);
            if (want < 1) want = 1;
            if (want > 4096) want = 4096;
            if (want > npt * 2) want = npt * 2;
            if (want < npt / 2) want = npt / 2 > 0 ? npt / 2 : 1;
            if (abs(want - npt) * 10 > npt) { npt = want; noid_gpu_set_nonces_per_thread(g, npt); }
        }
        st.npt = npt;
    }
    st.alive = false;
    noid_gpu_destroy(g);
}

// ---------------------------------------------------------------------------
// pool client (parano1d-stratum-v1)
// ---------------------------------------------------------------------------
struct Pending { std::string kind; int gpu; std::string nonce; uint64_t seq; bool block; int order; double work; };

static void pool_worker(const Config& cfg, std::string gpu_desc)
{
    TlsConn conn;
    size_t pool_idx = 0;
    int backoff = 2;
    std::string rig = cfg.user.find('.') != std::string::npos ? cfg.user.substr(cfg.user.find('.') + 1) : "rig";
    while (g_running) {
        std::string url = cfg.pools[pool_idx % cfg.pools.size()];
        LOG("pool: connecting to %s", url.c_str());
        if (!conn.connect(url, cfg.cafile, 10000)) {
            LOG("pool: %s", conn.error().c_str());
            pool_idx++;
            for (int i = 0; i < backoff * 10 && g_running; i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            backoff = backoff * 2 > 60 ? 60 : backoff * 2;
            continue;
        }
        LOG("pool: connected to %s (TLS certificate %s)", conn.host().c_str(), conn.verified() ? "verified" : "NOT verified");
        int next_id = 3;
        std::map<int, Pending> pending;
        bool authorized = false;
        std::string stats_method;
        bool stats_ok = true;
        auto t_conn = Clock::now();
        auto t_rx = Clock::now();
        auto t_stats = Clock::now();
        uint64_t stats_hashes = 0;
        for (auto& s : g_stats) stats_hashes += s->hashes;
        auto t_stats_h = Clock::now();
        conn.send_line(std::string("{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"NoidMiner/") + NOIDMINER_VERSION + "\"]}");
        bool lost = false;
        while (g_running && !lost) {
            std::vector<std::string> lines;
            if (!conn.poll_lines(lines, 100)) { LOG("pool: %s", conn.error().c_str()); lost = true; break; }
            if (!lines.empty()) t_rx = Clock::now();
            for (auto& line : lines) {
                json::Value v;
                if (!json::parse(line, v) || !v.is_obj()) { LOG("pool: unparsable line: %.200s", line.c_str()); continue; }
                const json::Value* method = v.get("method");
                if (method && method->is_str()) {
                    const std::string& m = method->s;
                    const json::Value* params = v.get("params");
                    const json::Value* p0 = (params && params->is_arr() && !params->arr.empty()) ? &params->arr[0] : (params && params->is_obj() ? params : nullptr);
                    if (m == "mining.notify" && p0 && p0->is_obj()) {
                        auto j = std::make_shared<PoolJob>();
                        j->job_id = p0->str("job_id");
                        j->work_domain_id = p0->str("work_domain_id");
                        j->assignment_id = p0->str("assignment_id");
                        std::string fh = p0->str("pow_fields_hex");
                        int nfi = (int)p0->num("nonce_field_index", 10);
                        uint8_t fb[256];
                        bool ok = nfi == 10 && noid::hex_to_bytes(fh, fb, 256)
                               && noid::hex_to_bytes(p0->str("share_target_hex"), j->share_target, 32)
                               && noid::hex_to_bytes(p0->str("block_target_hex"), j->block_target, 32);
                        j->prefix_hex = p0->str("nonce_prefix_hex");
                        uint8_t pb[8];
                        ok = ok && noid::hex_to_bytes(j->prefix_hex, pb, 8);
                        if (!ok) { LOG("pool: invalid mining.notify (nonce_field_index %d): %.300s", nfi, line.c_str()); continue; }
                        for (int i = 0; i < 16; i++) j->fields[i] = noid::u128_from_le(fb + 16 * i);
                        j->nonce_bits = (int)p0->num("nonce_bits", 64);
                        j->height = p0->num("height", 0);
                        long long exp = p0->num("expires_in_seconds", 30);
                        if (exp < 2) exp = 2;
                        j->expires = Clock::now() + std::chrono::milliseconds(exp * 1000 - 1000);
                        bool clean = p0->boolean("clean", true);
                        {
                            std::lock_guard<std::mutex> lk(g_mtx);
                            j->seq = ++g_job_seq;
                            g_job = j;
                            g_paused = false;
                            if (clean) g_valid_seqs.clear();
                            g_valid_seqs.push_back(j->seq);
                            if (g_valid_seqs.size() > 16) g_valid_seqs.erase(g_valid_seqs.begin());
                        }
                        LOG("pool: new job %s height %lld (share target %s...)", j->job_id.c_str(), j->height,
                            noid::bytes_to_hex(j->share_target + 24, 8).c_str());
                    } else if (m == "mining.pause") {
                        std::lock_guard<std::mutex> lk(g_mtx);
                        g_paused = true;
                        g_valid_seqs.clear();
                        g_stale_dropped += g_shares.size();
                        g_shares.clear();
                        g_pauses++;
                    } else if (m == "client.reconnect") {
                        LOG("pool: reconnect requested");
                        lost = true;
                    } else {
                        LOG("pool: unhandled method %s: %.200s", m.c_str(), line.c_str());
                    }
                    continue;
                }
                long long id = v.num("id", -1);
                const json::Value* result = v.get("result");
                const json::Value* error = v.get("error");
                bool has_err = error && !error->is_null();
                std::string errs = has_err ? json::dump(*error) : "";
                if (id == 1) {
                    if (has_err || !result || !result->is_obj()) { LOG("pool: subscribe failed: %s", line.c_str()); lost = true; break; }
                    LOG("pool: subscribed, protocol %s, namespace %s, nonce_bits %lld", result->str("protocol").c_str(),
                        result->str("session_namespace").c_str(), result->num("nonce_bits", 64));
                    const json::Value* hr = v.get("hashrate_report");
                    if (!hr) hr = result->get("hashrate_report");
                    if (hr && hr->is_arr() && !hr->arr.empty() && hr->arr[0].is_str()) stats_method = hr->arr[0].s;
                    conn.send_line("{\"id\":2,\"method\":\"mining.authorize\",\"params\":[" + json::quote(cfg.user) + "," + json::quote(cfg.pass) +
                                   ",{\"gpu\":" + json::quote(gpu_desc) + "}]}");
                } else if (id == 2) {
                    if (result && result->is_bool() && result->b) {
                        authorized = true;
                        backoff = 2;
                        LOG("pool: authorized as %s", cfg.user.c_str());
                    } else {
                        LOG("pool: authorization refused: %s", line.c_str());
                        lost = true;
                        backoff = 60;
                        break;
                    }
                } else {
                    auto it = pending.find((int)id);
                    if (it == pending.end()) continue;
                    Pending pd = it->second;
                    pending.erase(it);
                    if (pd.kind == "stats") {
                        if (has_err) {
                            long long code = error->is_obj() ? error->num("code", 0) : 0;
                            if (code == -32601) { stats_ok = false; LOG("pool: %s not supported, hashrate reports disabled", stats_method.c_str()); }
                        }
                        continue;
                    }
                    bool acc = result && result->is_bool() && result->b && !has_err;
                    if (acc) {
                        g_accepted++;
                        { std::lock_guard<std::mutex> lk(g_mtx); g_accepted_work += pd.work; }
                        if (g_ns_auto) g_ns_auto = false;         // nonce namespace format confirmed
                        if (pd.block) g_blocks++;
                        LOG("share accepted (GPU %d)%s  [%llu/%llu]", pd.gpu, pd.block ? " BLOCK" : "",
                            (unsigned long long)g_accepted.load(), (unsigned long long)(g_accepted.load() + g_rejected.load()));
                    } else {
                        g_rejected++;
                        LOG("share REJECTED (GPU %d) nonce %s: %s  [%llu/%llu]", pd.gpu, pd.nonce.c_str(), errs.c_str(),
                            (unsigned long long)g_accepted.load(), (unsigned long long)(g_accepted.load() + g_rejected.load()));
                        // first rejections may come from the namespace byte order: try the other one
                        std::string low = errs;
                        for (auto& ch : low) ch = (char)tolower((unsigned char)ch);
                        bool ns_issue = low.find("namespace") != std::string::npos || low.find("prefix") != std::string::npos ||
                                        low.find("upper") != std::string::npos;
                        if (g_ns_auto && pd.order == g_ns_order.load() && ns_issue) {
                            int no = 1 - g_ns_order.load();
                            g_ns_order = no;
                            LOG("pool: switching nonce namespace byte order to %s", no == 0 ? "integer" : "bytes");
                        }
                    }
                }
            }
            if (lost) break;
            // submit queued shares
            if (authorized) {
                std::deque<Share> q;
                uint64_t seq_now;
                {
                    std::lock_guard<std::mutex> lk(g_mtx);
                    q.swap(g_shares);
                    seq_now = g_job_seq;
                }
                std::vector<uint64_t> valid;
                {
                    std::lock_guard<std::mutex> lk(g_mtx);
                    valid = g_valid_seqs;
                }
                for (auto& s : q) {
                    (void)seq_now;
                    bool ok = false;
                    for (uint64_t v : valid) if (v == s.seq) ok = true;
                    if (!ok) { g_stale_dropped++; continue; }
                    int id = next_id++;
                    pending[id] = Pending{ "share", s.gpu, s.nonce_hex, s.seq, s.block, g_ns_order.load(), s.work };
                    std::string msg = "{\"id\":" + std::to_string(id) + ",\"method\":\"mining.submit\",\"params\":[" + json::quote(s.job_id) + "," +
                                      json::quote(s.nonce_hex) + "," + json::quote(s.work_domain_id) + "]}";
                    if (!conn.send_line(msg)) { LOG("pool: send failed: %s", conn.error().c_str()); lost = true; break; }
                }
            } else if (std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - t_conn).count() > 60) {
                LOG("pool: no authorization within 60 s");
                lost = true;
            }
            if (lost) break;
            // hashrate report every 60 s
            auto now = Clock::now();
            if (authorized && stats_ok && !stats_method.empty() && now - t_stats > std::chrono::seconds(60)) {
                uint64_t h = 0;
                for (auto& s : g_stats) h += s->hashes;
                double secs = std::chrono::duration<double>(now - t_stats_h).count();
                double rate = secs > 0 ? (double)(h - stats_hashes) / secs : 0;
                stats_hashes = h;
                t_stats_h = now;
                t_stats = now;
                int id = next_id++;
                pending[id] = Pending{ "stats", -1, "", 0, false, 0, 0.0 };
                char rs[64];
                snprintf(rs, sizeof rs, "%.0f", rate);
                std::string msg;
                if (stats_method == "miner.stats")
                    msg = "{\"id\":" + std::to_string(id) + ",\"method\":\"miner.stats\",\"params\":{\"nonce_rate\":" + rs + ",\"worker\":" + json::quote(rig) + "}}";
                else
                    msg = "{\"id\":" + std::to_string(id) + ",\"method\":" + json::quote(stats_method) + ",\"params\":[" + rs + "," + json::quote(rig) + "]}";
                conn.send_line(msg);
            }
            if (now - t_rx > std::chrono::seconds(300)) { LOG("pool: nothing received for 300 s, reconnecting"); lost = true; }
        }
        conn.close();
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            g_paused = true;
            g_job.reset();
            g_shares.clear();
            g_valid_seqs.clear();
        }
        if (!g_running) break;
        pool_idx++;
        LOG("pool: disconnected, retrying in %d s", backoff);
        for (int i = 0; i < backoff * 10 && g_running; i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        backoff = backoff * 2 > 60 ? 60 : backoff * 2;
    }
}

// ---------------------------------------------------------------------------
// tests and benchmark
// ---------------------------------------------------------------------------
static std::vector<int> selected_devices(const Config& cfg)
{
    int n = noid_gpu_device_count();
    std::vector<int> d;
    if (cfg.gpus.empty()) { for (int i = 0; i < n; i++) d.push_back(i); }
    else { for (int g : cfg.gpus) if (g >= 0 && g < n) d.push_back(g); }
    return d;
}

static bool load_vector(const std::string& path, int which, u128 fields[16], uint8_t digest[32])
{
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    char line[2048];
    int n = 0;
    bool ok = false;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || strlen(line) < 100) continue;
        if (n++ != which) continue;
        std::string l = trim(line);
        size_t sp = l.find(' ');
        std::string fs = l.substr(0, sp);
        for (int i = 0; i < 16; i++) {
            std::string h = fs.substr((size_t)i * 33, 32);
            uint64_t hi = strtoull(h.substr(0, 16).c_str(), nullptr, 16), lo = strtoull(h.substr(16).c_str(), nullptr, 16);
            fields[i] = noid::mk(lo, hi);
        }
        ok = noid::hex_to_bytes(l.substr(sp + 1), digest, 32);
        break;
    }
    fclose(f);
    return ok;
}

static int run_tests(const Config& cfg)
{
    std::string vec = cfg.exe_dir + "/vectors.txt";
    if (!file_exists(vec)) vec = cfg.exe_dir + "/../tests/vectors.txt";
    if (!file_exists(vec)) vec = "tests/vectors.txt";
    int fails = noid::self_test_vectors(vec.c_str(), 1);
    if (fails != 0) { LOG("TEST CPU reference: FAILED (%d)", fails); return 1; }
    LOG("TEST CPU reference: 64/64 official vectors OK");

    std::vector<int> devs = selected_devices(cfg);
    if (devs.empty()) { LOG("TEST: no CUDA device"); return 1; }
    int total_fail = 0;
    for (int dev : devs) {
        for (int mode = 0; mode < 3; mode++) {
          for (int lanes = 1; lanes <= (mode == NOID_MODE_TOWER ? 1 : 2); lanes++) {
            void* g = noid_gpu_create2(dev, mode, cfg.tpb, cfg.bps, 2, lanes);
            if (!g) { LOG("TEST GPU %d: create failed: %s", dev, noid_gpu_last_error(nullptr)); total_fail++; continue; }
            int fail = 0;
            for (int vi = 1; vi <= 3; vi++) {
                u128 fields[16];
                uint8_t want[32];
                if (!load_vector(vec, vi, fields, want)) { LOG("TEST: cannot load vector %d", vi); fail++; continue; }
                noid::Job cj;
                noid::prepare_job(cj, fields);
                u128 n0 = fields[10];
                uint32_t lo32 = (uint32_t)n0.lo;
                if (lo32 > 0xFFFF0000u) lo32 = 0xFFFF0000u;      // keep the test range inside the 32-bit window
                u128 base = noid::mk(n0.lo & 0xFFFFFFFF00000000ULL, n0.hi);
                uint8_t easy[32];
                memset(easy, 0, 32);
                NoidGpuJob gj;
                make_gpu_job(cj, base, easy, gj);
                noid_gpu_set_job(g, &gj);
                const int cnt = 4096;
                std::vector<uint32_t> st((size_t)cnt * 8);
                if (noid_gpu_digest_test(g, lo32, cnt, st.data()) != 0) { LOG("TEST GPU %d: %s", dev, noid_gpu_last_error(g)); fail++; continue; }
                int checked = 0, bad = 0;
                for (int i = 0; i < cnt; i += (i < 64 ? 1 : 97)) {
                    u128 n = noid::mk(base.lo | (uint64_t)(lo32 + (uint32_t)i), base.hi);
                    uint8_t d[32], gd[32];
                    noid::job_digest(cj, n, d);
                    u128 s0 = noid::mk(((uint64_t)st[8 * i + 1] << 32) | st[8 * i], ((uint64_t)st[8 * i + 3] << 32) | st[8 * i + 2]);
                    u128 s1 = noid::mk(((uint64_t)st[8 * i + 5] << 32) | st[8 * i + 4], ((uint64_t)st[8 * i + 7] << 32) | st[8 * i + 6]);
                    noid::u128_to_le(noid::flat_to_tower(s0), gd);
                    noid::u128_to_le(noid::flat_to_tower(s1), gd + 16);
                    if (memcmp(d, gd, 32) != 0) bad++;
                    if (noid::eq(n, n0) && memcmp(gd, want, 32) != 0) bad++;
                    checked++;
                }
                if (bad) LOG("TEST GPU %d mode %d vector %d: %d/%d digests WRONG", dev, mode, vi, bad, checked);
                fail += bad;
            }
            // candidate filter: target top64 = 2^56 -> about 1/256 of the nonces
            {
                u128 fields[16];
                uint8_t want[32];
                load_vector(vec, 5, fields, want);
                noid::Job cj;
                noid::prepare_job(cj, fields);
                uint8_t tgt[32];
                memset(tgt, 0xFF, 32);
                memset(tgt + 24, 0, 8);
                tgt[31] = 0x01;                 // top64 = 0x0100000000000000
                NoidGpuJob gj;
                u128 base = noid::mk(0x1234567800000000ULL, 0x0123456789abcdefULL);
                make_gpu_job(cj, base, tgt, gj);
                noid_gpu_set_job(g, &gj);
                std::vector<uint32_t> fnd(1024);
                int nf = 0;
                noid_gpu_set_nonces_per_thread(g, 1);
                uint32_t batch = noid_gpu_batch(g);
                noid_gpu_scan(g, 0, fnd.data(), 1024, &nf);
                double expect = batch / 256.0;
                int bad = 0, m = nf < 1024 ? nf : 1024;
                for (int i = 0; i < m && i < 200; i++) {
                    uint8_t d[32];
                    noid::job_digest(cj, noid::mk(base.lo | fnd[i], base.hi), d);
                    uint64_t top = 0;
                    for (int k = 31; k >= 24; k--) top = (top << 8) | d[k];
                    if (top > 0x0100000000000000ULL) bad++;
                }
                bool count_ok = nf > expect * 0.5 && nf < expect * 1.5;
                if (bad || !count_ok) LOG("TEST GPU %d mode %d candidates: %d found (expected ~%.0f), %d wrong", dev, mode, nf, expect, bad);
                fail += bad + (count_ok ? 0 : 1);
            }
            LOG("TEST GPU %d %s mode %s, %d lane(s): %s", dev, noid_gpu_name(g), mode_name(mode), lanes, fail ? "FAILED" : "OK");
            total_fail += fail;
            noid_gpu_destroy(g);
          }
        }
    }
    LOG("TEST RESULT: %s", total_fail ? "FAILED" : "ALL OK");
    return total_fail ? 1 : 0;
}

static int run_bench(const Config& cfg)
{
    std::vector<int> devs = selected_devices(cfg);
    if (devs.empty()) { LOG("BENCH: no CUDA device"); return 1; }
    auto j = std::make_shared<PoolJob>();
    std::mt19937_64 rng(42);
    for (int i = 0; i < 16; i++) j->fields[i] = noid::mk(rng(), rng());
    memset(j->share_target, 0, 32);           // impossible target: pure hashing
    memset(j->block_target, 0, 32);
    j->prefix_hex = "0123456789abcdef";
    j->nonce_bits = 64;
    j->expires = Clock::now() + std::chrono::hours(24);
    j->seq = 1;
    { std::lock_guard<std::mutex> lk(g_mtx); g_job = j; g_paused = false; g_job_seq = 1; g_valid_seqs.push_back(1); }
    for (size_t i = 0; i < devs.size(); i++) { g_stats.emplace_back(new GpuStat()); g_stats.back()->dev = devs[i]; }
    std::vector<std::thread> th;
    for (size_t i = 0; i < devs.size(); i++) th.emplace_back(gpu_worker, std::cref(cfg), (int)i, devs[i]);
    // warm-up 5 s, then measure
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::vector<uint64_t> h0;
    for (auto& s : g_stats) h0.push_back(s->hashes);
    auto t0 = Clock::now();
    for (int i = 0; i < cfg.bench_seconds && g_running; i++) std::this_thread::sleep_for(std::chrono::seconds(1));
    double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    double total = 0;
    for (size_t i = 0; i < g_stats.size(); i++) {
        double r = (g_stats[i]->hashes - h0[i]) / secs;
        total += r;
        LOG("BENCH GPU %d: %.3f MH/s  (npt %d, scan %d ms) %s", g_stats[i]->dev, r / 1e6, g_stats[i]->npt.load(), g_stats[i]->scan_ms.load(), g_stats[i]->name.c_str());
    }
    LOG("BENCH TOTAL: %.3f MH/s over %zu GPU(s), mode %s, %d threads/block, %d lane(s)", total / 1e6, g_stats.size(), mode_name(cfg.mode), cfg.tpb, cfg.lanes);
    g_running = false;
    for (auto& t : th) t.join();
    return 0;
}

// ---------------------------------------------------------------------------
// splices "--args-file FILE" (whitespace separated tokens) into the argument list
static std::vector<std::string> expand_args(int argc, char** argv)
{
    std::vector<std::string> out;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--args-file" && i + 1 < argc) {
            FILE* f = fopen(argv[++i], "r");
            if (!f) { fprintf(stderr, "cannot open args file %s\n", argv[i]); continue; }
            char tok[1024];
            while (fscanf(f, "%1023s", tok) == 1) out.push_back(tok);
            fclose(f);
            continue;
        }
        out.push_back(a);
    }
    return out;
}

int main(int argc, char** argv)
{
    Config cfg;
    cfg.exe_dir = exe_directory(argv[0]);
    std::vector<std::string> args = expand_args(argc, argv);
    std::string conf = cfg.exe_dir + "/noidminer.conf";
    if (!file_exists(conf)) conf = cfg.exe_dir + "/../noidminer.conf";
    for (size_t i = 0; i < args.size(); i++)
        if ((args[i] == "--config" || args[i] == "-c") && i + 1 < args.size()) conf = args[++i];
    bool have_conf = load_config_file(cfg, conf);
    std::vector<std::string> cli_pools;
    for (size_t i = 0; i < args.size(); i++) {
        const std::string& a = args[i];
        if (a == "--config" || a == "-c") { i++; continue; }
        if (a == "--test") { cfg.test = true; continue; }
        if (a == "--bench") {
            cfg.bench = true;
            if (i + 1 < args.size() && args[i + 1][0] != '-') cfg.bench_seconds = atoi(args[++i].c_str());
            continue;
        }
        if (a == "--help" || a == "-h") {
            printf("NoidMiner %s - ParanO(1)d (NOID) GPU miner\n"
                   "  noidminer [--config file] [--args-file file] [-o url] [-u addr.rig] [-p x] [--gpus 0,1|all]\n"
                   "            [--mode tower|table|kop] [--npt N] [--tpb 256|384|512] [--lanes 1|2] [--bps N] [--target-ms N] [--cafile f]\n"
                   "            [--ns-order auto|int|bytes] [--stats s] [--log file] [--test] [--bench [s]]\n", NOIDMINER_VERSION);
            return 0;
        }
        if (i + 1 < args.size()) {
            if (a == "-o" || a == "--pool" || a == "--url") { cli_pools.push_back(args[++i]); continue; }
            if (apply_option(cfg, a, args[i + 1])) { i++; continue; }
        }
        fprintf(stderr, "unknown argument %s (see --help)\n", a.c_str());
        return 2;
    }
    if (!cli_pools.empty()) cfg.pools = cli_pools;
    if (cfg.cafile.empty()) {
        const char* cands[] = { "C:\\msys64\\mingw64\\etc\\ssl\\certs\\ca-bundle.crt", "/etc/ssl/certs/ca-certificates.crt" };
        std::string local = cfg.exe_dir + "/ca-bundle.crt";
        if (file_exists(local)) cfg.cafile = local;
        else for (auto c : cands) if (file_exists(c)) { cfg.cafile = c; break; }
    }
    if (!cfg.logfile.empty()) g_logf = fopen(cfg.logfile.c_str(), "a");
    if (cfg.ns_order == "int") { g_ns_order = 0; g_ns_auto = false; }
    else if (cfg.ns_order == "bytes") { g_ns_order = 1; g_ns_auto = false; }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    LOG("NoidMiner %s (ParanO(1)d / NOID, Poseidon2b) - no dev fee%s", NOIDMINER_VERSION, have_conf ? "" : " - no config file");

    if (cfg.test) return run_tests(cfg);
    if (cfg.bench) return run_bench(cfg);

    if (cfg.pools.empty() || cfg.user.empty()) { LOG("missing pool url or user (address.rig), see noidminer.conf"); return 2; }
    std::vector<int> devs = selected_devices(cfg);
    if (devs.empty()) { LOG("no CUDA device selected"); return 1; }

    std::string gname;
    {
        void* g = noid_gpu_create(devs[0], cfg.mode, 256, cfg.bps, 1);
        if (g) { gname = noid_gpu_name(g); noid_gpu_destroy(g); }
        size_t p = gname.find(" (");
        if (p != std::string::npos) gname = gname.substr(0, p);
    }
    std::string gpu_desc = gname + " x " + std::to_string(devs.size());
    for (size_t i = 0; i < devs.size(); i++) { g_stats.emplace_back(new GpuStat()); g_stats.back()->dev = devs[i]; }
    std::vector<std::thread> th;
    for (size_t i = 0; i < devs.size(); i++) th.emplace_back(gpu_worker, std::cref(cfg), (int)i, devs[i]);
    std::thread net(pool_worker, std::cref(cfg), gpu_desc);

    std::vector<uint64_t> last(g_stats.size(), 0);
    auto t_last = Clock::now();
    auto t_start = Clock::now();
    while (g_running) {
        for (int i = 0; i < cfg.stats * 10 && g_running; i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!g_running) break;
        double secs = std::chrono::duration<double>(Clock::now() - t_last).count();
        t_last = Clock::now();
        double total = 0;
        std::string per;
        for (size_t i = 0; i < g_stats.size(); i++) {
            uint64_t h = g_stats[i]->hashes;
            double r = (h - last[i]) / secs;
            last[i] = h;
            total += r;
            char b[64];
            snprintf(b, sizeof b, "%sGPU%d %.2f", i ? " | " : "", g_stats[i]->dev, r / 1e6);
            per += b;
        }
        bool paused;
        double work;
        { std::lock_guard<std::mutex> lk(g_mtx); paused = g_paused; work = g_accepted_work; }
        double up = std::chrono::duration<double>(Clock::now() - t_start).count();
        LOG("%.2f MH/s total (%s) | shares %llu/%llu, stale dropped %llu | pool-side %.2f MH/s | blocks %llu%s", total / 1e6, per.c_str(),
            (unsigned long long)g_accepted.load(), (unsigned long long)(g_accepted.load() + g_rejected.load()),
            (unsigned long long)g_stale_dropped.load(), up > 0 ? work / up / 1e6 : 0.0,
            (unsigned long long)g_blocks.load(), paused ? " | waiting for work" : "");
    }
    LOG("stopping...");
    for (auto& t : th) t.join();
    net.join();
    return 0;
}
