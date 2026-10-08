// tls_conn.cpp - newline-delimited TLS connection (OpenSSL), Windows + POSIX.
#include "tls_conn.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET sock_t;
#define CLOSESOCK closesocket
#else
#include <sys/socket.h>
#include <sys/select.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
typedef int sock_t;
#define CLOSESOCK ::close
#endif

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/x509v3.h>

static const size_t MAX_LINE = 4u << 20;   // 4 MiB hard line limit (pool spec)

void net_global_init()
{
    static bool done = false;
    if (done) return;
    done = true;
#ifdef _WIN32
    WSADATA w;
    WSAStartup(MAKEWORD(2, 2), &w);
#endif
    SSL_library_init();
    SSL_load_error_strings();
}

TlsConn::TlsConn() : sock_(-1), ctx_(nullptr), ssl_(nullptr), tls_(true), verified_(false) {}
TlsConn::~TlsConn() { close(); }

bool TlsConn::parse_url(const std::string& url, std::string& host, int& port, bool& tls)
{
    std::string u = url;
    tls = true;
    size_t p = u.find("://");
    if (p != std::string::npos) {
        std::string scheme = u.substr(0, p);
        u = u.substr(p + 3);
        if (scheme.find("tcp") != std::string::npos && scheme.find("ssl") == std::string::npos && scheme.find("tls") == std::string::npos) tls = false;
    }
    size_t s = u.find('/');
    if (s != std::string::npos) u = u.substr(0, s);
    size_t c = u.rfind(':');
    if (c == std::string::npos) return false;
    host = u.substr(0, c);
    port = atoi(u.substr(c + 1).c_str());
    return !host.empty() && port > 0 && port < 65536;
}

static std::string ssl_err_string()
{
    unsigned long e = ERR_get_error();
    if (!e) return "unknown TLS error";
    char b[256];
    ERR_error_string_n(e, b, sizeof b);
    return b;
}

static bool wait_sock(sock_t s, bool for_write, int timeout_ms)
{
    fd_set fs;
    FD_ZERO(&fs);
    FD_SET(s, &fs);
    timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    int r = for_write ? select((int)s + 1, nullptr, &fs, nullptr, &tv) : select((int)s + 1, &fs, nullptr, nullptr, &tv);
    return r > 0;
}

bool TlsConn::connect(const std::string& url, const std::string& cafile, int timeout_ms)
{
    close();
    net_global_init();
    int port;
    if (!parse_url(url, host_, port, tls_)) { err_ = "bad pool url: " + url; return false; }

    addrinfo hints, *res = nullptr;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    char ps[16];
    snprintf(ps, sizeof ps, "%d", port);
    if (getaddrinfo(host_.c_str(), ps, &hints, &res) != 0 || !res) { err_ = "DNS lookup failed for " + host_; return false; }

    sock_t s = (sock_t)-1;
    for (addrinfo* a = res; a; a = a->ai_next) {
        s = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s == (sock_t)-1) continue;
        // non-blocking connect with timeout
#ifdef _WIN32
        u_long nb = 1; ioctlsocket(s, FIONBIO, &nb);
#else
        int fl = fcntl(s, F_GETFL, 0); fcntl(s, F_SETFL, fl | O_NONBLOCK);
#endif
        int r = ::connect(s, a->ai_addr, (int)a->ai_addrlen);
        bool ok = (r == 0);
        if (!ok) {
#ifdef _WIN32
            bool inprog = WSAGetLastError() == WSAEWOULDBLOCK;
#else
            bool inprog = errno == EINPROGRESS;
#endif
            if (inprog && wait_sock(s, true, timeout_ms)) {
                int so = 0;
                socklen_t sl = sizeof so;
                getsockopt(s, SOL_SOCKET, SO_ERROR, (char*)&so, &sl);
                ok = (so == 0);
            }
        }
#ifdef _WIN32
        nb = 0; ioctlsocket(s, FIONBIO, &nb);
#else
        fcntl(s, F_SETFL, fl);
#endif
        if (ok) break;
        CLOSESOCK(s);
        s = (sock_t)-1;
    }
    freeaddrinfo(res);
    if (s == (sock_t)-1) { err_ = "cannot connect to " + host_ + ":" + ps; return false; }
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof one);
    setsockopt(s, SOL_SOCKET, SO_KEEPALIVE, (const char*)&one, sizeof one);
    sock_ = (long long)s;

    if (!tls_) { buf_.clear(); return true; }

    SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) { err_ = "SSL_CTX_new: " + ssl_err_string(); close(); return false; }
    SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
    verified_ = false;
    bool have_ca = false;
    if (!cafile.empty()) {
        FILE* f = fopen(cafile.c_str(), "rb");
        if (f) { fclose(f); have_ca = SSL_CTX_load_verify_locations(ctx, cafile.c_str(), nullptr) == 1; }
    }
    SSL_CTX_set_verify(ctx, have_ca ? SSL_VERIFY_PEER : SSL_VERIFY_NONE, nullptr);
    ctx_ = ctx;

    SSL* ssl = SSL_new(ctx);
    ssl_ = ssl;
    SSL_set_tlsext_host_name(ssl, host_.c_str());
    if (have_ca) SSL_set1_host(ssl, host_.c_str());
    SSL_set_fd(ssl, (int)s);
    // handshake (blocking socket, bounded by receive timeout)
#ifdef _WIN32
    DWORD tmo = (DWORD)timeout_ms;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tmo, sizeof tmo);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tmo, sizeof tmo);
#else
    timeval tv; tv.tv_sec = timeout_ms / 1000; tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#endif
    if (SSL_connect(ssl) != 1) {
        long vr = SSL_get_verify_result(ssl);
        err_ = "TLS handshake with " + host_ + " failed: " + ssl_err_string();
        if (vr != X509_V_OK) err_ += std::string(" (certificate: ") + X509_verify_cert_error_string(vr) + ")";
        close();
        return false;
    }
    verified_ = have_ca && SSL_get_verify_result(ssl) == X509_V_OK;
    buf_.clear();
    return true;
}

void TlsConn::close()
{
    if (ssl_) { SSL_shutdown((SSL*)ssl_); SSL_free((SSL*)ssl_); ssl_ = nullptr; }
    if (ctx_) { SSL_CTX_free((SSL_CTX*)ctx_); ctx_ = nullptr; }
    if (sock_ != -1) { CLOSESOCK((sock_t)sock_); sock_ = -1; }
    buf_.clear();
}

bool TlsConn::raw_send(const char* p, size_t n)
{
    while (n > 0) {
        int w;
        if (tls_) {
            w = SSL_write((SSL*)ssl_, p, (int)n);
            if (w <= 0) { err_ = "TLS write failed"; return false; }
        } else {
            w = (int)::send((sock_t)sock_, p, (int)n, 0);
            if (w <= 0) { err_ = "socket write failed"; return false; }
        }
        p += w;
        n -= (size_t)w;
    }
    return true;
}

bool TlsConn::send_line(const std::string& line)
{
    if (sock_ == -1) return false;
    std::string l = line + "\n";
    if (!raw_send(l.data(), l.size())) { close(); return false; }
    return true;
}

bool TlsConn::poll_lines(std::vector<std::string>& out, int timeout_ms)
{
    if (sock_ == -1) return false;
    bool pending = tls_ && SSL_pending((SSL*)ssl_) > 0;
    if (!pending && !wait_sock((sock_t)sock_, false, timeout_ms)) return true;   // nothing yet
    char tmp[16384];
    int r;
    if (tls_) {
        r = SSL_read((SSL*)ssl_, tmp, sizeof tmp);
        if (r <= 0) {
            int e = SSL_get_error((SSL*)ssl_, r);
            if (e == SSL_ERROR_WANT_READ || e == SSL_ERROR_WANT_WRITE) return true;
            err_ = e == SSL_ERROR_ZERO_RETURN ? "connection closed by pool" : "TLS read failed";
            close();
            return false;
        }
    } else {
        r = (int)::recv((sock_t)sock_, tmp, sizeof tmp, 0);
        if (r <= 0) { err_ = "connection closed by pool"; close(); return false; }
    }
    buf_.append(tmp, (size_t)r);
    size_t p;
    while ((p = buf_.find('\n')) != std::string::npos) {
        std::string l = buf_.substr(0, p);
        if (!l.empty() && l.back() == '\r') l.pop_back();
        if (!l.empty()) out.push_back(l);
        buf_.erase(0, p + 1);
    }
    if (buf_.size() > MAX_LINE) { err_ = "line too long from pool"; close(); return false; }
    return true;
}
