// tls_conn.h - newline-delimited TLS (or plain TCP) connection, OpenSSL based.
#pragma once
#include <string>
#include <vector>
#include <stdint.h>

class TlsConn {
public:
    TlsConn();
    ~TlsConn();
    // url: stratum+ssl://host:port  stratum+tcp://host:port  host:port (TLS)
    bool connect(const std::string& url, const std::string& cafile, int timeout_ms);
    void close();
    bool connected() const { return sock_ != -1; }
    bool send_line(const std::string& line);       // appends '\n'
    // waits up to timeout_ms for data; appends complete lines to out.
    // returns false when the connection is lost.
    bool poll_lines(std::vector<std::string>& out, int timeout_ms);
    const std::string& error() const { return err_; }
    const std::string& host() const { return host_; }
    bool verified() const { return verified_; }

    static bool parse_url(const std::string& url, std::string& host, int& port, bool& tls);

private:
    long long sock_;
    void* ctx_;
    void* ssl_;
    bool tls_;
    bool verified_;
    std::string host_;
    std::string buf_;
    std::string err_;
    bool raw_send(const char* p, size_t n);
};

void net_global_init();
