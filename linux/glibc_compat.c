/*
 * glibc_compat.c - lets a NoidMiner binary built on a recent distribution
 * (glibc 2.38+) run on older ones (glibc 2.35+, e.g. Ubuntu 22.04, Debian 12).
 *
 * Recent headers redirect strtol/strtoul/strtoll/strtoull/fscanf to their
 * C23 variants (__isoc23_*, glibc 2.38) and OpenSSL uses arc4random (glibc
 * 2.36). The link wraps those symbols (-Wl,--wrap=...) to the functions below,
 * which call the classic versions. This file is compiled as plain C11 without
 * _GNU_SOURCE, so the calls here are NOT redirected again.
 * The only behaviour difference: no "0b" binary prefix parsing (unused).
 */
#define _DEFAULT_SOURCE 1
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>

long __wrap___isoc23_strtol(const char* s, char** e, int b) { return strtol(s, e, b); }
unsigned long __wrap___isoc23_strtoul(const char* s, char** e, int b) { return strtoul(s, e, b); }
long long __wrap___isoc23_strtoll(const char* s, char** e, int b) { return strtoll(s, e, b); }
unsigned long long __wrap___isoc23_strtoull(const char* s, char** e, int b) { return strtoull(s, e, b); }

int __wrap___isoc23_fscanf(FILE* f, const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int r = vfscanf(f, fmt, ap);
    va_end(ap);
    return r;
}

int __wrap___isoc23_sscanf(const char* s, const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int r = vsscanf(s, fmt, ap);
    va_end(ap);
    return r;
}

/* arc4random from the kernel CSPRNG (getrandom syscall, Linux 3.17+) */
static void fill_random(void* buf, size_t n)
{
    unsigned char* p = (unsigned char*)buf;
    while (n > 0) {
        long r = syscall(SYS_getrandom, p, n, 0);
        if (r > 0) { p += r; n -= (size_t)r; continue; }
        FILE* f = fopen("/dev/urandom", "rb");
        if (f) { size_t g = fread(p, 1, n, f); fclose(f); p += g; n -= g; if (g) continue; }
        abort();
    }
}

uint32_t __wrap_arc4random(void) { uint32_t v; fill_random(&v, sizeof v); return v; }
void __wrap_arc4random_buf(void* buf, size_t n) { fill_random(buf, n); }
uint32_t __wrap_arc4random_uniform(uint32_t bound)
{
    if (bound < 2) return 0;
    uint32_t min = (uint32_t)(-bound) % bound, r;
    do { r = __wrap_arc4random(); } while (r < min);
    return r % bound;
}
