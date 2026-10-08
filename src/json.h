// json.h - minimal JSON value + parser + writer helpers for the stratum client.
// Numbers keep their raw text (no precision loss on 64-bit values).
#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

namespace json {

struct Value {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    std::string s;                         // String text or Number raw text
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;

    bool is_null() const { return type == Null; }
    bool is_obj() const { return type == Object; }
    bool is_arr() const { return type == Array; }
    bool is_str() const { return type == String; }
    bool is_num() const { return type == Number; }
    bool is_bool() const { return type == Bool; }

    const Value* get(const std::string& k) const
    {
        if (type != Object) return nullptr;
        for (auto& kv : obj) if (kv.first == k) return &kv.second;
        return nullptr;
    }
    std::string str(const std::string& k, const std::string& def = "") const
    {
        const Value* v = get(k);
        if (!v) return def;
        if (v->type == String || v->type == Number) return v->s;
        if (v->type == Bool) return v->b ? "true" : "false";
        return def;
    }
    long long num(const std::string& k, long long def = 0) const
    {
        const Value* v = get(k);
        if (!v) return def;
        if (v->type == Number || v->type == String) return strtoll(v->s.c_str(), nullptr, 10);
        if (v->type == Bool) return v->b ? 1 : 0;
        return def;
    }
    bool boolean(const std::string& k, bool def = false) const
    {
        const Value* v = get(k);
        if (!v) return def;
        if (v->type == Bool) return v->b;
        if (v->type == Number) return strtoll(v->s.c_str(), nullptr, 10) != 0;
        if (v->type == String) return v->s == "true";
        return def;
    }
};

class Parser {
public:
    explicit Parser(const std::string& t) : t_(t), p_(0) {}
    bool parse(Value& out)
    {
        ws();
        if (!value(out, 0)) return false;
        ws();
        return p_ == t_.size();
    }
private:
    const std::string& t_;
    size_t p_;
    void ws() { while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\t' || t_[p_] == '\r' || t_[p_] == '\n')) p_++; }
    bool lit(const char* s)
    {
        size_t n = 0; while (s[n]) n++;
        if (t_.compare(p_, n, s) != 0) return false;
        p_ += n; return true;
    }
    static void utf8(std::string& o, unsigned cp)
    {
        if (cp < 0x80) o += (char)cp;
        else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
        else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 0x3F)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
    }
    bool hex4(unsigned& v)
    {
        if (p_ + 4 > t_.size()) return false;
        v = 0;
        for (int i = 0; i < 4; i++) {
            char c = t_[p_++]; v <<= 4;
            if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
            else return false;
        }
        return true;
    }
    bool string(std::string& o)
    {
        if (p_ >= t_.size() || t_[p_] != '"') return false;
        p_++;
        while (p_ < t_.size()) {
            char c = t_[p_++];
            if (c == '"') return true;
            if (c != '\\') { o += c; continue; }
            if (p_ >= t_.size()) return false;
            char e = t_[p_++];
            switch (e) {
            case '"': o += '"'; break;
            case '\\': o += '\\'; break;
            case '/': o += '/'; break;
            case 'b': o += '\b'; break;
            case 'f': o += '\f'; break;
            case 'n': o += '\n'; break;
            case 'r': o += '\r'; break;
            case 't': o += '\t'; break;
            case 'u': {
                unsigned cp;
                if (!hex4(cp)) return false;
                if (cp >= 0xD800 && cp < 0xDC00 && p_ + 6 <= t_.size() && t_[p_] == '\\' && t_[p_ + 1] == 'u') {
                    p_ += 2; unsigned lo;
                    if (!hex4(lo)) return false;
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                }
                utf8(o, cp);
                break;
            }
            default: return false;
            }
        }
        return false;
    }
    bool value(Value& v, int depth)
    {
        if (depth > 64 || p_ >= t_.size()) return false;
        char c = t_[p_];
        if (c == '{') {
            v.type = Value::Object; p_++; ws();
            if (p_ < t_.size() && t_[p_] == '}') { p_++; return true; }
            while (true) {
                ws();
                std::string k;
                if (!string(k)) return false;
                ws();
                if (p_ >= t_.size() || t_[p_] != ':') return false;
                p_++; ws();
                Value x;
                if (!value(x, depth + 1)) return false;
                v.obj.emplace_back(k, std::move(x));
                ws();
                if (p_ < t_.size() && t_[p_] == ',') { p_++; continue; }
                if (p_ < t_.size() && t_[p_] == '}') { p_++; return true; }
                return false;
            }
        }
        if (c == '[') {
            v.type = Value::Array; p_++; ws();
            if (p_ < t_.size() && t_[p_] == ']') { p_++; return true; }
            while (true) {
                ws();
                Value x;
                if (!value(x, depth + 1)) return false;
                v.arr.push_back(std::move(x));
                ws();
                if (p_ < t_.size() && t_[p_] == ',') { p_++; continue; }
                if (p_ < t_.size() && t_[p_] == ']') { p_++; return true; }
                return false;
            }
        }
        if (c == '"') { v.type = Value::String; return string(v.s); }
        if (lit("true")) { v.type = Value::Bool; v.b = true; return true; }
        if (lit("false")) { v.type = Value::Bool; v.b = false; return true; }
        if (lit("null")) { v.type = Value::Null; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            size_t s = p_;
            p_++;
            while (p_ < t_.size() && ((t_[p_] >= '0' && t_[p_] <= '9') || t_[p_] == '.' || t_[p_] == 'e' || t_[p_] == 'E' || t_[p_] == '+' || t_[p_] == '-')) p_++;
            v.type = Value::Number; v.s = t_.substr(s, p_ - s);
            return true;
        }
        return false;
    }
};

inline bool parse(const std::string& text, Value& out) { Parser p(text); return p.parse(out); }

inline std::string quote(const std::string& s)
{
    std::string o = "\"";
    for (char c : s) {
        switch (c) {
        case '"': o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break;
        case '\r': o += "\\r"; break;
        case '\t': o += "\\t"; break;
        default:
            if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", (unsigned char)c); o += b; }
            else o += c;
        }
    }
    return o + "\"";
}

// compact serialization (for logging unknown messages)
inline std::string dump(const Value& v)
{
    switch (v.type) {
    case Value::Null: return "null";
    case Value::Bool: return v.b ? "true" : "false";
    case Value::Number: return v.s;
    case Value::String: return quote(v.s);
    case Value::Array: {
        std::string o = "[";
        for (size_t i = 0; i < v.arr.size(); i++) { if (i) o += ","; o += dump(v.arr[i]); }
        return o + "]";
    }
    case Value::Object: {
        std::string o = "{";
        for (size_t i = 0; i < v.obj.size(); i++) { if (i) o += ","; o += quote(v.obj[i].first) + ":" + dump(v.obj[i].second); }
        return o + "}";
    }
    }
    return "null";
}

} // namespace json
