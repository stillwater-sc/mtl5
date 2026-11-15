#pragma once
// MTL5 -- Strict token parsing shared by the text readers (Matrix Market,
// element lists).
//
// Each helper parses one whole token and reports failure instead of stopping
// silently: a token with trailing characters, a negative index, an index that
// overflows, or a value the type cannot read is an error (#843). Floating-point
// values go through std::from_chars where the standard library has it for
// floating point (locale-independent), else strtod/strtof/strtold (which follow
// the C locale's decimal point), with the same grammar: one optional sign
// ("+-5" is an error), no hexadecimal floats. Both read the "nan", "inf",
// "infinity" and "-inf" that the writers emit; iostreams do not. Other value
// types are read with operator>> and must consume the whole token.
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace mtl::io::detail {

inline bool is_space(char ch) noexcept {
    return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\v' || ch == '\f';
}

/// True if the line holds nothing but whitespace.
inline bool is_blank(std::string_view line) noexcept {
    for (char ch : line)
        if (!is_space(ch)) return false;
    return true;
}

/// Splits a line into whitespace-separated tokens, one at a time.
class token_cursor {
public:
    explicit token_cursor(std::string_view line) noexcept : rest_(line) {}

    /// The next token, or false at the end of the line.
    bool next(std::string_view& tok) noexcept {
        std::size_t i = 0;
        while (i < rest_.size() && is_space(rest_[i])) ++i;
        if (i == rest_.size()) { rest_ = {}; return false; }
        std::size_t j = i;
        while (j < rest_.size() && !is_space(rest_[j])) ++j;
        tok = rest_.substr(i, j - i);
        rest_ = rest_.substr(j);
        return true;
    }

    /// True if no token is left.
    bool done() const noexcept { return is_blank(rest_); }

private:
    std::string_view rest_;
};

/// A non-negative decimal integer, the whole token. Rejects a sign, other
/// characters, and values above `max`.
inline bool parse_index(std::string_view tok, std::size_t& out,
                        std::size_t max = std::numeric_limits<std::size_t>::max()) noexcept {
    if (tok.empty() || tok.front() == '-' || tok.front() == '+') return false;
    unsigned long long v = 0;
    const auto [ptr, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), v);
    if (ec != std::errc{} || ptr != tok.data() + tok.size()) return false;
    if (v > max) return false;
    out = static_cast<std::size_t>(v);
    return true;
}

/// The strtof/strtod/strtold reading of [first, last), a token without a
/// leading '+': parse_value's path where std::from_chars is missing, or where
/// it reports a value out of range. strto* also reads hexadecimal floats
/// ("0x1p3"), which from_chars in its general format does not, so they are
/// refused here: the grammar is the same either way (#855). strto* would also
/// skip leading whitespace, but a token has none.
template <typename T>
bool parse_float_strto(const char* first, const char* last, T& out) {
    static_assert(std::is_floating_point_v<T>);
    if (first == last) return false;
    const char* p = (*first == '-') ? first + 1 : first;
    if (last - p >= 2 && p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) return false;
    const std::string s(first, last);  // strto* needs a terminated string
    char* end = nullptr;
    T w{};
    if constexpr (std::is_same_v<T, float>)       w = std::strtof(s.c_str(), &end);
    else if constexpr (std::is_same_v<T, double>) w = std::strtod(s.c_str(), &end);
    else                                          w = std::strtold(s.c_str(), &end);
    if (end != s.c_str() + s.size()) return false;
    out = w;
    return true;
}

/// One value, the whole token.
template <typename T>
bool parse_value(std::string_view tok, T& out) {
    if (tok.empty()) return false;
    if constexpr (std::is_floating_point_v<T>) {
        const char* first = tok.data();
        const char* last = tok.data() + tok.size();
        if (*first == '+') {               // from_chars takes '-' but not '+'
            ++first;
            if (first == last || *first == '-' || *first == '+') return false;   // "+-5" (#855)
        }
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
        // Locale-independent, and reads inf, infinity and nan(...) as well.
        T v{};
        const auto [ptr, ec] = std::from_chars(first, last, v);
        if (ec == std::errc{}) {
            if (ptr != last) return false;
            out = v;
            return true;
        }
        if (ec != std::errc::result_out_of_range) return false;
        // Out of range (an overflow or a subnormal underflow): strto* below
        // gives the rounded result, +-inf or a subnormal or 0.
#endif
        return parse_float_strto(first, last, out);
    } else if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        const char* first = tok.data();
        const char* last = tok.data() + tok.size();
        if (*first == '+') {               // from_chars takes '-' but not '+'
            ++first;
            if (first == last || *first == '-' || *first == '+') return false;   // "+-5" (#855)
        }
        T v{};
        const auto [ptr, ec] = std::from_chars(first, last, v);
        if (ec != std::errc{} || ptr != last) return false;
        out = v;
        return true;
    } else {
        std::istringstream ss{std::string(tok)};
        T v{};
        if (!(ss >> v)) return false;
        ss >> std::ws;
        if (!ss.eof()) return false;
        out = v;
        return true;
    }
}

} // namespace mtl::io::detail
