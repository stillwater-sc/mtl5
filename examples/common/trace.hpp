// trace.hpp -- opt-in JSON traces for the pedagogical examples (#933)
//
// The docs site animates what an example computes. It does not re-solve the
// problem in JavaScript: the example writes its own numbers to a JSON file, the
// file is committed under docs-site/src/data/traces/, and the page replays it.
//
// A program opts in by constructing a trace from its command line:
//
//     example_trace::trace tr(argc, argv);       // looks for --trace <file>
//     if (tr) tr["iterations"] = 42;             // only when asked for
//     ...                                        // written at the end of main
//
// Without --trace nothing is recorded and the program's output is unchanged.
// This header belongs to the examples, not to the library: nothing under
// include/mtl/ depends on it.

#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace example_trace {

/// A JSON value: null, bool, number, string, array or object. Objects keep
/// their keys in insertion order, so a trace reads in the order it was built.
///
/// Arrays and objects are deques, not vectors: a reference returned by [] or
/// push_back stays valid while more members or elements are added, so
/// `auto& a = v["a"]; v["b"] = ...; a = ...;` is safe. With a vector the second
/// insertion could reallocate and leave `a` dangling.
class value {
public:
    using array_t  = std::deque<value>;
    using object_t = std::deque<std::pair<std::string, value>>;

    value() = default;
    value(std::nullptr_t) {}
    value(bool b) : v_(b) {}
    value(double d) : v_(number{d, 17}) {}
    value(int i) : v_(number{static_cast<double>(i), 17}) {}
    value(long i) : v_(number{static_cast<double>(i), 17}) {}
    value(long long i) : v_(number{static_cast<double>(i), 17}) {}
    value(unsigned i) : v_(number{static_cast<double>(i), 17}) {}
    value(unsigned long i) : v_(number{static_cast<double>(i), 17}) {}
    value(unsigned long long i) : v_(number{static_cast<double>(i), 17}) {}
    value(const char* s) : v_(std::string(s)) {}
    value(std::string s) : v_(std::move(s)) {}
    value(std::string_view s) : v_(std::string(s)) {}
    value(array_t a) : v_(std::make_shared<array_t>(std::move(a))) {}

    /// A number written with `digits` significant digits. Plotting data needs
    /// far fewer than the 17 that round-trip a double, and a trace inlined in a
    /// page should stay small.
    static value rounded(double d, int digits) {
        value r;
        r.v_ = number{d, digits};
        return r;
    }

    /// An array of numbers, each with `digits` significant digits.
    template <typename Range>
    static value numbers(const Range& r, int digits = 17) {
        array_t a;
        for (const auto& x : r) a.push_back(rounded(static_cast<double>(x), digits));
        return value(std::move(a));
    }

    static value object() {
        value r;
        r.v_ = std::make_shared<object_t>();
        return r;
    }

    static value array() { return value(array_t{}); }

    /// Object member access; turns a null value into an empty object first.
    value& operator[](const std::string& key) {
        if (std::holds_alternative<std::monostate>(v_)) v_ = std::make_shared<object_t>();
        auto* obj = std::get_if<std::shared_ptr<object_t>>(&v_);
        if (!obj) throw std::logic_error("example_trace: [] on a non-object");
        for (auto& [k, v] : **obj)
            if (k == key) return v;
        (*obj)->emplace_back(key, value{});
        return (*obj)->back().second;
    }

    /// Append to an array; turns a null value into an empty array first.
    value& push_back(value x) {
        if (std::holds_alternative<std::monostate>(v_)) v_ = std::make_shared<array_t>();
        auto* arr = std::get_if<std::shared_ptr<array_t>>(&v_);
        if (!arr) throw std::logic_error("example_trace: push_back on a non-array");
        (*arr)->push_back(std::move(x));
        return (*arr)->back();
    }

    void write(std::ostream& os) const { write(os, 0); }

private:
    struct number {
        double d;
        int digits;
    };

    static void write_string(std::ostream& os, const std::string& s) {
        os << '"';
        for (char c : s) {
            switch (c) {
                case '"':  os << "\\\""; break;
                case '\\': os << "\\\\"; break;
                case '\n': os << "\\n"; break;
                case '\t': os << "\\t"; break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned>(c));
                        os << buf;
                    } else {
                        os << c;
                    }
            }
        }
        os << '"';
    }

    static void write_number(std::ostream& os, const number& n) {
        // JSON has no inf or NaN: they are written as null, and a reader that
        // needs to tell them apart records them separately.
        if (!std::isfinite(n.d)) { os << "null"; return; }
        char buf[40];
        std::snprintf(buf, sizeof buf, "%.*g", n.digits, n.d);
        os << buf;
    }

    // Arrays of scalars go on one line; anything nested is indented.
    void write(std::ostream& os, int indent) const {
        const std::string pad(static_cast<std::size_t>(indent), ' ');
        const std::string pad2(static_cast<std::size_t>(indent) + 2, ' ');
        if (std::holds_alternative<std::monostate>(v_)) {
            os << "null";
        } else if (auto* b = std::get_if<bool>(&v_)) {
            os << (*b ? "true" : "false");
        } else if (auto* n = std::get_if<number>(&v_)) {
            write_number(os, *n);
        } else if (auto* s = std::get_if<std::string>(&v_)) {
            write_string(os, *s);
        } else if (auto* a = std::get_if<std::shared_ptr<array_t>>(&v_)) {
            bool flat = true;
            for (const auto& x : **a)
                if (x.is_container()) { flat = false; break; }
            os << '[';
            for (std::size_t i = 0; i < (*a)->size(); ++i) {
                if (i) os << ',';
                if (!flat) os << '\n' << pad2;
                (**a)[i].write(os, indent + 2);
            }
            if (!flat && !(*a)->empty()) os << '\n' << pad;
            os << ']';
        } else if (auto* o = std::get_if<std::shared_ptr<object_t>>(&v_)) {
            os << '{';
            for (std::size_t i = 0; i < (*o)->size(); ++i) {
                if (i) os << ',';
                os << '\n' << pad2;
                write_string(os, (**o)[i].first);
                os << ": ";
                (**o)[i].second.write(os, indent + 2);
            }
            if (!(*o)->empty()) os << '\n' << pad;
            os << '}';
        }
    }

    bool is_container() const {
        return std::holds_alternative<std::shared_ptr<array_t>>(v_)
            || std::holds_alternative<std::shared_ptr<object_t>>(v_);
    }

    std::variant<std::monostate, bool, number, std::string,
                 std::shared_ptr<array_t>, std::shared_ptr<object_t>> v_;
};

/// The trace of one example run. Reads `--trace <file>` from the command
/// line; when it is absent the trace is disabled and records nothing. The
/// file is written when the trace is destroyed, at the end of main.
class trace {
public:
    trace(int argc, char** argv, std::string example_name) {
        for (int i = 1; i < argc; ++i) {
            std::string_view a = argv[i];
            if (a == "--trace") {
                if (i + 1 >= argc) throw std::invalid_argument("--trace needs a file name");
                path_ = argv[++i];
            }
        }
        if (enabled()) {
            root_["example"] = std::move(example_name);
            root_["generator"] = "examples/common/trace.hpp; refresh with tools/regen_example_traces.sh";
        }
    }

    trace(const trace&) = delete;
    trace& operator=(const trace&) = delete;

    /// Write the trace (when enabled) and return main's exit status:
    /// EXIT_FAILURE if the file could not be opened or written, so a script
    /// that regenerates traces does not keep a stale one. Call it as
    /// `return trace.close();` at the end of main.
    int close() {
        if (!enabled() || closed_) return EXIT_SUCCESS;
        closed_ = true;
        std::ofstream out(path_);
        if (out) {
            root_.write(out);
            out << '\n';
            out.close();
        }
        if (!out) {
            std::cerr << "trace: cannot write " << path_ << '\n';
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    /// A trace that main did not close (an early return) is still written.
    ~trace() { close(); }

    bool enabled() const { return !path_.empty(); }
    explicit operator bool() const { return enabled(); }

    value& operator[](const std::string& key) { return root_[key]; }

private:
    std::string path_;
    value root_ = value::object();
    bool closed_ = false;
};

} // namespace example_trace
