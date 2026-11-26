// test_trace.cpp -- self-test of the examples' trace writer (trace.hpp, #933).
// Registered with CTest; exits non-zero on the first failed check.
//
//   1. A reference into an object or array survives later insertions. With
//      vector storage, `auto& a = v["a"]; v["b"] = ...; a = ...;` wrote into
//      freed memory once the second insertion reallocated.
//   2. close() reports a trace that cannot be written as EXIT_FAILURE, so
//      tools/regen_example_traces.sh stops instead of keeping a stale trace.
//   3. A trace that is written round-trips its numbers and escapes strings.

#include "trace.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAILED: " << what << '\n';
        ++failures;
    }
}

std::string dump(const example_trace::value& v) {
    std::ostringstream os;
    v.write(os);
    return os.str();
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    using example_trace::value;

    // 1. References stay valid while the containers grow.
    {
        auto h = value::object();
        auto& first = h["first"];
        for (int k = 0; k < 100; ++k) h["m" + std::to_string(k)] = k;
        first = value::array();
        auto& element = first.push_back(1);
        for (int k = 0; k < 100; ++k) first.push_back(k);
        element = 42;
        const std::string s = dump(h);
        check(s.find("\"first\": [42,0,1,2") != std::string::npos, "member and element references survive growth");
    }

    // 2. An unwritable path makes close() fail; a writable one succeeds.
    const fs::path dir = fs::temp_directory_path() / "mtl5_test_trace";
    fs::create_directories(dir);
    {
        const std::string bad = (dir / "no_such_dir" / "t.json").string();
        std::vector<std::string> args = {"prog", "--trace", bad};
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(a.data());
        example_trace::trace tr(static_cast<int>(argv.size()), argv.data(), "test");
        tr["x"] = 1;
        check(tr.close() == EXIT_FAILURE, "close() reports a trace it could not write");
    }
    const fs::path good = dir / "t.json";
    {
        const std::string path = good.string();
        std::vector<std::string> args = {"prog", "--trace", path};
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(a.data());
        example_trace::trace tr(static_cast<int>(argv.size()), argv.data(), "quote \" and \\ slash");
        tr["pi"] = 3.141592653589793;
        tr["short"] = value::rounded(2.0 / 3.0, 4);
        tr["inf"] = std::numeric_limits<double>::infinity();
        tr["list"] = value::numbers(std::vector<int>{1, 2, 3});
        check(tr.close() == EXIT_SUCCESS, "close() succeeds on a writable path");
        check(tr.close() == EXIT_SUCCESS, "a second close() is a no-op");
    }
    {
        std::ifstream in(good);
        std::stringstream ss;
        ss << in.rdbuf();
        const std::string s = ss.str();
        check(s.find("\"example\": \"quote \\\" and \\\\ slash\"") != std::string::npos, "strings are escaped");
        check(s.find("\"pi\": 3.1415926535897931") != std::string::npos, "17 significant digits by default");
        check(s.find("\"short\": 0.6667") != std::string::npos, "rounded() keeps the digits asked for");
        check(s.find("\"inf\": null") != std::string::npos, "non-finite numbers are written as null");
        check(s.find("\"list\": [1,2,3]") != std::string::npos, "flat arrays on one line");
    }
    {
        // Disabled: no --trace, nothing written, close() succeeds.
        char prog[] = "prog";
        char* argv[] = {prog};
        example_trace::trace tr(1, argv, "off");
        check(!tr, "a trace without --trace is disabled");
        check(tr.close() == EXIT_SUCCESS, "a disabled trace closes successfully");
    }
    fs::remove_all(dir);

    if (failures == 0) std::cout << "trace.hpp: all checks passed\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
