// The element-access assertions of the block and MX types (#826): an index
// past the end must stop at the assertion in a debug build. One call, chosen
// by argv[1]; ctest passes it only when the output names the assertion.
// NDEBUG is undefined first, as in mult_alias.cpp, and SIGABRT exits normally.
#undef NDEBUG
#include <cassert>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mtl/mat/block_matrix_view.hpp>
#include <mtl/mat/mx_schemes.hpp>

extern "C" void mtl5_death_on_abort(int) { std::_Exit(0); }

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::signal(SIGABRT, mtl5_death_on_abort);
    std::vector<std::byte> bytes(2 * 33, std::byte{0x7F});
    const mtl::mat::block_matrix_view<mtl::mx::mxint8> A(2, 32, bytes);

    volatile float sink = 0;
    if (std::strcmp(argv[1], "block_row") == 0) {
        sink = static_cast<float>(A.row(2).size());
    } else if (std::strcmp(argv[1], "decode_element") == 0) {
        sink = mtl::mx::mxint8::decode_element(bytes.data(), 32);
    } else {
        return 2;
    }
    static_cast<void>(sink);
    return 1;   // the assertion did not fire
}
