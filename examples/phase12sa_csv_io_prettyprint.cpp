// phase12sa_csv_io_prettyprint.cpp — CSV I/O & Pretty-Print Workflow
//
// This example demonstrates practical data exchange and visualization:
//
//   1. Create a dense matrix and write it to CSV
//   2. Read it back and verify roundtrip fidelity
//   3. Create a sparse matrix and write/read triplet format
//   4. Pretty-print with configurable precision
//   5. Print sparse matrices in triplet format for debugging
//   6. Export matrices in MATLAB-pasteable format
//
// CSV and triplet I/O are the simplest interchange formats — no headers,
// no metadata, just values. They work with Excel, NumPy (numpy.loadtxt),
// MATLAB (readmatrix), R (read.csv), and any text editor.
//
// The pretty-print functions help with debugging and reporting:
// - print() gives precision control without polluting the stream state
// - print_sparse() shows only nonzeros — essential for large sparse matrices
// - print_matlab() produces copy-paste-ready output for MATLAB/Octave

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>

using namespace mtl;

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== CSV I/O & Pretty-Print Workflow ===\n\n";

    // ── 1. Dense CSV roundtrip ───────────────────────────────────────────
    // Create a 3x4 data matrix (e.g., sensor readings)
    mat::dense2D<double> sensors(3, 4);
    sensors(0, 0) = 23.456789; sensors(0, 1) = 45.123456;
    sensors(0, 2) = 67.891234; sensors(0, 3) = 12.345678;
    sensors(1, 0) = 34.567891; sensors(1, 1) = 56.789012;
    sensors(1, 2) = 78.901234; sensors(1, 3) = 90.123456;
    sensors(2, 0) = 11.222333; sensors(2, 1) = 44.555666;
    sensors(2, 2) = 77.888999; sensors(2, 3) = 99.000111;

    std::cout << "── 1. Original sensor data (3 sensors x 4 readings) ──\n";
    print(std::cout, sensors, 4);

    // Write to CSV file
    auto csv_path = std::filesystem::temp_directory_path() / "mtl5_sensors.csv";
    io::write_dense(csv_path.string(), sensors);
    std::cout << "\nWritten to: " << csv_path << "\n";

    // Read it back
    auto loaded = io::read_dense(csv_path.string());
    std::cout << "\n── 2. Loaded from CSV ──\n";
    print(std::cout, loaded, 4);

    // Verify roundtrip
    double max_err = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            max_err = std::max(max_err, std::abs(sensors(i, j) - loaded(i, j)));
    std::cout << "\nRoundtrip max error: " << max_err << "\n";

    // ── 2. Whitespace-delimited format ───────────────────────────────────
    auto ws_path = std::filesystem::temp_directory_path() / "mtl5_sensors.txt";
    io::write_dense(ws_path.string(), sensors, ' ');
    auto loaded_ws = io::read_dense(ws_path.string(), ' ');
    std::cout << "\n── 3. Whitespace-delimited roundtrip ──\n";
    print(std::cout, loaded_ws, 4);

    // ── 3. Sparse triplet I/O ────────────────────────────────────────────
    // A sparse 5x5 matrix (e.g., adjacency or stiffness)
    mat::compressed2D<double> S(5, 5);
    {
        mat::inserter<mat::compressed2D<double>> ins(S);
        ins[0][0] << 4.0;  ins[0][1] << -1.0;
        ins[1][0] << -1.0; ins[1][1] << 4.0;  ins[1][2] << -1.0;
        ins[2][2] << 4.0;  ins[2][3] << -1.0;
        ins[3][2] << -1.0; ins[3][3] << 4.0;  ins[3][4] << -1.0;
        ins[4][3] << -1.0; ins[4][4] << 4.0;
    }

    std::cout << "\n── 4. Sparse matrix (triplet view) ──\n";
    print_sparse(std::cout, S);

    auto tri_path = std::filesystem::temp_directory_path() / "mtl5_sparse.tri";
    io::write_sparse(tri_path.string(), S);
    std::cout << "\nWritten to: " << tri_path << "\n";

    auto S_loaded = io::read_sparse(tri_path.string(), 5, 5);
    std::cout << "\n── 5. Re-loaded sparse (triplet view) ──\n";
    print_sparse(std::cout, S_loaded);
    std::cout << "nnz original: " << S.nnz() << ", loaded: " << S_loaded.nnz() << "\n";

    // ── 4. Pretty-print with different precisions ────────────────────────
    std::cout << "\n── 6. Precision control ──\n";
    mat::dense2D<double> pi_mat(2, 2);
    pi_mat(0, 0) = 3.14159265358979;
    pi_mat(0, 1) = 2.71828182845905;
    pi_mat(1, 0) = 1.41421356237310;
    pi_mat(1, 1) = 1.73205080756888;

    std::cout << "Precision 3:\n";
    print(std::cout, pi_mat, 3);
    std::cout << "Precision 8:\n";
    print(std::cout, pi_mat, 8);
    std::cout << "Precision 15:\n";
    print(std::cout, pi_mat, 15);

    // ── 5. MATLAB-format output ──────────────────────────────────────────
    std::cout << "\n── 7. MATLAB-format output ──\n";
    std::cout << "Copy-paste these into MATLAB or Octave:\n\n";
    print_matlab(std::cout, sensors, "sensors", 4);
    std::cout << "\n";
    print_matlab(std::cout, pi_mat, "constants", 10);

    // ── 6. Vector pretty-print ───────────────────────────────────────────
    std::cout << "\n── 8. Vector pretty-print ──\n";
    vec::dense_vector<double> v({1.111111, 2.222222, 3.333333, 4.444444});
    std::cout << "Precision 2: ";
    print(std::cout, v, 2);
    std::cout << "\nPrecision 6: ";
    print(std::cout, v, 6);
    std::cout << "\n";

    // Cleanup temp files
    std::remove(csv_path.string().c_str());
    std::remove(ws_path.string().c_str());
    std::remove(tri_path.string().c_str());

    std::cout << "\nAll I/O and pretty-print operations completed successfully.\n";
    return 0;
}
