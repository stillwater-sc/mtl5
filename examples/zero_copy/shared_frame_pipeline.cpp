// shared_frame_pipeline.cpp - Zero-copy processing of frames in a shared buffer
//
// The pattern: one stage of a pipeline (a camera driver, an ADC's DMA engine, a
// network receiver) fills frames in a region of memory it shares with the
// next stage. The next stage runs an algorithm over each frame IN PLACE, and
// hands the same memory on. No stage copies a frame.
//
// Laying an MTL5 matrix over a frame turns the frame into an operand:
//
//     mat::dense2D<float> F(H, W, frame_ptr);   // a conduit to the shared frame
//     F = gain * (F - dark);                    // dark-frame and gain correction,
//                                               // written back into the frame
//
// and an ndarray view gives strided access to parts of it -- here, tiles,
// whose rows are not contiguous, so a dense2D cannot describe them:
//
//     array::ndarray<float, 2> nd(frame_ptr, {H, W});
//     auto tile = slice(nd, range(r0, r0 + T), range(c0, c0 + T));
//     tile -= mean(tile);                       // background flattening, in place
//
// MTL5 does not synchronize anything. The application decides who may touch
// a frame when; here a mutex and a condition variable hand each of two frame
// slots back and forth between a producer thread and the processing thread.
//
// The program checks every processed frame against values computed
// independently, and checks that each frame was processed in the memory the
// producer wrote, never in a copy.

#include <mtl/mtl.hpp>
#include <mtl/array/ndarray.hpp>
#include <mtl/array/slice.hpp>

#include <array>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using namespace mtl;

namespace {

constexpr std::size_t H = 96, W = 128;   // frame size
constexpr std::size_t T = 32;            // tile size; H and W are multiples of it
constexpr std::size_t slots = 2;         // double buffering
constexpr int frames = 24;
constexpr float gain = 2.5f;

// The scene the sensor sees, a per-tile background that drifts from frame to
// frame, and the sensor's fixed dark offset.
double scene(int k, std::size_t r, std::size_t c) {
    return 50.0 + 40.0 * std::sin(0.1 * double(r) + 0.07 * double(c) + 0.3 * k);
}
double background(int k, std::size_t r, std::size_t c) {
    const std::size_t tile = (r / T) * (W / T) + c / T;
    return 10.0 + 3.0 * double(tile) + double(k);
}
float dark_offset(std::size_t r, std::size_t c) {
    return 100.0f + float((r * 7 + c * 13) % 17);
}

// -- The shared region ---------------------------------------------------------
// One allocation holds both frame slots. The state array says who owns each
// slot: the producer (empty) or the processing stage (filled).

enum class slot_state { empty, filled };

struct shared_region {
    std::vector<float> pixels = std::vector<float>(slots * H * W);
    std::array<slot_state, slots> state{slot_state::empty, slot_state::empty};
    std::array<int, slots> frame_id{-1, -1};
    std::mutex m;
    std::condition_variable cv;

    float* slot(std::size_t s) { return pixels.data() + s * H * W; }
};

// -- Stage 1: the producer -------------------------------------------------------
// Stands in for a driver: raw = dark + (scene + background) / gain. It writes
// a slot only while it owns it, and releases it by marking it filled.

void producer(shared_region& sh) {
    for (int k = 0; k < frames; ++k) {
        const std::size_t s = std::size_t(k) % slots;
        std::unique_lock lock(sh.m);
        sh.cv.wait(lock, [&] { return sh.state[s] == slot_state::empty; });
        lock.unlock();                       // the slot is ours: write it unlocked

        float* raw = sh.slot(s);
        for (std::size_t r = 0; r < H; ++r)
            for (std::size_t c = 0; c < W; ++c)
                raw[r * W + c] = dark_offset(r, c)
                               + float((scene(k, r, c) + background(k, r, c)) / gain);

        lock.lock();
        sh.frame_id[s] = k;
        sh.state[s] = slot_state::filled;    // hand the slot to the next stage
        lock.unlock();
        sh.cv.notify_all();
    }
}

// -- Stage 2: in-place processing -------------------------------------------------

struct process_report {
    int frames = 0;
    int copied = 0;        // frames whose processing did not stay in the shared slot
    double max_error = 0;  // against the independently computed result
};

/// Processes the frame in place. Returns whether every view stayed on the
/// slot: the check that nothing detached into a private copy.
bool process_in_place(float* frame, const mat::dense2D<float>& dark) {
    // (a) Dark-frame and gain correction over the whole frame. F is a
    //     row-major H x W matrix over the slot: the frame has no row padding,
    //     which is what dense2D's layout requires.
    mat::dense2D<float> F(H, W, frame);
    F = gain * (F - dark);                   // element-local, so safe in place
    bool in_place = F.data() == frame && !F.owns_memory();

    // (b) Background flattening per tile. A tile's rows are W apart in
    //     memory, so it needs a strided view: an ndarray slice.
    array::ndarray<float, 2> nd(frame, {H, W});
    for (std::size_t r0 = 0; r0 < H; r0 += T)
        for (std::size_t c0 = 0; c0 < W; c0 += T) {
            auto tile = array::slice(nd, array::range(r0, r0 + T), array::range(c0, c0 + T));
            double sum = 0.0;
            tile.for_each_element([&](float& v) { sum += v; });
            tile -= float(sum / double(T * T));
            in_place = in_place && tile.is_view();
        }
    return in_place && nd.data() == frame;
}

void processor(shared_region& sh, process_report& rep) {
    // The calibration frame is the processing stage's own, owning matrix.
    mat::dense2D<float> dark(H, W);
    for (std::size_t r = 0; r < H; ++r)
        for (std::size_t c = 0; c < W; ++c) dark(r, c) = dark_offset(r, c);

    for (int k = 0; k < frames; ++k) {
        const std::size_t s = std::size_t(k) % slots;
        std::unique_lock lock(sh.m);
        sh.cv.wait(lock, [&] { return sh.state[s] == slot_state::filled; });
        const int id = sh.frame_id[s];
        lock.unlock();                       // the slot is ours until we release it

        float* frame = sh.slot(s);
        if (!process_in_place(frame, dark)) ++rep.copied;

        // Stage 3, the consumer of the result, reads the same slot. Here it
        // checks it: expected = scene - mean(scene over the tile), since the
        // background is constant per tile and the dark offset and gain cancel.
        ++rep.frames;
        {
            const auto out = mat::const_view(H, W, frame);   // read-only: this stage only reads
            for (std::size_t r0 = 0; r0 < H; r0 += T)
                for (std::size_t c0 = 0; c0 < W; c0 += T) {
                    double mean = 0.0;
                    for (std::size_t r = r0; r < r0 + T; ++r)
                        for (std::size_t c = c0; c < c0 + T; ++c) mean += scene(id, r, c);
                    mean /= double(T * T);
                    for (std::size_t r = r0; r < r0 + T; ++r)
                        for (std::size_t c = c0; c < c0 + T; ++c)
                            rep.max_error = std::max(rep.max_error,
                                std::abs(double(out(r, c)) - (scene(id, r, c) - mean)));
                }
        }

        lock.lock();
        sh.state[s] = slot_state::empty;     // give the slot back to the producer
        lock.unlock();
        sh.cv.notify_all();
    }
}

} // namespace

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Zero copy: in-place processing of frames in a shared buffer\n";
    std::cout << "=============================================================\n\n";

    shared_region sh;
    process_report rep;
    const float* const region_begin = sh.pixels.data();

    std::thread prod(producer, std::ref(sh));
    std::thread proc(processor, std::ref(sh), std::ref(rep));
    prod.join();
    proc.join();

    std::cout << frames << " frames of " << H << " x " << W << " float pixels, "
              << slots << " slots, " << sh.pixels.size() * sizeof(float) / 1024 << " KiB shared\n"
              << "  max error against the independent result: " << rep.max_error << "\n\n";

    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        std::cout << (ok ? "  [ok]   " : "  [FAIL] ") << what << "\n";
        if (!ok) ++failures;
    };
    check(rep.frames == frames, "every frame was processed");
    check(rep.max_error < 1e-3, "processed frames match the independent result");
    check(rep.copied == 0, "every frame was processed in its shared slot");
    check(sh.pixels.data() == region_begin, "the shared region was never reallocated");

    std::cout << "\n" << (failures ? "FAILED" : "All checks passed") << "\n";
    return failures ? 1 : 0;
}
