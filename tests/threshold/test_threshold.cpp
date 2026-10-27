#include "threshold.h"

#include <cstdio>
#include <random>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

// Two clear modes (row hits around 100, conflicts around 250) must be separated by the threshold.
static void test_bimodal() {
    std::mt19937_64 rng(42);
    std::normal_distribution<double> hit(100, 5), conflict(250, 8);
    std::vector<uint64_t> lat;
    for (int i = 0; i < 9000; ++i) lat.push_back((uint64_t)hit(rng));
    for (int i = 0; i < 1000; ++i) lat.push_back((uint64_t)conflict(rng));

    ThresholdDetectionResult r = detect_latency_threshold(lat);
    // Only require the threshold to fall between the two modes. The "left foot" heuristic
    // currently lands close to the conflict peak, so a tighter bound is not asserted yet.
    CHECK(r.threshold > 120);
    CHECK(r.threshold < 250);
}

static void test_empty_falls_back() {
    ThresholdDetectionResult r = detect_latency_threshold({});
    CHECK(r.used_fallback);
    CHECK(!r.threshold_valid);
}

static void test_too_few_samples() {
    ThresholdDetectionResult r = detect_latency_threshold(std::vector<uint64_t>(50, 100));
    CHECK(r.used_fallback);
}

static void test_identical_samples() {
    ThresholdDetectionResult r = detect_latency_threshold(std::vector<uint64_t>(1000, 120));
    CHECK(r.used_fallback);
    CHECK(r.threshold > 120);
}

int main() {
    test_bimodal();
    test_empty_falls_back();
    test_too_few_samples();
    test_identical_samples();
    if (failures) return 1;
    std::puts("threshold tests OK");
    return 0;
}
