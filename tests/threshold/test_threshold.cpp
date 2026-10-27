#include "threshold.h"

#include <cstdio>
#include <random>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

static std::vector<uint64_t> bimodal(double hm, double hs, double cm, double cs, int nh, int nc) {
    std::mt19937_64 rng(42);
    std::normal_distribution<double> hit(hm, hs), conflict(cm, cs);
    std::vector<uint64_t> lat;
    for (int i = 0; i < nh; ++i) lat.push_back((uint64_t)hit(rng));
    for (int i = 0; i < nc; ++i) lat.push_back((uint64_t)conflict(rng));
    return lat;
}

// Row hits around 100, conflicts around 250. The legacy heuristic only has to land between the
// modes (it tends to sit close to the conflict peak).
static void test_legacy_bimodal() {
    ThresholdDetectionResult r = detect_latency_threshold(bimodal(100, 5, 250, 8, 9000, 1000));
    CHECK(r.threshold > 120);
    CHECK(r.threshold < 250);
}

// The foot method must place the threshold in the gap / at the base of the conflict bump,
// at least ~2 sigma below the conflict mean, and report the result as valid.
static void test_foot_bimodal() {
    struct Case { double hm, hs, cm, cs; int nh, nc; } cases[] = {
        {100, 5, 250, 8, 9000, 1000},
        {100, 10, 250, 20, 9000, 1000},
        {100, 5, 250, 8, 5000, 5000},
        {100, 10, 250, 20, 5000, 5000},
        {100, 5, 250, 8, 9900, 100},
        {100, 10, 200, 15, 8000, 2000},
    };
    for (const Case& c : cases) {
        ThresholdDetectionResult r = detect_latency_threshold(
            bimodal(c.hm, c.hs, c.cm, c.cs, c.nh, c.nc), 7, 0.95, ThresholdMethod::Foot);
        CHECK(r.threshold_valid);
        CHECK(!r.used_fallback);
        CHECK(r.threshold > c.hm + 4 * c.hs);
        CHECK(r.threshold < c.cm - 1.5 * c.cs);
    }
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
    test_legacy_bimodal();
    test_foot_bimodal();
    test_empty_falls_back();
    test_too_few_samples();
    test_identical_samples();
    if (failures) return 1;
    std::puts("threshold tests OK");
    return 0;
}
