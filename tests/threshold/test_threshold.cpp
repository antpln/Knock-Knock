#include "threshold.h"

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    std::vector<uint64_t> latencies;
    latencies.reserve(1500);

    for (int i = 0; i < 500; ++i) latencies.push_back(100);
    for (int i = 0; i < 400; ++i) latencies.push_back(101);
    for (int i = 0; i < 300; ++i) latencies.push_back(102);

    for (int i = 0; i < 20; ++i) latencies.push_back(181);

    for (int i = 0; i < 200; ++i) latencies.push_back(240);
    for (int i = 0; i < 150; ++i) latencies.push_back(241);
    for (int i = 0; i < 120; ++i) latencies.push_back(242);

    auto result = detect_latency_threshold(latencies);
    assert(!result.used_fallback);
    assert(result.threshold_valid);
    assert(result.threshold >= 235 && result.threshold <= 239);
    assert(result.high_latency_peak_idx > result.main_peak_idx);
    assert(result.threshold < result.min_latency + result.high_latency_peak_idx);
    assert(result.separation_ratio >= 2.0);

    std::vector<uint64_t> flat_latencies(150, 210);
    auto flat_result = detect_latency_threshold(flat_latencies);
    assert(flat_result.used_fallback);
    assert(flat_result.threshold == 260);
    assert(!flat_result.threshold_valid);

    std::vector<uint64_t> descending_hist;
    descending_hist.reserve(900);
    descending_hist.insert(descending_hist.end(), 500, 100);
    descending_hist.insert(descending_hist.end(), 300, 101);
    descending_hist.insert(descending_hist.end(), 100, 102);

    auto quantile_result = detect_latency_threshold(descending_hist);
    assert(quantile_result.used_fallback);
    assert(quantile_result.threshold == 102);
    assert(!quantile_result.threshold_valid);

    auto empty_result = detect_latency_threshold({});
    assert(empty_result.used_fallback);
    assert(empty_result.threshold == 150);
    assert(!empty_result.threshold_valid);

    return 0;
}
