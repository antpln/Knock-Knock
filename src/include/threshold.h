#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ThresholdDetectionResult {
    uint64_t threshold;
    uint64_t min_latency;
    uint64_t max_latency;
    size_t main_peak_idx;
    size_t high_latency_peak_idx;
    size_t threshold_idx;
    bool used_fallback;
    bool threshold_valid;
    double separation_ratio;
    std::string fallback_reason;
    std::vector<int> histogram;
    std::vector<double> smoothed_histogram;
};

ThresholdDetectionResult detect_latency_threshold(
    const std::vector<uint64_t>& latencies,
    int smoothing_window = 7,
    double fallback_quantile = 0.95);

#endif // THRESHOLD_H
