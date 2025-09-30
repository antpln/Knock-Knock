#include "threshold.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <string>

ThresholdDetectionResult detect_latency_threshold(const std::vector<uint64_t>& latencies,
                                                  int smoothing_window,
                                                  double fallback_quantile) {
    ThresholdDetectionResult result{};

    result.threshold_valid = false;
    result.used_fallback = false;
    result.separation_ratio = 0.0;
    result.threshold_idx = 0;

    if (latencies.empty()) {
        result.threshold = 150;
        result.used_fallback = true;
        result.threshold_valid = false;
        result.fallback_reason = "No latency samples provided";
        return result;
    }

    std::vector<uint64_t> sorted_latencies = latencies;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());

    result.min_latency = sorted_latencies.front();
    result.max_latency = sorted_latencies.back();

    if (sorted_latencies.size() < 100) {
        result.threshold = 150;
        result.used_fallback = true;
        result.threshold_valid = false;
        result.fallback_reason = "Not enough latency samples for reliable histogram";
        return result;
    }

    if (result.max_latency <= result.min_latency) {
        result.threshold = result.max_latency + 50;
        result.used_fallback = true;
        result.threshold_valid = false;
        result.fallback_reason = "All measured latencies are identical";
        return result;
    }

    if (smoothing_window < 1) {
        smoothing_window = 1;
    }
    if (smoothing_window % 2 == 0) {
        ++smoothing_window;
    }

    const size_t range = static_cast<size_t>(result.max_latency - result.min_latency + 1);
    result.histogram.assign(range, 0);
    for (uint64_t lat : sorted_latencies) {
        const size_t idx = static_cast<size_t>(lat - result.min_latency);
        if (idx < result.histogram.size()) {
            result.histogram[idx]++;
        }
    }

    result.smoothed_histogram.assign(range, 0.0);
    const int half_window = smoothing_window / 2;
    for (size_t i = 0; i < range; ++i) {
        double sum = 0.0;
        int count = 0;
        for (int offset = -half_window; offset <= half_window; ++offset) {
            const long neighbor = static_cast<long>(i) + offset;
            if (neighbor >= 0 && neighbor < static_cast<long>(range)) {
                sum += result.histogram[static_cast<size_t>(neighbor)];
                ++count;
            }
        }
        if (count > 0) {
            result.smoothed_histogram[i] = sum / static_cast<double>(count);
        }
    }

    result.main_peak_idx = std::distance(
        result.smoothed_histogram.begin(),
        std::max_element(result.smoothed_histogram.begin(), result.smoothed_histogram.end()));

    std::vector<size_t> local_maxima;
    if (range == 1) {
        local_maxima.push_back(0);
    } else {
        for (size_t i = 1; i + 1 < range; ++i) {
            double left = result.smoothed_histogram[i - 1];
            double curr = result.smoothed_histogram[i];
            double right = result.smoothed_histogram[i + 1];
            if (curr >= left && curr >= right) {
                local_maxima.push_back(i);
            }
        }
        if (result.smoothed_histogram[0] >= result.smoothed_histogram[1]) {
            local_maxima.push_back(0);
        }
        if (result.smoothed_histogram[range - 1] >= result.smoothed_histogram[range - 2]) {
            local_maxima.push_back(range - 1);
        }
    }

    size_t high_peak_idx = 0;
    double high_peak_val = std::numeric_limits<double>::lowest();
    for (size_t idx : local_maxima) {
        if (idx > result.main_peak_idx) {
            double value = result.smoothed_histogram[idx];
            if (value > high_peak_val) {
                high_peak_val = value;
                high_peak_idx = idx;
            }
        }
    }

    if (high_peak_val == std::numeric_limits<double>::lowest() ||
        high_peak_idx <= result.main_peak_idx + 1) {
        size_t q_index = static_cast<size_t>(fallback_quantile * sorted_latencies.size());
        q_index = std::min(q_index, sorted_latencies.size() - 1);
        result.threshold = sorted_latencies[q_index];
        result.used_fallback = true;
        result.threshold_valid = false;
        result.fallback_reason = "Could not isolate a distinct high-latency peak";
        result.high_latency_peak_idx = result.main_peak_idx;
        return result;
    }

    bool found_valley = false;
    size_t threshold_idx = result.main_peak_idx + 1;

    for (size_t i = high_peak_idx; i > result.main_peak_idx + 1; --i) {
        double prev = result.smoothed_histogram[i - 1];
        double curr = result.smoothed_histogram[i];
        double next = (i + 1 < range) ? result.smoothed_histogram[i + 1] : curr;
        if (curr <= prev && curr <= next) {
            threshold_idx = i;
            found_valley = true;
            break;
        }
    }

    if (!found_valley) {
        double trough_val = result.smoothed_histogram[threshold_idx];
        for (size_t i = result.main_peak_idx + 1; i < high_peak_idx; ++i) {
            double value = result.smoothed_histogram[i];
            if (value <= trough_val) {
                trough_val = value;
                threshold_idx = i;
            }
        }
    }

    if (threshold_idx >= high_peak_idx) {
        threshold_idx = high_peak_idx - 1;
    }
    if (threshold_idx <= result.main_peak_idx) {
        threshold_idx = result.main_peak_idx + 1;
    }

    result.high_latency_peak_idx = high_peak_idx;
    result.threshold_idx = threshold_idx;
    result.threshold = result.min_latency + threshold_idx;

    // Evaluate quality of the valley between peaks
    const double main_peak_height = result.smoothed_histogram[result.main_peak_idx];
    const double high_peak_height = result.smoothed_histogram[result.high_latency_peak_idx];
    const double valley_height = result.smoothed_histogram[threshold_idx];
    const size_t peak_distance = (result.high_latency_peak_idx > result.main_peak_idx)
        ? (result.high_latency_peak_idx - result.main_peak_idx)
        : 0;

    if (valley_height <= 0.0) {
        result.separation_ratio = std::numeric_limits<double>::infinity();
    } else {
        const double limiting_peak = std::min(main_peak_height, high_peak_height);
        result.separation_ratio = (limiting_peak > 0.0)
            ? (limiting_peak / valley_height)
            : 0.0;
    }

    const bool peaks_reasonable = (main_peak_height > 0.0) && (high_peak_height > 0.0);
    const bool distance_ok = peak_distance >= 3;
    const bool valley_strong = result.separation_ratio >= 2.0;

    result.threshold_valid = peaks_reasonable && distance_ok && valley_strong;

    if (!result.threshold_valid) {
        if (result.fallback_reason.empty()) {
            if (!peaks_reasonable) {
                result.fallback_reason = "Peak heights too low for reliable separation";
            } else if (!distance_ok) {
                result.fallback_reason = "High-latency peak too close to main peak";
            } else if (!valley_strong) {
                result.fallback_reason = "Valley between peaks not pronounced enough";
            }
        }
    }

    return result;
}
