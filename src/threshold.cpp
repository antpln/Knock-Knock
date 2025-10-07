#include "threshold.h"

#include <algorithm>
#include <cmath>
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
    result.confidence = 0.0;

    if (latencies.empty()) {
        result.threshold = 150;
        result.used_fallback = true;
        result.threshold_valid = false;
        result.fallback_reason = "No latency samples provided";
        result.confidence = 0.0;
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
        result.confidence = 0.1;
        return result;
    }

    if (result.max_latency <= result.min_latency) {
        result.threshold = result.max_latency + 50;
        result.used_fallback = true;
        result.threshold_valid = false;
        result.fallback_reason = "All measured latencies are identical";
        result.confidence = 0.1;
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

    // Find the HIGHEST peak among all high-latency peaks (after main peak)
    // This handles the case where there are multiple small high-latency bumps
    size_t high_peak_idx = 0;
    double high_peak_val = std::numeric_limits<double>::lowest();
    
    // First, determine a minimum distance from main peak to consider
    // This avoids picking shoulders of the main peak
    const size_t min_distance_from_main = std::max(size_t(5), 
                                                     static_cast<size_t>(range * 0.02));
    
    for (size_t idx : local_maxima) {
        if (idx > result.main_peak_idx + min_distance_from_main) {
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
        result.confidence = 0.2;
        return result;
    }

    // "Find the Bump's Left Foot" heuristic:
    // Start from the HIGHEST high-latency peak and walk left to find where 
    // the slope changes from increasing (moving left) to decreasing or flat.
    // This is the "left foot" where the high-latency bump begins.
    
    bool found_left_foot = false;
    size_t threshold_idx = result.main_peak_idx + 1;
    
    // First, find the minimum (valley) between main peak and high peak
    size_t valley_idx = result.main_peak_idx + 1;
    double valley_val = result.smoothed_histogram[valley_idx];
    for (size_t i = result.main_peak_idx + 1; i < high_peak_idx; ++i) {
        double value = result.smoothed_histogram[i];
        if (value < valley_val) {
            valley_val = value;
            valley_idx = i;
        }
    }
    
    // Calculate a threshold for "significant rise" based on peak height
    const double peak_height = result.smoothed_histogram[high_peak_idx];
    const double valley_to_peak = peak_height - valley_val;
    
    // We want to find where the curve starts its significant ascent
    // Define "significant" as rising more than 5% of the valley-to-peak range
    const double rise_threshold = valley_to_peak * 0.05;
    
    // Walk from high_peak towards valley
    // Find the point where the derivative (rise rate) first exceeds our threshold
    // This is the "left foot" - where the bump starts rising
    for (size_t i = high_peak_idx; i > valley_idx + 1; --i) {
        double curr = result.smoothed_histogram[i];
        double prev = result.smoothed_histogram[i - 1];
        double rise = curr - prev; // positive when curve goes up as we move right
        
        // If we're still rising significantly as we move right, we're past the foot
        if (rise > rise_threshold) {
            // The left foot is where the rise starts to become significant
            threshold_idx = i - 1;
            found_left_foot = true;
            break;
        }
    }
    
    // Alternative method: find inflection point (where second derivative changes)
    if (!found_left_foot) {
        // Look for where the curve transitions from concave down to concave up
        // (second derivative changes from negative to positive)
        for (size_t i = valley_idx + 2; i < high_peak_idx - 1; ++i) {
            double prev_rise = result.smoothed_histogram[i] - result.smoothed_histogram[i - 1];
            double next_rise = result.smoothed_histogram[i + 1] - result.smoothed_histogram[i];
            double second_deriv = next_rise - prev_rise;
            
            // If second derivative becomes positive and we're rising significantly
            if (second_deriv > 0 && next_rise > rise_threshold) {
                threshold_idx = i;
                found_left_foot = true;
                break;
            }
        }
    }
    
    // Fallback: use a point slightly above the valley
    // This ensures we're not in the valley itself, but at the beginning of the rise
    if (!found_left_foot) {
        // Find where we've risen 15% of the way from valley to high_peak
        double target_height = valley_val + valley_to_peak * 0.15;
        
        for (size_t i = valley_idx; i < high_peak_idx; ++i) {
            if (result.smoothed_histogram[i] >= target_height) {
                threshold_idx = i;
                found_left_foot = true;
                break;
            }
        }
        
        // Ultimate fallback: slightly past the valley
        if (!found_left_foot) {
            threshold_idx = valley_idx + std::max(size_t(1), 
                                                   (high_peak_idx - valley_idx) / 6);
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
    const double limiting_peak = std::min(main_peak_height, high_peak_height);
    const size_t peak_distance = (result.high_latency_peak_idx > result.main_peak_idx)
        ? (result.high_latency_peak_idx - result.main_peak_idx)
        : 0;

    if (valley_height <= 0.0) {
        result.separation_ratio = std::numeric_limits<double>::infinity();
    } else {
        result.separation_ratio = (limiting_peak > 0.0)
            ? (limiting_peak / valley_height)
            : 0.0;
    }

    const bool peaks_reasonable = (main_peak_height > 0.0) && (high_peak_height > 0.0);
    const bool distance_ok = peak_distance >= 3;
    const bool valley_strong = result.separation_ratio >= 2.0;

    result.threshold_valid = peaks_reasonable && distance_ok && valley_strong;

    auto clamp01 = [](double value) {
        if (value < 0.0) {
            return 0.0;
        }
        if (value > 1.0) {
            return 1.0;
        }
        return value;
    };

    double separation_score = 0.0;
    if (std::isinf(result.separation_ratio)) {
        separation_score = 1.0;
    } else {
        separation_score = clamp01((result.separation_ratio - 1.0) / 4.0);
    }

    const double distance_score = clamp01(static_cast<double>(peak_distance) / 12.0);

    double valley_score = 0.0;
    if (limiting_peak > 0.0) {
        const double depth = 1.0 - (valley_height / limiting_peak);
        valley_score = clamp01(depth);
    }

    const double combined = (separation_score + distance_score + valley_score) / 3.0;
    result.confidence = clamp01(0.25 + 0.6 * combined);

    if (!result.threshold_valid) {
        result.confidence = std::min(result.confidence, 0.45);
    }
    if (result.used_fallback) {
        result.confidence = std::min(result.confidence, 0.25);
    }

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
