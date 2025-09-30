#include "full_analysis.h"
#include "rev-mc.h"
#include "utils.h"
#include "threshold.h"
#include <array>
#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "linalg.h"

// Forward declaration for function in rev-mc.cpp
uint64_t measure_one_block_access_time(volatile void* p1, volatile void* p2);
uint64_t virt_to_phys(uint64_t v_addr);


namespace {

struct RowSample {
    uint64_t diff;
    bool conflict;
    uint64_t latency;
};

inline uint64_t bool_vector_to_mask(const std::vector<bool>& bits) {
    uint64_t mask = 0;
    const size_t limit = std::min<size_t>(64, bits.size());
    for (size_t i = 0; i < limit; ++i) {
        if (bits[i]) {
            mask |= (1ULL << i);
        }
    }
    return mask;
}

inline std::vector<bool> mask_to_bool(uint64_t mask) {
    std::vector<bool> bits(64, false);
    for (size_t i = 0; i < 64; ++i) {
        if (mask & (1ULL << i)) {
            bits[i] = true;
        }
    }
    return bits;
}

inline size_t popcount64(uint64_t value) {
#if defined(__GNUG__)
    return static_cast<size_t>(__builtin_popcountll(value));
#else
    size_t count = 0;
    while (value) {
        value &= (value - 1);
        ++count;
    }
    return count;
#endif
}

class GF2Basis {
public:
    bool add(uint64_t value) {
        uint64_t v = value;
        for (int bit = 63; bit >= 0; --bit) {
            const uint64_t pivot = 1ULL << bit;
            if ((v & pivot) == 0) {
                continue;
            }
            if (basis_[bit] == 0) {
                basis_[bit] = v;
                ++rank_;
                return true;
            }
            v ^= basis_[bit];
        }
        return false;
    }

    size_t rank() const {
        return rank_;
    }

private:
    std::array<uint64_t, 64> basis_{};
    size_t rank_ = 0;
};

} // namespace


FullAnalysis::FullAnalysis(size_t mem_size, unsigned int delay_us)
    : mem_size_(mem_size), measurements_(50000), flags_(F_POPULATE), mem_region_(nullptr), threshold_(0), delay_us_(delay_us) {
    allocate_memory();
}

FullAnalysis::~FullAnalysis() {
    free_memory();
}

void FullAnalysis::run() {
    printf("[ LOG ] - Starting full analysis...\n");

    detect_threshold();
    find_bank_masks();
    find_row_masks();

    printf("[ LOG ] - Full analysis complete.\n");
}

void FullAnalysis::allocate_memory() {
    mem_region_ = mmap(NULL, mem_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem_region_ == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    if (flags_ & F_POPULATE) {
        printf("[ LOG ] - Populating memory...\n");
        for (size_t i = 0; i < mem_size_; i += 4096) {
            ((char*)mem_region_)[i] = 1;
        }
    }
}

void FullAnalysis::free_memory() {
    if (mem_region_) {
        munmap(mem_region_, mem_size_);
        mem_region_ = nullptr;
    }
}

std::vector<uint64_t> FullAnalysis::measure_random_latencies(size_t num_pairs) {
    std::vector<uint64_t> latencies;
    latencies.reserve(num_pairs);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, mem_size_ - 1);

    for (size_t i = 0; i < num_pairs; ++i) {
        void* p1 = (char*)mem_region_ + (dist(gen) & ~0x3f);
        void* p2 = (char*)mem_region_ + (dist(gen) & ~0x3f);
        
        uint64_t time = measure_one_block_access_time(p1, p2);
        latencies.push_back(time);

        if (delay_us_ > 0) {
            usleep(delay_us_);
        }
    }
    return latencies;
}

bool FullAnalysis::is_same_bank(uint64_t diff_mask) const {
    if (bank_mask_values_.empty()) {
        return false;
    }

    for (uint64_t mask : bank_mask_values_) {
        if (popcount64(diff_mask & mask) % 2 != 0) {
            return false;
        }
    }
    return true;
}

bool FullAnalysis::collect_same_bank_samples(size_t target_pairs,
                                             size_t min_hits,
                                             size_t min_conflicts,
                                             std::vector<uint64_t>& hit_diffs,
                                             std::vector<uint64_t>& conflict_diffs,
                                             std::vector<uint64_t>& hit_latencies,
                                             std::vector<uint64_t>& conflict_latencies) {
    hit_diffs.clear();
    conflict_diffs.clear();
    hit_latencies.clear();
    conflict_latencies.clear();

    if (bank_mask_values_.empty()) {
        printf("[ERROR] - No bank masks available for row analysis.\n");
        return false;
    }

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, mem_size_ - 1);

    const size_t max_measurements = measurements_ * 40;
    size_t attempts = 0;

    while (attempts < max_measurements) {
        void* p1 = (char*)mem_region_ + (dist(gen) & ~0x3f);
        void* p2 = (char*)mem_region_ + (dist(gen) & ~0x3f);

        uint64_t time = measure_one_block_access_time(p1, p2);
        ++attempts;

        if (delay_us_ > 0) {
            usleep(delay_us_);
        }

        uint64_t v1 = virt_to_phys((uint64_t)p1);
        uint64_t v2 = virt_to_phys((uint64_t)p2);

        if (v1 == 0 || v2 == 0) {
            continue;
        }

        uint64_t diff = v1 ^ v2;
        if (diff == 0) {
            continue;
        }

        if (!is_same_bank(diff)) {
            continue;
        }

        const bool conflict = time > threshold_;
        if (conflict) {
            conflict_diffs.push_back(diff);
            conflict_latencies.push_back(time);
        } else {
            hit_diffs.push_back(diff);
            hit_latencies.push_back(time);
        }

        const size_t collected = hit_diffs.size() + conflict_diffs.size();
        if (collected % 1000 == 0) {
            printf("[ LOG ] - Collected %zu same-bank samples (%zu hits / %zu conflicts)\n",
                   collected, hit_diffs.size(), conflict_diffs.size());
        }

        if (collected >= target_pairs &&
            hit_diffs.size() >= min_hits &&
            conflict_diffs.size() >= min_conflicts) {
            break;
        }
    }

    const bool enough_hits = hit_diffs.size() >= min_hits;
    const bool enough_conflicts = conflict_diffs.size() >= min_conflicts;

    if (!enough_hits || !enough_conflicts) {
        printf("[ERROR] - Insufficient same-bank samples (hits: %zu / %zu, conflicts: %zu / %zu).\n",
               hit_diffs.size(), min_hits,
               conflict_diffs.size(), min_conflicts);
        return false;
    }

    return true;
}

void FullAnalysis::detect_threshold() {
    printf("[ LOG ] - Step 1: Automatic threshold detection (using 'Find the Bump''s Left Foot' strategy)\n");

    size_t num_samples = 100000;
    std::vector<uint64_t> latencies = measure_random_latencies(num_samples);
    std::sort(latencies.begin(), latencies.end());

    if (latencies.empty()) {
        printf("[ERROR] - Threshold detection aborted: no latency samples recorded.\n");
        exit(EXIT_FAILURE);
    }

    FILE* lat_file = fopen("latencies.dat", "w");
    if (lat_file) {
        for (uint64_t lat : latencies) {
            fprintf(lat_file, "%lu\n", lat);
        }
        fclose(lat_file);
    }

    auto detection = detect_latency_threshold(latencies);

    if (!detection.smoothed_histogram.empty()) {
        FILE* smoothed_file = fopen("smoothed_histogram.dat", "w");
        if (smoothed_file) {
            for (size_t i = 0; i < detection.smoothed_histogram.size(); ++i) {
                fprintf(smoothed_file, "%lu,%f\n",
                        detection.min_latency + i,
                        detection.smoothed_histogram[i]);
            }
            fclose(smoothed_file);
        }
    }

    uint64_t auto_threshold = detection.threshold;

    if (!detection.threshold_valid) {
        printf("[ERROR] - Unable to determine a reliable threshold: %s\n",
               detection.fallback_reason.c_str());
        printf("[ERROR] - Aborting automated analysis. Provide cleaner measurements or adjust sampling parameters.\n");
        exit(EXIT_FAILURE);
    }

    FILE* points_file = fopen("analysis_points.dat", "w");
    if (points_file) {
        fprintf(points_file, "main_peak,%lu\n",
                detection.min_latency + detection.main_peak_idx);
        fprintf(points_file, "high_peak,%lu\n",
                detection.min_latency + detection.high_latency_peak_idx);
        fprintf(points_file, "threshold,%lu\n", auto_threshold);
        fclose(points_file);
    }

    printf("[ LOG ] - Main peak at %lu, high-latency peak at %lu. Threshold foot placed at %lu.\n",
           detection.min_latency + detection.main_peak_idx,
           detection.min_latency + detection.high_latency_peak_idx,
           auto_threshold);
    printf("[ LOG ] - Peak separation ratio %.2f across %zu bins.\n",
           detection.separation_ratio,
           detection.high_latency_peak_idx - detection.main_peak_idx);

    threshold_ = auto_threshold;
    printf("[ LOG ] - Final threshold set to %lu cycles\n", threshold_);
}

void FullAnalysis::find_bank_masks() {
    printf("[ LOG ] - Step 2: Bank reversing\n");

    bank_masks_.clear();
    bank_mask_values_.clear();

    std::vector<std::vector<bool>> conflict_diffs;
    
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, mem_size_ - 1);

    size_t measurements_done = 0;
    const size_t min_conflicts = 5000; // We need at least enough to constrain the problem
    const size_t max_measurements = measurements_ * 20; // Don't run forever

    while (conflict_diffs.size() < min_conflicts && measurements_done < max_measurements) {
        void* p1 = (char*)mem_region_ + (dist(gen) & ~0x3f);
        void* p2 = (char*)mem_region_ + (dist(gen) & ~0x3f);

        uint64_t time = measure_one_block_access_time(p1, p2);
        measurements_done++;

        if (time > threshold_) {
            uint64_t v1 = virt_to_phys((uint64_t)p1);
            uint64_t v2 = virt_to_phys((uint64_t)p2);
            
            if (v1 == 0 || v2 == 0) continue; // Skip if address translation failed

            uint64_t diff = v1 ^ v2;

            std::vector<bool> diff_bits(64);
            for(int i=0; i<64; ++i) {
                diff_bits[i] = (diff >> i) & 1;
            }
            conflict_diffs.push_back(diff_bits);
        }
        if(measurements_done % 100000 == 0) {
            printf("[ LOG ] - Performed %zu measurements, found %zu conflicts\n", measurements_done, conflict_diffs.size());
        }
    }

    if (conflict_diffs.empty()) {
        printf("[ERROR] - No conflicts found. Try adjusting the threshold or increasing measurements.\n");
        return;
    }

    printf("[ LOG ] - Found %zu conflicts. Building matrix and finding nullspace...\n", conflict_diffs.size());

    GF2Matrix mat(conflict_diffs.size(), 64, conflict_diffs);
    bank_masks_ = mat.nullspace();

    printf("[ LOG ] - Found %zu bank masks:\n", bank_masks_.size());
    for(const auto& mask_bits : bank_masks_) {
        uint64_t mask = 0;
        for(int i=0; i<64; ++i) {
            if(mask_bits[i]) {
                mask |= (1ULL << i);
            }
        }
        printf("[ LOG ] -   Mask: 0x%016lx\n", mask);
        bank_mask_values_.push_back(mask);
    }
}

void FullAnalysis::find_row_masks() {
    printf("[ LOG ] - Step 3: Row mask reversing\n");
    if (bank_masks_.empty()) {
        printf("[ WARN ] - No bank masks found, skipping row mask analysis.\n");
        return;
    }

    row_masks_.clear();
    row_mask_values_.clear();

    std::vector<uint64_t> hit_diffs;
    std::vector<uint64_t> conflict_diffs;
    std::vector<uint64_t> hit_latencies;
    std::vector<uint64_t> conflict_latencies;

    const size_t target_pairs = 8000;
    const size_t min_hits = 1000;
    const size_t min_conflicts = 1000;

    if (!collect_same_bank_samples(target_pairs,
                                   min_hits,
                                   min_conflicts,
                                   hit_diffs,
                                   conflict_diffs,
                                   hit_latencies,
                                   conflict_latencies)) {
        printf("[ERROR] - Row mask analysis aborted due to insufficient samples.\n");
        return;
    }

    const size_t total_hits = hit_diffs.size();
    const size_t total_conflicts = conflict_diffs.size();

    printf("[ LOG ] - Same-bank dataset: %zu hits (mean %0.2f cycles) / %zu conflicts (mean %0.2f cycles)\n",
           total_hits,
           total_hits ? std::accumulate(hit_latencies.begin(), hit_latencies.end(), 0.0) / total_hits : 0.0,
           total_conflicts,
           total_conflicts ? std::accumulate(conflict_latencies.begin(), conflict_latencies.end(), 0.0) / total_conflicts : 0.0);

    std::array<bool, 64> observed{};
    std::array<bool, 64> hit_varies{};
    std::array<bool, 64> conflict_varies{};

    auto accumulate_bits = [&](const std::vector<uint64_t>& diffs,
                               std::array<bool, 64>& varies) {
        for (uint64_t diff : diffs) {
            for (size_t bit = 0; bit < 64; ++bit) {
                const uint64_t mask = 1ULL << bit;
                if (diff & mask) {
                    observed[bit] = true;
                    varies[bit] = true;
                }
            }
        }
    };

    accumulate_bits(hit_diffs, hit_varies);
    accumulate_bits(conflict_diffs, conflict_varies);

    GF2Matrix bank_mat(bank_masks_.size(), 64, bank_masks_);
    auto nullspace = bank_mat.nullspace();

    if (nullspace.empty()) {
        printf("[ERROR] - Nullspace of bank masks is empty; cannot derive row masks.\n");
        return;
    }

    std::vector<uint64_t> candidates;
    candidates.reserve(nullspace.size());
    for (const auto& vec : nullspace) {
        uint64_t mask = bool_vector_to_mask(vec);
        if (mask != 0) {
            candidates.push_back(mask);
        }
    }

    if (candidates.empty()) {
        printf("[ERROR] - No non-zero row candidates discovered.\n");
        return;
    }

    std::vector<uint64_t> filtered;
    filtered.reserve(candidates.size());

    for (uint64_t mask : candidates) {
        if (mask == 0) {
            continue;
        }

        bool any_bit = false;
        bool all_observed = true;
        bool all_invariant = true;
        bool conflict_flag = false;

        for (size_t bit = 0; bit < 64; ++bit) {
            if ((mask & (1ULL << bit)) == 0) {
                continue;
            }
            any_bit = true;
            if (!observed[bit]) {
                all_observed = false;
                break;
            }
            if (hit_varies[bit]) {
                all_invariant = false;
                break;
            }
            if (conflict_varies[bit]) {
                conflict_flag = true;
            }
        }

        if (!any_bit || !all_observed || !all_invariant || !conflict_flag) {
            continue;
        }

        filtered.push_back(mask);
    }

    if (filtered.empty()) {
        printf("[ERROR] - No row mask candidates satisfied invariance constraints.\n");
        return;
    }

    std::sort(filtered.begin(), filtered.end(), [](uint64_t a, uint64_t b) {
        size_t wa = popcount64(a);
        size_t wb = popcount64(b);
        if (wa != wb) {
            return wa < wb;
        }
        return a < b;
    });

    GF2Basis basis;
    for (uint64_t mask : filtered) {
        if (basis.add(mask)) {
            row_mask_values_.push_back(mask);
            row_masks_.push_back(mask_to_bool(mask));
        }
    }

    if (row_masks_.empty()) {
        printf("[ERROR] - Unable to construct an independent set of row masks.\n");
        return;
    }

    printf("[ LOG ] - Identified %zu independent row mask(s):\n", row_mask_values_.size());
    for (size_t i = 0; i < row_mask_values_.size(); ++i) {
        const uint64_t value = row_mask_values_[i];
        std::vector<int> bits;
        for (size_t bit = 0; bit < 64; ++bit) {
            if (value & (1ULL << bit)) {
                bits.push_back(static_cast<int>(bit));
            }
        }
        printf("[ LOG ] -   Row %zu: 0x%016lx (weight %zu)\n",
               i,
               value,
               popcount64(value));
        printf("             Bits: ");
        for (size_t j = 0; j < bits.size(); ++j) {
            printf("%d%s", bits[j], (j + 1 == bits.size()) ? "\n" : ", ");
        }
    }
}
