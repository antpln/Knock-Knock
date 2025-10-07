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
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_map>
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

inline size_t count_true_bits(const std::array<bool, 64>& bits) {
    size_t count = 0;
    for (bool b : bits) {
        if (b) {
            ++count;
        }
    }
    return count;
}

inline size_t count_true_bits_intersection(const std::array<bool, 64>& a,
                                           const std::array<bool, 64>& b) {
    size_t count = 0;
    for (size_t i = 0; i < 64; ++i) {
        if (a[i] && b[i]) {
            ++count;
        }
    }
    return count;
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

struct CandidateMask {
    uint64_t value;
    size_t weight;
    size_t frequency;
};

struct BankMaskAnalysis {
    size_t subsample_rounds = 0;
    size_t successful_rounds = 0;
    size_t conflict_count = 0;
    std::vector<CandidateMask> ordered_candidates;
    std::vector<uint64_t> final_masks;
};

struct RowMaskAnalysis {
    size_t total_hits = 0;
    size_t total_conflicts = 0;
    std::array<bool, 64> observed{};
    std::array<bool, 64> hit_invariant{};
    std::array<bool, 64> conflict_varies{};
    std::vector<uint64_t> raw_candidates;
    std::vector<uint64_t> filtered_candidates;
    std::vector<uint64_t> final_masks;
};

static size_t safe_binomial(size_t n, size_t k) {
    if (k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    if (k > n - k) {
        k = n - k;
    }
    long double result = 1.0L;
    for (size_t i = 1; i <= k; ++i) {
        result *= static_cast<long double>(n - k + i);
        result /= static_cast<long double>(i);
        if (result > static_cast<long double>(std::numeric_limits<size_t>::max())) {
            return std::numeric_limits<size_t>::max();
        }
    }
    if (result < 1.0L) {
        result = 1.0L;
    }
    return static_cast<size_t>(result + 0.5L);
}

static bool find_minimal_weight_basis(const std::vector<CandidateMask>& candidates,
                                      size_t target_dimension,
                                      std::vector<uint64_t>& best_basis) {
    if (target_dimension == 0) {
        return false;
    }

    const size_t combination_limit = 500000;
    const size_t candidate_count = candidates.size();
    const size_t combinations = safe_binomial(candidate_count, target_dimension);

    std::vector<uint64_t> current_selection;
    current_selection.reserve(target_dimension);

    size_t best_weight = std::numeric_limits<size_t>::max();
    size_t best_frequency_sum = 0;

    bool attempted_depth_first = false;

    if (combinations <= combination_limit && target_dimension <= 10) {
        attempted_depth_first = true;
        std::function<void(size_t, GF2Basis, size_t, size_t)> dfs;
        dfs = [&](size_t index, GF2Basis basis, size_t weight_sum, size_t frequency_sum) {
            if (current_selection.size() == target_dimension) {
                if (weight_sum < best_weight ||
                    (weight_sum == best_weight && frequency_sum > best_frequency_sum)) {
                    best_weight = weight_sum;
                    best_frequency_sum = frequency_sum;
                    best_basis = current_selection;
                }
                return;
            }

            if (index >= candidate_count) {
                return;
            }

            const size_t remaining_needed = target_dimension - current_selection.size();
            const size_t remaining_candidates = candidate_count - index;
            if (remaining_candidates < remaining_needed) {
                return;
            }

            // Pruning: if current weight already exceeds best, skip
            if (weight_sum >= best_weight) {
                return;
            }

            // Try adding current candidate
            if (basis.add(candidates[index].value)) {
                current_selection.push_back(candidates[index].value);
                dfs(index + 1,
                    basis,
                    weight_sum + candidates[index].weight,
                    frequency_sum + candidates[index].frequency);
                current_selection.pop_back();
            }

            // Try skipping current candidate
            dfs(index + 1, basis, weight_sum, frequency_sum);
        };

        dfs(0, GF2Basis(), 0, 0);
        if (!best_basis.empty()) {
            printf("[ LOG ] - Minimal-weight basis resolved via depth-first search (%zu combinations).\n",
                   combinations);
            return true;
        }
    }

    GF2Basis greedy_basis;
    size_t greedy_weight = 0;
    size_t greedy_frequency = 0;
    std::vector<uint64_t> greedy_selection;
    greedy_selection.reserve(target_dimension);

    for (size_t i = 0; i < candidate_count && greedy_selection.size() < target_dimension; ++i) {
        if (greedy_basis.add(candidates[i].value)) {
            greedy_selection.push_back(candidates[i].value);
            greedy_weight += candidates[i].weight;
            greedy_frequency += candidates[i].frequency;
        }
    }

    if (greedy_selection.size() == target_dimension) {
        best_basis = greedy_selection;
        best_weight = greedy_weight;
        best_frequency_sum = greedy_frequency;
        if (attempted_depth_first) {
            printf("[ WARN ] - Depth-first search failed; falling back to greedy ordering.\n");
        } else {
            printf("[ LOG ] - Minimal-weight basis selected via weight-priority greedy ordering.\n");
        }
        return true;
    }

    std::vector<CandidateMask> frequency_sorted = candidates;
    std::sort(frequency_sorted.begin(), frequency_sorted.end(),
              [](const CandidateMask& a, const CandidateMask& b) {
                  if (a.frequency != b.frequency) {
                      return a.frequency > b.frequency;
                  }
                  if (a.weight != b.weight) {
                      return a.weight < b.weight;
                  }
                  return a.value < b.value;
              });

    greedy_basis = GF2Basis();
    greedy_selection.clear();
    greedy_weight = 0;
    greedy_frequency = 0;

    for (size_t i = 0; i < frequency_sorted.size() &&
                     greedy_selection.size() < target_dimension; ++i) {
        if (greedy_basis.add(frequency_sorted[i].value)) {
            greedy_selection.push_back(frequency_sorted[i].value);
            greedy_weight += frequency_sorted[i].weight;
            greedy_frequency += frequency_sorted[i].frequency;
        }
    }

    if (greedy_selection.size() == target_dimension) {
        best_basis = greedy_selection;
        best_weight = greedy_weight;
        best_frequency_sum = greedy_frequency;
        printf("[ WARN ] - Secondary frequency-priority greedy ordering assembled the basis.\n");
        return true;
    }

    printf("[ ERROR] - Unable to assemble minimal-weight basis (dimension=%zu, candidates=%zu).\n",
           target_dimension,
           candidates.size());
    return false;
}

static bool compute_bank_masks_from_conflicts(const std::vector<uint64_t>& conflict_masks,
                                              const std::vector<std::vector<bool>>& conflict_bits,
                                              const AnalysisConfig& config,
                                              BankMaskAnalysis& analysis) {
    analysis = BankMaskAnalysis();
    analysis.conflict_count = conflict_masks.size();

    if (conflict_masks.empty()) {
        return false;
    }

    uint64_t observed_mask = 0;
    for (size_t i = 0; i < conflict_masks.size(); ++i) {
        observed_mask |= conflict_masks[i];
    }

    size_t subsample_size = conflict_masks.size();
    if (subsample_size > config.bank_subsample_size) {
        subsample_size = config.bank_subsample_size;
    }
    if (subsample_size == 0) {
        return false;
    }

    size_t repeat_rounds = config.bank_repeat_rounds;
    if (conflict_masks.size() > 8000) {
        repeat_rounds = std::max(repeat_rounds, static_cast<size_t>(60));
    } else if (conflict_masks.size() > 4000) {
        repeat_rounds = std::max(repeat_rounds, static_cast<size_t>(50));
    }

    analysis.subsample_rounds = repeat_rounds;

    std::unordered_map<uint64_t, size_t> frequency_map;
    std::vector<size_t> indices(conflict_masks.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    std::random_device rd;
    std::mt19937_64 gen(rd());

    printf("[ LOG ] - Starting subsampling (rounds=%zu, subsample_size=%zu).\n",
           repeat_rounds,
           subsample_size);

    size_t total_nullspace_vectors = 0;

    for (size_t round = 0; round < repeat_rounds; ++round) {
        if (indices.size() > subsample_size) {
            std::shuffle(indices.begin(), indices.end(), gen);
        }

        const size_t take = std::min(subsample_size, indices.size());
        if (take == 0) {
            continue;
        }

    std::vector<std::vector<bool> > matrix_data;
    matrix_data.reserve(take);
    for (size_t j = 0; j < take; ++j) {
        matrix_data.push_back(conflict_bits[indices[j]]);
    }

        GF2Matrix mat(matrix_data.size(), 64, matrix_data);
        std::vector<std::vector<bool> > nullspace = mat.nullspace();

        if (!nullspace.empty()) {
            analysis.successful_rounds++;
            total_nullspace_vectors += nullspace.size();
            if ((round + 1) % 10 == 0 || round + 1 == repeat_rounds) {
                printf("[ LOG ] - Round %zu/%zu: rank=%zu, nullspace dim=%zu (vectors: %zu total)\n",
                       round + 1,
                       repeat_rounds,
                       64 - nullspace.size(),
                       nullspace.size(),
                       total_nullspace_vectors);
            }
        } else if ((round + 1) % 10 == 0 || round + 1 == repeat_rounds) {
            printf("[ WARN ] - Round %zu/%zu produced no nullspace vectors (full rank).\n",
                   round + 1,
                   repeat_rounds);
        }

        for (size_t j = 0; j < nullspace.size(); ++j) {
            uint64_t mask = bool_vector_to_mask(nullspace[j]);
            mask &= observed_mask;
            if (mask != 0) {
                frequency_map[mask]++;
            }
        }
    }

    printf("[ LOG ] - Subsampling complete: %zu/%zu rounds successful, %zu total vectors, %zu unique masks.\n",
           analysis.successful_rounds,
           repeat_rounds,
           total_nullspace_vectors,
           frequency_map.size());

    if (analysis.successful_rounds < repeat_rounds / 4) {
        printf("[ WARN ] - Low success rate (%zu/%zu rounds). Results may be unreliable.\n",
               analysis.successful_rounds,
               repeat_rounds);
        printf("           Consider: increasing conflict samples, adjusting threshold, or collecting more data.\n");
    }

    if (frequency_map.empty()) {
        printf("[ ERROR] - Subsampling produced no candidate bank masks.\n");
        return false;
    }

    std::vector<CandidateMask> ordered;
    ordered.reserve(frequency_map.size());
    for (std::unordered_map<uint64_t, size_t>::const_iterator it = frequency_map.begin();
         it != frequency_map.end(); ++it) {
        const uint64_t value = it->first & observed_mask;
        if (value == 0) {
            continue;
        }
        CandidateMask candidate;
        candidate.value = value;
        candidate.weight = popcount64(value);
        candidate.frequency = it->second;
        ordered.push_back(candidate);
    }

    if (ordered.empty()) {
        printf("[ ERROR] - All candidate masks collapsed to zero after filtering observed bits.\n");
        return false;
    }

    std::sort(ordered.begin(), ordered.end(),
              [](const CandidateMask& a, const CandidateMask& b) {
                  if (a.frequency != b.frequency) {
                      return a.frequency > b.frequency;
                  }
                  if (a.weight != b.weight) {
                      return a.weight < b.weight;
                  }
                  return a.value < b.value;
              });

    analysis.ordered_candidates = ordered;

    printf("[ LOG ] - Candidate statistics:\n");
    printf("           Total unique candidates: %zu\n", ordered.size());
    if (!ordered.empty()) {
        size_t min_freq = ordered[0].frequency;
        size_t max_freq = ordered[0].frequency;
        size_t min_weight = ordered[0].weight;
        size_t max_weight = ordered[0].weight;
        for (size_t i = 1; i < ordered.size(); ++i) {
            min_freq = std::min(min_freq, ordered[i].frequency);
            max_freq = std::max(max_freq, ordered[i].frequency);
            min_weight = std::min(min_weight, ordered[i].weight);
            max_weight = std::max(max_weight, ordered[i].weight);
        }
        printf("           Frequency range: %zu - %zu\n", min_freq, max_freq);
        printf("           Weight range: %zu - %zu bits\n", min_weight, max_weight);
        
        // Show top 5 candidates by frequency
        const size_t top_n = std::min<size_t>(5, ordered.size());
        printf("           Top %zu candidates by frequency:\n", top_n);
        for (size_t i = 0; i < top_n; ++i) {
            printf("             %zu. 0x%016lx (freq=%zu, weight=%zu)\n",
                   i + 1,
                   ordered[i].value,
                   ordered[i].frequency,
                   ordered[i].weight);
        }
    }

    GF2Basis rank_basis;
    size_t target_dimension = 0;
    for (size_t i = 0; i < ordered.size(); ++i) {
        if (rank_basis.add(ordered[i].value)) {
            ++target_dimension;
        }
    }

    printf("[ LOG ] - Target dimension (rank of candidate space): %zu\n", target_dimension);

    if (target_dimension == 0) {
        printf("[ ERROR] - Candidate masks are not independent (rank=0).\n");
        return false;
    }

    std::vector<CandidateMask> weight_sorted = ordered;
    std::sort(weight_sorted.begin(), weight_sorted.end(),
              [](const CandidateMask& a, const CandidateMask& b) {
                  if (a.weight != b.weight) {
                      return a.weight < b.weight;
                  }
                  if (a.frequency != b.frequency) {
                      return a.frequency > b.frequency;
                  }
                  return a.value < b.value;
              });

    std::vector<uint64_t> basis_masks;
    if (!find_minimal_weight_basis(weight_sorted, target_dimension, basis_masks)) {
        return false;
    }

    std::sort(basis_masks.begin(), basis_masks.end());
    analysis.final_masks = basis_masks;
    size_t total_weight = 0;
    for (size_t i = 0; i < analysis.final_masks.size(); ++i) {
        total_weight += popcount64(analysis.final_masks[i]);
    }
    printf("[ LOG ] - Minimal basis size: %zu (total weight=%zu).\n",
           analysis.final_masks.size(),
           total_weight);
    return !analysis.final_masks.empty();
}

static bool derive_row_masks(const std::vector<uint64_t>& hit_diffs,
                             const std::vector<uint64_t>& conflict_diffs,
                             const std::vector<std::vector<bool>>& bank_masks,
                             RowMaskAnalysis& analysis) {
    analysis = RowMaskAnalysis();
    analysis.total_hits = hit_diffs.size();
    analysis.total_conflicts = conflict_diffs.size();

    if (bank_masks.empty()) {
        printf("[ERROR] - Row mask analysis requires previously discovered bank masks.\n");
        return false;
    }

    std::array<bool, 64> observed{};
    std::array<bool, 64> hit_varies{};
    std::array<bool, 64> conflict_varies{};

    for (size_t i = 0; i < hit_diffs.size(); ++i) {
        const uint64_t diff = hit_diffs[i];
        if (diff == 0) {
            continue;
        }
        for (size_t bit = 0; bit < 64; ++bit) {
            const uint64_t mask = 1ULL << bit;
            if (diff & mask) {
                observed[bit] = true;
                hit_varies[bit] = true;
            }
        }
    }

    for (size_t i = 0; i < conflict_diffs.size(); ++i) {
        const uint64_t diff = conflict_diffs[i];
        if (diff == 0) {
            continue;
        }
        for (size_t bit = 0; bit < 64; ++bit) {
            const uint64_t mask = 1ULL << bit;
            if (diff & mask) {
                observed[bit] = true;
                conflict_varies[bit] = true;
            }
        }
    }

    analysis.observed = observed;
    analysis.conflict_varies = conflict_varies;
    for (size_t bit = 0; bit < 64; ++bit) {
        analysis.hit_invariant[bit] = observed[bit] && !hit_varies[bit];
    }

    printf("[ LOG ] - Row mask bit analysis:\n");
    printf("           Observed bits: %zu\n", count_true_bits(observed));
    printf("           Hit-invariant bits: %zu\n", count_true_bits(analysis.hit_invariant));
    printf("           Conflict-varying bits: %zu\n", count_true_bits(conflict_varies));

    GF2Matrix bank_mat(bank_masks.size(), 64, bank_masks);
    std::vector<std::vector<bool>> nullspace = bank_mat.nullspace();

    if (nullspace.empty()) {
        printf("[ERROR] - Nullspace of bank mask basis is empty; cannot derive row masks.\n");
        return false;
    }

    analysis.raw_candidates.reserve(nullspace.size());
    for (size_t i = 0; i < nullspace.size(); ++i) {
        const uint64_t candidate = bool_vector_to_mask(nullspace[i]);
        if (candidate != 0) {
            analysis.raw_candidates.push_back(candidate);
        }
    }

    if (analysis.raw_candidates.empty()) {
        printf("[ERROR] - Nullspace contained only the zero vector; no row mask candidates available.\n");
        return false;
    }

    std::vector<uint64_t> filtered;
    filtered.reserve(analysis.raw_candidates.size());

    for (size_t i = 0; i < analysis.raw_candidates.size(); ++i) {
        const uint64_t mask = analysis.raw_candidates[i];
        bool all_observed = true;
        bool all_invariant = true;
        bool conflict_flag = false;

        for (size_t bit = 0; bit < 64; ++bit) {
            if ((mask & (1ULL << bit)) == 0) {
                continue;
            }

            if (!analysis.observed[bit]) {
                all_observed = false;
                break;
            }

            if (!analysis.hit_invariant[bit]) {
                all_invariant = false;
                break;
            }

            if (analysis.conflict_varies[bit]) {
                conflict_flag = true;
            }
        }

        if (all_observed && all_invariant && conflict_flag) {
            filtered.push_back(mask);
        }
    }

    if (filtered.empty()) {
        printf("[ERROR] - No row mask candidates satisfied observed/invariance constraints.\n");
        return false;
    }

    std::sort(filtered.begin(), filtered.end());
    filtered.erase(std::unique(filtered.begin(), filtered.end()), filtered.end());
    std::sort(filtered.begin(), filtered.end(),
              [](uint64_t a, uint64_t b) {
                  const size_t wa = popcount64(a);
                  const size_t wb = popcount64(b);
                  if (wa != wb) {
                      return wa < wb;
                  }
                  return a < b;
              });

    analysis.filtered_candidates = filtered;

    GF2Basis row_basis;
    std::vector<uint64_t> final_masks;
    final_masks.reserve(filtered.size());

    for (size_t i = 0; i < filtered.size(); ++i) {
        if (row_basis.add(filtered[i])) {
            final_masks.push_back(filtered[i]);
        }
    }

    if (final_masks.empty()) {
        printf("[ERROR] - Filtered row masks were linearly dependent; basis construction failed.\n");
        return false;
    }

    analysis.final_masks = final_masks;
    return true;
}

} // namespace


FullAnalysis::FullAnalysis(size_t mem_size, unsigned int delay_us)
    : mem_size_(mem_size), measurements_(50000), flags_(F_POPULATE), mem_region_(nullptr), 
      threshold_(0), delay_us_(delay_us), config_(AnalysisConfig()) {
    allocate_memory();
    bank_measurements_.reserve(kMaxBankEvaluationSamples);
}

FullAnalysis::FullAnalysis(size_t mem_size, unsigned int delay_us, const AnalysisConfig& config)
    : mem_size_(mem_size), measurements_(50000), flags_(F_POPULATE), mem_region_(nullptr), 
      threshold_(0), delay_us_(delay_us), config_(config) {
    allocate_memory();
    bank_measurements_.reserve(kMaxBankEvaluationSamples);
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

bool FullAnalysis::predict_conflict(uint64_t diff_mask) const {
    return is_same_bank(diff_mask);
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

    size_t attempts = 0;
    size_t attempt_limit = config_.row_max_attempts;
    size_t expansions = 0;
    const size_t max_expansions = 3;

    while (true) {
        while (attempts < attempt_limit) {
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
                printf("[ LOG ] - Same-bank target met (hits=%zu, conflicts=%zu).\n",
                       hit_diffs.size(),
                       conflict_diffs.size());
                return true;
            }
        }

        if (attempts < attempt_limit) {
            printf("[ WARN ] - Sampling loop exited early (attempts=%zu, limit=%zu).\n",
                   attempts,
                   attempt_limit);
            break;
        }

        if (expansions >= max_expansions) {
            printf("[ ERROR] - Maximum same-bank sampling expansions reached.\n");
            break;
        }

        ++expansions;
        attempt_limit += measurements_ * 20;
        printf("[ WARN ] - Extending same-bank sampling budget to %zu attempts.\n", attempt_limit);
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

    // Check if manual threshold is specified
    if (config_.manual_threshold > 0) {
        threshold_ = config_.manual_threshold;
        printf("[ LOG ] - Using manual threshold: %lu cycles (auto-detection skipped)\n", threshold_);
        return;
    }

    size_t num_samples = config_.threshold_samples;
    printf("[ LOG ] - Collecting %zu latency samples for threshold detection...\n", num_samples);
    
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
        printf("[ LOG ] - Saved raw latency samples to latencies.dat\n");
    }

    auto detection = detect_latency_threshold(latencies);

    // Save raw histogram (unsmoothed)
    if (!detection.histogram.empty()) {
        FILE* raw_hist_file = fopen("raw_histogram.dat", "w");
        if (raw_hist_file) {
            fprintf(raw_hist_file, "# latency,count\n");
            for (size_t i = 0; i < detection.histogram.size(); ++i) {
                if (detection.histogram[i] > 0) {  // Only save non-zero bins
                    fprintf(raw_hist_file, "%lu,%d\n",
                            detection.min_latency + i,
                            detection.histogram[i]);
                }
            }
            fclose(raw_hist_file);
            printf("[ LOG ] - Saved raw histogram to raw_histogram.dat\n");
        }
    }

    // Save smoothed histogram
    if (!detection.smoothed_histogram.empty()) {
        FILE* smoothed_file = fopen("smoothed_histogram.dat", "w");
        if (smoothed_file) {
            fprintf(smoothed_file, "# latency,smoothed_count\n");
            for (size_t i = 0; i < detection.smoothed_histogram.size(); ++i) {
                fprintf(smoothed_file, "%lu,%f\n",
                        detection.min_latency + i,
                        detection.smoothed_histogram[i]);
            }
            fclose(smoothed_file);
            printf("[ LOG ] - Saved smoothed histogram to smoothed_histogram.dat\n");
        }
    }

    uint64_t auto_threshold = detection.threshold;

    if (!detection.threshold_valid) {
        printf("[WARN ] - Threshold quality check failed: %s\n",
               detection.fallback_reason.c_str());
        printf("[WARN ] - Proceeding with fallback threshold (%lu cycles). Results may be noisy.\n",
               auto_threshold);
    }

    FILE* points_file = fopen("analysis_points.dat", "w");
    if (points_file) {
        fprintf(points_file, "main_peak,%lu\n",
                detection.min_latency + detection.main_peak_idx);
        fprintf(points_file, "high_peak,%lu\n",
                detection.min_latency + detection.high_latency_peak_idx);
        fprintf(points_file, "threshold,%lu\n", auto_threshold);
        fprintf(points_file, "confidence,%.3f\n", detection.confidence);
        fclose(points_file);
        printf("[ LOG ] - Saved analysis points to analysis_points.dat\n");
    }

    // Save comprehensive summary for manual analysis
    FILE* summary_file = fopen("threshold_analysis_summary.txt", "w");
    if (summary_file) {
        fprintf(summary_file, "=== THRESHOLD DETECTION ANALYSIS SUMMARY ===\n\n");
        fprintf(summary_file, "Total samples collected: %zu\n", latencies.size());
        fprintf(summary_file, "Latency range: %lu - %lu cycles\n", 
                detection.min_latency, detection.max_latency);
        fprintf(summary_file, "\n--- Peak Analysis ---\n");
        fprintf(summary_file, "Main peak location: %lu cycles (index %zu)\n",
                detection.min_latency + detection.main_peak_idx,
                detection.main_peak_idx);
        fprintf(summary_file, "High-latency peak location: %lu cycles (index %zu)\n",
                detection.min_latency + detection.high_latency_peak_idx,
                detection.high_latency_peak_idx);
        fprintf(summary_file, "Peak separation: %zu bins\n",
                detection.high_latency_peak_idx - detection.main_peak_idx);
        fprintf(summary_file, "Separation ratio: %.3f\n", detection.separation_ratio);
        
        fprintf(summary_file, "\n--- Threshold Decision ---\n");
        fprintf(summary_file, "Auto-detected threshold: %lu cycles\n", auto_threshold);
        fprintf(summary_file, "Threshold confidence: %.3f\n", detection.confidence);
        fprintf(summary_file, "Threshold valid: %s\n", 
                detection.threshold_valid ? "YES" : "NO");
        
        if (!detection.threshold_valid) {
            fprintf(summary_file, "Fallback used: %s\n", 
                    detection.used_fallback ? "YES" : "NO");
            fprintf(summary_file, "Fallback reason: %s\n", 
                    detection.fallback_reason.c_str());
        }
        
        fprintf(summary_file, "\n--- Files Generated ---\n");
        fprintf(summary_file, "latencies.dat - Raw sorted latency samples (one per line)\n");
        fprintf(summary_file, "raw_histogram.dat - Histogram bins (latency,count)\n");
        fprintf(summary_file, "smoothed_histogram.dat - Smoothed histogram for visualization\n");
        fprintf(summary_file, "analysis_points.dat - Key points for plotting\n");
        
        fprintf(summary_file, "\n--- Manual Threshold Selection Guide ---\n");
        fprintf(summary_file, "If auto-detection failed or you want to verify:\n");
        fprintf(summary_file, "1. Plot the histogram: python3 plot_histogram.py\n");
        fprintf(summary_file, "2. Look for the 'left foot' of the high-latency bump\n");
        fprintf(summary_file, "3. The threshold should separate low-latency hits from conflicts\n");
        fprintf(summary_file, "4. Typical values: 150-250 cycles for most systems\n");
        fprintf(summary_file, "5. Re-run with: ./main --full-analysis --threshold <value>\n");
        
        fprintf(summary_file, "\n--- Quick Statistics ---\n");
        fprintf(summary_file, "Min latency: %lu cycles\n", detection.min_latency);
        fprintf(summary_file, "Max latency: %lu cycles\n", detection.max_latency);
        fprintf(summary_file, "Range: %lu cycles\n", 
                detection.max_latency - detection.min_latency);
        
        fclose(summary_file);
        printf("[ LOG ] - Saved threshold analysis summary to threshold_analysis_summary.txt\n");
    }

    printf("[ LOG ] - Main peak at %lu, high-latency peak at %lu. Threshold foot placed at %lu.\n",
           detection.min_latency + detection.main_peak_idx,
           detection.min_latency + detection.high_latency_peak_idx,
           auto_threshold);
    printf("[ LOG ] - Peak separation ratio %.2f across %zu bins.\n",
           detection.separation_ratio,
           detection.high_latency_peak_idx - detection.main_peak_idx);

    printf("[ LOG ] - Threshold confidence score: %.2f\n", detection.confidence);

    threshold_ = auto_threshold;
    printf("[ LOG ] - Final threshold set to %lu cycles\n", threshold_);
}

void FullAnalysis::find_bank_masks() {
    printf("[ LOG ] - Step 2: Bank reversing\n");

    bank_masks_.clear();
    bank_mask_values_.clear();
    bank_measurements_.clear();
    bank_measurements_.reserve(kMaxBankEvaluationSamples);

    std::vector<uint64_t> conflict_masks;
    std::vector<std::vector<bool>> conflict_bits;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, mem_size_ - 1);

    const size_t base_target_conflicts = config_.bank_target_conflicts;
    const size_t max_measurements = config_.bank_max_measurements;
    const size_t progress_interval = 100000;
    const size_t max_attempts = config_.bank_max_attempts;

    printf("[ LOG ] - Bank mask search config: target_conflicts=%zu, max_measurements=%zu, max_attempts=%zu.\n",
           base_target_conflicts,
           max_measurements,
           max_attempts);

    size_t desired_conflicts = base_target_conflicts;
    size_t total_measurements = 0;
    size_t attempts = 0;

    BankMaskAnalysis analysis;
    bool solved = false;

    conflict_masks.reserve(desired_conflicts);
    conflict_bits.reserve(desired_conflicts);

    while (!solved && total_measurements < max_measurements) {
        if (conflict_masks.capacity() < desired_conflicts) {
            conflict_masks.reserve(desired_conflicts);
        }
        if (conflict_bits.capacity() < desired_conflicts) {
            conflict_bits.reserve(desired_conflicts);
        }

        while (conflict_masks.size() < desired_conflicts && total_measurements < max_measurements) {
            void* p1 = (char*)mem_region_ + (dist(gen) & ~0x3f);
            void* p2 = (char*)mem_region_ + (dist(gen) & ~0x3f);

            uint64_t time = measure_one_block_access_time(p1, p2);
            ++total_measurements;

            if (delay_us_ > 0) {
                usleep(delay_us_);
            }

            uint64_t v1 = virt_to_phys((uint64_t)p1);
            uint64_t v2 = virt_to_phys((uint64_t)p2);
            if (v1 == 0 || v2 == 0) {
                if (total_measurements % progress_interval == 0) {
                    printf("[ LOG ] - Performed %zu measurements, conflicts gathered %zu\n",
                           total_measurements,
                           conflict_masks.size());
                }
                continue;
            }

            uint64_t diff = v1 ^ v2;
            if (diff == 0) {
                continue;
            }

            const bool conflict = time > threshold_;

            if (bank_measurements_.size() < kMaxBankEvaluationSamples) {
                bank_measurements_.push_back({diff, time, conflict});
            }

            if (!conflict) {
                if (total_measurements % progress_interval == 0) {
                    printf("[ LOG ] - Performed %zu measurements, conflicts gathered %zu\n",
                           total_measurements,
                           conflict_masks.size());
                }
                continue;
            }

            conflict_masks.push_back(diff);
            conflict_bits.push_back(mask_to_bool(diff));

            if (conflict_masks.size() % 1000 == 0) {
                printf("[ LOG ] - Collected %zu conflict samples\n", conflict_masks.size());
            }
        }

        if (conflict_masks.size() < desired_conflicts) {
            printf("[ WARN ] - Measurement budget exhausted (conflicts=%zu, target=%zu).\n",
                   conflict_masks.size(),
                   desired_conflicts);
            break;
        }

        if (compute_bank_masks_from_conflicts(conflict_masks, conflict_bits, config_, analysis)) {
            solved = true;
            break;
        }

        if (attempts + 1 >= max_attempts) {
            printf("[ ERROR] - Maximum retry attempts reached for bank mask derivation.\n");
            break;
        }

        ++attempts;
        desired_conflicts += desired_conflicts / 2;
        if (desired_conflicts > conflict_masks.size() + 2000) {
            desired_conflicts = conflict_masks.size() + 2000;
        }
        if (desired_conflicts > 20000) {
            desired_conflicts = 20000;
        }
        printf("[ WARN ] - Bank mask solving incomplete. Increasing target conflicts to %zu (attempt %zu/%zu).\n",
               desired_conflicts,
               attempts,
               max_attempts);
    }

    if (!solved || analysis.final_masks.empty()) {
        printf("[ERROR] - Unable to derive bank masks. Try longer runs or adjust threshold.\n");
        return;
    }

    bank_mask_values_ = analysis.final_masks;
    bank_masks_.clear();
    for (size_t i = 0; i < bank_mask_values_.size(); ++i) {
        bank_masks_.push_back(mask_to_bool(bank_mask_values_[i]));
    }

    printf("[ LOG ] - Conflict samples processed: %zu\n", conflict_masks.size());
    printf("[ LOG ] - Subsampling rounds: %zu (%zu successful)\n",
           analysis.subsample_rounds,
           analysis.successful_rounds);
    printf("[ LOG ] - Unique candidate masks: %zu\n", analysis.ordered_candidates.size());
    if (analysis.successful_rounds < analysis.subsample_rounds / 4) {
        printf("[ WARN ] - Only %zu of %zu rounds produced nullspace vectors; consider collecting more data.\n",
               analysis.successful_rounds,
               analysis.subsample_rounds);
    }

    printf("[ LOG ] - Final bank mask set (%zu masks):\n", bank_mask_values_.size());
    for (size_t i = 0; i < bank_mask_values_.size(); ++i) {
        const uint64_t value = bank_mask_values_[i];
        size_t frequency = 0;
        for (size_t j = 0; j < analysis.ordered_candidates.size(); ++j) {
            if (analysis.ordered_candidates[j].value == value) {
                frequency = analysis.ordered_candidates[j].frequency;
                break;
            }
        }

        std::vector<int> bits;
        for (size_t bit = 0; bit < 64; ++bit) {
            if (value & (1ULL << bit)) {
                bits.push_back(static_cast<int>(bit));
            }
        }

        printf("[ LOG ] -   Mask %zu: 0x%016lx (weight %zu, seen %zux)\n",
               i,
               value,
               popcount64(value),
               frequency);
        printf("             Bits: ");
        for (size_t j = 0; j < bits.size(); ++j) {
            printf("%d%s", bits[j], (j + 1 == bits.size()) ? "\n" : ", ");
        }
    }

    evaluate_bank_masks();
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

    const size_t target_pairs = config_.row_target_pairs;
    const size_t min_hits = config_.row_min_hits;
    const size_t min_conflicts = config_.row_min_conflicts;

    printf("[ LOG ] - Row mask search config: target_pairs=%zu, min_hits=%zu, min_conflicts=%zu.\n",
           target_pairs,
           min_hits,
           min_conflicts);

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

    RowMaskAnalysis analysis;
    if (!derive_row_masks(hit_diffs, conflict_diffs, bank_masks_, analysis)) {
        printf("[ERROR] - Row mask derivation failed.\n");
        return;
    }

    row_mask_values_ = analysis.final_masks;
    row_masks_.clear();
    row_masks_.reserve(row_mask_values_.size());
    for (size_t i = 0; i < row_mask_values_.size(); ++i) {
        row_masks_.push_back(mask_to_bool(row_mask_values_[i]));
    }

    const size_t observed_bits = count_true_bits(analysis.observed);
    const size_t invariant_bits = count_true_bits_intersection(analysis.observed, analysis.hit_invariant);
    const size_t conflict_bits = count_true_bits(analysis.conflict_varies);

    printf("[ LOG ] - Observed bit positions: %zu / 64\n", observed_bits);
    printf("[ LOG ] - Hit-invariant observed positions: %zu\n", invariant_bits);
    printf("[ LOG ] - Conflict-associated positions: %zu\n", conflict_bits);
    printf("[ LOG ] - Nullspace row candidates (non-zero): %zu\n", analysis.raw_candidates.size());
    printf("[ LOG ] - Row candidates after invariance filter: %zu\n", analysis.filtered_candidates.size());
    printf("[ LOG ] - Final independent row mask count: %zu\n", row_mask_values_.size());

    if (!row_mask_values_.empty()) {
        printf("[ LOG ] - Evaluating row mask performance on same-bank samples...\n");
        
        size_t hits_respecting = 0;
        for (size_t i = 0; i < hit_diffs.size(); ++i) {
            const uint64_t diff = hit_diffs[i];
            bool zero_delta = true;
            for (size_t j = 0; j < row_mask_values_.size(); ++j) {
                const uint64_t masked = diff & row_mask_values_[j];
                if (popcount64(masked) % 2 != 0) {
                    zero_delta = false;
                    break;
                }
            }
            if (zero_delta) {
                ++hits_respecting;
            }
        }

        size_t conflicts_triggering = 0;
        for (size_t i = 0; i < conflict_diffs.size(); ++i) {
            const uint64_t diff = conflict_diffs[i];
            bool toggles_row = false;
            for (size_t j = 0; j < row_mask_values_.size(); ++j) {
                const uint64_t masked = diff & row_mask_values_[j];
                if (popcount64(masked) % 2 != 0) {
                    toggles_row = true;
                    break;
                }
            }
            if (toggles_row) {
                ++conflicts_triggering;
            }
        }

        const double hit_ratio = hit_diffs.empty() ? 0.0 :
            100.0 * static_cast<double>(hits_respecting) / static_cast<double>(hit_diffs.size());
        const double conflict_ratio = conflict_diffs.empty() ? 0.0 :
            100.0 * static_cast<double>(conflicts_triggering) / static_cast<double>(conflict_diffs.size());

        printf("[ LOG ] - Row mask evaluation: %.2f%% of hits retain zero row delta, %.2f%% of conflicts toggle row bits.\n",
               hit_ratio,
               conflict_ratio);
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

void FullAnalysis::evaluate_bank_masks() const {
    if (bank_mask_values_.empty()) {
        printf("[ WARN ] - Bank mask evaluation skipped: no masks available.\n");
        return;
    }

    if (bank_measurements_.empty()) {
        printf("[ WARN ] - Bank mask evaluation skipped: no measurement samples recorded.\n");
        return;
    }

    size_t true_pos = 0;
    size_t false_pos = 0;
    size_t true_neg = 0;
    size_t false_neg = 0;

    for (size_t i = 0; i < bank_measurements_.size(); ++i) {
        const MeasurementRecord& rec = bank_measurements_[i];
        const bool predicted = predict_conflict(rec.diff);
        const bool actual = rec.conflict;

        if (predicted) {
            if (actual) {
                ++true_pos;
            } else {
                ++false_pos;
            }
        } else {
            if (actual) {
                ++false_neg;
            } else {
                ++true_neg;
            }
        }
    }

    const size_t total = true_pos + true_neg + false_pos + false_neg;
    if (total == 0) {
        printf("[ WARN ] - Bank mask evaluation produced zero classified samples.\n");
        return;
    }

    const size_t actual_conflicts = true_pos + false_neg;
    const size_t actual_hits = true_neg + false_pos;

    const double accuracy = 100.0 * static_cast<double>(true_pos + true_neg) / static_cast<double>(total);
    const double precision = (true_pos + false_pos) ?
        100.0 * static_cast<double>(true_pos) / static_cast<double>(true_pos + false_pos) : 0.0;
    const double recall = (true_pos + false_neg) ?
        100.0 * static_cast<double>(true_pos) / static_cast<double>(true_pos + false_neg) : 0.0;
    const double f1 = (precision + recall > 0.0) ?
        (2.0 * precision * recall) / (precision + recall) : 0.0;

    printf("[ LOG ] - Bank mask evaluation samples: %zu (conflicts=%zu, hits=%zu)%s\n",
           total,
           actual_conflicts,
           actual_hits,
           bank_measurements_.size() == kMaxBankEvaluationSamples ?
               " [truncated]" : "");

    printf("[ LOG ] - Confusion Matrix:\n");
    printf("             Predicted Conflict | Predicted Hit\n");
    printf("   Actual Conflict    %8zu | %8zu\n", true_pos, false_neg);
    printf("   Actual Hit         %8zu | %8zu\n", false_pos, true_neg);

    printf("[ LOG ] - Performance: accuracy=%.2f%% precision=%.2f%% recall=%.2f%% F1=%.2f\n",
           accuracy,
           precision,
           recall,
           f1);
}
