#ifndef FULL_ANALYSIS_H
#define FULL_ANALYSIS_H

#include <cstdint>
#include <vector>
#include <cstddef>

struct AnalysisConfig {
    // Threshold detection parameters
    size_t threshold_samples = 100000;
    uint64_t manual_threshold = 0;  // If non-zero, skip auto-detection
    
    // Bank mask detection parameters
    size_t bank_target_conflicts = 5000;
    size_t bank_max_measurements = 3000000;  // 50000 * 60
    size_t bank_subsample_size = 1000;
    size_t bank_repeat_rounds = 35;
    size_t bank_max_attempts = 3;
    
    // Row mask detection parameters
    size_t row_target_pairs = 8000;
    size_t row_min_hits = 1000;
    size_t row_min_conflicts = 1000;
    size_t row_max_attempts = 240000;  // 50000 * 40, then expandable
    
    // General parameters
    bool verbose = false;
    bool force_multiple_rounds = false;  // Force re-runs if first attempt fails
};

class FullAnalysis {
public:
    FullAnalysis(size_t mem_size, unsigned int delay_us);
    FullAnalysis(size_t mem_size, unsigned int delay_us, const AnalysisConfig& config);
    ~FullAnalysis();
    void run();

private:
    struct MeasurementRecord {
        uint64_t diff;
        uint64_t latency;
        bool conflict;
    };

    static const size_t kMaxBankEvaluationSamples = 120000;

    size_t mem_size_;
    size_t measurements_;
    uint64_t flags_;
    void* mem_region_;
    uint64_t threshold_;
    unsigned int delay_us_;
    AnalysisConfig config_;
    std::vector<std::vector<bool>> bank_masks_;
    std::vector<std::vector<bool>> row_masks_;
    std::vector<uint64_t> bank_mask_values_;
    std::vector<uint64_t> row_mask_values_;
    std::vector<MeasurementRecord> bank_measurements_;

    void allocate_memory();
    void free_memory();
    void detect_threshold();
    void find_bank_masks();
    void find_row_masks();
    
    std::vector<uint64_t> measure_random_latencies(size_t num_pairs);
    bool is_same_bank(uint64_t diff_mask) const;
    bool predict_conflict(uint64_t diff_mask) const;
    void evaluate_bank_masks() const;
    bool collect_same_bank_samples(size_t target_pairs,
                                   size_t min_hits,
                                   size_t min_conflicts,
                                   std::vector<uint64_t>& hit_diffs,
                                   std::vector<uint64_t>& conflict_diffs,
                                   std::vector<uint64_t>& hit_latencies,
                                   std::vector<uint64_t>& conflict_latencies);
};

#endif // FULL_ANALYSIS_H
