#ifndef FULL_ANALYSIS_H
#define FULL_ANALYSIS_H

#include <cstdint>
#include <vector>
#include <cstddef>

class FullAnalysis {
public:
    FullAnalysis(size_t mem_size, unsigned int delay_us);
    ~FullAnalysis();
    void run();

private:
    size_t mem_size_;
    size_t measurements_;
    uint64_t flags_;
    void* mem_region_;
    uint64_t threshold_;
    unsigned int delay_us_;
    std::vector<std::vector<bool>> bank_masks_;
    std::vector<std::vector<bool>> row_masks_;
    std::vector<uint64_t> bank_mask_values_;
    std::vector<uint64_t> row_mask_values_;

    void allocate_memory();
    void free_memory();
    void detect_threshold();
    void find_bank_masks();
    void find_row_masks();
    
    std::vector<uint64_t> measure_random_latencies(size_t num_pairs);
    bool is_same_bank(uint64_t diff_mask) const;
    bool collect_same_bank_samples(size_t target_pairs,
                                   size_t min_hits,
                                   size_t min_conflicts,
                                   std::vector<uint64_t>& hit_diffs,
                                   std::vector<uint64_t>& conflict_diffs,
                                   std::vector<uint64_t>& hit_latencies,
                                   std::vector<uint64_t>& conflict_latencies);
};

#endif // FULL_ANALYSIS_H
